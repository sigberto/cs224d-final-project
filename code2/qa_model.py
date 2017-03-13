from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
import util

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size #hidden size 
        self.vocab_dim = vocab_dim #embedding size
        self.num_perspectives = 50
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.agg_cell = tf.nn.rnn_cell.BasicLSTMCell(6*self.num_perspectives)
        self.attn_cell = None


    def last_hidden_state(self, fw_o, bw_o, srclen, concat):
        # https://danijar.com/variable-sequence-lengths-in-tensorflow/
        idx = tf.range(tf.shape(fw_o)[0]) * tf.shape(fw_o)[1] + (srclen - 1)
        last_forward_state = tf.gather(tf.reshape(fw_o, [-1, self.size]), idx)
        if bw_o is not None:
            last_backward_state = bw_o[:, 0, :]
            if concat:
                last_hidden_state = tf.concat(1, [last_forward_state, last_backward_state])
            else:
                last_hidden_state = (last_forward_state, last_backward_state)
        else:
            last_hidden_state = last_forward_state
        return last_hidden_state

    def encode(self, inputs, masks, concat, stage, f_init_state=None, b_init_state=None, scope="", reuse=False):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        if stage == 'context_rep':
            cell = self.cell
        elif stage == 'aggregation':
            cell = self.agg_cell
        else:
            raise ValueError('stage of processing is not one of ("context_rep", "aggregation")')
        with vs.variable_scope(scope, reuse=reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis = 1)
            (fw_o, bw_o), (f_o_state, b_o_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=inputs,
                                sequence_length=srclen, initial_state_fw=f_init_state, initial_state_bw=b_init_state, dtype=tf.float64)
            if concat:
                all_hidden_states = tf.concat(2, [fw_o, bw_o])
            else:
                all_hidden_states = (fw_o, bw_o)

            last_hidden_state = self.last_hidden_state(fw_o, bw_o, srclen, concat)

        return all_hidden_states, last_hidden_state #, f_o_state, b_o_state

    def f_m_vec_vec(self, vec1, vec2, scope):
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(vec1, dim=2) * W
            Wv2 = tf.expand_dims(vec2, dim=2) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=1)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=1)
            m = tf.reduce_sum(tf.multiply(norm_Wv1, norm_Wv2), axis=1)
            return m

    def f_m_mat_mat_pool(self, mat1, mat2, op, scope):
        # TODO incorporate paragraph_mask into result
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(mat1, dim=3) * W
            Wv2 = tf.expand_dims(mat2, dim=3) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=2)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=2)
            outer_product = tf.reduce_sum(tf.expand_dims(norm_Wv1, dim=2) * (norm_Wv2), axis=3)
            if op == 'max':
                return tf.reduce_max(outer_product, axis=2)
            elif op == 'mean':
                return tf.reduce_mean(outer_product, axis=2)
            else:
                raise ValueError('op type is not one of ("max", "mean")')

    def f_m_mat_vec(self, mat, vec, scope):
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(mat, dim=3) * W
            Wv2 = tf.expand_dims(vec, dim=2) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=2)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=1)

            m = tf.reduce_sum(tf.multiply(norm_Wv1, norm_Wv2), axis=2)
            return m

    def aggregate(self, scope):
        with vs.variable_scope(scope):


    def attn_mixer(self, reference_states, reference_masks, input_state):
        ht = tf.nn.rnn_cell._linear(input_state, 2*self.size, True, 1.0)
        ht = tf.expand_dims(ht, axis=1)
        scores = tf.reduce_sum(reference_states * ht, reduction_indices=2, keep_dims=True)

        alpha = tf.exp(scores)
        # mask the exponents so that entries not in paragraph don't contribute to weights
        srclen = tf.cast(reference_masks, tf.float64)
        srclen = tf.reshape(srclen, [-1, tf.shape(reference_states)[1], 1])
        alpha = alpha * srclen

        weighted_reference_states = tf.reduce_sum(reference_states * alpha, reduction_indices=1)

        sum_alpha = tf.reduce_sum(alpha, reduction_indices=1)

        return weighted_reference_states/sum_alpha

    def encode_w_attn(self, inputs, masks, encoder_outputs, encoder_masks, f_init_state = None, b_init_state = None, scope = "", reuse = False):

        fw_encoder_output = encoder_output[:, :, 0:self.size]
        bw_encoder_output = encoder_output[:, :, self.size:]

        self.attn_cell = GRUAttnCell(2*self.size, encoder_outputs, encoder_masks)
        # self.fw_attn_cell = GRUAttnCell(self.size, fw_encoder_outputs)
        # self.bw_attn_cell = GRUAttnCell(self.size, bw_encoder_outputs)

        with vs.variable_scope(scope, reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            # (fw_o, bw_o), (f_o_state, b_o_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_attn_cell, cell_bw=self.bw_attn_cell,
            #                                                                    inputs=inputs,
            #                                                                    sequence_length=srclen,
            #                                                                    initial_state_fw=f_init_state,
            #                                                                    initial_state_bw=b_init_state,
            #                                                                    dtype=tf.float64)
            o, state = tf.nn.dynamic_rnn(cell=self.attn_cell,
                                                   inputs=inputs,
                                                   sequence_length=srclen,
                                                   initial_state=f_init_state,
                                                   dtype=tf.float64)

            last_hidden_state = self.last_hidden_state(o, None, srclen)

        return o, last_hidden_state

class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_outputs, encoder_masks, scope=None):
        self.hs = encoder_outputs
        self.hs_masks = encoder_masks
        super(GRUAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn"):
                ht = tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)
                ht = tf.expand_dims(ht, axis = 1)
            scores = tf.reduce_sum(self.hs * ht, reduction_indices=2, keep_dims=True)

            alpha = tf.exp(scores)

            # mask the exponents so that entries not in paragraph don't contribute to weights
            srclen = tf.cast(self.hs_masks, tf.float64)
            srclen = tf.reshape(srclen, [-1, tf.shape(self.hs_masks)[1], 1])
            alpha = alpha * srclen

            sum_alpha = tf.reduce_sum(alpha, reduction_indices=1)

            context = tf.reduce_sum(self.hs * alpha, reduction_indices=1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(tf.nn.rnn_cell._linear([context/sum_alpha, gru_out], self._num_units, True, 1.0))

        return (out, out)



class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size #output size 

    def decode(self, h_q, h_p):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        # h_q, h_p: both are 2-d TF variables 
        with vs.variable_scope("answer_start"):
            a_s = tf.nn.rnn_cell._linear([h_q, h_p], output_size = self.output_size, bias=True)
        with vs.variable_scope("answer_end"):
            a_e = tf.nn.rnn_cell._linear([h_q, h_p], output_size = self.output_size, bias=True)
        return a_s, a_e


class QASystem(object):
    def __init__(self, encoder, decoder, FLAGS):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.FLAGS = FLAGS
        # ==== set up placeholder tokens ========
       
        self.question = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])
        self.paragraph = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])

        self.start_answer = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])
        self.end_answer = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])

        self.question_mask = tf.placeholder(tf.bool, [None,self.FLAGS.output_size])
        self.paragraph_mask = tf.placeholder(tf.bool, [None,self.FLAGS.output_size])

        self.encoder = encoder
        self.decoder = decoder

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        #params = tf.trainable_variables()
        self.updates = get_optimizer('adam')(self.FLAGS.learning_rate).minimize(self.loss)

    def filter_layer(self):
        norm_q = tf.nn.l2_normalize(self.question_var, dim=2)
        norm_p = tf.nn.l2_normalize(self.paragraph_var, dim=2)
        r_ij = tf.matmul(norm_q, norm_p, transpose_b=True)
        r_j = tf.reduce_max(r_ij, axis=1)
        p_prime = tf.multiply(tf.expand_dims(r_j, dim=2), self.paragraph_var)
        self.paragraph_var = p_prime

    def mpcm_layer(self, p_fw_all_h, p_bw_all_h, p_fw_last_h, p_bw_last_h, q_fw_all_h, q_bw_all_h, q_fw_last_h, q_bw_last_h):
        m_full_fw = self.encoder.f_m_mat_vec(p_fw_all_h, q_fw_last_h, scope='W1')
        m_full_bw = self.encoder.f_m_mat_vec(q_fw_all_h, p_fw_last_h, scope='W2')
        m_max_fw = self.encoder.f_m_mat_mat_pool(p_fw_all_h, q_fw_all_h, 'max', scope='W3')
        m_max_bw = self.encoder.f_m_mat_mat_pool(p_bw_all_h, q_bw_all_h, 'max', scope='W4')
        m_mean_fw = self.encoder.f_m_mat_mat_pool(p_fw_all_h, q_fw_all_h, 'mean', scope='W5')
        m_mean_bw = self.encoder.f_m_mat_mat_pool(p_bw_all_h, q_bw_all_h, 'mean', scope='W6')
        m = tf.concat(2, [m_full_fw, m_full_bw, m_max_fw, m_max_bw, m_mean_fw, m_mean_bw])
        return m

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # TODO: cut paragraph and question lens to actual lens
        self.filter_layer()
        (p_fw_all_h, p_bw_all_h), (p_fw_last_h, p_bw_last_h) = self.encoder.encode(self.paragraph_var, self.paragraph_mask, stage='context_rep', concat=False, scope='paragraph')
        (q_fw_all_h, q_bw_all_h), (q_fw_last_h, q_bw_last_h) = self.encoder.encode(self.question_var, self.question_mask, concat=False, scope='question')
        m = self.mpcm_layer(p_fw_all_h, p_bw_all_h, p_fw_last_h, p_bw_last_h, q_fw_all_h, q_bw_all_h, q_fw_last_h, q_bw_last_h)
        agg_all_h, agg_last_h = self.encoder.encode(m, self.paragraph_mask, stage='aggregation', concat=True, scope='aggregation')


        self.a_s, self.a_e = self.decoder.decode(h_q, atten_o_p)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            l1 = tf.nn.softmax_cross_entropy_with_logits(self.a_s, self.start_answer)
            l2 = tf.nn.softmax_cross_entropy_with_logits(self.a_e, self.end_answer)
            self.loss = tf.reduce_mean(l1+l2)
        


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        q_mask = tf.cast(tf.expand_dims(self.question_mask, axis=2), tf.float64)
        p_mask = tf.cast(tf.expand_dims(self.paragraph_mask, axis=2), tf.float64)

        with vs.variable_scope("embeddings"):
            glove_matrix = np.load(self.FLAGS.embed_path)['glove']
            embedding = tf.constant(glove_matrix)  # don't train the embeddings 
            self.paragraph_var = tf.nn.embedding_lookup(embedding, self.paragraph) * p_mask
            self.question_var = tf.nn.embedding_lookup(embedding, self.question) * q_mask
            # do we have to reshape?
            #self.question_var = tf.reshape(self.question_var, [-1, self.FLAGS.output_size, self.FLAGS.embedding_size])


    def optimize(self, session, paragraph, paragraph_mask, question, question_mask, answer_start, answer_end):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        #this does all the create_feed_dict 

        # we are iterating 
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.paragraph] = paragraph
        input_feed[self.question] = question
        input_feed[self.paragraph_mask] = paragraph_mask
        input_feed[self.question_mask] = question_mask
        input_feed[self.start_answer] = answer_start
        input_feed[self.end_answer] = answer_end

        # grad_norm, param_norm
        output_feed = [self.updates, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_paragraph, test_question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        input_feed[self.paragraph] = test_paragraph
        input_feed[self.question] = test_question

        output_feed = [self.a_s, self.a_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, paragraph, question, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        from evaluate import f1_score, exact_match_score
        # use these functions to determine how the model is actually doing 

        f1 = 0.
        em = 0.

        for p, q in zip(paragraph, question):
            a_s, a_e = self.answer(session, p, q)
            answer = p[a_s : a_e + 1]
            f1_score(prediction, ground_truth)

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def one_hot_func(self, answers):
        a_s = np.zeros((len(answers), self.FLAGS.output_size))
        a_e = np.zeros((len(answers), self.FLAGS.output_size))
        for i, a in enumerate(answers):
            a_s[i, a[0]] = 1
            a_e[i, a[1]] = 1
        return a_s, a_e

    def mask_and_pad(self, sentence_batch):
        masks = []
        padded_sentences = []
        for sentence in sentence_batch:
            padded_s = list(sentence)
            mask = [True]*len(sentence)
            pad_size = self.FLAGS.output_size - len(sentence)
            mask.extend([False]*pad_size)
            masks.append(mask)
            padded_s.extend([0]*pad_size)
            padded_sentences.append(padded_s)
        return np.asarray(padded_sentences), np.asarray(masks)

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code2 to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for e in range(self.FLAGS.epochs):
            for p, q, a in util.load_dataset("data/squad/train.ids.context", "data/squad/train.ids.question", "data/squad/train.span", self.FLAGS.batch_size, in_batches=True):

                a_s, a_e = self.one_hot_func(a)
                q, q_mask = self.mask_and_pad(q)
                p, p_mask = self.mask_and_pad(p)
                
                updates, loss = self.optimize(session, q, q_mask, p, p_mask, a_s, a_e)
                print(loss)
            # save the model
            saver = tf.Saver()


            val_loss = self.validate(p_val, q_val, a_val)

            self.evaluate_answer(session, p_val, q_val)
            self.evaluate_answer(session, q, p, sample = 100)

