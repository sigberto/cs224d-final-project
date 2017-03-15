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

def normalize_scores(scores, mask):
    int_mask = tf.expand_dims(tf.cast(mask, tf.float64), dim=2)
    scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=0, keep_dims=True))
    scores = int_mask * scores
    scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=0, keep_dims=True))
    return scores

class Encoder(object):
    def __init__(self, size, vocab_dim, num_perspectives):
        self.size = size #hidden size 
        self.vocab_dim = vocab_dim #embedding size 
        self.cell = tf.nn.rnn_cell.GRUCell(self.size)
        self.attn_cell = None

        # variables needed for the context + prediction steps
        self.num_perspectives = num_perspectives
        self.num_m = 8
        self.total_m_size = self.num_perspectives * self.num_m
        self.agg_cell = tf.nn.rnn_cell.BasicLSTMCell(self.total_m_size)

    def last_hidden_state(self, fw_o, bw_o, srclen, concat, hidden_size):
        # https://danijar.com/variable-sequence-lengths-in-tensorflow/
        idx = tf.range(tf.shape(fw_o)[0]) * tf.shape(fw_o)[1] + (srclen - 1)
        last_forward_state = tf.gather(tf.reshape(fw_o, [-1, hidden_size]), idx)
        if bw_o is not None:
            last_backward_state = bw_o[:, 0, :]
            if concat:
                last_hidden_state = tf.concat(1, [last_forward_state, last_backward_state])
            else:
                last_hidden_state = (last_forward_state, last_backward_state)
        else:
            last_hidden_state = last_forward_state
        return last_hidden_state

    '''
    def encode(self, inputs, masks, f_init_state = None, b_init_state = None, scope = "", reuse = False):
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
        with vs.variable_scope(scope, reuse=reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis = 1)
            (fw_o, bw_o), (f_o_state, b_o_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell, cell_bw=self.cell, inputs=inputs,
                                sequence_length=srclen, initial_state_fw=f_init_state, initial_state_bw=b_init_state, dtype=tf.float64)
            o = tf.concat(2, [fw_o, bw_o])
            last_hidden_state = self.last_hidden_state(fw_o, bw_o, srclen)

        return o, last_hidden_state, f_o_state, b_o_state
    '''

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
            hidden_size = self.size
        elif stage == 'aggregation':
            cell = self.agg_cell
            hidden_size = self.total_m_size
        else:
            raise ValueError('stage of processing is not one of ("context_rep", "aggregation")')
        with vs.variable_scope(scope, reuse=reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            (fw_o, bw_o), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell,
                                                                                   inputs=inputs,
                                                                                   sequence_length=srclen,
                                                                                   initial_state_fw=f_init_state,
                                                                                   initial_state_bw=b_init_state,
                                                                                   dtype=tf.float64)
            if concat:
                all_hidden_states = tf.concat(2, [fw_o, bw_o])
            else:
                all_hidden_states = (fw_o, bw_o)

            last_hidden_state = self.last_hidden_state(fw_o, bw_o, srclen, concat, hidden_size)

        return all_hidden_states, last_hidden_state

    def encode_gru(self, inputs, masks, init_state = None, scope = "", reuse = False):
        with vs.variable_scope(scope, reuse=reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis = 1)
            states, _ = tf.nn.dynamic_rnn(cell=self.cell, inputs=inputs, sequence_length=srclen, initial_state=init_state, dtype=tf.float64)
            last_hidden_state = self.last_hidden_state(states, None, srclen, self.size)
            return states, last_hidden_state

    # vector, vector to get cosine scores
    def f_m_vec_vec(self, vec1, vec2, scope):
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(vec1, dim=2) * W
            Wv2 = tf.expand_dims(vec2, dim=2) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=1)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=1)
            m = tf.reduce_sum(tf.multiply(norm_Wv1, norm_Wv2), axis=1)
            return m

    # gets all cosine scores given a matrix and matrix
    def f_m_mat_mat_pool(self, mat1, mat2, op, scope):
        # TODO incorporate paragraph_mask into result
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())
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

    # given a matrix, gets all cosine scores
    def f_m_mat_vec(self, mat, vec, scope):
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(mat, dim=3) * W
            Wv2 = tf.expand_dims(vec, dim=2) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=2)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=1)
            m = tf.reduce_sum(tf.multiply(norm_Wv1, norm_Wv2), axis=2)
            return m

    def attn_mixer(self, reference_states, reference_masks, input_state):
        ht = tf.nn.rnn_cell._linear(input_state, self.size, True, 1.0)
        ht = tf.expand_dims(ht, axis=1)
        scores = tf.reduce_sum(reference_states * ht, reduction_indices=2, keep_dims=True)
        norm_scores = normalize_scores(scores, reference_masks)
        weighted_reference_states = reference_states * norm_scores

        srclen = tf.reduce_sum(tf.cast(reference_masks, tf.int32), axis=1)
        return weighted_reference_states, self.last_hidden_state(weighted_reference_states, None, srclen)

    '''
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
    '''

    def encode_w_gru_attn(self, inputs, masks, encoder_outputs, encoder_masks, init_state = None, scope = "", reuse = False):
        self.attn_cell = GRUAttnCell(self.size, encoder_outputs, encoder_masks)
        with vs.variable_scope(scope, reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            states, _ = tf.nn.dynamic_rnn(cell=self.attn_cell, inputs=inputs, sequence_length=srclen, initial_state=init_state, dtype=tf.float64)
            last_hidden_state = self.last_hidden_state(states, None, srclen)
        return states, last_hidden_state

    def attentive_matching(self, mat1, mat2, op, scope):
        with vs.variable_scope(scope):
            norm_mat1 = tf.nn.l2_normalize(mat1, dim=2)
            norm_mat2 = tf.nn.l2_normalize(mat2, dim=2)
            alpha = tf.reduce_sum(tf.expand_dims(norm_mat1, dim=2) * norm_mat2, axis=3) # (?, 750, 60)
            if op == 'mean':
                sum_alpha = 1e-6 + tf.reduce_sum(alpha, axis=2, keep_dims=True)
                weighted_mat2 = mat2 * tf.expand_dims(alpha, dim=3)
                mean_mat2 = tf.reduce_sum(weighted_mat2, axis=2)
                input_mat_2 = mean_mat2 / sum_alpha
            elif op == 'max':
                max_alpha_inds = tf.argmax(alpha, axis=2)

                # a = tf.cast(tf.range(0, tf.shape(mat1)[0]), dtype=tf.int64)
                # cat_idx = tf.concat(0, [a, max_alpha_inds])

                # max_mat1 = tf.gather_nd(mat2, max_alpha_inds)
                max_mat2 = tf.gather(mat2, max_alpha_inds)
                a=0

            else:
                raise ValueError('op was not one of ("mean", "max")')
            m_att = self.f_m_mat_mat_one_to_one(mat1, input_mat_2, scope)
            return m_att

    def f_m_mat_mat_one_to_one(self, mat1, mat2, scope):
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(mat1, dim=3) * W
            Wv2 = tf.expand_dims(mat2, dim=3) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=2)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=2)
            cos_sim = tf.reduce_sum(norm_Wv1 * norm_Wv2, axis=2)
            return cos_sim

class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_outputs, encoder_masks, scope=None):
        self.hs = encoder_outputs
        self.hs_masks = encoder_masks
        super(GRUAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, _ = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn"):
                ht = tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)
                ht = tf.expand_dims(ht, axis = 1)
            scores = tf.reduce_sum(self.hs * ht, reduction_indices=2, keep_dims=True)
            norm_scores = normalize_scores(scores, self.hs_masks)
            # alpha = tf.exp(scores)

            # mask the exponents so that entries not in paragraph don't contribute to weights
            # alpha = tf.expand_dims(int_mask, dim=2) * alpha
            # srclen = tf.reshape(srclen, [-1, tf.shape(self.hs_masks)[1], 1])

            # sum_alpha = tf.reduce_sum(alpha, reduction_indices=1)
            context = tf.reduce_sum(self.hs * norm_scores, reduction_indices=1)

            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(tf.nn.rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))

        return (out, out)



class Decoder(object):
    def __init__(self, output_size, num_perspectives):
        self.output_size = output_size #output size
        self.intermediate_size_factor = 10
        self.num_perspectives = num_perspectives

    def decode(self, knowledge_rep):
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
            intermediate_a_s = tf.nn.rnn_cell._linear(knowledge_rep, output_size = self.intermediate_size_factor * self.output_size, bias=True, scope='intermediate')
            intermediate_relu =tf.nn.relu(intermediate_a_s)
            a_s = tf.nn.rnn_cell._linear(intermediate_relu, output_size = self.output_size, bias=True, scope='final')
        with vs.variable_scope("answer_end"):
            intermediate_a_e = tf.nn.rnn_cell._linear(knowledge_rep, output_size = self.intermediate_size_factor * self.output_size, bias=True, scope='intermediate')
            intermediate_relu = tf.nn.relu(intermediate_a_e)
            a_e = tf.nn.rnn_cell._linear(intermediate_relu, output_size = self.output_size, bias=True, scope='final')
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
       
        self.question = tf.placeholder(tf.int32, [None, self.FLAGS.max_question_size])
        self.paragraph = tf.placeholder(tf.int32, [None, self.FLAGS.max_paragraph_size])

        self.start_answer = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])
        self.end_answer = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])

        self.question_mask = tf.placeholder(tf.bool, [None,self.FLAGS.max_question_size])
        self.paragraph_mask = tf.placeholder(tf.bool, [None,self.FLAGS.max_paragraph_size])

        self.encoder = encoder
        self.decoder = decoder



        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        #params = tf.trainable_variables()
        optimizer = get_optimizer('adam')(self.FLAGS.learning_rate)
        grad_var = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grad = [x[0] for x in grad_var]
        grad, _ = tf.clip_by_global_norm(grad, self.FLAGS.max_gradient_norm)
        grad_var = zip(grad, tf.trainable_variables())
        train_op = optimizer.apply_gradients(grad_var)
        self.norm_grad = tf.global_norm(grad)
        self.updates = train_op

    def filter_layer(self):
        norm_q = tf.nn.l2_normalize(self.question_var, dim=2)
        norm_p = tf.nn.l2_normalize(self.paragraph_var, dim=2)
        r_ij = tf.matmul(norm_q, norm_p, transpose_b=True)
        r_j = tf.reduce_max(r_ij, axis=1)
        p_prime = tf.multiply(tf.expand_dims(r_j, dim=2), self.paragraph_var)
        self.paragraph_var = p_prime

    def bmpm_layer(self, from_fw_all_h, from_bw_all_h, p_fw_last_h, p_bw_last_h, to_fw_all_h, to_bw_all_h, to_fw_last_h,
                   to_bw_last_h, scope):
        with vs.variable_scope(scope):
            m_full_fw = self.encoder.f_m_mat_vec(from_fw_all_h, to_fw_last_h, scope='W1')
            m_full_bw = self.encoder.f_m_mat_vec(from_bw_all_h, to_bw_last_h, scope='W2')
            m_max_fw = self.encoder.f_m_mat_mat_pool(from_fw_all_h, to_fw_all_h, 'max', scope='W3')
            m_max_bw = self.encoder.f_m_mat_mat_pool(from_bw_all_h, to_bw_all_h, 'max', scope='W4')
            m_attentive_matching_mean_fw = self.encoder.attentive_matching(from_fw_all_h, to_fw_all_h, 'mean', scope='W5')
            m_attentive_matching_mean_bw = self.encoder.attentive_matching(from_bw_all_h, to_bw_all_h, 'mean', scope='W6')
            # m_attentive_matching_max_fw = self.encoder.attentive_matching(from_fw_all_h, to_fw_all_h, 'max', scope='W7')
            # m_attentive_matching_max_bw = self.encoder.attentive_matching(from_bw_all_h, to_bw_all_h, 'max', scope='W8')

            m = tf.concat(2, [m_full_fw, m_full_bw, m_max_fw, m_max_bw, m_attentive_matching_mean_fw,
                              m_attentive_matching_mean_bw, m_attentive_matching_mean_fw, m_attentive_matching_mean_bw])
        return m



    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        '''
        ###### BASELINE ######
        # step 1
        q_states, q_last_state = self.encoder.encode_gru(self.question_var, self.question_mask, scope='question')
        # step 2
        p_states, p_last_state = self.encoder.encode_w_gru_attn(self.paragraph_var, self.paragraph_mask, encoder_outputs=q_states,
                                                                encoder_masks=self.question_mask, scope='paragraph')
        # step 3
        knowledge_rep, last_knowledge_rep = self.encoder.attn_mixer(p_states, self.paragraph_mask, q_last_state)

        # step 4
        self.a_s, self.a_e = self.decoder.decode(last_knowledge_rep)
        '''

        ###### Bilateral ######
        # TODO: cut paragraph and question lens to actual lens

        (p_fw_all_h, p_bw_all_h), (p_fw_last_h, p_bw_last_h) = self.encoder.encode(self.paragraph_var, self.paragraph_mask, stage='context_rep', concat=False, scope='paragraph')
        (q_fw_all_h, q_bw_all_h), (q_fw_last_h, q_bw_last_h) = self.encoder.encode(self.question_var, self.question_mask, stage='context_rep', concat=False, scope='question')
        m_p = self.bmpm_layer(p_fw_all_h, p_bw_all_h, p_fw_last_h, p_bw_last_h, q_fw_all_h, q_bw_all_h, q_fw_last_h, q_bw_last_h, scope ='p_bmpm')
        m_q = self.bmpm_layer(q_fw_all_h, q_bw_all_h, q_fw_last_h, q_bw_last_h, p_fw_all_h, p_bw_all_h, p_fw_last_h, p_bw_last_h, scope ='q_bmpm')
        (_, _), (p_agg_fw_last_h, p_agg_bw_last_h) = self.encoder.encode(m_p,
                                                                       self.paragraph_mask,
                                                                       stage='aggregation', concat=False,
                                                                       scope='p_aggregation')
        (_, _), (q_agg_fw_last_h, q_agg_bw_last_h) = self.encoder.encode(m_q,
                                                                       self.question_mask,
                                                                       stage='aggregation', concat=False,
                                                                       scope='q_aggregation')
        knowledge_rep = tf.concat(1, [p_agg_fw_last_h, p_agg_bw_last_h, q_agg_fw_last_h, q_agg_bw_last_h])
        self.a_s, self.a_e = self.decoder.decode(knowledge_rep)

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
        with vs.variable_scope("embeddings"):
            glove_matrix = np.load(self.FLAGS.embed_path)['glove']
            embedding = tf.constant(glove_matrix)  # don't train the embeddings 
            self.paragraph_var = tf.nn.embedding_lookup(embedding, self.paragraph)
            self.question_var = tf.nn.embedding_lookup(embedding, self.question)

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
        output_feed = [self.updates, self.loss, self.norm_grad]

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

    def mask_and_pad(self, sentence_batch, text_type):
        masks = []
        padded_sentences = []
        if text_type == 'paragraph':
            max_len = self.FLAGS.max_paragraph_size
        elif text_type == 'question':
            max_len = self.FLAGS.max_question_size
        else:
            raise ValueError('text_type was not one of ("paragraph", "question")')
        for sentence in sentence_batch:
            padded_s = list(sentence)
            sentence_len = len(sentence)
            if sentence_len > max_len:
                padded_s = padded_s[:max_len]
                mask = [True] * max_len
            else:
                pad_len = max_len - sentence_len
                padded_s.extend([0]*pad_len)
                mask = [True] * sentence_len + [False] * pad_len
            masks.append(mask)
            padded_sentences.append(padded_s)
        return np.asarray(padded_sentences, dtype=np.int32), np.asarray(masks, dtype=np.int32)

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

        count = 0
        for e in range(self.FLAGS.epochs):
            for p, q, a in util.load_dataset("data/squad/train.ids.context", "data/squad/train.ids.question", "data/squad/train.span", self.FLAGS.batch_size, in_batches=True):

                a_s, a_e = self.one_hot_func(a)
                q, q_mask = self.mask_and_pad(q, 'question')
                p, p_mask = self.mask_and_pad(p, 'paragraph')
                
                updates, loss, norm = self.optimize(session, p, p_mask, q, q_mask, a_s, a_e)

                count += 1
                print(str(count) + ": Loss is " + str(loss)  + " ----- Norm is " + str(norm))
            # save the model
            saver = tf.Saver()


            val_loss = self.validate(p_val, q_val, a_val)

            self.evaluate_answer(session, p_val, q_val)
            self.evaluate_answer(session, q, p, sample = 100)

