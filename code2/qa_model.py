from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

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
    scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
    scores = int_mask * scores
    scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))
    return scores


class Encoder(object):
    def __init__(self, size, vocab_dim, num_perspectives):
        self.size = size  # hidden size
        self.vocab_dim = vocab_dim  # embedding size
        self.dropout = 0.15
        self.cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(self.size),
            output_keep_prob=self.dropout)
        self.attn_cell = None

        # variables needed for the context + prediction steps
        self.num_perspectives = num_perspectives
        self.num_m = 8
        self.total_m_size = self.num_perspectives * self.num_m
        self.agg_cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(self.total_m_size),
            output_keep_prob=self.dropout)

    def last_hidden_state(self, fw_o, bw_o, srclen, concat, hidden_size):
        # https://danijar.com/variable-sequence-lengths-in-tensorflow/
        idx = tf.range(tf.shape(fw_o)[0]) * tf.shape(fw_o)[1] + (srclen - 1)
        last_forward_state = tf.gather(tf.reshape(fw_o, [-1, hidden_size]), idx)
        if bw_o is not None:
            last_hidden_state = tf.concat(1, [last_forward_state, bw_o[:, 0, :]])
            last_backward_state = bw_o[:, 0, :]
            if concat:
                last_hidden_state = tf.concat(1, [last_forward_state, last_backward_state])
            else:
                last_hidden_state = (last_forward_state, last_backward_state)
        else:
            last_hidden_state = last_forward_state
        return last_hidden_state

    def encode(self, inputs, masks, f_init_state=None, b_init_state=None, scope="", reuse=False):
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
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            (fw_o, bw_o), (f_o_state, b_o_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell, cell_bw=self.cell,
                                                                                   inputs=inputs,
                                                                                   sequence_length=srclen,
                                                                                   initial_state_fw=f_init_state,
                                                                                   initial_state_bw=b_init_state,
                                                                                   dtype=tf.float64)
            o = tf.concat(2, [fw_o, bw_o])
            last_hidden_state = self.last_hidden_state(fw_o, bw_o, srclen)

        return o, last_hidden_state, f_o_state, b_o_state

    def encode_bilstm(self, inputs, masks, concat, stage, f_init_state=None, b_init_state=None, scope="", reuse=False):
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
            (fw_o, bw_o), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=inputs,
                                                              sequence_length=srclen, initial_state_fw=f_init_state,
                                                              initial_state_bw=b_init_state, dtype=tf.float64)
            if concat:
                all_hidden_states = tf.concat(2, [fw_o, bw_o])
            else:
                all_hidden_states = (fw_o, bw_o)

            last_hidden_state = self.last_hidden_state(fw_o, bw_o, srclen, concat, hidden_size)

        return all_hidden_states, last_hidden_state

    def encode_gru(self, inputs, masks, init_state=None, scope="", reuse=False):
        with vs.variable_scope(scope, reuse=reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            states, _ = tf.nn.dynamic_rnn(cell=self.cell, inputs=inputs, sequence_length=srclen,
                                          initial_state=init_state, dtype=tf.float64)
            last_hidden_state = self.last_hidden_state(states, None, srclen, False, self.size)
            return states, last_hidden_state


    def attn_mixer(self, reference_states, reference_masks, input_state, scope="", reuse=False):
        with vs.variable_scope(scope, reuse=reuse):
            ht = tf.nn.rnn_cell._linear(input_state, self.size, True, 1.0)
            ht = tf.expand_dims(ht, axis=1)
            scores = tf.reduce_sum(reference_states * ht, reduction_indices=2, keep_dims=True)
            norm_scores = normalize_scores(scores, reference_masks)
            weighted_reference_states = reference_states * norm_scores

            srclen = tf.reduce_sum(tf.cast(reference_masks, tf.int32), axis=1)
            return weighted_reference_states, self.last_hidden_state(weighted_reference_states, None, srclen, False, self.size)


    def encode_w_gru_attn(self, inputs, masks, encoder_outputs, encoder_masks, init_state=None, scope="", reuse=False):
        self.attn_cell = GRUAttnCell(self.size, encoder_outputs, encoder_masks)
        with vs.variable_scope(scope, reuse):
            srclen = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
            states, _ = tf.nn.dynamic_rnn(cell=self.attn_cell, inputs=inputs, sequence_length=srclen,
                                          initial_state=init_state, dtype=tf.float64)
            last_hidden_state = self.last_hidden_state(states, None, srclen, False, self.size)
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
            norm_Wv1_t = tf.transpose(norm_Wv1, [0,3,1,2]) #(?, num_perspectives, num_mat1_states, hidden_size)
            norm_Wv2_t = tf.transpose(norm_Wv2, [0,3,2,1]) #(?, num_perspectives, hidden_size, num_mat2_states)
            compared_states = tf.matmul(norm_Wv1_t, norm_Wv2_t) #(?, num_perspectives, num_mat1_states, num_mat2_states)
            if op == 'max':
                result = tf.reduce_max(compared_states, axis=3)
            elif op == 'mean':
                result = tf.reduce_mean(compared_states, axis=3)
            else:
                raise ValueError('op type is not one of ("max", "mean")')
            m = tf.transpose(result, [0, 2, 1])
            return m

    # given a matrix, gets all cosine scores
    def f_m_mat_vec(self, mat, vec, scope):
        with vs.variable_scope(scope):
            W = tf.get_variable('W', shape=(self.size, self.num_perspectives), dtype=tf.float64,
                                initializer=tf.contrib.layers.xavier_initializer())
            Wv1 = tf.expand_dims(mat, dim=3) * W
            Wv2 = tf.expand_dims(vec, dim=2) * W
            norm_Wv1 = tf.nn.l2_normalize(Wv1, dim=2)
            norm_Wv2 = tf.nn.l2_normalize(Wv2, dim=1)
            num_states = mat.get_shape()[1]
            tile_tensor = tf.constant([1, int(num_states), 1, 1], dtype=tf.int32)
            tiled_norm_Wv2 = tf.tile(tf.expand_dims(norm_Wv2, dim=1), tile_tensor)
            product = norm_Wv1 * tiled_norm_Wv2
            m = tf.reduce_sum(product, axis=2)
            return m

    def compare_states(self, states1, states2):
        norm_states1 = tf.nn.l2_normalize(states1, dim=2)
        norm_states2 = tf.nn.l2_normalize(states2, dim=2)
        alpha = tf.matmul(norm_states1, norm_states2, transpose_b=True)  # (?, num_perspectives, num_mat1_states, num_mat2_states)
        return alpha

    def attentive_matching(self, mat1, mat2, alpha, op, scope):
        with vs.variable_scope(scope):
            # norm_mat1 = tf.nn.l2_normalize(mat1, dim=2)
            # norm_mat2 = tf.nn.l2_normalize(mat2, dim=2)
            if op == 'mean':
                weighted_mat2 = batch_matmul(alpha, mat2)
                sum_alpha = 1e-6 + tf.reduce_sum(alpha, axis=2, keep_dims=True)
                input_mat_2 = weighted_mat2 / sum_alpha
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
                ht = tf.expand_dims(ht, axis=1)
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
        self.output_size = output_size  # output size
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
            a_s = tf.nn.rnn_cell._linear(knowledge_rep, output_size=self.output_size, bias=True)
        with vs.variable_scope("answer_end"):
            a_e = tf.nn.rnn_cell._linear(knowledge_rep, output_size=self.output_size, bias=True)
        return a_s, a_e

    # def decode_gru(self, start_knowledge_rep, end_knowledge_rep):
    #     with vs.variable_scope("answer_start"):
    #         a_s = tf.nn.rnn_cell._linear(start_knowledge_rep, output_size=self.output_size, bias=True)
    #     with vs.variable_scope("answer_end"):
    #         a_e = tf.nn.rnn_cell._linear(end_knowledge_rep, output_size=self.output_size, bias=True)
    #     return a_s, a_e


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

        # self.start_answer = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])
        self.start_answer = tf.placeholder(tf.int32, [None,])
        # self.end_answer = tf.placeholder(tf.int32, [None, self.FLAGS.output_size])
        self.end_answer = tf.placeholder(tf.int32, [None,])

        self.question_mask = tf.placeholder(tf.bool, [None, self.FLAGS.max_question_size])
        self.paragraph_mask = tf.placeholder(tf.bool, [None,self.FLAGS.max_paragraph_size])

        # self.dropout = tf.placeholder(tf.float64, ())

        self.encoder = encoder
        self.decoder = decoder

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system_baseline()
            # self.setup_system_bmpm()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        # params = tf.trainable_variables()
        optimizer = get_optimizer('adam')(self.FLAGS.learning_rate)
        grad_var = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grad = [x[0] for x in grad_var]
        grad, _ = tf.clip_by_global_norm(grad, self.FLAGS.max_gradient_norm)
        grad_var = zip(grad, tf.trainable_variables())
        train_op = optimizer.apply_gradients(grad_var)
        self.norm_grad = tf.global_norm(grad)
        self.updates = train_op
        self.grad_norm = tf.global_norm(grad)

    def setup_system_baseline(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        """
        # step 1
        q_states, q_last_state = self.encoder.encode_gru(self.question_var, self.question_mask, scope='question')
        # step 2
        p_states, p_last_state = self.encoder.encode_w_gru_attn(self.paragraph_var, self.paragraph_mask, encoder_outputs=q_states,
                                                                encoder_masks=self.question_mask, scope='paragraph')
        q_2_states, q_last_state = self.encoder.encode_w_gru_attn(q_states, self.question_mask, encoder_outputs=p_states,
                                                                encoder_masks=self.paragraph_mask, scope='q_attn')
        # step 3
        p_3_states, _ = self.encoder.attn_mixer(p_states, self.paragraph_mask, q_last_state, scope='attn_to_q')
        q_3_states, _ = self.encoder.attn_mixer(q_2_states, self.question_mask, p_last_state, scope='attn_to_p')


        p_4_states, p_4_last_state = self.encoder.encode_gru(p_3_states, self.paragraph_mask, scope='p4')
        q_4_states, q_4_last_state = self.encoder.encode_gru(q_3_states, self.question_mask, scope='q4')

        # _, p_5_last_state = self.encoder.encode_gru(p_4_states, self.paragraph_mask, scope='p5')
        # step 4
        self.a_s, self.a_e = self.decoder.decode_gru(p_4_last_state, q_4_last_state)
        """

        # step 1
        q_states, q_last_state = self.encoder.encode_gru(self.question_var, self.question_mask, scope='question')
        # step 2
        p_attn_states, p_attn_last_state = self.encoder.encode_w_gru_attn(self.paragraph_var, self.paragraph_mask,
                                                                encoder_outputs=q_states,
                                                                encoder_masks=self.question_mask, scope='paragraph_attn')

        q_attn_states, q_attn_last_state = self.encoder.encode_w_gru_attn(q_states, self.question_mask,
                                                                          encoder_outputs=p_attn_states,
                                                                          encoder_masks=self.paragraph_mask,
                                                                          scope='question_attn')
        # step 3
        p_3_states, p_3_last_state = self.encoder.attn_mixer(p_attn_states, self.paragraph_mask, q_last_state, scope='paragraph_mix')
        p_3_sum_state = tf.reduce_sum(p_3_states, reduction_indices = 1)

        q_3_states, q_3_last_state = self.encoder.attn_mixer(q_attn_states, self.question_mask, p_attn_last_state, scope='question_mix')
        q_3_sum_state = tf.reduce_sum(q_3_states, reduction_indices = 1)

        p_4_states, p_4_last_state = self.encoder.encode_gru(p_3_states, self.paragraph_mask, scope='p4')
        q_4_states, q_4_last_state = self.encoder.encode_gru(q_3_states, self.question_mask, scope='q4')

        #p_4_concat_state = tf.concat(1, [p_3_sum_state, q_3_sum_state])
        p_4_concat_state = tf.concat(1, [p_4_last_state, q_4_last_state])

        # step 4
        self.a_s, self.a_e = self.decoder.decode(p_4_concat_state)


    def bmpm_layer(self, from_fw_all_h, from_bw_all_h, p_fw_last_h, p_bw_last_h, to_fw_all_h, to_bw_all_h, to_fw_last_h,
                   to_bw_last_h, alpha_fw, alpha_bw, scope):
        with vs.variable_scope(scope):
            m_full_fw = self.encoder.f_m_mat_vec(from_fw_all_h, to_fw_last_h, scope='W1')
            m_full_bw = self.encoder.f_m_mat_vec(from_bw_all_h, to_bw_last_h, scope='W2')
            m_max_fw = self.encoder.f_m_mat_mat_pool(from_fw_all_h, to_fw_all_h, 'max', scope='W3')
            m_max_bw = self.encoder.f_m_mat_mat_pool(from_bw_all_h, to_bw_all_h, 'max', scope='W4')
            m_attentive_matching_mean_fw = self.encoder.attentive_matching(from_fw_all_h, to_fw_all_h, alpha_fw, 'mean', scope='W5')
            m_attentive_matching_mean_bw = self.encoder.attentive_matching(from_bw_all_h, to_bw_all_h, alpha_bw, 'mean', scope='W6')
            # m_attentive_matching_max_fw = self.encoder.attentive_matching(from_fw_all_h, to_fw_all_h, 'max', scope='W7')
            # m_attentive_matching_max_bw = self.encoder.attentive_matching(from_bw_all_h, to_bw_all_h, 'max', scope='W8')

            m = tf.concat(2, [m_full_fw, m_full_bw, m_max_fw, m_max_bw, m_attentive_matching_mean_fw,
                              m_attentive_matching_mean_bw, m_attentive_matching_mean_fw, m_attentive_matching_mean_bw])
        return m

    def setup_system_bmpm(self):

        (p_fw_all_h, p_bw_all_h), (p_fw_last_h, p_bw_last_h) = self.encoder.encode_bilstm(self.paragraph_var, self.paragraph_mask, stage='context_rep', concat=False, scope='paragraph')
        (q_fw_all_h, q_bw_all_h), (q_fw_last_h, q_bw_last_h) = self.encoder.encode_bilstm(self.question_var, self.question_mask, stage='context_rep', concat=False, scope='question')
        alpha_fw = self.encoder.compare_states(p_fw_all_h, q_fw_all_h)
        alpha_bw = self.encoder.compare_states(p_bw_all_h, q_bw_all_h)
        m_p = self.bmpm_layer(p_fw_all_h, p_bw_all_h, p_fw_last_h, p_bw_last_h, q_fw_all_h, q_bw_all_h, q_fw_last_h, q_bw_last_h,
                              alpha_fw, alpha_bw, scope='p_bmpm')
        m_q = self.bmpm_layer(q_fw_all_h, q_bw_all_h, q_fw_last_h, q_bw_last_h, p_fw_all_h, p_bw_all_h, p_fw_last_h, p_bw_last_h,
                              tf.transpose(alpha_fw, [0,2,1]), tf.transpose(alpha_bw, [0,2,1]), scope='q_bmpm')
        (_, _), (p_agg_fw_last_h, p_agg_bw_last_h) = self.encoder.encode_bilstm(m_p,
                                                                       self.paragraph_mask,
                                                                       stage='aggregation', concat=False,
                                                                       scope='p_aggregation')
        (_, _), (q_agg_fw_last_h, q_agg_bw_last_h) = self.encoder.encode_bilstm(m_q,
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
            l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_s, self.start_answer)
            l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_e, self.end_answer)
            self.loss = tf.reduce_mean(l1 + l2)

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
        # this does all the create_feed_dict

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
        # input_feed[self.dropout] = dropout

        # grad_norm, param_norm
        output_feed = [self.updates, self.loss, self.grad_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, paragraph, paragraph_mask, question, question_mask, answer_start, answer_end):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """

        # A create_dict_dict helper may be a good idea ...
        input_feed = {}
        input_feed[self.paragraph] = paragraph
        input_feed[self.question] = question
        input_feed[self.paragraph_mask] = paragraph_mask
        input_feed[self.question_mask] = question_mask
        input_feed[self.start_answer] = answer_start
        input_feed[self.end_answer] = answer_end

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss

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

    def validate(self, session):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.
        This method calls self.test() which explicitly calculates validation cost.
        How you implement this function is dependent on how you design
        your data iteration function
        :return:
        """
        validate_prefix = self.FLAGS.data_dir + '/val.'
        context_file = validate_prefix + 'ids.context'
        question_file = validate_prefix + 'ids.question'
        answer_file = validate_prefix + 'span'

        valid_cost = 0

        for p, q, a in util.load_dataset(context_file, question_file, answer_file, self.FLAGS.batch_size,
                                         in_batches=True):
            a_s, a_e = zip(*a) # self.one_hot_func(a)
            q, q_mask = self.mask_and_pad(q, 'question')
            p, p_mask = self.mask_and_pad(p, 'paragraph')

            valid_cost += self.test(session, p, p_mask, q, q_mask, a_s, a_e)

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
            answer = p[a_s: a_e + 1]
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
                padded_s.extend([0] * pad_len)
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
        saver = tf.train.Saver()
        '''
        for e in range(self.FLAGS.epochs):
            batch_num = 0
            for p, q, a in util.load_dataset("data/squad/train.ids.context", "data/squad/train.ids.question",
                                             "data/squad/train.span", self.FLAGS.batch_size,
                                             self.FLAGS.max_paragraph_size, in_batches=True):
                a_s, a_e = zip(*a) # self.one_hot_func(a)
                q, q_mask = self.mask_and_pad(q, 'question')
                p, p_mask = self.mask_and_pad(p, 'paragraph')

                updates, loss, grad_norm = self.optimize(session, p, p_mask, q, q_mask, a_s, a_e)
                logging.info('Epoch: {}. Batch: {}. Loss:{}'.format(e, batch_num, loss))
                batch_num += 1
                print("Loss: " + str(loss) + " ------ Gradient Norm: " + str(grad_norm))

            saver.save(session, self.FLAGS.log_dir + '/model-weights', global_step=e)

            val_loss = self.validate(session)
            print(val_loss)

            self.evaluate_answer(session, q, p, sample=100)

        '''
        for e in range(80):

            p, q, a = util.load_single_dataset("data/squad/train.ids.context", "data/squad/train.ids.question",
                                             "data/squad/train.span", self.FLAGS.batch_size)
            a_s, a_e = zip(*a) # self.one_hot_func(a)
            q, q_mask = self.mask_and_pad(q, 'question')
            p, p_mask = self.mask_and_pad(p, 'paragraph')

            updates, loss, grad_norm = self.optimize(session, p, p_mask, q, q_mask, a_s, a_e)
            logging.info('Epoch: {}. Loss:{}'.format(e, loss))
            print("Loss: " + str(loss) + " ------ Gradient Norm: " + str(grad_norm))

