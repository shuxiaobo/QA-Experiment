#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Desc:  
 @Author: Shane
 @Contact: iamshanesue@gmail.com  
 @Software: PyCharm  @since:python 3.6.4 
 @Created by Shane on 2018/5/28
 """

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, GRUCell, DropoutWrapper

from models.rc_base import RcBase
from utils.log import logger


class Simple_model1(RcBase):
    """
    """

    def create_model(self):
        num_layers = self.args.num_layers
        hidden_size = self.args.hidden_size
        cell = LSTMCell if self.args.use_lstm else GRUCell

        q_input = tf.placeholder(dtype = tf.int32, shape = [None, self.q_len], name = 'questions_bt')
        candidate_idxs = tf.placeholder(dtype = tf.int32, shape = [None, self.dataset.A_len], name = 'candidates_bi')
        d_input = tf.placeholder(dtype = tf.int32, shape = [None, self.d_len], name = 'documents_bt')

        y_true_idx = tf.placeholder(dtype = tf.float32, shape = [None, self.dataset.A_len], name = 'y_true_bi')

        init_embed = tf.constant(self.embedding_matrix, dtype = tf.float32)
        embedding_matrix = tf.get_variable(name = 'embdding_matrix', initializer = init_embed, dtype = tf.float32)
        can_embedding_matrix = tf.get_variable(name = 'can_embdding_matrix', initializer = init_embed, dtype = tf.float32,
                                               trainable = False)

        q_real_len = tf.reduce_sum(tf.sign(tf.abs(q_input)), axis = 1)
        d_real_len = tf.reduce_sum(tf.sign(tf.abs(d_input)), axis = 1)
        d_mask = tf.sequence_mask(dtype = tf.float32, maxlen = self.d_len, lengths = d_real_len)
        _EPSILON = 10e-8
        self.d_mask = d_mask

        with tf.variable_scope('q_encoder') as scp:
            q_embed = tf.nn.embedding_lookup(embedding_matrix, q_input)

            q_rnn_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
            q_rnn_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])

            outputs, q_last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = q_rnn_f, cell_bw = q_rnn_b, inputs = q_embed,
                                                                     sequence_length = q_real_len, initial_state_bw = None,
                                                                     dtype = "float32", parallel_iterations = None,
                                                                     swap_memory = True, time_major = False, scope = None)

            # last_states -> (output_state_fw, output_state_bw)
            # q_emb_bi = tf.concat([q_last_states[0][-1], q_last_states[1][-1]], axis = -1)
            q_emb_bi = tf.concat(outputs, axis = -1)

            logger("q_encoded_bf shape {}".format(q_emb_bi.get_shape()))

        # with tf.variable_scope('d_encoder'):
        #     d_embed = tf.nn.embedding_lookup(embedding_matrix, d_input)
        #
        #     d_rnn_f = MultiRNNCell(
        #         cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
        #     d_rnn_b = MultiRNNCell(
        #         cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
        #
        #     d_rnn_out, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_rnn_b, cell_fw = d_rnn_f, inputs = d_embed,
        #                                                              sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
        #
        #     d_emb_bi = tf.concat(d_rnn_out, axis = -1)
        #     logger("d_encoded_bf shape {}".format(d_emb_bi.get_shape()))

        def cell_fn(a, x):
            score = tf.matmul(a[1][0], q_last_states[0][-1])
            tf.assign(a[1][0], tf.multiply(score, a[1][0]))
            return d_rnn_f(x, a[1])

        with tf.variable_scope('d_encoder_new') as scp:
            d_rnn_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
            d_rnn_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
            d_embed = tf.nn.embedding_lookup(embedding_matrix, d_input)
            d_forward, d_forward_last = tf.scan(fn = cell_fn, elems = [tf.transpose(d_embed, [1, 0, 2])],
                                                initializer = (tf.zeros([self.args.batch_size, self.args.hidden_size]),
                                                               d_rnn_f.zero_state(batch_size = self.args.batch_size, dtype = tf.float32)))
            d_backward, d_backward_last = tf.scan(fn = cell_fn, elems = [tf.transpose(tf.reverse(d_embed, axis = 1), [1, 0, 2])],
                                                  initializer = (tf.zeros([self.args.batch_size, self.args.hidden_size]),
                                                                 d_rnn_b.zero_state(batch_size = self.args.batch_size, dtype = tf.float32)))
            d_emb_bi = tf.concat([d_forward, d_backward], axis = -1)
            logger("d_encoded_bf shape {}".format(d_emb_bi.get_shape()))

        with tf.variable_scope('attention_dq'):
            # atten_d_q = tf.matmul(d_emb_bi, tf.expand_dims(q_emb_bi, axis = -1), adjoint_b = False)
            atten_d_q = tf.matmul(d_emb_bi, q_emb_bi, adjoint_b = True)  # ？ * d_len * hidden, 这里由于数据稀疏问题，会造成数据都很小，整个atten_d_q都会很小。
            self.atten_d_q = atten_d_q
            atten_d = tf.reduce_sum(atten_d_q, axis = -1)
            attened_d_masked = self.softmax_with_mask(atten_d, axis = -1, mask = d_mask, name = 'attened_d_softmax')
            # attened_softmax = tf.nn.softmax(logits = attened_d_masked, name = 'attened_d_softmax', dim = -1)
            self.attened_d_masked = attened_d_masked
            # there should be [None, seq_len, hidden_size]
            attened_d = tf.multiply(d_emb_bi, tf.expand_dims(attened_d_masked, -1))
            self.attened_d = attened_d

        with tf.variable_scope('candidate'):
            candi_embed = tf.nn.embedding_lookup(params = can_embedding_matrix, ids = candidate_idxs)
            # [None, can_len, 1]
            # candi_score_d = tf.matmul(candi_embed, attened_d, transpose_b = True)
            can_w_qd = tf.get_variable(name = 'can_w_qd', dtype = tf.float32, shape = [self.args.hidden_size * 2, self.args.embedding_dim])
            sha = attened_d.get_shape()
            candi_score_d = tf.matmul(candi_embed,
                                      tf.reshape(tf.matmul(tf.reshape(attened_d, [-1, attened_d.get_shape()[-1]]), can_w_qd),
                                                 [-1, sha[1], self.args.embedding_dim]), transpose_b = True)
            candi_score = tf.reduce_mean(tf.nn.relu(candi_score_d), axis = -1)
            self.candi_score = candi_score
            # candi_score_sfm = tf.nn.log_softmax(logits = candi_score, name = 'candi_score_sfm', dim = -1)
            # manual computation of crossentropy
            output_bi = candi_score / tf.reduce_sum(candi_score, axis = -1, keep_dims = True)
            self.output_bi = output_bi
            epsilon = tf.convert_to_tensor(_EPSILON, output_bi.dtype.base_dtype, name = "epsilon")
            candi_score_sfm = tf.clip_by_value(output_bi, epsilon, 1. - epsilon)
            self.candi_score_sfm = candi_score_sfm

        self.loss = tf.reduce_mean(
            -tf.reduce_sum(y_true_idx * tf.log(candi_score_sfm), axis = -1) + tf.reduce_sum((1 - y_true_idx) * candi_score_sfm))
        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(tf.equal(tf.argmax(y_true_idx, 1), tf.argmax(candi_score_sfm, 1)), 'float')))

    @staticmethod
    def softmax_with_mask(logits, axis, mask, epsilon = 10e-8, name = None):  # 1. normalize 2. softmax
        with tf.name_scope(name, 'softmax', [logits, mask]):
            max_axis = tf.reduce_max(logits, axis, keep_dims = True)
            target_exp = tf.exp(logits - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis, keep_dims = True)
            softmax = target_exp / (normalize + epsilon)
            return softmax
