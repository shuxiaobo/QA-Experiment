#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Desc:  
 @Author: Shane
 @Contact: iamshanesue@gmail.com  
 @Software: PyCharm  @since:python 3.6.4 
 @Created by Shane on 2018/6/12
 """

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, GRUCell, DropoutWrapper
from tensorflow.contrib.layers import fully_connected

from models.rc_base import RcBase
from utils.log import logger

"""
this implement is for BiDAF.
"""


class BiDAF(RcBase):
    """
    """

    def create_model(self):
        num_layers = self.args.num_layers
        hidden_size = self.args.hidden_size
        char_hidden_size = self.args.char_hidden_size
        char_embedding_dim = self.args.char_embedding_dim
        cell = LSTMCell if self.args.use_lstm else GRUCell

        q_input = tf.placeholder(dtype = tf.int32, shape = [None, self.q_len], name = 'questions_bt')
        d_input = tf.placeholder(dtype = tf.int32, shape = [None, self.d_len], name = 'documents_bt')
        answer_s = tf.placeholder(dtype = tf.float32, shape = [None, None], name = 'answer_start')
        answer_e = tf.placeholder(dtype = tf.float32, shape = [None, None], name = 'answer_end')
        q_input_char = tf.placeholder(dtype = tf.int32, shape = [None, self.q_len, self.q_char_len], name = 'questions_bt_char')
        d_input_char = tf.placeholder(dtype = tf.int32, shape = [None, self.d_len, self.d_char_len], name = 'documents_bt_char')

        init_embed = tf.constant(self.embedding_matrix, dtype = tf.float32)
        embedding_matrix = tf.get_variable(name = 'embdding_matrix', initializer = init_embed, dtype = tf.float32)
        # can_embedding_matrix = tf.get_variable(name = 'can_embdding_matrix', initializer = init_embed, dtype = tf.float32,
        #                                        trainable = False)

        q_real_len = tf.reduce_sum(tf.sign(tf.abs(q_input)), axis = 1)
        d_real_len = tf.reduce_sum(tf.sign(tf.abs(d_input)), axis = 1)
        d_mask = tf.sequence_mask(dtype = tf.float32, maxlen = self.d_len, lengths = d_real_len)
        q_mask = tf.sequence_mask(dtype = tf.float32, maxlen = self.q_len, lengths = d_real_len)
        _EPSILON = 10e-8

        batch_size = tf.shape(q_input)[0]

        if self.args.use_char_embedding:
            char_embedding = tf.get_variable(name = 'can_embdding_matrix',
                                             initializer = tf.constant(self.char_embedding_matrix, dtype = tf.float32), dtype = tf.float32,
                                             trainable = True)

            with tf.variable_scope('char_embedding', reuse = tf.AUTO_REUSE) as scp:
                q_char_embed = tf.nn.embedding_lookup(char_embedding, q_input_char)  # B * Q * C * emb
                d_char_embed = tf.nn.embedding_lookup(char_embedding, d_input_char)  # B * D * C * emb

                # q_char_embed = tf.reshape(q_char_embed, [-1, self.q_len, self.d_char_len * char_embedding_dim])  # B * Q * C * emb
                # d_char_embed = tf.reshape(d_char_embed, [-1, self.d_len, self.q_char_len * char_embedding_dim])  # B * D * C * emb
                # char_rnn_f = MultiRNNCell(
                #     cells = [DropoutWrapper(cell(char_hidden_size), output_keep_prob = self.args.keep_prob)])
                # char_rnn_b = MultiRNNCell(
                #     cells = [DropoutWrapper(cell(char_hidden_size), output_keep_prob = self.args.keep_prob)])
                #
                # d_char_embed_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = char_rnn_f, cell_bw = char_rnn_b, inputs = d_char_embed,
                #                                                       sequence_length = d_real_len, initial_state_bw = None,
                #                                                       dtype = "float32", parallel_iterations = None,
                #                                                       swap_memory = True, time_major = False, scope = 'char_rnn')
                # q_char_embed_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = char_rnn_f, cell_bw = char_rnn_b, inputs = q_char_embed,
                #                                                       sequence_length = q_real_len, initial_state_bw = None,
                #                                                       dtype = "float32", parallel_iterations = None,
                #                                                       swap_memory = True, time_major = False, scope = 'char_rnn')

                q_char_embed = tf.nn.dropout(q_char_embed, keep_prob = self.args.keep_prob)
                d_char_embed = tf.nn.dropout(d_char_embed, keep_prob = self.args.keep_prob)
                with tf.variable_scope('char_conv', reuse = tf.AUTO_REUSE) as scp:

                    q_char_embed = tf.transpose(q_char_embed, perm = [0, 2, 3, 1])  # [batch, height, width, channels]
                    filter = tf.get_variable('q_filter_w',
                                             shape = [5, 5, self.q_len,
                                                      self.q_len])  # [filter_height, filter_width, in_channels, out_channels]
                    cnned_char = tf.nn.conv2d(q_char_embed, filter, strides = [1, 1, 1, 1], padding = 'VALID', use_cudnn_on_gpu = True,
                                              data_format = "NHWC",
                                              name = None)  # [B, (char_len-filter_size/stride), (word_len-filter_size/stride), d_len]

                    q_char_embed_out = tf.nn.max_pool(cnned_char, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'VALID',
                                                      data_format = "NHWC",
                                                      name = None)

                    char_out_size = q_char_embed_out.get_shape().as_list()[1] * q_char_embed_out.get_shape().as_list()[2]
                    q_char_embed_out = tf.reshape(tf.transpose(q_char_embed_out, perm = [0, 3, 1, 2]), shape = [batch_size, self.q_len, char_out_size])

                    d_char_embed = tf.transpose(d_char_embed, perm = [0, 2, 3, 1])  # [batch, height, width, channels]
                    filter = tf.get_variable('d_filter_w',
                                             shape = [5, 5, self.d_len,
                                                      self.d_len])  # [filter_height, filter_width, in_channels, out_channels]
                    cnned_char = tf.nn.conv2d(d_char_embed, filter, strides = [1, 1, 1, 1], padding = 'VALID', use_cudnn_on_gpu = True,
                                              data_format = "NHWC",
                                              name = None)  # [B, (char_len-filter_size/stride), (word_len-filter_size/stride), d_len]

                    d_char_embed_out = tf.nn.max_pool(cnned_char, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'VALID', data_format = "NHWC",
                                                      name = None)
                    char_out_size = d_char_embed_out.get_shape().as_list()[1] * d_char_embed_out.get_shape().as_list()[2]
                    d_char_embed_out = tf.reshape(tf.transpose(d_char_embed_out, perm = [0, 3, 1, 2]),
                                                  shape = [batch_size, self.d_len, char_out_size])

                    d_char_embed_out = tf.reshape(d_char_embed_out, shape = [batch_size, self.d_len, char_out_size])

                d_char_out = tf.concat(d_char_embed_out, -1)
                q_char_out = tf.concat(q_char_embed_out, -1)

        with tf.variable_scope('q_encoder') as scp:
            q_embed = tf.nn.embedding_lookup(embedding_matrix, q_input)

            if self.args.use_char_embedding:
                q_embed = tf.concat([q_embed, q_char_out], -1)
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

        with tf.variable_scope('d_encoder'):
            d_embed = tf.nn.embedding_lookup(embedding_matrix, d_input)

            if self.args.use_char_embedding:
                d_embed = tf.concat([d_embed, d_char_out], -1)

            d_rnn_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
            d_rnn_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])

            d_rnn_out, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_rnn_b, cell_fw = d_rnn_f, inputs = d_embed,
                                                                     sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
            d_emb_bi = tf.concat(d_rnn_out, axis = -1)
            logger("d_encoded_bf shape {}".format(d_emb_bi.get_shape()))

        # def attention1(x, y, w):
        #     return tf.squeeze(tf.scan(fn = lambda pre, xx: tf.squeeze(tf.concat([xx, y, tf.multiply(xx, y)], axis = -1)) @ w, elems = [x],
        #                               initializer = tf.zeros(shape = [tf.shape(y)[0], 1])), axis = -1)

        with tf.variable_scope('ctq_att'):
            ctq_w = tf.get_variable(shape = [hidden_size * 6, 1], name = 'ctq_w')
            # dq_dot = tf.scan(fn = lambda pre, x: attention1(tf.transpose(d_emb_bi, perm = [1, 0, 2]), x, ctq_w),
            #                  elems = [tf.transpose(q_emb_bi, perm = [1, 0, 2])],
            #                  initializer = tf.zeros(shape = [self.d_len, tf.shape(q_emb_bi)[1]]))  # should be Q * D * B
            # dq_dot = tf.transpose(dq_dot, perm = [0, 2, 1]) # Q * B * D
            d_expanded = tf.tile(tf.expand_dims(d_emb_bi, 2), [1, 1, self.q_len, 1])
            q_expanded = tf.tile(tf.expand_dims(q_emb_bi, 1), [1, self.d_len, 1, 1])
            dq_dot = tf.concat([d_expanded, q_expanded, d_expanded * q_expanded], axis = -1)
            dq_dot = tf.squeeze(tf.tensordot(dq_dot, ctq_w, axes = ((-1,), (0,))), axis = -1)
            dq_dot_softmax = self.softmax_with_mask(logits = dq_dot, axis = 2,
                                                    mask = tf.tile(tf.expand_dims(q_mask, axis = 1), [1, self.d_len, 1]))  # Q * B
            U_hat = tf.einsum("bij,bjk->bik", dq_dot_softmax, q_emb_bi) # B * D * hidden*2
            # U_hat = tf.transpose(U_hat, [1, 0, 2])
            max_atten = self.softmax_with_mask(tf.reduce_max(dq_dot, axis = -1), mask = d_mask, axis = -1)  # B * D
            H_hat = tf.tile(tf.expand_dims(tf.reduce_sum(tf.multiply(tf.expand_dims(max_atten, axis = -1), d_emb_bi), 1), axis = 1),
                            [1, self.d_len, 1])  # B * D * hidden*2,

            G_belta = tf.concat([d_emb_bi, U_hat, d_emb_bi * U_hat, d_emb_bi * H_hat], axis = -1)

        with tf.variable_scope('model_layer') as scp:
            model_cell_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob)])
            model_cell_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob)])

            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = model_cell_f, cell_bw = model_cell_b, inputs = G_belta,
                                                                   sequence_length = d_real_len, swap_memory = True, dtype = 'float32')
            M = tf.concat(outputs, axis = -1)

        with tf.variable_scope('output_layer') as scp:
            w_p_1 = tf.get_variable('w_p_1', shape = [hidden_size * 10, 1])
            out_cell_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob)])
            out_cell_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob)])

            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = out_cell_f, cell_bw = out_cell_b, inputs = M,
                                                                   sequence_length = d_real_len, dtype = 'float32')
            M_2 = tf.concat(outputs, axis = -1)
            w_p_2 = tf.get_variable('w_p_2', shape = [hidden_size * 10, 1])

            p1 = self.softmax_with_mask(
                logits = tf.reshape(tf.matmul(tf.reshape(tf.concat([G_belta, M], -1), [-1, hidden_size * 10]), w_p_1), [-1, self.d_len]),
                axis = -1, mask = d_mask)
            self.result_s = p1
            p2 = self.softmax_with_mask(
                logits = tf.reshape(tf.matmul(tf.reshape(tf.concat([G_belta, M_2], -1), [-1, hidden_size * 10]), w_p_2), [-1, self.d_len]),
                axis = -1, mask = d_mask)
            self.result_e = p2
        self.answer_s = answer_s
        self.answer_e = answer_e
        epsilon = tf.convert_to_tensor(_EPSILON, p1.dtype.base_dtype, name = "epsilon")
        p1 = tf.clip_by_value(p1, epsilon, 1. - epsilon)
        p2 = tf.clip_by_value(p2, epsilon, 1. - epsilon)
        self.p1 = p1
        self.p2 = p2
        # self.loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(p1), answer_s) + tf.multiply(tf.log(p2), answer_e)))
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.p1, labels = tf.argmax(self.answer_s, -1))
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.p2, labels = tf.argmax(self.answer_e, -1))
        self.loss = tf.reduce_mean(losses)

        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(
                tf.logical_and(
                    tf.equal(tf.argmax(self.answer_s, 1, output_type = tf.int32), tf.argmax(self.result_s, -1, output_type = tf.int32)),
                    tf.equal(tf.argmax(self.answer_e, 1, output_type = tf.int32), tf.argmax(self.result_e, -1, output_type = tf.int32))
                ), dtype = 'float'
            )))

        self.begin_acc = tf.reduce_sum(
            tf.sign(
                tf.cast(tf.equal(tf.argmax(self.answer_s, 1, output_type = tf.int32), tf.argmax(self.result_s, -1, output_type = tf.int32)),
                        dtype = 'float')))
        self.end_acc = tf.reduce_sum(
            tf.sign(
                tf.cast(tf.equal(tf.argmax(self.answer_e, 1, output_type = tf.int32), tf.argmax(self.result_e, -1, output_type = tf.int32)),
                        dtype = 'float')))

    @staticmethod
    def softmax_with_mask(logits, axis, mask, epsilon = 10e-8, name = None):  # 1. normalize 2. softmax
        with tf.name_scope(name, 'softmax', [logits, mask]):
            max_axis = tf.reduce_max(logits, axis, keep_dims = True)
            target_exp = tf.exp(logits - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis, keep_dims = True)
            softmax = target_exp / (normalize + epsilon)
            logger("softmax shape {}".format(softmax.get_shape()))
            return softmax


def gated_attention(doc, qry, inter, mask, gating_fn = 'tf.multiply'):
    # doc: B x N x D
    # qry: B x Q x D
    # inter: B x N x Q
    # mask (qry): B x Q
    alphas_r = tf.nn.softmax(inter) * tf.cast(tf.expand_dims(mask, axis = 1), tf.float32)
    alphas_r = alphas_r / tf.expand_dims(tf.reduce_sum(alphas_r, axis = 2), axis = -1)  # B x N x Q
    q_rep = tf.matmul(alphas_r, qry)  # B x N x D
    return eval(gating_fn)(doc, q_rep)


def pairwise_interaction(doc, qry):
    # doc: B x N x D
    # qry: B x Q x D
    shuffled = tf.transpose(qry, perm = [0, 2, 1])  # B x D x Q
    return tf.matmul(doc, shuffled)  # B x N x Q


def attention_sum(doc, qry, cand, cloze, cand_mask = None):
    # doc: B x N x D
    # qry: B x Q x D
    # cand: B x N x C
    # cloze: B x 1
    # cand_mask: B x N
    idx = tf.concat(
        [tf.expand_dims(tf.range(tf.shape(qry)[0]), axis = 1),
         tf.expand_dims(cloze, axis = 1)], axis = 1)
    q = tf.gather_nd(qry, idx)  # B x D
    p = tf.squeeze(
        tf.matmul(doc, tf.expand_dims(q, axis = -1)), axis = -1)  # B x N
    pm = tf.nn.softmax(p) * tf.cast(cand_mask, tf.float32)  # B x N
    pm = pm / tf.expand_dims(tf.reduce_sum(pm, axis = 1), axis = -1)  # B x N
    pm = tf.expand_dims(pm, axis = 1)  # B x 1 x N
    return tf.squeeze(
        tf.matmul(pm, tf.cast(cand, tf.float32)), axis = 1)  # B x C
