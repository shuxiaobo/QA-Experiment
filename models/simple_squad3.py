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
from utils.layers import *

from models.rc_base import RcBase
from utils.log import logger


class SimpleModelSQuad3(RcBase):
    """
    attention计算方式：
        1. 内容
            self attention
            c2q: lasthidden or outputs
            q2c: lasthidden or outputs
            interact and update with time or static
        2. 方法
            dot
            bilinear: qWp or qW + pW or v(qW + pW)
            取sum or max
    interact :
        1. 计算p中词在p中每个词的相似度attention
        2. 计算p中词与q的相似度attention， query-aware
        3. 句子级别attention算出中心句子？
        4.
、
    answer :
        pointer

    Loss:
        reduce_mean(log(p1) * answer_s + log(p2) * answer_e) + semantic loss
    """

    def create_model(self):
        """
        当前Model
        rnn q & p
        p2q atten1 :<p_emb_bi|q_emb_bi>
        self atten2 : W*d_emb_bi, W*q_emb_bi
        new_d_emb_bi : softmax(atten1 + atten2) * d_emb_bi
        rnn(new_d_emb_bi)
        :return:
        """
        num_layers = self.args.num_layers
        hidden_size = self.args.hidden_size
        cell = LSTMCell if self.args.use_lstm else GRUCell

        q_input = tf.placeholder(dtype = tf.int32, shape = [None, self.q_len], name = 'questions_bt')
        d_input = tf.placeholder(dtype = tf.int32, shape = [None, self.d_len], name = 'documents_bt')
        answer_s = tf.placeholder(dtype = tf.float32, shape = [None, None], name = 'answer_start')
        answer_e = tf.placeholder(dtype = tf.float32, shape = [None, None], name = 'answer_end')

        init_embed = tf.constant(self.embedding_matrix, dtype = tf.float32)
        embedding_matrix = tf.get_variable(name = 'embdding_matrix', initializer = init_embed, dtype = tf.float32)
        # char_embedding = tf.get_variable(name = 'can_embdding_matrix', initializer = init_embed, dtype = tf.float32,
        #                                  trainable = False)

        q_real_len = tf.reduce_sum(tf.sign(tf.abs(q_input)), axis = 1)
        d_real_len = tf.reduce_sum(tf.sign(tf.abs(d_input)), axis = 1)
        d_mask = tf.sequence_mask(dtype = tf.float32, maxlen = self.d_len, lengths = d_real_len)
        q_mask = tf.sequence_mask(dtype = tf.float32, maxlen = self.q_len, lengths = d_real_len)
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
            q_last_states = tf.concat(q_last_states, -1)

            logger("q_encoded_bf shape {}".format(q_emb_bi.get_shape()))

        with tf.variable_scope('d_encoder'):
            d_embed = tf.nn.embedding_lookup(embedding_matrix, d_input)

            d_rnn_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
            d_rnn_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])

            d_rnn_out, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_rnn_b, cell_fw = d_rnn_f, inputs = d_embed,
                                                                     sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
            d_emb_bi = tf.concat(d_rnn_out, axis = -1)
            self.d_emb_bi = d_emb_bi
            logger("d_encoded_bf shape {}".format(d_emb_bi.get_shape()))

        with tf.variable_scope('attention') as scp:
            dq_atten = tf.einsum('bij,bjk->bik', d_emb_bi, tf.transpose(q_emb_bi, perm = [0, 2, 1]))  # B * D * Q
            d_att = tf.reduce_sum(dq_atten, -1)
            q_att = tf.reduce_max(dq_atten, -2)
            d_atten_w = tf.get_variable('d_atten_w', shape = [d_emb_bi.get_shape()[-1], 1])
            q_atten_w = tf.get_variable(name = 'q_atten_w', shape = [q_last_states.get_shape()[-1], 1])
            d_emb_self_atten = tf.squeeze(tf.einsum('bij,jk->bik', d_emb_bi, d_atten_w), -1)
            q_emb_self_atten = tf.squeeze(tf.einsum('bij,jk->bik', q_emb_bi, q_atten_w), -1)
            # logger('d_emb_self_atten shape {}, d_att shape {}'.format(d_emb_self_atten.get_shape(), d_att.get_shape()))
            d_emb_att = tf.expand_dims(self.softmax_with_mask(tf.nn.relu(d_att + d_emb_self_atten), mask = d_mask, axis = -1), -1)
            q_emb_att = tf.expand_dims(self.softmax_with_mask(tf.nn.relu(q_att + q_emb_self_atten), mask = q_mask, axis = -1), -1)
            self.d_emb_self_atten = d_emb_self_atten
            self.d_att = d_att
            # self.sess.run([tf.squeeze(self.d_emb_att), self.d_emb_bi_attened_pre, self.d_att_rnn_out, self.d_att_rnn_out2, self.d_emb_bi_attened], data)
            self.d_emb_att = d_emb_att
            d_emb_bi_attened = d_emb_bi + d_emb_bi * d_emb_att
            q_emb_bi_attened = q_emb_bi + q_emb_bi * q_emb_att
            self.d_emb_bi_attened_pre = d_emb_bi_attened  # 这一层有问题，结果太小

        with tf.variable_scope("interact") as scp:
            # f_cell = MatchCell(cell(hidden_size), d_emb_bi_attened.get_shape()[-1].value, self.q_len)
            # f_init_state = f_cell.zero_state(tf.shape(d_emb_bi)[0], dtype = tf.float32)
            # b_cell = MatchCell(cell(hidden_size), d_emb_bi_attened.get_shape()[-1].value, self.q_len)
            # b_init_state = b_cell.zero_state(tf.shape(d_emb_bi)[0], dtype = tf.float32)
            # reshaped_q = tf.reshape(q_emb_bi_attened, [tf.shape(q_emb_bi_attened)[0], self.q_len * hidden_size * 2])
            # d_att_rnn_out, _ = tf.scan(
            #     fn = lambda pre, x: f_cell(tf.concat([tf.reshape(x, [-1, d_emb_bi_attened.get_shape()[-1].value]), q_mask, reshaped_q], -1),
            #                                pre, scope = 'f'),
            #     elems = [tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])],
            #     initializer = (tf.zeros(tf.shape(f_init_state)), f_init_state))
            # d_att_rnn_out2, _ = tf.scan(
            #     fn = lambda pre, x: b_cell(tf.concat([tf.reshape(x, [-1, d_emb_bi_attened.get_shape()[-1].value]), q_mask, reshaped_q], -1),
            #                                pre, scope = 'b'),
            #     elems = [tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])],
            #     initializer = (tf.zeros(tf.shape(b_init_state)), b_init_state))
            d_atten_cell_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = 1.) for _ in range(num_layers)])
            d_atten_cell_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = 1.) for _ in range(num_layers)])
            d_att_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_atten_cell_b, cell_fw = d_atten_cell_f,
                                                               inputs = tf.concat([d_emb_bi_attened, tf.tile(
                                                                   tf.transpose(q_last_states, perm = [1, 0, 2]),
                                                                   [1, d_emb_bi_attened.get_shape()[1], 1])], -1),
                                                               sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
            d_emb_bi_attened = tf.concat(d_att_rnn_out, -1)
            # d_emb_bi_attened = tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])
            # d_atten_cell_f = MultiRNNCell(
            #     cells = [
            #         DropoutWrapper(MatchCell(cell(hidden_size), d_emb_bi_attened.get_shape()[-1].value, self.q_len), output_keep_prob = 1.)
            #         for _ in range(num_layers)])
            # d_atten_cell_b = MultiRNNCell(
            #     cells = [
            #         DropoutWrapper(MatchCell(cell(hidden_size), d_emb_bi_attened.get_shape()[-1].value, self.q_len), output_keep_prob = 1.)
            #         for _ in range(num_layers)])
            # d_att_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_atten_cell_b, cell_fw = d_atten_cell_f,
            #                                                    inputs = tf.concat([d_emb_bi_attened,
            #                                                                        tf.tile(tf.expand_dims(q_mask, 1), [1, self.d_len, 1]),
            #                                                                        tf.tile(tf.expand_dims(reshaped_q, 1),
            #                                                                                [1, self.d_len, 1])], -1),
            #                                                    sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
            # d_emb_bi_attened = tf.concat([d_att_rnn_out, d_att_rnn_out2], -1)

            # self.d_att_rnn_out = d_att_rnn_out
            # self.d_att_rnn_out2 = d_att_rnn_out2
            # d_emb_bi_attened = tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])
            self.d_emb_bi_attened = d_emb_bi_attened

        with tf.variable_scope('pointer') as scp:
            pointer = ptr_net(batch = tf.shape(q_last_states)[1], hidden = q_last_states.get_shape().as_list(
            )[-1], keep_prob = self.args.keep_prob, is_train = tf.constant(True, dtype = tf.bool, shape = []))
            p1, p2 = pointer(tf.squeeze(q_last_states, axis = 0), d_emb_bi_attened, hidden_size, d_mask)
            p1 = tf.nn.softmax(p1)
            p2 = tf.nn.softmax(p2)
        #
        # with tf.variable_scope('answer') as scp:
        #     answer_s_w = tf.get_variable('answer_s_w', shape = [q_emb_bi_attened.get_shape()[1], 1])
        #     answer_e_w = tf.get_variable('answer_e_w', shape = [q_emb_bi_attened.get_shape()[1], 1])
        #     prob1 = self.softmax_with_mask(tf.reduce_sum(tf.einsum('bij,jk->bik', tf.einsum('bij,bjk->bik', d_emb_bi_attened,
        #                                                                                     tf.transpose(q_emb_bi_attened, perm = [0, 2,
        #                                                                                                                            1])) * tf.expand_dims(
        #         d_mask, -1), answer_s_w), -1), axis = -1, mask = d_mask)
        #     prob2 = self.softmax_with_mask(tf.reduce_sum(tf.einsum('bij,jk->bik', tf.einsum('bij,bjk->bik', d_emb_bi_attened,
        #                                                                                     tf.transpose(q_emb_bi_attened, perm = [0, 2,
        #                                                                                                                            1])) * tf.expand_dims(
        #         d_mask, -1), answer_e_w), -1), axis = -1, mask = d_mask)
        #     # answer_s_w = tf.get_variable('answer_s_w', shape = [d_emb_bi.get_shape()[-1], q_emb_bi.get_shape()[-1]])
        #     # answer_e_w = tf.get_variable('answer_e_w', shape = [d_emb_bi.get_shape()[-1], q_emb_bi.get_shape()[-1]])
        #     # prob1 = self.softmax_with_mask(tf.reduce_sum(tf.einsum('bij,bjk->bik', tf.einsum('bij,jk->bik', d_emb_bi_attened, answer_s_w),
        #     #                                                        tf.transpose(q_emb_bi_attened, perm = [0, 2, 1])) * tf.expand_dims(
        #     #     d_mask, -1), -1), axis = -1, mask = d_mask)
        #     # prob2 = self.softmax_with_mask(tf.reduce_sum(tf.einsum('bij,bjk->bik', tf.einsum('bij,jk->bik', d_emb_bi_attened, answer_e_w),
        #     #                                                        tf.transpose(q_emb_bi_attened, perm = [0, 2, 1])) * tf.expand_dims(
        #     #     d_mask, -1), -1), axis = -1, mask = d_mask)
        #
        #     epsilon = tf.convert_to_tensor(_EPSILON, prob1.dtype.base_dtype, name = "epsilon")
        #     prob1 = prob1 + (1 - d_mask)
        #     prob2 = prob2 + (1 - d_mask)
        #     p1 = tf.clip_by_value(prob1, epsilon, 1. - epsilon)
        #     p2 = tf.clip_by_value(prob2, epsilon, 1. - epsilon)

        self.p1 = p1
        self.p2 = p2
        # 如果使用log，那mask必须为1
        self.loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(p1), answer_s) + tf.multiply(tf.log(p2), answer_e)))

        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(
                tf.logical_and(
                    tf.equal(tf.argmax(answer_s, 1), tf.argmax(p1, 1)),
                    tf.equal(tf.argmax(answer_e, 1), tf.argmax(p2, 1))
                ), dtype = 'float'
            )))

    @staticmethod
    def softmax_with_mask(logits, axis, mask, epsilon = 10e-8, name = None):  # 1. normalize 2. softmax
        with tf.name_scope(name, 'softmax', [logits, mask]):
            max_axis = tf.reduce_max(logits, axis, keep_dims = True)
            target_exp = tf.exp(logits - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis, keep_dims = True)
            softmax = target_exp / (normalize + epsilon)
            logger("softmax shape {}".format(softmax.get_shape()))
            return softmax
