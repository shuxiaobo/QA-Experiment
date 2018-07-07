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

from models.attention_wrapper import *

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
            general qWp
            concat W[q,p]
            perceptron: qW + pW or v*tanh(qW + pW)
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
                                                                     swap_memory = True, time_major = False, scope = 'd_rnn')

            q_emb_bi = tf.concat(outputs, axis = -1)
            if self.args.use_lstm:
                q_last_states_con = tf.concat([q_last_states[0][-1][-1], q_last_states[1][-1][-1]], axis = -1)
            else:
                q_last_states_con = tf.concat(q_last_states, -1)

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

        # with tf.variable_scope('attention') as scp:
        #     dq_atten = tf.einsum('bij,bjk->bik', d_emb_bi, tf.transpose(q_emb_bi, perm = [0, 2, 1]))  # B * D * Q
        #     d_att = tf.reduce_sum(dq_atten, -1)
        #     q_att = tf.reduce_max(dq_atten, -2)
        #     d_atten_w = tf.get_variable(name = 'd_atten_w', shape = [d_emb_bi.get_shape()[-1], d_emb_bi.get_shape()[-1]])
        #     q_atten_w = tf.get_variable(name = 'q_atten_w', shape = [q_last_states_con.get_shape()[-1], q_last_states_con.get_shape()[-1]])
        #     d_emb_self_atten = tf.reduce_sum(
        #         tf.einsum('bij,bjk->bik', tf.einsum('bij,jk->bik', d_emb_bi, d_atten_w), tf.transpose(d_emb_bi, perm = [0, 2, 1])), -1)
        #     q_emb_self_atten = tf.reduce_sum(tf.einsum('bij,jk->bik', q_emb_bi, q_atten_w), -1)
        #     d_emb_att = tf.expand_dims(tf.nn.tanh(d_att + d_emb_self_atten) * d_mask, -1)
        #     q_emb_att = tf.expand_dims(tf.nn.tanh(q_att + q_emb_self_atten) * q_mask, -1)
        #     # self.sess.run([tf.squeeze(self.d_emb_att), self.d_emb_bi_attened_pre, self.d_att_rnn_out, self.d_att_rnn_out2, self.d_emb_bi_attened], data)
        #     d_emb_bi = d_emb_bi * d_emb_att
        #     q_emb_bi = q_emb_bi * q_emb_att
        # #     self.d_emb_bi_attened_pre = d_emb_bi_attened  # 这一层有问题，结果太小
        with tf.variable_scope('attention_dq'):
            atten_q2d, atten_d2q = context_query_attention(context = d_emb_bi, query = q_emb_bi, scope = 'context_query_att',
                                                       reuse = None)
            d_emb_bi_con = tf.concat([tf.add(d_emb_bi, atten_d2q), tf.add(d_emb_bi, atten_q2d), d_emb_bi], axis = -1)
            d_emb_bi_attened = d_emb_bi_con
            # attented_d_w = tf.get_variable('attented_d_w', shape = [d_emb_bi_con.get_shape()[-1], d_emb_bi.get_shape()[-1]])
            # d_emb_bi_attened = tf.einsum('bij,jk->bik', d_emb_bi_con, attented_d_w)
            atten_q2d, atten_d2q = context_query_attention(context =q_emb_bi , query = d_emb_bi, scope = 'context_query_att',
                                                           reuse = True)
            q_emb_bi_con = tf.concat([tf.add(q_emb_bi, atten_d2q), tf.add(q_emb_bi, atten_q2d), q_emb_bi], axis = -1)
            q_emb_bi_attened = q_emb_bi_con

            # 只用这一层 + pointer, 学不到东西。

        """
        貌似loss由整个模型的结构和lr决定，acc由一些细节决定
        """
        with tf.variable_scope("match_rnn") as scp:
            # Match RNN via attention wrapper
            # with tf.variable_scope("match_lstm_attender"):
            #     match_lstm_cell_attention_fn = lambda curr_input, state: tf.concat([curr_input, state], axis = -1)
            #     attention_mechanism_match_lstm = BahdanauAttention(q_emb_bi_con.get_shape()[-1], q_emb_bi_con, memory_sequence_length = q_real_len)
            #     cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple = True)
            #     lstm_attender = AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention = False,
            #                                      attention_input_fn = match_lstm_cell_attention_fn)
            #
            #     # we don't mask the passage because masking the memories will be handled by the pointerNet
            #
            #     output_attender, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_attender, cell_bw = lstm_attender, inputs = d_emb_bi_attened, dtype = tf.float32, swap_memory=True, scope = "rnn")
            #     d_emb_bi_attened = tf.concat(output_attender, axis = -1)

            # Match RNN via tf.scan
            f_cell = MatchCell(cell(hidden_size), d_emb_bi_attened.get_shape()[-1].value, self.q_len)
            f_init_state = f_cell.zero_state(tf.shape(d_emb_bi)[0], dtype = tf.float32)
            b_cell = MatchCell(cell(hidden_size), d_emb_bi_attened.get_shape()[-1].value, self.q_len)
            b_init_state = b_cell.zero_state(tf.shape(d_emb_bi)[0], dtype = tf.float32)
            reshaped_q = tf.reshape(q_emb_bi_attened, [tf.shape(q_emb_bi_attened)[0], -1])
            d_att_rnn_out, _ = tf.scan(
                fn = lambda pre, x: f_cell(tf.concat([tf.reshape(x, [-1, d_emb_bi_attened.get_shape()[-1].value]), q_mask, reshaped_q], -1),
                                           pre[1], scope = 'fo'),
                elems = [tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])],
                initializer = (tf.zeros(tf.shape(f_init_state[1])), f_init_state),swap_memory=True)
            d_att_rnn_out2, _ = tf.scan(
                fn = lambda pre, x: b_cell(tf.concat([tf.reshape(x, [-1, d_emb_bi_attened.get_shape()[-1].value]), q_mask, reshaped_q], -1),
                                           pre[1], scope = 'b'),
                elems = [tf.reverse_sequence(tf.transpose(d_emb_bi_attened, perm = [1, 0, 2]), d_real_len,
                                             seq_axis = 0,
                                             batch_axis = 1)],
                initializer = (tf.zeros(tf.shape(b_init_state[1])), b_init_state), swap_memory=True)
            d_emb_bi_attened = tf.concat([d_att_rnn_out, d_att_rnn_out2], -1)
            d_emb_bi_attened = tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])

            # 普通的RNN
            # d_atten_cell_f = MultiRNNCell(
            #     cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = 1.) for _ in range(num_layers)])
            # d_atten_cell_b = MultiRNNCell(
            #     cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = 1.) for _ in range(num_layers)])
            # d_att_rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_atten_cell_b, cell_fw = d_atten_cell_f,
            #                                                    inputs = tf.concat([d_emb_bi_attened, tf.tile(
            #                                                        tf.transpose(q_last_states_con, perm = [1, 0, 2]),
            #                                                        [1, d_emb_bi_attened.get_shape()[1], 1])], -1),
            #                                                    sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
            # d_emb_bi_attened = tf.concat(d_att_rnn_out, -1)
            # d_emb_bi_attened = tf.transpose(d_emb_bi_attened, perm = [1, 0, 2])

            # Match RNN via dynamic rnn
            # reshaped_q = tf.reshape(q_emb_bi_attened, [tf.shape(q_emb_bi_attened)[0], q_emb_bi_attened.get_shape()[1] *  q_emb_bi_attened.get_shape()[2]])
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
            # d_emb_bi_attened = tf.concat(d_att_rnn_out, -1)

            self.d_emb_bi_attened = d_emb_bi_attened

        with tf.variable_scope('pointer') as scp:
            pointer = ptr_net(batch = tf.shape(q_last_states_con)[0], hidden = q_last_states_con.get_shape().as_list(
            )[-1], keep_prob = self.args.keep_prob, is_train = tf.constant(True, dtype = tf.bool, shape = []))
            p1, p2 = pointer(q_last_states_con, d_emb_bi_attened, hidden_size, d_mask)
            self.p1, self.p2 = p1, p2
        self.answer_s = answer_s
        self.answer_e = answer_e  # self.sess.run([tf.equal(tf.argmax(self.answer_s, 1), tf.argmax(self.p1, 1)), tf.equal(tf.argmax(self.answer_e, 1), tf.argmax(self.p2, 1)),tf.argmax(self.p1, 1),tf.argmax(self.answer_s, 1), self.p1],data)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.p1, labels = tf.argmax(self.answer_s, -1))
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.p2, labels = tf.argmax(self.answer_e, -1))
        self.loss = tf.reduce_mean(losses)

        # self.loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(p1), answer_s) + tf.multiply(tf.log(p2), answer_e)))

        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(
                tf.logical_and(
                    tf.equal(tf.argmax(self.answer_s, 1, output_type = tf.int32), tf.argmax(self.p1, -1, output_type = tf.int32)),
                    tf.equal(tf.argmax(self.answer_e, 1, output_type = tf.int32), tf.argmax(self.p2, -1, output_type = tf.int32))
                ), dtype = 'float'
            )))

        self.begin_acc = tf.reduce_sum(
            tf.sign(
                tf.cast(tf.equal(tf.argmax(self.answer_s, 1, output_type = tf.int32), tf.argmax(self.p1, -1, output_type = tf.int32)),
                        dtype = 'float')))
        self.end_acc = tf.reduce_sum(
            tf.sign(
                tf.cast(tf.equal(tf.argmax(self.answer_e, 1, output_type = tf.int32), tf.argmax(self.p2, -1, output_type = tf.int32)),
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
