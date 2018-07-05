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


class SimpleModelSQuad4(RcBase):
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
        self.d_real_len = d_real_len

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

        with tf.variable_scope('attention_dq'):
            atten_q2d, atten_d2q = context_query_attention(context = d_emb_bi, query = q_emb_bi, scope = 'context_query_att',
                                                           reuse = None)
            attened_d = tf.concat([tf.multiply(d_emb_bi, atten_d2q), tf.multiply(d_emb_bi, atten_q2d), d_emb_bi], axis = -1)
            # computing c dot b
            # atten_d_q = tf.einsum('bij,bjk->bik', d_emb_bi, tf.transpose(q_emb_bi, perm = [0, 2, 1]))
            # atten_d = tf.reduce_sum(atten_d_q, axis = -1)
            # attened_d_masked = atten_d / tf.expand_dims(tf.reduce_sum(atten_d, -1), -1) * d_mask
            # there should be [None, seq_len, hidden_size]
            # attened_d = tf.multiply(d_emb_bi, tf.expand_dims(attened_d_masked,
            #                                                  -1))  # self.sess.run([self.atten_d, self.attened_d, self.result_s[-1], self.result_e[-1]], data)
        self.attened_d = attened_d
        q_emb_rl = q_last_states_con
        memory = tf.concat([q_last_states[0][-1][-1], q_last_states[1][-1][-1]], axis = -1)
        memory_cell = cell(hidden_size * 4)
        m_state = memory_cell.zero_state(batch_size = tf.shape(d_emb_bi)[0], dtype = tf.float32)
        candi_embed = d_embed
        result_ss = tf.zeros(shape = [tf.shape(d_emb_bi)[0], tf.shape(d_emb_bi)[1]])
        result_ee = tf.zeros(shape = [tf.shape(d_emb_bi)[0], tf.shape(d_emb_bi)[1]])
        activ = 'tanh'

        def inference(hidden_d, memory, m_state, result_s, result_e):
            # position = tf.stack([tf.range(0, tf.shape(d_real_len)[0], dtype = tf.int32),
            #                      tf.mod(i, d_real_len)], axis = 1)  # Fuck, x.get_shape()[0] is not equal tf.shape(x)[0], fuck!!!

            # hidden_d = tf.reshape(tf.gather_nd(attened_d, position), shape = [-1, d_emb_bi.get_shape()[-1]])
            x_context, m_state = memory_cell(tf.concat([memory, hidden_d], axis = -1),
                                             state = m_state)  # just use for gru cell, x = m_state
            # tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
            # Cause by GPU memory full
            # update memory: use the question and the context to update
            with tf.variable_scope('reinforce', reuse = tf.AUTO_REUSE) as scp:
                context_and_q = tf.concat([x_context, hidden_d, q_emb_rl], axis = -1)
                rl_w = tf.get_variable(name = 'w', shape = [context_and_q.get_shape()[-1], context_and_q.get_shape()[-1]])
                rl_b = tf.get_variable(name = 'b', shape = [1])
                if activ == 'tanh':
                    rl_mul_context_q = tf.tanh(tf.matmul(context_and_q, rl_w))
                else:
                    rl_mul_context_q = tf.nn.relu(tf.matmul(context_and_q, rl_w))
                out = tf.nn.tanh(rl_mul_context_q)  # b * 1, Note: should use the bias here, while select_prob == 0 !!!!!
                memory_update_w = tf.get_variable("memory_update_w", shape = [context_and_q.get_shape()[-1], memory.get_shape()[-1]])
                memory = tf.multiply(tf.nn.tanh(tf.matmul(out, memory_update_w)), memory)
            # inference : use the new memory to inference the answer
            with tf.variable_scope('inference', reuse = tf.AUTO_REUSE) as scp:
                context = tf.concat([memory, hidden_d, q_emb_rl], -1)
                infer_bilinear_start = tf.get_variable('infer_bilinear_start',
                                                       shape = [context.get_shape()[-1], candi_embed.get_shape()[-1]])
                pre_anw = tf.squeeze(
                    tf.einsum('bij,bjk->bik', candi_embed, tf.expand_dims(tf.matmul(context, infer_bilinear_start), -1)),
                    axis = -1)
                # pre_anw = tf.reduce_sum(
                #     tf.multiply(tf.transpose(candi_embed, [1, 0, 2]), tf.nn.relu(tf.matmul(context, infer_bilinear_start))), axis = -1)
                pre_anw_pro_s = pre_anw
                infer_bilinear_end = tf.get_variable('infer_bilinear_end', shape = [context.get_shape()[-1], candi_embed.get_shape()[-1]])
                pre_anw = tf.squeeze(
                    tf.einsum('bij,bjk->bik', candi_embed, tf.expand_dims(tf.matmul(context, infer_bilinear_end), -1)),
                    axis = -1)
                # pre_anw = tf.reduce_sum(
                #     tf.multiply(tf.transpose(candi_embed, [1, 0, 2]), tf.nn.relu(tf.matmul(context, infer_bilinear_end))), axis = -1)
                pre_anw_pro_e = pre_anw
                # pre_anw_pro = self.softmax_with_mask(tf.transpose(pre_anw), mask = d_mask, axis = -1)

            # with tf.variable_scope('interence_end', reuse = tf.AUTO_REUSE):
            logger("pre_anw_pro_s shape {}".format(pre_anw_pro_s.get_shape()))
            return [memory, m_state, pre_anw_pro_s, pre_anw_pro_e]

        _, self.m_state, result_s, result_e = tf.scan(fn = lambda pre, x: inference(tf.squeeze(x, 0), *pre),
                                                      elems = [tf.transpose(attened_d, perm = [1, 0, 2])],
                                                      initializer = [memory, m_state, result_ss, result_ee], name = 'scan',
                                                      swap_memory = True)
        # result_s = tf.reduce_sum(result_s, 0)
        # result_e = tf.reduce_sum(result_e, 0)
        # epsilon = tf.convert_to_tensor(_EPSILON, tf.float32, name = "epsilon")
        # result_prob_s = tf.clip_by_value(tf.nn.relu(result_s) / tf.reduce_sum(tf.nn.relu(result_s)), epsilon, 1. - epsilon)
        # result_prob_e = tf.clip_by_value(tf.nn.relu(result_e) / tf.reduce_sum(tf.nn.relu(result_e)), epsilon, 1. - epsilon)
        # self.logits_start = result_prob_s
        # self.logits_end = result_prob_e
        self.result_s = result_s[-1]
        self.result_e = result_e[-1]
        self.answer_s = answer_s
        self.answer_e = answer_e

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.result_s, labels = tf.argmax(answer_s, -1))
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.result_e, labels = tf.argmax(answer_e, -1))
        self.loss = tf.reduce_mean(losses)
        # 如果使用log，那mask必须为1
        # self.loss = -tf.reduce_mean( tf.reduce_sum(tf.multiply(tf.log(result_prob_s), answer_s) + tf.multiply(tf.log(result_prob_e), answer_e)))
        # self.add_loss(answer_s, answer_e)
        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(
                tf.logical_and(
                    tf.equal(tf.argmax(self.answer_s, 1, output_type = tf.int32), tf.argmax(self.result_s, -1, output_type = tf.int32)),
                    tf.equal(tf.argmax(self.answer_e, 1, output_type = tf.int32), tf.argmax(self.result_e, -1, output_type = tf.int32))
                ), dtype = 'float'
            )))

        self.begin_acc = tf.reduce_sum(
            tf.sign(tf.cast(tf.equal(tf.argmax(self.answer_s, 1, output_type = tf.int32), tf.argmax(self.result_s, -1, output_type = tf.int32)),
                            dtype = 'float')))
        self.end_acc = tf.reduce_sum(
            tf.sign(tf.cast(tf.equal(tf.argmax(self.answer_e, 1, output_type = tf.int32), tf.argmax(self.result_e, -1, output_type = tf.int32)),
                            dtype = 'float')))

    @staticmethod
    def softmax_with_mask(logits, axis, mask, epsilon = 10e-8, name = None):  # 1. normalize 2. softmax
        with tf.name_scope(name, 'softmax', [logits, mask]):
            max_axis = tf.reduce_max(logits, axis, keep_dims = True)
            target_exp = tf.exp(logits - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis, keep_dims = True)
            softmax = target_exp / (normalize + epsilon)
            # logger("softmax shape {}".format(softmax.get_shape()))
            return softmax
