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
from tensorflow.contrib.layers import fully_connected

from models.rc_base import RcBase
from utils.log import logger


class Simple_modelrl(RcBase):
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

        with tf.variable_scope('d_encoder'):
            d_embed = tf.nn.embedding_lookup(embedding_matrix, d_input)

            d_rnn_f = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])
            d_rnn_b = MultiRNNCell(
                cells = [DropoutWrapper(cell(hidden_size), output_keep_prob = self.args.keep_prob) for _ in range(num_layers)])

            d_rnn_out, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw = d_rnn_b, cell_fw = d_rnn_f, inputs = d_embed,
                                                                     sequence_length = d_real_len, swap_memory = True, dtype = "float32", )
            d_emb_bi = tf.concat(d_rnn_out, axis = -1)
            logger("d_encoded_bf shape {}".format(d_emb_bi.get_shape()))

        with tf.variable_scope('attention_dq'):
            atten_d_q = tf.matmul(d_emb_bi, q_emb_bi, adjoint_b = True)
            atten_d = tf.reduce_sum(atten_d_q, axis = -1)
            attened_d_masked = self.softmax_with_mask(atten_d, axis = -1, mask = d_mask, name = 'attened_d_softmax')
            # there should be [None, seq_len, hidden_size]
            attened_d = tf.multiply(d_emb_bi, tf.expand_dims(attened_d_masked, -1))

        q_emb_rl = tf.concat([q_last_states[0][-1], q_last_states[1][-1]], axis = -1)
        memory = tf.concat([q_last_states[0][-1], q_last_states[1][-1]], axis = -1)
        memory_cell = cell(hidden_size * 4)
        m_state = memory_cell.zero_state(batch_size = tf.shape(d_emb_bi)[0], dtype = tf.float32)
        candi_embed = tf.nn.embedding_lookup(params = can_embedding_matrix, ids = candidate_idxs)
        result = None
        activ = 'tanh'

        # def reinforce(i, m_state, memory, result, activ = 'tanh'):
        #     i = tf.add(i, 1)
        #     position = tf.mod(i, d_real_len)
        #     hidden_d = tf.reshape(tf.gather(d_emb_bi, position, axis = 1), shape = [-1, d_emb_bi.get_shape()[-1]])
        #     x_context, m_state = memory_cell(tf.concat([memory, hidden_d], axis = -1),
        #                                      state = m_state)  # just use for gru cell, x = m_state
        #     # update memory: use the question and the context to update
        #     with tf.variable_scope('reinforce') as scp:
        #         context_and_q = tf.concat([x_context, q_emb_rl], axis = -1)
        #         rl_w = tf.get_variable(name = 'w', shape = [context_and_q.get_shape()[-1], 1])
        #         rl_b = tf.get_variable(name = 'b', shape = [1])
        #         if activ == 'tanh':
        #             rl_mul_context_q = tf.tanh(tf.matmul(context_and_q, rl_w))
        #         else:
        #             rl_mul_context_q = tf.nn.relu(tf.matmul(context_and_q, rl_w))
        #         out = tf.nn.softmax(
        #             logits = tf.add(rl_mul_context_q, rl_b))  # b * 1, Note: should use the bias here, while select_prob == 0 !!!!!
        #         select_prob.append(out)  # b * 1
        #         memory_update_w = tf.get_variable("memory_update_w", shape = [context_and_q.get_shape()[-1], memory.get_shape()[-1]])
        #         memory = tf.add(tf.matmul(tf.multiply(out, context_and_q), memory_update_w), memory)
        #     #     rl_w = tf.get_variable(name = 'w', shape = [context_and_q.get_shape()[-1], memory.get_shape()[-1]])
        #     #     rl_b = tf.get_variable(name = 'b', shape = [1])
        #     #     if activ == 'tanh':
        #     #         rl_mul_context_q = tf.tanh(tf.matmul(context_and_q, rl_w))
        #     #     else:
        #     #         rl_mul_context_q = tf.nn.relu(tf.matmul(context_and_q, rl_w))
        #     # out = tf.nn.softmax(logits = tf.add(rl_mul_context_q, rl_b))
        #     # memory = tf.multiply(out, memory)
        #     # return new_memory
        #
        #     # inference : use the new memory to inference the answer
        #     with tf.variable_scope('inference') as scp:
        #         infer_bilinear = tf.get_variable('infer_bilinear', shape = [memory.get_shape()[-1], candi_embed.get_shape()[-1]])
        #         pre_anw = tf.reduce_sum(tf.multiply(tf.transpose(candi_embed, [1, 0, 2]), tf.nn.relu(tf.matmul(memory, infer_bilinear))),
        #                                 axis = -1)
        #         pre_anw_pro = tf.nn.softmax(tf.transpose(pre_anw), dim = -1)
        #         answers.append(pre_anw_pro)  # b * can * 1
        #     result = tf.add(result, tf.multiply(out, pre_anw_pro))
        #     return i
        #
        # i = tf.constant(0)
        # i = tf.while_loop(cond = lambda x: x < 200, body = lambda x: reinforce(x, m_state, memory, result, 'tanh'), loop_vars = [i])

        for i in range(20):
            position = tf.stack([tf.range(0, tf.shape(d_real_len)[0], dtype = tf.int32),
                                 tf.mod(i, d_real_len)], axis = 1)  # Fuck, x.get_shape()[0] is not equal tf.shape(x)[0], fuck!!!

            hidden_d = tf.reshape(tf.gather_nd(attened_d, position), shape = [-1, d_emb_bi.get_shape()[-1]])
            x_context, m_state = memory_cell(tf.concat([memory, hidden_d], axis = -1),
                                             state = m_state)  # just use for gru cell, x = m_state
            # update memory: use the question and the context to update
            with tf.variable_scope('reinforce', reuse = tf.AUTO_REUSE) as scp:
                context_and_q = tf.concat([x_context, q_emb_rl], axis = -1)
                rl_w = tf.get_variable(name = 'w', shape = [context_and_q.get_shape()[-1], 1])
                rl_b = tf.get_variable(name = 'b', shape = [1])
                if activ == 'tanh':
                    rl_mul_context_q = tf.tanh(tf.matmul(context_and_q, rl_w))
                else:
                    rl_mul_context_q = tf.nn.relu(tf.matmul(context_and_q, rl_w))
                out = tf.nn.softmax(
                    logits = tf.add(rl_mul_context_q, rl_b))  # b * 1, Note: should use the bias here, while select_prob == 0 !!!!!
                memory_update_w = tf.get_variable("memory_update_w", shape = [context_and_q.get_shape()[-1], memory.get_shape()[-1]])
                memory = tf.add(tf.matmul(tf.multiply(out, context_and_q), memory_update_w), memory)
            # inference : use the new memory to inference the answer
            with tf.variable_scope('inference', reuse = tf.AUTO_REUSE) as scp:
                infer_bilinear = tf.get_variable('infer_bilinear', shape = [memory.get_shape()[-1], candi_embed.get_shape()[-1]])
                pre_anw = tf.reduce_sum(tf.multiply(tf.transpose(candi_embed, [1, 0, 2]), tf.nn.relu(tf.matmul(memory, infer_bilinear))),
                                        axis = -1)
                pre_anw_pro = tf.nn.softmax(tf.transpose(pre_anw), dim = -1)
            if i == 0:
                result = tf.multiply(out, pre_anw_pro)
            else:
                result = tf.add(result, tf.multiply(out, pre_anw_pro))

        epsilon = tf.convert_to_tensor(_EPSILON, tf.float32, name = "epsilon")
        result_prob = tf.clip_by_value(tf.nn.relu(result) / tf.reduce_sum(tf.nn.relu(result)), epsilon, 1. - epsilon)
        self.result = result_prob
        self.loss = tf.reduce_mean(
            -tf.reduce_sum(tf.multiply(tf.log(result_prob), y_true_idx)) + tf.reduce_sum(tf.multiply(result_prob, 1 - y_true_idx)))

        # self.loss = tf.reduce_mean(-tf.reduce_sum(y_true_idx * tf.log(candi_score_sfm), axis = -1))
        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(tf.equal(tf.argmax(y_true_idx, 1), tf.argmax(result, 1)), 'float')))
        # self.correct_prediction = tf.reduce_sum(
        #     tf.sign(tf.cast(tf.equal(tf.argmax(y_true_idx, 1), tf.argmax(answers[-1], 1)), 'float')))

    @staticmethod
    def softmax_with_mask(logits, axis, mask, epsilon = 10e-8, name = None):  # 1. normalize 2. softmax
        with tf.name_scope(name, 'softmax', [logits, mask]):
            max_axis = tf.reduce_max(logits, axis, keep_dims = True)
            target_exp = tf.exp(logits - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis, keep_dims = True)
            softmax = target_exp / (normalize + epsilon)
            logger("softmax shape {}".format(softmax.get_shape()))
            return softmax
