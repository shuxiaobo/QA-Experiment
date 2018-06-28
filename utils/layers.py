#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Desc:  
 @Author: Shane
 @Contact: iamshanesue@gmail.com  
 @Software: PyCharm  @since:python 3.6.4 
 @Created by Shane on 2018/6/27
 """
import tensorflow as tf
from functools import reduce
from operator import mul
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.util import nest

from tensorflow.python.ops import rnn_cell_impl

_linear = rnn_cell_impl._linear
VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


class MatchCell(RNNCell):
    """
    Match Rnn Cell
    match all the question word with the sigle context word. rebuild the question with attention and send the concat([question, context]) to rnncell
    tile state and x, the compute (Wx + Ws + Wq) + b, then softmax to attention update the question send the concat([question, context]) to rnncell
    """

    def __init__(self, cell, input_size, q_len):
        self._cell = cell
        self._input_size = input_size
        # FIXME : This won't be needed with good shape guessing
        self._q_len = q_len

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope = None):
        """
        :param inputs: [N, d + JQ + JQ * d] : input + q_mask + question
        :param state: [N, d]
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or self.__class__.__name__):
            c_prev, h_prev = state
            x = tf.slice(inputs, [0, 0], [-1, self._input_size])
            q_mask = tf.slice(inputs, [0, self._input_size], [-1, self._q_len])  # [N, JQ]
            qs = tf.slice(inputs, [0, self._input_size + self._q_len], [-1, -1])
            qs = tf.reshape(qs, [-1, self._q_len, self._input_size])  # [N, JQ, d]
            x_tiled = tf.tile(tf.expand_dims(x, 1), [1, self._q_len, 1])  # [N, JQ, d]
            h_prev_tiled = tf.tile(tf.expand_dims(h_prev, 1), [1, self._q_len, 1])  # [N, JQ, d]
            f = tf.tanh(linear([qs, x_tiled, h_prev_tiled], self._input_size, True, scope = 'f'))  # [N, JQ, d]
            a = tf.nn.softmax(exp_mask(linear(f, 1, True, squeeze = True, scope = 'a'), q_mask))  # [N, JQ]
            q = tf.reduce_sum(qs * tf.expand_dims(a, -1), 1)
            z = tf.concat([x, q], 1)  # [N, 2d]
            return self._cell(z, state[1])


def linear(args, output_size, bias, bias_start = 0.0, scope = None, squeeze = False, wd = 0.0, input_keep_prob = 1.0,
           is_train = None):
    with tf.variable_scope(scope):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        flat_args = [flatten(arg, 1) for arg in args]
        if input_keep_prob < 1.0:
            assert is_train is not None
            flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                         for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias)
        out = reconstruct(flat_out, args[0], 1)
        if squeeze:
            out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
        if wd:
            add_wd(wd)

    return out


def exp_mask(val, mask, name = None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name = name)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_wd(wd, scope = None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
    with tf.name_scope("weight_decay"):
        for var in variables:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name = "{}/wd".format(var.op.name))
            tf.add_to_collection('losses', weight_decay)


def padded_reshape(tensor, shape, mode = 'CONSTANT', name = None):
    paddings = [[0, shape[i] - tf.shape(tensor)[i]] for i in range(len(shape))]
    return tf.pad(tensor, paddings, mode = mode, name = name)


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


class ptr_net:
    """
    Pointer net:
        dropout(match)
        pointer the (question_sumed ,qc_match): -> B * Hidden, B * Len * Hidden
            score = W * tanh(W * [tile(question_sumed), qc_match]) -> B * Len * 1
            res = softmax(score) * qc_match
            return res, softmax(score)
        res = dropout(res)
        state = get last state of rnn(res, question_sumed)
        res2 = pointer the (question_sumed ,state)
        :return res , res2
    """

    def __init__(self, batch, hidden, keep_prob = 1.0, is_train = None, scope = "ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype = tf.float32), keep_prob = keep_prob, is_train = is_train)

    def __call__(self, init, match, d, mask):
        """
        :param init: B * Hidden
        :param match: B * Len * Hidden
        :param d:
        :param mask:
        :return:
        """
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob = self.keep_prob,
                              is_train = self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask)
            d_inp = dropout(inp, keep_prob = self.keep_prob,
                            is_train = self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask)
            return logits1, logits2


def dropout(args, keep_prob, is_train, mode = "recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape = noise_shape) * scale, lambda: args)
    return args


def pointer(inputs, state, hidden, mask, scope = "pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis = 1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis = 2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias = False, scope = "s0"))
        s = dense(s0, 1, use_bias = False, scope = "s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis = 2)
        res = tf.reduce_sum(a * inputs, axis = 1)
        return res, s1

def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res

def softmax_mask(val, mask):
    val = val * mask
    # return -float('inf') * (1 - tf.cast(mask, tf.float32)) + val
    return -VERY_SMALL_NUMBER * (1 - tf.cast(mask, tf.float32)) + val


def dense(inputs, hidden, use_bias = True, scope = "dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer = tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
