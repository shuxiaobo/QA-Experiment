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
            return self._cell(z, state)


class MatchAddCell(RNNCell):
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
            return self._cell(z, h_prev)


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


def summ(memory, hidden, mask, keep_prob = 1.0, is_train = None, scope = "summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob = keep_prob, is_train = is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope = "s0"))
        s = dense(s0, 1, use_bias = False, scope = "s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis = 2)
        res = tf.reduce_sum(a * memory, axis = 1)
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


from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.
    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.
    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob = self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob = self.keep_prob)

    def build_graph(self, inputs, masks, input_lens):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.
        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with tf.variable_scope("RNNEncoder"):
            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens,
                                                                  dtype = tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.
        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with tf.variable_scope("SimpleSoftmaxLayer"):
            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs = 1, activation_fn = None)  # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis = [2])  # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.
    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".
    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.
    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.
        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with tf.variable_scope("BasicAttn"):
            # Calculate attention distribution
            values_t = tf.transpose(values, perm = [0, 2, 1])  # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t)  # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1)  # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask,
                                          2)  # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values)  # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.
    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax
    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def context_query_attention(context, query, scope = "context_query_att", reuse = None):
    '''
    Defines a context-query attention layer
    This layer computes both the context-to-query attention and query-to-context attention
    '''
    # dimensions=[B, N, d] ([batch_size, max_words_context, word_dimension])
    B, N, d = context.get_shape().as_list()
    # dimensions=[B, M, d] ([batch_size, max_words_question, word_dimension])
    B, M, d = query.get_shape().as_list()
    with tf.variable_scope(scope, reuse = reuse):
        # apply manual broadcasting to compute pair wise trilinear similarity score
        # trilinear similarity score is computed between all pairs of context words and question words
        # dimensions=[B, N, d] -> [B, N, M, d]
        context_expand = tf.tile(tf.expand_dims(context, 2), [1, 1, M, 1])
        # dimensions=[B, M, d] -> [B, N, M, d]
        query_expand = tf.tile(tf.expand_dims(query, 1), [1, N, 1, 1])
        # concat(q, c, (q)dot(c)) which is the input to the trilinear similarity score computation function
        mat = tf.concat((query_expand, context_expand, query_expand * context_expand), axis = 3)
        # apply trilinear function as a linear dense layer
        # dimensions=[B, N, M, 1]
        similarity = tf.layers.dense(mat, 1, name = "dense", reuse = reuse)
        # dimensions=[B, N, M]
        similarity = tf.squeeze(similarity, -1)
        # normalizing by applying softmax over rows of similarity matrix
        similarity_row_normalized = tf.nn.softmax(similarity, dim = 1)
        # normalizing by applying softmax over columns of similarity matrix
        similarity_column_normalized = tf.nn.softmax(similarity, dim = 2)
        # computing A = S_bar X Question
        matrix_a = tf.matmul(similarity_row_normalized, query)
        # computing B = S_bar X S_double_bar X Context
        matrix_b = tf.matmul(tf.matmul(similarity_row_normalized, tf.transpose(similarity_column_normalized, [0, 2, 1])), context)
        return matrix_a, matrix_b
