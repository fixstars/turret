# -*- coding: utf-8 -*-
import numpy as np

from ..foundational import Weights

from .builtin import rnn
from .builtin import rnn_v2
from .builtin import RNNOperation
from .builtin import RNNInputMode
from .builtin import RNNDirection
from .builtin import RNNGateType

from .rnn import RNNParameterSet


def _roundrobin(a, b):
    result = []
    for x, y in zip(a, b):
        result.append(x)
        result.append(y)
    return result

def _rnn(input, max_seq_len, weights, bias, op, mode,
         direction, hidden_state, sequence_lengths):
    # TODO check shapes of weigths and biases
    dtype = input.network.weight_type
    num_layers = len(weights)
    out_size, in_size = weights[0].W_input.shape
    if direction == RNNDirection.BIDIRECTION:
        num_layers //= 2
    weights_list = []
    for i, w in enumerate(weights):
        weights_list.append((
            i, RNNGateType.INPUT, True,
            Weights(w.W_input.flatten(), dtype)))
        weights_list.append((
            i, RNNGateType.INPUT, False,
            Weights(w.U_input.flatten(), dtype)))
    bias_list = []
    for i, b in enumerate(bias):
        bias_list.append((
            i, RNNGateType.INPUT, True,
            Weights(b.W_input.flatten(), dtype)))
        bias_list.append((
            i, RNNGateType.INPUT, False,
            Weights(b.U_input.flatten(), dtype)))
    return rnn_v2(input, num_layers, out_size, max_seq_len, op, mode,
                  direction, weights_list, bias_list, hidden_state,
                  sequence_lengths=sequence_lengths)

def rnn_relu_v2(input, max_seq_len, weights, bias, mode=RNNInputMode.LINEAR,
                hidden_state=None, sequence_lengths=None):
    """Layer for RNN with ReLU.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of RNN.
        sequence_lengths(turret.Tensor): The sequence lengths specified for
            the RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.RELU,
                mode, RNNDirection.UNIDIRECTION, hidden_state,
                sequence_lengths)

def brnn_relu_v2(input, max_seq_len, fwd_weights, bwd_weights,
                 fwd_bias, bwd_bias, mode=RNNInputMode.LINEAR,
                 hidden_state=None, sequence_lengths=None):
    """Layer for BiRNN with ReLU.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        fwd_weights(turret.Weights): Forward weights.
        bwd_weights(turret.Weights): Backward weights.
        fwd_bias(turret.Weights): Forward bias.
        bwd_bias(turret.Weights): Backward bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of BiRNN.
        sequence_lengths(turret.Tensor): The sequence lengths specified for
            the BiRNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    # TODO check shapes of weights and biases
    weights = _roundrobin(fwd_weights, bwd_weights)
    bias = _roundrobin(fwd_bias, bwd_bias)
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.RELU,
                mode, RNNDirection.BIDIRECTION, hidden_state,
                sequence_lengths)

def rnn_tanh_v2(input, max_seq_len, weights, bias, mode=RNNInputMode.LINEAR,
                hidden_state=None, sequence_lengths=None):
    """Layer for RNN with tanh.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of RNN.
        sequence_lengths(turret.Tensor): The sequence lengths specified for
            the RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.TANH,
                mode, RNNDirection.UNIDIRECTION, hidden_state,
                sequence_lengths)

def brnn_tanh_v2(input, max_seq_len, fwd_weights, bwd_weights,
                 fwd_bias, bwd_bias, mode=RNNInputMode.LINEAR,
                 hidden_state=None, sequence_lengths=None):
    """Layer for BiRNN with tanh.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        fwd_weights(turret.Weights): Forward weights.
        bwd_weights(turret.Weights): Backward weights.
        fwd_bias(turret.Weights): Forward bias.
        bwd_bias(turret.Weights): Backward bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of BiRNN.
        sequence_lengths(turret.Tensor): The sequence lengths specified for
            the BiRNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    # TODO check shapes of weights and biases
    weights = _roundrobin(fwd_weights, bwd_weights)
    bias = _roundrobin(fwd_bias, bwd_bias)
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.TANH,
                mode, RNNDirection.BIDIRECTION, hidden_state,
                sequence_lengths)

