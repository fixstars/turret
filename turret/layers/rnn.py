# -*- coding: utf-8 -*-
import numpy as np

from ..foundational import Weights

from .builtin import rnn
from .builtin import RNNOperation
from .builtin import RNNInputMode
from .builtin import RNNDirection


def _roundrobin(a, b):
    result = []
    for x, y in zip(a, b):
        result.append(x)
        result.append(y)
    return result

class RNNParameterSet:
    """The object for parameters(weights, bias) of RNN layer.

    Attribute:
        W_input(np.ndarray): The initial input parameter for data.
        U_input(np.ndarray): The initial input parameter for hidden.
    """
    def __init__(self, W_input=None, U_input=None):
        self.W_input = W_input
        self.U_input = U_input

def _rnn(input, max_seq_len, weights, bias, op, mode, direction,
         hidden_state):
    # TODO check shapes of weigths and biases
    num_layers = len(weights)
    if direction == RNNDirection.BIDIRECTION:
        num_layers = num_layers // 2
    out_size, in_size = weights[0].W_input.shape
    weights_list = []
    for w in weights:
        weights_list.append(w.W_input.flatten())
        weights_list.append(w.U_input.flatten())
    bias_list = []
    for b in bias:
        bias_list.append(b.W_input.flatten())
        bias_list.append(b.U_input.flatten())
    dtype = input.network.weight_type
    return rnn(input, num_layers, out_size, max_seq_len,
               op, mode, direction,
               Weights(np.concatenate(weights_list), dtype=dtype),
               Weights(np.concatenate(bias_list), dtype=dtype),
               hidden_state)

def rnn_relu(input, max_seq_len, weights, bias,
             mode=RNNInputMode.LINEAR, hidden_state=None):
    """Layer for RNN with ReLU.

    Note:
        This function is not recommended. Use rnn_relu_v2.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.RELU,
                mode, RNNDirection.UNIDIRECTION, hidden_state)

def brnn_relu(input, max_seq_len,
              fwd_weights, bwd_weights, fwd_bias, bwd_bias,
              mode=RNNInputMode.LINEAR, hidden_state=None):
    """Layer for BiRNN with ReLU.

    Note:
        This function is not recommended. Use brnn_relu_v2.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        fwd_weights(turret.Weights): Forward weights.
        bwd_weights(turret.Weights): Backward weights.
        fwd_bias(turret.Weights): Forward bias.
        bwd_bias(turret.Weights): Backward bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    # TODO check shapes of weights and biases
    weights = _roundrobin(fwd_weights, bwd_weights)
    bias = _roundrobin(fwd_bias, bwd_bias)
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.RELU,
                mode, RNNDirection.BIDIRECTION, hidden_state)

def rnn_tanh(input, max_seq_len, weights, bias,
             mode=RNNInputMode.LINEAR, hidden_state=None):
    """Layer for RNN with tanh.

    Note:
        This function is not recommended. Use rnn_tanh_v2.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.TANH,
                mode, RNNDirection.UNIDIRECTION, hidden_state)

def brnn_tanh(input, max_seq_len,
              fwd_weights, bwd_weights, fwd_bias, bwd_bias,
              mode=RNNInputMode.LINEAR, hidden_state=None):
    """Layer for BiRNN with tanh.

    Note:
        This function is not recommended. Use brnn_tanh_v2.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        fwd_weights(turret.Weights): Forward weights.
        bwd_weights(turret.Weights): Backward weights.
        fwd_bias(turret.Weights): Forward bias.
        bwd_bias(turret.Weights): Backward bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    # TODO check shapes of weights and biases
    weights = _roundrobin(fwd_weights, bwd_weights)
    bias = _roundrobin(fwd_bias, bwd_bias)
    return _rnn(input, max_seq_len, weights, bias, RNNOperation.TANH,
                mode, RNNDirection.BIDIRECTION, hidden_state)


class GRUParameterSet:
    """The object for parameters(weights, bias) of GRU layer.

    Attribute:
        W_update(np.ndarray): The initial input parameter for data.
        U_update(np.ndarray): The initial input parameter for hidden.
        W_reset(np.ndarray): The initial reset parameter for data.
        U_reset(np.ndarray): The initial reset parameter for hidden.
        W_hidden(np.ndarray): The initial hidden parameter for data.
        U_hidden(np.ndarray): The initial hidden parameter for hidden.
    """
    def __init__(self):
        self.W_update = None
        self.U_update = None
        self.W_reset = None
        self.U_reset = None
        self.W_hidden = None
        self.U_hidden = None

def gru(input, max_seq_len, weights, bias=None,
        mode=RNNInputMode.LINEAR,
        direction=RNNDirection.UNIDIRECTION,
        hidden_state=None):
    """Layer for GRU.

    Note:
        This function is not implemented.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        direction(turret.RNNDirection): The direction of RNN.
        hidden_state(turret.Tensor): Initial hidden status of RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    pass

def bias(input, b):
    """Layer for applying bias.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        b(np.ndarray): The bias.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    c = input.dimensions[0].size
    if b.shape != (c,):
        raise ValueError("shape mismatch")
    return scale(input, ScaleMode.CHANNEL, shift=b)
