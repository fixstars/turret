# -*- coding: utf-8 -*-
import numpy as np

from ..foundational import Weights

from .builtin import rnn_v2
from .builtin import RNNOperation
from .builtin import RNNInputMode
from .builtin import RNNDirection
from .builtin import RNNGateType

from .lstm import LSTMParameterSet


def _roundrobin(a, b):
    result = []
    for x, y in zip(a, b):
        result.append(x)
        result.append(y)
    return result

def _lstm(input, max_seq_len, weights, bias, mode, direction,
          hidden_state, cell_state, sequence_lengths):
    # TODO check shapes of weigths and biases
    dtype = input.network.weight_type
    num_layers = len(weights)
    if direction == RNNDirection.BIDIRECTION:
        num_layers //= 2
    out_size, in_size = weights[0].W_input.shape
    weights_list = []
    for i, w in enumerate(weights):
        weights_list.append((i, RNNGateType.FORGET, True,  Weights(w.W_forget.flatten(), dtype)))
        weights_list.append((i, RNNGateType.FORGET, False, Weights(w.U_forget.flatten(), dtype)))
        weights_list.append((i, RNNGateType.INPUT,  True,  Weights(w.W_input.flatten(),  dtype)))
        weights_list.append((i, RNNGateType.INPUT,  False, Weights(w.U_input.flatten(),  dtype)))
        weights_list.append((i, RNNGateType.CELL,   True,  Weights(w.W_cell.flatten(),   dtype)))
        weights_list.append((i, RNNGateType.CELL,   False, Weights(w.U_cell.flatten(),   dtype)))
        weights_list.append((i, RNNGateType.OUTPUT, True,  Weights(w.W_output.flatten(), dtype)))
        weights_list.append((i, RNNGateType.OUTPUT, False, Weights(w.U_output.flatten(), dtype)))
    bias_list = []
    for i, b in enumerate(bias):
        bias_list.append((i, RNNGateType.FORGET, True,  Weights(b.W_forget.flatten(), dtype)))
        bias_list.append((i, RNNGateType.FORGET, False, Weights(b.U_forget.flatten(), dtype)))
        bias_list.append((i, RNNGateType.INPUT,  True,  Weights(b.W_input.flatten(),  dtype)))
        bias_list.append((i, RNNGateType.INPUT,  False, Weights(b.U_input.flatten(),  dtype)))
        bias_list.append((i, RNNGateType.CELL,   True,  Weights(b.W_cell.flatten(),   dtype)))
        bias_list.append((i, RNNGateType.CELL,   False, Weights(b.U_cell.flatten(),   dtype)))
        bias_list.append((i, RNNGateType.OUTPUT, True,  Weights(b.W_output.flatten(), dtype)))
        bias_list.append((i, RNNGateType.OUTPUT, False, Weights(b.U_output.flatten(), dtype)))
    return rnn_v2(input, num_layers, out_size, max_seq_len, RNNOperation.LSTM,
                  mode, direction, weights_list, bias_list, hidden_state,
                  cell_state, sequence_lengths)

def lstm_v2(input, max_seq_len, weights, bias, mode=RNNInputMode.LINEAR,
            hidden_state=None, cell_state=None, sequence_lengths=None):
    """Layer for LSTM.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of LSTM.
        cell_state(turret.Tensor): Initial cell state of LSTM.
        sequence_lengths(turret.Tensor): The sequence lengths specified for
                    the LSTM.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return _lstm(input, max_seq_len, weights, bias, mode,
                 RNNDirection.UNIDIRECTION, hidden_state, cell_state,
                 sequence_lengths)

def blstm_v2(input, max_seq_len, fwd_weights, bwd_weights, fwd_bias, bwd_bias,
             mode=RNNInputMode.LINEAR, hidden_state=None, cell_state=None,
             sequence_lengths=None):
    """Layer for BiLSTM.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        fwd_weights(turret.Weights): Forward weights.
        bwd_weights(turret.Weights): Backward weights.
        fwd_bias(turret.Weights): Forward Bias.
        bwd_bias(turret.Weights): Backward Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of BiLSTM.
        cell_state(turret.Tensor): Initial cell state of BiLSTM.
        sequence_lengths(turret.Tensor): The sequence lengths specified for
                    the BiLSTM.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    weights = _roundrobin(fwd_weights, bwd_weights)
    bias = _roundrobin(fwd_bias, bwd_bias)
    return _lstm(input, max_seq_len, weights, bias, mode,
                 RNNDirection.BIDIRECTION, hidden_state, cell_state,
                 sequence_lengths)
