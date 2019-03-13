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

class LSTMParameterSet:
    """The object for parameters(weights, bias) of LSTM layer.

    Attribute:
        W_input(np.ndarray): The initial input parameter for data.
        U_input(np.ndarray): The initial input parameter for hidden.
        W_cell(np.ndarray): The initial cell parameter for data.
        U_cell(np.ndarray): The initial cell parameter for hidden.
        W_forget(np.ndarray): The initial forget parameter for data.
        U_forget(np.ndarray): The initial forget parameter for hidden.
        W_output(np.ndarray): The initial output parameter for data.
        U_output(np.ndarray): The initial output parameter for hidden.
    """
    def __init__(self, W_input=None, U_input=None, W_cell=None, U_cell=None,
                 W_forget=None, U_forget=None, W_output=None, U_output=None):
        self.W_input = W_input
        self.U_input = U_input
        self.W_cell = W_cell
        self.U_cell = U_cell
        self.W_forget = W_forget
        self.U_forget = U_forget
        self.W_output = W_output
        self.U_output = U_output

def _lstm(input, max_seq_len, weights, bias, mode, direction,
          hidden_state, cell_state):
    # TODO check shapes of weigths and biases
    num_layers = len(weights)
    if direction == RNNDirection.BIDIRECTION:
        num_layers = num_layers // 2
    out_size, in_size = weights[0].W_input.shape
    weights_list = []
    for w in weights:
        weights_list.append(w.W_forget.flatten())
        weights_list.append(w.W_input.flatten())
        weights_list.append(w.W_cell.flatten())
        weights_list.append(w.W_output.flatten())
        weights_list.append(w.U_forget.flatten())
        weights_list.append(w.U_input.flatten())
        weights_list.append(w.U_cell.flatten())
        weights_list.append(w.U_output.flatten())
    bias_list = []
    for b in bias:
        bias_list.append(b.W_forget.flatten())
        bias_list.append(b.W_input.flatten())
        bias_list.append(b.W_cell.flatten())
        bias_list.append(b.W_output.flatten())
        bias_list.append(b.U_forget.flatten())
        bias_list.append(b.U_input.flatten())
        bias_list.append(b.U_cell.flatten())
        bias_list.append(b.U_output.flatten())
    dtype = input.network.weight_type
    return rnn(input, num_layers, out_size, max_seq_len,
               RNNOperation.LSTM, mode, direction,
               Weights(np.concatenate(weights_list), dtype=dtype),
               Weights(np.concatenate(bias_list), dtype=dtype),
               hidden_state, cell_state)

def lstm(input, max_seq_len, weights, bias, mode=RNNInputMode.LINEAR,
         hidden_state=None, cell_state=None):
    """Layer for LSTM.

    Note:
        This function is not recommended. Use lstm_v2.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        max_seq_len(int): The maximum length of the time sequence.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        mode(turret.RNNInputMode): RNN input mode.
        hidden_state(turret.Tensor): Initial hidden status of LSTM.
        cell_state(turret.Tensor): Initial cell state of LSTM.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return _lstm(input, max_seq_len, weights, bias, mode,
                 RNNDirection.UNIDIRECTION, hidden_state, cell_state)

def blstm(input, max_seq_len, fwd_weights, bwd_weights, fwd_bias, bwd_bias,
          mode=RNNInputMode.LINEAR, hidden_state=None, cell_state=None):
    """Layer for BiLSTM.

    Note:
        This function is not recommended. Use blstm_v2.

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

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    weights = _roundrobin(fwd_weights, bwd_weights)
    bias = _roundrobin(fwd_bias, bwd_bias)
    return _lstm(input, max_seq_len, weights, bias, mode,
                 RNNDirection.BIDIRECTION, hidden_state, cell_state)
