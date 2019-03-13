# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np
import turret
import turret.layers as L

from util import execute_inference


class ParameterSet:
    def __init__(self, batch_size, seq_len, num_layers,
                 data_size, hidden_size):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.data_size = data_size
        self.hidden_size = hidden_size


def _unidirection_rnn_layer(input, hidden, weights, bias, activation):
    batch_size, seq_len, data_size = input.shape
    hidden_size = hidden.shape[1]
    y_last = hidden
    result = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float32)
    for t in range(seq_len):
        x = input[:, t, :]
        y = np.dot(x, weights.W_input.T) \
          + np.dot(y_last, weights.U_input.T) \
          + bias.W_input \
          + bias.U_input
        y = activation(y)
        y_last = y
        result[:, t, :] = y
    return result

def unidirection_rnn(input, hidden, weights, bias, activation):
    batch_dims  = input.shape[:-2]
    seq_len     = input.shape[-2]
    emb_size    = input.shape[-1]
    hidden_size = hidden.shape[-1]
    num_layers  = hidden.shape[-2]
    x = input.reshape(-1, seq_len, emb_size)
    h = hidden.reshape(-1, num_layers, hidden_size)
    for l in range(num_layers):
        x = _unidirection_rnn_layer(
            x, h[:, l, :], weights[l], bias[l], activation)
    return x.reshape(batch_dims + (seq_len, hidden_size))


def _bidirection_rnn_layer(input, hidden, fwd_weights, bwd_weights,
                           fwd_bias, bwd_bias, activation):
    hidden_size = hidden.shape[1] // 2
    fwd_hidden = hidden[:, :hidden_size]
    bwd_hidden = hidden[:, hidden_size:]
    y_fwd = _unidirection_rnn_layer(
        input, fwd_hidden, fwd_weights, fwd_bias, activation)
    y_bwd = _unidirection_rnn_layer(
        input[:, ::-1], bwd_hidden, bwd_weights, bwd_bias, activation)
    y_bwd = y_bwd[:, ::-1]
    return np.concatenate((y_fwd, y_bwd), axis=2)

def bidirection_rnn(input, hidden, fwd_weights, bwd_weights,
                    fwd_bias, bwd_bias, activation):
    batch_dims  = input.shape[:-2]
    seq_len     = input.shape[-2]
    emb_size    = input.shape[-1]
    hidden_size = hidden.shape[-1] // 2
    num_layers  = hidden.shape[-2]
    x = input.reshape(-1, seq_len, emb_size)
    h = hidden.reshape(-1, num_layers, hidden_size * 2)
    for l in range(num_layers):
        x = _bidirection_rnn_layer(
            x, h[:, l, :], fwd_weights[l], bwd_weights[l],
            fwd_bias[l], bwd_bias[l], activation)
    return x.reshape(batch_dims + (seq_len, hidden_size * 2))
