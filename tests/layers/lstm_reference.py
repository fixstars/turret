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


def _sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

def _unidirection_lstm_layer(input, hidden, cell, weights, bias):
    batch_size, seq_len, data_size = input.shape
    hidden_size = hidden.shape[1]
    y_last = hidden
    result = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float32)
    for t in range(seq_len):
        x = input[:, t, :]
        it = _sigmoid(np.dot(x, weights.W_input.T) \
                    + np.dot(y_last, weights.U_input.T) \
                    + bias.W_input \
                    + bias.U_input)
        ft = _sigmoid(np.dot(x, weights.W_forget.T) \
                    + np.dot(y_last, weights.U_forget.T) \
                    + bias.W_forget \
                    + bias.U_forget)
        ot = _sigmoid(np.dot(x, weights.W_output.T) \
                    + np.dot(y_last, weights.U_output.T) \
                    + bias.W_output
                    + bias.U_output)
        ct = np.tanh(np.dot(x, weights.W_cell.T) \
                   + np.dot(y_last, weights.U_cell.T) \
                   + bias.W_cell \
                   + bias.U_cell)
        cell = ft * cell + it * ct
        y = ot * np.tanh(cell)
        y_last = y
        result[:, t, :] = y
    return result

def unidirection_lstm(input, hidden, cell, weights, bias):
    batch_dims  = input.shape[:-2]
    seq_len     = input.shape[-2]
    emb_size    = input.shape[-1]
    hidden_size = hidden.shape[-1]
    num_layers  = hidden.shape[-2]
    x = input.reshape(-1, seq_len, emb_size)
    h = hidden.reshape(-1, num_layers, hidden_size)
    c = cell.reshape(-1, num_layers, hidden_size)
    for l in range(num_layers):
        x = _unidirection_lstm_layer(
            x, h[:, l], c[:, l], weights[l], bias[l])
    return x.reshape(batch_dims + (seq_len, hidden_size))


def _bidirection_lstm_layer(input, hidden, cell, fwd_weights, bwd_weights,
                            fwd_bias, bwd_bias):
    hidden_size = hidden.shape[1] // 2
    fwd_hidden = hidden[:, :hidden_size]
    bwd_hidden = hidden[:, hidden_size:]
    fwd_cell = cell[:, :hidden_size]
    bwd_cell = cell[:, hidden_size:]
    y_fwd = _unidirection_lstm_layer(
        input, fwd_hidden, fwd_cell, fwd_weights, fwd_bias)
    y_bwd = _unidirection_lstm_layer(
        input[:, ::-1], bwd_hidden, bwd_cell, bwd_weights, bwd_bias)
    y_bwd = y_bwd[:, ::-1]
    return np.concatenate((y_fwd, y_bwd), axis=2)

def bidirection_lstm(input, hidden, cell, fwd_weights, bwd_weights,
                      fwd_bias, bwd_bias):
    batch_dims  = input.shape[:-2]
    seq_len     = input.shape[-2]
    emb_size    = input.shape[-1]
    hidden_size = hidden.shape[-1] // 2
    num_layers  = hidden.shape[-2]
    x = input.reshape(-1, seq_len, emb_size)
    h = hidden.reshape(-1, num_layers, hidden_size * 2)
    c = cell.reshape(-1, num_layers, hidden_size * 2)
    for l in range(num_layers):
        x = _bidirection_lstm_layer(
            x, h[:, l], c[:, l], fwd_weights[l], bwd_weights[l],
            fwd_bias[l], bwd_bias[l])
    return x.reshape(batch_dims + (seq_len, hidden_size * 2))
