# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference
from lstm_reference import ParameterSet
from lstm_reference import unidirection_lstm
from lstm_reference import bidirection_lstm


class LSTMv2Test(unittest.TestCase):

    def _make_input(self, params):
        c = params.batch_size
        h = params.seq_len
        w = params.data_size
        return np.random.rand(1, c, h, w).astype(np.float32)

    def _make_hidden(self, params):
        c = params.batch_size
        h = params.num_layers
        w = params.hidden_size
        return np.random.rand(1, c, h, w).astype(np.float32)
    
    def _make_cell(self, params):
        c = params.batch_size
        h = params.num_layers
        w = params.hidden_size
        return np.random.rand(1, c, h, w).astype(np.float32)

    def _make_weights(self, params, skip=False):
        weights = []
        for i in range(params.num_layers):
            w_out = params.hidden_size
            w_in = params.hidden_size if i > 0 or skip else params.data_size
            weights.append(L.LSTMParameterSet(
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32),
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32),
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32),
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32)))
        return weights

    def _make_bias(self, params):
        bias = []
        for i in range(params.num_layers):
            bias.append(L.LSTMParameterSet(
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32)))
        return bias

    def test_default_output(self):
        cases = [
            ParameterSet(2, 10, 3, 20, 15)
        ]
        for case in cases:
            input = self._make_input(case)
            hidden = self._make_hidden(case)
            cell = self._make_cell(case)
            weights = self._make_weights(case)
            bias = self._make_bias(case)

            def build_network(network):
                h = network.add_input(
                    "input", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.batch_size, case.seq_len, case.data_size))
                h_hidden = network.add_input(
                    "hidden", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.batch_size, case.num_layers, case.hidden_size))
                h_cell = network.add_input(
                    "cell", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.batch_size, case.num_layers, case.hidden_size))
                h, _, _ = L.lstm_v2(h, case.seq_len, weights, bias,
                                    hidden_state=h_hidden, cell_state=h_cell)
                network.mark_output("output", h)
            actual = execute_inference(
                {"input": input, "hidden": hidden, "cell": cell},
                build_network)

            expect = unidirection_lstm(input, hidden, cell, weights, bias)
            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))


class BLSTMv2Test(unittest.TestCase):

    def _make_input(self, params):
        c = params.batch_size
        h = params.seq_len
        w = params.data_size
        return np.random.rand(1, c, h, w).astype(np.float32)

    def _make_hidden(self, params):
        c = params.batch_size
        h = params.num_layers
        w = params.hidden_size
        return np.random.rand(1, c, h, w * 2).astype(np.float32)
    
    def _make_cell(self, params):
        c = params.batch_size
        h = params.num_layers
        w = params.hidden_size
        return np.random.rand(1, c, h, w * 2).astype(np.float32)

    def _make_weights(self, params, skip=False):
        weights = []
        for i in range(params.num_layers):
            w_out = params.hidden_size
            w_in = params.data_size
            if i > 0 or skip:
                w_in = params.hidden_size * 2
            weights.append(L.LSTMParameterSet(
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32),
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32),
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32),
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32)))
        return weights

    def _make_bias(self, params):
        bias = []
        for i in range(params.num_layers):
            bias.append(L.LSTMParameterSet(
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32)))
        return bias

    def test_default_output(self):
        cases = [
            ParameterSet(2, 10, 3, 20, 15)
        ]
        for case in cases:
            input = self._make_input(case)
            hidden = self._make_hidden(case)
            cell = self._make_cell(case)
            fwd_weights = self._make_weights(case)
            bwd_weights = self._make_weights(case)
            fwd_bias = self._make_bias(case)
            bwd_bias = self._make_bias(case)

            def build_network(network):
                h = network.add_input(
                    "input", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.batch_size, case.seq_len, case.data_size))
                h_hidden = network.add_input(
                    "hidden", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.batch_size, case.num_layers, case.hidden_size * 2))
                h_cell = network.add_input(
                    "cell", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.batch_size, case.num_layers, case.hidden_size * 2))
                h, _, _ = L.blstm_v2(h, case.seq_len, fwd_weights, bwd_weights,
                                  fwd_bias, bwd_bias, hidden_state=h_hidden,
                                  cell_state=h_cell)
                network.mark_output("output", h)

            actual = execute_inference(
                {"input": input, "hidden": hidden, "cell": cell},
                build_network)
            expect = bidirection_lstm(input, hidden, cell,
                                      fwd_weights, bwd_weights,
                                      fwd_bias, bwd_bias)
            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))
