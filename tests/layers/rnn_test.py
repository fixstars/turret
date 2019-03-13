# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import unittest
import pytest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference
from rnn_reference import ParameterSet
from rnn_reference import unidirection_rnn
from rnn_reference import bidirection_rnn


class RNNTest(unittest.TestCase):

    def _make_input(self, params):
        c = params.seq_len
        h = params.batch_size
        w = params.data_size
        return np.random.rand(1, c, h, w).astype(np.float32)

    def _make_hidden(self, params):
        c = params.num_layers
        h = params.batch_size
        w = params.hidden_size
        return np.random.rand(1, c, h, w).astype(np.float32)

    def _make_weights(self, params, skip=False):
        weights = []
        for i in range(params.num_layers):
            w_out = params.hidden_size
            w_in = params.hidden_size if i > 0 or skip else params.data_size
            weights.append(L.RNNParameterSet(
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32)))
        return weights

    def _make_bias(self, params):
        bias = []
        for i in range(params.num_layers):
            bias.append(L.RNNParameterSet(
                np.random.rand(params.hidden_size).astype(np.float32),
                np.random.rand(params.hidden_size).astype(np.float32)))
        return bias

    def test_rnn_relu_default_output(self):
        cases = [
            ParameterSet(2, 10, 3, 20, 15)
        ]
        for case in cases:
            input = self._make_input(case)
            hidden = self._make_hidden(case)
            weights = self._make_weights(case)
            bias = self._make_bias(case)

            def build_network(network):
                h = network.add_input(
                    "input", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.seq_len, case.batch_size, case.data_size))
                h_hidden = network.add_input(
                    "hidden", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.num_layers, case.batch_size, case.hidden_size))
                h, _ = L.rnn_relu(h, case.seq_len, weights, bias,
                                  hidden_state=h_hidden)
                network.mark_output("output", h)

            actual = execute_inference(
                {"input": input, "hidden": hidden}, build_network,
                max_batch_size=1)

            t_input = input.transpose(0, 2, 1, 3)
            t_hidden = hidden.transpose(0, 2, 1, 3)
            t_expect = unidirection_rnn(
                t_input, t_hidden, weights, bias, lambda x: np.maximum(x, 0.0))
            expect = t_expect.transpose(0, 2, 1, 3)

            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))

    def test_tanh_default_output(self):
        cases = [
            ParameterSet(2, 10, 3, 20, 15)
        ]
        for case in cases:
            input = self._make_input(case)
            hidden = self._make_hidden(case)
            weights = self._make_weights(case)
            bias = self._make_bias(case)

            def build_network(network):
                h = network.add_input(
                    "input", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.seq_len, case.batch_size, case.data_size))
                h_hidden = network.add_input(
                    "hidden", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.num_layers, case.batch_size, case.hidden_size))
                h, _ = L.rnn_tanh(h, case.seq_len, weights, bias,
                                  hidden_state=h_hidden)
                network.mark_output("output", h)

            actual = execute_inference(
                {"input": input, "hidden": hidden}, build_network,
                max_batch_size=1)

            t_input = input.transpose(0, 2, 1, 3)
            t_hidden = hidden.transpose(0, 2, 1, 3)
            t_expect = unidirection_rnn(
                t_input, t_hidden, weights, bias, lambda x: np.tanh(x))
            expect = t_expect.transpose(0, 2, 1, 3)

            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))


class BRNNTest(unittest.TestCase):

    def _make_input(self, params):
        c = params.seq_len
        h = params.batch_size
        w = params.data_size
        return np.random.rand(1, c, h, w).astype(np.float32)

    def _make_hidden(self, params):
        c = params.num_layers
        h = params.batch_size
        w = params.hidden_size
        return np.random.rand(1, c, h, w * 2).astype(np.float32)

    def _make_weights(self, params, skip=False):
        weights = []
        for i in range(params.num_layers):
            w_out = params.hidden_size
            w_in = params.data_size
            if i > 0 or skip:
                w_in = params.hidden_size * 2
            weights.append(L.RNNParameterSet(
                np.random.rand(w_out, w_in).astype(np.float32),
                np.random.rand(w_out, w_out).astype(np.float32)))
        return weights

    def _make_bias(self, params):
        bias = []
        for i in range(params.num_layers):
            bias.append(L.RNNParameterSet(
                np.zeros(params.hidden_size).astype(np.float32),
                np.zeros(params.hidden_size).astype(np.float32)))
        return bias

    @pytest.mark.skip(reason="TensorRT 4.0 bug?")
    def test_brnn_relu_default_output(self):
        cases = [
            ParameterSet(2, 10, 3, 20, 15)
        ]
        for case in cases:
            input = self._make_input(case)
            hidden = self._make_hidden(case)
            fwd_weights = self._make_weights(case)
            bwd_weights = self._make_weights(case)
            fwd_bias = self._make_bias(case)
            bwd_bias = self._make_bias(case)

            def build_network(network):
                h = network.add_input(
                    "input", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.seq_len, case.batch_size, case.data_size))
                h_hidden = network.add_input(
                    "hidden", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.num_layers, case.batch_size, case.hidden_size * 2))
                h, _ = L.brnn_relu(h, case.seq_len, fwd_weights, bwd_weights,
                                   fwd_bias, bwd_bias, hidden_state=h_hidden)
                network.mark_output("output", h)

            actual = execute_inference(
                {"input": input, "hidden": hidden}, build_network,
                max_batch_size=1)

            t_input = input.transpose(0, 2, 1, 3)
            t_hidden = hidden.transpose(0, 2, 1, 3)
            t_expect = bidirection_rnn(
                t_input, t_hidden, fwd_weights, bwd_weights,
                fwd_bias, bwd_bias, lambda x: np.maximum(x, 0.0))
            expect = t_expect.transpose(0, 2, 1, 3)

            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))

    @pytest.mark.skip(reason="TensorRT 4.0 bug?")
    def test_brnn_tanh_default_output(self):
        cases = [
            ParameterSet(2, 10, 3, 20, 15)
        ]
        for case in cases:
            input = self._make_input(case)
            hidden = self._make_hidden(case)
            fwd_weights = self._make_weights(case)
            bwd_weights = self._make_weights(case)
            fwd_bias = self._make_bias(case)
            bwd_bias = self._make_bias(case)

            def build_network(network):
                h = network.add_input(
                    "input", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.seq_len, case.batch_size, case.data_size))
                h_hidden = network.add_input(
                    "hidden", turret.DataType.FLOAT,
                    turret.Dimensions.CHW(
                        case.num_layers, case.batch_size, case.hidden_size * 2))
                h, _ = L.brnn_tanh(h, case.seq_len, fwd_weights, bwd_weights,
                                   fwd_bias, bwd_bias, hidden_state=h_hidden)
                network.mark_output("output", h)

            actual = execute_inference(
                {"input": input, "hidden": hidden}, build_network,
                max_batch_size=1)

            t_input = input.transpose(0, 2, 1, 3)
            t_hidden = hidden.transpose(0, 2, 1, 3)
            t_expect = bidirection_rnn(
                t_input, t_hidden, fwd_weights, bwd_weights,
                fwd_bias, bwd_bias, lambda x: np.tanh(x))
            expect = t_expect.transpose(0, 2, 1, 3)

            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))
