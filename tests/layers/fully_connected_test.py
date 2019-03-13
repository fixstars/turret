# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class FullyConnectedTest(unittest.TestCase):

    def test_default(self):
        N, K, C = 5, 10, 100
        input = np.random.rand(N, C).astype(np.float32)
        weights = np.random.rand(K, C).astype(np.float32)
        biases = np.random.rand(K).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(1, 1, C))
            h = L.fully_connected(h, weights, biases)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input": input.reshape((N, 1, 1, C))}, build_network)

        expect = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for k in range(K):
                expect[i, k] = biases[k]
                for c in range(C):
                    expect[i, k] += input[i, c] * weights[k, c]
        expect = expect.reshape((N, K, 1, 1))
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


    def test_no_biases(self):
        N, K, C = 5, 10, 100
        input = np.random.rand(N, C).astype(np.float32)
        weights = np.random.rand(K, C).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(1, 1, C))
            h = L.fully_connected(h, weights)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input": input.reshape((N, 1, 1, C))}, build_network)

        expect = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for k in range(K):
                for c in range(C):
                    expect[i, k] += input[i, c] * weights[k, c]
        expect = expect.reshape((N, K, 1, 1))
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
