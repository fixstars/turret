# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class RaggedSoftmaxTest(unittest.TestCase):

    def _softmax(self, x):
        y = np.exp(x - x.max())
        return y / y.sum()

    def _ragged_softmax(self, x, bounds):
        N, C, K = x.shape
        y = np.zeros((N, C, K), dtype=x.dtype)
        for i in range(N):
            for j in range(C):
                k = bounds[i, j, 0]
                y[i, j, 0:k] = self._softmax(x[i, j, 0:k])
        return y

    def test_default(self):
        N, C, K = 5, 3, 20
        input = np.random.rand(N, C, K).astype(np.float32)
        bounds = np.broadcast_to(
            np.asarray([[[10], [15], [20]]], dtype=np.int32), (N, C, 1))

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.HW(C, K))
            b = network.add_input("bounds", turret.DataType.INT32,
                                  turret.Dimensions.HW(C, 1))
            h = L.ragged_softmax(h, b)
            network.mark_output("output", h)
        actual = execute_inference({
                "input": input,
                "bounds": bounds
            }, build_network)

        expect = self._ragged_softmax(input, bounds)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
