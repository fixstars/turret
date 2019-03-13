# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class ReduceTest(unittest.TestCase):
    def test_sum(self):
        N, C, H, W = 3, 5, 7, 11
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.reduce(h, turret.ReduceOperation.SUM, axes=(0, 1))
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.sum(input, axis=(1, 2), keepdims=False)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
