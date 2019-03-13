# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class LeakyReLUTest(unittest.TestCase):
    def test_default(self):
        N = 5
        C, H, W = 3, 20, 30
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.leaky_relu(h, slope=0.5)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input": input.reshape((N, C, H, W))}, build_network)

        expect = np.zeros((N, C, H, W), dtype=np.float32)
        expect = np.maximum(input, input * 0.5)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
