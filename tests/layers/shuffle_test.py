# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class ShuffleTest(unittest.TestCase):
    def test_default(self):
        N = 5
        C, H, W = 3, 20, 30
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.shuffle(h, (1, 2, 0))
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.transpose(input, (0, 2, 3, 1))
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
