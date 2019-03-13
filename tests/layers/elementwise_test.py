# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class ElementwiseTest(unittest.TestCase):

    def test_sum(self):
        N, C, H, W = 3, 4, 7, 9
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.elementwise(h0, h1, turret.ElementWiseOperation.SUM)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1}, build_network)

        expect = input0 + input1
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
