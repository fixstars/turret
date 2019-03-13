# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class ReshapeTest(unittest.TestCase):
    def test_default(self):
        N = 5
        C_in, H_in, W_in = 3, 20, 30
        C_out, H_out, W_out = 6, 5, 60
        input = np.random.rand(N, C_in, H_in, W_in).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C_in, H_in, W_in))
            h = L.reshape(h, turret.Dimensions.CHW(C_out, H_out, W_out))
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.reshape(input, (N, C_out, H_out, W_out))
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
