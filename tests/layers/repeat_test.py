# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class RepeatTest(unittest.TestCase):

    def test_repeat(self):
        N, C, H, W = 10, 13, 11, 17
        repeats, axis = 4, 1
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.repeat(h, repeats, axis=axis)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.repeat(input, repeats, axis=axis+1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
