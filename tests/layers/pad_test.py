# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class PadTest(unittest.TestCase):
    def test_positive(self):
        N, C, H, W = 3, 5, 20, 30
        pad_width = (1, 2)
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.pad(h, pad_width)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        Hp = H + 2 * pad_width[0]
        Wp = W + 2 * pad_width[1]
        top, left = pad_width
        expect = np.zeros((N, C, Hp, Wp), dtype=np.float32)
        expect[:, :, top:H+top, left:W+left] = input

        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_negative(self):
        N, C, H, W = 3, 5, 20, 30
        pad_width = (-1, -2)
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.pad(h, pad_width)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        Hp = H + 2 * pad_width[0]
        Wp = W + 2 * pad_width[1]
        top, left = -pad_width[0], -pad_width[1]
        expect = input[:, :, top:top+Hp, left:left+Wp]

        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
