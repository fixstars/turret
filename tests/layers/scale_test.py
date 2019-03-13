# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class ScaleTest(unittest.TestCase):

    def test_shift_uniform(self):
        N, C, H, W = 3, 4, 7, 9
        input = np.random.rand(N, C, H, W).astype(np.float32)
        shift = np.random.rand(1).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.scale(h, turret.ScaleMode.UNIFORM, shift=shift)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = input + shift
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


    def test_shift_channel(self):
        N, C, H, W = 3, 4, 7, 9
        input = np.random.rand(N, C, H, W).astype(np.float32)
        shift = np.random.rand(C).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.scale(h, turret.ScaleMode.CHANNEL, shift=shift)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.copy(input)
        for i in range(N):
            for c in range(C):
                expect[i, c] += shift[c]
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


    def test_shift_elementwise(self):
        N, C, H, W = 3, 4, 7, 9
        input = np.random.rand(N, C, H, W).astype(np.float32)
        shift = np.random.rand(C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.scale(h, turret.ScaleMode.ELEMENTWISE, shift=shift)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.copy(input)
        for i in range(N):
            expect[i] += shift
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
