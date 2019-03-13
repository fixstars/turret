# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class ConcatTest(unittest.TestCase):

    def test_single(self):
        N, C, H, W = 3, 4, 7, 9
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.concat([h])
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = input
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


    def test_double(self):
        N, C, H, W = 3, 4, 7, 9
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.concat([h, h])
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.concatenate((input, input), axis=1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


    def test_different(self):
        N, C0, C1, H, W = 3, 4, 7, 7, 9
        input0 = np.random.rand(N, C0, H, W).astype(np.float32)
        input1 = np.random.rand(N, C1, H, W).astype(np.float32)

        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C0, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C1, H, W))
            h = L.concat([h0, h1])
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1}, build_network)

        expect = np.concatenate((input0, input1), axis=1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


    def test_axis(self):
        N, C, H, W = 3, 4, 7, 9
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.concat([h, h], axis=1)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = np.concatenate((input, input), axis=2)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
