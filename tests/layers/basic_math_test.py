# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class BasicMathTest(unittest.TestCase):

    def test_sum_ternary(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        input2 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h2 = network.add_input("input2", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.sum(h0, h1, h2)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1, "input2": input2},
            build_network)

        expect = input0 + input1 + input2
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_prod_ternary(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        input2 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h2 = network.add_input("input2", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.prod(h0, h1, h2)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1, "input2": input2},
            build_network)

        expect = input0 * input1 * input2
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_max_ternary(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        input2 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h2 = network.add_input("input2", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.max(h0, h1, h2)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1, "input2": input2},
            build_network)

        expect = np.maximum(np.maximum(input0, input1), input2)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_min_ternary(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        input2 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h2 = network.add_input("input2", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.min(h0, h1, h2)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1, "input2": input2},
            build_network)

        expect = np.minimum(np.minimum(input0, input1), input2)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_sub(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        input2 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.sub(h0, h1)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1}, build_network)

        expect = input0 - input1
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_div(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32) + 0.5
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.div(h0, h1)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1}, build_network)

        expect = input0 / input1
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_pow(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        input1 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                   turret.Dimensions.CHW(C, H, W))
            h = L.pow(h0, h1)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0, "input1": input1}, build_network)

        expect = np.power(input0, input1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_exp(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.exp(h)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0}, build_network)

        expect = np.exp(input0)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_log(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32) + 0.5
        def build_network(network):
            h = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.log(h)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0}, build_network)

        expect = np.log(input0)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual, atol=1e-7))

    def test_sqrt(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.sqrt(h)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0}, build_network)

        expect = np.sqrt(input0)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_recip(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32) + 0.5
        def build_network(network):
            h = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.recip(h)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0}, build_network)

        expect = 1.0 / input0
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_abs(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.abs(h)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0}, build_network)

        expect = np.abs(input0)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_neg(self):
        N, C, H, W = 3, 5, 7, 11
        input0 = np.random.rand(N, C, H, W).astype(np.float32)
        def build_network(network):
            h = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.neg(h)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input0": input0}, build_network)

        expect = -input0
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
