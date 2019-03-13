# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


def max_pooling_2d(input, window_size, stride, padding):
    N, C, H, W = input.shape
    Hp = H + 2 * padding[0]
    Wp = W + 2 * padding[1]
    x = np.zeros((N, C, Hp, Wp), dtype=np.float32)
    x[:, :, padding[0]:H+padding[0], padding[1]:W+padding[1]] = input
    Hs = Hp - window_size[0] + 1
    Ws = Wp - window_size[1] + 1
    y = x[:, :, :Hs, :Ws]
    for i in range(window_size[0]):
        for j in range(window_size[1]):
            y = np.maximum(y, x[:, :, i:i+Hs, j:j+Ws])
    return y[:, :, ::stride[0], ::stride[1]]


def average_pooling_2d(input, window_size, stride, padding):
    N, C, H, W = input.shape
    Hp = H + 2 * padding[0]
    Wp = W + 2 * padding[1]
    x = np.zeros((N, C, Hp, Wp), dtype=np.float32)
    x[:, :, padding[0]:H+padding[0], padding[1]:W+padding[1]] = input
    m = np.zeros((N, C, Hp, Wp), dtype=np.float32)
    m[:, :, padding[0]:H+padding[0], padding[1]:W+padding[1]] = 1.0
    Hs = Hp - window_size[0] + 1
    Ws = Wp - window_size[1] + 1
    y = np.zeros((N, C, Hs, Ws), dtype=np.float32)
    c = np.zeros((N, C, Hs, Ws), dtype=np.float32)
    for i in range(window_size[0]):
        for j in range(window_size[1]):
            y = y + x[:, :, i:i+Hs, j:j+Ws]
            c = c + m[:, :, i:i+Hs, j:j+Ws]
    return (y / c)[:, :, ::stride[0], ::stride[1]]


class MaxPooling2DTest(unittest.TestCase):
    def test_default(self):
        N, C, H, W = 5, 3, 20, 30
        window_size = (3, 2)
        stride = (2, 3)
        padding = (0, 1)
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.max_pooling_2d(h, window_size, stride, padding)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input": input.reshape((N, C, H, W))}, build_network)

        expect = max_pooling_2d(input, window_size, stride, padding)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


class AveragePooling2DTest(unittest.TestCase):
    def test_default(self):
        N, C, H, W = 5, 3, 20, 30
        window_size = (3, 2)
        stride = (2, 3)
        padding = (0, 1)
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.average_pooling_2d(h, window_size, stride, padding)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input": input.reshape((N, C, H, W))}, build_network)

        expect = average_pooling_2d(input, window_size, stride, padding)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))


class MaxAverageBlendPooling2DTest(unittest.TestCase):
    def test_default(self):
        N, C, H, W = 5, 3, 20, 30
        window_size = (3, 2)
        stride = (2, 3)
        padding = (0, 1)
        blend = 0.3
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.max_average_blend_pooling_2d(h, window_size, stride,
                                               padding, blend)
            network.mark_output("output", h)
        actual = execute_inference(
            {"input": input.reshape((N, C, H, W))}, build_network)

        expect = (1 - blend) * max_pooling_2d(input, window_size, stride, padding) \
               + blend * average_pooling_2d(input, window_size, stride, padding)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
