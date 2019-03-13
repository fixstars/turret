# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class TopKTest(unittest.TestCase):

    def _topk_indices(self, x, k, axis):
        return np.split(np.argsort(x, axis=axis), [k], axis=axis)[0]

    def _topk(self, x, k, axis):
        return np.split(np.sort(x, axis=axis), [k], axis=axis)[0]

    def test_default(self):
        N, C, H, W = 5, 13, 17, 19
        k = 3
        input = np.random.rand(N, C, H, W).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.top_k(h, turret.TopKOperation.MIN, k, 1)
            network.mark_output("output", h[0])
        actual = execute_inference({"input": input}, build_network)

        expect = self._topk(input, k, 1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    # TODO receive INT32 output
    # def test_default_indices(self):
    #     N, C, H, W = 5, 13, 17, 19
    #     k = 3
    #     input = np.random.rand(N, C, H, W).astype(np.float32)

    #     def build_network(network):
    #         h = network.add_input("input", turret.DataType.FLOAT,
    #                               turret.Dimensions.CHW(C, H, W))
    #         h = L.top_k(h, turret.TopKOperation.MIN, k, 1)
    #         network.mark_output("output", h[1])
    #     actual = execute_inference({"input": input}, build_network)

    #     expect = self._topk_indices(input, k, 1)
    #     self.assertEqual(expect.shape, actual.shape)
    #     self.assertTrue(np.allclose(expect, actual))
