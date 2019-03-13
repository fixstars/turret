# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class GatherTest(unittest.TestCase):

    def _gather(self, x, indices, axis):
        N = x.shape[0]
        y = []
        for i in range(N):
            y.append(np.take(x[i], indices[i], axis=axis))
        return np.asarray(y)

    def test_1d_indices(self):
        N, C, K, H, W = 5, 13, 17, 19, 23
        data = np.random.rand(N, C, H, W).astype(np.float32)
        indices = np.random.randint(0, C - 1, (N, K), dtype=np.int32)

        def build_network(network):
            d = network.add_input("data", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            i = network.add_input("indices", turret.DataType.INT32,
                                  turret.Dimensions((
                                      (K, turret.DimensionType.INDEX),)))
            h = L.gather(d, i, 0)
            network.mark_output("output", h)
        actual = execute_inference({
                "data": data,
                "indices": indices 
            }, build_network)

        expect = self._gather(data, indices, 0)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_2d_indices(self):
        N, C, K1, K2, H, W = 5, 13, 17, 19, 23, 29
        data = np.random.rand(N, C, H, W).astype(np.float32)
        indices = np.random.randint(0, H - 1, (N, K1, K2), dtype=np.int32)

        def build_network(network):
            d = network.add_input("data", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            i = network.add_input("indices", turret.DataType.INT32,
                                  turret.Dimensions((
                                      (K1, turret.DimensionType.INDEX),
                                      (K2, turret.DimensionType.INDEX))))
            h = L.gather(d, i, 1)
            network.mark_output("output", h)
        actual = execute_inference({
                "data": data,
                "indices": indices 
            }, build_network)

        expect = self._gather(data, indices, 1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_constant(self):
        N, C, K, H, W = 5, 13, 17, 19, 23
        data = np.random.rand(C, H, W).astype(np.float32)
        indices = np.random.randint(0, C - 1, (N, K), dtype=np.int32)

        def build_network(network):
            d = network.add_constant(data, turret.Dimensions.CHW(C, H, W))
            i = network.add_input("indices", turret.DataType.INT32,
                                  turret.Dimensions((
                                      (K, turret.DimensionType.INDEX),)))
            h = L.gather(d, i, 0)
            network.mark_output("output", h)
        actual = execute_inference({"indices": indices}, build_network)

        expect = self._gather(np.broadcast_to(data, (N, C, H, W)), indices, 0)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
