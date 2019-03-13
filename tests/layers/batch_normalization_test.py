# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class BatchNormalizationTest(unittest.TestCase):
    def test_default(self):
        N, C, H, W = 9, 3, 20, 30
        eps = 2e-5
        input = np.random.rand(N, C, H, W).astype(np.float32)
        input = np.ones((N, C, H, W), dtype=np.float32)
        mean = np.random.rand(C).astype(np.float32)
        var = np.random.rand(C).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.batch_normalization(h, mean, var, eps=eps)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        bcast_mean = np.broadcast_to(mean, (W, H, C)).transpose()
        bcast_std = np.broadcast_to(var, (W, H, C)).transpose() ** 0.5
        expect = (input - bcast_mean) / (bcast_std + eps)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_gamma_beta(self):
        N, C, H, W = 9, 3, 20, 30
        eps = 2e-5
        input = np.random.rand(N, C, H, W).astype(np.float32)
        input = np.ones((N, C, H, W), dtype=np.float32)
        mean = np.random.rand(C).astype(np.float32)
        var = np.random.rand(C).astype(np.float32)
        gamma = np.random.rand(C).astype(np.float32)
        beta = np.random.rand(C).astype(np.float32)

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT,
                                  turret.Dimensions.CHW(C, H, W))
            h = L.batch_normalization(h, mean, var, gamma=gamma, beta=beta, eps=eps)
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        bcast_mean = np.broadcast_to(mean, (W, H, C)).transpose()
        bcast_std = np.broadcast_to(var, (W, H, C)).transpose() ** 0.5
        bcast_gamma = np.broadcast_to(gamma, (W, H, C)).transpose()
        bcast_beta = np.broadcast_to(beta, (W, H, C)).transpose()
        expect = bcast_gamma * (input - bcast_mean) / (bcast_std + eps) + bcast_beta
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
