# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class Convolution2DTest(unittest.TestCase):

    def _convolution_2d(self, x, W, b, stride, pad):
        N, C, iH, iW = x.shape
        K, _, kH, kW = W.shape
        assert W.shape[1] == C
        sH, sW = stride
        pH, pW = pad
        oH = (iH + 2*pH - kH) // sH + 1
        oW = (iW + 2*pW - kW) // sW + 1
        y = np.zeros((N, K, oH, oW), dtype=np.float32)
        for i in range(N):
            for k in range(K):
                for oy in range(oH):
                    for ox in range(oW):
                        s = 0.0 if b is None else b[k]
                        for c in range(C):
                            for ky in range(kH):
                                for kx in range(kW):
                                    iy = oy*sH - pH + ky
                                    ix = ox*sW - pW + kx
                                    if 0 <= iy < iH and 0 <= ix < iW:
                                        s +=  x[i, c, iy, ix] \
                                            * W[k, c, ky, kx]
                        y[i, k, oy, ox] = s
        return y

    def test_default(self):
        cases = [
            # N, K, C, H, W, kH, kW
            ( 1, 4, 5, 7, 8,  3,  3),
            ( 4, 8, 1, 7, 8,  3,  2),
            ( 1, 1, 2, 3, 5,  1,  4),
        ]
        for case in cases:
            N, K, C, H, W, kH, kW = case
            input = np.random.rand(N, C, H, W).astype(np.float32)
            filters = np.random.rand(K, C, kH, kW).astype(np.float32)
            biases = np.random.rand(K).astype(np.float32)

            def build_network(network):
                h = network.add_input("input", turret.DataType.FLOAT,
                                      turret.Dimensions.CHW(C, H, W))
                h = L.convolution_2d(h, filters, biases)
                network.mark_output("output", h)
            actual = execute_inference({"input": input}, build_network)

            expect = self._convolution_2d(input, filters, biases,
                                          (1, 1), (0, 0))
            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))


    def test_no_biases(self):
        cases = [
            # N, K, C, H, W, kH, kW
            ( 1, 4, 5, 7, 8,  3,  3),
            ( 4, 8, 1, 7, 8,  3,  2),
            ( 1, 1, 2, 3, 5,  1,  4),
        ]
        for case in cases:
            N, K, C, H, W, kH, kW = case
            input = np.random.rand(N, C, H, W).astype(np.float32)
            filters = np.random.rand(K, C, kH, kW).astype(np.float32)

            def build_network(network):
                h = network.add_input("input", turret.DataType.FLOAT,
                                      turret.Dimensions.CHW(C, H, W))
                h = L.convolution_2d(h, filters)
                network.mark_output("output", h)
            actual = execute_inference({"input": input}, build_network)

            expect = self._convolution_2d(input, filters,
                                          np.zeros(K, dtype=np.float32),
                                          (1, 1), (0, 0))
            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))


    def test_stride_and_padding(self):
        cases = [
            # N, K, C, H, W, kH, kW, sH, sW, pH, pW
            ( 1, 4, 5, 7, 8,  3,  3,  3,  1,  0,  0),
            ( 4, 8, 1, 7, 8,  3,  2,  1,  1,  2,  1),
            #( 4, 8, 1, 7, 8,  3,  2,  1,  1,  2,  4), # fails on TensorRT 2.1
            #( 1, 1, 2, 3, 5,  1,  4,  2,  2,  4,  4), # fails on TensorRT 2.1
        ]
        for case in cases:
            N, K, C, H, W, kH, kW, sH, sW, pH, pW = case
            input = np.random.rand(N, C, H, W).astype(np.float32)
            filters = np.random.rand(K, C, kH, kW).astype(np.float32)
            biases = np.random.rand(K).astype(np.float32)

            def build_network(network):
                h = network.add_input("input", turret.DataType.FLOAT,
                                      turret.Dimensions.CHW(C, H, W))
                h = L.convolution_2d(h, filters, biases,
                                     stride=(sH, sW), padding=(pH, pW))
                network.mark_output("output", h)
            actual = execute_inference({"input": input}, build_network)

            expect = self._convolution_2d(input, filters, biases,
                                          (sH, sW), (pH, pW))
            self.assertEqual(expect.shape, actual.shape)
            self.assertTrue(np.allclose(expect, actual))
