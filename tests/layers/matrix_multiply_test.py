# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class MatrixMultiplyTest(unittest.TestCase):

    def _matrix_multiply(self, x0, x1):
        N, A, B = x0.shape
        _, _, C = x1.shape
        y = np.zeros((N, A, C), dtype=x0.dtype)
        for i in range(N):
            y[i] = np.matmul(x0[i], x1[i])
        return y

    def test_default(self):
        # AxB * BxC => AxC
        N, A, B, C = 5, 31, 23, 37
        input0 = np.random.rand(N, A, B).astype(np.float32)
        input1 = np.random.rand(N, B, C).astype(np.float32)

        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.HW(A, B))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                  turret.Dimensions.HW(B, C))
            h = L.matrix_multiply(h0, False, h1, False)
            network.mark_output("output", h)
        actual = execute_inference({
                "input0": input0,
                "input1": input1
            }, build_network)

        expect = self._matrix_multiply(input0, input1)
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))

    def test_transposed(self):
        # BxA^T * CxB^T => AxC
        N, A, B, C = 5, 31, 23, 37
        input0 = np.random.rand(N, B, A).astype(np.float32)
        input1 = np.random.rand(N, C, B).astype(np.float32)

        def build_network(network):
            h0 = network.add_input("input0", turret.DataType.FLOAT,
                                  turret.Dimensions.HW(B, A))
            h1 = network.add_input("input1", turret.DataType.FLOAT,
                                  turret.Dimensions.HW(C, B))
            h = L.matrix_multiply(h0, True, h1, True)
            network.mark_output("output", h)
        actual = execute_inference({
                "input0": input0,
                "input1": input1
            }, build_network)

        expect = self._matrix_multiply(
            input0.transpose((0, 2, 1)),
            input1.transpose((0, 2, 1)))
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
