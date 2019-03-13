# -*- coding: utf-8 -*-
import unittest
import numpy as np

import turret
import turret.layers as L

from util import execute_inference


class SplitTest(unittest.TestCase):

    def test_split_w(self):
        N, C, H, W = 3, 10, 13, 11 
        sections = [2, 5, 6, 8]
        input = np.random.rand(N, C, H, W).astype(np.float32)

        for i in range(len(sections) + 1):
            def build_network(network):
                h = network.add_input("input", turret.DataType.FLOAT,
                                      turret.Dimensions.CHW(C, H, W))
                h = L.split(h, sections, axis=2)
                network.mark_output("output", h[i])
            actual = execute_inference({"input": input}, build_network)

            expect = np.split(input, sections, axis=3)[i]
            self.assertEqual(expect.shape, actual.shape)
            if not np.allclose(expect, actual):
                print(expect)
                print(actual)
            self.assertTrue(np.allclose(expect, actual))


    def test_split_h(self):
        N, C, H, W = 3, 10, 13, 11 
        sections = [2, 5, 6, 8]
        input = np.random.rand(N, C, H, W).astype(np.float32)

        for i in range(len(sections) + 1):
            def build_network(network):
                h = network.add_input("input", turret.DataType.FLOAT,
                                      turret.Dimensions.CHW(C, H, W))
                h = L.split(h, sections, axis=1)
                network.mark_output("output", h[i])
            actual = execute_inference({"input": input}, build_network)

            expect = np.split(input, sections, axis=2)[i]
            self.assertEqual(expect.shape, actual.shape)
            if not np.allclose(expect, actual):
                print(expect)
                print(actual)
            self.assertTrue(np.allclose(expect, actual))


    def test_split_c(self):
        N, C, H, W = 3, 10, 13, 11 
        sections = [2, 5, 6, 8]
        input = np.random.rand(N, C, H, W).astype(np.float32)

        for i in range(len(sections) + 1):
            def build_network(network):
                h = network.add_input("input", turret.DataType.FLOAT,
                                      turret.Dimensions.CHW(C, H, W))
                h = L.split(h, sections, axis=0)
                network.mark_output("output", h[i])
            actual = execute_inference({"input": input}, build_network)

            expect = np.split(input, sections, axis=1)[i]
            self.assertEqual(expect.shape, actual.shape)
            if not np.allclose(expect, actual):
                print(expect)
                print(actual)
            self.assertTrue(np.allclose(expect, actual))
