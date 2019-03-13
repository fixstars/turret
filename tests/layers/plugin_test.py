# -*- coding: utf-8 -*-
import unittest
import numpy as np

import pycuda.autoinit
from pycuda.compiler import SourceModule

import turret
import turret.layers as L

from util import execute_inference


_THRU_MODULE = SourceModule("""
    __global__ void thru(float *dest, const float *src){
        const int i = blockIdx.x;
        dest[i] = src[i];
    }
""")

class PluginTest(unittest.TestCase):

    def test_thru(self):
        N, C, H, W = 5, 10, 30, 40
        io_dims = turret.Dimensions.CHW(C, H, W)
        input = np.random.rand(N, C, H, W).astype(np.float32)
        test = self

        class ThruPlugin(turret.PluginBase):
            @classmethod
            def module_name(cls):
                return "thru"
            def serialize(self, stream):
                pass
            @classmethod
            def deserialize(cls, stream):
                return cls()
            def get_num_outputs(self):
                return 1
            def get_output_dimensions(self, in_dims):
                test.assertEqual(1, len(in_dims))
                test.assertEqual(io_dims, in_dims[0])
                return [io_dims]
            def configure(self, in_dims, out_dims, max_batch_size):
                test.assertEqual(1, len(in_dims))
                test.assertEqual(io_dims, in_dims[0])
                test.assertEqual(1, len(out_dims))
                test.assertEqual(io_dims, out_dims[0])
            def initialize(self):
                return 0
            def terminate(self):
                pass
            def get_workspace_size(self, max_batch_size):
                return 0
            def enqueue(self, batch_size, inputs, outputs, workspace, stream):
                test.assertEqual(N, batch_size)
                n = batch_size * io_dims.size
                func = _THRU_MODULE.get_function("thru")
                func(outputs[0], inputs[0],
                     block=(1, 1, 1), grid=(n, 1, 1), stream=stream)
                return 0

        def build_network(network):
            h = network.add_input("input", turret.DataType.FLOAT, io_dims)
            h = L.plugin(h, ThruPlugin())
            network.mark_output("output", h)
        actual = execute_inference({"input": input}, build_network)

        expect = input
        self.assertEqual(expect.shape, actual.shape)
        self.assertTrue(np.allclose(expect, actual))
