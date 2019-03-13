# -*- coding: utf-8 -*-
import pickle

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import turret


_kernel = SourceModule("""
    __global__ void fully_connected(
        float *dest, const float *src,
        const float *weights, const float *bias,
        const int N, const int C, const int K)
    {
        const int i = threadIdx.x + blockDim.x * blockIdx.x;
        const int j = threadIdx.y + blockDim.y * blockIdx.y;
        if(i >= N || j >= K){ return; }
        float sum = bias[j];
        for(int k = 0; k < C; ++k){
            sum += weights[j * C + k] * src[i * C + k];
        }
        dest[i * K + j] = sum;
    }
""")

class FullyConnectedPlugin(turret.PluginBase):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.device_weights = None
        self.device_bias = None

    @classmethod
    def module_name(cls):
        return "fully_connected"

    def serialize(self, stream):
        pickle.dump((self.weights, self.bias), stream)

    @classmethod
    def deserialize(cls, stream):
        fc = FullyConnectedPlugin(None, None)
        fc.weights, fc.bias = pickle.load(stream)
        return fc

    def get_num_outputs(self):
        return 1

    def get_output_dimensions(self, in_dims):
        in_dims = in_dims[0]
        out_dims = turret.Dimensions.CHW(self.bias.size, 1, 1)
        return [out_dims]

    def configure(self, in_dims, out_dims, max_batch_size):
        pass

    def initialize(self):
        self.device_weights = cuda.to_device(self.weights)
        self.device_bias = cuda.to_device(self.bias)
        return 0

    def terminate(self):
        self.device_weights.free()
        self.device_bias.free()

    def get_workspace_size(self, max_batch_size):
        return 0

    def enqueue(self, batch_size, inputs, outputs, workspace, stream):
        N = batch_size
        C = self.weights.size // self.bias.size
        K = self.bias.size
        block = (16, 16, 1)
        grid = ((N+block[0]-1)//block[0], (K+block[1]-1)//block[1], 1)
        func = _kernel.get_function("fully_connected")
        func(outputs[0], inputs[0], self.device_weights, self.device_bias,
             np.int32(N), np.int32(C), np.int32(K),
             block=block, grid=grid, stream=stream)
        return 0
