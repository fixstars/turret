# -*- coding: utf-8 -*-
import pickle

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .builtin import plugin
from ..foundational import Dimensions
from ..plugin.plugin_base import PluginBase
from ..plugin.auto_register import auto_register


_kernel = SourceModule("""
    __global__ void repeat(
        float *dest, const float *src,
        const unsigned int lower_size,
        const unsigned int upper_size,
        const unsigned int repeats)
    {
        const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
        const unsigned int y = blockIdx.y;
        if(x >= lower_size || y >= upper_size){ return; }
        const float value = src[y * lower_size + x];
        dest += y * lower_size * repeats + x;
        for(unsigned int i = 0; i < repeats; ++i){
            *dest = value;
            dest += lower_size;
        }
    }
""")


def _ceil_div(x, y):
    return (x + y - 1) // y


@auto_register
class RepeatPlugin(PluginBase):

    def __init__(self, repeats, axis):
        self.repeats = repeats
        self.axis = axis
        self.in_dims = None

    @classmethod
    def module_name(cls):
        return "turret.layers.repeat"

    def serialize(self, stream):
        pickle.dump((self.repeats, self.axis, self.in_dims), stream)

    @classmethod
    def deserialize(cls, stream):
        repeats, axis, in_dims = pickle.load(stream)
        p = RepeatPlugin(repeats, axis)
        p.in_dims = in_dims
        return p

    def get_num_outputs(self):
        return 1

    def get_output_dimensions(self, in_dims):
        self.in_dims = in_dims[0]
        out_dims = []
        for i, d in enumerate(in_dims[0]):
            out_dims.append((
                (d.size * self.repeats) if i == self.axis else d.size,
                d.type))
        return [Dimensions(out_dims)]

    def enqueue(self, batch_size, inputs, outputs, workspace, stream):
        upper_size = self.in_dims[:self.axis+1].size * batch_size
        lower_size = self.in_dims[self.axis+1:].size
        block = (256, 1, 1)
        grid = (_ceil_div(lower_size, block[0]), upper_size, 1)
        func = _kernel.get_function("repeat")
        func(outputs[0], inputs[0], np.int32(lower_size),
             np.int32(upper_size), np.int32(self.repeats),
             block=block, grid=grid, stream=stream)


def repeat(input, repeats, axis=0):
    """Plugin layer.

    Args:
        input(turret.Tensor): Tensors which will be processed.
        repeats(int): The number of repeat.
        axis(int): Axis to repeat.

    Returns:
        tensor(turret.Tensor): Processed tensor.
    """
    return plugin(input, RepeatPlugin(repeats, axis))
