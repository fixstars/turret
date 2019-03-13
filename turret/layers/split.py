# -*- coding: utf-8 -*-
import pickle

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .builtin import plugin
from ..foundational import Dimension
from ..foundational import Dimensions
from ..plugin.plugin_base import PluginBase
from ..plugin.auto_register import auto_register


@auto_register
class SplitPlugin(PluginBase):

    def __init__(self, sections, axis):
        self.sections = list(sections)
        self.axis = axis
        self.in_dims = None

    @classmethod
    def module_name(cls):
        return "turret.layers.split"

    def serialize(self, stream):
        pickle.dump((self.sections, self.axis, self.in_dims), stream)

    @classmethod
    def deserialize(cls, stream):
        sections, axis, in_dims = pickle.load(stream)
        p = SplitPlugin(sections, axis)
        p.in_dims = in_dims
        return p

    def get_num_outputs(self):
        return len(self.sections)

    def get_output_dimensions(self, in_dims):
        self.in_dims = in_dims[0]
        out_dims = []
        last = 0
        for pos in self.sections:
            dims = list(self.in_dims)
            dims[self.axis] = Dimension(pos - last, dims[self.axis].type)
            out_dims.append(Dimensions(dims))
            last = pos
        return out_dims

    def enqueue(self, batch_size, inputs, output, workspace, stream):
        ELEM_SIZE = 4
        in_dims = list(self.in_dims)
        last = 0
        for i, pos in enumerate(self.sections):
            acc_size_lo, acc_size_hi = 1, 1
            for d in list(self.in_dims)[self.axis+1:]:
                acc_size_lo *= d.size
            for d in list(self.in_dims)[:self.axis]:
                acc_size_hi *= d.size

            src_offset = acc_size_lo * last
            src_pitch = acc_size_lo * self.in_dims[self.axis].size
            dst_pitch = acc_size_lo * (pos - last)
            height = acc_size_hi * batch_size

            copy = cuda.Memcpy2D()
            copy.set_src_device(int(inputs[0]))
            copy.set_dst_device(int(output[i]))
            copy.src_x_in_bytes = src_offset * ELEM_SIZE
            copy.src_pitch = src_pitch * ELEM_SIZE
            copy.dst_pitch = dst_pitch * ELEM_SIZE
            copy.width_in_bytes = dst_pitch * ELEM_SIZE
            copy.height = height
            copy(stream)
            last = pos


def split(input, indices_or_sections, axis):
    """Layer to split the tensor.

    Args:
        input(turret.Tensor): Tensor which will be splited.
        indices_or_sections(tuple): The indices or sections to split.
        axis(int): The axis to be split.

    Returns:
        tensor(turret.Tensor): The splited tensor.
    """
    if type(indices_or_sections) is int:
        step = indices_or_sections
        size = input.dimensions[axis].size
        indices_or_sections = list(range(step, size, step)) + [size]
    else:
        size = input.dimensions[axis].size
        indices_or_sections = list(indices_or_sections) + [size]
    return plugin(input, SplitPlugin(indices_or_sections, axis))
