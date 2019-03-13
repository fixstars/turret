# -*- coding: utf-8 -*-
import numpy as np

import pycuda.autoinit  # noqa
import pycuda.driver as cuda


class _CalibratorBuffer:

    def __init__(self, bindings, batch_size):
        self.bindings = bindings
        self.allocations = {}
        for binding in self.bindings.values():
            elem_size = binding.type.size
            elem_count = binding.dimensions.size
            self.allocations[binding.name] = \
                cuda.mem_alloc(batch_size * elem_size * elem_count)

    def release(self):
        for mem in self.allocations.values():
            mem.free()
        self.allocations = {}

    def put(self, name, index, value):
        binding = self.bindings[name]
        if value.dtype != binding.type.nptype:
            raise TypeError()
        if value.shape != binding.dimensions.shape:
            raise ValueError()
        allocation = self.allocations[name]
        elem_size = binding.type.size
        elem_count = binding.dimensions.size
        dstptr = int(allocation) + index * elem_size * elem_count
        if value.flags["C_CONTIGUOUS"]:
            cuda.memcpy_htod(dstptr, value)
        else:
            cuda.memcpy_htod(dstptr, np.ascontiguousarray(value))


class Int8Calibrator:
    """The object to use INT8 calibrator.

    Args:
        samples(object): The samples for INT8 calibrator.
        batch_size(int): The batch size.

    """

    def __init__(self, samples, batch_size):
        self.batch_size = batch_size
        self.iterator = iter(samples)
        self.network = None
        self.buffer = None

    def get_batch(self, names):
        """Get the batch of input for calibration.

        Args:
            names(list): The names of the network input.

        Returns:
            batch(list): The batch of input for calibration.

        """
        assert self.network is not None
        if self.buffer is not None:
            self.buffer.release()
            self.buffer = None
        self.buffer = _CalibratorBuffer(
                self.network.input_bindings, self.batch_size)
        for i in range(self.batch_size):
            try:
                sample = next(self.iterator)
            except StopIteration:
                self.buffer.release()
                self.buffer = None
                return None
            if type(sample) is not dict:
                if len(names) == 1:
                    sample = {names[0]: sample}
                else:
                    raise ValueError()
            for key in names:
                self.buffer.put(key, i, sample[key])
        return [self.buffer.allocations[key] for key in names]

    def get_batch_size(self):
        """Get the batch size.

        Returns:
            batch_size(int): The batch size.
        """
        return self.batch_size
