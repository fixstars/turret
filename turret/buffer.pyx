# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import numpy as np
cimport numpy as np

from libc.stdint cimport uintptr_t

import pycuda.autoinit
import pycuda.driver as cuda

from .engine cimport Binding
from .engine cimport InferenceEngine


cdef class InferenceBuffer:
    """An object of input and output buffers for the network."""
    def __cinit__(self, InferenceEngine engine):
        self.engine = engine
        self.allocations = []
        self.bindings.clear()
        self.batch_size = 0
        cdef:
            int max_batch_size
            Binding binding
            ssize_t elem_size, elem_count
            uintptr_t allocation_ptr
        max_batch_size = engine.max_batch_size
        for binding in engine.bindings:
            elem_size = binding.type.size
            elem_count = binding.dimensions.size
            self.allocations.append(cuda.mem_alloc(
                max_batch_size * elem_count * elem_size))
        for allocation in self.allocations:
            allocation_ptr = <uintptr_t>int(allocation)
            self.bindings.push_back(<void *>allocation_ptr)

    def __dealloc__(self):
        for allocation in self.allocations:
            allocation.free()

    cdef void **binding_pointers(self):
        return self.bindings.data()

    def put(self, str name, np.ndarray batch):
        """Putting data on buffer.

        Args:
            name(str): Buffer name on the network.
            batch(np.ndarray): Data to put on buffer.
        """
        if batch.shape[0] > self.engine.max_batch_size:
            raise ValueError("too large batch size")
        cdef Binding binding = self.engine.bindings[name]
        cdef int index = binding.index
        cdef tuple shape = binding.dimensions.shape
        if (<object>batch).shape[1:] != shape:
            raise ValueError("shape mismatch")
        if batch.flags['C_CONTIGUOUS']:
            cuda.memcpy_htod(self.allocations[index], batch)
        else:
            cuda.memcpy_htod(self.allocations[index],
                             np.ascontiguousarray(batch))
        self.batch_size = batch.shape[0]

    def get(self, str name):
        """Getting data from buffer.

        Args:
            name(str): Buffer name on the network.

        Returns:
            np.ndarray: Data got from buffer.
        """
        cdef Binding binding = self.engine.bindings[name]
        cdef int index = binding.index
        cdef tuple shape = (self.batch_size,) + binding.dimensions.shape
        cdef np.ndarray hmem = np.empty(shape, binding.type.nptype)
        cuda.memcpy_dtoh(hmem, self.allocations[index])
        return hmem

    def dimensions(self, str name):
        """Getting dimensions of data on buffer.

        Args:
            name(str): Buffer name on the network.

        Returns:
            tuple: dimensions of data.
        """
        cdef Binding binding = self.engine.bindings[name]
        return binding.dimensions

    def swap(self, str a, str b):
        """Swapping data on buffer.

        Args:
            a(str): Buffer name on the network to swap.
            b(str): Buffer name on the network to swap.
        """
        cdef Binding a_binding = self.engine.bindings[a]
        cdef Binding b_binding = self.engine.bindings[b]
        # WA for TensorRT4 suspicious behavior: dimension type mismatch
        #if a_binding.dimensions != b_binding.dimensions:
        if a_binding.dimensions.shape != b_binding.dimensions.shape:
            raise ValueError("shape mismatch")
        a_index = a_binding.index
        b_index = b_binding.index
        self.allocations[a_index], self.allocations[b_index] = \
                self.allocations[b_index], self.allocations[a_index]
        self.bindings[a_index], self.bindings[b_index] = \
                self.bindings[b_index], self.bindings[a_index]

    def get_device_memory(self, str name):
        """Getting a cuda memory for the buffer data.

        Args:
            name(str): Buffer name on the network.

        Returns:
            binary: A cuda memory for the buffer data.
        """
        cdef Binding binding = self.engine.bindings[name]
        cdef int index = binding.index
        return self.allocations[index]
