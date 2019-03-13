from libc.stdint cimport uintptr_t
from cpython cimport ref

from ..nvinfer cimport nvinfer


cdef extern from "native_plugin_proxy.hpp" namespace "turret":
    cdef cppclass NativePluginProxy(nvinfer.IPlugin):
        NativePluginProxy()
        NativePluginProxy(ref.PyObject *)


cdef class CyPluginProxy:
    cdef object plugin
    cdef object stream_registory
    cdef int num_inputs
    cdef int num_outputs

    cdef int get_nb_outputs(self) except *
    cdef nvinfer.Dims get_output_dimensions(
        self, int index, const nvinfer.Dims *inputs,
        int nb_input_dims) except *
    cdef void configure(
        self, const nvinfer.Dims *input_dims, int nb_input_dims,
        const nvinfer.Dims *output_dims, int nb_output_dims,
        int max_batch_size) except *
    cdef int initialize(self) except *
    cdef void terminate(self) except *
    cdef size_t get_workspace_size(self, int max_batch_size) except *
    cdef int enqueue(
        self, int batch_size, const void * const *inputs,
        void **outputs, void *workspace, uintptr_t stream) except *
    cdef size_t get_serialization_size(self) except *
    cdef void serialize(self, void *dest) except *

    cdef void _serialize_header(self, stream)


cdef class PluginProxy:
    cdef NativePluginProxy *native_proxy
    cdef object stream_registory
