# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
from libc.stdint cimport uintptr_t
from libcpp cimport bool

from ..nvinfer cimport nvinfer
from .cy_plugin_proxy cimport CyPluginProxy


cdef public api int cy_call_get_num_outputs(object self) except *:
    cdef CyPluginProxy proxy = self
    return proxy.get_nb_outputs()

cdef public api nvinfer.Dims cy_call_get_output_dimensions(
        object self, int index, const nvinfer.Dims *inputs,
        int nb_input_dims) except *:
    cdef CyPluginProxy proxy = self
    return proxy.get_output_dimensions(index, inputs, nb_input_dims)


cdef public api void cy_call_plugin_configure(
        object self, const nvinfer.Dims *input_dims,
        int nb_input_dims, const nvinfer.Dims *output_dims,
        int nb_output_dims, int max_batch_size) except *:
    cdef CyPluginProxy proxy = self
    proxy.configure(input_dims, nb_input_dims, output_dims,
                    nb_output_dims, max_batch_size)

cdef public api int cy_call_plugin_initialize(object self) except *:
    cdef CyPluginProxy proxy = self
    return proxy.initialize()

cdef public api void cy_call_plugin_terminate(object self) except *:
    cdef CyPluginProxy proxy = self
    proxy.terminate()


cdef public api size_t cy_call_plugin_get_workspace_size(
        object self, int max_batch_size) except *:
    cdef CyPluginProxy proxy = self
    return proxy.get_workspace_size(max_batch_size)


cdef public api int cy_call_plugin_enqueue(
        object self, int batch_size, const void * const * inputs,
        void **outputs, void *workspace, uintptr_t stream) except *:
    cdef CyPluginProxy proxy = self
    return proxy.enqueue(batch_size, inputs, outputs, workspace, stream)


cdef public api size_t cy_call_plugin_serialization_size(
        object self) except *:
    cdef CyPluginProxy proxy = self
    return proxy.get_serialization_size()

cdef public api void cy_call_plugin_serialize(
        object self, void *dest) except *:
    cdef CyPluginProxy proxy = self
    proxy.serialize(dest)
