# distutils: language = c++
# distutils: sources = ["turret/plugin/native_plugin_proxy.cpp"]
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import io
import json

import numpy as np
cimport numpy as np

from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

from ..nvinfer cimport nvinfer
from ..foundational cimport Dimensions

from .stream import StreamRegistory
from .stream import PluginStream


cdef list _make_dimensions_list(const nvinfer.Dims *ptr, int n):
    return [Dimensions.from_nvinfer_dims(ptr[i]) for i in range(n)]


cdef class CyPluginProxy:

    def __init__(self, plugin, stream_registory, proxy_params):
        self.plugin = plugin
        self.stream_registory = stream_registory
        if proxy_params:
            self.num_inputs = proxy_params["num_inputs"]
            self.num_outputs = proxy_params["num_outputs"]

    cdef int get_nb_outputs(self) except *:
        return self.plugin.get_num_outputs()

    cdef nvinfer.Dims get_output_dimensions(
            self, int index, const nvinfer.Dims *inputs,
            int nb_input_dims) except *:
        # TODO cache output dimensions
        outputs = self.plugin.get_output_dimensions(
                _make_dimensions_list(inputs, nb_input_dims))
        cdef Dimensions d = outputs[index]
        return d.to_nvinfer_dims()

    cdef void configure(
            self, const nvinfer.Dims *input_dims, int nb_input_dims,
            const nvinfer.Dims *output_dims, int nb_output_dims,
            int max_batch_size) except *:
        self.num_inputs = nb_input_dims
        self.num_outputs = nb_output_dims
        self.plugin.configure(
                _make_dimensions_list(input_dims, nb_input_dims),
                _make_dimensions_list(output_dims, nb_output_dims),
                max_batch_size)

    cdef int initialize(self) except *:
        self.plugin.initialize()
        return 0

    cdef void terminate(self) except *:
        self.plugin.terminate()

    cdef size_t get_workspace_size(self, int max_batch_size) except *:
        return self.plugin.get_workspace_size(max_batch_size)

    cdef int enqueue(
            self, int batch_size, const void * const *inputs,
            void **outputs, void *workspace, uintptr_t stream) except *:
        py_inputs = [
            np.intp(<uintptr_t>inputs[i])
            for i in range(self.num_inputs)]
        py_outputs = [
            np.intp(<uintptr_t>outputs[i])
            for i in range(self.num_outputs)]
        py_workspace = np.intp(<uintptr_t>workspace)
        with PluginStream(stream, self.stream_registory) as py_stream:
            self.plugin.enqueue(batch_size, py_inputs, py_outputs,
                                py_workspace, py_stream)
        return 0

    cdef size_t get_serialization_size(self) except *:
        # TODO cache serialized data
        with io.BytesIO() as stream:
            self._serialize_header(stream)
            self.plugin.serialize(stream)
            return len(stream.getvalue())

    cdef void serialize(self, void *dest) except *:
        # TODO streaming write to dest
        with io.BytesIO() as stream:
            self._serialize_header(stream)
            self.plugin.serialize(stream)
            serialized = stream.getvalue()
            memcpy(dest, <const char *>serialized, len(serialized))

    cdef void _serialize_header(self, stream):
        obj = {
            "name": self.plugin.module_name(),
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs
        }
        stream.write(json.dumps(
            obj, separators=(",", ":")).encode("utf-8") + b"\0")


cdef class PluginProxy:
    """The proxy object of Turret PluginBase.

    Args:
        plugin(turret.PluginBase): The object of PluginBase
        proxy_params(dict): The parameter of plugin proxy."num_inputs"(int)
            and "num_outputs"(int) are required.
    """

    def __cinit__(self, plugin, proxy_params=None):
        stream_registory = StreamRegistory()
        cy_proxy = CyPluginProxy(plugin, stream_registory, proxy_params)
        self.stream_registory = stream_registory
        self.native_proxy = new NativePluginProxy(<ref.PyObject *>cy_proxy)

    def __dealloc__(self):
        if self.native_proxy:
            del self.native_proxy

    def register_stream(self, stream):
        """Register cuda stream on the instance.

        Args:
            stream(cuda.Stream): The cuda stream to be used on the instance.
        """
        self.stream_registory.register(stream)

    def unregister_stream(self, stream):
        """Unregister cuda stream from the instance.

        Args:
            stream(cuda.Stream): The cuda stream to be used on the instance.
        """
        self.stream_registory.unregister(stream)
