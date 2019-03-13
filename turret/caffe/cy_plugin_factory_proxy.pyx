# distutils: language = c++
# distutils: sources = ["turret/caffe/native_plugin_factory_proxy.cpp"]
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvcaffe_parser", "nvinfer"]
from libcpp cimport bool

import numpy as np
cimport numpy as np

from ..foundational cimport DataType
from ..nvinfer cimport nvinfer
from ..plugin.cy_plugin_proxy cimport PluginProxy


cdef class CyPluginFactoryProxy:
    """The class for internal processing. Users need not to use."""

    def __init__(self, factory):
        self.factory = factory

    cdef bool is_plugin(self, const char *layer_name):
        return self.factory.is_plugin(layer_name.decode("utf-8"))

    cdef PluginProxy create_plugin(
            self, const char *layer_name, const nvinfer.Weights *weights,
            int num_weights):
        py_weights = []
        for i in range(num_weights):
            # TODO float16, int8
            w = weights[i]
            dtype = DataType(<int>w.type)
            n = w.count
            a = np.copy(<float[:n]>w.values)
            py_weights.append(a)
        return self.factory.create(
                layer_name.decode("utf-8"), py_weights)


cdef class PluginFactoryProxy:
    """The class for internal processing. Users need not to use."""

    def __cinit__(self, factory):
        cdef CyPluginFactoryProxy cy_proxy = CyPluginFactoryProxy(factory)
        self.native_proxy = \
                new NativePluginFactoryProxy(<ref.PyObject *>cy_proxy)
        self.factory = factory

    def __dealloc__(self):
        if self.native_proxy:
            del self.native_proxy
