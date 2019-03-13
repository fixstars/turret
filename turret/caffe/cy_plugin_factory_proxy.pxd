from libcpp cimport bool
from cpython cimport ref

from ..nvinfer cimport nvinfer
from ..nvinfer cimport caffe
from ..plugin.cy_plugin_proxy cimport PluginProxy


cdef extern from "native_plugin_factory_proxy.hpp" namespace "turret::caffe":
    cdef cppclass NativePluginFactoryProxy(caffe.IPluginFactory):
        NativePluginFactoryProxy()
        NativePluginFactoryProxy(ref.PyObject *)


cdef class CyPluginFactoryProxy:
    cdef object factory

    cdef bool is_plugin(self, const char *layer_name)
    cdef PluginProxy create_plugin(
            self, const char *layer_name, const nvinfer.Weights *weights,
            int num_weights)


cdef class PluginFactoryProxy:
    cdef NativePluginFactoryProxy *native_proxy
    cdef object factory
