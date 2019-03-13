from cpython cimport ref

from ..nvinfer cimport nvinfer
from .cy_plugin_proxy cimport PluginProxy

cdef extern from "native_plugin_factory_proxy.hpp" namespace "turret":
    cdef cppclass NativePluginFactoryProxy(nvinfer.IPluginFactory):
        NativePluginFactoryProxy()
        NativePluginFactoryProxy(ref.PyObject *)


cdef class CyPluginFactoryProxy:
    cdef object factory

    cdef PluginProxy create_plugin(
            self, const char *layer_name, const void *serial_data,
            size_t serial_length)


cdef class PluginFactoryProxy:
    cdef NativePluginFactoryProxy *native_proxy
