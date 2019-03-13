# distutils: language = c++
# distutils: sources = ["turret/plugin/native_plugin_factory_proxy.cpp"]
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import io

from ..nvinfer cimport nvinfer
from .cy_plugin_proxy cimport PluginProxy


cdef class CyPluginFactoryProxy:

    def __init__(self, factory):
        self.factory = factory

    cdef PluginProxy create_plugin(
            self, const char *layer_name, const void *serial_data,
            size_t serial_length):
        cdef const char *ptr = <const char *>serial_data
        with io.BytesIO(ptr[:serial_length]) as stream:
            return self.factory.create(stream)


cdef class PluginFactoryProxy:

    def __cinit__(self, factory):
        cdef CyPluginFactoryProxy cy_proxy = CyPluginFactoryProxy(factory)
        self.native_proxy = \
                new NativePluginFactoryProxy(<ref.PyObject *>cy_proxy)

    def __dealloc__(self):
        if self.native_proxy:
            del self.native_proxy
