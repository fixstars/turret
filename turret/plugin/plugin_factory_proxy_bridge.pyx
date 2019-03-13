# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
from ..nvinfer cimport nvinfer
from .cy_plugin_proxy cimport PluginProxy
from .cy_plugin_factory_proxy cimport CyPluginFactoryProxy


cdef public api nvinfer.IPlugin *cy_call_create_plugin(
        object self, const char *layer_name,
        const void *serial_data, size_t serial_length):
    cdef CyPluginFactoryProxy factory = self
    cdef PluginProxy plugin = \
            factory.create_plugin(layer_name, serial_data, serial_length)
    return plugin.native_proxy
