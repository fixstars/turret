# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvcaffe_parser", "nvinfer"]
from libcpp  cimport bool

from ..nvinfer cimport nvinfer
from ..plugin.cy_plugin_proxy cimport PluginProxy
from .cy_plugin_factory_proxy cimport CyPluginFactoryProxy


cdef public api bool cy_call_is_plugin(object self, const char *layer_name):
    cdef CyPluginFactoryProxy factory = self
    return factory.is_plugin(layer_name)

cdef public api nvinfer.IPlugin *cy_call_create_plugin(
        object self, const char *layer_name,
        const nvinfer.Weights *weights, int num_weights):
    cdef CyPluginFactoryProxy factory = self
    cdef PluginProxy plugin = \
            factory.create_plugin(layer_name, weights, num_weights)
    return plugin.native_proxy
