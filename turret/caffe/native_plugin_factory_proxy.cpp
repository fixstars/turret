#include "native_plugin_factory_proxy.hpp"
#include "plugin_factory_proxy_bridge_api.h"

namespace turret {
namespace caffe {

NativePluginFactoryProxy::NativePluginFactoryProxy()
	: m_impl(nullptr)
{ }

NativePluginFactoryProxy::NativePluginFactoryProxy(PyObject *impl)
	: m_impl(nullptr)
{
	if(import_turret__caffe__plugin_factory_proxy_bridge() != 0){ return; }
	m_impl = impl;
	Py_XINCREF(m_impl);
}

NativePluginFactoryProxy::~NativePluginFactoryProxy(){
	if(m_impl){ Py_XDECREF(m_impl); }
}


bool NativePluginFactoryProxy::isPlugin(const char *layer_name){
	return cy_call_is_plugin(m_impl, layer_name);
}

nvinfer1::IPlugin *NativePluginFactoryProxy::createPlugin(
	const char *layer_name, const nvinfer1::Weights *weights, int num_weights)
{
	return cy_call_create_plugin(m_impl, layer_name, weights, num_weights);
}

}
}
