#include <iostream>
#include "native_plugin_factory_proxy.hpp"
#include "plugin_factory_proxy_bridge_api.h"
#include "../error.hpp"

namespace turret {

NativePluginFactoryProxy::NativePluginFactoryProxy()
	: m_impl(nullptr)
{ }

NativePluginFactoryProxy::NativePluginFactoryProxy(PyObject *impl)
	: m_impl(nullptr)
{
	if(import_turret__plugin__plugin_factory_proxy_bridge() != 0){ return; }
	m_impl = impl;
	Py_XINCREF(m_impl);
}

NativePluginFactoryProxy::~NativePluginFactoryProxy(){
	if(m_impl){ Py_XDECREF(m_impl); }
}


nvinfer1::IPlugin *NativePluginFactoryProxy::createPlugin(
	const char *layerName, const void *serialData, size_t serialLength)
{
	const auto ret = cy_call_create_plugin(
		m_impl, layerName, serialData, serialLength);
	pyerror_test();
	return ret;
}

}
