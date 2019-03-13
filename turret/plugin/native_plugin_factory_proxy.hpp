#ifndef TURRET_PLUGIN_NATIVE_PLUGIN_FACTORY_PROXY_HPP
#define TURRET_PLUGIN_NATIVE_PLUGIN_FACTORY_PROXY_HPP

#include "NvInfer.h"
#include "Python.h"

namespace turret {

class NativePluginFactoryProxy : public nvinfer1::IPluginFactory {

private:
	PyObject *m_impl;

public:
	NativePluginFactoryProxy();
	explicit NativePluginFactoryProxy(PyObject *impl);
	virtual ~NativePluginFactoryProxy();

	virtual nvinfer1::IPlugin *createPlugin(
		const char *layerName, const void *serialData,
		size_t serialLength) override;

};

}

#endif
