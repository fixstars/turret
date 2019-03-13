#ifndef TURRET_CAFFE_NATIVE_PLUGIN_FACTORY_PROXY_HPP
#define TURRET_CAFFE_NATIVE_PLUGIN_FACTORY_PROXY_HPP

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "Python.h"

namespace turret {
namespace caffe {
	
class NativePluginFactoryProxy : public nvcaffeparser1::IPluginFactory {

private:
	PyObject *m_impl;

public:
	NativePluginFactoryProxy();
	explicit NativePluginFactoryProxy(PyObject *impl);
	virtual ~NativePluginFactoryProxy();

	virtual bool isPlugin(const char *layer_name) override;
	virtual nvinfer1::IPlugin *createPlugin(
		const char *layer_name, const nvinfer1::Weights *weights,
		int num_weights) override;

};

}
}

#endif
