#ifndef TURRET_PLUGIN_NATIVE_PLUGIN_PROXY_HPP
#define TURRET_PLUGIN_NATIVE_PLUGIN_PROXY_HPP

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "Python.h"

namespace turret {

class NativePluginProxy : public nvinfer1::IPlugin {

private:
	PyObject *m_impl;

public:
	NativePluginProxy();
	NativePluginProxy(PyObject *impl);
	virtual ~NativePluginProxy();

	virtual int getNbOutputs() const override;
	virtual nvinfer1::Dims getOutputDimensions(
		int index, const nvinfer1::Dims *inputs, int nbInputDims) override;

	virtual void configure(
		const nvinfer1::Dims *inputDims, int nbInputs,
		const nvinfer1::Dims *outputDims, int nbOutputs, int maxBatchSize) override;

	virtual int initialize() override;
	virtual void terminate() override;

	virtual size_t getWorkspaceSize(int maxBatchSize) const override;

	virtual int enqueue(
		int batchSize, const void * const *inputs, void **outputs,
		void *workspace, cudaStream_t stream) override;

	virtual size_t getSerializationSize() override;
	virtual void serialize(void *buffer) override;

};

}

#endif
