#include <cstdint>
#include "native_plugin_proxy.hpp"
#include "plugin_proxy_bridge_api.h"
#include "../error.hpp"

namespace turret {

NativePluginProxy::NativePluginProxy()
	: m_impl(nullptr)
{ }

NativePluginProxy::NativePluginProxy(PyObject *impl)
	: m_impl(impl)
{
	if(import_turret__plugin__plugin_proxy_bridge() != 0){ return; }
	Py_XINCREF(m_impl);
}

NativePluginProxy::~NativePluginProxy(){
	if(m_impl){ Py_XDECREF(m_impl); }
}


int NativePluginProxy::getNbOutputs() const {
	const auto ret = cy_call_get_num_outputs(m_impl);
	pyerror_test();
	return ret;
}

nvinfer1::Dims NativePluginProxy::getOutputDimensions(
	int index, const nvinfer1::Dims *inputs, int nbInputDims)
{
	const auto ret = cy_call_get_output_dimensions(
		m_impl, index, inputs, nbInputDims);
	pyerror_test();
	return ret;
}

void NativePluginProxy::configure(
	const nvinfer1::Dims *inputDims, int nbInputs,
	const nvinfer1::Dims *outputDims, int nbOutputs, int maxBatchSize)
{
	cy_call_plugin_configure(
		m_impl, inputDims, nbInputs, outputDims, nbOutputs, maxBatchSize);
	pyerror_test();
}


int NativePluginProxy::initialize(){
	const auto ret = cy_call_plugin_initialize(m_impl);
	pyerror_test();
	return ret;
}

void NativePluginProxy::terminate(){
	cy_call_plugin_terminate(m_impl);
	pyerror_test();
}


size_t NativePluginProxy::getWorkspaceSize(int maxBatchSize) const {
	const auto ret = cy_call_plugin_get_workspace_size(m_impl, maxBatchSize);
	pyerror_test();
	return ret;
}


int NativePluginProxy::enqueue(
	int batchSize, const void * const *inputs, void **outputs,
	void *workspace, cudaStream_t stream)
{
	const auto ret = cy_call_plugin_enqueue(
		m_impl, batchSize, inputs, outputs, workspace,
		reinterpret_cast<uintptr_t>(stream));
	pyerror_test();
	return ret;
}


size_t NativePluginProxy::getSerializationSize(){
	const auto ret = cy_call_plugin_serialization_size(m_impl);
	pyerror_test();
	return ret;
}

void NativePluginProxy::serialize(void *buffer){
	cy_call_plugin_serialize(m_impl, buffer);
	pyerror_test();
}

}
