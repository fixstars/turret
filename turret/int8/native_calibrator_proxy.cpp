#include "native_calibrator_proxy.hpp"
#include "calibrator_proxy_bridge_api.h"
#include "../error.hpp"

namespace turret {

NativeCalibratorProxy::NativeCalibratorProxy()
	: m_impl(nullptr)
{ }

NativeCalibratorProxy::NativeCalibratorProxy(PyObject *impl)
	: m_impl(nullptr)
{
	if(import_turret__int8__calibrator_proxy_bridge() != 0){
		throw std::runtime_error("import_turret__int8__calibrator_proxy_bridge");
	}
	m_impl = impl;
	Py_XINCREF(m_impl);
}

NativeCalibratorProxy::~NativeCalibratorProxy(){
	if(m_impl){ Py_XDECREF(m_impl); }
}

int NativeCalibratorProxy::getBatchSize() const {
	const auto ret = cy_call_get_batch_size(m_impl);
	pyerror_test();
	return ret;
}

bool NativeCalibratorProxy::getBatch(
	void *bindings[], const char *names[], int nbBindings)
{
	const auto ret =
		cy_call_get_batch(m_impl, bindings, names, nbBindings);
	pyerror_test();
	return ret;
}

const void *NativeCalibratorProxy::readCalibrationCache(std::size_t& length){
	const auto ret =
		cy_call_read_calibration_cache(m_impl, length);
	pyerror_test();
	return ret;
}

void NativeCalibratorProxy::writeCalibrationCache(
	const void *ptr, std::size_t length)
{
	cy_call_write_calibration_cache(m_impl, ptr, length);
	pyerror_test();
}

}
