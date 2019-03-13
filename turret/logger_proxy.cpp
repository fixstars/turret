#include <iostream>
#include "logger_proxy.hpp"
#include "logger_bridge_api.h"

namespace turret {

NativeLoggerProxy::NativeLoggerProxy()
	: m_impl(nullptr)
{ }

NativeLoggerProxy::NativeLoggerProxy(PyObject *impl)
	: m_impl(impl)
{
	if(import_turret__logger_bridge() != 0){ return; }
	Py_XINCREF(m_impl);
}

NativeLoggerProxy::~NativeLoggerProxy(){
	if(m_impl){ Py_XDECREF(m_impl); }
}

void NativeLoggerProxy::log(
	nvinfer1::ILogger::Severity severity,
	const char *message)
{
	cy_call_logger_log(m_impl, severity, message);
}

}
