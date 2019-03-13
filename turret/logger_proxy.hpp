#ifndef TURRET_LOGGER_PROXY_H
#define TURRET_LOGGER_PROXY_H

#include "NvInfer.h"
#include "Python.h"

namespace turret {

class NativeLoggerProxy : public nvinfer1::ILogger {

private:
	PyObject *m_impl;

public:
	NativeLoggerProxy();
	NativeLoggerProxy(PyObject *impl);
	virtual ~NativeLoggerProxy();

	virtual void log(
		nvinfer1::ILogger::Severity severity,
		const char *message) override;

};

}

#endif
