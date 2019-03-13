#ifndef TURRET_INT8_NATIVE_CALIBRATOR_PROXY_HPP
#define TURRET_INT8_NATIVE_CALIBRATOR_PROXY_HPP

#include "NvInfer.h"
#include "Python.h"

namespace turret {

class NativeCalibratorProxy
	: public nvinfer1::IInt8EntropyCalibrator
{

private:
	PyObject *m_impl;

public:
	NativeCalibratorProxy();
	NativeCalibratorProxy(PyObject *impl);
	virtual ~NativeCalibratorProxy();

	virtual int getBatchSize() const override;

	virtual bool getBatch(
		void *bindings[], const char *names[], int nbBindings) override;

	virtual const void *readCalibrationCache(std::size_t& length) override;

	virtual void writeCalibrationCache(
		const void *ptr, std::size_t length) override;

};

}

#endif
