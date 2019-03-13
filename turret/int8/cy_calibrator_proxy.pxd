from libcpp cimport bool
from cpython cimport ref

from ..nvinfer cimport nvinfer

cdef extern from "native_calibrator_proxy.hpp" namespace "turret":
    cdef cppclass NativeCalibratorProxy(nvinfer.IInt8EntropyCalibrator):
        NativeCalibratorProxy()
        NativeCalibratorProxy(ref.PyObject *)


cdef class CyCalibratorProxy:
    cdef object calibrator

    cdef int get_batch_size(self) except *
    cdef bool get_batch(self, void **bindings, const char **names,
                        int nb_bindings) except *
    cdef const void *read_calibration_cache(self, size_t& length) except *
    cdef void write_calibration_cache(self, const void *ptr,
                                      size_t length) except *


cdef class CalibratorProxy:
    cdef NativeCalibratorProxy *native_proxy
