# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
from libcpp cimport bool

from .cy_calibrator_proxy cimport CyCalibratorProxy


cdef public api int cy_call_get_batch_size(object self) except *:
    cdef CyCalibratorProxy proxy = self
    return proxy.get_batch_size()

cdef public api bool cy_call_get_batch(
        object self, void **bindings, const char **names,
        int nb_bindings) except *:
    cdef CyCalibratorProxy proxy = self
    return proxy.get_batch(bindings, names, nb_bindings)

cdef public api const void *cy_call_read_calibration_cache(
        object self, size_t& length) except *:
    cdef CyCalibratorProxy proxy = self
    return proxy.read_calibration_cache(length)

cdef public api void cy_call_write_calibration_cache(
        object self, const void *ptr, size_t length) except *:
    cdef CyCalibratorProxy proxy = self
    proxy.write_calibration_cache(ptr, length)
