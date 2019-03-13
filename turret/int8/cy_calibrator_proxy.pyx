# distutils: language = c++
# distutils: sources = ["turret/int8/native_calibrator_proxy.cpp"]
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
from libcpp cimport bool
from libc.stdint cimport uintptr_t

cdef class CyCalibratorProxy:
    """The class for internal processing. Users need not to use."""

    def __cinit__(self, calibrator):
        self.calibrator = calibrator

    cdef int get_batch_size(self) except *:
        return self.calibrator.get_batch_size()

    cdef bool get_batch(self, void **bindings, const char **names,
                        int nb_bindings) except *:
        py_names = [names[i].decode("utf-8") for i in range(nb_bindings)]
        py_allocs = self.calibrator.get_batch(py_names)
        if py_allocs is None:
            return False
        for i in range(nb_bindings):
            ptr = <uintptr_t>int(py_allocs[i])
            bindings[i] = <void *>ptr
        return True

    cdef const void *read_calibration_cache(self, size_t& length) except *:
        return NULL

    cdef void write_calibration_cache(self, const void *ptr,
                                      size_t length) except *:
        pass


cdef class CalibratorProxy:
    """The class for internal processing. Users need not to use."""

    def __cinit__(self, calibrator):
        cy_proxy = CyCalibratorProxy(calibrator)
        self.native_proxy = \
                new NativeCalibratorProxy(<ref.PyObject *>cy_proxy)

    def __dealloc__(self):
        if self.native_proxy:
            del self.native_proxy
