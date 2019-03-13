from cpython cimport ref

from .nvinfer cimport nvinfer


cdef extern from "logger_proxy.hpp" namespace "turret":
    cdef cppclass NativeLoggerProxy(nvinfer.ILogger):
        NativeLoggerProxy() except +
        NativeLoggerProxy(ref.PyObject *) except +
        void log(nvinfer.Severity, const char *) except +

cdef class Severity:
    cdef nvinfer.Severity thisobj

cdef class LoggerProxy:
    cdef NativeLoggerProxy *proxy
