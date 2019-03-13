from libcpp cimport bool

from .nvinfer cimport nvinfer

from .logger cimport LoggerProxy
from .foundational cimport DataType
from .foundational cimport Dimensions


cdef class InferenceEngineBuilder:
    cdef nvinfer.IBuilder *thisptr
    cdef LoggerProxy logger
    cdef object int8_calibrator

cdef class Binding:
    cdef readonly unicode name
    cdef readonly int index
    cdef readonly bool is_input
    cdef readonly DataType type
    cdef readonly Dimensions dimensions

cdef class BindingSet:
    cdef list from_index
    cdef dict from_name

    @staticmethod
    cdef BindingSet create(nvinfer.ICudaEngine *)

cdef class InferenceEngine:
    cdef nvinfer.ICudaEngine *thisptr
    cdef readonly BindingSet bindings
    cdef list plugins

    @staticmethod
    cdef InferenceEngine create(nvinfer.ICudaEngine *, list)

cdef class InferenceRuntime:
    cdef nvinfer.IRuntime *thisptr
    cdef LoggerProxy logger

cdef class ExecutionContext:
    cdef nvinfer.IExecutionContext *thisptr
    cdef InferenceEngine engine
    cdef object default_stream
