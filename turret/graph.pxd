from .nvinfer cimport nvinfer

from .foundational cimport DataType
from .engine cimport InferenceEngineBuilder


cdef class Tensor:
    cdef nvinfer.ITensor *thisptr
    cdef readonly NetworkDefinition network

    @staticmethod
    cdef Tensor create(nvinfer.ITensor *rawptr, NetworkDefinition network)

cdef class NetworkDefinition:
    cdef nvinfer.INetworkDefinition *thisptr
    cdef InferenceEngineBuilder builder
    cdef readonly DataType dtype
    cdef list plugins
    cdef list resources
