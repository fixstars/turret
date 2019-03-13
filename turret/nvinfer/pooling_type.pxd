from .nvinfer cimport PoolingType

cdef extern from "NvInfer.h" namespace "nvinfer1::PoolingType":
    cdef PoolingType kMAX
    cdef PoolingType kAVERAGE
    cdef PoolingType kMAX_AVERAGE_BLEND
