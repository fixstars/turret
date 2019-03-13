from .nvinfer cimport DimensionType

cdef extern from "NvInfer.h" namespace "nvinfer1::DimensionType":
    cdef DimensionType kSPATIAL
    cdef DimensionType kCHANNEL
    cdef DimensionType kINDEX
    cdef DimensionType kSEQUENCE
