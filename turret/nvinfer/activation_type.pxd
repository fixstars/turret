from .nvinfer cimport ActivationType

cdef extern from "NvInfer.h" namespace "nvinfer1::ActivationType":
    cdef ActivationType kRELU
    cdef ActivationType kSIGMOID
    cdef ActivationType kTANH
