from .nvinfer cimport Severity

cdef extern from "NvInfer.h" namespace "nvinfer1::ILogger::Severity":
    cdef Severity kINTERNAL_ERROR
    cdef Severity kERROR
    cdef Severity kWARNING
    cdef Severity kINFO
