from .nvinfer cimport DataType

cdef extern from "NvInfer.h" namespace "nvinfer1::DataType":
    cdef DataType kFLOAT
    cdef DataType kHALF
    cdef DataType kINT8
    cdef DataType kINT32
