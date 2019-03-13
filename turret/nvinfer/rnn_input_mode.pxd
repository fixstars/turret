# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass RNNInputMode:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::RNNInputMode":
    cdef RNNInputMode kLINEAR
    cdef RNNInputMode kSKIP
