# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass RNNOperation:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::RNNOperation":
    cdef RNNOperation kRELU
    cdef RNNOperation kTANH
    cdef RNNOperation kLSTM
    cdef RNNOperation kGRU
