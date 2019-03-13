# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass RNNGateType:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::RNNGateType":
    cdef RNNGateType kINPUT
    cdef RNNGateType kOUTPUT
    cdef RNNGateType kFORGET
    cdef RNNGateType kUPDATE
    cdef RNNGateType kRESET
    cdef RNNGateType kCELL
    cdef RNNGateType kHIDDEN
