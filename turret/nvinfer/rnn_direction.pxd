# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass RNNDirection:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::RNNDirection":
    cdef RNNDirection kUNIDIRECTION
    cdef RNNDirection kBIDIRECTION
