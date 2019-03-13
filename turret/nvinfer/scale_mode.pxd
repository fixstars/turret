# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass ScaleMode:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::ScaleMode":
    cdef ScaleMode kUNIFORM
    cdef ScaleMode kCHANNEL
    cdef ScaleMode kELEMENTWISE
