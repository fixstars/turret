# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass ReduceOperation:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::ReduceOperation":
    cdef ReduceOperation kSUM
    cdef ReduceOperation kPROD
    cdef ReduceOperation kMAX
    cdef ReduceOperation kMIN
    cdef ReduceOperation kAVG
