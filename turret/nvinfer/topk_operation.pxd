# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass TopKOperation:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::TopKOperation":
    cdef TopKOperation kMAX
    cdef TopKOperation kMIN
