# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass ElementWiseOperation:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::ElementWiseOperation":
    cdef ElementWiseOperation kSUM
    cdef ElementWiseOperation kPROD
    cdef ElementWiseOperation kMAX
    cdef ElementWiseOperation kMIN
    cdef ElementWiseOperation kSUB
    cdef ElementWiseOperation kDIV
    cdef ElementWiseOperation kPOW
