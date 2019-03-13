# distutils: language = c++

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass UnaryOperation:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1::UnaryOperation":
    cdef UnaryOperation kEXP
    cdef UnaryOperation kLOG
    cdef UnaryOperation kSQRT
    cdef UnaryOperation kRECIP
    cdef UnaryOperation kABS
    cdef UnaryOperation kNEG
