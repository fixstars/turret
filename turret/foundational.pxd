cimport numpy as np

from .nvinfer cimport nvinfer


cdef class DataType:
    cdef nvinfer.DataType thisobj

cdef class DimensionType:
    cdef nvinfer.DimensionType thisobj

cdef class Dimension:
    cdef readonly int size
    cdef readonly DimensionType type

cdef class Dimensions:
    cdef tuple dims
    cdef nvinfer.Dims to_nvinfer_dims(self)
    @staticmethod
    cdef Dimensions from_nvinfer_dims(nvinfer.Dims)

cdef class Weights:
    cdef nvinfer.Weights thisobj
    cdef np.ndarray values
