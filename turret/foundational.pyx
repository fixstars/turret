# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import copy

import numpy as np
cimport numpy as np

from libc.stdint cimport uintptr_t
from libc.stdint cimport uint16_t
from cpython cimport ref

from .nvinfer cimport nvinfer
from .nvinfer cimport data_type
from .nvinfer cimport dimension_type

from . import type_check


cdef class DataType:
    """The type of weights and tensors.

    Attributes:
        FLOAT(turret.DataType): FP32 format.
        HALF(turret.DataType): FP16 format.
        INT8(turret.DataType): INT8 format.
        INT32(turret.DataType): INT32 format.
    """

    FLOAT = DataType(<int>data_type.kFLOAT)
    """FP32 format."""

    HALF = DataType(<int>data_type.kHALF)
    """FP16 format."""

    INT8 = DataType(<int>data_type.kINT8)
    """INT8 format."""

    INT32 = DataType(<int>data_type.kINT32)
    """INT32 format."""

    _NAME_TABLE = {
        <int>data_type.kFLOAT: 'FLOAT',
        <int>data_type.kHALF: 'HALF',
        <int>data_type.kINT8: 'INT8',
        <int>data_type.kINT32: 'INT32',
    }

    def __init__(self, x):
        cdef int ivalue = <int>x
        self.thisobj = <nvinfer.DataType>ivalue

    def __repr__(self):
        cdef int ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj

    def __getstate__(self):
        return int(self)

    def __setstate__(self, state):
        cdef int ivalue = state
        self.thisobj = <nvinfer.DataType>ivalue

    def __reduce__(self):
        return _reconstruct_data_type, (self.__getstate__(),)

    property size:
        """Number of bytes consumed by an element of this type."""

        def __get__(self):
            cdef int thisobj = <int>self.thisobj
            if thisobj == <int>data_type.kFLOAT:
                return 4
            if thisobj == <int>data_type.kHALF:
                return 2
            if thisobj == <int>data_type.kINT8:
                return 1
            if thisobj == <int>data_type.kINT32:
                return 4
            assert False, "Invalid thisobj"

    property nptype:
        """NumPy data-type for this type."""

        def __get__(self):
            cdef int thisobj = <int>self.thisobj
            if thisobj == <int>data_type.kFLOAT:
                return np.float32
            if thisobj == <int>data_type.kHALF:
                return np.float16
            if thisobj == <int>data_type.kINT8:
                return np.int8
            if thisobj == <int>data_type.kINT32:
                return np.int32
            assert False, "Invalid thisobj"

def _reconstruct_data_type(state):
    return DataType(state)


cdef class DimensionType:
    """The type of data encoded across this dimension."""

    SPATIAL = DimensionType(<int>dimension_type.kSPATIAL)
    """Elements correspond to different spatial data."""

    CHANNEL = DimensionType(<int>dimension_type.kCHANNEL)
    """Elements correspond to different channels."""

    INDEX = DimensionType(<int>dimension_type.kINDEX)
    """Elements correspond to different batch index."""

    SEQUENCE = DimensionType(<int>dimension_type.kSEQUENCE)
    """Elements correspond to different sequence values."""

    _NAME_TABLE = {
        <int>dimension_type.kSPATIAL: 'SPATIAL',
        <int>dimension_type.kCHANNEL: 'CHANNEL',
        <int>dimension_type.kINDEX: 'INDEX',
        <int>dimension_type.kSEQUENCE: 'SEQUENCE',
    }

    def __init__(self, x):
        cdef int ivalue = <int>x
        self.thisobj = <nvinfer.DimensionType>ivalue

    def __repr__(self):
        cdef int ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj

    def __getstate__(self):
        return int(self)

    def __setstate__(self, state):
        cdef int ivalue = state
        self.thisobj = <nvinfer.DimensionType>ivalue

    def __reduce__(self):
        return _reconstruct_dimension_type, (self.__getstate__(),)

    def __richcmp__(DimensionType x, DimensionType y, op):
        cdef int a = <int>x.thisobj
        cdef int b = <int>y.thisobj
        if op == 2:
            return a == b
        if op == 3:
            return a != b
        raise NotImplementedError()

def _reconstruct_dimension_type(state):
    return DimensionType(state)


cdef class Dimension:
    """Structure to define a dimension.
    
    Args:
        size (int): The size of the dimension.
        type (turret.DimensionType): The type of the dimension.

    Attributes:
        size (int): The size of the dimension.
        type (turret.DimensionType): The type of the dimension.
    """

    def __init__(self, int size, DimensionType type):
        self.size = size
        self.type = type

    def __repr__(self):
        return "<{}: ({}, {})>".format(
            self.__class__.__name__, self.size, self.type)

    def __str__(self):
        return "({}, {})".format(self.size, self.type)

    def __getstate__(self):
        return (self.size, self.type)

    def __setstate__(self, state):
        self.size = state[0]
        self.type = state[1]

    def __reduce__(self):
        return _reconstruct_dimension, (self.__getstate__(),)

    def __richcmp__(Dimension a, Dimension b, op):
        if op == 2:
            return a.size == b.size and a.type == b.type
        if op == 3:
            return a.dims != b.dims or a.type != b.type
        raise NotImplementedError()

def _reconstruct_dimension(state):
    return Dimension(state[0], state[1])


cdef class Dimensions:
    """Structure to define the dimensions of a tensor."""

    def __init__(self, dims):
        def is_dimension_tuple(d):
            if type(d) is not tuple:
                return False
            if len(d) != 2:
                return False
            if type(d[0]) is not int:
                return False
            if type(d[1]) is not DimensionType:
                return False
            return True

        def to_dimension(d):
            if type(d) is Dimension:
                return d
            elif is_dimension_tuple(d):
                return Dimension(d[0], d[1])
            else:
                raise TypeError("dimensions not understood")

        self.dims = tuple([to_dimension(d) for d in dims])

    def __iter__(self):
        return iter(self.dims)

    def __len__(self):
        return len(self.dims)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.dims[key]
        else:
            return Dimensions(self.dims[key])

    def __repr__(self):
        return repr(self.dims)

    def __getstate__(self):
        return self.dims

    def __setstate__(self, state):
        self.dims = state

    def __reduce__(self):
        return _reconstruct_dimensions, (self.__getstate__(),)

    def __richcmp__(Dimensions a, Dimensions b, op):
        if op == 2:
            return a.dims == b.dims
        if op == 3:
            return a.dims != b.dims
        raise NotImplementedError()

    @property
    def shape(self):
        """Tuple of dimensions."""
        return tuple([d.size for d in self.dims])

    @property
    def size(self):
        """Number of elements."""
        cdef int prod = 1
        for d in self.dims:
            prod *= d.size
        return prod

    @staticmethod
    def HW(int h, int w):
        """Make descriptor for 2D spatial data."""
        return Dimensions((
            (h, DimensionType.SPATIAL),
            (w, DimensionType.SPATIAL)))

    @staticmethod
    def CHW(int c, int h, int w):
        """Make descriptor for 2D spatial data with channels."""
        return Dimensions((
            (c, DimensionType.CHANNEL),
            (h, DimensionType.SPATIAL),
            (w, DimensionType.SPATIAL)))

    @staticmethod
    def NCHW(int n, int c, int h, int w):
        """Make descriptor for 2D spatial data with channels and indices."""
        return Dimensions((
            (n, DimensionType.INDEX),
            (c, DimensionType.CHANNEL),
            (h, DimensionType.SPATIAL),
            (w, DimensionType.SPATIAL)))

    cdef nvinfer.Dims to_nvinfer_dims(self):
        """Construct `nvinfer1::Dims` from `self`."""
        cdef:
            nvinfer.Dims dest
            Dimension dim
            int i
        dest.nbDims = len(self.dims)
        for i, dim in enumerate(self.dims):
            dest.d[i] = dim.size
            dest.type[i] = dim.type.thisobj
        return dest

    @staticmethod
    cdef Dimensions from_nvinfer_dims(nvinfer.Dims src):
        """Construct descriptor from `nvinfer1::Dims`."""
        return Dimensions([
            (src.d[i], DimensionType(<int>src.type[i]))
            for i in range(src.nbDims)
        ])

def _reconstruct_dimensions(state):
    return Dimensions(state)


cdef class Weights:
    """An array of weights used as a layer parameter.

    Args:
        values (numpy.ndarray): The weight values.
        dtype (data-type): The type of the weights.
            `values.dtype` will be used as it when `dtype` is None.
    """

    def __init__(self, np.ndarray values=None, dtype=None):
        cdef DataType datatype = type_check.get_datatype(
                dtype if dtype else values.dtype)
        if values is None:
            values = np.empty((0,), dtype=datatype.nptype)
        elif values.dtype != datatype.nptype:
            values = values.astype(datatype.nptype)
        values = values.flatten()
        self.values = values
        self.thisobj.type = datatype.thisobj
        self.thisobj.values = values.data if len(values) > 0 else NULL
        self.thisobj.count = len(values)

    def __getstate__(self):
        return self.values

    def __setstate__(self, state):
        self.__init__(state)

    def __reduce__(self):
        return _reconstruct_weights, (self.__getstate__(),)

def _reconstruct_weights(state):
    return Weights(state)
