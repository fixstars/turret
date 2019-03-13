# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import numpy as np
cimport numpy as np

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from cpython cimport ref

import pycuda.autoinit
import pycuda.driver as cuda

from .nvinfer cimport nvinfer
from .nvinfer cimport data_type
from .nvinfer cimport dimension_type
from .nvinfer cimport calibration_algo_type

from .logger cimport LoggerProxy

from .foundational cimport DataType
from .foundational cimport DimensionType
from .foundational cimport Dimensions
from .foundational cimport Weights

from .engine cimport Binding
from .engine cimport InferenceEngineBuilder


cdef class Tensor:
    """A tensor in a network definition."""

    def __cinit__(self):
        self.thisptr = NULL
        self.network = None

    @staticmethod
    cdef Tensor create(nvinfer.ITensor *rawptr, NetworkDefinition network):
        t = Tensor()
        t.thisptr = rawptr
        t.network = network
        return t

    property dimensions:
        """Get the dimensions of the tensor."""

        def __get__(self):
            return Dimensions.from_nvinfer_dims(self.thisptr.getDimensions())

    property type:
        """Get the data type of the tensor."""

        def __get__(self):
            return DataType(<int>self.thisptr.getType())

    property is_network_input:
        """Returns True if this tensor is a network input."""

        def __get__(self):
            return self.thisptr.isNetworkInput()

    property is_network_output:
        """Returns True if this tensor is a network output."""

        def __get__(self):
            return self.thisptr.isNetworkOutput()

    property broadcast_across_batch:
        """Returns True if this tensor is broadcast across the batch."""

        def __get__(self):
            return self.thisptr.getBroadcastAcrossBatch()


cdef class NetworkDefinition:
    """A network definition."""

    def __cinit__(self, InferenceEngineBuilder builder, DataType dtype):
        self.thisptr = builder.thisptr.createNetwork()
        self.builder = builder
        self.dtype = dtype
        self.plugins = []
        self.resources = []

    def __dealloc__(self):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL

    def add_input(self, str name, DataType type, Dimensions dims,
                  bool broadcast_across_batch=False):
        """Add an input tensor to the network.

        Args:
            name (str): The name of the input tensor.
            type (turret.DataType): The data type of the input tensor.
            dims (turret.Dimensions): The dimensions of the input tensor.
            broadcast_across_batch(bool, optional): Whether to enable
                broadcast of tensor accross the batch. Default to False.

        Returns:
            turret.Tensor: The tensor that is added by this operation.
        """
        t = Tensor()
        t.network = self
        t.thisptr = self.thisptr.addInput(
            name.encode("utf-8"), type.thisobj, dims.to_nvinfer_dims())
        if t.thisptr == NULL:
            raise RuntimeError("failed to create input tensor")
        t.thisptr.setBroadcastAcrossBatch(broadcast_across_batch)
        return t

    def add_constant(self, np.ndarray weights, Dimensions dims=None):
        """Add an constant input to the networks.

        Args:
            weights (np.ndarray): The input values.
            dims (Dimensions, optional): The dimensions of this input. Default
                to None.

        Returns:
            turret.Tensor: The tensor that is added by this operation.
        """
        w_shape = tuple([weights.shape[i] for i in range(weights.ndim)])
        if dims is None:
            dims = Dimensions(tuple(
                [(x, DimensionType.SPATIAL) for x in w_shape]))
        if w_shape != dims.shape:
            raise ValueError("shape mismatch")
        w_weights = Weights(weights, self.weight_type)
        self.add_resource(w_weights)
        layer = self.thisptr.addConstant(
            dims.to_nvinfer_dims(), w_weights.thisobj)
        if layer == NULL:
            raise RuntimeError("failed to create constant layer")
        return Tensor.create(layer.getOutput(0), self)

    def mark_output(self, str name, Tensor tensor):
        """Mark an tensor as an output of the network.

        Args:
            name (str): The name of the output tensor.
            tensor (turret.Tensor): The tensor that will be marked as
                an output.
        """
        if tensor.network is not self:
            raise ValueError("tensor is not in this network")
        tensor.thisptr.setName(name.encode("utf-8"))
        self.thisptr.markOutput(tensor.thisptr[0])

    property input_bindings:
        """Get binding informations of input tensors."""

        def __get__(self):
            result = {}
            for i in range(self.thisptr.getNbInputs()):
                tensor = self.thisptr.getInput(i)
                binding = Binding()
                binding.name = tensor.getName().decode("utf-8")
                binding.index = i
                binding.is_input = True
                binding.type = DataType(<int>tensor.getType())
                binding.dimensions = \
                        Dimensions.from_nvinfer_dims(tensor.getDimensions())
                result[binding.name] = binding
            return result

    property weight_type:
        """Get data type for weights."""

        def __get__(self):
            if self.dtype == DataType.HALF:
                return DataType.HALF
            else:
                return DataType.FLOAT

    def add_plugin(self, object plugin):
        """Add an plugin layer to the networks.

        Args:
            plugin (turret.Plugin): The layer plugin.
        """
        self.plugins.append(plugin)

    def add_resource(self, object resource):
        """Add an resource(ex. weights, bias, etc.) to the networks.

        Args:
            resource (turret.Tensor): an resource.
        """
        self.resources.append(resource)
