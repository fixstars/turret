# distutils: language = c++
# distutils: sources = []
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import numpy as np
cimport numpy as np

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.vector cimport vector

from ..nvinfer cimport nvinfer
from ..nvinfer cimport activation_type
from ..nvinfer cimport pooling_type
from ..nvinfer cimport scale_mode
from ..nvinfer cimport element_wise_operation
from ..nvinfer cimport unary_operation
from ..nvinfer cimport reduce_operation
from ..nvinfer cimport rnn_operation
from ..nvinfer cimport rnn_direction
from ..nvinfer cimport rnn_input_mode
from ..nvinfer cimport rnn_gate_type
from ..nvinfer cimport topk_operation

from ..graph cimport Tensor
from ..foundational cimport Weights
from ..foundational cimport Dimensions
from ..plugin.cy_plugin_proxy cimport PluginProxy


#--------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------
cdef nvinfer.DimsHW _make_dims_hw(x):
    if hasattr(x, "__getitem__"):
        return nvinfer.DimsHW(x[0], x[1])
    return nvinfer.DimsHW(x, x)

cdef uint32_t _convert_axes(a):
    if isinstance(a, int):
        return <uint32_t>(1 << a)
    else:
        value = 0
        for x in a:
            value |= (1 << x)
        return <uint32_t>value


#--------------------------------------------------------------------
# 2D Convolution
#--------------------------------------------------------------------
def convolution_2d(Tensor input, np.ndarray filters,
                   np.ndarray biases=None, stride=1, padding=0):
    """Layer for 2D Convolution.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        filters(np.ndarray): Filters for convolution.
        biases(np.ndarray, optional): Biases for convolution. Defaults to None.
        stride(int, optional): Stride for convolution. Defaults to 1.
        padding(int, optional): Padding for convolution. Defaults to 0.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    if filters.ndim != 4:
        raise ValueError()
    if biases is not None and biases.ndim != 1:
        raise ValueError()
    k = filters.shape[0]
    c = filters.shape[1]
    h = filters.shape[2]
    w = filters.shape[3]
    if c != input.dimensions[0].size:
        raise ValueError()
    if biases is not None and biases.shape[0] != k:
        raise ValueError()

    w_filters = Weights(filters, input.network.weight_type)
    w_biases = Weights(biases, input.network.weight_type)
    input.network.add_resource((w_filters, w_biases))

    raw_network = input.network.thisptr
    layer = raw_network.addConvolution(
        input.thisptr[0], k, _make_dims_hw((h, w)),
        w_filters.thisobj, w_biases.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create convolution layer")
    layer.setStride(_make_dims_hw(stride))
    layer.setPadding(_make_dims_hw(padding))
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Fully connected
#--------------------------------------------------------------------
def fully_connected(Tensor input, np.ndarray weights,
                    np.ndarray biases=None):
    """Layer for fully connected.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        weights(np.ndarray): Filters for fully connected layer.
        biases(np.ndarray, optional): Biases for fully connected layer.
            Defaults to None.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    if len(input.dimensions) < 3:
        raise ValueError()
    if weights.ndim != 2:
        raise ValueError()
    if biases is not None and biases.ndim != 1:
        raise ValueError()

    k, c = weights.shape[0], weights.shape[1]
    if input.dimensions[-3:].size != c:
        raise ValueError()
    if biases is not None and biases.shape[0] != k:
        raise ValueError()

    w_weights = Weights(weights, input.network.weight_type)
    w_biases = Weights(biases, input.network.weight_type)
    input.network.add_resource((w_weights, w_biases))

    raw_network = input.network.thisptr
    layer = raw_network.addFullyConnected(
        input.thisptr[0], k, w_weights.thisobj, w_biases.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create fully connected layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Activation
#--------------------------------------------------------------------
cdef class ActivationType:
    """
    enumerates the types of activation to perform in an activation
    layer.

    Attributes:
        RELU(turret.ActivationType): Rectified linear activation.
        SIGMOID(turret.ActivationType): Sigmoid activation.
        TANH(turret.ActivationType): TanH activation.
    """

    cdef nvinfer.ActivationType thisobj

    RELU    = ActivationType(<int>activation_type.kRELU)
    """Rectified linear activation."""

    SIGMOID = ActivationType(<int>activation_type.kSIGMOID)
    """Sigmoid activation."""

    TANH    = ActivationType(<int>activation_type.kTANH)
    """TanH activation."""

    _NAME_TABLE = {
        <int>activation_type.kRELU:    "RELU",
        <int>activation_type.kSIGMOID: "SIGMOID",
        <int>activation_type.kTANH:    "TANH",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.ActivationType>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def builtin_activation(Tensor input, ActivationType type):
    """Layer for activation.

    Args:
        input(turret.Tensor): Tensor which will be processed by activation.
        type(turret.ActivationType): Type of Activation.

    Returns:
        tensor(turret.Tensor): Tensor processed by activation.
    """
    assert input.network
    assert input.network.thisptr
    raw_network = input.network.thisptr
    layer = raw_network.addActivation(input.thisptr[0], type.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create activation layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Pooling
#--------------------------------------------------------------------
cdef class PoolingType:
    """Type of pooling to perform in a pooling layer.

    Attributes:
        MAX(turret.PoolingType): Maximum over elements.
        AVERAGE(turret.PoolingType): Average over elements. If the tensor is
            padded, the count includes the padding.
        MAX_AVERAGE_BLEND(turret.PoolingType): Blending between the max
            pooling and average pooling:
            (1-blendFactor)*maxPool + blendFactor*avgPool
    """

    cdef nvinfer.PoolingType thisobj

    MAX = PoolingType(<int>pooling_type.kMAX)
    """Maximum over elements."""

    AVERAGE = PoolingType(<int>pooling_type.kAVERAGE)
    """
    Average over elements. If the tensor is padded, the count
    includes the padding.
    """

    MAX_AVERAGE_BLEND = PoolingType(<int>pooling_type.kMAX_AVERAGE_BLEND)
    """
    Blending between the max pooling and average pooling:
    (1-blendFactor)*maxPool + blendFactor*avgPool
    """

    _NAME_TABLE = {
        <int>pooling_type.kMAX:               "MAX",
        <int>pooling_type.kAVERAGE:           "AVERAGE",
        <int>pooling_type.kMAX_AVERAGE_BLEND: "MAX_AVERAGE_BLEND",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.PoolingType>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def builtin_pooling(Tensor input, PoolingType type,
                    window_size, stride, padding, blend=None):
    """Layer for pooling.

    Args:
        input(turret.Tensor): Tensor which will be processed by pooling.
        type(turret.PoolingType): Type of pooling.
        window_size(tupple): Window size for pooling.
        stride(int): Stride for pooling.
        padding: Padding for pooling.
        blend(float, optional): Blending factor for the max_average_blend mode:
            max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
            blendFactor is a user value in [0,1] with the default value of 0.0
            This value only applies for the MAX_AVERAGE_BLEND mode. Default
            to None.

    Returns:
        tensor(turret.Tensor): Tensor processed by pooling.
    """
    assert input.network
    assert input.network.thisptr
    raw_network = input.network.thisptr
    layer = raw_network.addPooling(input.thisptr[0], type.thisobj,
                                   _make_dims_hw(window_size))
    if layer == NULL:
        raise RuntimeError("failed to create pooling layer")
    layer.setStride(_make_dims_hw(stride))
    layer.setPadding(_make_dims_hw(padding))
    if blend is not None:
        layer.setBlendFactor(blend)
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Local repsonse normalization
#--------------------------------------------------------------------
def local_response_normalization(Tensor input, n=5, k=2.0,
                                 alpha=1e-4, beta=0.75):
    """Layer for local responce normalization(LRN).

    Args:
        input(turret.Tensor): Tensor which will be processed by LRN layer.
        n(int, optional): Window size. Default to 5.
        k(float, optional): LRN K value. Default to 2.0.
        alpha(float, optional): LRN alpha value. Default to 1e-4.
        beta(float, optional): LRN beta value. Default to 0.75.

    Returns:
        tensor(turret.Tensor): Tensor processed by LRN layer.
    """
    assert input.network
    assert input.network.thisptr
    raw_network = input.network.thisptr
    layer = raw_network.addLRN(input.thisptr[0], n, alpha, beta, k)
    if layer == NULL:
        raise RuntimeError("failed to create LRN layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Scale
#--------------------------------------------------------------------
cdef class ScaleMode:
    """Controls how scale is applied in a Scale layer.

    Attributes:
        UNIFORM(turret.ScaleMode): Identical coefficients across all elements
            of the tensor.
        CHANNEL(turret.ScaleMode): Per-channel coefficients.
        ELEMENTWISE(turret.ScaleMode): Elementwise coefficients.
    """

    cdef nvinfer.ScaleMode thisobj

    UNIFORM = ScaleMode(<int>scale_mode.kUNIFORM)
    """Identical coefficients across all elements of the tensor."""

    CHANNEL = ScaleMode(<int>scale_mode.kCHANNEL)
    """Per-channel coefficients."""

    ELEMENTWISE = ScaleMode(<int>scale_mode.kELEMENTWISE)
    """Elementwise coefficients."""

    _NAME_TABLE = {
        <int>scale_mode.kUNIFORM:     "UNIFORM",
        <int>scale_mode.kCHANNEL:     "CHANNEL",
        <int>scale_mode.kELEMENTWISE: "ELEMENTWISE",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.ScaleMode>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def scale(Tensor input, ScaleMode mode, np.ndarray shift=None,
          np.ndarray scale=None, np.ndarray power=None):
    """Layer for scale.

    Args:
        input(turret.Tensor): Tensor which will be processed by scale layer.
        mode(turret.ScaleMode): The scaling mode.
        shift(np.ndarray, optional): The shift value. Default to None.
        scale(np.ndarray, optional): The scale value. Default to None.
        power(np.ndarray, optional): The power value. Default to None.

    Returns:
        tensor(turret.Tensor): Tensor processed by scale layer.
    """
    assert input.network
    assert input.network.thisptr
    input_shape = input.dimensions.shape

    if mode == ScaleMode.UNIFORM:
        def validate(object a):
            if a is None: return True
            return a.shape == (1,)
    elif mode == ScaleMode.CHANNEL:
        def validate(object a):
            if a is None: return True
            return a.shape == (input.dimensions.shape[0],)
    elif mode == ScaleMode.ELEMENTWISE:
        def validate(object a):
            if a is None: return True
            return a.shape == input.dimensions.shape
    if not validate(shift):
        raise ValueError("shape mismatch for shift")
    if not validate(scale):
        raise ValueError("shape mismatch for scale")
    if not validate(power):
        raise ValueError("shape mismatch for power")

    w_shift = Weights(shift, input.network.weight_type)
    w_scale = Weights(scale, input.network.weight_type)
    w_power = Weights(power, input.network.weight_type)
    input.network.add_resource((w_shift, w_scale, w_power))

    raw_network = input.network.thisptr
    layer = raw_network.addScale(input.thisptr[0], mode.thisobj,
                                 w_shift.thisobj, w_scale.thisobj,
                                 w_power.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create scale layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Softmax
#--------------------------------------------------------------------
def softmax(Tensor input, axes=0):
    """Layer for softmax.

    Args:
        input(turret.Tensor): Tensor which will be processed by softmax layer.
        axes(int, optional): Axes to apply layer. Default to 0.

    Returns:
        tensor(turret.Tensor): Tensor processed by softmax layer.
    """
    assert input.network
    assert input.network.thisptr

    raw_network = input.network.thisptr
    layer = raw_network.addSoftMax(input.thisptr[0])
    if layer == NULL:
        raise RuntimeError("failed to create softmax layer")
    layer.setAxes(_convert_axes(axes))
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Concatenation
#--------------------------------------------------------------------
def concat(object tensors, int axis=0):
    """Layer for concatenation.

    Args:
        tensors(object): Tensors which will be processed by concat layer.
        axes(int, optional): Axes to apply layer. Default to 0.

    Returns:
        tensor(turret.Tensor): Tensor processed by concat layer.
    """
    assert len(tensors) > 0

    cdef:
        Tensor input0 = tensors[0]
        vector[nvinfer.ITensor*] v_tensors
        Tensor t
    for t in tensors:
        v_tensors.push_back(t.thisptr)
    raw_network = input0.network.thisptr
    layer = raw_network.addConcatenation(v_tensors.data(), v_tensors.size())
    if layer == NULL:
        raise RuntimeError("failed to create concatenation layer")
    layer.setAxis(axis)
    return Tensor.create(layer.getOutput(0), input0.network)


#--------------------------------------------------------------------
# Deconvolution
#--------------------------------------------------------------------
def deconvolution_2d(Tensor input, np.ndarray filters,
                     np.ndarray biases=None, stride=1, padding=0):
    """Layer for 2D Deconvolution.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        filters(np.ndarray): Filters for deconvolution.
        biases(np.ndarray, optional): Biases for deconvolution. Default to
            None.
        stride(int, optional): Stride for deconvolution. Default to 1.
        padding(int, optional): Padding for deconvolution. Default to 0.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    if filters.ndim != 4:
        raise ValueError()
    if biases is not None and biases.ndim != 1:
        raise ValueError()
    c = filters.shape[0]
    k = filters.shape[1]
    h = filters.shape[2]
    w = filters.shape[3]
    if c != input.dimensions[0].size:
        raise ValueError()
    if biases is not None and biases.shape[0] != k:
        raise ValueError()

    w_filters = Weights(filters, input.network.weight_type)
    w_biases = Weights(biases, input.network.weight_type)
    input.network.add_resource((w_filters, w_biases))

    raw_network = input.network.thisptr
    layer = raw_network.addDeconvolution(
        input.thisptr[0], k, _make_dims_hw((h, w)),
        w_filters.thisobj, w_biases.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create deconvolution layer")
    layer.setStride(_make_dims_hw(stride))
    layer.setPadding(_make_dims_hw(padding))
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Elementwise
#--------------------------------------------------------------------
cdef class ElementWiseOperation:
    """
    Enumerates the binary operations that may be performed by an
    elementwise layer.

    Attributes:
        SUM(turret.ElementWiseOperation): Sum of the two elements.
        PROD(turret.ElementWiseOperation): Product of the two elements.
        MAX(turret.ElementWiseOperation): Maximum of the two elements.
        MIN(turret.ElementWiseOperation): Minimum of the two elements.
        SUB(turret.ElementWiseOperation): Substract the second element from
            the first.
        DIV(turret.ElementWiseOperation): Divide the first element by the
            second.
        POW(turret.ElementWiseOperation): The first element to the power of
            the second element.
    """

    cdef nvinfer.ElementWiseOperation thisobj

    SUM = ElementWiseOperation(<int>element_wise_operation.kSUM)
    """Sum of the two elements."""

    PROD = ElementWiseOperation(<int>element_wise_operation.kPROD)
    """Product of the two elements."""

    MAX = ElementWiseOperation(<int>element_wise_operation.kMAX)
    """Maximum of the two elements."""

    MIN = ElementWiseOperation(<int>element_wise_operation.kMIN)
    """Minimum of the two elements."""

    SUB = ElementWiseOperation(<int>element_wise_operation.kSUB)
    """Substract the second element from the first."""

    DIV = ElementWiseOperation(<int>element_wise_operation.kDIV)
    """Divide the first element by the second."""

    POW = ElementWiseOperation(<int>element_wise_operation.kPOW)
    """The first element to the power of the second element."""

    _NAME_TABLE = {
        <int>element_wise_operation.kSUM:  "SUM",
        <int>element_wise_operation.kPROD: "PROD",
        <int>element_wise_operation.kMAX:  "MAX",
        <int>element_wise_operation.kMIN:  "MIN",
        <int>element_wise_operation.kSUB:  "SUB",
        <int>element_wise_operation.kDIV:  "DIV",
        <int>element_wise_operation.kPOW:  "POW",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.ElementWiseOperation>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def elementwise(Tensor input0, Tensor input1, ElementWiseOperation op):
    """Layer for elementwise.

    Args:
        input0(turret.Tensor): First tensor which will be processed by layer.
        input1(turret.Tensor): Second tensor which will be processed by layer.
        op(turret.ElementWiseOperation): Elementwise operator.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input0.network
    assert input1.network
    if input0.network is not input1.network:
        raise ValueError("input0 and input1 are not in same network")

    raw_network = input0.network.thisptr
    layer = raw_network.addElementWise(input0.thisptr[0], input1.thisptr[0],
                                       op.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create elementwise layer")
    return Tensor.create(layer.getOutput(0), input0.network)


#--------------------------------------------------------------------
# RNN
#--------------------------------------------------------------------
cdef class RNNOperation:
    """Enumerates the RNN operations.

    Attributes:
        RELU(turret.RNNOperation): Single gate RNN with ReLU activation
            function.
        TANH(turret.RNNOperation): Single gate RNN with TANH activation
            function.
        LSTM(turret.RNNOperation): Four-gate LSTM network without peephole
            connections.
        GRU(turret.RNNOperation): Three-gate network consisting of Gated
            Recurrent Units.
    """

    cdef nvinfer.RNNOperation thisobj

    RELU = RNNOperation(<int>rnn_operation.kRELU)
    """Single gate RNN with ReLU activation function."""

    TANH = RNNOperation(<int>rnn_operation.kTANH)
    """Single gate RNN with TANH activation function."""

    LSTM = RNNOperation(<int>rnn_operation.kLSTM)
    """Four-gate LSTM network without peephole connections."""

    GRU = RNNOperation(<int>rnn_operation.kGRU)
    """Three-gate network consisting of Gated Recurrent Units."""

    _NAME_TABLE = {
        <int>rnn_operation.kRELU: "RELU",
        <int>rnn_operation.kTANH: "TANH",
        <int>rnn_operation.kLSTM: "LSTM",
        <int>rnn_operation.kGRU: "GRU",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.RNNOperation>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


cdef class RNNDirection:
    """Enumerates the RNN directions.

    Attributes:
        UNIDIRECTION(turret.RNNDirection): Network iterations from first input
            to last input.
        BIDIRECTION(turret.RNNDirection): Network iterates from first to last
            and vice versa and outputs concatenated.
    """

    cdef nvinfer.RNNDirection thisobj

    UNIDIRECTION = RNNDirection(<int>rnn_direction.kUNIDIRECTION)
    """Network iterations from first input to last input."""

    BIDIRECTION = RNNDirection(<int>rnn_direction.kBIDIRECTION)
    """Network iterates from first to last and vice versa and outputs concatenated."""

    _NAME_TABLE = {
        <int>rnn_direction.kUNIDIRECTION: "UNIDIRECTION",
        <int>rnn_direction.kBIDIRECTION: "BIDIRECTION",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.RNNDirection>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


cdef class RNNInputMode:
    """Enumerates the RNN input modes.

    Attributes:
        LINEAR(turret.RNNInputMode): Perform the normal matrix multiplication
            in the first recurrent layer.
        SKIP(turret.RNNInputMode): No operation is performed on the first
            recurrent layer.
    """

    cdef nvinfer.RNNInputMode thisobj

    LINEAR = RNNInputMode(<int>rnn_input_mode.kLINEAR)
    """Perform the normal matrix multiplication in the first recurrent layer."""

    SKIP = RNNInputMode(<int>rnn_input_mode.kSKIP)
    """No operation is performed on the first recurrent layer."""

    _NAME_TABLE = {
        <int>rnn_input_mode.kLINEAR: "LINEAR",
        <int>rnn_input_mode.kSKIP: "SKIP",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.RNNInputMode>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def rnn(Tensor input, int num_layers, int hidden_size,
        int max_sequence_length, RNNOperation op,
        RNNInputMode mode, RNNDirection direction,
        Weights weights, Weights bias,
        Tensor hidden_state, Tensor cell_state=None):
    """Layer for RNN.

    Note:
        This function is not recommended. Use rnn_v2.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        num_layers(int): The number of layers in the RNN.
        hidden_size(int): The size of the internal hidden state for each layer.
        max_sequence_length(int): The maximum length of the time sequence.
        op(turret.RNNOperation): RNN operator.
        mode(turret.RNNInputMode): RNN input mode.
        direction(turret.RNNDirection): Direction of RNN.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        hidden_state(turret.Tensor): Initial hidden status of RNN.
        cell_state(turret.Tensor, optional): Initial cell state of RNN.
            Default to None.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    input.network.add_resource((weights, bias))

    raw_network = input.network.thisptr
    layer = raw_network.addRNN(
        input.thisptr[0], num_layers, hidden_size, max_sequence_length,
        op.thisobj, mode.thisobj, direction.thisobj,
        weights.thisobj, bias.thisobj)

    if hidden_state is not None:
        layer.setHiddenState(hidden_state.thisptr[0])
    if cell_state is not None:
        layer.setCellState(cell_state.thisptr[0])

    if op == RNNOperation.LSTM:
        return (Tensor.create(layer.getOutput(0), input.network),
                Tensor.create(layer.getOutput(1), input.network),
                Tensor.create(layer.getOutput(2), input.network))
    else:
        return (Tensor.create(layer.getOutput(0), input.network),
                Tensor.create(layer.getOutput(1), input.network))



#--------------------------------------------------------------------
# RNNv2
#--------------------------------------------------------------------
cdef class RNNGateType:
    """Identifies an individual gate within an RNN cell.

    Attributes:
        INPUT(turret.RNNGateType): Input gate (i).
        OUTPUT(turret.RNNGateType): Output gate (o).
        FORGET(turret.RNNGateType): Forget gate (f).
        UPDATE(turret.RNNGateType): Update gate (u).
        RESET(turret.RNNGateType): Reset gate (r).
        CELL(turret.RNNGateType): Cell gate (c).
        INPUT(turret.RNNGateType): Hidden gate (h).
    """

    cdef nvinfer.RNNGateType thisobj

    INPUT = RNNGateType(<int>rnn_gate_type.kINPUT)
    """Input gate (i)."""

    OUTPUT = RNNGateType(<int>rnn_gate_type.kOUTPUT)
    """Output gate (o)."""

    FORGET = RNNGateType(<int>rnn_gate_type.kFORGET)
    """Forget gate (f)."""

    UPDATE = RNNGateType(<int>rnn_gate_type.kUPDATE)
    """Update gate (u)."""

    RESET = RNNGateType(<int>rnn_gate_type.kRESET)
    """Reset gate (r)."""

    CELL = RNNGateType(<int>rnn_gate_type.kCELL)
    """Cell gate (c)."""

    HIDDEN = RNNGateType(<int>rnn_gate_type.kHIDDEN)
    """Hidden gate (h)."""

    _NAME_TABLE = {
        <int>rnn_gate_type.kINPUT:  "INPUT",
        <int>rnn_gate_type.kOUTPUT: "OUTPUT",
        <int>rnn_gate_type.kFORGET: "FORGET",
        <int>rnn_gate_type.kUPDATE: "UPDATE",
        <int>rnn_gate_type.kRESET:  "RESET",
        <int>rnn_gate_type.kCELL:   "CELL",
        <int>rnn_gate_type.kHIDDEN: "HIDDEN",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.RNNGateType>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def rnn_v2(Tensor input, int num_layers, int hidden_size,
          int max_sequence_length, RNNOperation op,
          RNNInputMode mode, RNNDirection direction,
          list weights, list bias,
          Tensor hidden_state, Tensor cell_state=None,
          Tensor sequence_lengths=None):
    """Layer for RNN.

    Args:
        input(turret.Tensor): Tensor which will be processed by layer.
        num_layers(int): The number of layers in the RNN.
        hidden_size(int): The size of the internal hidden state for each layer.
        max_sequence_length(int): The maximum length of the time sequence.
        op(turret.RNNOperation): RNN operator.
        mode(turret.RNNInputMode): RNN input mode.
        direction(turret.RNNDirection): Direction of RNN.
        weights(turret.Weights): Weights.
        bias(turret.Weights): Bias.
        hidden_state(turret.Tensor): Initial hidden status of RNN.
        cell_state(turret.Tensor, optional): Initial cell state of RNN.
            Default to None.
        sequence_lengths(turret.Tensor, optional): The sequence lengths
            specified for the RNN.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    input.network.add_resource((weights, bias))

    raw_network = input.network.thisptr
    layer = raw_network.addRNNv2(
        input.thisptr[0], num_layers, hidden_size,
        max_sequence_length, op.thisobj)

    layer.setDirection(direction.thisobj)

    for w in weights:
        layer.setWeightsForGate(
            w[0], (<RNNGateType>w[1]).thisobj, w[2], (<Weights>w[3]).thisobj)
    for b in bias:
        layer.setBiasForGate(
            b[0], (<RNNGateType>b[1]).thisobj, b[2], (<Weights>b[3]).thisobj)

    if hidden_state is not None:
        layer.setHiddenState(hidden_state.thisptr[0])
    if cell_state is not None:
        layer.setCellState(cell_state.thisptr[0])
    if sequence_lengths is not None:
        layer.setSequenceLengths(sequence_lengths.thisptr[0])

    if op == RNNOperation.LSTM:
        return (Tensor.create(layer.getOutput(0), input.network),
                Tensor.create(layer.getOutput(1), input.network),
                Tensor.create(layer.getOutput(2), input.network))
    else:
        return (Tensor.create(layer.getOutput(0), input.network),
                Tensor.create(layer.getOutput(1), input.network))



#--------------------------------------------------------------------
# Plugin
#--------------------------------------------------------------------
def plugin(object tensors, object instance):
    """Plugin layer.

    Args:
        tensors(turret.Tensor): Tensors which will be processed by plugin.
        instance(turret.PluginBase): Plugin layer.

    Returns:
        tensor(turret.Tensor): Tensor processed by plugin.
    """
    if type(tensors) is not list:
        tensors = [tensors]
    assert len(tensors) > 0

    cdef:
        Tensor input0 = tensors[0]
        vector[nvinfer.ITensor*] v_tensors
        Tensor t
    for t in tensors:
        v_tensors.push_back(t.thisptr)

    network = input0.network
    raw_network = network.thisptr

    proxy = PluginProxy(instance)
    network.add_plugin(proxy)
    layer = raw_network.addPlugin(v_tensors.data(), v_tensors.size(),
                                  proxy.native_proxy[0])
    if layer == NULL:
        raise RuntimeError("failed to create plugin layer")

    n_output = instance.get_num_outputs()
    outputs = [
        Tensor.create(layer.getOutput(i), network)
        for i in range(n_output)
    ]
    if n_output == 0:
        return None
    elif n_output == 1:
        return outputs[0]
    else:
        return tuple(outputs)


#--------------------------------------------------------------------
# Unary
#--------------------------------------------------------------------
cdef class UnaryOperation:
    """
    Enumerates the unary operations that may be performed by an
    unary layer.

    Attributes:
        EXP(turret.UnaryOperation): Exponentation.
        LOG(turret.UnaryOperation): log (base e).
        SQRT(turret.UnaryOperation): Square root.
        RECIP(turret.UnaryOperation): reciprocal.
        ABS(turret.UnaryOperation): Absolute value.
        NEG(turret.UnaryOperation): Negation.
    """

    cdef nvinfer.UnaryOperation thisobj

    EXP = UnaryOperation(<int>unary_operation.kEXP)
    """Exponentation."""

    LOG = UnaryOperation(<int>unary_operation.kLOG)
    """log (base e)."""

    SQRT = UnaryOperation(<int>unary_operation.kSQRT)
    """Square root."""

    RECIP = UnaryOperation(<int>unary_operation.kRECIP)
    """reciprocal."""

    ABS = UnaryOperation(<int>unary_operation.kABS)
    """Absolute value."""

    NEG = UnaryOperation(<int>unary_operation.kNEG)
    """Negation."""

    _NAME_TABLE = {
        <int>unary_operation.kEXP:   "EXP",
        <int>unary_operation.kLOG:   "LOG",
        <int>unary_operation.kSQRT:  "SQRT",
        <int>unary_operation.kRECIP: "RECIP",
        <int>unary_operation.kABS:   "ABS",
        <int>unary_operation.kNEG:   "NEG",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.UnaryOperation>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def unary(Tensor input, UnaryOperation op):
    """Layer for unary operation.

    Args:
        input(turret.Tensor): The tensor to be done unary operation.
        op(turret.UnaryOperation): Unary operation.

    Returns:
        tensor(turret.Tensor): Operated tensor.
    """
    assert input.network
    assert input.network.thisptr

    raw_network = input.network.thisptr
    layer = raw_network.addUnary(input.thisptr[0], op.thisobj)
    if layer == NULL:
        raise RuntimeError("failed to create unary layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Reduce
#--------------------------------------------------------------------
cdef class ReduceOperation:
    """
    Enumerates the reduce operations that may be performed by an
    reduce layer.

    Attributes:
        SUM(turret.ReduceOperation): Sum of the elements.
        PROD(turret.ReduceOperation): Product of the elements.
        MAX(turret.ReduceOperation): Maximum of the elements.
        MIN(turret.ReduceOperation): Minimum of the elements.
        AVG(turret.ReduceOperation): Average of the elements.
    """

    cdef nvinfer.ReduceOperation thisobj

    SUM = ReduceOperation(<int>reduce_operation.kSUM)
    """Sum of the elements."""

    PROD = ReduceOperation(<int>reduce_operation.kPROD)
    """Product of the elements."""

    MAX = ReduceOperation(<int>reduce_operation.kMAX)
    """Maximum of the elements."""

    MIN = ReduceOperation(<int>reduce_operation.kMIN)
    """Minimum of the elements."""

    AVG = ReduceOperation(<int>reduce_operation.kAVG)
    """Average of the elements."""

    _NAME_TABLE = {
        <int>reduce_operation.kSUM:  "SUM",
        <int>reduce_operation.kPROD: "PROD",
        <int>reduce_operation.kMAX:  "MAX",
        <int>reduce_operation.kMIN:  "MIN",
        <int>reduce_operation.kAVG:  "AVG",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.ReduceOperation>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def reduce(Tensor input, ReduceOperation op, axes, bool keepdims=False):
    """Layer for unary operation.

    Args:
        input(turret.Tensor): The tensor to be done reduce operation.
        op(turret.ReduceOperation): Reduce operation.
        axes(int): Axes to apply layer.
        keepdims(bool, optional): True if keep dimensions of tensor. Default
            to False.

    Returns:
        tensor(turret.Tensor): Operated tensor.
    """
    assert input.network
    assert input.network.thisptr

    raw_network = input.network.thisptr
    layer = raw_network.addReduce(
        input.thisptr[0], op.thisobj, _convert_axes(axes), keepdims)
    if layer == NULL:
        raise RuntimeError("failed to create softmax layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Padding
#--------------------------------------------------------------------
def padding(Tensor input, pre, post):
    """Layer for padding.

    Args:
        input(turret.Tensor): The tensor which will be processed by layer.
        pre(float): The padding that is applied at the start of the tensor.
        post(float): The padding that is applied at the end of the tensor.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    raw_network = input.network.thisptr
    layer = raw_network.addPadding(
        input.thisptr[0], _make_dims_hw(pre), _make_dims_hw(post))
    if layer == NULL:
        raise RuntimeError("failed to create padding layer")
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# Shuffle
#--------------------------------------------------------------------
cdef nvinfer.Permutation _tuple2permutation(t):
   cdef nvinfer.Permutation perm
   for i, x in enumerate(t):
       perm.order[i] = x
   for i in range(len(t), 8):
       perm.order[i] = i
   return perm

def shuffle_and_reshape(Tensor input, tuple order1,
                        Dimensions shape, tuple order2):
    """Layer to shuffle and reshape.

    Args:
        input(turret.Tensor): The tensor which will be processed by layer.
        order1(tuple): The permutation applied by the first transpose
            operation.
        shape(turret.Dimensions): Reshaped dimensions.
        order2(tuple): The permutation applied by the second transpose
            operation.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    # TODO validation

    raw_network = input.network.thisptr
    layer = raw_network.addShuffle(input.thisptr[0])
    if layer == NULL:
        raise RuntimeError("failed to create shuffle layer")

    if order1 is not None:
        layer.setFirstTranspose(_tuple2permutation(order1))
    if shape is not None:
        layer.setReshapeDimensions(shape.to_nvinfer_dims())
    if order2 is not None:
        layer.setSecondTranspose(_tuple2permutation(order2))
    return Tensor.create(layer.getOutput(0), input.network)


#--------------------------------------------------------------------
# TopK
#--------------------------------------------------------------------
cdef class TopKOperation:
    """Enumerates the operations that may be performed by a TopK layer.

    Attributes:
        MAX(turret.TopKOperation): Maximum of the elements.
        MIN(turret.TopKOperation): Minimum of the elements.
    """

    cdef nvinfer.TopKOperation thisobj

    MAX = TopKOperation(<int>topk_operation.kMAX)
    """Maximum of the elements."""

    MIN = TopKOperation(<int>topk_operation.kMIN)
    """Minimum of the elements."""

    _NAME_TABLE = {
        <int>topk_operation.kMAX: "MAX",
        <int>topk_operation.kMIN: "MIN",
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.TopKOperation>x

    def __repr__(self):
        ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj


def top_k(Tensor input, TopKOperation op, k, reduce_axes):
    """Layer for TopK reduction.

    Args:
        input(turret.Tensor): The tensor which will be processed by layer.
        op(turret.TopKOperation): The TopK operation.
        k(int): The k value.
        reduce_axes(int): Axes to reduce for the layer.

    Returns:
        tensor(turret.Tensor):
            The two tensors. The first is the value, and the second is the indices.
    """
    assert input.network
    assert input.network.thisptr

    # TODO validation

    raw_network = input.network.thisptr
    layer = raw_network.addTopK(
        input.thisptr[0], op.thisobj, <int>k, <uint32_t>reduce_axes);
    if layer == NULL:
        raise RuntimeError("failed to create top-k layer")

    return (Tensor.create(layer.getOutput(0), input.network),
            Tensor.create(layer.getOutput(1), input.network))


#--------------------------------------------------------------------
# MatrixMultiply
#--------------------------------------------------------------------
def matrix_multiply(Tensor input0, transpose0,
                    Tensor input1, transpose1):
    """Layer for matrix multiply.

    Args:
        input0(turret.Tensor): The tensor which will be first processed by
            layer.
        transpose0(bool): True if you transpose input0.
        input1(turret.Tensor): The tensor which will be second processed by
            layer.
        transpose1(bool): True if you transpose input1.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input0.network
    assert input0.network.thisptr

    if input0.network is not input1.network:
        raise ValueError("input0 and input1 are not in same network")

    # TODO validation

    raw_network = input0.network.thisptr
    layer = raw_network.addMatrixMultiply(
        input0.thisptr[0], <bool>transpose0,
        input1.thisptr[0], <bool>transpose1)
    if layer == NULL:
        raise RuntimeError("failed to create matrix multiply layer")

    return Tensor.create(layer.getOutput(0), input0.network)


#--------------------------------------------------------------------
# Gather
#--------------------------------------------------------------------
def gather(Tensor data, Tensor indices, int axis):
    """Layer for gathering.

    Args:
        data(turret.Tensor): The tensor to gather values from.
        indices(turret.Tensor): The tensor to get indices from to populate
            the output tensor.
        axis(int): The non-batch dimension axis in the data tensor to gather
            on.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert data.network
    assert data.network.thisptr

    if data.network is not indices.network:
        raise ValueError("data and indices are not in same network")

    # TODO validation

    raw_network = data.network.thisptr
    layer = raw_network.addGather(
        data.thisptr[0], indices.thisptr[0], axis)
    if layer == NULL:
        raise RuntimeError("failed to create gather layer")

    return Tensor.create(layer.getOutput(0), data.network)


#--------------------------------------------------------------------
# RaggedSoftMax
#--------------------------------------------------------------------
def ragged_softmax(Tensor input, Tensor bounds):
    """Layer for ragged softmax.

    Args:
        input(turret.Tensor): The ZxS input tensor.
        bounds(turret.Tensor): The Zx1 bounds tensor.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    assert input.network
    assert input.network.thisptr

    # TODO validation
    # input: ZxS, bounds: Zx1

    if input.network is not bounds.network:
        raise ValueError("input and bounds are not in same network")

    raw_network = input.network.thisptr
    layer = raw_network.addRaggedSoftMax(input.thisptr[0],
                                         bounds.thisptr[0])
    if layer == NULL:
        raise RuntimeError("failed to create ragged softmax layer")
    return Tensor.create(layer.getOutput(0), input.network)
