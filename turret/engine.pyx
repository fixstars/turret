# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
import numpy as np
cimport numpy as np

from libc.stdint cimport uintptr_t
from libcpp cimport bool

import pycuda.autoinit
import pycuda.driver as cuda

from .nvinfer cimport nvinfer

from .logger cimport LoggerProxy

from .foundational cimport DataType
from .foundational cimport Dimensions
from .graph cimport NetworkDefinition
from .buffer cimport InferenceBuffer

from .int8.cy_calibrator_proxy cimport CalibratorProxy
from .plugin.cy_plugin_factory_proxy cimport PluginFactoryProxy


cdef class InferenceEngineBuilder:
    """A builder for constructing inference engines."""

    def __cinit__(self, logger):
        self.logger = LoggerProxy(logger)
        self.thisptr = nvinfer.createInferBuilder(self.logger.proxy[0])
        self.int8_calibrator = None

    def __dealloc__(self):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL

    def build(self, NetworkDefinition network):
        """Build an inference engine.

        Args:
            network (turret.NetworkDefinition): The network definition.

        Returns:
            turret.InferenceEngine: The constructed inference engine.
        """

        cdef InferenceEngine engine
        self.thisptr.setInt8Mode(network.dtype == DataType.INT8)
        self.thisptr.setHalf2Mode(network.dtype == DataType.HALF)
        if network.builder is not self:
            raise ValueError()  # TODO write message
        if self.int8_calibrator:
            self.int8_calibrator.network = network
            calibrator_proxy = CalibratorProxy(self.int8_calibrator)
            self.thisptr.setInt8Calibrator(calibrator_proxy.native_proxy)
        engine = InferenceEngine.create(
            self.thisptr.buildCudaEngine(network.thisptr[0]),
            network.plugins)
        return engine

    def create_network(self, dtype=DataType.FLOAT):
        """Create an empty network definition.

        Args:
            dtype (turret.DataType, optional): The data type for inference.
                Default to DataType.FLOAT.

        Returns:
            turret.NetworkDefinition: The empty network definition.
        """
        return NetworkDefinition(self, dtype)

    property max_batch_size:
        """The maximum batch size."""

        def __get__(self):
            return self.thisptr.getMaxBatchSize()

        def __set__(self, int batch_size):
            self.thisptr.setMaxBatchSize(batch_size)

    property max_workspace_size:
        """The maximum workspace size in bytes."""

        def __get__(self):
            return self.thisptr.getMaxWorkspaceSize()

        def __set__(self, size_t workspace_size):
            self.thisptr.setMaxWorkspaceSize(workspace_size)

    property average_find_iterations:
        """The number of minimization iterations in minimization."""

        def __get__(self):
            return self.thisptr.getAverageFindIterations()

        def __set__(self, int avg_find):
            self.thisptr.setAverageFindIterations(avg_find)

    property min_find_iterations:
        """The number of minimization iterations in averaging."""

        def __get__(self):
            return self.thisptr.getMinFindIterations()

        def __set__(self, int min_find):
            self.thisptr.setMinFindIterations(min_find)

    property debug_sync:
        """The debug sync flag."""

        def __get__(self):
            return self.thisptr.getDebugSync()

        def __set__(self, bool sync):
            self.thisptr.setDebugSync(sync)

    property int8_calibrator:
        """INT8 calibration interface."""

        def __get__(self):
            return self.int8_calibrator

        def __set__(self, object calibrator):
            self.int8_calibrator = calibrator

    def platform_has_fast_fp16(self):
        """Returns True when the platform has fast native fp16."""
        return self.thisptr.platformHasFastFp16()

    def platform_has_fast_int8(self):
        """Returns True when the platform has fast native int8."""
        return self.thisptr.platformHasFastInt8()


cdef class Binding:
    """Binding information for network inputs and outputs.

    Attributes:
        name (str): The name of the tensor.
        index (int): The index of the buffer.
        is_input (bool): True if the tensor is an input.
        type (turret.DataType): The data type of the tensor.
        dimensions (turret.Dimensions): The dimensions of the tensor.
    """
    pass


cdef class BindingSet:
    """A set of binding informations for network inputs and outputs."""
    def __cinit__(self):
        self.from_index = []
        self.from_name = {}

    @staticmethod
    cdef BindingSet create(nvinfer.ICudaEngine *raw_engine):
        cdef:
            BindingSet bindings
            Binding b
            int i
        bindings = BindingSet()
        for i in range(raw_engine.getNbBindings()):
            b = Binding()
            b.name = raw_engine.getBindingName(i).decode('utf-8')
            b.index = i
            b.is_input = raw_engine.bindingIsInput(i)
            b.type = DataType(<int>raw_engine.getBindingDataType(i))
            b.dimensions = Dimensions.from_nvinfer_dims(
                raw_engine.getBindingDimensions(i))
            bindings.from_index.append(b)
            bindings.from_name[b.name] = b
        return bindings

    def __iter__(self):
        return iter(self.from_index)

    def __len__(self):
        return len(self.from_index)

    def __getitem__(self, key):
        if type(key) is int:
            return self.from_index[key]
        elif type(key) is str:
            return self.from_name[key]
        return None


cdef class InferenceEngine:
    """An engine for executing inference on a built network."""

    def __cinit__(self):
        self.thisptr = NULL
        self.bindings = BindingSet()
        self.plugins = []

    def __dealloc__(self):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL

    @staticmethod
    cdef InferenceEngine create(nvinfer.ICudaEngine *raw_engine,
                                list plugins):
        """Create InferenceEngine of Turret from raw cuda engine.

        Args:
            raw_engine (nvinfer.ICudaEngine): Raw cuda engine.
            plugins (list): List of plugins.

        Returns:
            turret.InferenceEngine: InferenceEngine of Turret.
        """
        cdef InferenceEngine engine = InferenceEngine()
        engine.thisptr = raw_engine
        engine.bindings = BindingSet.create(raw_engine)
        engine.plugins = plugins
        return engine

    property max_batch_size:
        """The maximum batch size."""

        def __get__(self):
            return self.thisptr.getMaxBatchSize()

    property nlayers:
        """The number of layers in the network."""

        def __get__(self):
            return self.thisptr.getNbLayers()

    property workspace_size:
        """The size of workspace in bytes."""

        def __get__(self):
            return self.thisptr.getWorkspaceSize()

    def serialize(self, stream):
        """Serialize the network to a stream.

        Args:
            stream (binary stream): A stream to write serialized engine.
        """
        hmem = self.thisptr.serialize()
        cdef char *data = <char *>hmem.data()
        cdef Py_ssize_t size = hmem.size()
        try:
            stream.write(data[:size])
        finally:
            hmem.destroy()


cdef class InferenceRuntime:
    """A runtime to allow a serialized engine to be deserialized."""
    def __cinit__(self, logger):
        self.logger = LoggerProxy(logger)
        self.thisptr = nvinfer.createInferRuntime(self.logger.proxy[0])

    def __dealloc__(self):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL

    def deserialize_engine(self, stream, plugin_factory=None):
        """Deserialize the network using a stream.

        Args:
            stream (binary stream): A stream to read serialized engine.
            plugin_factory (turret.PluginFactory, optional): The plugin
                factory. Default to None.

        Returns:
            turret.InferenceEngine: Deserialized InferenceEngine.
        """
        # TODO exception handling
        cdef:
            bytes blob = stream.read()
            char *blob_ptr = blob
        factory = PluginFactoryProxy(plugin_factory)
        raw_engine = self.thisptr.deserializeCudaEngine(
            blob_ptr, len(blob), factory.native_proxy)
        # TODO reset plugin_factory.created
        plugins = [] if plugin_factory is None else plugin_factory.created
        return InferenceEngine.create(raw_engine, plugins)


cdef class ExecutionContext:
    """Context for executing inference using an engine.
    Multiple execution contexts may exist for one InferenceEngine instance,
    allowing the same engine to be used for the execution of multiple batches
    simultaneously.
    """
    def __cinit__(self, InferenceEngine engine):
        self.thisptr = engine.thisptr.createExecutionContext()
        self.engine = engine
        self.default_stream = cuda.Stream()

    def __dealloc__(self):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_name, exc_type, traceback):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL
        self.engine = None

    def create_buffer(self):
        """Returns InferenceBuffer for this context.

        Returns:
            turret.InferenceBuffer: InferenceBuffer for this context.
        """
        return InferenceBuffer(self.engine)

    def execute(self, InferenceBuffer buf):
        """Synchronously execute inference on a batch.

        Args:
            buf (InferenceBuffer): InferenceBuffer for this context.
        """
        ret = self.enqueue(buf, self.default_stream)
        if ret:
            self.default_stream.synchronize()
        return ret

    def enqueue(self, InferenceBuffer buf, object stream):
        """Asynchronously execute inference on a batch.

        Args:
            buf (InferenceBuffer): InferenceBuffer for this context.
            stream (CUDA stream): A cuda stream on which the inference kernels
                                  will be enqueued.
        """
        cdef uintptr_t handle = stream.handle
        for plugin in self.engine.plugins:
            plugin.register_stream(stream)
        ret = self.thisptr.enqueue(buf.batch_size, buf.binding_pointers(),
                                   <nvinfer.cudaStream_t>handle, NULL)
        for plugin in self.engine.plugins:
            plugin.unregister_stream(stream)
        return ret
