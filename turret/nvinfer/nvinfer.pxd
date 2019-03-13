from libc.stdint cimport int32_t
from libc.stdint cimport uint32_t
from libc.stdint cimport int64_t
from libcpp cimport bool

cdef extern from "driver_types.h":
    ctypedef void * cudaStream_t
    ctypedef void * cudaEvent_t

cdef extern from "NvInfer.h" namespace "nvinfer1":

    cdef cppclass DataType:
        pass

    cdef cppclass DimensionType:
        pass


    cdef cppclass Dims:
        int nbDims
        int d[8]
        DimensionType type[8]

    cdef cppclass DimsHW(Dims):
        DimsHW()
        DimsHW(int, int)
        int& h()
        int  h() const
        int& w()
        int  w() const

    cdef cppclass DimsCHW(Dims):
        DimsCHW()
        DimsCHW(int, int, int)
        int& c()
        int  c() const
        int& h()
        int  h() const
        int& w()
        int  w() const

    cdef cppclass DimsNCHW(Dims):
        DimsNCHW()
        DimsNCHW(int, int, int, int)
        int& n()
        int  n() const
        int& c()
        int  c() const
        int& h()
        int  h() const
        int& w()
        int  w() const


    cdef cppclass Weights:
        DataType type
        const void *values
        int64_t count


    cdef cppclass IHostMemory:
        void *data() except +
        size_t size() except +
        DataType type() except +
        void destroy() except +


    cdef cppclass LayerType:
        pass


    cdef cppclass ITensor:
        void setName(const char *) except +
        const char *getName() except +
        void setDimensions(Dims) except +
        Dims getDimensions() except +
        void setType(DataType) except +
        DataType getType() except +
        bool isNetworkInput() except +
        bool isNetworkOutput() except +
        void setBroadcastAcrossBatch(bool) except +
        bool getBroadcastAcrossBatch() except +


    cdef cppclass ILayer:
        LayerType getType() except +
        void setName(const char *) except +
        const char *getName() except +
        int getNbInputs() except +
        ITensor *getInput(int) except +
        int getNbOutputs() except +
        ITensor *getOutput(int) except +

    cdef cppclass IConvolutionLayer(ILayer):
        void setKernelSize(DimsHW) except +
        DimsHW getKernelSize() except +
        void setNbOutputMaps(int) except +
        int getNbOutputMaps() except +
        void setStride(DimsHW) except +
        DimsHW getStride() except +
        void setPadding(DimsHW) except +
        DimsHW getPadding() except +
        void setNbGroups(int) except +
        int getNbGroups() except +
        void setKernelWeights(Weights) except +
        Weights getKernelWeights() except +
        void setBiasWeights(Weights) except +
        Weights getBiasWeights() except +
        void setDilation(DimsHW) except +
        DimsHW getDilation() except +

    cdef cppclass IFullyConnectedLayer(ILayer):
        void setNbOutputChannels(int) except +
        int getNbOutputChannels() except +
        void setKernelWeights(Weights) except +
        Weights getKernelWeights() except +
        void setBiasWeights(Weights) except +
        Weights getBiasWeights() except +

    cdef cppclass ActivationType:
        pass

    cdef cppclass IActivationLayer(ILayer):
        void setActivationType(ActivationType) except +
        ActivationType getActivationType() except +

    cdef cppclass PoolingType:
        pass

    cdef cppclass IPoolingLayer(ILayer):
        void setPoolingType(PoolingType) except +
        PoolingType getPoolingType() except +
        void setWindowSize(DimsHW) except +
        DimsHW getWindowSize() except +
        void setStride(DimsHW) except +
        DimsHW getStride() except +
        void setPadding(DimsHW) except +
        DimsHW getPadding() except +
        void setBlendFactor(float) except +
        float getBlendFactor() except +

    cdef cppclass ILRNLayer(ILayer):
        void setWindowSize(int) except +
        int getWindowSize() except +
        void setAlpha(float) except +
        float getAlpha() except +
        void setBeta(float) except +
        float getBeta() except +
        void setK(float) except +
        float getK() except +

    cdef cppclass ScaleMode:
        pass

    cdef cppclass IScaleLayer(ILayer):
        void setMode(ScaleMode) except +
        ScaleMode getMode() except +
        void setShift(Weights) except +
        Weights getShift() except +
        void setScale(Weights) except +
        Weights getScale(Weights) except +
        void setPower(Weights) except +
        Weights getPower() except +

    cdef cppclass ISoftMaxLayer(ILayer):
        void setAxes(uint32_t) except +
        uint32_t getAxes() except +

    cdef cppclass IConcatenationLayer(ILayer):
        void setAxis(int) except +
        int getAxis() except +

    cdef cppclass IDeconvolutionLayer(ILayer):
        void setKernelSize(DimsHW) except +
        DimsHW getKernelSize() except +
        void setNbOutputMaps(int) except +
        int getNbOutputMaps() except +
        void setStride(DimsHW) except +
        DimsHW getStride() except +
        void setPadding(DimsHW) except +
        DimsHW getPadding() except +
        void setNbGroups(int) except +
        int getNbGroups() except +
        void setKernelWeights(Weights) except +
        Weights getKernelWeights() except +
        void setBiasWeights(Weights) except +
        Weights getBiasWeights() except +

    cdef cppclass ElementWiseOperation:
        pass

    cdef cppclass IElementWiseLayer(ILayer):
        void setOperation(ElementWiseOperation)
        ElementWiseOperation getOperation() const

    cdef cppclass IGatherLayer(ILayer):
        void setGatherAxis(int) except +
        int getGatherAxis() except +

    cdef cppclass IPlugin:
        int getNbOutputs() except +
        Dims getOutputDimensions(int, const Dims *, int) except +
        void configure(const Dims *, int, const Dims *, int,
                       int) except +
        int initialize() except +
        void terminate() except +
        size_t getWorkspaceSize(int) except +
        int enqueue(int, const void * const *, void **, void *,
                    cudaStream_t) except +
        size_t getSerializationSize() except +
        void serialize(void *) except +

    cdef cppclass IPluginLayer(ILayer):
        IPlugin& getPlugin() except +

    cdef cppclass UnaryOperation:
        pass

    cdef cppclass IUnaryLayer(ILayer):
        void setOperation(UnaryOperation) except +
        UnaryOperation getOperation() except +

    cdef cppclass ReduceOperation:
        pass

    cdef cppclass IReduceLayer(ILayer):
        void setOperation(ReduceOperation) except +
        ReduceOperation getOperation() except +
        void setReduceAxes(uint32_t) except +
        uint32_t getReduceAxes() except +
        void setKeepDimensions(bool) except +
        bool getKeepDimensions() except +

    cdef cppclass IPaddingLayer(ILayer):
        void setPrePadding(DimsHW) except +
        DimsHW getPrePadding() except +
        void setPostPadding(DimsHW) except +
        DimsHW getPostPadding() except +

    cdef cppclass RNNOperation:
        pass

    cdef cppclass RNNDirection:
        pass

    cdef cppclass RNNInputMode:
        pass

    cdef cppclass IRNNLayer(ILayer):
        unsigned int getLayerCount() except +
        size_t getHiddenSize() except +
        int getSeqLength() except +
        void setOperation(RNNOperation) except +
        RNNOperation getOperation() except +
        void setInputMode(RNNInputMode) except +
        RNNInputMode getInputMode() except +
        void setDirection(RNNDirection) except +
        RNNDirection getDirection() except +
        void setWeights(Weights) except +
        Weights getWeights() except +
        void setBias(Weights) except +
        Weights getBias() except +
        int getDataLength() except +
        void setHiddenState(ITensor&) except +
        ITensor *getHiddenState() except +
        void setCellState(ITensor&) except +
        ITensor *getCellState() except +

    cdef cppclass RNNGateType:
        pass

    cdef cppclass IRNNv2Layer(ILayer):
        int32_t getLayerCount() except +
        int32_t getHiddenSize() except +
        int32_t getMaxSeqLength() except +
        int32_t getDataLength() except +
        void setSequenceLengths(ITensor&) except +
        ITensor *getSequenceLengths() except +
        void setOperation(RNNOperation) except +
        RNNOperation getOperation() except +
        void setInputMode(RNNInputMode) except +
        RNNInputMode getInputMode() except +
        void setDirection(RNNDirection) except +
        RNNDirection getDirection() except +
        void setWeightsForGate(int, RNNGateType, bool, Weights) except +
        Weights getWeightsForGate(int, RNNGateType, bool) except +
        void setBiasForGate(int, RNNGateType, bool, Weights) except +
        Weights getBiasForGate(int, RNNGateType, bool) except +
        void setHiddenState(ITensor&) except +
        ITensor *getHiddenState() except +
        void setCellState(ITensor&) except +
        ITensor *getCellState() except +


    cdef cppclass Permutation:
        int order[8]

    cdef cppclass IShuffleLayer(ILayer):
        void setFirstTranspose(Permutation) except +
        Permutation getFirstTranspose() except +
        void setReshapeDimensions(Dims) except +
        Dims getReshapeDimensions() except +
        void setSecondTranspose(Permutation) except +
        Permutation getSecondTranspose() except +


    cdef cppclass TopKOperation:
        pass

    cdef cppclass ITopKLayer(ILayer):
        void setOperation(TopKOperation) except +
        TopKOperation getOperation() except +
        void setK(int) except +
        int getK() except +
        void setReduceAxes(uint32_t) except +
        uint32_t getReduceAxes() except +

    cdef cppclass IMatrixMultiplyLayer(ILayer):
        void setTranspose(int, bool) except +
        bool getTranspose(int) except +

    cdef cppclass IRaggedSoftMaxLayer(ILayer):
        pass

    cdef cppclass IConstantLayer(ILayer):
        void setWeights(Weights) except +
        Weights getWeights() except +
        void setDimensions(Dims) except +
        Dims getDimensions() except +


    cdef cppclass INetworkDefinition:
        ITensor *addInput(const char *, DataType, Dims) except +
        void markOutput(ITensor&) except +
        IConvolutionLayer *addConvolution(
            ITensor&, int, DimsHW, Weights, Weights) except +
        IFullyConnectedLayer *addFullyConnected(
            ITensor&, int, Weights, Weights) except +
        IActivationLayer *addActivation(
            ITensor&, ActivationType) except +
        IPoolingLayer *addPooling(
            ITensor&, PoolingType, DimsHW) except +
        ILRNLayer *addLRN(ITensor&, int, float, float, float) except +
        IScaleLayer *addScale(
            ITensor&, ScaleMode, Weights, Weights, Weights) except +
        ISoftMaxLayer *addSoftMax(ITensor&) except +
        IConcatenationLayer *addConcatenation(
            ITensor * const *, int) except +
        IDeconvolutionLayer *addDeconvolution(
            ITensor&, int, DimsHW, Weights, Weights) except +
        IElementWiseLayer *addElementWise(
            ITensor&, ITensor&, ElementWiseOperation) except +
        IPluginLayer *addPlugin(
            ITensor * const *, int, IPlugin&) except +
        IUnaryLayer *addUnary(ITensor&, UnaryOperation) except +
        IReduceLayer *addReduce(ITensor&, ReduceOperation, uint32_t, bool) except +
        IPaddingLayer *addPadding(ITensor&, DimsHW, DimsHW) except +
        IRNNLayer *addRNN(
            ITensor&, int, size_t, int, RNNOperation, RNNInputMode,
            RNNDirection, Weights, Weights) except +
        IRNNv2Layer *addRNNv2(
            ITensor&, int32_t, int32_t, int32_t, RNNOperation) except +
        IShuffleLayer *addShuffle(ITensor&) except +
        ITopKLayer *addTopK(ITensor&, TopKOperation, int, uint32_t) except +
        IMatrixMultiplyLayer *addMatrixMultiply(
            ITensor&, bool, ITensor&, bool) except +
        IGatherLayer *addGather(ITensor&, ITensor&, int) except +
        IRaggedSoftMaxLayer *addRaggedSoftMax(ITensor&, ITensor&) except +
        IConstantLayer *addConstant(Dims, Weights) except +
        int getNbLayers() except +
        ILayer *getLayer(int) except +
        int getNbInputs() except +
        ITensor *getInput(int) except +
        int getNbOutputs() except +
        ITensor *getOutput(int) except +
        void destroy() except +

    cdef cppclass ICudaEngine

    cdef cppclass IExecutionContext:
        bool execute(int, void **) except +
        bool enqueue(
            int, void **, cudaStream_t, cudaEvent_t *) except +
        void setDebugSync(bool) except +
        bool getDebugSync() except +
        const ICudaEngine& getEngine() except +
        void destroy() except +

    cdef cppclass ICudaEngine:
        int getNbBindings() except +
        int getBindingIndex(const char *) except +
        const char *getBindingName(int) except +
        bool bindingIsInput(int) except +
        Dims getBindingDimensions(int) except +
        DataType getBindingDataType(int) except +
        int getMaxBatchSize() except +
        int getNbLayers() except +
        size_t getWorkspaceSize() except +
        IHostMemory *serialize() except +
        IExecutionContext *createExecutionContext() except +
        void destroy() except +


    cdef cppclass CalibrationAlgoType:
        pass

    cdef cppclass IInt8Calibrator:
        int getBatchSize() except +
        bool getBatch(void *[], const char *[], int) except +
        const void *readCalibrationCache(size_t&) except +
        void writeCalibrationCache(const void *, size_t) except +
        CalibrationAlgoType getAlgorithm() except +

    cdef cppclass IInt8EntropyCalibrator(IInt8Calibrator):
        pass

    cdef cppclass IInt8LegacyCalibrator(IInt8Calibrator):
        double getQuantile() except +
        double getRegressionCutoff() except +
        const void *readHistogramCache(size_t&) except +
        void writeHistogramCache(const void *, size_t) except +


    cdef cppclass IBuilder:
        INetworkDefinition *createNetwork() except +
        void setMaxBatchSize(int) except +
        int getMaxBatchSize() except +
        void setMaxWorkspaceSize(size_t) except +
        size_t getMaxWorkspaceSize() except +
        void setHalf2Mode(bool) except +
        bool getHalf2Mode() except +
        void setDebugSync(bool) except +
        bool getDebugSync() except +
        void setMinFindIterations(int) except +
        int getMinFindIterations() except +
        void setAverageFindIterations(int) except +
        int getAverageFindIterations() except +
        ICudaEngine *buildCudaEngine(INetworkDefinition& network) except +
        bool platformHasFastFp16() except +
        bool platformHasFastInt8() except +
        void destroy() except +
        void setInt8Mode(bool) except +
        bool getInt8Mode() except +
        void setInt8Calibrator(IInt8Calibrator *) except +


    cdef cppclass IPluginFactory:
        IPlugin *createPlugin(const char *, const void *, size_t) except +

    cdef cppclass IRuntime:
        ICudaEngine *deserializeCudaEngine(
            const void *, size_t, IPluginFactory *) except +
        void destroy() except +


cdef extern from "NvInfer.h" namespace "nvinfer1::ILogger":
    cdef cppclass Severity:
        pass

cdef extern from "NvInfer.h" namespace "nvinfer1":
    cdef cppclass ILogger:
        void log(Severity, const char *) except +


    cdef IBuilder *createInferBuilder(ILogger&) except +
    cdef IRuntime *createInferRuntime(ILogger&) except +
