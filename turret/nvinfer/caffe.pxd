from libc.stdint cimport int64_t
from libcpp cimport bool

from .nvinfer cimport DataType
from .nvinfer cimport DimsNCHW
from .nvinfer cimport Weights
from .nvinfer cimport ITensor
from .nvinfer cimport IPlugin
from .nvinfer cimport INetworkDefinition

cdef extern from "NvCaffeParser.h" namespace "nvcaffeparser1":

    cdef cppclass IBlobNameToTensor:
        ITensor *find(const char *) except +

    cdef cppclass IBinaryProtoBlob:
        const void *getData() except +
        DimsNCHW getDimensions() except +
        DataType getDataType() except +
        void destroy() except +

    cdef cppclass IPluginFactory:
        bool isPlugin(const char *) except +
        IPlugin *createPlugin(const char *, const Weights *, int) except +

    cdef cppclass ICaffeParser:
        const IBlobNameToTensor *parse(
                const char *, const char *, INetworkDefinition&,
                DataType) except +
        void setProtobufBufferSize(size_t) except +
        void setPluginFactory(IPluginFactory *) except +
        IBinaryProtoBlob *parseBinaryProto(const char *) except +
        void destroy() except +

    cdef ICaffeParser *createCaffeParser() except +
    void shutdownProtobufLibrary() except +
