# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvcaffe_parser", "nvinfer"]
from ..nvinfer cimport caffe

from ..foundational cimport DataType
from ..graph cimport Tensor
from ..graph cimport NetworkDefinition
from ..engine cimport InferenceEngineBuilder

from .cy_plugin_factory_proxy cimport PluginFactoryProxy

cdef class NamedTensorSet:
    """The object that contains the data extracted from caffe model. This
    object can be used as dictionary, and The data got from this object is
    ``turret.Tensor``.
    """
    cdef const caffe.IBlobNameToTensor *thisptr
    cdef NetworkDefinition network
    # IBlobNameToTensor is owned by the ICaffeParser.
    # Keep reference to avoid unexpected deallocation for thisptr.
    cdef CaffeParser parser

    def __cinit__(self):
        self.thisptr = NULL
        self.network = None
        self.parser = None

    @staticmethod
    cdef NamedTensorSet create(const caffe.IBlobNameToTensor *rawptr,
                               NetworkDefinition network,
                               CaffeParser parser):
        s = NamedTensorSet()
        s.thisptr = rawptr
        s.network = network
        s.parser = parser
        return s;

    def __getitem__(self, str key):
        ptr = self.thisptr.find(key.encode("utf-8")) 
        if ptr == NULL:
            raise KeyError(key)
        return Tensor.create(ptr, self.network)


cdef class CaffeParser:
    """Class used for parsing Caffe models.
    Allows users to export models trained using Caffe to TRT.
    """
    cdef caffe.ICaffeParser *thisptr
    cdef PluginFactoryProxy plugin_factory

    def __cinit__(self):
        self.thisptr = NULL
        self.plugin_factory = None

    def __dealloc__(self):
        if self.thisptr:
            self.thisptr.destroy()
            self.thisptr = NULL

    @staticmethod
    cdef create(caffe.ICaffeParser *rawptr):
        p = CaffeParser()
        p.thisptr = rawptr
        return p

    def set_plugin_factory(self, factory):
        """Set the plugin factory to parser.

        Args:
            factory(turret.PluginFactory): The plugin factory.
        """
        if factory is None:
            self.plugin_factory = None
            self.thisptr.setPluginFactory(NULL)
        else:
            self.plugin_factory = PluginFactoryProxy(factory)
            self.thisptr.setPluginFactory(self.plugin_factory.native_proxy)

    def parse(self, NetworkDefinition network, str deploy, str model):
        """Parse a prototxt file and a binaryproto Caffe model to extract
        network configuration and weights associated with the network,
        respectively.

        Args:
            network(turret.NetworkDefinition): The network in which the
                CaffeParser will fill the layers.
            deploy(str): The plain text, prototxt file used to define the
                network configuration.
            model(str): The binaryproto Caffe model that contains the weights
                associated with the network.

        Returns:
            tensorset(turret.NamedTensorSet): The object that contains the extracted data.
        """
        cdef DataType dtype = \
            DataType.HALF if network.dtype == DataType.HALF else DataType.FLOAT
        name_to_tensor = self.thisptr.parse(
            deploy.encode("utf-8"), model.encode("utf-8"),
            network.thisptr[0], dtype.thisobj)
        if name_to_tensor == NULL:
            raise RuntimeError(
                "failed to parse model (deploy: '{}', model: '{}')".format(deploy, model))
        if self.plugin_factory is not None:
            for plugin in self.plugin_factory.factory.created:
                network.add_plugin(plugin)
        return NamedTensorSet.create(name_to_tensor, network, self)


def import_caffemodel(NetworkDefinition network, str deploy, str model,
                      object plugin_factory=None):
    """Import caffe model to Turret.

    Args:
        network(turret.NetworkDefinition): The network in which the
            CaffeParser will fill the layers.
        deploy(str): The plain text, prototxt file used to define the network
            configuration.
        model(str): The binaryproto Caffe model that contains the weights
            associated with the network.
        plugin_factory(turret.PluginFactory, optional): The plugin factory.
            Default to None.

    Returns:
        tensorset(turret.NamedTensorSet): The object that contains the extracted data.
    """
    raw_parser = caffe.createCaffeParser()
    if raw_parser == NULL:
        raise RuntimeError("failed to createCaffeParser")
    parser = CaffeParser.create(raw_parser)
    parser.set_plugin_factory(plugin_factory)
    return parser.parse(network, deploy, model)
