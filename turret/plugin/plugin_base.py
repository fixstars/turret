# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pickle
from abc import ABCMeta, abstractmethod

from six import with_metaclass


class PluginBase(with_metaclass(ABCMeta)):
    """The superclass to create the plugin of turret."""

    @classmethod
    @abstractmethod
    def module_name(cls):
        """Get the plugin name.

        Note:
            This is a abstractmethod. Be sure to implement on the subclass.

        Returns:
            module_name(str): The plugin name.
        """
        raise NotImplementedError()

    def serialize(self, stream):
        """Serialize the binary to a stream.

        Args:
            stream(binary stream): A stream to write serialized engine.
        """
        pickle.dump(self, stream)

    @classmethod
    def deserialize(cls, stream):
        """Deserialize the plugin using a stream.

        Args:
            stream(binary stream): A stream to read serialized engine.
        """
        return pickle.load(stream)

    @abstractmethod
    def get_num_outputs(self):
        """Get the number of outputs.

        Note:
            This is a abstractmethod. Be sure to implement on the subclass.

        Returns:
            num_outputs(str): The number of outputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_dimensions(self, in_dims):
        """Get the dimension of outputs.

        Note:
            This is a abstractmethod. Be sure to implement on the subclass.

        Args:
            in_dims: The dimension of inputs.

        Returns:
            num_outputs(str): The dimension of outputs.
        """
        raise NotImplementedError()

    def configure(self, in_dims, out_dims, max_batch_size):
        """Configure the dimensions and the maximum of batch size.

        Args:
            in_dims: The dimension of inputs.
            out_dims: The dimension of outputs.
            max_batch_size(int): The maximum of batch size.
        """
        pass

    def initialize(self):
        """Initialize the plugin."""
        pass

    def terminate(self):
        """Terminate the plugin."""
        pass

    def get_workspace_size(self, max_batch_size):
        """Get the workspace size.

        Args:
            max_batch_size(int): The maximum of batch size.

        Returns:
            workspace_size(int): The workspace size.
        """
        return 0

    @abstractmethod
    def enqueue(self, batch_size, inputs, outputs, workspace, stream):
        """Asynchronously execute inference on a batch.

        Note:
            This is a abstractmethod. Be sure to implement on the subclass.

        Args:
            batch_size(int): The batch size.
            inputs: The input data.
            outputs: The buffer to output data.
            workspace: The workspace.
            stream(cuda.Stream): The cuda stream.
        """
        raise NotImplementedError()
