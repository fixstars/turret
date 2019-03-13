# -*- coding: utf-8 -*-
from ..plugin.cy_plugin_proxy import PluginProxy


class PluginFactory:
    """The PluginFactory for caffe model."""

    def __init__(self):
        self.factories = {}
        self.created = []

    def is_plugin(self, layer_name):
        """Check whether the layer is in the plugin factory.

        Args:
            layer_name(str): The name of layer.

        Returns:
            result(bool): true if the layer name is in the plugin factory.
        """
        return layer_name in self.factories

    def create(self, layer_name, weights):
        """Create the plugin.

        Args:
            layer_name(str): The name of layer.
            weights(turret.Weights): The weight parameter for the layer.

        Returns:
            plugin(turret.PluginProxy): The plugin object for the layer.
        """
        if layer_name not in self.factories:
            raise KeyError("unregistered layer '{}'".format(layer_name))
        factory = self.factories[layer_name]
        plugin = factory(layer_name, weights)
        proxy = PluginProxy(plugin)
        self.created.append(proxy)
        return proxy

    def register_plugin(self, layer_name, factory):
        """Register the plugin.

        Args:
            layer_name(str): The name of layer.
            factory: The object of plugin.
        """
        # TODO check overwrite
        self.factories[layer_name] = factory
