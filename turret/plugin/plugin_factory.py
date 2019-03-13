# -*- coding: utf-8 -*-
import json

from .cy_plugin_proxy import PluginProxy
from .auto_register import registered_classes


class PluginFactory:
    """The class of custom layer factory."""

    def __init__(self):
        self.classes = {}
        self.created = []

    @staticmethod
    def _read_until(stream, terminal=b"\0"):
        result = b""
        while True:
            c = stream.read(1)
            if c == terminal:
                break
            result += c
        return result

    def create(self, stream):
        """
        Create the plugin from plugin factory.

        Args:
            stream(binary stream): The binary stream to deserialize plugin.

        Returns:
            proxy(PluginProxy): The created plugin.
        """
        proxy_params = json.loads(self._read_until(stream).decode("utf-8"))
        module_name = proxy_params["name"]
        if module_name in self.classes:
            plugin = self.classes[module_name].deserialize(stream)
        elif module_name in registered_classes:
            plugin = registered_classes[module_name].deserialize(stream)
        else:
            raise ValueError("unregistered plugin")
        proxy = PluginProxy(plugin, proxy_params)
        self.created.append(proxy)
        return proxy

    def register_plugin(self, klass):
        """
        Register the plugin on the  plugin factory.

        Args:
            klass: The plugin.
        """
        # TODO check overwrite?
        self.classes[klass.module_name()] = klass
