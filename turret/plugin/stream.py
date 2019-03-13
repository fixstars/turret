# -*- coding: utf-8 -*-
from pycuda import driver as cuda

from .temporary_stream_context import TemporaryStreamContext


class StreamRegistory:
    """The class for internal processing. Users need not to use."""

    def __init__(self):
        self.map = {}

    def __len__(self):
        return len(self.map)

    def __getitem__(self, key):
        return self.map[key]

    def __contains__(self, key):
        return key in self.map

    def register(self, stream):
        assert isinstance(stream, cuda.Stream)
        self.map[stream.handle] = stream

    def unregister(self, stream):
        assert isinstance(stream, cuda.Stream)
        if stream.handle in self.map:
            del self.map[stream.handle]


class PluginStream:
    """The class for internal processing. Users need not to use."""

    def __init__(self, main_stream, stream_registory):
        self.main_stream = main_stream
        self.stream_registory = stream_registory
        self.temporary_context = None

    def __enter__(self):
        assert self.temporary_context is None
        if self.main_stream in self.stream_registory:
            return self.stream_registory[self.main_stream]
        else:
            self.temporary_context = TemporaryStreamContext(self.main_stream)
            return self.temporary_context.enter()

    def __exit__(self, exc_name, exc_type, traceback):
        if self.temporary_context:
            self.temporary_context.exit()
            self.temporary_context = None
