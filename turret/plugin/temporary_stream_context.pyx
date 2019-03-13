# distutils: language = c++
# distutils: sources = ["turret/plugin/temporary_stream_context_deleter.cpp"]
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["cuda"]
from libc.stdint cimport uintptr_t

from pycuda import driver as cuda


cdef object _plugin_stream = cuda.Stream()

cdef class TemporaryStreamContext:
    """The class for internal processing. Users need not to use."""
    cdef CUstream main_stream
    cdef CUevent main_completion_event
    cdef CUevent plugin_completion_event

    def __cinit__(self, main_stream):
        cdef uintptr_t main_stream_handle = <uintptr_t>int(main_stream)
        self.main_stream = <CUstream>main_stream_handle
        cuEventCreate(&self.main_completion_event, 0)
        cuEventCreate(&self.plugin_completion_event, 0)

    cdef CUstream _to_stream(self, object s):
        assert isinstance(s, cuda.Stream)
        cdef uintptr_t h = <uintptr_t>s.handle
        return <CUstream>h

    def enter(self):
        e = self.main_completion_event
        cuEventRecord(e, self.main_stream)
        cuStreamWaitEvent(self._to_stream(_plugin_stream), e, 0)
        return _plugin_stream

    def exit(self):
        e = self.plugin_completion_event
        cuEventRecord(e, self._to_stream(_plugin_stream))
        cuStreamWaitEvent(self.main_stream, e, 0)
        ret = cuStreamAddCallback(
                self.main_stream, release_temporary_stream_events,
                new TemporaryStreamContextDeleter(
                    self.main_completion_event,
                    self.plugin_completion_event),
                0)
