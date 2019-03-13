# distutils: language = c++
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
from .nvinfer cimport nvinfer

from .logger cimport Severity


cdef public api int cy_call_logger_log(
        object self, nvinfer.Severity severity, const char *message):
    self.log(Severity(<int>severity), message.decode("utf-8"))
    return 0
