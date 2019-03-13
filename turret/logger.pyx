# distutils: language = c++
# distutils: sources = ["turret/logger_proxy.cpp"]
# distutils: extra_compile_args = ["-std=c++11"]
# distutils: extra_link_args = ["-std=c++11"]
# distutils: libraries = ["nvinfer"]
from abc import ABCMeta, abstractmethod

from cpython cimport ref

from .nvinfer cimport nvinfer
from .nvinfer cimport severity


cdef class Severity:
    """The severity corresponding to a log message.

    Attributes:
        INTERNAL_ERROR(turret.Severity): An internal error has occurred.
            Execution is unrecoverable.
        ERROR(turret.Severity): An application error has occurred.
        WARNING(turret.Severity): An application error has been discovered,
            but TensorRT has recovered or fallen back to a default.
        INFO(turret.Severity): Informational messages.
    """

    INTERNAL_ERROR = Severity(<int>severity.kINTERNAL_ERROR)
    """An internal error has occurred. Execution is unrecoverable."""

    ERROR = Severity(<int>severity.kERROR)
    """An application error has occurred."""

    WARNING = Severity(<int>severity.kWARNING)
    """
    An application error has been discovered, but TensorRT has
    recovered or fallen back to a default.
    """

    INFO = Severity(<int>severity.kINFO)
    """Informational messages."""

    _NAME_TABLE = {
        <int>severity.kINTERNAL_ERROR: 'INTERNAL_ERROR',
        <int>severity.kERROR:          'ERROR',
        <int>severity.kWARNING:        'WARNING',
        <int>severity.kINFO:           'INFO',
    }

    def __cinit__(self, int x):
        self.thisobj = <nvinfer.Severity>x

    def __repr__(self):
        cdef int ivalue = <int>self.thisobj
        return "<{}.{}: {}>".format(
            self.__class__.__name__, self._NAME_TABLE[ivalue], ivalue)

    def __str__(self):
        return self._NAME_TABLE[<int>self.thisobj]

    def __int__(self):
        return <int>self.thisobj

    def __richcmp__(Severity x, Severity y, op):
        cdef int a = <int>x.thisobj
        cdef int b = <int>y.thisobj
        if op == 0: return a < b
        if op == 1: return a <= b
        if op == 2: return a == b
        if op == 3: return a != b
        if op == 4: return a > b
        if op == 5: return a >= b


class Logger(metaclass=ABCMeta):
    """
    Application-implemented logging interface for the builder,
    engine and runtime.
    """

    @abstractmethod
    def log(self, severity, message):
        """
        A callback implemented by the application to handle logging
        messages.

        Args:
            severity (Severity): The severity of the message.
            message (str): The log message.
        """
        raise NotImplementedError()


cdef class LoggerProxy:
    def __cinit__(self, logger):
        if not isinstance(logger, Logger):
            raise TypeError("logger must implement Logger")
        self.proxy = new NativeLoggerProxy(<ref.PyObject *>logger)

    def __dealloc__(self):
        if self.proxy:
            del self.proxy
