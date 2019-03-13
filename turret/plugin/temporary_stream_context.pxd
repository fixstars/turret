from cpython cimport ref

cdef extern from "cuda.h":
    ctypedef void *CUstream
    ctypedef void *CUevent
    ctypedef int CUresult

    # TODO add __stdcall for Windows
    CUresult cuStreamAddCallback(
            CUstream, void (*)(CUstream, CUresult, void *),
            void *, unsigned int)
    CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned int)
    CUresult cuEventCreate(CUevent *, unsigned int)
    CUresult cuEventDestroy(CUevent)
    CUresult cuEventRecord(CUevent, CUstream)


cdef extern from "temporary_stream_context_deleter.hpp" namespace "turret":
    cdef cppclass TemporaryStreamContextDeleter:
        TemporaryStreamContextDeleter()
        TemporaryStreamContextDeleter(CUevent, CUevent)

    cdef void release_temporary_stream_events(CUstream, CUresult, void *)
