from libcpp.vector cimport vector

from .engine cimport InferenceEngine


cdef class InferenceBuffer:
    cdef InferenceEngine engine
    cdef list allocations
    cdef vector[void*] bindings
    cdef public int batch_size

    cdef void **binding_pointers(self)
