from .definitions cimport hsize_t
from .hdf5extension cimport Leaf

cdef class VLArray(Leaf):
    cdef hsize_t nrecords

