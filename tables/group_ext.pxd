from .definitions cimport hid_t
from .hdf5extension cimport Node

cdef class Group(Node):
    cdef hid_t group_id
