cdef extern from "hdf5.h":
  ctypedef int hid_t
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  # hsize_t should be unsigned, but Windows platform does not support
  # such a unsigned long long type.
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t


# Declaration of instance variables for shared classes
cdef class Node:
  cdef char  *name
  cdef hid_t  parent_id

cdef class Leaf(Node):
  cdef hid_t   dataset_id
  cdef hid_t   type_id
  cdef hid_t   base_type_id

cdef class Array(Leaf):
  cdef int     rank
  cdef hsize_t *dims
  cdef hsize_t *maxdims
  cdef hsize_t *dims_chunk
