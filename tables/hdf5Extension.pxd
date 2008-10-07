from definitions cimport hid_t, hsize_t, ndarray


# Declaration of instance variables for shared classes
cdef class Node:
  cdef char  *name
  cdef hid_t  parent_id

cdef class Leaf(Node):
  cdef hid_t   dataset_id
  cdef hid_t   type_id
  cdef hid_t   base_type_id
  cdef hid_t   disk_type_id
  cdef hsize_t *dims     # Necessary to be here because of Leaf._g_truncate()
  cdef _get_type_ids(self)
  cdef _convertTime64(self, ndarray nparr, int sense)

cdef class Array(Leaf):
  cdef int      rank
  cdef hsize_t *maxdims
  cdef hsize_t *dims_chunk



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
