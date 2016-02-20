from numpy cimport ndarray

from .definitions cimport hid_t, hsize_t
from .hdf5extension cimport Array

cdef class CacheArray(Array):
  cdef hid_t ___NOTHING___

  cdef initread(self, int nbounds)
  cdef read_slice(self, hsize_t nrow, hsize_t start, hsize_t stop, ndarray buf)

