from numpy cimport ndarray

from .definitions cimport hid_t, hsize_t
from .hdf5extension cimport Array
from .cache_array_ext cimport CacheArray
from .lrucacheextension cimport NumCache

cdef class IndexArray(Array):
  cdef void    *rbufst
  cdef void    *rbufln
  cdef void    *rbufrv
  cdef void    *rbufbc
  cdef void    *rbuflb
  cdef hid_t   mem_space_id
  cdef int     l_chunksize, l_slicesize, nbounds, indsize
  cdef CacheArray bounds_ext
  cdef NumCache boundscache, sortedcache
  cdef ndarray bufferbc, bufferlb

  cdef void *_g_read_sorted_slice(self, hsize_t irow, hsize_t start, hsize_t stop)
  cdef void *get_lru_bounds(self, int nrow, int nbounds)
  cdef void *get_lru_sorted(self, int nrow, int ncs, int nchunk, int cs)
