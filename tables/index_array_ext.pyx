from .definitions cimport hsize_t, herr_t, H5Screate_simple, H5Sclose
from .exceptions import HDF5ExtError

# Types, constants, functions, classes & other objects from everywhere
from numpy cimport import_array, ndarray, \
    npy_int8, npy_int16, npy_int32, npy_int64, \
    npy_uint8, npy_uint16, npy_uint32, npy_uint64, \
    npy_float32, npy_float64, \
    npy_float, npy_double, npy_longdouble

import numpy as np
cimport numpy as np

# These two types are defined in npy_common.h but not in cython's numpy.pxd
ctypedef unsigned char npy_bool
ctypedef npy_uint16 npy_float16


# Functions for optimized operations for dealing with indexes
cdef extern from "idx-opt.h" nogil:
  int bisect_left_b(npy_int8 *a, long x, int hi, int offset)
  int bisect_left_ub(npy_uint8 *a, long x, int hi, int offset)
  int bisect_right_b(npy_int8 *a, long x, int hi, int offset)
  int bisect_right_ub(npy_uint8 *a, long x, int hi, int offset)
  int bisect_left_s(npy_int16 *a, long x, int hi, int offset)
  int bisect_left_us(npy_uint16 *a, long x, int hi, int offset)
  int bisect_right_s(npy_int16 *a, long x, int hi, int offset)
  int bisect_right_us(npy_uint16 *a, long x, int hi, int offset)
  int bisect_left_i(npy_int32 *a, long x, int hi, int offset)
  int bisect_left_ui(npy_uint32 *a, npy_uint32 x, int hi, int offset)
  int bisect_right_i(npy_int32 *a, long x, int hi, int offset)
  int bisect_right_ui(npy_uint32 *a, npy_uint32 x, int hi, int offset)
  int bisect_left_ll(npy_int64 *a, npy_int64 x, int hi, int offset)
  int bisect_left_ull(npy_uint64 *a, npy_uint64 x, int hi, int offset)
  int bisect_right_ll(npy_int64 *a, npy_int64 x, int hi, int offset)
  int bisect_right_ull(npy_uint64 *a, npy_uint64 x, int hi, int offset)
  int bisect_left_e(npy_float16 *a, npy_float64 x, int hi, int offset)
  int bisect_right_e(npy_float16 *a, npy_float64 x, int hi, int offset)
  int bisect_left_f(npy_float32 *a, npy_float64 x, int hi, int offset)
  int bisect_right_f(npy_float32 *a, npy_float64 x, int hi, int offset)
  int bisect_left_d(npy_float64 *a, npy_float64 x, int hi, int offset)
  int bisect_right_d(npy_float64 *a, npy_float64 x, int hi, int offset)
  int bisect_left_g(npy_longdouble *a, npy_longdouble x, int hi, int offset)
  int bisect_right_g(npy_longdouble *a, npy_longdouble x, int hi, int offset)


# Functions for optimized operations with ARRAY for indexing purposes
cdef extern from "H5ARRAY-opt.h" nogil:
  herr_t H5ARRAYOread_readSlice(
    hid_t dataset_id, hid_t type_id,
    hsize_t irow, hsize_t start, hsize_t stop, void *data)
  herr_t H5ARRAYOread_readSortedSlice(
    hid_t dataset_id, hid_t mem_space_id, hid_t type_id,
    hsize_t irow, hsize_t start, hsize_t stop, void *data)


cdef class IndexArray(Array):
  """Container for keeping sorted and indices values."""

  def _read_index_slice(self, hsize_t irow, hsize_t start, hsize_t stop,
                      ndarray idx):
    cdef herr_t ret

    # Do the physical read
    with nogil:
        ret = H5ARRAYOread_readSlice(self.dataset_id, self.type_id,
                                     irow, start, stop, idx.data)

    if ret < 0:
      raise HDF5ExtError("Problems reading the index indices.")


  def _init_sorted_slice(self, index):
    """Initialize the structures for doing a binary search."""

    cdef long ndims
    cdef int  rank, buflen, cachesize
    cdef char *bname
    cdef hsize_t count[2]
    cdef ndarray starts, lengths, rvcache
    cdef object maxslots, rowsize

    dtype = self.atom.dtype
    # Create the buffer for reading sorted data chunks if not created yet
    if <object>self.bufferlb is None:
      # Internal buffers
      self.bufferlb = np.empty(dtype=dtype, shape=self.chunksize)
      # Get the pointers to the different buffer data areas
      self.rbuflb = self.bufferlb.data
      # Init structures for accelerating sorted array reads
      rank = 2
      count[0] = 1
      count[1] = self.chunksize
      self.mem_space_id = H5Screate_simple(rank, count, NULL)
      # Cache some counters in local extension variables
      self.l_chunksize = self.chunksize
      self.l_slicesize = self.slicesize

    # Get the addresses of buffer data
    starts = index.starts
    lengths = index.lengths
    self.rbufst = starts.data
    self.rbufln = lengths.data
    # The 1st cache is loaded completely in memory and needs to be reloaded
    rvcache = index.ranges[:]
    self.rbufrv = rvcache.data
    index.rvcache = <object>rvcache
    # Init the bounds array for reading
    self.nbounds = index.bounds.shape[1]
    self.bounds_ext = <CacheArray>index.bounds
    self.bounds_ext.initread(self.nbounds)
    if str(dtype) in self._v_parent.opt_search_types:
      # The next caches should be defined only for optimized search types.
      # The 2nd level cache will replace the already existing ObjectCache and
      # already bound to the boundscache attribute. This way, the cache will
      # not be duplicated (I know, this smells badly, but anyway).
      params = self._v_file.params
      rowsize = (self.bounds_ext._v_chunkshape[1] * dtype.itemsize)
      maxslots = params['BOUNDS_MAX_SIZE'] / rowsize
      self.boundscache = <NumCache>NumCache(
        (maxslots, self.nbounds), dtype, 'non-opt types bounds')
      self.bufferbc = np.empty(dtype=dtype, shape=self.nbounds)
      # Get the pointer for the internal buffer for 2nd level cache
      self.rbufbc = self.bufferbc.data
      # Another NumCache for the sorted values
      rowsize = (self.chunksize*dtype.itemsize)
      maxslots = params['SORTED_MAX_SIZE'] / (self.chunksize*dtype.itemsize)
      self.sortedcache = <NumCache>NumCache(
        (maxslots, self.chunksize), dtype, 'sorted')


  cdef void *_g_read_sorted_slice(self, hsize_t irow, hsize_t start, hsize_t stop):
    """Read the sorted part of an index."""

    with nogil:
        ret = H5ARRAYOread_readSortedSlice(
          self.dataset_id, self.mem_space_id, self.type_id,
          irow, start, stop, self.rbuflb)

    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    return self.rbuflb


  # This is callable from python
  def _read_sorted_slice(self, hsize_t irow, hsize_t start, hsize_t stop):
    """Read the sorted part of an index."""

    self._g_read_sorted_slice(irow, start, stop)
    return self.bufferlb


  cdef void *get_lru_bounds(self, int nrow, int nbounds):
    """Get the bounds from the cache, or read them."""

    cdef void *vpointer
    cdef long nslot

    nslot = self.boundscache.getslot_(nrow)
    if nslot >= 0:
      vpointer = self.boundscache.getitem1_(nslot)
    else:
      # Bounds row is not in cache. Read it and put it in the LRU cache.
      self.bounds_ext.read_slice(nrow, 0, nbounds, self.rbufbc)
      self.boundscache.setitem_(nrow, self.rbufbc, 0)
      vpointer = self.rbufbc
    return vpointer

  # can't time machine since get_lru_bounds() function is cdef'd

  cdef void *get_lru_sorted(self, int nrow, int ncs, int nchunk, int cs):
    """Get the sorted row from the cache or read it."""

    cdef void *vpointer
    cdef npy_int64 nckey
    cdef long nslot
    cdef hsize_t start, stop

    # Compute the number of chunk read and use it as the key for the cache.
    nckey = nrow*ncs+nchunk
    nslot = self.sortedcache.getslot_(nckey)
    if nslot >= 0:
      vpointer = self.sortedcache.getitem1_(nslot)
    else:
      # The sorted chunk is not in cache. Read it and put it in the LRU cache.
      start = cs*nchunk
      stop = cs*(nchunk+1)
      vpointer = self._g_read_sorted_slice(nrow, start, stop)
      self.sortedcache.setitem_(nckey, vpointer, 0)
    return vpointer

  # can't time machine since get_lru_sorted() function is cdef'd

  # Optimized version for int8
  def _search_bin_na_b(self, long item1, long item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_int8 *rbufrv
    cdef npy_int8 *rbufbc = NULL
    cdef npy_int8 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_int8 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1

      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_int8 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_b(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_int8 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_b(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_int8 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_b(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_int8 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_b(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for uint8
  def _search_bin_na_ub(self, long item1, long item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_uint8 *rbufrv
    cdef npy_uint8 *rbufbc = NULL
    cdef npy_uint8 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_uint8 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1

      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_uint8 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_ub(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_uint8 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_ub(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_uint8 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_ub(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_uint8 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_ub(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for int16
  def _search_bin_na_s(self, long item1, long item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_int16 *rbufrv
    cdef npy_int16 *rbufbc = NULL
    cdef npy_int16 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_int16 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_int16 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_s(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_int16 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_s(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_int16 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_s(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_int16 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_s(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for uint16
  def _search_bin_na_us(self, long item1, long item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_uint16 *rbufrv
    cdef npy_uint16 *rbufbc = NULL
    cdef npy_uint16 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_uint16 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_uint16 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_us(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_uint16 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_us(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_uint16 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_us(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_uint16 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_us(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for int32
  def _search_bin_na_i(self, long item1, long item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_int32 *rbufrv
    cdef npy_int32 *rbufbc = NULL
    cdef npy_int32 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_int32 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_int32 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_i(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_int32 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_i(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_int32 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_i(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_int32 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_i(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for uint32
  def _search_bin_na_ui(self, npy_uint32 item1, npy_uint32 item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_uint32 *rbufrv
    cdef npy_uint32 *rbufbc = NULL
    cdef npy_uint32 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_uint32 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_uint32 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_ui(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_uint32 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_ui(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_uint32 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_ui(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_uint32 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_ui(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for int64
  def _search_bin_na_ll(self, npy_int64 item1, npy_int64 item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_int64 *rbufrv
    cdef npy_int64 *rbufbc = NULL
    cdef npy_int64 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_int64 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_int64 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_ll(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_int64 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_ll(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_int64 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_ll(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_int64 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_ll(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for uint64
  def _search_bin_na_ull(self, npy_uint64 item1, npy_uint64 item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_uint64 *rbufrv
    cdef npy_uint64 *rbufbc = NULL
    cdef npy_uint64 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufrv = <npy_uint64 *>self.rbufrv
    tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_uint64 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_ull(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_uint64 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_ull(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_uint64 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_ull(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_uint64 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_ull(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for float16
  def _search_bin_na_e(self, npy_float64 item1, npy_float64 item2):
    cdef int cs, ss, ncs, nrow, nrows, nrow2, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_float16 *rbufrv
    cdef npy_float16 *rbufbc = NULL
    cdef npy_float16 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    tlength = 0
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    # Limits not in cache, do a lookup
    rbufrv = <npy_float16 *>self.rbufrv
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1

      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_float16 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_e(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_float16 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_e(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_float16 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_e(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_float16 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_e(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for float32
  def _search_bin_na_f(self, npy_float64 item1, npy_float64 item2):
    cdef int cs, ss, ncs, nrow, nrows, nrow2, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln
    # Variables with specific type
    cdef npy_float32 *rbufrv
    cdef npy_float32 *rbufbc = NULL
    cdef npy_float32 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    tlength = 0
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln

    # Limits not in cache, do a lookup
    rbufrv = <npy_float32 *>self.rbufrv
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_float32 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_f(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_float32 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_f(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_float32 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_f(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_float32 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_f(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for float64
  def _search_bin_na_d(self, npy_float64 item1, npy_float64 item2):
    cdef int cs, ss, ncs, nrow, nrows, nrow2, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_float64 *rbufrv
    cdef npy_float64 *rbufbc = NULL
    cdef npy_float64 *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    tlength = 0
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln

    # Limits not in cache, do a lookup
    rbufrv = <npy_float64 *>self.rbufrv
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1

      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_float64 *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_d(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_float64 *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_d(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_float64 *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_d(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_float64 *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_d(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  # Optimized version for npy_longdouble/float96/float128
  def _search_bin_na_g(self, npy_longdouble item1, npy_longdouble item2):
    cdef int cs, ss, ncs, nrow, nrows, nrow2, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst
    cdef int *rbufln

    # Variables with specific type
    cdef npy_longdouble *rbufrv
    cdef npy_longdouble *rbufbc = NULL
    cdef npy_longdouble *rbuflb = NULL

    cs = self.l_chunksize
    ss = self.l_slicesize
    ncs = ss / cs
    nbounds = self.nbounds
    nrows = self.nrows
    tlength = 0
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln

    # Limits not in cache, do a lookup
    rbufrv = <npy_longdouble *>self.rbufrv
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      nchunk = -1

      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <npy_longdouble *>self.get_lru_bounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_g(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <npy_longdouble *>self.get_lru_sorted(nrow, ncs, nchunk, cs)
          start = bisect_left_g(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = ss
      else:
        start = 0
      # Now, for item2
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            # Get the bounds row from the LRU cache or read them.
            rbufbc = <npy_longdouble *>self.get_lru_bounds(nrow, nbounds)
          nchunk2 = bisect_right_g(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <npy_longdouble *>self.get_lru_sorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_g(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start
      tlength = tlength + length
      rbufst[nrow] = start
      rbufln[nrow] = length
    return tlength


  def _g_close(self):
    super(Array, self)._g_close()
    # Release specific resources of this class
    if self.mem_space_id > 0:
      H5Sclose(self.mem_space_id)


