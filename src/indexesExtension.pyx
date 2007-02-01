#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: May 18, 2006
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Pyrex interface for keeping indexes classes.

Classes (type extensions):

    IndexArray
    CacheArray
    LastRowArray

Functions:

Misc variables:

    __version__
"""

import sys
import os
import warnings
import pickle
import cPickle

import numpy

from tables.exceptions import HDF5ExtError
from hdf5Extension cimport Array
from tables.constants import \
     SORTED_MAX_SIZE, BOUNDS_MAX_SIZE, INDICES_MAX_SIZE


# numpy functions & objects
from definitions cimport \
     memcpy, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, \
     import_array, ndarray, \
     hid_t, herr_t, hsize_t, \
     H5Dget_space, H5Screate_simple, H5Sclose


from lrucacheExtension cimport NumCache


__version__ = "$Revision$"

#-------------------------------------------------------------------

# External C functions

# Functions for optimized operations with ARRAY for indexing purposes
cdef extern from "H5ARRAY-opt.h":
  herr_t H5ARRAYOinit_readSlice(hid_t dataset_id, hid_t type_id,
                                hid_t *space_id,  hid_t *mem_space_id,
                                hsize_t count)
  herr_t H5ARRAYOread_readSlice(hid_t dataset_id, hid_t space_id,
                                hid_t type_id, hsize_t irow,
                                hsize_t start, hsize_t stop,
                                void *data)
  herr_t H5ARRAYOread_index_sparse(hid_t dataset_id, hid_t space_id,
                                   hid_t type_id, hsize_t ncoords,
                                   void *coords, void *data)
  herr_t H5ARRAYOread_readSortedSlice(hid_t dataset_id, hid_t space_id,
                                      hid_t mem_space_id, hid_t type_id,
                                      hsize_t irow, hsize_t start,
                                      hsize_t stop, void *data)
  herr_t H5ARRAYOread_readBoundsSlice(hid_t dataset_id, hid_t space_id,
                                      hid_t mem_space_id, hid_t type_id,
                                      hsize_t irow, hsize_t start,
                                      hsize_t stop, void *data)
  herr_t H5ARRAYOreadSliceLR(hid_t dataset_id, hsize_t start,
                             hsize_t stop, void *data)

# Functions for optimized operations for dealing with indexes
cdef extern from "idx-opt.h":
  int bisect_left_d(double *a, double x, int hi, int offset)
  int bisect_left_i(int *a, int x, int hi, int offset)
  int bisect_left_ll(long long *a, long long x, int hi, int offset)
  int bisect_right_d(double *a, double x, int hi, int offset)
  int bisect_right_i(int *a, int x, int hi, int offset)
  int bisect_right_ll(long long *a, long long x, int hi, int offset)
  int get_sorted_indices(int nrows, long long *rbufC,
                         int *rbufst, int *rbufln, int ssize)
  int convert_addr64(int nrows, int nelem, long long *rbufA,
                     int *rbufR, int *rbufln)



#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#---------------------------------------------------------------------------

# Functions


# Classes


cdef class Index:
  pass


cdef class CacheArray(Array):
  """Container for keeping index caches of 1st and 2nd level."""
  cdef hid_t space_id
  cdef hid_t mem_space_id


  cdef initRead(self, int nbounds):
    "Actions to accelerate the reads afterwards."

    # Precompute the space_id and mem_space_id
    if (H5ARRAYOinit_readSlice(self.dataset_id, self.type_id,
                               &self.space_id, &self.mem_space_id,
                               nbounds) < 0):
      raise HDF5ExtError("Problems initializing the bounds array data.")
    return


  cdef readSlice(self, hsize_t nrow, hsize_t start, hsize_t stop, void *rbuf):
    "Read an slice of bounds."

    if (H5ARRAYOread_readBoundsSlice(self.dataset_id, self.space_id,
                                     self.mem_space_id, self.type_id,
                                     nrow, start, stop, rbuf) < 0):
      raise HDF5ExtError("Problems reading the bounds array data.")
    return


  def _g_close(self):
    super(Array, self)._g_close()
    # Release specific resources of this class
    if self.space_id > 0:
      H5Sclose(self.space_id)
    if self.mem_space_id > 0:
      H5Sclose(self.mem_space_id)



cdef class IndexArray(Array):
  """Container for keeping sorted and indices values."""
  cdef void    *rbufst, *rbufln, *rbufrv, *rbufbc, *rbuflb
  cdef void    *rbufC, *rbufA
  cdef hid_t   space_id, mem_space_id
  cdef int     l_chunksize, l_slicesize, nbounds
  cdef CacheArray bounds_ext
  cdef NumCache boundscache, sortedcache, indicescache
  cdef ndarray arrAbs, coords, bufferbc, bufferlb


  def _initIndexSlice(self, index, ncoords):
    "Initialize the structures for doing a binary search"
    cdef long buflen
    cdef ndarray starts, lengths
    cdef object maxslots

    # Create buffers for reading reverse index data
    if <object>self.arrAbs is None or len(self.arrAbs) < ncoords:
      self.coords = numpy.empty(dtype=numpy.int64, shape=ncoords)
      self.arrAbs = numpy.empty(dtype=numpy.int64, shape=ncoords)
      # Get the pointers to the buffer data area
      self.rbufC = self.coords.data
      self.rbufA = self.arrAbs.data
      # Access starts and lengths in parent index.
      # This sharing method should be improved.
      starts = <ndarray>index.starts;  lengths = <ndarray>index.lengths
      self.rbufst = starts.data;  self.rbufln = lengths.data
      if not self.space_id:
        # Initialize the index array for reading
        self.space_id = H5Dget_space(self.dataset_id )
      # cache some counters in local extension variables
      # nrows cannot be cached because it can grow!
      self.l_slicesize = index.slicesize
      self.l_chunksize = index.chunksize
      if <object>self.indicescache is None:
        # Define a LRU cache for indices
        maxslots = INDICES_MAX_SIZE / (self.l_chunksize*8)
        self.indicescache = <NumCache>NumCache(
          shape=(maxslots, 1), itemsize=8, name="indices")


  cdef _readIndex(self, hsize_t irow, hsize_t start, hsize_t stop,
                  int offsetl):
    cdef herr_t ret
    cdef long long *rbufA

    # Correct the start of the buffer with offsetl
    rbufA = <long long *>self.rbufA + offsetl
    # Do the physical read
    ##Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOread_readSlice(self.dataset_id, self.space_id, self.type_id,
                                 irow, start, stop, rbufA)
    ##Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    return


  # The next method is able to read data from last row if necessary
  cdef _readIndex_single(self, hsize_t coord, int relcoord):
    cdef herr_t ret
    cdef hsize_t irow, icol
    cdef long long *rbufA

    irow = coord / self.l_slicesize
    icol = coord - irow * self.l_slicesize
    # Do the physical read
    rbufA = <long long *>self.rbufA + relcoord
    if irow < self.nrows:
      # Py_BEGIN_ALLOW_THREADS
      ret = H5ARRAYOread_readSlice(self.dataset_id, self.space_id,
                                   self.type_id, irow, icol, icol+1, rbufA)
      # Py_END_ALLOW_THREADS
      if ret < 0:
        raise HDF5ExtError("_readIndex_single: Problems reading the indices.")
    else:
      self._v_parent.indicesLR._readIndexSlice(self, icol, icol+1, relcoord)

    return


  def _initSortedSlice(self, index):
    "Initialize the structures for doing a binary search."
    cdef long ndims
    cdef int  rank, buflen, cachesize
    cdef char *bname
    cdef hsize_t count[2]
    cdef ndarray starts, lengths, rvcache
    cdef object maxslots, rowsize

    dtype = self.dtype
    # Create the buffer for reading sorted data chunks if not created yet
    if <object>self.bufferlb is None:
      self.bufferlb = numpy.empty(dtype=dtype, shape=self.chunksize)
      # Internal buffers
      # Get the pointers to the different buffer data areas
      self.rbuflb = self.bufferlb.data
      # Init structures for accelerating sorted array reads
      self.space_id = H5Dget_space(self.dataset_id)
      rank = 2
      count[0] = 1; count[1] = self.chunksize;
      self.mem_space_id = H5Screate_simple(rank, count, NULL)
      # cache some counters in local extension variables
      self.l_slicesize = index.slicesize
      self.l_chunksize = index.chunksize

    # Get the addresses of buffer data
    starts = index.starts;  lengths = index.lengths
    self.rbufst = starts.data
    self.rbufln = lengths.data
    # The 1st cache is loaded completely in memory and needs to be reloaded
    rvcache = index.ranges[:]
    self.rbufrv = rvcache.data
    index.rvcache = <object>rvcache
    # Init the bounds array for reading
    self.nbounds = index.bounds.shape[1]
    self.bounds_ext = <CacheArray>index.bounds
    self.bounds_ext.initRead(self.nbounds)
    if str(dtype) in self._v_parent.opt_search_types:
      # The next caches should be defined only for optimized search types.
      # The 2nd level cache will replace the already existing ObjectCache and
      # already bound to the boundscache attribute. This way, the cache will
      # not be duplicated (I know, this smells badly, but anyway).
      rowsize = (self.bounds_ext._v_chunkshape[1] * dtype.itemsize)
      maxslots = BOUNDS_MAX_SIZE / rowsize
      self.boundscache = <NumCache>NumCache(
        (maxslots, self.nbounds), dtype.itemsize, 'non-opt types bounds')
      self.bufferbc = numpy.empty(dtype=dtype, shape=self.nbounds)
      # Get the pointer for the internal buffer for 2nd level cache
      self.rbufbc = self.bufferbc.data
      # Another NumCache for the sorted values
      rowsize = (self.chunksize*dtype.itemsize)
      maxslots = SORTED_MAX_SIZE / (self.chunksize*dtype.itemsize)
      self.sortedcache = <NumCache>NumCache(
        (maxslots, self.chunksize), dtype.itemsize, 'sorted')


  cdef void *_g_readSortedSlice(self, hsize_t irow, hsize_t start, hsize_t stop):
    "Read the sorted part of an index."

    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       irow, start, stop, self.rbuflb)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    return self.rbuflb


  # This is callable from python
  def _readSortedSlice(self, hsize_t irow, hsize_t start, hsize_t stop):
    "Read the sorted part of an index."

    self._g_readSortedSlice(irow, start, stop)
    return self.bufferlb


# This has been copied from the standard module bisect.
# Checks for the values out of limits has been added at the beginning
# because I forsee that this should be a very common case.
# 2004-05-20
  def _bisect_left(self, a, x, int hi):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, i points just
    before the leftmost x already there.

    """
    cdef int lo, mid

    lo = 0
    if x <= a[0]: return 0
    if a[-1] < x: return hi
    while lo < hi:
        mid = (lo+hi)/2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo


  # This accelerates quite a bit (~25%) respect to _bisect_left
  # Besides, it can manage general python objects
  cdef _bisect_left_optim(self, a, x, int hi, int stride):
    cdef int lo, mid

    lo = 0
    if x <= getPythonScalar(a, 0): return 0
    if getPythonScalar(a, (hi-1)*stride) < x: return hi
    while lo < hi:
        mid = (lo+hi)/2
        if getPythonScalar(a, mid*stride) < x: lo = mid+1
        else: hi = mid
    return lo


  def _bisect_right(self, a, x, int hi):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, i points just
    beyond the rightmost x already there.

    """
    cdef int lo, mid

    lo = 0
    if x < a[0]: return 0
    if a[-1] <= x: return hi
    while lo < hi:
      mid = (lo+hi)/2
      if x < a[mid]: hi = mid
      else: lo = mid+1
    return lo


  # This accelerates quite a bit (~25%) respect to _bisect_right
  # Besides, it can manage general python objects
  cdef _bisect_right_optim(self, a, x, int hi, int stride):
    cdef int lo, mid

    lo = 0
    if x < getPythonScalar(a, 0): return 0
    if getPythonScalar(a, (hi-1)*stride) <= x: return hi
    while lo < hi:
      mid = (lo+hi)/2
      if x < getPythonScalar(a, mid*stride): hi = mid
      else: lo = mid+1
    return lo


  # Get the bounds from the cache, or read them
  cdef void *getLRUbounds(self, int nrow, int nbounds):
    cdef void *vpointer
    cdef long nslot

    nslot = self.boundscache.getslot(nrow)
    if nslot >= 0:
      vpointer = self.boundscache.getitem(nslot)
    else:
      # Bounds row is not in cache. Read it and put it in the LRU cache.
      self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
      self.boundscache.setitem(nrow, self.rbufbc, 0)
      vpointer = self.rbufbc
    return vpointer


  # Get the sorted row from the cache or read it.
  cdef void *getLRUsorted(self, int nrow, int ncs, int nchunk, int cs):
    cdef void *vpointer
    cdef long long nckey
    cdef long nslot

    # Compute the number of chunk read and use it as the key for the cache.
    nckey = nrow*ncs+nchunk
    nslot = self.sortedcache.getslot(nckey)
    if nslot >= 0:
      vpointer = self.sortedcache.getitem(nslot)
    else:
      # The sorted chunk is not in cache. Read it and put it in the LRU cache.
      vpointer = self._g_readSortedSlice(nrow, cs*nchunk, cs*(nchunk+1))
      self.sortedcache.setitem(nckey, vpointer, 0)
    return vpointer


  # Optimized version for doubles
  def _searchBinNA_d(self, double item1, double item2):
    cdef int cs, ss, ncs, nrow, nrows, nrow2, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst, *rbufln
    # Variables with specific type
    cdef double *rbufrv, *rbufbc, *rbuflb

    cs = self.l_chunksize;  ss = self.l_slicesize;  ncs = ss / cs
    nbounds = self.nbounds;  nrows = self.nrows;  tlength = 0
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    # Limits not in cache, do a lookup
    rbufrv = <double *>self.rbufrv
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2;  bread = 0;  nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <double *>self.getLRUbounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_d(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <double *>self.getLRUsorted(nrow, ncs, nchunk, cs)
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
            rbufbc = <double *>self.getLRUbounds(nrow, nbounds)
          nchunk2 = bisect_right_d(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <double *>self.getLRUsorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_d(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start;  tlength = tlength + length
      rbufst[nrow] = start;  rbufln[nrow] = length;
    return tlength


  # Optimized version for ints
  def _searchBinNA_i(self, double item1, double item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst, *rbufln
    # Variables with specific type
    cdef int *rbufrv, *rbufbc, *rbuflb

    cs = self.l_chunksize;  ss = self.l_slicesize; ncs = ss / cs
    nbounds = self.nbounds;  nrows = self.nrows
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufrv = <int *>self.rbufrv; tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2;  bread = 0;  nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <int *>self.getLRUbounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_i(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <int *>self.getLRUsorted(nrow, ncs, nchunk, cs)
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
            rbufbc = <int *>self.getLRUbounds(nrow, nbounds)
          nchunk2 = bisect_right_i(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <int *>self.getLRUsorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_i(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start;  tlength = tlength + length
      rbufst[nrow] = start;  rbufln[nrow] = length;
    return tlength


  # Optimized version for long long
  def _searchBinNA_ll(self, double item1, double item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlength, length, bread, nchunk, nchunk2
    cdef int *rbufst, *rbufln
    # Variables with specific type
    cdef long long *rbufrv, *rbufbc, *rbuflb

    cs = self.l_chunksize;  ss = self.l_slicesize; ncs = ss / cs
    nbounds = self.nbounds;  nrows = self.nrows
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufrv = <long long *>self.rbufrv; tlength = 0
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2;  bread = 0;  nchunk = -1
      # Look if item1 is in this row
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row from the LRU cache or read them.
          rbufbc = <long long *>self.getLRUbounds(nrow, nbounds)
          bread = 1
          nchunk = bisect_left_ll(rbufbc, item1, nbounds, 0)
          # Get the sorted row from the LRU cache or read it.
          rbuflb = <long long *>self.getLRUsorted(nrow, ncs, nchunk, cs)
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
            rbufbc = <long long *>self.getLRUbounds(nrow, nbounds)
          nchunk2 = bisect_right_ll(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            # Get the sorted row from the LRU cache or read it.
            rbuflb = <long long *>self.getLRUsorted(nrow, ncs, nchunk2, cs)
          stop = bisect_right_ll(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = ss
      else:
        stop = 0
      length = stop - start;  tlength = tlength + length
      rbufst[nrow] = start;  rbufln[nrow] = length;
    return tlength


  # This version of getCoords reads the indexes in chunks.
  # Because of that, it can be used in iterators.
  def _getCoords(self, index, int startcoords, int ncoords):
    cdef int nrow, nrows, leni, len1, len2, nidxelem
    cdef int relcoord, bcoords
    cdef int startl, stopl, incr, stop
    cdef int *rbufst, *rbufln
    cdef long long coord
    cdef long nslot

    len1 = 0; len2 = 0; bcoords = 0
    # Correction against asking too many elements
    nidxelem = index.nelements
    if startcoords + ncoords > nidxelem:
      ncoords = nidxelem - startcoords
    # create buffers for indices
    self._initIndexSlice(index, ncoords)
    arrAbs = self.arrAbs
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    nrows = index.nrows
    for nrow from 0 <= nrow < nrows:
      leni = rbufln[nrow]; len2 = len2 + leni
      if (leni > 0 and len1 <= startcoords < len2):
        startl = rbufst[nrow] + (startcoords-len1)
        # Read ncoords as maximum
        stopl = startl + ncoords
        # Correction if stopl exceeds the limits
        if stopl > rbufst[nrow] + rbufln[nrow]:
          stopl = rbufst[nrow] + rbufln[nrow]
        # Use the cache for reading reverse coordinates
        for relcoord from 0 <= relcoord < stopl-startl:
          coord = nrow * self.l_slicesize + startl + relcoord
          nslot = self.indicescache.getslot(coord)
          if nslot >= 0:
            self.indicescache.getitem2(nslot, self.rbufA, bcoords+relcoord)
          else:
            # The coord is not in cache. Read it and put it in the LRU cache.
            self._readIndex_single(coord, bcoords+relcoord)
            self.indicescache.setitem(coord, self.rbufA, bcoords+relcoord)
        incr = stopl - startl
        bcoords = bcoords + incr
        startcoords = startcoords + incr
        ncoords = ncoords - incr
        if ncoords == 0:
          break
      len1 = len1 + leni
    return arrAbs[:bcoords]


  # This version of getCoords reads all the indexes in one pass.
  # Because of that, it is not meant to be used on iterators.
  # This is aproximately a 25% faster than _getCoords above.
  # If there is a last row with interesting values on it, this has been
  # optimised as well.
  def _getCoords_sparse(self, index, int ncoords):
    cdef int nrow, nrows, startl, stopl, lenl, relcoord
    cdef int *rbufst, *rbufln
    cdef long long *rbufC, *rbufA
    cdef long long coord
    cdef object nckey
    cdef long nslot

    nrows = self._v_parent.nrows  # Get the nrows of Index!
    # Initialize the index dataset
    self._initIndexSlice(index, ncoords)
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufC = <long long *>self.rbufC
    rbufA = <long long *>self.rbufA

    # Get the sorted indices
    get_sorted_indices(nrows, rbufC, rbufst, rbufln, self.l_slicesize)

    # Retrieve the reverse coordinates
    for relcoord from 0 <= relcoord < ncoords:
      coord = rbufC[relcoord]
      # Look at the cache for this coord
      nslot = self.indicescache.getslot(coord)
      if nslot >= 0:
        self.indicescache.getitem2(nslot, self.rbufA, relcoord)
      else:
        # The coord is not in cache. Read it and put it in the LRU cache.
        self._readIndex_single(coord, relcoord) # Puts result in self.rbufA
        self.indicescache.setitem(coord, self.rbufA, relcoord)

    # Return ncoords as maximum because arrAbs can have more elements
    return self.arrAbs[:ncoords]


  def _g_close(self):
    super(Array, self)._g_close()
    # Release specific resources of this class
    if self.space_id > 0:
      H5Sclose(self.space_id)
    if self.mem_space_id > 0:
      H5Sclose(self.mem_space_id)



cdef class LastRowArray(Array):
  """Container for keeping sorted and indices values of last rows of an index."""


  def _readIndexSlice(self, IndexArray indices, hsize_t start, hsize_t stop,
                      int offsetl):
    "Read the reverse index part of an LR index."
    cdef long long *rbufA

    rbufA = <long long *>indices.rbufA + offsetl
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOreadSliceLR(self.dataset_id, start, stop, rbufA)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the index data in Last Row.")

    return


  def _readSortedSlice(self, IndexArray sorted,
                              hsize_t start, hsize_t stop):
    "Read the sorted part of an LR index."
    cdef void  *rbuflb

    rbuflb = sorted.rbuflb  # direct access to rbuflb: very fast.
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOreadSliceLR(self.dataset_id, start, stop, rbuflb)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the index data.")

    return sorted.bufferlb



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
