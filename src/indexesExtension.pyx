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

import numarray
from numarray import records, strings, memory

from tables.exceptions import HDF5ExtError
from hdf5Extension cimport Array, hid_t, herr_t, hsize_t
from constants import SORTED_CACHE_SIZE, BOUNDS_CACHE_SIZE, INDICES_CACHE_SIZE

from definitions cimport import_libnumarray, NA_getPythonScalar, \
     NA_getBufferPtrAndSize, Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, \
     PyInt_AsLong, PyLong_AsLongLong



__version__ = "$Revision$"

#-------------------------------------------------------------------


# C functions and variable declaration from its headers

# Standard C functions.
cdef extern from "stdlib.h":
  ctypedef long size_t

cdef extern from "string.h":
  void *memcpy(void *dest, void *src, size_t n)

cdef extern from "Python.h":
  object PyDict_GetItem(object p, object key)
  int PyDict_Contains(object p, object key)
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)

# Functions, structs and types from HDF5
cdef extern from "hdf5.h":
  hid_t H5Dget_space (hid_t dset_id)
  hid_t H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[])
  herr_t H5Sclose(hid_t space_id)


# Functions for optimized operations for ARRAY
cdef extern from "H5ARRAY-opt.h":

  herr_t H5ARRAYOinit_readSlice( hid_t dataset_id,
                                 hid_t type_id,
                                 hid_t *space_id,
                                 hid_t *mem_space_id,
                                 hsize_t count )

  herr_t H5ARRAYOread_readSlice( hid_t dataset_id,
                                 hid_t space_id,
                                 hid_t type_id,
                                 hsize_t irow,
                                 hsize_t start,
                                 hsize_t stop,
                                 void *data )

  herr_t H5ARRAYOread_index_sparse( hid_t dataset_id,
                                    hid_t space_id,
                                    hid_t type_id,
                                    hsize_t ncoords,
                                    void *coords,
                                    void *data )

  herr_t H5ARRAYOread_readSortedSlice( hid_t dataset_id,
                                       hid_t space_id,
                                       hid_t mem_space_id,
                                       hid_t type_id,
                                       hsize_t irow,
                                       hsize_t start,
                                       hsize_t stop,
                                       void *data )


  herr_t H5ARRAYOread_readBoundsSlice( hid_t dataset_id,
                                       hid_t space_id,
                                       hid_t mem_space_id,
                                       hid_t type_id,
                                       hsize_t irow,
                                       hsize_t start,
                                       hsize_t stop,
                                       void *data )

  herr_t H5ARRAYOreadSliceLR( hid_t dataset_id,
                              hsize_t start,
                              hsize_t stop,
                              void *data )


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

# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()

#---------------------------------------------------------------------------

# Functions

cdef getPythonScalar(object a, long i):
  return NA_getPythonScalar(a, i)


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


cdef class NumCache  # Forward declaration

cdef class IndexArray(Array):
  """Container for keeping sorted and indices values."""
  cdef void    *rbufst, *rbufln, *rbufrv, *rbufbc, *rbuflb
  cdef void    *rbufC, *rbufA
  cdef hid_t   space_id, mem_space_id
  cdef int     l_chunksize, l_slicesize, l_nrows, nbounds
  cdef object  bufferbc, bufferlb
  cdef CacheArray bounds_ext
  cdef NumCache boundscache, sortedcache, indicescache


  def _initIndexSlice(self, index, ncoords):
    "Initialize the structures for doing a binary search"
    cdef long buflen

    # Create buffers for reading reverse index data
    if self.arrAbs is None or len(self.arrAbs) < ncoords:
      #print "_initIndexSlice"
      self.coords = numarray.zeros(type="Int64", shape=ncoords)
      self.arrAbs = numarray.zeros(type="Int64", shape=ncoords)
      # Get the pointers to the buffer data area
      NA_getBufferPtrAndSize(self.coords._data, 1, &self.rbufC)
      NA_getBufferPtrAndSize(self.arrAbs._data, 1, &self.rbufA)
      # Access starts and lengths in parent index.
      # This sharing method should be improved.
      NA_getBufferPtrAndSize(index.starts._data, 1, &self.rbufst)
      NA_getBufferPtrAndSize(index.lengths._data, 1, &self.rbufln)
      if not self.space_id:
        # Initialize the index array for reading
        self.space_id = H5Dget_space(self.dataset_id )
      # cache some counters in local extension variables
      self.l_nrows = self.nrows
      self.l_slicesize = index.slicesize
      self.l_chunksize = index.chunksize
      if index.is_pro:  #XYX it is necessary to check against cache dirtiness
        # Define a cache for indices here (until I find a better place for it)
        self.indicescache = <NumCache>NumCache(INDICES_CACHE_SIZE, 1,
                                               "Int64", 8, "indices")


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


  cdef long long _readIndex_sparse(self, long long coord):
    cdef herr_t ret
    cdef hsize_t irow, icol
    cdef long long abscoord

    irow = coord / self.l_slicesize;  icol = coord - irow * self.l_slicesize
    # Do the physical read
    ##Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOread_readSlice(self.dataset_id, self.space_id, self.type_id,
                                 irow, icol, icol+1, &abscoord)
    ##Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("_readIndex_sparse: Problems reading the indices.")

    return abscoord


  def _initSortedSlice(self, index, pro=0):
    "Initialize the structures for doing a binary search"
    cdef long ndims
    cdef int  rank, buflen, cachesize
    cdef char *bname
    cdef hsize_t count[2]

    # Create the buffer for reading sorted data chunks
    if self.bufferlb is None:
      if str(self.type) == "CharType":
        self.bufferlb = strings.array(None, itemsize=self.itemsize,
                                      shape=self.chunksize)
      else:
        self.bufferlb = numarray.array(None, type=self.type,
                                       shape=self.chunksize)
      # Internal buffers
      # Get the pointers to the different buffer data areas
      NA_getBufferPtrAndSize(self.bufferlb._data, 1, &self.rbuflb)
      # index.starts and index.lengths are assigned here for allow access
      # from _initIndexSlice. I should find a better way to share them.
      index.starts = numarray.array(None, shape=index.nrows,
                                    type = numarray.Int32)
      index.lengths = numarray.array(None, shape=index.nrows,
                                     type = numarray.Int32)
      NA_getBufferPtrAndSize(index.starts._data, 1, &self.rbufst)
      NA_getBufferPtrAndSize(index.lengths._data, 1, &self.rbufln)
      # Init structures for accelerating sorted array reads
      self.space_id = H5Dget_space(self.dataset_id)
      rank = 2
      count[0] = 1; count[1] = self.chunksize;
      self.mem_space_id = H5Screate_simple(rank, count, NULL)
      # cache some counters in local extension variables
      self.l_nrows = self.nrows
      self.l_slicesize = index.slicesize
      self.l_chunksize = index.chunksize
    if pro and not index.cache :
      # This 1st cache is loaded completely in memory
      index.rvcache = index.ranges[:]
      NA_getBufferPtrAndSize(index.rvcache._data, 1, &self.rbufrv)
      index.cache = True
      # The 2nd level cache and sorted values will be cached in a NumCache
      self.bounds_ext = <CacheArray>index.bounds
      self.nbounds = index.bounds.shape[1]
      # The <NumCache> cast is for keeping the C compiler happy
      self.boundscache = <NumCache>NumCache(BOUNDS_CACHE_SIZE, self.nbounds,
                                            self.type, self.itemsize, 'bounds')
      if str(self.type) == "CharType":
        self.bufferbc = strings.array(None, itemsize=self.itemsize,
                                      shape=self.nbounds)
      else:
        self.bufferbc = numarray.array(None, type=self.type,
                                       shape=self.nbounds)
      # Init the bounds array for reading
      self.bounds_ext.initRead(self.nbounds)
      # Get the pointer for the internal buffer for 2nd level cache
      NA_getBufferPtrAndSize(self.bufferbc._data, 1, &self.rbufbc)
      # Another NumCache for the sorted values
      self.sortedcache = <NumCache>NumCache(SORTED_CACHE_SIZE, self.chunksize,
                                            self.type, self.itemsize, 'sorted')


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


  def _interSearch_left(self, int nrow, int chunksize, item, int lo, int hi):
    cdef int niter, mid, start, result, beginning

    niter = 0
    beginning = 0
    while lo < hi:
      mid = (lo+hi)/2
      start = (mid/chunksize)*chunksize
      buffer = self._readSrotedSlice(nrow, start, start+chunksize)
      #buffer = xrange(start,start+chunksize) # test
      niter = niter + 1
      result = self._bisect_left(buffer, item, chunksize)
      if result == 0:
        if buffer[result] == item:
          lo = start
          beginning = 1
          break
        # The item is at left
        hi = mid
      elif result == chunksize:
        # The item is at the right
        lo = mid+1
      else:
        # Item has been found. Exit the loop and return
        lo = result+start
        break
    return (lo, beginning, niter)


  def _interSearch_right(self, int nrow, int chunksize, item, int lo, int hi):
    cdef int niter, mid, start, result, ending
    cdef void *rbuflb
    cdef object buffer

    niter = 0;  ending = 0
    while lo < hi:
      mid = (lo+hi)/2
      start = (mid/chunksize)*chunksize
      buffer = self._readSortedSlice(nrow, start, start+chunksize)
      niter = niter + 1
      result = self._bisect_right(buffer, item, chunksize)
      if result == 0:
        # The item is at left
        hi = mid
      elif result == chunksize:
        if buffer[result-1] == item:
          lo = start+chunksize
          ending = 1
          break
        # The item is at the right
        lo = mid+1
      else:
        # Item has been found. Exit the loop and return
        lo = result+start
        break
    return (lo, ending, niter)


  # Get the bounds from the cache, or read them
  cdef void *getLRUbounds(self, int nrow, int nbounds):
    cdef void *vpointer
    cdef object buf

    if self.boundscache.contains(nrow):
      vpointer = self.boundscache.getitem(nrow)
    else:
      # Bounds row is not in cache. Read it and put it in the LRU cache.
      self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
      self.boundscache.setitem(nrow, self.rbufbc, 0, nbounds)
      vpointer = self.rbufbc
    return vpointer


  # Get the sorted row from the cache or read it.
  cdef void *getLRUsorted(self, int nrow, int ncs, int nchunk, int cs):
    cdef void *vpointer
    cdef long long nckey

    # Compute the number of chunk read and use it as the key for the cache.
    nckey = nrow*ncs+nchunk
    if self.sortedcache.contains(nckey):
      vpointer = self.sortedcache.getitem(nckey)
    else:
      # The sorted chunk is not in cache. Read it and put it in the LRU cache.
      vpointer = self._g_readSortedSlice(nrow, cs*nchunk, cs*(nchunk+1))
      self.sortedcache.setitem(nckey, vpointer, 0, cs)
    return vpointer


  # Optimized version for doubles
  def _searchBinNA_d(self, double item1, double item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlen, len, bread, nchunk, nchunk2
    cdef int *rbufst, *rbufln
    # Variables with specific type
    cdef double *rbufrv, *rbufbc, *rbuflb

    cs = self.l_chunksize;  ss = self.l_slicesize; ncs = ss / cs
    nbounds = self.nbounds;  nrows = self.l_nrows
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufrv = <double *>self.rbufrv; tlen = 0
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
      len = stop - start;  tlen = tlen + len
      rbufst[nrow] = start;  rbufln[nrow] = len;
    return tlen


  # Optimized version for ints
  def _searchBinNA_i(self, double item1, double item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlen, len, bread, nchunk, nchunk2
    cdef int *rbufst, *rbufln
    # Variables with specific type
    cdef int *rbufrv, *rbufbc, *rbuflb

    cs = self.l_chunksize;  ss = self.l_slicesize; ncs = ss / cs
    nbounds = self.nbounds;  nrows = self.l_nrows
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufrv = <int *>self.rbufrv; tlen = 0
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
      len = stop - start;  tlen = tlen + len
      rbufst[nrow] = start;  rbufln[nrow] = len;
    return tlen


  # Optimized version for long long
  def _searchBinNA_ll(self, double item1, double item2):
    cdef int cs, ss, ncs, nrow, nrows, nbounds, rvrow
    cdef int start, stop, tlen, len, bread, nchunk, nchunk2
    cdef int *rbufst, *rbufln
    # Variables with specific type
    cdef long long *rbufrv, *rbufbc, *rbuflb

    cs = self.l_chunksize;  ss = self.l_slicesize; ncs = ss / cs
    nbounds = self.nbounds;  nrows = self.l_nrows
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufrv = <long long *>self.rbufrv; tlen = 0
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
      len = stop - start;  tlen = tlen + len
      rbufst[nrow] = start;  rbufln[nrow] = len;
    return tlen


  # This version of getCoords reads the indexes in chunks.
  # Because of that, it can be used in iterators.
  def _getCoords(self, index, int startcoords, int ncoords):
    cdef int nrow, nrows, leni, len1, len2, relcoords, nidxelem
    cdef int *rbufst, *rbufln
    cdef int startl, stopl, incr, stop

    len1 = 0; len2 = 0; relcoords = 0
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
        if nrow < self.nrows:
          self._readIndex(nrow, startl, stopl, relcoords)
        else:
          # Get indices for last row
          stop = relcoords+(stopl-startl)
          indicesLR = index.indicesLR
          # The next line can be optimised by calling indicesLR._g_readSlice()
          # directly although I don't know if it is worth the effort.
          arrAbs[relcoords:stop] = indicesLR[startl:stopl]
        incr = stopl - startl
        relcoords = relcoords + incr
        startcoords = startcoords + incr
        ncoords = ncoords - incr
        if ncoords == 0:
          break
      len1 = len1 + leni
    return arrAbs[:relcoords]


  # This version of getCoords reads all the indexes in one pass.
  # Because of that, it is not meant to be used on iterators.
  # This is aproximately a 25% faster than _getCoords above.
  # If there is a last row with interesting values on it, this has been
  # optimised as well.
  def _getCoords_sparse(self, index, int ncoords):
    cdef int nrow, startl, stopl, lenl, relcoord
    cdef int *rbufst, *rbufln
    cdef long long *rbufC, *rbufA
    cdef long long coord
    cdef object nckey

    # Initialize the index dataset
    self._initIndexSlice(index, ncoords)
    rbufst = <int *>self.rbufst;  rbufln = <int *>self.rbufln
    rbufC = <long long *>self.rbufC
    rbufA = <long long *>self.rbufA

    # Get the sorted indices
    get_sorted_indices(self.l_nrows, rbufC, rbufst, rbufln, self.l_slicesize)

    # Retrieve the reverse coordinates
    nrows = self.nrows;  relcoords = 0
    for relcoord from 0 <= relcoord < ncoords:
      coord = rbufC[relcoord]
      # Look at the cache for this coord
      if self.indicescache.contains(coord):
        self.indicescache.getitem2(coord, self.rbufA, relcoord)
      else:
        # The coord is not in cache. Read it and put it in the LRU cache.
        rbufA[relcoord] = self._readIndex_sparse(coord)
        self.indicescache.setitem(coord, self.rbufA, relcoord, 1)

    # Get possible values in last slice
    if (index.nrows > nrows and rbufln[nrows] > 0):
      # Get indices for last row
      startl = rbufst[nrows]
      stopl = startl + rbufln[nrows]
      lenl = ncoords - rbufln[nrows]
      index.indicesLR._readIndexSlice(self, startl, stopl, lenl)

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
      raise HDF5ExtError("Problems reading the index data.")

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


#  Minimalistic LRU cache implementation for numerical data

#*********************** Important note! ****************************
#The code behind has been carefully tuned to serve the needs of
#caching numarray (or numpy) data. As a consequence, it is no longer
#appropriate as a general LRU cache implementation. You have been
#warned!. F. Altet 2006-08-09
#********************************************************************

from heapq import heappush, heappop, heapreplace, heapify
from constants import ENABLE_EVERY_CYCLES, LOWEST_HIT_RATIO

cdef class LRUNode:
  """Record of a cached value. Not for public consumption."""
  cdef object key
  cdef long nslot, atime, slotsize

  def __init__(self, object key, long nslot, long slotsize, long atime):
    object.__init__(self)
    self.key = key
    self.nslot = nslot
    self.slotsize = slotsize
    self.atime = atime


  def __cmp__(self, LRUNode other):
    #return cmp(self.atime_, other.atime_)
    # This optimization makes the comparison more than twice as faster
    if self.atime < self.atime:
      return -1
    elif self.atime > other.atime:
      return 1
    else:
      return 0


  def __repr__(self):
    return "<%s %s => %s (accessed at %s)>" % \
           (self.__class__, self.key, self.obj, self.atime)


cdef class NumCache:
  """Least-Recently-Used (LRU) cache specific for Numerical data.
  """
  cdef object __dict, __heap
  cdef long seqn_, nextslot, setcount, getcount, cyclecount
  cdef long enableeverycycles
  cdef double lowesthr
  cdef int iscachedisabled, itemsize, nslots, maxslotsize
  cdef object name, cacheobj
  cdef void *rcache

  def __init__(self, long nslots, int maxslotsize, object type,
               int itemsize, object name):
    """Maximum size of the cache.
    If more than 'nslots' elements are added to the cache,
    the least-recently-used ones will be discarded."""
    cdef object shape

    if nslots <= 0:
      raise ValueError, nslots
    self.__heap = [];  self.__dict = {}
    self.nslots = nslots;  self.maxslotsize = maxslotsize;
    self.itemsize = itemsize;  self.name = name
    self.seqn_ = 0;  self.nextslot = 0
    self.setcount = 0;  self.getcount = 0;  self.cyclecount = 0
    self.iscachedisabled = False  # Cache is enabled by default
    self.enableeverycycles = ENABLE_EVERY_CYCLES
    self.lowesthr = LOWEST_HIT_RATIO
    # The cache object where all data will go
    shape = (nslots, maxslotsize)
    if str(type) == "CharType":
      self.cacheobj = strings.array(None, itemsize=itemsize, shape=shape)
    else:
      self.cacheobj = numarray.array(None, type=type, shape=shape)
    NA_getBufferPtrAndSize(self.cacheobj._data, 1, &self.rcache)


  cdef long incseqn(self):
    cdef LRUNode node

    self.seqn_ = self.seqn_ + 1
    if self.seqn_ < 0:
      # Ooops, the counter has run out of range!
      # Reset all the priorities to 0
      for node in self.__heap:
        node.atime = 0
      # Set the counter to 1 (to indicate that it is newer than existing ones)
      self.seqn_ = 1
    return self.seqn_


  cdef int contains(self, key):
    return PyDict_Contains(self.__dict, key)


  # Machinery for determining whether the hit ratio is being effective
  # or not.  If not, the cache will be disabled. The efficency will be
  # checked every cycle (the time that the cache would be refilled
  # completely).  In situations where the cache is not being re-filled
  # (i.e. it is not enabled) for a long time, it is forced to be
  # re-enabled when a certain number of cycles has passed so as to
  # check whether a new scenario where the cache can be useful again
  # has come.
  # F. Altet 2006-08-09
  cdef int checkhitratio(self):
    cdef double hitratio

    if self.setcount > self.nslots:
      self.cyclecount = self.cyclecount + 1
      # Check whether the cache is being effective or not
      hitratio = <double>self.getcount / (self.setcount+self.getcount)
      if hitratio < self.lowesthr:
        # Hit ratio is low. Disable the cache.
        self.iscachedisabled = True
      else:
        # Hit ratio is acceptable. (Re-)Enable the cache.
        self.iscachedisabled = False
      # Reset the counters to 0
      self.setcount = 0; self.getcount = 0
      if self.cyclecount > self.enableeverycycles:
        # We have reached the time for forcing the cache to act again
        self.iscachedisabled = False
        self.cyclecount = 0
    return not self.iscachedisabled

  cdef long setitem(self, object key, void *data, long start, long slotsize):
    cdef LRUNode node, lru
    cdef long nslot, base1, base2

    self.setcount = self.setcount + 1
    if self.checkhitratio():
      # Check if we are growing out of space
      if len(self.__heap) == self.nslots:
        lru = heappop(self.__heap)
        nslot = lru.nslot
        PyObject_DelItem(self.__dict, lru.key)
      else:
        nslot = self.nextslot;  self.nextslot = self.nextslot + 1
        assert nslot < self.nslots, "Number of nodes exceeding cache capacity."
      # Copy the data to the appropriate row in cache
      base1 = nslot * self.maxslotsize * self.itemsize
      base2 = start * self.itemsize
      memcpy(self.rcache + base1, data + base2, slotsize * self.itemsize)
      # Add a new node with references to the copied data to the LRUCache
      node = LRUNode(key, nslot, slotsize, self.incseqn())
      heappush(self.__heap, node)
      PyObject_SetItem(self.__dict, key, node)  # Can't fail
    return nslot


  # Return the pointer to the data in cache
  cdef void *getitem(self, object key):
    cdef LRUNode node
    cdef long base

    self.getcount = self.getcount + 1
    node = PyObject_GetItem(self.__dict, key)
    node.atime = self.incseqn()
    base = node.nslot * self.maxslotsize * self.itemsize
    return self.rcache + base


  # This version copies data in cache to data+start.
  # The user should be responsible to provide a large enough data buffer
  # to the memcpy to succeed.
  cdef long getitem2(self, object key, void *data, long start):
    cdef LRUNode node
    cdef long base
    cdef void *src

    src = self.getitem(key)
    # Copy the data in cache to destination
    base = start * self.itemsize
    memcpy(data + base, src, node.slotsize * self.itemsize)
    return node.nslot


  def __repr__(self):
    return "<%s (%d elements)>" % (str(self.__class__), len(self.__heap))
