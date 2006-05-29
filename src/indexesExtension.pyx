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

from definitions cimport import_libnumarray, NA_getPythonScalar, \
     NA_getBufferPtrAndSize, Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, \
     PyInt_AsLong, PyLong_AsLongLong



__version__ = "$Revision$"

#-------------------------------------------------------------------

# C functions and variable declaration from its headers

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
                                    hid_t mem_type_id,
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
  int get_sorted_indices(int nrows, long long *rbufR2,
                         int *rbufst, int *rbufln)
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


cdef class IndexArray(Array):
  """Container for keeping sorted and indices values."""
  cdef void    *rbufst, *rbufln, *rbufrv, *rbufbc, *rbuflb
  cdef void    *rbufR, *rbufR2, *rbufA
  cdef hid_t   space_id, mem_space_id
  cdef int     nbounds
  cdef object  bufferl
  cdef CacheArray bounds_ext

  def _initIndexSlice(self, ncoords):
    "Initialize the structures for doing a binary search"
    cdef long buflen

    # Create buffers for reading reverse index data
    if self.arrRel is None or len(self.arrRel) < ncoords:
      self.arrRel = numarray.zeros(type="Int32", shape=ncoords)
      self.coords = numarray.zeros(type="Int64", shape=(ncoords, 2))
      self.arrAbs = numarray.zeros(type="Int64", shape=ncoords)
      # Get the pointers to the buffer data area
      NA_getBufferPtrAndSize(self.arrRel._data, 1, &self.rbufR)
      NA_getBufferPtrAndSize(self.coords._data, 1, &self.rbufR2)
      NA_getBufferPtrAndSize(self.arrAbs._data, 1, &self.rbufA)
      NA_getBufferPtrAndSize(self._v_parent.starts._data, 1, &self.rbufst)
      NA_getBufferPtrAndSize(self._v_parent.lengths._data, 1, &self.rbufln)
      # Initialize the index array for reading
      self.space_id = H5Dget_space(self.dataset_id )
      self.isopen_for_read = True

  def _readIndex(self, hsize_t irow, hsize_t start, hsize_t stop,
                 long offsetl):
    cdef herr_t ret
    cdef long buflen
    cdef int *rbufR
    cdef long long *rbufA
    cdef long long offset
    cdef long j, len

    # Correct the start of the buffer with offsetl
    rbufR = <int *>self.rbufR + offsetl
    rbufA = <long long *>self.rbufA + offsetl
    # Do the physical read
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOread_readSlice(self.dataset_id, self.space_id, self.type_id,
                                 irow, start, stop, rbufR)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Now, compute the absolute coords for table rows by adding the offset
    len = stop-start
    offset = irow*self.nelemslice
    for j from 0 <= j < len:
      rbufA[j] = rbufR[j] + offset

    return

  def _readIndex_sparse(self, hsize_t ncoords):
    cdef herr_t ret

    # Do the physical read
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOread_index_sparse(self.dataset_id, self.space_id, self.type_id,
                                    ncoords, self.rbufR2, self.rbufR)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("_readIndex_sparse: Problems reading the array data.")

    return

  def _initSortedSlice(self, pro=0):
    "Initialize the structures for doing a binary search"
    cdef long ndims
    cdef int  buflen
    cdef char *bname
    cdef int  rank
    cdef hsize_t count[2]

    index = self._v_parent
    # Create the buffer for reading sorted data chunks
    if self.bufferl is None:
      if str(self.type) == "CharType":
        self.bufferl = strings.array(None, itemsize=self.itemsize,
                                     shape=self.chunksize)
      else:
        self.bufferl = numarray.array(None, type=self.type,
                                      shape=self.chunksize)
      # Internal buffers
      # (starts and lengths for interesting values along the slices)
      index.starts = numarray.array(None, shape=index.nrows,
                                    type = numarray.Int32)
      index.lengths = numarray.array(None, shape=index.nrows,
                                     type = numarray.Int32)
      # Get the pointers to the different buffer data areas
      NA_getBufferPtrAndSize(self.bufferl._data, 1, &self.rbuflb)
      NA_getBufferPtrAndSize(index.starts._data, 1, &self.rbufst)
      NA_getBufferPtrAndSize(index.lengths._data, 1, &self.rbufln)
      # Init structures for accelerating sorted array reads
      self.space_id = H5Dget_space(self.dataset_id)
      rank = 2
      count[0] = 1; count[1] = self.chunksize;
      self.mem_space_id = H5Screate_simple(rank, count, NULL)
      self.isopen_for_read = True
    if pro and not index.cache :
      index.rvcache = index.ranges[:]
      NA_getBufferPtrAndSize(index.rvcache._data, 1, &self.rbufrv)
      index.cache = True
      # Protection against using too big cache for bounds values
      self.bounds_ext = <CacheArray>index.bounds
      self.nbounds = index.bounds.shape[1]
      # Avoid loading too much data from second-level cache
      # If too much data is pre-loaded, this affects negatively to the
      # first data selection.
      #if self.nrows * self.nbounds < 100000:    # for testing purposes
      if self.nrows * self.nbounds < 4000:
        self.boundscache = index.bounds[:]
        self.bcache = True
      else:
        self.boundscache = numarray.array(None, type=self.type,
                                          shape=self.nbounds)
        # Init the bounds array for reading
        index.bounds_ext.initRead(self.nbounds)
        self.bcache = False
      NA_getBufferPtrAndSize(self.boundscache._data, 1, &self.rbufbc)

  def _readSortedSlice(self, hsize_t irow, hsize_t start, hsize_t stop):
    "Read the sorted part of an index."

    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       irow, start, stop, self.rbuflb)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    return self.bufferl

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
      buffer = self._readSortedSlice(nrow, start, start+chunksize)
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

    niter = 0
    ending = 0
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

  # Optimized version for doubles
  def _searchBinNA_d(self, double item1, double item2):
    cdef int cs, nchunk, nchunk2, nrow, nrows, nbounds, rvrow
    cdef int *rbufst, *rbufln
    cdef double *rbufbc, *rbuflb, *rbufrv
    cdef int start, stop, nslice, tlen, len, bread, bcache

    cs = self.chunksize
    nrows = self.nrows
    nbounds = self.nbounds
    nslice = self.nelemslice
    bcache = self.bcache
    tlen = 0
    rbuflb = <double *>self.rbuflb
    rbufrv = <double *>self.rbufrv
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row
          if bcache:
            rbufbc = <double *>self.rbufbc + nrow*nbounds
          else:
            # Bounds is not in cache. Read the appropriate row.
            self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
            rbufbc = <double *>self.rbufbc
          bread = 1
          nchunk = bisect_left_d(rbufbc, item1, nbounds, 0)
          H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       nrow, cs*nchunk, cs*(nchunk+1),
                                       self.rbuflb)
          start = bisect_left_d(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = nslice
      else:
        start = 0
      # The next optimization takes more time! besides, it gives some
      # seg faults from time to time!
#       if start > 0 and item1 > rbuflb[start] and item2 < rbuflb[start+1]:
#         # Not interesting values here
#         stop = start
#       elif item2 >= rbufrv[rvrow]:
      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            if bcache:
              rbufbc = <double *>self.rbufbc + nrow*nbounds
            else:
              # Bounds is not in cache. Read the appropriate row.
              self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
              rbufbc = <double *>self.rbufbc
          nchunk2 = bisect_right_d(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                         self.mem_space_id, self.type_id,
                                         nrow, cs*nchunk2, cs*(nchunk2+1),
                                         self.rbuflb)
        # The next optimization in calls does not buy any real speed-up
#           offset[0] = nrow
#           offset[1] = cs*nchunk2
#           H5Sselect_hyperslab(self.space_id, H5S_SELECT_SET, offset, stride,
#                               count, NULL)
#           H5Dread(self.dataset_id, self.type_id, self.mem_space_id,
#                   self.space_id, H5P_DEFAULT, self.rbuflb)
          stop = bisect_right_d(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = nslice
      else:
        stop = 0
      rbufst[nrow] = start
      len = stop - start
      rbufln[nrow] = len
      tlen = tlen + len
    return tlen

  # Vectorial version. This is a bit slower, but perhaps gcc 4.x would be
  # able to automatically paralelize this.
  def _searchBinNA_d_vec(self, double item1, double item2):
    cdef int cs, nchunk, nchunk2, nrow, nrows, nbounds, rvrow
    cdef int *rbufst, *rbufln
    cdef int start, stop, nslice, tlen, len, bcache
    cdef double *rbufbc, *rbuflb, *rbufrv

    cs = self.chunksize
    nrows = self.nrows
    nbounds = self.nbounds
    nslice = self.nelemslice
    bcache = self.bcache
    tlen = 0
    rbuflb = <double *>self.rbuflb
    rbufrv = <double *>self.rbufrv
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      start = (item1 <= rbufrv[rvrow]) + (item1 > rbufrv[rvrow+1]) * nslice
      stop  = (item2 < rbufrv[rvrow]) + (item2 >= rbufrv[rvrow+1]) * nslice
      if start == 0 or stop == 0:
        # Get the bounds row
        if bcache:
          rbufbc = <double *>self.rbufbc + nrow*nbounds
        else:
          # Bounds is not in cache. Read the appropriate row.
          self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
          rbufbc = <double *>self.rbufbc
      if start == 0:
        nchunk = bisect_left_d(rbufbc, item1, nbounds, 0)
        # self._readSortedSlice(nrow, cs*nchunk, cs*(nchunk+1))
        H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                     self.mem_space_id, self.type_id,
                                     nrow, cs*nchunk, cs*(nchunk+1),
                                     self.rbuflb)
        start = bisect_left_d(rbuflb, item1, cs, 0) + cs*nchunk
      if stop == 0:
        nchunk2 = bisect_right_d(rbufbc, item2, nbounds, 0)
        if nchunk2 <> nchunk:
          # self._readSortedSlice(nrow, cs*nchunk2, cs*(nchunk2+1))
          H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       nrow, cs*nchunk2, cs*(nchunk2+1),
                                       self.rbuflb)
        stop = bisect_right_d(rbuflb, item2, cs, 0) + cs*nchunk2
      rbufst[nrow] = start
      len = stop - start
      rbufln[nrow] = len
      tlen = tlen + len
    return tlen

  # Optimized version for ints
  def _searchBinNA_i(self, int item1, int item2):
    cdef int cs, nchunk, nchunk2, nrow, nrows, nbounds, rvrow
    cdef int *rbufst, *rbufln
    cdef int start, stop, nslice, tlen, len, bread, bcache
    # Variables with specific type
    cdef int *rbufbc, *rbuflb, *rbufrv

    cs = self.chunksize
    nrows = self.nrows
    nbounds = self.nbounds
    nslice = self.nelemslice
    bcache = self.bcache
    tlen = 0
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbuflb = <int *>self.rbuflb   # specific type
    rbufrv = <int *>self.rbufrv   # specific type
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row
          if bcache:
            rbufbc = <int *>self.rbufbc + nrow*nbounds  # specific type
          else:
            # Bounds is not in cache. Read the appropriate row.
            self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
            rbufbc = <int *>self.rbufbc  # specific type
          bread = 1
          nchunk = bisect_left_i(rbufbc, item1, nbounds, 0)
          H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       nrow, cs*nchunk, cs*(nchunk+1),
                                       self.rbuflb)
          start = bisect_left_i(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = nslice
      else:
        start = 0

      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            if bcache:
              rbufbc = <int *>self.rbufbc + nrow*nbounds  # specific type
            else:
              # Bounds is not in cache. Read the appropriate row.
              self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
              rbufbc = <int *>self.rbufbc  # specific type
          nchunk2 = bisect_right_i(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                         self.mem_space_id, self.type_id,
                                         nrow, cs*nchunk2, cs*(nchunk2+1),
                                         self.rbuflb)
          stop = bisect_right_i(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = nslice
      else:
        stop = 0
      rbufst[nrow] = start
      len = stop - start
      rbufln[nrow] = len
      tlen = tlen + len
    return tlen

  # Vectorial version. This is a bit slower, but perhaps gcc 4.x would be
  # able to automatically paralelize this.
  def _searchBinNA_i_vec(self, int item1, int item2):
    cdef int cs, nchunk, nchunk2, nrow, nrows, nbounds, rvrow
    cdef int *rbufst, *rbufln
    cdef int *rbufbc, *rbuflb, *rbufrv
    cdef int start, stop, nslice, tlen, len, bcache

    cs = self.chunksize
    nrows = self.nrows
    nbounds = self.nbounds
    bcache = self.bcache
    nslice = self.nelemslice
    tlen = 0
    rbuflb = <int *>self.rbuflb
    rbufrv = <int *>self.rbufrv
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      start = (item1 <= rbufrv[rvrow]) + (item1 > rbufrv[rvrow+1]) * nslice
      stop = (item2 < rbufrv[rvrow]) + (item2 >= rbufrv[rvrow+1]) * nslice
      if start == 0 or stop == 0:
        # Get the bounds row
        if bcache:
          rbufbc = <int *>self.rbufbc + nrow*nbounds
        else:
          # Bounds is not in cache. Read the appropriate row.
          self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
          rbufbc = <int *>self.rbufbc
      if start == 0:
        nchunk = bisect_left_i(rbufbc, item1, nbounds, 0)
        #self._readSortedSlice(nrow, cs*nchunk, cs*(nchunk+1))
        H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                     self.mem_space_id, self.type_id,
                                     nrow, cs*nchunk, cs*(nchunk+1),
                                     self.rbuflb)
        start = bisect_left_i(rbuflb, item1, cs, 0) + cs*nchunk
      if stop == 0:
        nchunk2 = bisect_right_i(rbufbc, item2, nbounds, 0)
        if nchunk2 <> nchunk:
          #self._readSortedSlice(nrow, cs*nchunk2, cs*(nchunk2+1))
          H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       nrow, cs*nchunk2, cs*(nchunk2+1),
                                       self.rbuflb)
        stop = bisect_right_i(rbuflb, item2, cs, 0) + cs*nchunk2
      rbufst[nrow] = start
      len = stop - start
      rbufln[nrow] = len
      tlen = tlen + len
    return tlen


  # Optimized version for long long
  def _searchBinNA_ll(self, long long item1, long long item2):
    cdef int cs, nchunk, nchunk2, nrow, nrows, nbounds, rvrow
    cdef int *rbufst, *rbufln
    cdef int start, stop, nslice, tlen, len, bread, bcache
    # Variables with specific type
    cdef long long *rbufbc, *rbuflb, *rbufrv

    cs = self.chunksize
    nrows = self.nrows
    nbounds = self.nbounds
    nslice = self.nelemslice
    bcache = self.bcache
    tlen = 0
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbuflb = <long long *>self.rbuflb   # specific type
    rbufrv = <long long *>self.rbufrv   # specific type
    for nrow from 0 <= nrow < nrows:
      rvrow = nrow*2
      bread = 0
      if item1 > rbufrv[rvrow]:
        if item1 <= rbufrv[rvrow+1]:
          # Get the bounds row
          if bcache:
            rbufbc = <long long *>self.rbufbc + nrow*nbounds  # specific type
          else:
            # Bounds is not in cache. Read the appropriate row.
            self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
            rbufbc = <long long *>self.rbufbc  # specific type
          bread = 1
          nchunk = bisect_left_ll(rbufbc, item1, nbounds, 0)
          H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                       self.mem_space_id, self.type_id,
                                       nrow, cs*nchunk, cs*(nchunk+1),
                                      self.rbuflb)
          start = bisect_left_ll(rbuflb, item1, cs, 0) + cs*nchunk
        else:
          start = nslice
      else:
        start = 0

      if item2 >= rbufrv[rvrow]:
        if item2 < rbufrv[rvrow+1]:
          if not bread:
            if bcache:
              rbufbc = <long long *>self.rbufbc + nrow*nbounds  # specific type
            else:
              # Bounds is not in cache. Read the appropriate row.
              self.bounds_ext.readSlice(nrow, 0, nbounds, self.rbufbc)
              rbufbc = <long long *>self.rbufbc  # specific type
          nchunk2 = bisect_right_ll(rbufbc, item2, nbounds, 0)
          if nchunk2 <> nchunk:
            H5ARRAYOread_readSortedSlice(self.dataset_id, self.space_id,
                                         self.mem_space_id, self.type_id,
                                         nrow, cs*nchunk2, cs*(nchunk2+1),
                                         self.rbuflb)
          stop = bisect_right_ll(rbuflb, item2, cs, 0) + cs*nchunk2
        else:
          stop = nslice
      else:
        stop = 0
      rbufst[nrow] = start
      len = stop - start
      rbufln[nrow] = len
      tlen = tlen + len
    return tlen


  # Vectorial version for values of any type. It doesn't seem to work well though :-(
  def _searchBinNA_vec(self, item1, item2):
    cdef int cs, nbounds, nchunk, nchunk2, nrow, nrows, stride1
    cdef object boundscache, ibounds, chunk
    cdef int *rbufst, *rbufln
    cdef int start, stop, nslice, tlen, len

    tlen = 0
    cs = self.chunksize
    nrows = self.nrows
    if self.bcache:
      boundscache = self.boundscache
      stride1 = boundscache.type().bytes
    else:
      boundscache = self._v_parent.bounds
      stride1 = boundscache.type.bytes
    rvc = self._v_parent.rvcache
    nslice = self.nelemslice
    nbounds = self.nbounds
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    for nrow from 0 <= nrow < nrows:
      ibounds = boundscache[nrow]
      start = (item1 <= rvc[nrow,0]) + (item1 > rvc[nrow,1]) * nslice
      stop = (item2 < rvc[nrow,0]) + (item2 >= rvc[nrow,1]) * nslice
      if start == 0:
        nchunk = self._bisect_left_optim(ibounds, item1, nbounds, stride1)
        chunk = self._readSortedSlice(nrow, cs*nchunk, cs*(nchunk+1))
        start = self._bisect_left_optim(chunk, item1, cs, stride1) + cs*nchunk
      if stop == 0:
        nchunk2 = self._bisect_right_optim(ibounds, item2, nbounds, stride1)
        if nchunk2 <> nchunk:
          # The chunk for item2 is different. Read the new chunk.
          chunk = self._readSortedSlice(nrow, cs*nchunk2, cs*(nchunk2+1))
        stop = self._bisect_right_optim(chunk, item2, cs, stride1) + cs*nchunk2
      rbufst[nrow] = start
      len = stop - start
      rbufln[nrow] = len
      tlen = tlen + len
    return tlen

  # This version of getCoords reads the indexes in chunks.
  # Because of that, it can be used in iterators.
  def _getCoords(self, int startcoords, int ncoords):
    cdef int nrow, nrows, leni, len1, len2, relcoords, nindexedrows
    cdef int *rbufst, *rbufln
    cdef int startl, stopl, incr, stop, offset

    len1 = 0; len2 = 0; relcoords = 0
    # Correction against asking too many elements
    nindexedrows = self._v_parent.nelements
    if startcoords + ncoords > nindexedrows:
      ncoords = nindexedrows - startcoords
    # create buffers for indices
    self._initIndexSlice(ncoords)
    arrAbs = self.arrAbs
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    nrows = self._v_parent.nrows
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
          offset = nrow*self.nelemslice
          stop = relcoords+(stopl-startl)
          indicesLR = self._v_parent.indicesLR
          # The next line can be optimised by calling indicesLR._g_readSlice()
          # directly although I don't know if it is worth the effort.
          arrAbs[relcoords:stop] = indicesLR[startl:stopl] + offset
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
  def _getCoords_sparse(self, int ncoords):
    cdef int irow, jrow, nrows, len1, offset
    cdef int *rbufR, *rbufst, *rbufln
    cdef long long *rbufA, *rbufR2

    nrows = self.nrows
    lengths = self._v_parent.lengths
    starts = self._v_parent.starts
    # Initialize the index dataset
    self._initIndexSlice(ncoords)
    rbufR = <int *>self.rbufR
    rbufst = <int *>self.rbufst
    rbufln = <int *>self.rbufln
    rbufA = <long long *>self.rbufA
    rbufR2 = <long long *>self.rbufR2

    # Get the sorted indices
#     len1 = 0
#     for irow from 0 <= irow < nrows:
#       for jrow from 0 <= jrow < rbufln[irow]:
#         rbufR2[len1] = irow
#         len1 = len1 + 1
#         rbufR2[len1] = rbufst[irow] + jrow
#         len1 = len1 + 1
    # C version of above
    get_sorted_indices(nrows, rbufR2, rbufst, rbufln)

    # Given the sorted indices, get the real ones
    self._readIndex_sparse(ncoords)

    # Get possible values in last slice
    if (self._v_parent.nrows > nrows and rbufln[nrows] > 0):
      # Get indices for last row
      irow = nrows
      startl = rbufst[irow]
      stopl = startl + rbufln[irow]
      len1 = ncoords - rbufln[irow]
      #offset = irow * self.nelemslice
      #self.arrAbs[len1:ncoords] = self._v_parent.indicesLR[startl:stopl] + offset
      self._v_parent.indicesLR._readIndexSlice(self, startl, stopl, len1)
      nrows = nrows + 1  # Add the last row for later conversion to 64-bit

    # Finally, convert the values to full 64-bit addresses
#     len1 = 0
#     offset = self.nelemslice
#     for irow from 0 <= irow < nrows:
#       for jrow from 0 <= jrow < rbufln[irow]:
#         rbufA[len1] = rbufR[len1] + irow * offset
#         len1 = len1 + 1
    # C version of above
    offset = self.nelemslice
    convert_addr64(nrows, offset, rbufA, rbufR, rbufln)

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
    cdef int *rbufR

    rbufR = <int *>indices.rbufR + offsetl
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOreadSliceLR(self.dataset_id, start, stop, rbufR)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the index data.")

    return

  def _readSortedSlice(self, IndexArray sorted, hsize_t start, hsize_t stop):
    "Read the sorted part of an LR index."
    cdef void  *rbuflb

    rbuflb = sorted.rbuflb  # direct access to rbuflb: very fast.
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYOreadSliceLR(self.dataset_id, start, stop, rbuflb)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the index data.")

    return sorted.bufferl
