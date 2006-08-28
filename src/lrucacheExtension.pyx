#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: Aug 13, 2006
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id: indexesExtension.pyx 1756 2006-08-13 10:19:29Z faltet $
#
########################################################################

"""Pyrex interface for several LRU cache systems.

Classes (type extensions):

    NodeCache
    NumCache

Functions:

Misc variables:

    __version__
"""

import sys

import numpy
from numpydefs cimport import_array, ndarray, \
     PyArray_GETITEM, PyArray_SETITEM

from lrucache import CacheKeyError
from constants import ENABLE_EVERY_CYCLES, LOWEST_HIT_RATIO




#----------------------------------------------------------------------

# Standard C functions.
cdef extern from "stdlib.h":
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  void *memcpy(void *dest, void *src, size_t n)
  int strcmp(char *s1, char *s2)

# Python API functions.
cdef extern from "Python.h":
  ctypedef int Py_ssize_t
  int PySequence_DelItem(object o, Py_ssize_t i)
  int PyList_Append(object list, object item)
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)
  long PyObject_Length(object o)
  int PyObject_Compare(object o1, object o2)
  char *PyString_AsString(object string)

# # External C functions for dealing with binary searchs
# cdef extern from "idx-opt.h":
#   int bisect_left_ll2(long long *a, long long x, int hi)


#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#---------------------------------------------------------------------------



# ------- Minimalist NodeCache for nodes in PyTables ---------

# The next NodeCache code relies on the fact that a node that is
# fetched from the cache will be removed for it. Said in other words:
# "A node cannot be alive and dead at the same time."

# Thanks to the above behaviour, the next code has been stripped down
# to a bare minimum (the info in cache is kept in just 2 lists).

#*********************** Important note! *****************************
#The code behind has been carefully tuned to serve the needs of
#PyTables cache for nodes. As a consequence, it is no longer
#appropriate as a general LRU cache implementation. You have been
#warned!.  F. Altet 2006-08-08
#*********************************************************************


cdef class NodeCache:
  """Least-Recently-Used (LRU) cache for PyTables nodes.
  """
  # This class variables are declared in utilsExtension.pxd


  def __init__(self, size):
    """Maximum size of the cache.
    If more than 'size' elements are added to the cache,
    the least-recently-used ones will be discarded."""

    if size <= 0:
      raise ValueError, size
    elif type(size) is not type(0):
      raise TypeError, size
    self.nodes = [];  self.paths = []
    self.size = size;  self.lsize = 0


  def __len__(self):
    return len(self.nodes)


  def __setitem__(self, path, node):
    self.setitem(path, node)


  # Puts a new node in the node list
  cdef setitem(self, object path, object node):

    # Add the node and path to the end of its lists
    PyList_Append(self.nodes, node);  PyList_Append(self.paths, path)
    self.lsize = self.lsize + 1
    # Check if we are growing out of space
    if self.lsize == self.size:
      # Remove the LRU node and path (the start of the lists)
      PyObject_DelItem(self.nodes, 0);  PyObject_DelItem(self.paths, 0)
      self.lsize = self.lsize - 1


  def __contains__(self, path):
    if self.contains(path) == -1:
      return 0
    else:
      return 1


  # Checks whether path is in this cache or not
  cdef long contains(self, object path):
    cdef long i, idx

    idx = -1  # -1 means not found
    # Start looking from the trailing values (most recently used)
    for i from self.lsize > i >= 0:
      if path == PyObject_GetItem(self.paths, i):
        idx = i
        break
    return idx


  def pop(self, path):
    return self.cpop(path)


  cdef object cpop(self, object path):
    cdef object idx

    idx = self.contains(path)
    node = PyObject_GetItem(self.nodes, idx)
    PyObject_DelItem(self.nodes, idx);  PyObject_DelItem(self.paths, idx)
    self.lsize = self.lsize - 1
    return node


  def __iter__(self):
    return iter(self.paths)


  def __repr__(self):
    return "<%s (%d elements)>" % (str(self.__class__), len(self.paths))



#  Minimalistic LRU cache implementation for numerical data

#*********************** Important note! ****************************
#The code behind has been carefully tuned to serve the needs of
#caching numerical data. As a consequence, it is no longer appropriate
#as a general LRU cache implementation. You have been warned!.
#F. Altet 2006-08-09
#********************************************************************

cdef class NumCache:
  """Least-Recently-Used (LRU) cache specific for Numerical data.
  """

  def __init__(self, object shape, int itemsize, object name):
    """Maximum size of the cache.
    If more than 'nslots' elements are added to the cache,
    the least-recently-used ones will be discarded.

    Parameters:
    shape - The rectangular shape of the cache (nslots, nelemsperslot)
    itemsize - The size of the element base in cache
    name - A descriptive name for this cache
    """

    self.nslots = shape[0];  self.slotsize = shape[1]*itemsize
    self.itemsize = itemsize;  self.name = name
    if self.nslots <= 0:
      raise ValueError, "Negative or zero number (%s) of slots!" % self.nslots
    if self.nslots >= 2**16:
      raise ValueError, "Too many slots (%s) in cache!" % self.nslots
    self.seqn_ = 0;  self.nextslot = 0
    self.setcount = 0;  self.getcount = 0;  self.cyclecount = 0
    self.iscachedisabled = False  # Cache is enabled by default
    self.enableeverycycles = ENABLE_EVERY_CYCLES
    self.lowesthr = LOWEST_HIT_RATIO
    # The cache object where all data will go
    self.cacheobj = numpy.empty(shape=(self.nslots, self.slotsize),
                                dtype=numpy.uint8)
    self.rcache = self.cacheobj.data
    # The arrays for keeping the indexes of slots
    self.sorted = -numpy.ones(shape=self.nslots, dtype=numpy.int64)
    self.rsorted = <long long *>self.sorted.data
    # 16-bits is more than enough for keeping the slot numbers
    self.indices = numpy.arange(self.nslots, dtype=numpy.uint16)
    self.rindices = <unsigned short *>self.indices.data
    # The array for keeping the access times (using long ints here)
    self.atimes = numpy.zeros(shape=self.nslots, dtype=numpy.int_)
    self.ratimes = <long *>self.atimes.data


  cdef long incseqn(self):

    self.seqn_ = self.seqn_ + 1
    if self.seqn_ < 0:
      # Ooops, the counter has run out of range! Reset all the priorities to 0.
      self.atimes[:] = 0
      # Set the counter to 1 (to indicate that it is newer than existing ones)
      self.seqn_ = 1
    return self.seqn_


  # Return the index for element x in array self.rindices.
  # This should never fail because x should be always present.
  cdef long slotlookup(self, unsigned short x):
    cdef long idx
    cdef unsigned short *a

    a = self.rindices; idx = 0
    while x != a[idx]:
      idx = idx + 1
      assert idx < self.nslots
    return idx


#   # Check where x should be found in array self.rsorted
#   cdef long keylookup(self, long long x):
#     cdef long lo, hi, mid
#     cdef long long *a

#     lo = 0;  hi = self.nslots;  a = self.rsorted
#     while lo < hi:
#       mid = (lo+hi)/2
#       if a[mid] < x: lo = mid+1
#       else: hi = mid
#     return lo


#   # Tells in which slot key is. If not found, -1 is returned.
#   cdef int contains(self, long long key):
#     cdef long idx

#     idx = self.keylookup(key)
#     if key == self.rsorted[idx]:
#       return self.rindices[idx]
#     else:
#       return -1

#   # This version optimized in C doesn't go faster than Pyrex.
#   cdef int contains(self, long long key):
#     cdef int idx

#     idx = bisect_left_ll2(self.rsorted, key, self.nslots)
#     if key == self.rsorted[idx]:
#       return self.rindices[idx]
#     else:
#       return -1

  # The next contains is a 15% faster than the splitted version.
  # Tells in which slot key is. If not found, -1 is returned.
  cdef int contains(self, long long key):
    cdef long lo, hi, mid
    cdef long long *rsorted

    rsorted = self.rsorted
    lo = 0;  hi = self.nslots
    while lo < hi:
      mid = (lo+hi)/2
      if rsorted[mid] < key: lo = mid+1
      else: hi = mid
    if key == rsorted[lo]:
      return self.rindices[lo]
    else:
      return -1


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


  cdef long setitem(self, long long key, void *data, long start):
    cdef long nslot, nidx, base1, base2

    self.setcount = self.setcount + 1
    if self.checkhitratio():
      # Check if we are growing out of space
      if self.nextslot == self.nslots:
        # Get the least recently used slot
        nslot = self.atimes.argmin()
        self.nextslot = self.nextslot - 1
      else:
        # Get the next available slot
        nslot = self.nextslot
      assert nslot < self.nslots, "Wrong slot index!"
      # Copy the data to the appropriate row in cache
      base1 = nslot * self.slotsize;  base2 = start * self.itemsize
      memcpy(self.rcache + base1, data + base2, self.slotsize)
      # Refresh the atimes, sorted and indices data with the new slot info
      self.ratimes[nslot] = self.incseqn()
      nidx = self.slotlookup(nslot)
      self.rsorted[nidx] = key
      self.indices[:] = self.indices[self.sorted.argsort()]
      # The take() method seems similar in speed. This is striking,
      # because documentation says that it should be faster.
      # Profiling is saying that take maybe using memmove() and
      # this is *very* inneficient (at least with Linux/i386).
      #self.indices[:] = self.indices.take(self.sorted.argsort())
      self.sorted.sort()
      self.nextslot = self.nextslot + 1
    return nslot


  # Return the pointer to the data in cache
  cdef void *getitem(self, long nslot):

    self.getcount = self.getcount + 1
    self.ratimes[nslot] = self.incseqn()
    return self.rcache + nslot * self.slotsize


  # This version copies data in cache to data+start.
  # The user should be responsible to provide a large enough data buffer
  # to keep all the data.
  cdef long getitem2(self, long nslot, void *data, long start):
    cdef long base1, base2

    self.getcount = self.getcount + 1
    self.ratimes[nslot] = self.incseqn()
    # Copy the data in cache to destination
    base1 = start * self.itemsize;   base2 = nslot * self.slotsize
    memcpy(data + base1, self.rcache + base2, self.slotsize)
    return nslot


  def __repr__(self):
    return "<%s (%d elements)>" % (str(self.__class__), self.nslots)
