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

    LRUArray
    NumArray

Functions:

Misc variables:

    __version__
"""

from heapq import heappush, heappop, heapreplace, heapify

import numarray

from lrucache import CacheKeyError
from constants import ENABLE_EVERY_CYCLES, LOWEST_HIT_RATIO

from definitions cimport import_libnumarray, NA_getBufferPtrAndSize


#----------------------------------------------------------------------

# Standard C functions.
cdef extern from "stdlib.h":
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  void *memcpy(void *dest, void *src, size_t n)

# Python API functions.
cdef extern from "Python.h":
  # How to declare a Py_ssize_t ??
  #ctypedef long Py_ssize_t
  int PySequence_DelItem(object o, long i)
  object PyDict_GetItem(object p, object key)
  int PyDict_Contains(object p, object key)
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)
  long PyObject_Length(object o)
  int PyObject_Compare(object o1, object o2)


#----------------------------------------------------------------------------

# Initialization code

# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()

#---------------------------------------------------------------------------


# ------- Implementation of the LRUcache classes in Pyrex ---------

#*********************** Important note! *****************************
#The code behind has been carefully tuned to serve the needs of
#PyTables cache for nodes. As a consequence, it is no longer
#appropriate as a general LRU cache implementation. You have been
#warned!.  F. Altet 2006-08-08
#*********************************************************************


cdef class LRUNode:
  """Record of a cached value. Not for public consumption."""
  cdef object key, obj
  cdef long  atime


  def __init__(self, key, obj, timestamp):
    object.__init__(self)
    self.key = key
    self.obj = obj
    self.atime = timestamp


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


DEFAULT_SIZE = 16
"""Default size of a new LRUCache object, if no 'size' argument is given."""


cdef class LRUCache:
  """Least-Recently-Used (LRU) cache.
  """
  # This class variables are declared in utilsExtension.pxd


  cdef long incseqn(self):
    cdef LRUNode node

    self.seqn_ = self.seqn_ + 1
    # Check that the counter is ok
    #if self.seqn_ > 1000:  # Only for testing!
    if self.seqn_ < 0:
      # Ooops, the counter has run out of range!
      # Reset all the priorities to 0
      for node in self.__heap:
        node.atime = 0
      # Set the counter to 1 (to indicate that it is newer than existing ones)
      self.seqn_ = 1

    #print self.seqn_
    return self.seqn_


  property seqn:
    "The sequential key."
    def __get__(self):
      return self.incseqn()


  def __init__(self, size=DEFAULT_SIZE):
    """Maximum size of the cache.
    If more than 'size' elements are added to the cache,
    the least-recently-used ones will be discarded."""

    if size <= 0:
      raise ValueError, size
    elif type(size) is not type(0):
      raise TypeError, size
    self.__heap = []
    self.__dict = {}
    self.seqn_ = 0
    self.size = size

  def __len__(self):
    return len(self.__heap)


  def __contains__(self, key):
    return PyDict_Contains(self.__dict, key)


  # This is meant to be called from Pyrex extensions
  cdef int contains(self, key):
    # We don't check for -1 as this should never fail
    return PyDict_Contains(self.__dict, key)


  def __setitem__(self, key, obj):
    self.setitem(key, obj)


  # This version is meant to be called from extensions
  cdef setitem(self, key, obj):
    cdef LRUNode node, lru
    if PyDict_Contains(self.__dict, key):
      node = PyObject_GetItem(self.__dict, key)
      node.obj = obj
      node.atime = self.incseqn()
      heapify(self.__heap)
    else:
      node = LRUNode(key, obj, self.incseqn())
      PyObject_SetItem(self.__dict, key, node)
      # Check if we are growing out of space
      if PyObject_Length(self.__heap) == self.size:
        lru = heapreplace(self.__heap, node)
        PyObject_DelItem(self.__dict, lru.key)
      else:
        heappush(self.__heap, node)


  def __getitem__(self, key):
    cdef LRUNode node

    node = self.__dict.pop(key, None)
    if <object>node is None:
      raise CacheKeyError(key)
    else:
      node.atime = self.incseqn()
      heapify(self.__heap)
      return node.obj


  def __delitem__(self, key):
    self.pop(key)


  def pop(self, key):
    return self.cpop(key)


  cdef object cpop(self, object key):
    cdef LRUNode node, node2
    cdef long idx

    #node = self.__dict.pop(key)
    node = PyObject_GetItem(self.__dict, key)
    PyObject_DelItem(self.__dict, key)
    # The next line makes a segfault to happen
    #self.__heap.remove(<object>node)
    # Workaround. This workaround has the virtue that only a heapify is
    # done in case the node is the LRU.
    for idx from 0 <= idx < len(self.__heap):
      node2 = PyObject_GetItem(self.__heap, idx)
      if node2 is node:
      # The next line is not equivalent to "is"...
      #if PyObject_Compare(node, node2) == 0:
        if idx == 0:
          # It turns that the element to be removed is the LRU.
          # Extract it and let the heap invariant.
          heappop(self.__heap)
        else:
          # The node to be removed is in the middle of the heap, so we
          # don't need to maintain the heap invariant (the next
          # insertion will do that).
          # Using del here causes another segfault, I don't know why,
          # but this has probably to do with wrong ref counts in Pyrex :-(
          # Fortunately, .pop() method seems to work...
          # del self.__heap[idx]
          #self.__heap.pop(idx)
          PySequence_DelItem(self.__heap, idx)
          # PyObject delitem also works
          #PyObject_DelItem(self.__heap, idx)
        break
    return node.obj


  def __iter__(self):

    self.copy = self.__heap[:]
    self.niter = len(self.copy)
    return self


  def __next__(self):
    cdef LRUNode node

    if self.niter > 0:
      node = heappop(self.copy)
      self.niter = self.niter - 1
      return node.key
    raise StopIteration


  def __repr__(self):
    return "<%s (%d elements)>" % (str(self.__class__), len(self.__heap))



#  Minimalistic LRU cache implementation for numerical data

#*********************** Important note! ****************************
#The code behind has been carefully tuned to serve the needs of
#caching numarray (or numpy) data. As a consequence, it is no longer
#appropriate as a general LRU cache implementation. You have been
#warned!. F. Altet 2006-08-09
#********************************************************************

cdef class NumNode:
  """Record of a cached value. Not for public consumption."""
  cdef object key
  cdef long nslot, atime

  def __init__(self, object key, long nslot, long atime):
    object.__init__(self)
    self.key = key
    self.nslot = nslot
    self.atime = atime


  def __cmp__(self, NumNode other):
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

  def __init__(self, object shape, int itemsize, object name):
    """Maximum size of the cache.
    If more than 'nslots' elements are added to the cache,
    the least-recently-used ones will be discarded.

    Parameters:
    shape - The rectangular shape of the cache (nslots, nelemsperslot)
    itemsize - The size of the element base in cache
    name - A descriptive name for this cache
    """

    self.nslots = shape[0];  self.slotsize = shape[1]*itemsize;
    self.itemsize = itemsize;  self.name = name
    if self.nslots <= 0:
      raise ValueError, self.nslots
    self.__heap = [];  self.__dict = {}
    self.seqn_ = 0;  self.nextslot = 0
    self.setcount = 0;  self.getcount = 0;  self.cyclecount = 0
    self.iscachedisabled = False  # Cache is enabled by default
    self.enableeverycycles = ENABLE_EVERY_CYCLES
    self.lowesthr = LOWEST_HIT_RATIO
    # The cache object where all data will go
    self.cacheobj = numarray.array(None, type="UInt8",
                                   shape=(self.nslots, self.slotsize))
    NA_getBufferPtrAndSize(self.cacheobj._data, 1, &self.rcache)


  cdef long incseqn(self):
    cdef NumNode node

    self.seqn_ = self.seqn_ + 1
    if self.seqn_ < 0:
      # Ooops, the counter has run out of range!
      # Reset all the priorities to 0
      for node in self.__heap:
        node.atime = 0
      # Set the counter to 1 (to indicate that it is newer than existing ones)
      self.seqn_ = 1
    return self.seqn_


  cdef int contains(self, object key):
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

  cdef long setitem(self, object key, void *data, long start):
    cdef NumNode node, lru
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
      base1 = nslot * self.slotsize;  base2 = start * self.itemsize
      memcpy(self.rcache + base1, data + base2, self.slotsize)
      # Add a new node with references to the copied data to the LRUCache
      node = NumNode(key, nslot, self.incseqn())
      heappush(self.__heap, node)
      PyObject_SetItem(self.__dict, key, node)  # Can't fail
    return nslot


  # Return the pointer to the data in cache
  cdef void *getitem(self, object key):
    cdef NumNode node
    cdef long base

    self.getcount = self.getcount + 1
    node = PyObject_GetItem(self.__dict, key)
    node.atime = self.incseqn()
    base = node.nslot * self.slotsize
    return self.rcache + base


  # This version copies data in cache to data+start.
  # The user should be responsible to provide a large enough data buffer
  # to the memcpy to succeed.
  cdef long getitem2(self, object key, void *data, long start):
    cdef NumNode node
    cdef long base1, base2

    self.getcount = self.getcount + 1
    node = PyObject_GetItem(self.__dict, key)
    node.atime = self.incseqn()
    # Copy the data in cache to destination
    base1 = start * self.itemsize;   base2 = node.nslot * self.slotsize
    memcpy(data + base1, self.rcache + base2, self.slotsize)
    return node.nslot


  def __repr__(self):
    return "<%s (%d elements)>" % (str(self.__class__), len(self.__heap))
