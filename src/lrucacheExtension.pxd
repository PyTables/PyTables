from numpydefs cimport ndarray

# Declaration of instance variables for shared classes
# The NodeCache class is useful for caching general objects (like Nodes).
cdef class NodeCache:
  cdef object nodes, paths
  cdef int size, lsize
  cdef object setitem(self, object path, object node)
  cdef long getslot(self, object path)
  cdef object cpop(self, object path)


# Base class for other caches
cdef class BaseCache:
  cdef int iscachedisabled
  cdef long setcount, getcount, cyclecount, enableeverycycles
  cdef double lowesthr
  cdef int checkhitratio(self, int cachesize)


# The NumCache class is useful for caching numerical data in an efficient way.
cdef class NumCache(BaseCache):
  cdef long seqn_, nextslot
  cdef int itemsize, nslots, slotsize
  cdef object name
  cdef ndarray cacheobj, sorted, indices, atimes
  cdef void *rcache
  cdef long long *rsorted
  cdef unsigned short *rindices
  cdef long *ratimes
  cdef long incseqn(self)
  cdef long slotlookup(self, unsigned short x)
  cdef long getslot(self, long long key)
  cdef long setitem(self, long long key, void *data, long start)
  cdef void *getitem(self, long nslot)
  cdef long getitem2(self, long nslot, void *data, long start)


# The ObjectCache class is useful for python objects in an efficient way.
cdef class ObjectCache(BaseCache):
  cdef long seqn_, nextslot
  cdef int itemsize, nslots, slotsize
  cdef object name, mrunode
  cdef object __heap,  __list,  __dict
  cdef long incseqn(self)
  cdef long setitem_(self, object key, object value)
  cdef long getslot_(self, object key)
  cdef object getitem_(self, long nslot)
