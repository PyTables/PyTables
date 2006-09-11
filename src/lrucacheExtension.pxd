from numpydefs cimport ndarray

# Declaration of instance variables for shared classes
# The NodeCache class is useful for caching general objects (like Nodes).
cdef class NodeCache:
  cdef long nextslot, nslots
  cdef object nodes, paths
  cdef object setitem(self, object path, object node)
  cdef long getslot(self, object path)
  cdef object cpop(self, object path)


# Base class for other caches
cdef class BaseCache:
  cdef int iscachedisabled, incsetcount
  cdef long setcount, getcount, containscount
  cdef long cyclecount, enableeverycycles
  cdef long seqn_, nextslot, nslots
  cdef long *ratimes
  cdef double lowesthr
  cdef ndarray atimes
  cdef object name
  cdef int checkhitratio(self)
  cdef int couldenablecache_(self)
  cdef long incseqn(self)


# The ObjectCache class is useful for general python objects
cdef class ObjectCache(BaseCache):
  cdef long maxcachesize, cachesize, maxobjsize
  cdef long *rsizes
  cdef ndarray sizes
  cdef object __list, __dict, mrunode
  cdef removeslot_(self, long nslot)
  cdef addslot_(self, long nslot, long size, object key, object value)
  cdef long setitem_(self, object key, object value, long size)
  cdef long getslot_(self, object key)
  cdef object getitem_(self, long nslot)


# The NumCache class is useful for caching numerical data in an efficient way
cdef class NumCache(BaseCache):
  cdef long itemsize, slotsize
  cdef ndarray cacheobj, sorted, indices
  cdef void *rcache
  cdef long long *rsorted
  cdef unsigned short *rindices
  cdef removeslot_(self, long nslot)
  cdef addslot_(self, long nslot, long start, long long key, void *data)
  cdef long setitem(self, long long key, void *data, long start)
  cdef long slotlookup(self, unsigned short x)
  cdef long getslot(self, long long key)
  cdef void *getitem(self, long nslot)
  cdef long getitem2(self, long nslot, void *data, long start)

