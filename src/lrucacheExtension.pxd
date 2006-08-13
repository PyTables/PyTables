# Declaration of instance variables for shared classes
# The LRUCache class is useful for caching general objects (like Nodes).
cdef class LRUCache:
  cdef object __dict, __heap
  cdef int size
  cdef long niter
  cdef long seqn_
  cdef object copy
  cdef long incseqn(self)
  cdef int contains(self, object key)
  cdef object cpop(self, object key)
  cdef object setitem(self, object key, object obj)

# The NumCache class is useful for caching numerical data in an efficient way.
cdef class NumCache:
  cdef object __dict, __heap
  cdef long seqn_, nextslot, setcount, getcount, cyclecount
  cdef long enableeverycycles
  cdef double lowesthr
  cdef int iscachedisabled, itemsize, nslots, slotsize
  cdef object name, cacheobj
  cdef void *rcache
  cdef long incseqn(self)
  cdef int checkhitratio(self)
  cdef int contains(self, object key)
  cdef long setitem(self, object key, void *data, long start)
  cdef void *getitem(self, object key)
  cdef long getitem2(self, object key, void *data, long start)

