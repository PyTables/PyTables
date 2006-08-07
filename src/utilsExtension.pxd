# Declaration of instance variables for shared classes
# The LRUCache class will be accessed in indexesExtension.pyx
cdef class LRUCache:
  cdef int size
  cdef long niter
  cdef long seqn_
  cdef object __dict
  cdef object __heap
  cdef object copy
  cdef long incseqn(self)
  cdef int contains(self, object key)
  cdef object cpop(self, object key)
  cdef object getitem(self, object key)
  cdef object getitem2(self, object key)
