# API for NumPy objects
cdef extern from "numpy/arrayobject.h":
  ctypedef int npy_intp

  ctypedef extern class numpy.dtype [object PyArray_Descr]:
    cdef int type_num, elsize, alignment
    cdef char type, kind, byteorder, hasobject
    cdef object fields, typeobj

  ctypedef extern class numpy.ndarray [object PyArrayObject]:
    cdef char *data
    cdef int nd
    cdef npy_intp *dimensions
    cdef npy_intp *strides
    cdef object base
    cdef dtype descr
    cdef int flags

  # Functions
  object PyArray_GETITEM(object arr, void *itemptr)
  int PyArray_SETITEM(object arr, void *itemptr, object obj)

  # The numpy initialization funtion
  void import_array()

