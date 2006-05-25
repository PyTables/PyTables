#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: June 20, 2005
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id: definitions.pyd 1018 2005-06-20 09:43:34Z faltet $
#
########################################################################

"""Here are some definitions for sharing between extensions.

"""

# Some helper routines from the Python API
cdef extern from "Python.h":

  # references
  void Py_INCREF(object)
  void Py_DECREF(object)

  # To access tuples
  object PyTuple_New(int)
  int PyTuple_SetItem(object, int, object)
  object PyTuple_GetItem(object, int)
  int PyTuple_Size(object tuple)

  # To access integers
  object PyInt_FromLong(long)
  long PyInt_AsLong(object)
  object PyLong_FromLongLong(long long)
  long long PyLong_AsLongLong(object)

  # To access double
  object PyFloat_FromDouble(double)

  # To access strings
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  object PyString_FromString(char *)

  # To access to Memory (Buffer) objects presents in numarray
  object PyBuffer_FromMemory(void *ptr, int size)
  object PyBuffer_FromReadWriteMemory(void *ptr, int size)
  object PyBuffer_New(int size)
  int PyObject_CheckReadBuffer(object)
  int PyObject_AsReadBuffer(object, void **rbuf, int *len)
  int PyObject_AsWriteBuffer(object, void **rbuf, int *len)

  # To release global interpreter lock (GIL) for threading
  void Py_BEGIN_ALLOW_THREADS()
  void Py_END_ALLOW_THREADS()


# Functions from numarray API
cdef extern from "numarray/libnumarray.h":

  cdef enum:
    MAXDIM  # Maximum dimensionality for arrays

  object NA_getPythonScalar(object, long)
  int  NA_setFromPythonScalar(object, long, object)
  long NA_getBufferPtrAndSize(object, int, void**)

  # The numarray initialization funtion
  void import_libnumarray()

