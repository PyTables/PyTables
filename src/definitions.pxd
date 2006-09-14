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

  # To access dicts
  int PyDict_Contains(object p, object key)
  object PyDict_GetItem(object p, object key)

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


# API for NumPy objects
cdef extern from "numpy/arrayobject.h":

  # Types
  ctypedef int npy_intp

  cdef enum NPY_TYPES:
    NPY_BOOL
    NPY_BYTE
    NPY_UBYTE
    NPY_SHORT
    NPY_USHORT
    NPY_INT
    NPY_UINT
    NPY_LONG
    NPY_ULONG
    NPY_LONGLONG
    NPY_ULONGLONG
    NPY_FLOAT
    NPY_DOUBLE
    NPY_LONGDOUBLE
    NPY_CFLOAT
    NPY_CDOUBLE
    NPY_CLONGDOUBLE
    NPY_OBJECT
    NPY_STRING
    NPY_UNICODE
    NPY_VOID
    NPY_NTYPES
    NPY_NOTYPE

  # Functions
  object PyArray_GETITEM(object arr, void *itemptr)
  int PyArray_SETITEM(object arr, void *itemptr, object obj)

  # Classes
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

  # The numpy initialization funtion
  void import_array()


#-----------------------------------------------------------------------------

# Structs and types from HDF5
cdef extern from "hdf5.h":

  ctypedef int hid_t  # In H5Ipublic.h
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  # hsize_t should be unsigned, but Windows platform does not support
  # such an unsigned long long type.
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t

  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT
  int H5P_DEFAULT, H5P_DATASET_XFER, H5S_ALL
  int H5P_FILE_CREATE, H5P_FILE_ACCESS
  int H5FD_LOG_LOC_WRITE, H5FD_LOG_ALL
  int H5I_INVALID_HID

  ctypedef struct hvl_t:
    size_t len                 # Length of VL data (in base type units) */
    void *p                    # Pointer to VL data */

  ctypedef enum H5G_obj_t:
    H5G_UNKNOWN = -1,           # Unknown object type          */
    H5G_LINK,                   # Object is a symbolic link    */
    H5G_GROUP,                  # Object is a group            */
    H5G_DATASET,                # Object is a dataset          */
    H5G_TYPE,                   # Object is a named data type  */
    H5G_RESERVED_4,             # Reserved for future use      */
    H5G_RESERVED_5,             # Reserved for future use      */
    H5G_RESERVED_6,             # Reserved for future use      */
    H5G_RESERVED_7              # Reserved for future use      */

  cdef struct H5G_stat_t:
    unsigned long fileno[2]
    unsigned long objno[2]
    unsigned nlink
    H5G_obj_t type  # new in HDF5 1.6
    time_t mtime
    size_t linklen
    #H5O_stat_t ohdr           # Object header information. New in HDF5 1.6

  cdef enum H5T_class_t:
    H5T_NO_CLASS         = -1,  #error                                      */
    H5T_INTEGER          = 0,   #integer types                              */
    H5T_FLOAT            = 1,   #floating-point types                       */
    H5T_TIME             = 2,   #date and time types                        */
    H5T_STRING           = 3,   #character string types                     */
    H5T_BITFIELD         = 4,   #bit field types                            */
    H5T_OPAQUE           = 5,   #opaque types                               */
    H5T_COMPOUND         = 6,   #compound types                             */
    H5T_REFERENCE        = 7,   #reference types                            */
    H5T_ENUM             = 8,   #enumeration types                          */
    H5T_VLEN             = 9,   #Variable-Length types                      */
    H5T_ARRAY            = 10,  #Array types                                */
    H5T_NCLASSES                #this must be last                          */

  # The difference between a single file and a set of mounted files
  cdef enum H5F_scope_t:
    H5F_SCOPE_LOCAL     = 0,    # specified file handle only
    H5F_SCOPE_GLOBAL    = 1,    # entire virtual file
    H5F_SCOPE_DOWN      = 2     # for internal use only

  cdef enum H5T_sign_t:
    H5T_SGN_ERROR        = -1,  #error                                      */
    H5T_SGN_NONE         = 0,   #this is an unsigned type                   */
    H5T_SGN_2            = 1,   #two's complement                           */
    H5T_NSGN             = 2    #this must be last!                         */

  cdef enum H5G_link_t:
    H5G_LINK_ERROR      = -1,
    H5G_LINK_HARD       = 0,
    H5G_LINK_SOFT       = 1

  # Native types
  cdef enum:
    H5T_NATIVE_CHAR
    H5T_NATIVE_SCHAR
    H5T_NATIVE_UCHAR
    H5T_NATIVE_SHORT
    H5T_NATIVE_USHORT
    H5T_NATIVE_INT
    H5T_NATIVE_UINT
    H5T_NATIVE_LONG
    H5T_NATIVE_ULONG
    H5T_NATIVE_LLONG
    H5T_NATIVE_ULLONG
    H5T_NATIVE_FLOAT
    H5T_NATIVE_DOUBLE
    H5T_NATIVE_LDOUBLE

  ctypedef enum H5S_seloper_t:
    H5S_SELECT_NOOP      = -1,
    H5S_SELECT_SET       = 0,
    H5S_SELECT_OR,
    H5S_SELECT_AND,
    H5S_SELECT_XOR,
    H5S_SELECT_NOTB,
    H5S_SELECT_NOTA,
    H5S_SELECT_APPEND,
    H5S_SELECT_PREPEND,
    H5S_SELECT_INVALID    # Must be the last one


#-----------------------------------------------------------------------------


# Conversion tables

# Conversion from NumPy types supported for native HDF5 attributes
NPTypeToHDF5 = {
  NPY_BYTE      : H5T_NATIVE_SCHAR,
  NPY_SHORT     : H5T_NATIVE_SHORT,
  NPY_INT       : H5T_NATIVE_INT,
  NPY_LONGLONG  : H5T_NATIVE_LLONG,
  NPY_UBYTE     : H5T_NATIVE_UCHAR,
  NPY_USHORT    : H5T_NATIVE_USHORT,
  NPY_UINT      : H5T_NATIVE_UINT,
  NPY_ULONGLONG : H5T_NATIVE_ULLONG,
  NPY_FLOAT32   : H5T_NATIVE_FLOAT,
  NPY_FLOAT64   : H5T_NATIVE_DOUBLE
  }


# Conversion from numpy int codes to strings
NPTypeToString = {
  NPY_BOOL:     'Bool',
  NPY_BYTE:     'Int8',      NPY_UBYTE:'UInt8',
  NPY_SHORT:    'Int16',     NPY_USHORT:'UInt16',
  NPY_INT:      'Int32',     NPY_UINT:'UInt32',
  NPY_LONGLONG: 'Int64',     NPY_ULONGLONG:'UInt64',
  NPY_FLOAT:    'Float32',   NPY_DOUBLE:'Float64',
  NPY_CFLOAT:   'Complex64', NPY_CDOUBLE:'Complex128',
  NPY_STRING:   'CharType',
  # Special cases:
  ord('t'):     'Time32',    ord('T'):'Time64',
  ord('e'):     'Enum',
  }

