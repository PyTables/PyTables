#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: June 20, 2005
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id: definitions.pyd 1018 2005-06-20 09:43:34Z faltet $
#
########################################################################

"""Here are some definitions for sharing between extensions.

"""

import sys

# Standard C functions.
cdef extern from "stdlib.h":
  ctypedef long size_t
  ctypedef long uintptr_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  char *strchr(char *s, int c)
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)
  void *memcpy(void *dest, void *src, size_t n)

cdef extern from "time.h":
  ctypedef int time_t


# Some helper routines from the Python API
cdef extern from "Python.h":

  # special types
  ctypedef int Py_ssize_t

  # references
  void Py_INCREF(object)
  void Py_DECREF(object)

  # To release global interpreter lock (GIL) for threading
  void Py_BEGIN_ALLOW_THREADS()
  void Py_END_ALLOW_THREADS()

  # Functions for integers
  object PyInt_FromLong(long)
  long PyInt_AsLong(object)
  object PyLong_FromLongLong(long long)
  long long PyLong_AsLongLong(object)

  # Functions for floating points
  object PyFloat_FromDouble(double)

  # Functions for strings
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  object PyString_FromString(char *)

  # Functions for lists
  int PyList_Append(object list, object item)

  # Functions for tuples
  object PyTuple_New(int)
  int PyTuple_SetItem(object, int, object)
  object PyTuple_GetItem(object, int)
  int PyTuple_Size(object tuple)

  # Functions for dicts
  int PyDict_Contains(object p, object key)
  object PyDict_GetItem(object p, object key)

  # Functions for objects
  object PyObject_GetItem(object o, object key)
  int PyObject_SetItem(object o, object key, object v)
  int PyObject_DelItem(object o, object key)
  long PyObject_Length(object o)
  int PyObject_Compare(object o1, object o2)
  int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)



#-----------------------------------------------------------------------------

# API for NumPy objects
cdef extern from "numpy/arrayobject.h":

  # Platform independent types
  ctypedef int npy_intp
  ctypedef signed int npy_int8
  ctypedef unsigned int npy_uint8
  ctypedef signed int npy_int16
  ctypedef unsigned int npy_uint16
  ctypedef signed int npy_int32
  ctypedef unsigned int npy_uint32
  ctypedef signed long long npy_int64
  ctypedef unsigned long long npy_uint64
  ctypedef float npy_float32
  ctypedef double npy_float64

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

  # Platform independent types
  cdef enum:
    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128

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

  # Functions
  object PyArray_GETITEM(object arr, void *itemptr)
  int PyArray_SETITEM(object arr, void *itemptr, object obj)
  dtype PyArray_DescrFromType(int type)
  object PyArray_Scalar(void *data, dtype descr, object base)

  # The NumPy initialization function
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
  ctypedef long long int64_t
  ctypedef long long haddr_t

  ctypedef struct hvl_t:
    size_t len                 # Length of VL data (in base type units)
    void *p                    # Pointer to VL data

  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT
  int H5P_DEFAULT, H5P_DATASET_XFER, H5S_ALL
  int H5P_FILE_CREATE, H5P_FILE_ACCESS
  int H5FD_LOG_LOC_WRITE, H5FD_LOG_ALL
  int H5I_INVALID_HID

  # The difference between a single file and a set of mounted files
  cdef enum H5F_scope_t:
    H5F_SCOPE_LOCAL     = 0,    # specified file handle only
    H5F_SCOPE_GLOBAL    = 1,    # entire virtual file
    H5F_SCOPE_DOWN      = 2     # for internal use only

  cdef enum H5G_obj_t:
    H5G_UNKNOWN = -1,           # Unknown object type
    H5G_GROUP,                  # Object is a group
    H5G_DATASET,                # Object is a dataset
    H5G_TYPE,                   # Object is a named data type
    H5G_LINK,                   # Object is a symbolic link
    ##H5G_UDLINK,                 # Object is a user-defined link  # 1.8.x

  # Values for fill value status
  cdef enum H5D_fill_value_t:
    H5D_FILL_VALUE_ERROR        = -1,
    H5D_FILL_VALUE_UNDEFINED    = 0,
    H5D_FILL_VALUE_DEFAULT      = 1,
    H5D_FILL_VALUE_USER_DEFINED = 2

  # HDF5 layouts
  cdef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    # raw data is very small
    H5D_CONTIGUOUS      = 1,    # the default
    H5D_CHUNKED         = 2,    # slow and fancy
    H5D_NLAYOUTS        = 3     # this one must be last!

  # Byte orders
  cdef enum H5T_order_t:
    H5T_ORDER_ERROR      = -1,  # error
    H5T_ORDER_LE         = 0,   # little endian
    H5T_ORDER_BE         = 1,   # bit endian
    H5T_ORDER_VAX        = 2,   # VAX mixed endian
    H5T_ORDER_NONE       = 3    # no particular order (strings, bits,..)

  # HDF5 signed enums
  cdef enum H5T_sign_t:
    H5T_SGN_ERROR        = -1,  # error
    H5T_SGN_NONE         = 0,   # this is an unsigned type
    H5T_SGN_2            = 1,   # two's complement
    H5T_NSGN             = 2    # this must be last!

  # HDF5 type classes
  cdef enum H5T_class_t:
    H5T_NO_CLASS         = -1,  # error
    H5T_INTEGER          = 0,   # integer types
    H5T_FLOAT            = 1,   # floating-point types
    H5T_TIME             = 2,   # date and time types
    H5T_STRING           = 3,   # character string types
    H5T_BITFIELD         = 4,   # bit field types
    H5T_OPAQUE           = 5,   # opaque types
    H5T_COMPOUND         = 6,   # compound types
    H5T_REFERENCE        = 7,   # reference types
    H5T_ENUM             = 8,   # enumeration types
    H5T_VLEN             = 9,   # variable-length types
    H5T_ARRAY            = 10,  # array types
    H5T_NCLASSES                # this must be last

  # Native types
  cdef enum:
    H5T_C_S1
    H5T_NATIVE_B8
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

  # "Standard" types
  cdef enum:
    H5T_STD_I8LE
    H5T_STD_I16LE
    H5T_STD_I32LE
    H5T_STD_I64LE
    H5T_STD_U8LE
    H5T_STD_U16LE
    H5T_STD_U32LE
    H5T_STD_U64LE
    H5T_STD_B8LE
    H5T_STD_B16LE
    H5T_STD_B32LE
    H5T_STD_B64LE
    H5T_IEEE_F32LE
    H5T_IEEE_F64LE
    H5T_STD_I8BE
    H5T_STD_I16BE
    H5T_STD_I32BE
    H5T_STD_I64BE
    H5T_STD_U8BE
    H5T_STD_U16BE
    H5T_STD_U32BE
    H5T_STD_U64BE
    H5T_STD_B8BE
    H5T_STD_B16BE
    H5T_STD_B32BE
    H5T_STD_B64BE
    H5T_IEEE_F32BE
    H5T_IEEE_F64BE

  # Types which are particular to UNIX (for Time types)
  cdef enum:
    H5T_UNIX_D32LE
    H5T_UNIX_D64LE
    H5T_UNIX_D32BE
    H5T_UNIX_D64BE

  # The order to retrieve atomic native datatype
  cdef enum H5T_direction_t:
    H5T_DIR_DEFAULT     = 0,    #default direction is inscendent
    H5T_DIR_ASCEND      = 1,    #in inscendent order
    H5T_DIR_DESCEND     = 2     #in descendent order

  # Codes for defining selections
  cdef enum H5S_seloper_t:
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

  # Character set to use for text strings
  cdef enum H5T_cset_t:
    H5T_CSET_ERROR       = -1,  # error
    H5T_CSET_ASCII       = 0,   # US ASCII
    H5T_CSET_UTF8        = 1,   # UTF-8 Unicode encoding
    H5T_CSET_RESERVED_2  = 2,
    H5T_CSET_RESERVED_3  = 3,
    H5T_CSET_RESERVED_4  = 4,
    H5T_CSET_RESERVED_5  = 5,
    H5T_CSET_RESERVED_6  = 6,
    H5T_CSET_RESERVED_7  = 7,
    H5T_CSET_RESERVED_8  = 8,
    H5T_CSET_RESERVED_9  = 9,
    H5T_CSET_RESERVED_10 = 10,
    H5T_CSET_RESERVED_11 = 11,
    H5T_CSET_RESERVED_12 = 12,
    H5T_CSET_RESERVED_13 = 13,
    H5T_CSET_RESERVED_14 = 14,
    H5T_CSET_RESERVED_15 = 15


  #------------------------------------------------------------------

  # HDF5 API

  # Version functions
  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum )
  herr_t H5check_version(unsigned majnum, unsigned minnum,
                         unsigned relnum )

  # Operations with files
  hid_t  H5Fcreate(char *filename, unsigned int flags,
                   hid_t create_plist, hid_t access_plist)
  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id)
  herr_t H5Fclose (hid_t file_id)
  htri_t H5Fis_hdf5(char *name)
  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope)
  herr_t H5Fget_vfd_handle(hid_t file_id, hid_t fapl_id, void **file_handle)

  # Operations with groups
  hid_t  H5Gcreate(hid_t loc_id, char *name, size_t size_hint )
  hid_t  H5Gopen(hid_t loc_id, char *name )
  herr_t H5Gclose(hid_t group_id)
  herr_t H5Gunlink (hid_t file_id, char *name)
  herr_t H5Gmove(hid_t loc_id, char *src, char *dst)
  herr_t H5Gmove2(hid_t src_loc_id, char *src_name,
                  hid_t dst_loc_id, char *dst_name )
  # For dealing with datasets
  hid_t  H5Dopen(hid_t file_id, char *name)
  herr_t H5Dclose(hid_t dset_id)
  herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                 hid_t file_space_id, hid_t plist_id, void *buf)
  herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf)
  hid_t H5Dget_type(hid_t dset_id)
  hid_t H5Dget_space(hid_t dset_id)
  herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id, hid_t plist_id,
                         void *buf)
  hid_t H5Dget_create_plist(hid_t dataset_id)

  # Functions for dealing with dataspaces
  hid_t H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[])
  int H5Sget_simple_extent_ndims(hid_t space_id)
  int H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[],
                                hsize_t maxdims[])
  herr_t H5Sselect_all(hid_t spaceid)
  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,
                             hsize_t start[], hsize_t _stride[],
                             hsize_t count[], hsize_t _block[])
  herr_t H5Sselect_elements(hid_t space_id, H5S_seloper_t op,
                            size_t num_elements, hsize_t *coord)
  herr_t H5Sclose(hid_t space_id)


  # Functions for dealing with datatypes
  H5T_class_t H5Tget_class(hid_t type_id)
  hid_t H5Tget_super(hid_t type)
  H5T_sign_t H5Tget_sign(hid_t type_id)
  H5T_order_t H5Tget_order(hid_t type_id)
  size_t H5Tget_size(hid_t type_id)
  herr_t H5Tset_size(hid_t type_id, size_t size)
  herr_t H5Tset_precision(hid_t type_id, size_t prec)
  hid_t  H5Tcreate(H5T_class_t type, size_t size)
  hid_t  H5Tcopy(hid_t type_id)
  herr_t H5Tclose(hid_t type_id)

  # Operations defined on string data types
  htri_t H5Tis_variable_str(hid_t dtype_id)

  # Operations for compound data types
  int    H5Tget_nmembers(hid_t type_id)
  char  *H5Tget_member_name(hid_t type_id, unsigned membno)
  hid_t  H5Tget_member_type(hid_t type_id, unsigned membno)
  hid_t  H5Tget_native_type(hid_t type_id, H5T_direction_t direction)
  herr_t H5Tget_member_value(hid_t type_id, int membno, void *value)
  int    H5Tget_offset(hid_t type_id)
  herr_t H5Tinsert(hid_t parent_id, char *name, size_t offset,
                   hid_t member_id)
  herr_t H5Tpack(hid_t type_id)

  # Operations for enumerated data types
  hid_t H5Tenum_create(hid_t base_id)
  herr_t H5Tenum_insert(hid_t type, char *name, void *value)

  # Operations for array data types
  hid_t H5Tarray_create(hid_t base_id, int ndims, hsize_t dims[], int perm[])
  int   H5Tget_array_ndims(hid_t type_id)
  int   H5Tget_array_dims(hid_t type_id, hsize_t dims[], int perm[])

  # Operations with attributes
  herr_t H5Adelete(hid_t loc_id, char *name)
  int    H5Aget_num_attrs(hid_t loc_id)
  size_t H5Aget_name(hid_t attr_id, size_t buf_size, char *buf)
  hid_t  H5Aopen_idx(hid_t loc_id, unsigned int idx)
  herr_t H5Aread(hid_t attr_id, hid_t mem_type_id, void *buf)
  herr_t H5Aclose(hid_t attr_id)

  # Operations with properties
  hid_t  H5Pcreate(hid_t plist_id)
  herr_t H5Pclose(hid_t plist_id)
  herr_t H5Pset_cache(hid_t plist_id, int mdc_nelmts, int rdcc_nelmts,
                      size_t rdcc_nbytes, double rdcc_w0)
  herr_t H5Pset_sieve_buf_size(hid_t fapl_id, hsize_t size)
  herr_t H5Pset_fapl_log(hid_t fapl_id, char *logfile,
                         unsigned int flags, size_t buf_size)
  H5D_layout_t H5Pget_layout(hid_t plist)
  int H5Pget_chunk(hid_t plist, int max_ndims, hsize_t *dims)
  herr_t H5Pset_fapl_core(hid_t fapl_id, size_t increment,
                          hbool_t backing_store)


# Specific HDF5 functions for PyTables
cdef extern from "H5ATTR.h":
  herr_t H5ATTRget_attribute(hid_t loc_id, char *attr_name,
                             hid_t type_id, void *data)
  herr_t H5ATTRget_attribute_string(hid_t loc_id, char *attr_name,
                                    char **attr_value)
  herr_t H5ATTRset_attribute(hid_t obj_id, char *attr_name,
                             hid_t type_id, size_t rank,  hsize_t *dims,
                             char *attr_data )
  herr_t H5ATTRset_attribute_string(hid_t loc_id, char *attr_name,
                                    char *attr_data)
  herr_t H5ATTRfind_attribute(hid_t loc_id, char *attr_name)
  herr_t H5ATTRget_type_ndims(hid_t loc_id, char *attr_name,
                              hid_t *type_id, H5T_class_t *class_id,
                              size_t *type_size, int *rank)
  herr_t H5ATTRget_dims(hid_t loc_id, char *attr_name, hsize_t *dims)


# Functions for operations with ARRAY
cdef extern from "H5ARRAY.h":
  herr_t H5ARRAYget_ndims(hid_t dataset_id, int *rank)
  herr_t H5ARRAYget_info(hid_t dataset_id, hid_t type_id, hsize_t *dims,
                         hsize_t *maxdims, H5T_class_t *super_class_id,
                         char *byteorder)


# Some utilities
cdef extern from "utils.h":
  herr_t set_cache_size(hid_t file_id, size_t cache_size)
  int get_objinfo(hid_t loc_id, char *name)
  object Giterate(hid_t parent_id, hid_t loc_id, char *name)
  object Aiterate(hid_t loc_id)
  object H5UIget_info(hid_t loc_id, char *name, char *byteorder)
  hsize_t get_len_of_range(hsize_t lo, hsize_t hi, hsize_t step)
  hid_t  create_ieee_complex64(char *byteorder)
  hid_t  create_ieee_complex128(char *byteorder)
  herr_t set_order(hid_t type_id, char *byteorder)
  herr_t get_order(hid_t type_id, char *byteorder)
  int    is_complex(hid_t type_id)
  herr_t truncate_dset(hid_t dataset_id, int maindim, hsize_t size)

# Type conversion routines
cdef extern from "typeconv.h":
  void conv_float64_timeval32(void *base,
                              unsigned long byteoffset,
                              unsigned long bytestride,
                              long long nrecords,
                              unsigned long nelements,
                              int sense)

# Blosc registration
cdef extern from "blosc_filter.h":
  int register_blosc(char **version, char **date)
