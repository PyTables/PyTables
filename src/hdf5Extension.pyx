#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/hdf5Extension.pyx,v $
#       $Id: hdf5Extension.pyx,v 1.146 2004/10/05 19:22:20 falted Exp $
#
########################################################################

"""Pyrex interface between PyTables and HDF5 library.

See the metaIsRecord for a deep explanation on how exactly this works.

Classes (type extensions):

    File
    Group
    Table
    Array

Functions:

    isHDF5(filename)
    isPyTablesFile(filename)
    getHDF5LibraryVersion()
    getExtCVSVersion()
    getPyTablesVersion()
    

Misc variables:

    __version__

"""

__version__ = "$Revision: 1.146 $"


import sys, os
import warnings
import types, cPickle
import numarray
from numarray import records
from numarray import strings
from numarray import memmap
from utils import calcBufferSize
try:
  import zlib
  zlib_imported = 1
except:
  zlib_imported = 0

# C funtions and variable declaration from its headers

# Type size_t is defined in stdlib.h
cdef extern from "stdlib.h":
  #ctypedef int size_t
  # The correct correspondence between size_t and a basic type is *long*
  # instead of int, because they are the same size even for 64-bit platforms
  # F. Alted 2003-01-08
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)
  double atof(char *nptr)

cdef extern from "time.h":
  ctypedef int time_t

# The next has been substituted by equivalents in Python, so that this
# functions could be accessible in Windows systems
# Thanks to Shack Toms for this!
# F. Alted 2004-10-01
# cdef extern from "math.h":
#   double nextafter(double x, double y)
#   float nextafterf(float x, float y)

# Funtions for printing in C
cdef extern from "stdio.h":
  int sprintf(char *str,  char *format, ...)
  int snprintf(char *str, size_t size, char *format, ...)

cdef extern from "string.h":
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)
  void *memcpy(void *dest, void *src, size_t n)

# Some helper routines from the Python API
cdef extern from "Python.h":
  # For parsing tuples
  int PyArg_ParseTuple(object args, char *format, ...)

  # To access tuples
  object PyTuple_New(int)
  int PyTuple_SetItem(object, int, object)
  object PyTuple_GetItem(object, int)
  int PyTuple_Size(object tuple)

  int Py_DECREF(object)
  int Py_INCREF(object)
  
  # To access integers
  object PyInt_FromLong(long)
  long PyInt_AsLong(object)
  object PyLong_FromLongLong(long long)

  # To access double
  object PyFloat_FromDouble(double)
  
  # To access strings
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  object PyString_FromString(char *)

# To access to str and tuple structures. This does not work with Pyrex 0.8
# This is not necessary, though
#  ctypedef class __builtin__.str [object PyStringObject]:
#    cdef char *ob_sval
#    cdef int  ob_size

#   ctypedef class __builtin__.tuple [object PyTupleObject]:
#     cdef object ob_item
#     cdef int    ob_size

  # To access to Memory (Buffer) objects presents in numarray
  object PyBuffer_FromMemory(void *ptr, int size)
  object PyBuffer_FromReadWriteMemory(void *ptr, int size)
  object PyBuffer_New(int size)
  int PyObject_CheckReadBuffer(object)
  int PyObject_AsReadBuffer(object, void **rbuf, int *len)
  int PyObject_AsWriteBuffer(object, void **rbuf, int *len)

  # To get the indices of a slice in python 2.2
  int PySlice_GetIndices(object r, int length,
                         int *start, int *stop, int *step)

# Structs and functions from numarray
cdef extern from "numarray/numarray.h":

  ctypedef enum NumRequirements:
    NUM_CONTIGUOUS
    NUM_NOTSWAPPED
    NUM_ALIGNED
    NUM_WRITABLE
    NUM_C_ARRAY
    NUM_UNCONVERTED

  ctypedef enum NumarrayByteOrder:
    NUM_LITTLE_ENDIAN
    NUM_BIG_ENDIAN

  cdef enum:
    UNCONVERTED
    C_ARRAY

  ctypedef enum NumarrayType:
    tAny
    tBool
    tInt8
    tUInt8
    tInt16
    tUInt16
    tInt32
    tUInt32
    tInt64
    tUInt64
    tFloat32
    tFloat64
    tComplex32
    tComplex64
    tObject
    tDefault
    tLong
  
# Declaration for the PyArrayObject
# This does not work with pyrex 0.8 and better anymore. It's worth
# analyzing what's going on.
  
  struct PyArray_Descr:
     int type_num, elsize
     char type

  #ctypedef class numarray.numarraycore.NumArray [object PyArrayObject]:
  # This does not work because NumArray is actually a python class
  # derived from the c extension class _numarray.
  # Thanks to Simon Burton for pointing out this. 2003-01-12
  ctypedef class numarray._numarray._numarray [object PyArrayObject]:
    # Compatibility with Numeric
    cdef char *data
    cdef int nd
    cdef int *dimensions, *strides
    cdef object base
    cdef PyArray_Descr *descr
    cdef int flags
    # New attributes for numarray objects
    cdef object _data         # object must meet buffer API */
    cdef object _shadows      # ill-behaved original array. */
    cdef int    nstrides      # elements in strides array */
    cdef long   byteoffset    # offset into buffer where array data begins */
    cdef long   bytestride    # basic seperation of elements in bytes */
    cdef long   itemsize      # length of 1 element in bytes */
    cdef char   byteorder     # NUM_BIG_ENDIAN, NUM_LITTLE_ENDIAN */
    cdef char   _aligned      # test override flag */
    cdef char   _contiguous   # test override flag */

  # The numarray initialization funtion
  void import_libnumarray()
  #void import_array()
    
# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()
#import_array()

# CharArray type
#CharType = recarray.CharType
CharType = records.CharType

# Conversion tables from/to classes to the numarray enum types
toenum = {numarray.Bool:tBool,   # Boolean type added
          numarray.Int8:tInt8,       numarray.UInt8:tUInt8,
          numarray.Int16:tInt16,     numarray.UInt16:tUInt16,
          numarray.Int32:tInt32,     numarray.UInt32:tUInt32,
          numarray.Int64:tInt64,     numarray.UInt64:tUInt64,
          numarray.Float32:tFloat32, numarray.Float64:tFloat64,
          numarray.Complex32:tComplex32, numarray.Complex64:tComplex64,
          CharType:97   # ascii(97) --> 'a' # Special case (to be corrected)
          }

toclass = {tBool:numarray.Bool,  # Boolean type added
           tInt8:numarray.Int8,       tUInt8:numarray.UInt8,
           tInt16:numarray.Int16,     tUInt16:numarray.UInt16,
           tInt32:numarray.Int32,     tUInt32:numarray.UInt32,
           tInt64:numarray.Int64,     tUInt64:numarray.UInt64,
           tFloat32:numarray.Float32, tFloat64:numarray.Float64,
           tComplex32:numarray.Complex32, tComplex64:numarray.Complex64,
           97:CharType   # ascii(97) --> 'a' # Special case (to be corrected)
          }

# Define the CharType code as a constant
cdef enum:
  CHARTYPE = 97

# Functions from numarray API
cdef extern from "numarray/libnumarray.h":
  object NA_InputArray (object, NumarrayType, int)
  object NA_OutputArray (object, NumarrayType, int)
  object NA_IoArray (object, NumarrayType, int)
  object PyArray_FromDims(int nd, int *d, int type)
  object NA_Empty(int nd, int *d, NumarrayType type)
  object NA_updateDataPtr(object)
  #object NA_getPythonScalar(object, long)
  #object NA_setFromPythonScalar(object, int, object)
  object NA_getPythonScalar(_numarray, long)
  object NA_setFromPythonScalar(_numarray, int, _numarray)
  int getWriteBufferDataPtr(object, void**)
  int getReadBufferDataPtr(object, void**)
  long NA_getBufferPtrAndSize(object, int, void**)
  int isBufferWriteable(object)

  object PyArray_ContiguousFromObject(object op, int type,
                                      int min_dim, int max_dim)
# Functions from HDF5
cdef extern from "hdf5.h":
  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT
  int H5P_DEFAULT, H5S_ALL
  #int H5F_SCOPE_GLOBAL, H5F_SCOPE_LOCAL
  #int H5T_NATIVE_CHAR, H5T_NATIVE_INT, H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE
  int H5P_FILE_CREATE, H5P_FILE_ACCESS
  int H5FD_LOG_LOC_WRITE, H5FD_LOG_ALL

  ctypedef int hid_t  # In H5Ipublic.h
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t

  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    #raw data is very small                     */
    H5D_CONTIGUOUS      = 1,    #the default                                */
    H5D_CHUNKED         = 2,    #slow and fancy                             */
    H5D_NLAYOUTS        = 3     #this one must be last!                     */

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
    #int type     # pre HDF5 1.4
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

  cdef enum H5S_seloper_t:
    H5S_SELECT_NOOP      = -1,  # error                                     */
    H5S_SELECT_SET       = 0,   # Select "set" operation                    */
    H5S_SELECT_OR,        # Binary "or" operation for hyperslabs
    H5S_SELECT_AND,       # Binary "and" operation for hyperslabs
    H5S_SELECT_XOR,       # Binary "xor" operation for hyperslabs
    H5S_SELECT_NOTB,      # Binary "not" operation for hyperslabs
    H5S_SELECT_NOTA,      # Binary "not" operation for hyperslabs
    H5S_SELECT_APPEND,    # Append elements to end of point selection */
    H5S_SELECT_PREPEND,   # Prepend elements to beginning of point selection */
    H5S_SELECT_INVALID    # Invalid upper bound on selection operations *
    

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

                
# Functions from HDF5
cdef extern from "H5public.h":
  hid_t  H5Fcreate(char *filename, unsigned int flags,
                   hid_t create_plist, hid_t access_plist)
  
  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id)
                
  herr_t H5Fclose (hid_t file_id)

  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope ) 

  htri_t H5Fis_hdf5(char *name)
  
  hid_t  H5Dopen (hid_t file_id, char *name)
  
  herr_t H5Dclose (hid_t dset_id)
  
  herr_t H5Dread (hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf)

  hid_t  H5Gcreate(hid_t loc_id, char *name, size_t size_hint )
  
  herr_t H5Gget_objinfo(hid_t loc_id,
                        char *name,
                        hbool_t follow_link,
                        H5G_stat_t *statbuf )
  
  hid_t  H5Gopen(hid_t loc_id, char *name )

  herr_t H5Gclose(hid_t group_id)

  herr_t H5Glink (hid_t file_id, H5G_link_t link_type,
                  char *current_name, char *new_name)
  
  herr_t H5Gunlink (hid_t file_id, char *name)

  herr_t H5Gmove(hid_t loc_id, char *src, char *dst)

  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum )

  herr_t H5check_version(unsigned majnum, unsigned minnum,
          unsigned relnum )

  herr_t H5Adelete(hid_t loc_id, char *name )

  herr_t H5Tclose(hid_t type_id)

  #hid_t H5Pcreate(H5P_class_t type )  # Wrong in documentation!
  hid_t H5Pcreate(hid_t type )

  herr_t H5Pset_cache(hid_t plist_id, int mdc_nelmts, int rdcc_nelmts,
                      size_t rdcc_nbytes, double rdcc_w0 )
  
  herr_t H5Pset_sieve_buf_size( hid_t fapl_id, hsize_t size )

  herr_t H5Pset_fapl_log( hid_t fapl_id, char *logfile,
                          unsigned int flags, size_t buf_size ) 

  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,
                             hssize_t start[],
                             hsize_t _stride[],
                             hsize_t count[],
                             hsize_t _block[])
     
# Functions from HDF5 HL Lite
cdef extern from "H5LT.h":

  herr_t H5LTmake_dataset( hid_t loc_id, char *dset_name, int rank,
                           hsize_t *dims, hid_t type_id, void *data )
  
  herr_t H5LTread_dataset( hid_t loc_id, char *dset_name,
                           hid_t type_id, void *data )
                           
  herr_t H5LTget_dataset_ndims ( hid_t loc_id, char *dset_name, int *rank )
  
  herr_t H5LTget_dataset_info ( hid_t loc_id, char *dset_name,
                                hsize_t *dims, H5T_class_t *class_id,
                                size_t *type_size )

  herr_t H5LT_get_attribute_disk(hid_t loc_id, char *attr_name, void *attr_out)
          
  herr_t H5LTget_attribute_ndims( hid_t loc_id, 
                                  char *obj_name, 
                                  char *attr_name,
                                  int *rank )
  herr_t H5LTget_attribute_info( hid_t loc_id, char *obj_name, char *attr_name,
                                 hsize_t *dims, H5T_class_t *class_id,
                                 size_t *type_size )

  herr_t H5LTset_attribute_string( hid_t loc_id, char *obj_name,
                                   char *attr_name, char *attr_data )

  herr_t H5LTset_attribute_char( hid_t loc_id, 
                                 char *obj_name, 
                                 char *attr_name,
                                 char *data,
                                 size_t size )
  
  herr_t H5LTset_attribute_short( hid_t loc_id, 
                                  char *obj_name, 
                                  char *attr_name,
                                  short *data,
                                  size_t size )
  
  herr_t H5LTset_attribute_int( hid_t loc_id, 
                                char *obj_name, 
                                char *attr_name,
                                int *data,
                                size_t size )
  
  herr_t H5LTset_attribute_long   ( hid_t loc_id, 
                                    char *obj_name, 
                                    char *attr_name,
                                    long *data,
                                    size_t size )
  
  herr_t H5LTset_attribute_float( hid_t loc_id, 
                                  char *obj_name, 
                                  char *attr_name,
                                  float *data,
                                  size_t size )
  
  herr_t H5LTset_attribute_double( hid_t loc_id, 
                                   char *obj_name, 
                                   char *attr_name,
                                   double *data,
                                   size_t size )
  
  herr_t H5LTget_attribute( hid_t loc_id, 
                            char *obj_name, 
                            char *attr_name,
                            hid_t mem_type_id,
                            void *data )
  
  herr_t H5LTget_attribute_string( hid_t loc_id, char *obj_name,
                                   char *attr_name, char *attr_data )

  herr_t H5LTget_attribute_char( hid_t loc_id, 
                                 char *obj_name, 
                                 char *attr_name,
                                 char *data ) 

  herr_t H5LTget_attribute_short( hid_t loc_id, 
                                  char *obj_name, 
                                  char *attr_name,
                                  short *data ) 

  herr_t H5LTget_attribute_int( hid_t loc_id, 
                                char *obj_name, 
                                char *attr_name,
                                int *data ) 

  herr_t H5LTget_attribute_long( hid_t loc_id, 
                                 char *obj_name, 
                                 char *attr_name,
                                 long *data ) 

  herr_t H5LTget_attribute_float( hid_t loc_id, 
                                  char *obj_name, 
                                  char *attr_name,
                                  float *data ) 

  herr_t H5LTget_attribute_double( hid_t loc_id, 
                                   char *obj_name, 
                                   char *attr_name,
                                   double *data )
  
  herr_t H5LTfind_dataset(hid_t loc_id, char *name)

  herr_t H5LT_find_attribute(hid_t loc_id, char *attr_name )


# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h":  

  herr_t H5ARRAYmake( hid_t loc_id, char *dset_name, char *klass,
                      char *title, char *flavor, char *obversion,
                      int rank, hsize_t *dims, int extdim,
                      hid_t type_id, hsize_t *dims_chunk, void *fill_data,
                      int complevel, char  *complib, int shuffle,
                      int fletcher32, void *data)

  herr_t H5ARRAYappend_records( hid_t loc_id, char *dset_name,
                                int rank, hsize_t *dims_orig,
                                hsize_t *dims_new, int extdim, void *data )

  herr_t H5ARRAYread( hid_t loc_id, char *dset_name,
                      hsize_t start,  hsize_t nrows, hsize_t step,
                      int extdim, void *data )

  herr_t H5ARRAYreadSlice( hid_t loc_id, char *dset_name,
                           hsize_t *start, hsize_t *stop,
                           hsize_t *step, void *data )

  herr_t H5ARRAYreadIndex( hid_t loc_id, char *dset_name, int notequal,
                           hsize_t *start, hsize_t *stop, hsize_t *step,
                           void *data )

  herr_t H5ARRAYget_ndims( hid_t loc_id, char *dset_name, int *rank )

  hid_t H5ARRAYget_info( hid_t loc_id, char *dset_name,
                         hsize_t *dims, hid_t *super_type_id,
                         H5T_class_t *super_class_id, char *byteorder)

# Functions for optimized operations for ARRAY
cdef extern from "H5ARRAY-opt.h":

  herr_t H5ARRAYOopen_readSlice( hid_t *dataset_id,
                                 hid_t *space_id,
                                 hid_t *type_id,
                                 hid_t loc_id, 
                                 char *dset_name)

  herr_t H5ARRAYOread_readSlice( hid_t dataset_id,
                                 hid_t space_id,
                                 hid_t type_id,
                                 hsize_t irow,
                                 hsize_t start,
                                 hsize_t stop,
                                 void *data )

  herr_t H5ARRAYOclose_readSlice(hid_t dataset_id,
                                 hid_t space_id,
                                 hid_t type_id)

# Functions for VLEN Arrays
cdef extern from "H5VLARRAY.h":
  
  hid_t H5VLARRAYmake( hid_t loc_id, char *dset_name, char *title,
                       char *flavor, char *obversion, int rank, int scalar,
                       hsize_t *dims, hid_t type_id, hsize_t chunk_size,
                       void *fill_data, int complevel, char *complib,
                       int shuffle, int flecther32, void *data)
  
  herr_t H5VLARRAYappend_records( hid_t loc_id, char *dset_name,
                                  int nobjects, hsize_t nrecords,
                                  void *data )  

  herr_t H5VLARRAYread( hid_t loc_id, char *dset_name,
                        hsize_t start,  hsize_t nrows, hsize_t step,
                        hvl_t *rdata, hsize_t *rdatalen )

  herr_t H5VLARRAYget_ndims( hid_t loc_id, char *dset_name, int *rank )

  herr_t H5ARRAYget_chunksize( hid_t loc_id, char *dset_name,
                               int rank, hsize_t *dims_chunk)

  hid_t H5VLARRAYget_info( hid_t loc_id, char *dset_name,
                           hsize_t *nrecords, hsize_t *base_dims,
                           hid_t *base_type_id, char *base_byteorder)


# Funtion to compute the HDF5 type from a numarray enum type
cdef extern from "arraytypes.h":
    
  hid_t convArrayType(int fmt, size_t size, char *byteorder)
  size_t getArrayType(hid_t type_id, int *fmt) 

                   
# I define this constant, but I should not, because it should be defined in
# the HDF5 library, but having problems importing it
cdef enum:
  MAX_FIELDS = 255

# Maximum size for strings
cdef enum:
  MAX_CHARS = 256

cdef extern from "H5TB.h":

  hid_t H5TBmake_table( char *table_title, hid_t loc_id, 
                        char *dset_name, hsize_t nfields,
                        hsize_t nrecords, size_t rowsize,
                        char **field_names, size_t *field_offset,
                        hid_t *field_types, hsize_t chunk_size,
                        void *fill_data, int compress, char *complib,
                        int shuffle, int fletcher32, void *data )
                         
  herr_t H5TBappend_records ( hid_t loc_id, char *dset_name,
                              hsize_t nrecords, size_t type_size, 
                              size_t *field_offset, void *data )
                                        
  herr_t H5TBwrite_records ( hid_t loc_id, char *dset_name,
                             hsize_t start, hsize_t nrecords,
                             size_t type_size, size_t *field_offset,
                             void *data )
                                        
  herr_t H5TBOwrite_records ( hid_t loc_id,  char *dset_name,
                              hsize_t start, hsize_t nrecords,
                              hsize_t step, size_t type_size,
                              size_t *field_offset, void *data )
  
  herr_t H5TBwrite_fields_name( hid_t loc_id, char *dset_name,
                                char *field_names, hsize_t start,
                                hsize_t nrecords, size_t type_size,
                                size_t *field_offset, void *data )
  
  herr_t H5TBget_table_info ( hid_t loc_id, char *table_name,
                              hsize_t *nfields, hsize_t *nrecords )

  herr_t H5TBget_field_info ( hid_t loc_id, char *table_name,
                              char *field_names[], size_t *field_sizes,
                              size_t *field_offsets, size_t *type_size  )

  herr_t H5TBread_table ( hid_t loc_id, char *table_name,
                          size_t dst_size, size_t *dst_offset, 
                          size_t *dst_sizes, void *dst_buf )
                         
#  herr_t H5TBread_records ( hid_t loc_id, char *table_name,
#                            hsize_t start, hsize_t nrecords, size_t type_size,
#                            size_t *field_offset, void *data )

  herr_t H5TBread_fields_name ( hid_t loc_id, char *table_name,
                                char *field_names, hsize_t start,
                                hsize_t nrecords, hsize_t step,
                                size_t type_size,
                                size_t *field_offset, void *data )

  herr_t H5TBread_fields_index( hid_t loc_id, char *dset_name,
                                hsize_t nfields, int *field_index,
                                hsize_t start, hsize_t nrecords,
                                size_t type_size, size_t *field_offset,
                                void *data )

  herr_t H5TBdelete_record( hid_t loc_id, 
                            char *dset_name,
                            hsize_t start,
                            hsize_t nrecords,
                            hsize_t maxtuples)


cdef extern from "H5TB-opt.h":

  herr_t H5TBOopen_read( hid_t *dataset_id,
                         hid_t *space_id,
                         hid_t *mem_type_id,
                         hid_t loc_id,
                         char *dset_name,
                         hsize_t nfields,
                         char **field_names,
                         size_t type_size,
                         size_t *field_offset)

  herr_t H5TBOread_records( hid_t *dataset_id, hid_t *space_id,
                            hid_t *mem_type_id, hsize_t start,
                            hsize_t nrecords, void *data )

  herr_t H5TBOread_elements( hid_t *dataset_id,
                             hid_t *space_id,
                             hid_t *mem_type_id,
                             hsize_t nrecords,
                             void *coords,
                             void *data )

  herr_t H5TBOclose_read( hid_t *dataset_id,
                          hid_t *space_id,
                          hid_t *mem_type_id )

  herr_t H5TBOopen_append( hid_t *dataset_id,
                           hid_t *mem_type_id,
                           hid_t loc_id, 
                           char *dset_name,
                           hsize_t nfields,
                           size_t type_size,
                           size_t *field_offset )
  
  herr_t H5TBOappend_records( hid_t *dataset_id,
                              hid_t *mem_type_id,
                              hsize_t nrecords,
                              hsize_t nrecords_orig,
                              void *data )

  herr_t H5TBOclose_append(hid_t *dataset_id,
                           hid_t *mem_type_id,
                           hsize_t ntotal_records,
                           char *dset_name,
                           hid_t parent_id)

# Declarations from PyTables local functions

# Funtion to compute the offset of a struct format
cdef extern from "calcoffset.h":
  
  int calcoffset(char *fmt, int *nvar, hid_t *types,
                 size_t *field_sizes, size_t *field_offsets)

# Funtion to get info from fields in a table
cdef extern from "getfieldfmt.h":
  herr_t getfieldfmt ( hid_t loc_id, char *table_name,
                       char *field_names[], size_t *field_sizes,
                       size_t *field_offset, size_t *rowsize,
                       hsize_t *nrecords, hsize_t *nfields,
                       object shapes, object type_sizes,
                       object types, char *fmt )

# Helper routines
cdef extern from "utils.h":
  object _getTablesVersion()
  #object getZLIBVersionInfo()
  object getHDF5VersionInfo()
  object createNamesTuple(char *buffer[], int nelements)
  object get_filter_names( hid_t loc_id, char *dset_name)
  object Giterate(hid_t parent_id, hid_t loc_id, char *name)
  object Aiterate(hid_t loc_id)
  H5T_class_t getHDF5ClassID(hid_t loc_id, char *name, H5D_layout_t *layout)
  object H5UIget_info( hid_t loc_id, char *name, char *byteorder)

  # To access to the slice.indices function in 2.2
  int GetIndicesEx(object s, int length,
                   int *start, int *stop, int *step, int *slicelength)

  object get_attribute_string_sys( hid_t loc_id, char *obj_name,
                                   char *attr_name)


# LZO library
cdef int lzo_version
cdef extern from "H5Zlzo.h":
  int register_lzo()
  object getLZOVersionInfo()

# Initialize & register lzo
lzo_version = register_lzo()


# UCL library
cdef int ucl_version
cdef extern from "H5Zucl.h":
  int register_ucl()
  object getUCLVersionInfo()
  
# Initialize & register ucl
ucl_version = register_ucl()

# utility funtions (these can be directly invoked from Python)

# def PyNextAfter(double x, double y):
#   return nextafter(x, y)

# def PyNextAfterF(float x, float y):
#   return nextafterf(x, y)

def getIndices(object s, int length):
  cdef int start, stop, step, slicelength

  if GetIndicesEx(s, length, &start, &stop, &step, &slicelength) < 0:
    raise ValueError("Problems getting the indices on slice '%s'" % s)
  return (start, stop, step)

def whichLibVersion(char *name):
  "Tell if an optional library is available or not"
    
  if (strcmp(name, "hdf5") == 0):
    binver, strver = getHDF5VersionInfo()
    return (binver, strver, None)     # Should be always available
  elif (strcmp(name, "zlib") == 0):
    if zlib_imported:
      return (1, zlib.ZLIB_VERSION, None)
    else:
      return (0, 0, None)
    # We want to avoid dependencies of the zlib library
#     binver, strver = getZLIBVersionInfo()
#     return (binver, strver, None)   # Should be always available
  elif (strcmp(name, "lzo") == 0):
    if lzo_version:
      (lzo_version_string, lzo_version_date) = getLZOVersionInfo()
      return (lzo_version, lzo_version_string, lzo_version_date)
    else:
      return (0, 0, None)
  elif (strcmp(name, "ucl") == 0):
    if ucl_version:
      (ucl_version_string, ucl_version_date) = getUCLVersionInfo()
      return (ucl_version, ucl_version_string, ucl_version_date)
    else:
      return (0, 0, None)
  else:
    print "Asking version for:", name, "library, but the only supported library names are: 'hdf5', 'zlib', 'ucl', 'lzo' (note the lower case namimg)." 
    return (0, None, None)
    
def whichClass( hid_t loc_id, char *name):
  cdef H5T_class_t  class_id
  cdef H5D_layout_t layout

  class_id = getHDF5ClassID(loc_id, name, &layout)
  # Check if this a dataset of supported classtype for ARRAY
  if class_id == H5T_ARRAY:
    warnings.warn( \
 """Dataset object '%s' contains unsupported H5T_ARRAY datatypes.""" % (name),
 UserWarning)
  #if ((class_id == H5T_ARRAY)   or
  if  ((class_id == H5T_INTEGER)  or
       (class_id == H5T_FLOAT)    or
       (class_id == H5T_BITFIELD) or
       (class_id == H5T_STRING)):
    if layout == H5D_CHUNKED:
      return "EARRAY"
    else:
      return "ARRAY"
  elif class_id == H5T_VLEN:
    return "VLARRAY"
  elif class_id == H5T_COMPOUND:
    return "TABLE"

  # Fallback 
  return "UNSUPPORTED"

def isHDF5(char *filename):
  """Determines whether a file is in the HDF5 format.

  When successful, returns a positive value, for TRUE, or 0 (zero),
  for FALSE. Otherwise returns a negative value.

  To this function to work, it needs a closed file.

  """
  
  return H5Fis_hdf5(filename)
  ##return 1

def isPyTablesFile(char *filename):
  """Determines whether a file is in the PyTables format.

  When successful, returns the format version string, for TRUE, or 0
  (zero), for FALSE. Otherwise returns a negative value.

  To this function to work, it needs a closed file.

  """
  
  cdef hid_t file_id

  isptf = "unknown"
  if os.path.isfile(filename) and H5Fis_hdf5(filename) > 0:
    # The file exists and is HDF5, that's ok
    # Open it in read-only mode
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT)
    isptf = read_f_attr(file_id, 'PYTABLES_FORMAT_VERSION')
    # Close the file
    H5Fclose(file_id)

  if isptf == "unknown":
    return None    # Means that this is not a pytables file
  else:
    return isptf

def getHDF5Version():
  """Get the underlying HDF5 library version"""
  
  return getHDF5VersionInfo()[1]

def getExtVersion():
  """Return this extension CVS version"""
  
  # We need to do that here because
  # the cvsid really gives the CVS version of the generated C file (because
  # it is also in CVS!."""
  # But the $Id will be processed whenever a cvs commit is issued.
  # So, if you make a cvs commit *before* a .c generation *and*
  # you don't modify anymore the .pyx source file, you will get a cvsid
  # for the C file, not the Pyrex one!. The solution is not trivial!.
  return "$Id: hdf5Extension.pyx,v 1.146 2004/10/05 19:22:20 falted Exp $ "

def getPyTablesVersion():
  """Return this extension version."""
  
  #return PYTABLES_VERSION
  return _getTablesVersion()

def read_f_attr(hid_t file_id, char *attr_name):
  """Return the PyTables file attributes.

  When successful, returns the format version string, for TRUE, or 0
  (zero), for FALSE. Otherwise returns a negative value.

  To this function to work, it needs a closed file.

  """

  cdef hid_t root_id
  cdef herr_t ret
  cdef char attr_value[256]

  # Check if attribute exists
  # Open the root group
  root_id =  H5Gopen(file_id, "/")
  strcpy(attr_value, "unknown")  # Default value
  if H5LT_find_attribute(root_id, attr_name):
    # Read the format_version attribute
    ret = H5LT_get_attribute_disk(root_id, attr_name, attr_value)
    if ret < 0:
      strcpy(attr_value, "unknown")

  # Close root group
  H5Gclose(root_id)
      
  return attr_value

def flush_leaf(where, name):
  "Flush the buffers of a Leaf"
  cdef hid_t   parent_id, dataset_id
  cdef char    *dataset_name

  # Get the object ID. Do this until a standard objectID is implemented
  parent_id = where._v_objectID
  # Open the dataset
  dataset_name = strdup(name)
  dataset_id = H5Dopen( parent_id, dataset_name )
  # Flush the leaf
  H5Fflush(dataset_id, H5F_SCOPE_GLOBAL)
  H5Dclose( dataset_id )

# Utility function
def _getFilters(parent_id, name):
  "Get a dictionary with the filter names and cd_values"
  return get_filter_names(parent_id, name)

# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef hid_t   access_plist
  cdef char    *name

  def __new__(self, char *name, char *mode, char *title, int new,
              object trTable, char *root, int isPTFile, object filters):
    # Create a new file using default properties
    self.name = name
    self.mode = mode
    if (strcmp(mode, "r") == 0 or strcmp(mode, "r+") == 0):
      if (os.path.isfile(name) and H5Fis_hdf5(name) > 0):
        if strcmp(mode, "r") == 0:
          # Just a test for disabling the cache for metadata
          access_plist = H5Pcreate(H5P_FILE_ACCESS)
#           H5Pset_cache(access_plist, 0, 0, 0, 0.0)
#           H5Pset_sieve_buf_size(access_plist, 0 ) 
#           self.file_id = H5Fopen(name, H5F_ACC_RDONLY, access_plist)
          self.file_id = H5Fopen(name, H5F_ACC_RDONLY, H5P_DEFAULT)
        elif strcmp(mode, "r+") == 0:
          self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
      else:
        raise RuntimeError("File \'%s\' doesn't exist or is not a HDF5 file." \
                           % self.name )
    elif strcmp(mode, "a") == 0:
      # Fixes bug #988547
      exists_name = os.path.exists(name)
      if exists_name and os.path.isfile(name):
      #if os.path.isfile(name):
        if H5Fis_hdf5(name) > 0:
          # A test for logging
          access_plist = H5Pcreate(H5P_FILE_ACCESS)
#           H5Pset_cache(access_plist, 0, 0, 0, 0.0)
#           H5Pset_sieve_buf_size(access_plist, 0 ) 
##           H5Pset_fapl_log (access_plist, "test.log", H5FD_LOG_LOC_WRITE,
##                            0)
#           self.file_id = H5Fopen(name, H5F_ACC_RDWR, access_plist)
          self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
        else:
          raise RuntimeError("File \'%s\' exist but is not a HDF5 file." % \
                             self.name )
      elif exists_name:
        raise RuntimeError("\'%s\' is not an ordinary file." % self.name)
      else:
        self.file_id = H5Fcreate(name, H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT)
    elif strcmp(mode, "w") == 0:
      self.file_id = H5Fcreate(name, H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT)
    else:
      raise RuntimeError("Invalid mode \'%s\' for opening a file." % \
                         self.mode )

  # Accessor definitions
  def _getFileId(self):
    return self.file_id

  def _flushFile(self, scope):
    # Close the file
    H5Fflush(self.file_id, scope) 

  def _closeFile(self):
    # Close the file
    H5Fclose( self.file_id )
    self.file_id = 0    # Means file closed

  # This method is moved out of scope, until we provide code to delete
  # the memory booked by this extension types
  def __dealloc__(self):
    cdef int ret
    #print "Destroying object File in Extension"
    if self.file_id:
      #print "Closing the HDF5 file", name," because user didn't do that!."
      ret = H5Fclose(self.file_id)
      if ret < 0:
        raise RuntimeError("Problems closing the file %s" % self.name )


cdef class AttributeSet:
  cdef hid_t   parent_id
  cdef char    *name
  cdef object  node

  def _g_new(self, node):
    self.node = node
    # Initialize the C attributes of Node object
    self.name =  PyString_AsString(node._v_hdf5name)
    # The parent group id of the node
    self.parent_id = node._v_parent._v_objectID
    
  def _g_listAttr(self):
    cdef object attrlist
    cdef hid_t loc_id

    if isinstance(self.node, Group):
      # Return a tuple with the attribute list
      attrlist = Aiterate(self.node._v_objectID)
    else:
      # Get the dataset ID (the Leaf objects are always closed)
      loc_id = H5Dopen(self.parent_id, self.name)
      # Keep the object ID in the objectID attribute in the parent dataset
      # The existing datsets are always opened here, so this would
      # be enough to get the objectID for existing datasets
      self.node.objectID = loc_id 
      if loc_id < 0:
        raise RuntimeError("Cannot open the dataset '%s'" % self.name)
      attrlist = Aiterate(loc_id)
      # Close this dataset
      ret = H5Dclose(loc_id)
      if ret < 0:
        raise RuntimeError("Cannot close the dataset '%s'" % self.name)

    return attrlist

  def _g_setAttr(self, char *name, object value):
    cdef int ret
    cdef int valint
    cdef double valdouble

    ret = 0
    # Append this attribute on disk
    if isinstance(value, types.StringType):
      #self._g_setAttrStr(name, value)
      ret = H5LTset_attribute_string(self.parent_id, self.name, name, value)
    elif isinstance(value, types.IntType):
      #self._g_setAttrInt(name, value)
      valint = <int>PyInt_AsLong(value)
      ret = H5LTset_attribute_int(self.parent_id, self.name, name, &valint, 1)
    elif isinstance(value, types.FloatType):
      self._g_setAttrDouble(name, value)
      # I'm having problems with that in the C generated code:
      #   __pyx_v_valdouble = ((double )__pyx_v_value);
      # src/hdf5Extension.c:1295: error: pointer value used where a floating point value was expected
      # I don't know why!
      #valdouble = <double>value
      #ret = H5LTset_attribute_double(self.parent_id, self.name,
      #                               name, &valdouble, 1)
    else:
      # Convert this object to a null-terminated string
      # (binary pickles are not supported at this moment)
      pickledvalue = cPickle.dumps(value, 0)
      self._g_setAttrStr(name, pickledvalue)

    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (name, self.node))

  def _g_setAttrStr(self, char *attrname, char *attrvalue):
    cdef int ret

    ret = H5LTset_attribute_string(self.parent_id, self.name,
                                   attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_setAttrChar(self, char *attrname, char attrvalue):
    cdef int ret

    ret = H5LTset_attribute_char(self.parent_id, self.name,
                                 attrname, &attrvalue, 1)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_setAttrShort(self, char *attrname, short attrvalue):
    cdef int ret

    ret = H5LTset_attribute_short(self.parent_id, self.name,
                                  attrname, &attrvalue, 1)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_setAttrInt(self, char *attrname, int attrvalue):
    cdef int ret

    ret = H5LTset_attribute_int(self.parent_id, self.name,
                                attrname, &attrvalue, 1)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_setAttrFloat(self, char *attrname, float attrvalue):
    cdef int ret

    ret = H5LTset_attribute_float(self.parent_id, self.name,
                                  attrname, &attrvalue, 1)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_setAttrDouble(self, char *attrname, double attrvalue):
    cdef int ret

    ret = H5LTset_attribute_double(self.parent_id, self.name,
                                   attrname, &attrvalue, 1)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_getAttr(self, char *attrname):
    cdef object attrvalue
    cdef hid_t loc_id
    if isinstance(self.node, Group):
      attrvalue = self._g_getNodeAttr(self.parent_id, self.node._v_objectID,
                                      self.name, attrname)
    else:
      # Get the dataset ID
      loc_id = H5Dopen(self.parent_id, self.name)
      if loc_id < 0:
        raise RuntimeError("Cannot open the dataset '%s' in node '%s'" % \
                           (self.name, self._v_parent._v_name))

      attrvalue = self._g_getNodeAttr(self.parent_id, loc_id,
                                      self.name, attrname)

      # Close this dataset
      ret = H5Dclose(loc_id)
      if ret < 0:
        raise RuntimeError("Cannot close the dataset '%s'" % self.name)

    return attrvalue

  # Get a system attribute (they should be only strings)
  def _g_getSysAttr(self, char *attrname):
    ret = get_attribute_string_sys(self.parent_id, self.name, attrname)
    return ret

  # This funtion is useful to retrieve system attributes of Leafs
  def _g_getChildSysAttr(self, char *dsetname, char *attrname):
    ret = get_attribute_string_sys(self.node._v_objectID, dsetname, attrname)
    return ret

  def _g_getChildAttr(self, char *dsetname, char *attrname):
    cdef object attrvalue
    cdef hid_t loc_id
    # Get the dataset ID
    loc_id = H5Dopen(self.node._v_objectID, dsetname)
    if loc_id < 0:
      raise RuntimeError("Cannot open the child '%s' of node '%s'" % \
                         (dsetname, self.name))

    attrvalue = self._g_getNodeAttr(self.node._v_objectID, loc_id,
                                    dsetname, attrname)
    # Close this dataset
    ret = H5Dclose(loc_id)
    if ret < 0:
      raise RuntimeError("Cannot close the dataset '%s'" % dsetname)

    return attrvalue

  # Get attributes
  def _g_getNodeAttr(self, hid_t parent_id, hid_t loc_id,
                        char *dsetname, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef char  *attrvaluestr
    cdef char  attrvaluechar
    cdef short  attrvalueshort
    cdef long  attrvaluelong
    cdef float  attrvaluefloat
    cdef double  attrvaluedouble
    cdef long long attrvaluelonglong
    cdef object retvalue
    cdef hid_t mem_type
    cdef int rank
    cdef int ret, i

    # Check if attribute exists
    if H5LT_find_attribute(loc_id, attrname) <= 0:
      # If the attribute does not exists, return None
      # and do not even warn the user
      return None

    ret = H5LTget_attribute_ndims(parent_id, dsetname, attrname, &rank )
    if ret < 0:
      raise RuntimeError("Can't get ndims on attribute %s in node %s." %
                             (attrname, dsetname))

    if rank > 1:
      print \
"""Info: Can't deal with multidimensional attribute '%s' in node '%s'. Sorry about that!""" % (attrname, dsetname)
      return None
    
    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))

    ret = H5LTget_attribute_info(parent_id, dsetname, attrname,
                                 dims, &class_id, &type_size)
    if ret < 0:
        raise RuntimeError("Can't get info on attribute %s in node %s." %
                               (attrname, dsetname))

    if rank > 0 and dims[0] > 1:
      print \
"""Info: Can't deal with multidimensional attribute '%s' in node '%s'.""" % \
(attrname, dsetname)
      free(<void *> dims)
      return None
    elif rank > 0:
      free(<void *> dims)
      
    if class_id == H5T_INTEGER:
      if type_size == 1:
        ret = H5LTget_attribute_char(parent_id, dsetname,
                                     attrname, &attrvaluechar)
        retvalue = PyInt_FromLong(<long>attrvaluechar)
      if type_size == 2:
        ret = H5LTget_attribute_short(parent_id, dsetname,
                                      attrname, &attrvalueshort)
        retvalue = PyInt_FromLong(<long>attrvalueshort)
      if type_size == 4:
        ret = H5LTget_attribute_long(parent_id, dsetname,
                                     attrname, &attrvaluelong)
        retvalue = PyInt_FromLong(attrvaluelong)
      if type_size == 8:
        ret = H5LTget_attribute(parent_id,dsetname,
                                attrname, H5T_NATIVE_LLONG, &attrvaluelonglong)
        retvalue = PyLong_FromLongLong(attrvaluelonglong)
    elif class_id == H5T_FLOAT:
      if type_size == 4:
        ret = H5LTget_attribute_float(parent_id, dsetname,
                                      attrname, &attrvaluefloat)
        retvalue = PyFloat_FromDouble(<double>attrvaluefloat)
      if type_size == 8:
        ret = H5LTget_attribute_double(parent_id, dsetname,
                                       attrname, &attrvaluedouble)
        retvalue = PyFloat_FromDouble(attrvaluedouble)
    elif class_id == H5T_STRING:
      attrvaluestr = <char *>malloc(type_size * sizeof(char))
      ret = H5LTget_attribute_string(parent_id, dsetname,
                                     attrname, attrvaluestr)
      retvalue = PyString_FromString(attrvaluestr)
      free(<void *> attrvaluestr)   # To avoid memory leak!
    else:
      print \
"""Info: Type of attribute '%s' in node '%s' is not supported. Sorry about that!""" % (attrname, dsetname)
      return None

    # Check the return value of H5LTget_attribute_* call
    if ret < 0:
      raise RuntimeError("Attribute %s exists in node %s, but can't get it."\
                         % (attrname, dsetname))

    return retvalue

  def _g_remove(self, attrname):
    cdef int ret
    cdef hid_t loc_id
    
    if isinstance(self.node, Group):
      ret = H5Adelete(self.node._v_objectID, attrname ) 
      if ret < 0:
        raise RuntimeError("Attribute '%s' exists in node '%s', but cannot be deleted." \
                         % (attrname, dsetname))
    else:
      # Open the dataset 
      loc_id = H5Dopen(self.parent_id, self.name)
      if loc_id < 0:
        raise RuntimeError("Cannot open the dataset '%s' in node '%s'" % \
                           (dsetname, self.name))

    
      ret = H5Adelete(loc_id, attrname ) 
      if ret < 0:
        raise RuntimeError("Attribute '%s' exists in node '%s', but cannot be deleted." \
                         % (attrname, self.name))

      # Close this dataset
      ret = H5Dclose(loc_id)
      if ret < 0:
        raise RuntimeError("Cannot close the dataset '%s'" % self.name)

  def __dealloc__(self):
    cdef int ret
    #print "Destroying object AttributeSet in Extension"
    self.node = None
    self.parent_id = 0


cdef class Group:
  cdef hid_t   group_id
  cdef hid_t   parent_id
  cdef char    *name

  def _g_new(self, where, name):
    # Initialize the C attributes of Group object
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_objectID
    
  def _g_createGroup(self):
    cdef hid_t ret
    
    # Create a new group
    ret = H5Gcreate(self.parent_id, self.name, 0)
    if ret < 0:
      raise RuntimeError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_openGroup(self):
    cdef hid_t ret
    
    ret = H5Gopen(self.parent_id, self.name)
    if ret < 0:
      raise RuntimeError("Can't open the group: '%s'." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_openIndex(self):
    cdef hid_t ret
    
    ret = H5Gopen(self.parent_id, self.name)
    if ret < 0:
      raise RuntimeError("Can't open the index: '%s'." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_listGroup(self, hid_t parent_id, hid_t loc_id, char *name):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(parent_id, loc_id, name)

  def _g_getGChildAttr(self, char *group_name, char *attr_name):
    """Return an attribute of a child Group.

    When successful, returns the format version string, for TRUE, or 0
    (zero), for FALSE. Otherwise returns a negative value.

    """

    cdef hid_t gchild_id
    cdef herr_t ret
    cdef char attr_value[256]

    # Check if attribute exists
    # Open the group
    gchild_id = H5Gopen(self.group_id, group_name)
    strcpy(attr_value, "unknown")  # Default value
    if H5LT_find_attribute(gchild_id, attr_name):
      # Read the attr_name attribute
      ret = H5LT_get_attribute_disk(gchild_id, attr_name, attr_value)
      if ret < 0:
        strcpy(attr_value, "unknown")

    # Close child group
    H5Gclose(gchild_id)
    return attr_value

  def _g_getAttr(self, char *attr_name):
    """Return an attribute of a child Group.

    When successful, returns the format version string, for TRUE, or 0
    (zero), for FALSE. Otherwise returns a negative value.

    """
    cdef herr_t ret
    cdef char attr_value[256]

    # Check if attribute exists
    strcpy(attr_value, "unknown")  # Default value
    if H5LT_find_attribute(self.group_id, attr_name):
      # Read the attr_name attribute
      ret = H5LT_get_attribute_disk(self.group_id, attr_name, attr_value)
      if ret < 0:
        strcpy(attr_value, "unknown")

    return attr_value

  def _g_flushGroup(self):
    # Close the group
    H5Fflush(self.group_id, H5F_SCOPE_GLOBAL) 

  def _g_closeGroup(self):
    cdef int ret
    
    ret = H5Gclose(self.group_id)
    if ret < 0:
      raise RuntimeError("Problems closing the Group %s" % self.name )
    self.group_id = 0  # indicate that this group is closed

  def _g_renameNode(self, char *oldname, char *newname):
    cdef int ret

    ret = H5Gmove(self.group_id, oldname, newname)
    if ret < 0:
      raise RuntimeError("Problems renaming the node %s" % oldname )
    return ret

  def _g_deleteGroup(self):
    cdef int ret

    # Delete this group
    ret = H5Gunlink(self.parent_id, self.name)
    if ret < 0:
      raise RuntimeError("Problems deleting the Group %s" % self.name )
    return ret

  def _g_deleteLeaf(self, char *dsetname):
    cdef int ret

    # Delete the leaf child
#     print "Deleting leaf in -->", self.name, dsetname
#     list1 = self._g_listGroup(self.parent_id, self.group_id, self.name)
#     print "Childs-->", list1
    ret = H5Gunlink(self.group_id, dsetname)
    if ret < 0:
      raise RuntimeError("Problems deleting the Leaf '%s'" % dsetname )
    return ret

#   def _g_deleteLeaf2(self, char *dsetname, object hdf5object):
#     cdef int ret

#     # Delete the leaf child
#     hdf5object.close()
#     ret = H5Gunlink(self.group_id, dsetname)
#     if ret < 0:
#       raise RuntimeError("Problems deleting the Leaf '%s'" % dsetname )
#     return ret

  def __dealloc__(self):
    cdef int ret
    # print "Destroying object Group in Extension"
    if self.group_id <> 0 and 0:
      print "Group open: ", self.name
    free(<void *>self.name)


cdef class Table:
  # instance variables
  cdef size_t  field_offset[MAX_FIELDS]
  cdef size_t  field_sizes[MAX_FIELDS]
  cdef hsize_t nfields
  cdef void    *rbuf, *mmrbuf
  cdef hsize_t totalrecords
  cdef hid_t   parent_id, loc_id
  cdef char    *name, *xtitle
  cdef char    *fmt
  cdef char    *field_names[MAX_FIELDS]
  cdef int     _open
  cdef char    *complib
  cdef hid_t   dataset_id, space_id, mem_type_id
  cdef object  mmfilew, mmfiler

  def _g_new(self, where, name):
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_objectID
    self._open = 0

  def _createTable(self, char *title, char *complib):
    cdef int     nvar, offset
    cdef int     i, ret
    cdef long    buflen
    cdef hid_t   oid
    cdef hid_t   field_types[MAX_FIELDS]
    cdef void    *fill_data, *data
    cdef hsize_t nrecords

    # This check has to be done before assigning too much columns in 
    # self.field_names C array
    if (len(self.colnames) > MAX_FIELDS):
        raise IndexError("A maximum of %d fields on tables is allowed" % \
                         MAX_FIELDS)
                         
    # Get a pointer to the table format
    self.fmt = PyString_AsString(self._v_fmt)
      
    # Assign the field_names pointers to the Tuple colnames strings
    i = 0
    for name in self.colnames:
      if (len(name) >= MAX_CHARS):
        raise NameError("A maximum length of %d on column names is allowed" % \
                         (MAX_CHARS - 1))
      # Copy the column names to an internal buffer
      self.field_names[i] = strdup(name)
      i = i + 1

    # Compute the field type sizes, offsets, # fields, ...
    self.rowsize = calcoffset(self.fmt, &nvar, field_types,
                              self.field_sizes, self.field_offset)
    self.nfields = nvar

    # Protection against row sizes too large (HDF5 refuse to work
    # with row sizes larger than 10 KB or so).
    # This limitation was consequence of my buffer size computation that was
    # quite bad. Now, I think it is safe to release this limitation for most
    # uses
    # I'll revert to a 512 KB limit (just because banana 640 KB limitation)
    if self.rowsize > 524288:
            raise RuntimeError, \
    """Row size too large. Maximum size is 8192 bytes, and you are asking
    for a row size of %s bytes.""" % (self.rowsize)

    # test if there is data to be saved initially
    if hasattr(self, "_v_recarray"):
      self.totalrecords = self.nrows
      buflen = NA_getBufferPtrAndSize(self._v_recarray._data, 1, &data)
      # Correct the offset in the buffer
      offset = self._v_recarray._byteoffset
      data = <void *>(<char *>data + offset)
    else:
      self.totalrecords = 0
      data = NULL

    # Compute some values for buffering and I/O parameters
    (self._v_maxTuples, self._v_chunksize) = \
                        calcBufferSize(self.rowsize, self._v_expectedrows,
                                       self.filters.complevel)
    # The next is settable if we have default values
    fill_data = NULL
    nrecords = <hsize_t>PyInt_AsLong(nvar)
    oid = H5TBmake_table(title, self.parent_id, self.name,
                         nrecords, self.nrows, self.rowsize,
                         self.field_names,
                         self.field_offset, field_types, self._v_chunksize,
                         fill_data, self.filters.complevel, complib,
                         self.filters.shuffle, self.filters.fletcher32,
                         data)
    if oid < 0:
      raise RuntimeError("Problems creating the table")
    self.objectID = oid  

    # Release resources to avoid memory leaks
    for i from  0 <= i < nvar:
      H5Tclose(field_types[i])
    
  def _open_append(self, object recarr):
    cdef long buflen

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &self.rbuf)

    # Open the table for appending
    if ( H5TBOopen_append(&self.dataset_id, &self.mem_type_id,
                          self.parent_id, self.name, self.nfields,
                          self.rowsize, self.field_offset) < 0 ):
      raise RuntimeError("Problems opening table for append.")

    self._open = 1
    self.objectID = self.dataset_id

  # A version of Table._saveBufferRows in Pyrex is available in 0.7.2,
  # but it is not faster than the Python version, so I removed it
  
  def _append_records(self, object recarr, int nrecords):
    cdef int ret

    if not self._open:
      self._open_append(recarr)

    # Append the records:
    ret = H5TBOappend_records(&self.dataset_id, &self.mem_type_id,
                              nrecords, self.totalrecords, self.rbuf)
    if ret < 0:
      raise RuntimeError("Problems appending the records.")

    self.totalrecords = self.totalrecords + nrecords
    
  def _close_append(self):

    if self._open > 0:
      # Close the table for append
      if ( H5TBOclose_append(&self.dataset_id, &self.mem_type_id,
                             self.totalrecords, self.name,
                             self.parent_id) < 0 ):
        raise RuntimeError("Problems closing table for append.")

    self._open = 0

  def _modify_records(self, hsize_t start, hsize_t stop,
                       hsize_t step, object recarr):
    cdef int ret
    cdef void *rbuf
    cdef hsize_t nrecords, nrows

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &rbuf)

    # Compute the number of records to modify
    nrecords = len(recarr)
    nrows = ((stop - start - 1) / step) + 1 
    if nrecords > nrows:
      nrecords = nrows
    # Modify the records:
    ret = H5TBOwrite_records(self.parent_id, self.name,
                             start, nrecords, step, self.rowsize,
                             self.field_offset, rbuf )
    if ret < 0:
      raise RuntimeError("Problems modifying the records.")

# The next doesn't work well. We use the method _modify_records for all
# cases. 2004-08-09
#   def _modify_records_names(self, hsize_t start, object recarr, object names):
#     cdef int ret
#     cdef void *rbuf
#     cdef hsize_t nrecords
#     cdef int nfields
#     cdef char *field_names
#     cdef size_t type_size
#     cdef size_t *field_offset

#     # Get the pointer to the buffer data area
#     buflen = NA_getBufferPtrAndSize(recarr._data, 1, &rbuf)

#     # Get a pointer to the field names
#     field_names = PyString_AsString("\n".join(names))
#     # Modify the records:
#     nrecords = len(recarr)
#     type_size = recarr._itemsize
#     nfields = recarr._nfields
#     sizes = recarr._sizes
#     field_offset = <size_t *>malloc(nrecords * sizeof(size_t))
#     field_offset[0] = 0
#     print "field_offset-->", field_offset[0],
#     for i in range(nfields-1):
#       field_offset[i+1] = field_offset[i]+sizes[i]
#       print "field_offset-->", field_offset[i+1],
#     print
#     ret = H5TBwrite_fields_name(self.parent_id, self.name, field_names,
#                                 start, nrecords, type_size,
#                                 self.field_offset, rbuf )
#     if ret < 0:
#       raise RuntimeError("Problems modifying the records.")
#     free(field_offset)
    
  def _getTableInfo(self):
    "Get info from a table on disk. This method is standalone."
    cdef int     i, ret
    cdef hsize_t nrecords, nfields
    cdef hsize_t dims[1] # Tables are one-dimensional
    cdef H5T_class_t class_id
    cdef object  names, type_sizes, types
    cdef size_t  rowsize
    cdef char    fmt[2048]
    cdef long    buflen
    
    shapes = []
    type_sizes = []
    types = []
    ret = getfieldfmt(self.parent_id, self.name, self.field_names,
                      self.field_sizes, self.field_offset,
                      &rowsize, &nrecords, &nfields,
                      shapes, type_sizes, types, fmt)
    if ret < 0:
      raise RuntimeError("Problems getting field format")

    # Assign the values to class variables
    self.fmt = fmt
    self.nfields = nfields
    self.totalrecords = nrecords
    self.rowsize = rowsize
    
    # Create a python tuple with the field names
    names = []
    for i in xrange(nfields):
      names.append(self.field_names[i])
    names = tuple(names)

    # Return the buffer as a Python String
    return (nrecords, names, rowsize, type_sizes, shapes, types, fmt)

  def _open_read(self, object recarr):
    cdef long buflen
    cdef object recarr2

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &self.rbuf)

    # Readout to the buffer
    if ( H5TBOopen_read(&self.dataset_id, &self.space_id, &self.mem_type_id,
                        self.parent_id, self.name, self.nfields,
                        self.field_names, self.rowsize,
                        self.field_offset) < 0 ):
      raise RuntimeError("Problems opening table for read.")
    # Test
#     recarr2 = records.fromfile("prova.out",
#                               formats=self.description._v_recarrfmt)
#     print repr(recarr2._data)
#     if ( PyObject_AsWriteBuffer(recarr2._data, &self.mmrbuf, &buflen) < 0 ):
#       raise RuntimeError("Problems getting the pointer to the mm buffer")
#     print "despres de writebuffer"

    # Aquesta es la part vàlida (comentem...)
#     self.mmfiler = memmap.Memmap(self.name+".mmap",mode="r")
#     recarr2 = records.RecArray(self.mmfiler[:],
#                                formats=self.description._v_recarrfmt,
#                                shape=self.totalrecords)
#     print repr(recarr2._data._buffer)
#     if ( PyObject_AsReadBuffer(recarr2._data._buffer,
#                                &self.mmrbuf, &buflen) < 0 ):
#       raise RuntimeError("Problems getting the pointer to the mm buffer")


  def _read_field_name(self, object arr, hsize_t start, hsize_t stop,
                       hsize_t step, char *field_name):
    cdef int i, fieldpos
    cdef void *rbuf
    cdef hsize_t nrecords
    cdef long buflen

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(arr._data, 1, &rbuf)

    # Correct the number of records to read, if needed
    if stop > self.totalrecords:
      nrecords = self.totalrecords - start
    else:
      nrecords = stop - start
      
    # Search the field position
    fieldpos = -1
    for i in xrange(self.nfields):
      if strcmp(self.field_names[i], field_name) == 0:
        fieldpos = i
    if fieldpos == -1:
      raise RuntimeError("Problems searching the field_name: %s" % field_name)
    # Read the column
    if ( H5TBread_fields_name(self.parent_id, self.name, field_name,
                              start, nrecords, step,
                              self.field_sizes[fieldpos], NULL, rbuf) < 0):
      raise RuntimeError("Problems reading table column.")
    
    return nrecords

  def _read_field_index(self, object arr, hsize_t start, hsize_t stop,
                        int index):
    cdef int index_list[1]
    cdef void *rbuf
    cdef hsize_t nrecords
    cdef int buflen

    # Correct the number of records to read, if needed
    if stop > self.totalrecords:
      nrecords = self.totalrecords - start
    else:
      nrecords = stop - start

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(arr._data, 1, &rbuf)
    #if (PyObject_AsReadBuffer(arr._data, &rbuf, &buflen) < 0):
    #  print "Error getting buffer location"

    index_list[0] = index
    # Read the column
    if ( H5TBread_fields_index(self.parent_id, self.name, 1, index_list,
                              start, nrecords, self.field_sizes[index],
                              NULL, rbuf) < 0):
      raise RuntimeError("Problems reading table column.")
    
    return nrecords

  def _read_records(self, hsize_t start, hsize_t nrecords):

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    if ( H5TBOread_records(&self.dataset_id, &self.space_id,
                           &self.mem_type_id, start,
                           nrecords, self.rbuf) < 0 ):
      raise RuntimeError("Problems reading records.")

    return nrecords

  def _read_elements(self, size_t shift, object elements):
    cdef long buflen
    cdef hsize_t nrecords
    cdef void *coords

    # Get the chunk of the coords that correspond to a buffer
    nrecords = len(elements)
    coords_array = numarray.array(elements+shift, type=numarray.Int64)
    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(coords_array._data, 1, &coords)
    
    if ( H5TBOread_elements(&self.dataset_id, &self.space_id,
                            &self.mem_type_id, nrecords,
                            coords, self.rbuf) < 0 ):
      raise RuntimeError("Problems reading records.")

    return nrecords

  def _close_read(self):

    #self.mmfiler.close()  # Test  # Comentem...
    if ( H5TBOclose_read(&self.dataset_id, &self.space_id,
                         &self.mem_type_id) < 0 ):
      raise RuntimeError("Problems closing table for read.")

  def _remove_row(self, nrow, nrecords):

    if (H5TBdelete_record(self.parent_id, self.name, nrow, nrecords,
                          self._v_maxTuples) < 0):
      #raise RuntimeError("Problems deleting records.")
      print "Problems deleting records."
      # Return no removed records
      return 0
    else:
      self.totalrecords = self.totalrecords - nrecords
      # Return the number of records removed
      return nrecords

  def __dealloc__(self):
    #print "Destroying object Table in Extension"
    free(<void *>self.name)
    for i from  0 <= i < self.nfields:
      free(<void *>self.field_names[i])


cdef class Row:
  """Row Class

  This class hosts accessors to a recarray row. The fields on a
  recarray can be accessed both as items (__getitem__/__setitem__),
  i.e. following the "map" protocol.
    
  """

  cdef object _table   # To allow compilation under MIPSPro C in SGI machines
  #cdef Table _table   # To allow access C methods in Table
  cdef object _fields, _recarray, _saveBufferedRows, _indexes
  cdef long long _row, _nrowinbuf, _unsavednrows
  cdef int _strides
  #cdef readonly int _nrow # This is allowed from Pyrex 0.9 on
  # But defining it as long long makes it unaccessible from python!
  cdef long long _nrow
  cdef long long start, stop, step, nextelement
  cdef long long nrowsinbuf, nrows, nrowsread, stopindex
  cdef int bufcounter, counter, startb, stopb,  _all
  cdef int *_scalar, *_enumtypes, _r_initialized_buffer,_w_initialized_buffer
  cdef int indexChunk
  cdef object indexValid, coords, bufcoords, index
  cdef int whereCond, indexed
  cdef double startcond, stopcond
  cdef int op1, op2
  cdef char *colname

  def __new__(self, table):
  #def __new__(self, Table table):
    cdef int nfields, i
    
    # The MIPSPro C compiler on a SGI does not like to have an assignation
    # of a type Table to a type object. For now, as we do not have to call
    # C methods in Tables, I'll declare table as object.
    # F. Alted 2004-02-11
    self._table = table
    self._unsavednrows = 0
    self._row = 0
    self._nrow = 0
    self._r_initialized_buffer = 0
    self._w_initialized_buffer = 0
    self._saveBufferedRows = self._table._saveBufferedRows

  def __call__(self, start=0, stop=0, step=1, coords=None, ncoords=0):
    """ return the row for this record object and update counters"""

    self._initLoop(start, stop, step, coords, ncoords)
    return iter(self)

  def __iter__(self):
    "Iterator that traverses all the data in the Table"

    return self

  def _newBuffer(self, write):
    "Create the recarray for I/O buffering"
    
    if write:
      buff = self._table._v_buffer = self._table._newBuffer(init=1)
      self._table._v_buffercpy = self._table._newBuffer(init=1)
      # Flag that tells that the buffer has been initialized for writing
      self._w_initialized_buffer = 1
      self._r_initialized_buffer = 1   # and also for reading...
    else:
      buff = self._table._v_buffer = self._table._newBuffer(init=0)
      # Flag that tells that the buffer has been initialized for reading
      self._r_initialized_buffer = 1
      self._table._v_buffercpy = None  # Decrement the reference to the buffer
                                       # copy (if it exists!)
      self._w_initialized_buffer = 0   # buffer is not more valid for writing

    self._recarray = buff
    self.nrows = self._table.nrows  # This value may change
    self.nrowsinbuf = self._table._v_maxTuples    # Need to fetch this value
    self._fields = buff._fields
    self._strides = buff._strides[0]
    nfields = buff._nfields
    # Create a dictionary with the index columns of the recarray
    # and other tables
    i = 0
    self._indexes = {}
    self._scalar = <int *>malloc(nfields * sizeof(int))
    self._enumtypes = <int *>malloc(nfields * sizeof(int))
    for field in buff._names:
      self._indexes[field] = i
      # If _repeats[i] = (1,) this is a numarray object, and not a scalar
      #if buff._repeats[i] == 1 or buff._repeats[i] == (1,):
      if buff._repeats[i] == 1:
        self._scalar[i] = 1
      else:
        self._scalar[i] = 0
      self._enumtypes[i] = toenum[buff._fmt[i]]
      i = i + 1

  def _initLoop(self, int start, int stop, int step,
                object coords, int ncoords):
    "Initialization for the __iter__ iterator"

    if not self._r_initialized_buffer:
      self._newBuffer(write=0)
    self.start = start
    self.stop = stop
    self.step = step
    self.coords = coords
    self.startb = 0
    self.nrowsread = start
    self._nrow = start - self.step
    self._table._open_read(self._recarray)  # Open the table for reading
    self.whereCond = 0
    self.indexed = 0
    # Do we have in-kernel selections?
    if (hasattr(self._table, "whereColname") and
        self._table.whereColname is not None):
      self.whereCond = 1
      self.colname = PyString_AsString(self._table.whereColname)
      # Is this column indexed and ready to use?
      if self._table.colindexed[self.colname] and ncoords >= 0:
        self.indexed = 1
        self.index = self._table.cols[self.colname].index
        # create buffers for indices
        self.index.indices._initIndexSlice(self.nrowsinbuf)
        self.nrowsread = 0
        self.nextelement = 0
    if self.coords is not None:
      self.stopindex = len(coords)
      self.nrowsread = 0
      self.nextelement = 0
    elif self.indexed:
      self.stopindex = ncoords

  def __next__(self):
    "next() method for __iter__() that is called on each iteration"
    if self.indexed or self.coords is not None:
      #print "indexed"
      return self.__next__indexed()
    elif self.whereCond:
      #print "inKernel"
      return self.__next__inKernel()
    else:
      #print "general"
      return self.__next__general()

  cdef __next__indexed(self):
    """The version of next() for indexed columns or with user coordinates"""
    cdef long offset
    cdef object indexValid1, indexValid2
    cdef int ncond, op, recout
    cdef long long stop
    cdef object opValue, field
    cdef long long nextelement

    while self.nextelement < self.stopindex:
      if self.nextelement >= self.nrowsread:
        # Correction for avoiding reading past self.stopindex
        if self.nrowsread+self.nrowsinbuf > self.stopindex:
          stop = self.stopindex-self.nrowsread
        else:
          stop = self.nrowsinbuf
        if self.coords is not None:
          self.bufcoords = self.coords[self.nrowsread:self.nrowsread+stop]
          nrowsread = len(self.bufcoords)
        else:
          self.bufcoords = self.index.getCoords(self.nrowsread, stop)
          nrowsread = len(self.bufcoords)
          tmp = self.bufcoords
          # If a step was specified, select first the strided elements
          if len(tmp) > 0 and self.step > 1:
            tmp2=(tmp-self.start) % self.step
            tmp = tmp[tmp2.__eq__(0)]
          # Now, select those indices in the range start, stop:
          if len(tmp) > 0 and tmp[0] < self.start:
            # Pyrex can't use the tmp>=number notation when tmp is a numarray
            # object. Why?
            #tmp = tmp[tmp>=self.start]
            tmp = tmp[tmp.__ge__(self.start)]
          if len(tmp) > 0 and tmp[-1] >= self.stop:
            tmp = tmp[numarray.where(tmp.__lt__(self.stop))]
          self.bufcoords = tmp
        self._row = -1
        if len(self.bufcoords):
          recout = self._table._read_elements(0, self.bufcoords)
          if self._table.byteorder <> sys.byteorder:
            self._recarray._byteswap()
        else:
          recout = 0
        self.nrowsread = self.nrowsread + nrowsread
        # Correction for elements that are eliminated by its
        # [start:stop:step] range
        self.nextelement = self.nextelement + nrowsread - recout
        if recout == 0:
          # no items where read, skipping
          continue
      self._row = self._row + 1
      self._nrow = self.bufcoords[self._row]
      self.nextelement = self.nextelement + 1
      # Return this row
      return self
    else:
      # Re-initialize the possible cuts in columns
      self.indexed = 0
      if self.coords is None:
        self.index.indices._destroyIndexSlice()  # Remove buffers in indices
        nextelement = self.index.nelemslice * self.index.nrows
        # Correct this for step size > 1
        correct = (nextelement - self.start) % self.step
        if self.step > 1 and correct:
          nextelement = nextelement + self.step - correct
      else:
        self.coords = None
        # All the elements has been read for this mode
        nextelement = self.nrows
      if nextelement >= self.nrows:
        self._table._close_read()  # Close the table
        self._table.ops = []
        self._table.opsValues = []
        self._table.whereColname = None
        self.index = 0
        self.whereCond = 0
        raise StopIteration        # end of iteration
      else:
        # Continue the iteration with the __next__inKernel() method
        self.start = nextelement
        self.startb = 0
        self.nrowsread = self.start
        self._nrow = self.start - self.step
        return self.__next__inKernel()

# This version of __next__indexed is fully operational, but it does not work
# with start, stop, step ranges. 2004-08-11
#   cdef __next__indexed_original(self):
#     """The version of next() for indexed columns or with user coordinates"""
#     cdef long offset
#     cdef object indexValid1, indexValid2
#     cdef int ncond, op, recout
#     cdef long long stop
#     cdef object opValue, field
#     cdef long long nextelement

#     while self.nextelement < self.stop:
#       if self.nextelement >= self.nrowsread:
#         # Correction for avoiding reading past self.stop
#         if self.nrowsread+self.nrowsinbuf > self.stop:
#           stop = self.stop-self.nrowsread
#         else:
#           stop = self.nrowsinbuf
#         if self.coords is not None:
#           self.bufcoords = self.coords[self.nrowsread:self.nrowsread+stop]
#         else:
#           self.bufcoords = self.index.getCoords(self.nrowsread, stop)
#         self._row = -1
#         recout = self._table._read_elements(0, self.bufcoords)
#         if self._table.byteorder <> sys.byteorder:
#           self._recarray._byteswap()
#         self.nrowsread = self.nrowsread + recout
#       self._row = self._row + 1
#       self._nrow = self.bufcoords[self._row]
#       self.nextelement = self.nextelement + 1
#       # Return this row
#       return self
#     else:
#       # Re-initialize the possible cuts in columns
#       self.indexed = 0
#       if self.coords is None:
#         self.index.indices._destroyIndexSlice()  # Remove buffers in indices
#         nextelement = self.index.nelemslice * self.index.nrows
#       else:
#         self.coords = None
#         # All the elements has been read for this mode
#         nextelement = self.nrows
#       if nextelement >= self.nrows:
#         self._table._close_read()  # Close the table
#         self._table.ops = []
#         self._table.opsValues = []
#         self._table.whereColname = None
#         self.index = 0
#         self.whereCond = 0
#         raise StopIteration        # end of iteration
#       else:
#         # Continue the iteration with the __next__inKernel() method
#         self.start = nextelement
#         self.stop = self.nrows
#         self.step = 1
#         self.startb = 0
#         self.nrowsread = self.start
#         self._nrow = self.start - self.step
#         return self.__next__inKernel()

  cdef __next__inKernel(self):
    """The version of next() in case of in-kernel conditions"""
    cdef long offset
    cdef object indexValid1, indexValid2
    cdef int ncond, op, recout, correct
    cdef object opValue, field

    self.nextelement = self._nrow + self.step
    while self.nextelement < self.stop:
      if self.nextelement >= self.nrowsread:
        # Skip until there is interesting information
        while self.nextelement >= self.nrowsread + self.nrowsinbuf:
          self.nrowsread = self.nrowsread + self.nrowsinbuf
        # Compute the end for this iteration
        self.stopb = self.stop - self.nrowsread
        if self.stopb > self.nrowsinbuf:
          self.stopb = self.nrowsinbuf
        self._row = self.startb - self.step
        # Read a chunk
        recout = self._table._read_records(self.nextelement,
                                           self.nrowsinbuf)
        self.nrowsread = self.nrowsread + recout
        if self._table.byteorder <> sys.byteorder:
          self._recarray._byteswap()
        # The next assignment should be faster, but does not work!
        #self._recarray._byteorder = self._table.byteorder
        self.indexChunk = -self.step
        # Iterate over the conditions
        ncond = 0
        for op in self._table.ops:
          opValue = self._table.opsValues[ncond]
          # Copying first on a non-strided array, reduces the speed
          # in a factor of 20%
          #field = self._fields[self.colname].copy()
          if op == 1:
            indexValid1 = self._fields[self.colname].__lt__(opValue)
          elif op == 2:
            indexValid1 = self._fields[self.colname].__le__(opValue)
          elif op == 3:
            indexValid1 = self._fields[self.colname].__gt__(opValue)
          elif op == 4:
            indexValid1 = self._fields[self.colname].__ge__(opValue)
          elif op == 5:
            indexValid1 = self._fields[self.colname].__eq__(opValue)
          elif op == 6:
            indexValid1 = self._fields[self.colname].__ne__(opValue)
          # Consolidate the valid indexes
          if ncond == 0:
            self.indexValid = indexValid1
          else:
            self.indexValid = self.indexValid.__and__(indexValid1)
          ncond = ncond + 1

        # This indexing operation is *very* costly, so it is better
        # to keep the boolean (indexValid) approach.
        #result = self._recarray[self.indexValid]
        #if len(result) == 0:
        # Is still there any interesting information in this buffer?
        if not numarray.sometrue(self.indexValid):
          # No, so take the next one
          if self.step >= self.nrowsinbuf:
            self.nextelement = self.nextelement + self.step
          else:
            self.nextelement = self.nextelement + self.nrowsinbuf
            # Correction for step size > 1
            if self.step > 1:
              correct = (self.nextelement - self.start) % self.step
              self.nextelement = self.nextelement + self.step - correct
          continue
      
      self._row = self._row + self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = 0

      self.nextelement = self._nrow + self.step
      # Return only if this value is interesting
      self.indexChunk = self.indexChunk + self.step
      if self.indexValid[self.indexChunk]:
        return self
    else:
      self._table._close_read()  # Close the table
      # Re-initialize the possible cuts in columns
      self._table.ops = []
      self._table.opsValues = []
      self._table.whereColname = None
      self.whereCond = 0
      raise StopIteration        # end of iteration

  # This is the most general __next__ version, simple, but effective
  cdef __next__general(self):
    """The version of next() for the general cases"""
    cdef long offset
    cdef object indexValid1, indexValid2
    cdef int ncond, op, recout
    cdef object opValue, field

    self.nextelement = self._nrow + self.step
    while self.nextelement < self.stop:
      if self.nextelement >= self.nrowsread:
        # Skip until there is interesting information
        while self.nextelement >= self.nrowsread + self.nrowsinbuf:
          self.nrowsread = self.nrowsread + self.nrowsinbuf
        # Compute the end for this iteration
        self.stopb = self.stop - self.nrowsread
        if self.stopb > self.nrowsinbuf:
          self.stopb = self.nrowsinbuf
        self._row = self.startb - self.step
        # Read a chunk
        recout = self._table._read_records(self.nrowsread,
                                           self.nrowsinbuf)
        self.nrowsread = self.nrowsread + recout
        if self._table.byteorder <> sys.byteorder:
          self._recarray._byteswap()
        # This should be faster but doesn't seem to work
        #self._recarray._byteorder = self._table.byteorder
      
      self._row = self._row + self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = (self._row + self.step) % self.nrowsinbuf

      self.nextelement = self._nrow + self.step
      # Return this value
      return self
    else:
      self._table._close_read()  # Close the table
      raise StopIteration        # end of iteration

  def _fillCol(self, result, start, stop, step, field):
    "Read a field from a table on disk and put the result in result"
    cdef int startr, stopr, i, j, istartb, istopb
    cdef int istart, istop, istep, inrowsinbuf, inextelement, inrowsread
    cdef object fields
    
    self._initLoop(start, stop, step, None, 0)
    istart, istop, istep = (self.start, self.stop, self.step)
    inrowsinbuf, inextelement, inrowsread = (self.nrowsinbuf, istart, istart)
    istartb, startr = (self.startb, 0)
    if field:
      # If field is not None, select it
      fields = self._recarray._fields[field]
    else:
      # if don't, select all fields
      fields = self._recarray
    i = istart
    while i < istop:
      if (inextelement >= inrowsread + inrowsinbuf):
        inrowsread = inrowsread + inrowsinbuf
        i = i + inrowsinbuf
        continue
      # Compute the end for this iteration
      istopb = istop - inrowsread
      if istopb > inrowsinbuf:
        istopb = inrowsinbuf
      stopr = startr + ((istopb - istartb - 1) / istep) + 1
      # Read a chunk
      inrowsread = inrowsread + self._table._read_records(i, inrowsinbuf)
      # Assign the correct part to result
      # The bottleneck is in this assignment. Hope that the numarray
      # people might improve this in the short future
      result[startr:stopr] = fields[istartb:istopb:istep]
      # Compute some indexes for the next iteration
      startr = stopr
      j = istartb + ((istopb - istartb - 1) / istep) * istep
      istartb = (j+istep) % inrowsinbuf
      inextelement = inextelement + istep
      i = i + inrowsinbuf
    self._table._close_read()  # Close the table
    return

  def nrow(self):
    """ get the global row number for this table """
    return self._nrow

  def append(self):
    """Append self object to the output buffer.
    
    """
    assert self._table._v_file.mode <> "r", "Attempt to write over a file opened in read-only mode"
    if not self._w_initialized_buffer:
      # Create the arrays for buffering
      self._newBuffer(write=1)
    self._row = self._row + 1 # update the current buffer read counter
    self._unsavednrows = self._unsavednrows + 1
    if self._table.indexed:
      self._table._unsaved_indexedrows = self._table._unsaved_indexedrows + 1
    # When the buffer is full, flush it
    if self._unsavednrows == self.nrowsinbuf:
      # Save the records on disk
      self._saveBufferedRows()
      # Get again the self._fields of the new buffer
      self._fields = self._table._v_buffer._fields
      
    return
      
  def _setUnsavedNRows(self, row):
    """ set the buffer row number for this buffer """
    self._unsavednrows = row
    self._row = row # set the current buffer read counter

  def _getUnsavedNRows(self):
    """ get the buffer row number for this buffer """
    return self._unsavednrows

  def _incUnsavedNRows(self):
    """ set the row for this record object """
    self._row = self._row + 1 # update the current buffer read counter
    self._unsavednrows = self._unsavednrows + 1
    return self._unsavednrows

#   def __getitem__orig(self, fieldName):
#     try:
#       return self._fields[fieldName][self._row]
#       #return 40  # Just for testing purposes
#     except:
#       (type, value, traceback) = sys.exc_info()
#       raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
#             (fieldName, "Error was: \"%s: %s\"" % (type,value))

  # This method is twice as faster than __getattr__ because there is
  # not a lookup in the local dictionary
  def __getitem__(self, fieldName):
    cdef int index
    cdef long offset

    # Optimization follows for the case that the field dimension is
    # == 1, i.e. columns elements are scalars, and the column is not
    # of CharType. This code accelerates the access to column
    # elements a 20%

    try:
      # Get the column index. This is very fast!
      index = self._indexes[fieldName]
      if (self._enumtypes[index] <> CHARTYPE and self._scalar[index]):
        #return 40   # Just for tests purposes
        # if not NA_updateDataPtr(self._fields[fieldName]):
        #  return None
        # This optimization sucks when using numarray 0.4!
        # And it works better with python 2.2 than python 2.3
        # I'll disable it until further study of it is done
        #
        # I'm going to activate this optimization from 0.7.1 on
        # 2003/08/08
        offset = self._row * self._strides
        return NA_getPythonScalar(self._fields[fieldName], offset)
        #return self._fields[fieldName][self._row]
      elif (self._enumtypes[index] == CHARTYPE and self._scalar[index]):
        # Case of a plain string in the cell
        # Call the universal indexing function
        return self._fields[fieldName][self._row]
      else:  # Case when dimensions > 1
        # Call the universal indexing function
        # Make a copy of the (multi) dimensional array
        # so that the user does not have to do that!
        arr = self._fields[fieldName][self._row].copy()
        return arr
    except:
      (type, value, traceback) = sys.exc_info()
      raise KeyError, "Error accessing \"%s\" field.\n %s" % \
	    (fieldName, "Error was: \"%s: %s\"" % (type,value))

  # This is slightly faster (around 3%) than __setattr__
  def __setitem__(self, object fieldName, object value):

    assert self._table._v_file.mode <> "r", "Attempt to write over a file opened in read-only mode"
    if not self._w_initialized_buffer:
      # Create the arrays for buffering
      self._newBuffer(write=1)

    try:
      self._fields[fieldName][self._unsavednrows] = value
    except:
      (type, value, traceback) = sys.exc_info()
      raise KeyError, "Error setting \"%s\" field.\n %s" % \
           (fieldName, "Error was: \"%s: %s\"" % (type,value))

  # Delete the I/O buffers
  def _cleanup(self):
    self._fields = None         # Decrement the pointer to recarray fields
    self._table._v_buffer = None   # Decrement the pointer to recarray
    self._table._v_buffercpy = None  # Decrement the pointer to recarray copy
    # Flag that tells that the buffer has been uninitialized
    self._r_initialized_buffer = 0
    self._w_initialized_buffer = 0
    
  def __str__(self):
    """ represent the record as an string """
        
    outlist = []
    # Special case where Row has not been initialized yet
    if self._recarray == None:
      return "Row object has not been initialized for table:\n  %s\n %s" % \
             (self._table, \
    "You will normally want to use row objects in combination with iterators.")
    for name in self._recarray._names:
      outlist.append(`self._fields[name][self._row]`)
    return "(" + ", ".join(outlist) + ")"

  def __repr__(self):
    """ represent the record as an string """

    return str(self)

  def __dealloc__(self):
    #print "Deleting Row object"
    free(<void *>self._scalar)
    free(<void *>self._enumtypes)


cdef class Array:
  # Instance variables
  cdef hid_t   parent_id, type_id
  cdef char    *name
  cdef int     rank
  cdef hsize_t *dims
  cdef hsize_t *dims_chunk
  cdef int     enumtype
  cdef hsize_t stride[2]

  def _g_new(self, where, name):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_objectID

  def _createArray(self, object naarr, char *title):
    cdef int i
    cdef herr_t ret
    cdef hid_t oid
    cdef void *rbuf
    cdef long buflen
    cdef int itemsize, offset
    cdef char *byteorder
    cdef char *flavor, *complib, *version
    cdef int extdim
    cdef object  type

    if isinstance(naarr, strings.CharArray):
      type = CharType
      self.enumtype = toenum[CharType]
    else:
      type = naarr._type
      try:
        self.enumtype = toenum[naarr._type]
      except KeyError:
        raise TypeError, \
      """Type class '%s' not supported rigth now. Sorry about that.
      """ % repr(naarr._type)

    itemsize = naarr._itemsize
    byteorder = PyString_AsString(self.byteorder)
    self.type_id = convArrayType(self.enumtype, itemsize, byteorder)
    if self.type_id < 0:
      raise TypeError, \
        """type '%s' is not supported right now. Sorry about that.""" \
    % type

    # Get the pointer to the buffer data area
    # the second parameter means whether the buffer is read-only or not
    buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)
    # Correct the start of the buffer with the _byteoffset
    offset = naarr._byteoffset
    rbuf = <void *>(<char *>rbuf + offset)

    # Allocate space for the dimension axis info
    self.rank = len(naarr.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < self.rank:
        self.dims[i] = naarr.shape[i]

    # Save the array
    flavor = PyString_AsString(self.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    oid = H5ARRAYmake(self.parent_id, self.name, "ARRAY", title,
                      flavor, version, self.rank, self.dims, self.extdim,
                      self.type_id, NULL, rbuf,
                      self.filters.complevel, complib,
                      self.filters.shuffle, self.filters.fletcher32,
                      rbuf)
    if oid < 0:
      raise RuntimeError("Problems creating the EArray.")
    self.objectID = oid
    H5Tclose(self.type_id)    # Release resources

    return type
    
  def _createEArray(self, char *klass, char *title):
    cdef int i
    cdef herr_t ret
    cdef hid_t oid
    cdef void *rbuf
    cdef char *byteorder
    cdef char *flavor, *complib, *version

    try:
      self.enumtype = toenum[self.type]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported rigth now. Sorry about that.
            """ % repr(self.type)

    byteorder = PyString_AsString(self.byteorder)
    self.type_id = convArrayType(self.enumtype, self.atom.itemsize, byteorder)
    if self.type_id < 0:
      raise TypeError, \
        """type '%s' is not supported right now. Sorry about that.""" \
    % self.type

    self.rank = len(self.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
        self.dims[i] = self.shape[i]
        self.dims_chunk[i] = self._v_chunksize[i]

    rbuf = NULL   # The data pointer. We don't have data to save initially
    # Manually convert some string values that can't be done automatically
    flavor = PyString_AsString(self.atom.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    # Create the EArray
    oid = H5ARRAYmake(self.parent_id, self.name, klass, title,
                      flavor, version, self.rank, self.dims, self.extdim,
                      self.type_id, self.dims_chunk, rbuf,
                      self.filters.complevel, complib,
                      self.filters.shuffle, self.filters.fletcher32,
                      rbuf)
    if oid < 0:
      raise RuntimeError("Problems creating the EArray.")
    self.objectID = oid
    # Release resources
    H5Tclose(self.type_id)
    return
    
  def _append(self, object naarr):
    cdef int ret, rank
    cdef hsize_t *dims_arr
    cdef void *rbuf
    cdef long offset
    cdef int buflen
    cdef object shape

    # Allocate space for the dimension axis info
    rank = len(naarr.shape)
    dims_arr = <hsize_t *>malloc(rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < rank:
        dims_arr[i] = naarr.shape[i]

    # Get the pointer to the buffer data area
    # Both methods do the same
    buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)
    #if ( PyObject_AsReadBuffer(naarr._data, &rbuf, &buflen) < 0 ):
    #  raise RuntimeError, "Error getting the buffer location"
    # Correct the start of the buffer with the _byteoffset
    offset = naarr._byteoffset
    rbuf = <void *>(<char *>rbuf + offset)

    # Append the records:
    ret = H5ARRAYappend_records(self.parent_id, self.name, self.rank,
                                self.dims, dims_arr, self.extdim, rbuf)

    if ret < 0:
      raise RuntimeError("Problems appending the records.")
    free(dims_arr)
    # Update the new dimensionality
    shape = list(self.shape)
    shape[self.extdim] = self.dims[self.extdim]
    self.shape = tuple(shape)
    self.nrows = self.dims[self.extdim]
    
  def _openArray(self):
    cdef object shape
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef H5T_sign_t sign
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i
    cdef hid_t oid
    cdef herr_t ret
    cdef int extdim
    cdef char flavor[256]
    cdef char version[8]
    cdef double fversion

    # Get the rank for this array object
    ret = H5ARRAYget_ndims(self.parent_id, self.name, &self.rank)
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type size
    oid = H5ARRAYget_info(self.parent_id, self.name, self.dims,
                          &self.type_id, &class_id, byteorder)
    self.objectID = oid
    strcpy(flavor, "unknown")  # Default value
    if self._v_file._isPTFile:
      H5LTget_attribute_string(self.parent_id, self.name, "FLAVOR", flavor)
      H5LTget_attribute_string(self.parent_id, self.name, "VERSION", version)
      fversion = atof(version)
      if (self.__class__.__name__ == "EArray" or 
          self.__class__.__name__ == "IndexArray"):
        # For EArray, EXTDIM attribute exists
        H5LTget_attribute_int(self.parent_id, self.name, "EXTDIM", &extdim)
        self.extdim = extdim
        # Allocate space for the dimension chunking info
        self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
        if ( (H5ARRAYget_chunksize(self.parent_id, self.name,
                                   self.rank, self.dims_chunk)) < 0):
          raise RuntimeError, "Problems getting the chunksizes!"
    self.flavor = flavor  # Gives class visibility to flavor

    # Get the array type
    type_size = getArrayType(self.type_id, &self.enumtype)
    if type_size < 0:
      raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id

    H5Tclose(self.type_id)    # Release resources

    # We had problems when creating Tuples directly with Pyrex!.
    # A bug report has been sent to Greg Ewing and here is his answer:
    """

    It's impossible to call PyTuple_SetItem and PyTuple_GetItem
    correctly from Pyrex, because they don't follow the standard
    reference counting protocol (PyTuple_GetItem returns a borrowed
    reference, and PyTuple_SetItem steals a reference).

    It's best to use Python constructs to create tuples if you
    can. Otherwise, you could create wrapppers for these functions in
    an external C file which provide standard reference counting
    behaviour.

    """
    # So, I've decided to create the shape tuple using Python constructs
    shape = []
    chunksizes = []
    for i from 0 <= i < self.rank:
      # The <int> cast avoids returning a Long integer
      shape.append(<int>self.dims[i])
      if (self.__class__.__name__ == "EArray" or 
          self.__class__.__name__ == "IndexArray"):
        chunksizes.append(<int>self.dims_chunk[i])
    shape = tuple(shape)
    chunksizes = tuple(chunksizes)

    return (toclass[self.enumtype], shape, type_size, byteorder, chunksizes)
  
  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step,
                 object buf):
    cdef herr_t ret
    cdef void *rbuf
    cdef long buflen
    cdef hsize_t nrows
    cdef int extdim

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(buf, 1, &rbuf)

    # Number of rows to read
    nrows = ((stop - start - 1) / step) + 1  # (stop-start)/step  do not work
    if hasattr(self, "extdim"):
      extdim = self.extdim
    else:
      exdim = -1
    ret = H5ARRAYread(self.parent_id, self.name, start, nrows, step,
                      extdim, rbuf)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    return 

  def _g_readSlice(self, startl, stopl, stepl, bufferl):
    cdef herr_t ret
    cdef long ndims, buflen
    cdef void *startlb, *stoplb, *steplb, *rbuflb
    cdef long offset

    # Get the pointer to the buffer data area of startl, stopl and stepl arrays
    ndims = NA_getBufferPtrAndSize(startl._data, 1, &startlb)
    ndims = NA_getBufferPtrAndSize(stopl._data, 1, &stoplb)
    ndims = NA_getBufferPtrAndSize(stepl._data, 1, &steplb)
    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(bufferl._data, 1, &rbuflb)
    # Correct the start of the buffer with the _byteoffset
    offset = bufferl._byteoffset
    rbuflb = <void *>(<char *>rbuflb + offset)
    # Do the physical read
    ret = H5ARRAYreadSlice(self.parent_id, self.name,
                           <hsize_t *>startlb, <hsize_t *>stoplb,
                           <hsize_t *>steplb, rbuflb)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    return 

  def __dealloc__(self):
    #print "Destroying object Array in Extension"
    free(<void *>self.dims)
    free(<void *>self.name)
    if self.dims_chunk:
      free(self.dims_chunk)


cdef class IndexArray(Array):
  """Homogeneous dataset for keeping sorted and index values"""
  cdef void    *rbuflb, *vrbufR, *vrbufA
  cdef hid_t   dataset_id, type_id2, space_id
  # It is necessary for arrRel and arrAbs to be accessible in IndexArray,
  # so let's this commented out
  #cdef object arrRel, arrAbs

  def _initIndexSlice(self, maxCoords):
    "Initialize the structures for doing a binary search"
    cdef long buflen

    # Get the pointer to the buffer data area
    self.arrRel = numarray.zeros(type="Int32",shape=(maxCoords,))
    self.arrAbs = numarray.zeros(type="Int64",shape=(maxCoords,))
    buflen = NA_getBufferPtrAndSize(self.arrRel._data, 1, &self.vrbufR)
    buflen = NA_getBufferPtrAndSize(self.arrAbs._data, 1, &self.vrbufA)

    # Open the array for reading
    if (H5ARRAYOopen_readSlice(&self.dataset_id, &self.space_id,
                               &self.type_id2, self.parent_id, self.name) < 0):
      raise RuntimeError("Problems opening the sorted array data.")

  def _g_readIndex(self, hsize_t irow, hsize_t start, hsize_t stop,
                   long offsetl):
    cdef herr_t ret
    cdef long buflen
    cdef int *rbufR
    cdef long long *rbufA
    cdef long long offset
    cdef long j, len

    # Correct the start of the buffer with offsetl
    rbufR = <int *>self.vrbufR + offsetl
    rbufA = <long long *>self.vrbufA + offsetl
    # Do the physical read
    ret = H5ARRAYOread_readSlice(self.dataset_id, self.space_id, self.type_id2,
                                 irow, start, stop, rbufR)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    # Now, compute the absolute coords for table rows by adding the offset
    len = stop-start
    offset = irow*self.nelemslice
    for j from 0 <= j < len:
      rbufA[j] = rbufR[j] + offset
      
    return 

  def _destroyIndexSlice(self):
    # Close the array for reading
    if H5ARRAYOclose_readSlice(self.dataset_id, self.space_id,
                               self.type_id2) < 0:
      raise RuntimeError("Problems closing the sorted array data.")

  def _initSortedSlice(self, int bufsize):
    "Initialize the structures for doing a binary search"
    cdef long ndims
    cdef int buflen

    # Create the buffer for reading sorted data chunks
    if str(self.type) == "CharType":
      self.bufferl = strings.array(None, itemsize=self.itemsize, shape=bufsize)
    else:
      self.bufferl = numarray.array(None, type=self.type, shape=bufsize)
      # Set the same byteorder than on-disk
      self.bufferl._byteorder = self.byteorder

    # Get the pointer to the buffer data area
    # Both methods do the same
    buflen = NA_getBufferPtrAndSize(self.bufferl._data, 1, &self.rbuflb)
    #if (PyObject_AsReadBuffer(self.bufferl._data, &self.rbuflb, &buflen) < 0):
    #  print "Error getting buffer location"

    # Open the array for reading
    if (H5ARRAYOopen_readSlice(&self.dataset_id, &self.space_id,
                               &self.type_id2, self.parent_id, self.name) < 0):
      raise RuntimeError("Problems opening the sorted array data.")

  def _readSortedSlice(self, hsize_t irow, hsize_t start, hsize_t stop):
    "Read the sorted part of an index"

    ret = H5ARRAYOread_readSlice(self.dataset_id, self.space_id, self.type_id2,
                                 irow, start, stop, self.rbuflb)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    return self.bufferl

  def _destroySortedSlice(self):
    del self.bufferl
    # Close the array for reading
    if H5ARRAYOclose_readSlice(self.dataset_id, self.space_id,
                               self.type_id2) < 0:
      raise RuntimeError("Problems closing the sorted array data.")

# This has been copied from the standard module bisect.
# Checks for the values out of limits has been added at the beginning
# because I forsee that this should be a very common case.
# 2004-05-20
  def _bisect_left(self, a, x, int hi):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, i points just
    before the leftmost x already there.

    """
    cdef int lo, mid

    lo = 0
    if x <= a[0]: return 0
    if a[-1] < x: return hi
    while lo < hi:
        mid = (lo+hi)/2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo

  def _bisect_right(self, a, x, int hi):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, i points just
    beyond the rightmost x already there.

    """
    cdef int lo, mid

    lo = 0
    if x < a[0]: return 0
    if a[-1] <= x: return hi
    while lo < hi:
      mid = (lo+hi)/2
      if x < a[mid]: hi = mid
      else: lo = mid+1
    return lo

  def _interSearch_left(self, int nrow, int chunksize, item, int lo, int hi):
    cdef int niter, mid, start, result, beginning
    
    niter = 0
    beginning = 0
    while lo < hi:
      mid = (lo+hi)/2
      start = (mid/chunksize)*chunksize
      buffer = self._readSortedSlice(nrow, start, start+chunksize)
      #buffer = xrange(start,start+chunksize) # test
      niter = niter + 1
      result = self._bisect_left(buffer, item, chunksize)
      if result == 0:
        if buffer[result] == item:
          lo = start
          beginning = 1
          break
        # The item is at left
        hi = mid
      elif result == chunksize:
        # The item is at the right
        lo = mid+1
      else:
        # Item has been found. Exit the loop and return
        lo = result+start
        break
    return (lo, beginning, niter)

  def _interSearch_right(self, int nrow, int chunksize, item, int lo, int hi):
    cdef int niter, mid, start, result, ending
    
    niter = 0
    ending = 0
    while lo < hi:
      mid = (lo+hi)/2
      start = (mid/chunksize)*chunksize
      buffer = self._readSortedSlice(nrow, start, start+chunksize)
      niter = niter + 1
      result = self._bisect_right(buffer, item, chunksize)
      if result == 0:
        # The item is at left
        hi = mid
      elif result == chunksize:
        if buffer[result-1] == item:
          lo = start+chunksize
          ending = 1
          break
        # The item is at the right
        lo = mid+1
      else:
        # Item has been found. Exit the loop and return
        lo = result+start
        break
    return (lo, ending, niter)

# This is coded in python space as well, but the improvement in speed
# here is very little. So, it's better to let _searchBin live there.

#   def _searchBin(self, int nrow, object item):
#     cdef int hi, lo, chunksize, niter, item1done, item2done
#     cdef int result1, result2, tmpresult1, tmpresult2, nelemslice
#     cdef int beginning, ending, iter

#     nelemslice = self.shape[1]
#     hi = nelemslice   
#     item1, item2 = item
#     item1done = 0; item2done = 0
#     chunksize = self._v_chunksize[1] # Number of elements/chunksize

#     # First, look at the beginning of the slice (that could save lots of time)
#     buffer = self._readSortedSlice(nrow, 0, chunksize)
#     #buffer = xrange(0, chunksize)  # test  # 0.02 over 0.5 seg
#     # Look for items at the beginning of sorted slices
#     niter = 1
#     result1 = self._bisect_left(buffer, item1, chunksize)
#     if 0 <= result1 < chunksize:
#       item1done = 1
#     result2 = self._bisect_right(buffer, item2, chunksize)
#     #print "item1done, item2done-->", item1done, item2done
#     #print "result1, result2-->", result1, result2
#     if 0 <= result2 < chunksize:
#       item2done = 1
#       # Commented out. The value can be repeated in the next chunk
# #     elif buffer[chunksize-1] == item2:
# #       item2done = 1
#     if item1done and item2done:
#       #print "done 1"
#       return (result1, result2, niter)
    
#     # Then, look for items at the end of the sorted slice
#     buffer = self._readSortedSlice(nrow, hi-chunksize, hi)
#     #buffer = xrange(hi-chunksize, hi)  # test
#     niter = 2
#     #print "item1done, item2done-->", item1done, item2done
#     if not item1done:
#       result1 = self._bisect_left(buffer, item1, chunksize)
#       if 0 < result1 <= chunksize:
#         item1done = 1
#         result1 = hi - chunksize + result1
#         # Commented out. The value can be repeated in the previous chunk
# #       elif buffer[0] == item1:
# #         item1done = 1
# #         result1 = hi - chunksize
#     #print "item1done, item2done-->", item1done, item2done
#     if not item2done:
#       result2 = self._bisect_right(buffer, item2, chunksize)
#       if 0 < result2 <= chunksize:
#         item2done = 1
#         result2 = hi - chunksize + result2
#     if item1done and item2done:
#       #print "done 2"
#       return (result1, result2, niter)
#     #print "item1done, item2done-->", item1done, item2done
    
#     # Finally, do a lookup for item1 and item2 if they were not found
#     # Lookup in the middle of slice for item1
#     if not item1done:
#       lo = 0
#       hi = nelemslice
#       beginning = 1
#       result1 = 1  # a number different from 0
#       while beginning and result1 != 0:
#         (result1, beginning, iter) = self._interSearch_left(nrow, chunksize,
#                                                             item1, lo, hi)
#         tmpresult1 = result1
#         niter = niter + iter
#         if result1 == hi:  # The item is completely at right
#           break
#         else:
#           hi = result1        # one chunk to the left
#           lo = hi - chunksize  
#           #print "lo, hi, beginning-->", lo, hi, beginning
#       result1 = tmpresult1
#     # Lookup in the middle of slice for item1
#     if not item2done:
#       lo = 0
#       hi = nelemslice
#       ending = 1
#       result2 = 1  # a number different from 0
#       while ending and result2 != nelemslice:
#         (result2, ending, iter) = self._interSearch_right(nrow, chunksize,
#                                                           item2, lo, hi)
#         tmpresult2 = result2
#         niter = niter + iter
#         if result2 == lo:  # The item is completely at left
#           break
#         else:
#           hi = result2 + chunksize      # one chunk to the right
#           lo = result2
#           #print "lo, hi, ending-->", lo, hi, ending
#       result2 = tmpresult2
#       niter = niter + iter
#     #print "done 3"
#     return (result1, result2, niter)


cdef class Index:
  """Container for sorted and indices datasets"""
  # Instance variables

  # Methods
  pass
#   def search(self, item, notequal):
#     """Do a binary search in this index for an item"""
#     cdef long ntotaliter, tlen, bufsize, i, start, stop, niter
    
#     #t1=time.time()
#     ntotaliter = 0; tlen = 0
#     self.starts = []; self.lengths = []
#     bufsize = self.sorted._v_chunksize[1] # number of elements/chunksize
#     self.nelemslice = self.sorted.nelemslice   # number of columns/slice
#     self.sorted._initSortedSlice(bufsize)
#     # Do the lookup for values fullfilling the conditions
#     #for i in xrange(self.sorted.nrows):
#     for i from 0 <= i < self.sorted.nrows:
#       (start, stop, niter) = self.sorted._searchBin(i, item)
#       self.starts.append(start)
#       self.lengths.append(stop - start)
#       ntotaliter = ntotaliter + niter
#       tlen = tlen + (stop - start)
#     self.sorted._destroySortedSlice()
#     #print "time reading indices:", time.time()-t1
#     return tlen

#   def getCoords(self, long long startCoords, long maxCoords):
#     """Get the coordinates of indices satisfiying the cuts"""
#     cdef long len1, len2, leni, stop, relCoords, irow, startl, stopl
    
#     #t1=time.time()
#     len1 = 0; len2 = 0;
#     stop = 0; relCoords = 0
#     for irow from  0 <= irow < self.sorted.nrows: 
#       leni = self.lengths[irow]; len2 = len2 + leni
#       if (leni > 0 and len1 <= startCoords < len2):
#         startl = self.starts[irow] + (startCoords-len1)
#         if maxCoords >= leni - (startCoords-len1):
#           # Values fit on buffer
#           stopl = startl + leni
#         else:
#           # Stop after this iteration
#           stopl = startl + maxCoords
#           stop = 1
#         self.indices._g_readIndex(irow, startl, stopl, relCoords)
#         relCoords = relCoords + stopl - startl
#         if stop:
#           break
#         maxCoords = maxCoords - (leni - (startCoords-len1))
#         startCoords = startCoords + (leni - (startCoords-len1))
#       len1 = len1 + leni
                
#     selections = numarray.sort(self.indices.arrAbs[:relCoords])
#     #selections = self.indices.arrAbs[:relCoords]
#     #print "time doing revIndexing:", time.time()-t1
#     return selections


cdef class VLArray:
  # Instance variables
  cdef hid_t   parent_id
  cdef hid_t   oid
  cdef char    *name
  cdef int     rank
  cdef hsize_t *dims
  cdef object  type
  cdef int     enumtype
  cdef hid_t   type_id
  cdef hsize_t nrecords
  cdef int     scalar

  def _g_new(self, where, name):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_objectID

  def _createArray(self, char *title):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef char *byteorder
    cdef char *flavor, *complib, *version

    self.type = self.atom.type
    try:
      self.enumtype = toenum[self.type]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported rigth now. Sorry about that.
            """ % repr(self.type)

    byteorder = PyString_AsString(self.byteorder)
    # Get the HDF5 type id
    self.type_id = convArrayType(self.enumtype, self.atom.itemsize, byteorder)
    if self.type_id < 0:
      raise TypeError, \
        """type '%s' is not supported right now. Sorry about that.""" \
    % self.type

    # Allocate space for the dimension axis info
    if isinstance(self.atom.shape, types.IntType):
      self.rank = 1
      self.scalar = 1
    else:
      self.rank = len(self.atom.shape)
      self.scalar = 0
      
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      if isinstance(self.atom.shape, types.IntType):
        self.dims[i] = self.atom.shape
      else:
        self.dims[i] = self.atom.shape[i]

    rbuf = NULL   # We don't have data to save initially

    # Manually convert some string values that can't be done automatically
    flavor = PyString_AsString(self.atom.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    # Create the vlarray
    oid = H5VLARRAYmake(self.parent_id, self.name, title,
                        flavor, version, self.rank, self.scalar,
                        self.dims, self.type_id, self._v_chunksize, rbuf,
                        self.filters.complevel, complib,
                        self.filters.shuffle, self.filters.fletcher32,
                        rbuf)
    if oid < 0:
      raise RuntimeError("Problems creating the VLArray.")
    self.objectID = oid
    self.nrecords = 0  # Initialize the number of records saved

    return self.type
    
  def _append(self, object naarr, int nobjects):
    cdef int ret
    cdef void *rbuf
    cdef long buflen, offset

    # Get the pointer to the buffer data area
    if nobjects:
      buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)
      # Correct the start of the buffer with the _byteoffset
      offset = naarr._byteoffset
      rbuf = <void *>(<char *>rbuf + offset)
    else:
      rbuf = NULL

    # Append the records:
    ret = H5VLARRAYappend_records(self.parent_id, self.name,
                                  nobjects, self.nrecords,
                                  rbuf)
    if ret < 0:
      raise RuntimeError("Problems appending the records.")

    self.nrecords = self.nrecords + 1

    return self.nrecords

  def _openArray(self):
    cdef object shape
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i
    cdef hid_t oid
    cdef herr_t ret
    cdef hsize_t nrecords[1]
    cdef char flavor[256]
    
    # Get the rank for the atom in the array object
    ret = H5VLARRAYget_ndims(self.parent_id, self.name, &self.rank)
    # Allocate space for the dimension axis info
    if self.rank:
      self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    else:
      self.dims = NULL;
    # Get info on dimensions, class and type size
    oid = H5VLARRAYget_info(self.parent_id, self.name, nrecords,
                            self.dims, &self.type_id, byteorder)
    self.objectID = oid
    ret = H5LTget_attribute_string(self.parent_id, self.name, "FLAVOR", flavor)
    if ret < 0:
      strcpy(flavor, "unknown")

    # Give to these variables class visibility
    self.flavor = flavor
    self.byteorder = byteorder

    # Get the array type
    self._basesize = getArrayType(self.type_id, &self.enumtype)
    if self._basesize < 0:
      raise TypeError, "The HDF5 class of object does not seem VLEN. Sorry!"

    # Get the type of the atomic type
    self._atomictype = toclass[self.enumtype]
    # Get the size and shape of the atomic type
    self._atomicsize = self._basesize
    if self.rank:
      shape = []
      for i from 0 <= i < self.rank:
        # The <int> cast avoids returning a Long integer
        shape.append(<int>self.dims[i])
        self._atomicsize = self._atomicsize * <int>self.dims[i]
      shape = tuple(shape)
    else:
      # rank zero means a scalar
      shape = 1

    self._atomicshape = shape
    # The <int> cast avoids returning a Long integer
    return <int>nrecords[0]

  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step):
    cdef herr_t ret
    cdef hvl_t *rdata   # Information read in
    cdef size_t vllen
    cdef hsize_t rdatalen
    cdef object rbuf, naarr, shape, datalist
    cdef int i
    cdef hsize_t nrows

    # Compute the number of rows to read
    nrows = ((stop - start - 1) / step) + 1  # (stop-start)/step  do not work
    rdata = <hvl_t *>malloc(<size_t>nrows*sizeof(hvl_t))
    ret = H5VLARRAYread(self.parent_id, self.name, start, nrows, step, rdata,
                        &rdatalen)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    datalist = []
    for i from 0 <= i < nrows:
      # Number of atoms in row
      vllen = rdata[i].len
      # Get the pointer to the buffer data area
      if vllen > 0:
        rbuf = PyBuffer_FromReadWriteMemory(rdata[i].p, vllen*self._atomicsize)
        if not rbuf:
          raise RuntimeError("Problems creating python buffer for read data.")
      else:
        # Case where there is info with zero lentgh
        rbuf = None
      # Compute the shape for the read array
      if (isinstance(self._atomicshape, types.TupleType)):
        shape = list(self._atomicshape)
        shape.insert(0, vllen)  # put the length at the beginning of the shape
      elif self._atomicshape > 1:
        shape = (vllen, self._atomicshape)
      else:
        # Case of scalars (self._atomicshape == 1)
        shape = (vllen,)
      # Create an array to keep this info
      if str(self._atomictype) == "CharType":
        naarr = strings.array(rbuf, itemsize=self._basesize, shape=shape)
      else:
        naarr = numarray.array(rbuf, type=self._atomictype, shape=shape)
      datalist.append(naarr)

    # Release resources
    free(rdata)
    
    return datalist

  def __dealloc__(self):
    #print "Destroying object VLArray in Extension"
    H5Tclose(self.type_id)    # To avoid memory leaks
    if self.dims:
      free(<void *>self.dims)
    free(<void *>self.name)


cdef class UnImplemented:
  # Instance variables
  cdef hid_t   parent_id
  cdef char    *name

  def _g_new(self, where, name):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_objectID

  def _openUnImplemented(self):
    cdef object shape
    cdef char byteorder[16]  # "non-relevant" fits easily here

    # Get info on dimensions
    shape = H5UIget_info(self.parent_id, self.name, byteorder)
    return (shape, byteorder)
  
  def __dealloc__(self):
    free(<void *>self.name)
