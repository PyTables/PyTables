#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: December 21, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /cvsroot/pytables/pytables/src/hdf5Extension.pyx,v $
#       $Id: hdf5Extension.pyx,v 1.153 2004/12/17 10:27:14 falted Exp $
#
########################################################################

"""Pyrex extension for VLTable object

Classes (type extensions):

    VLTable

Misc variables:

    __version__

"""

__version__ = "$Revision: 1.153 $"


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
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)
  double atof(char *nptr)

cdef extern from "time.h":
  ctypedef int time_t

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

  # To release global interpreter lock (GIL) for threading
  void Py_BEGIN_ALLOW_THREADS()
  void Py_END_ALLOW_THREADS()

  # To access to Memory (Buffer) objects presents in numarray
  object PyBuffer_FromMemory(void *ptr, int size)
  object PyBuffer_FromReadWriteMemory(void *ptr, int size)
  object PyBuffer_New(int size)
  int PyObject_CheckReadBuffer(object)
  int PyObject_AsReadBuffer(object, void **rbuf, int *len)
  int PyObject_AsWriteBuffer(object, void **rbuf, int *len)

# Structs and functions from numarray
cdef extern from "numarray/numarray.h":
  # The numarray initialization funtion
  void import_libnumarray()

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
    
# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()

# CharArray type
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
  object PyArray_FromDims(int nd, int *d, int type)
  object NA_updateDataPtr(object)
  object NA_getPythonScalar(object, long)
  object NA_setFromPythonScalar(object, int, object)
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

  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope) 

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

  herr_t H5ARRAYwrite_records( hid_t loc_id,  char *dset_name,
                               int rank, hsize_t *start, hsize_t *step,
                               hsize_t *count, void *data )

  herr_t H5ARRAYtruncate( hid_t loc_id, char *dset_name,
                          int extdim, hsize_t size)
  
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

  herr_t H5VLARRAYmodify_records( hid_t loc_id, char *dset_name,
                                  hsize_t nrow, int nobjects,
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
  MAX_FIELDS = 256

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
  object getHDF5VersionInfo()
  object createNamesTuple(char *buffer[], int nelements)
  object get_filter_names( hid_t loc_id, char *dset_name)
  object Giterate(hid_t parent_id, hid_t loc_id, char *name)
  object Aiterate(hid_t loc_id)
  H5T_class_t getHDF5ClassID(hid_t loc_id, char *name, H5D_layout_t *layout)
  object H5UIget_info( hid_t loc_id, char *name, char *byteorder)

  # To access to the slice.indices function in 2.2
  int GetIndicesEx(object s, hsize_t length,
                   int *start, int *stop, int *step,
                   int *slicelength)

  object get_attribute_string_sys( hid_t loc_id, char *obj_name,
                                   char *attr_name)



cdef class VLTable:
  # instance variables
  cdef size_t  field_offset[MAX_FIELDS]
  cdef size_t  field_sizes[MAX_FIELDS]
  cdef char    *field_names[MAX_FIELDS]
  cdef hsize_t nfields
  cdef void    *rbuf, *mmrbuf
  cdef hsize_t totalrecords
  cdef hid_t   parent_id, loc_id
  cdef char    *name, *xtitle
  cdef char    *fmt
  cdef int     _open
  cdef char    *complib
  cdef hid_t   dataset_id, space_id, mem_type_id
  cdef object  mmfilew, mmfiler

  def _g_new(self, where, name):
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_objectID
    self._open = 0

  def _g_createVLTable(self):
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

    # Protection against too large row sizes
    # Set to a 512 KB limit (just because banana 640 KB limitation)
    if self.rowsize > 512*1024:
            raise RuntimeError, \
    """Row size too large. Maximum size is 512 Kbytes, and you are asking
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
                        calcBufferSize(self.rowsize, self._v_expectedrows)
    # The next is settable if we have default values
    fill_data = NULL
    nrecords = <hsize_t>PyInt_AsLong(nvar)
    # Create a new group
    ret = H5Gcreate(self.parent_id, self.name, 0)
    if ret < 0:
      raise RuntimeError("Can't create the vltable %s." % self.name)
    self.objectID = oid
    # Because VLTable is an hybrid between Group and Table
    self._v_objectID = oid  

    # Release resources to avoid memory leaks
    for i from  0 <= i < nvar:
      H5Tclose(field_types[i])
    
