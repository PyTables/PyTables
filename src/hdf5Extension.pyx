#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Pyrex interface between several PyTables classes and HDF5 library.

Classes (type extensions):

    File
    AttributeSet
    Group
    Array
    VLArray
    UnImplemented

Functions:

Misc variables:

    __version__
"""

import sys
import os
import warnings
import cPickle

import numarray
from numarray import records, strings, memory

from tables.exceptions import HDF5ExtError
from tables.enum import Enum
from tables.utils import checkFileAccess

from tables.utilsExtension import  \
     enumToHDF5, enumFromHDF5, getTypeEnum, \
     convertTime64, getLeafHDF5Type, isHDF5File, isPyTablesFile, \
     naEnumToNAType, naTypeToNAEnum

from definitions cimport import_libnumarray, NA_getBufferPtrAndSize, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, PyString_AsString, \
     PyString_FromStringAndSize


__version__ = "$Revision$"


#-------------------------------------------------------------------

# C funtions and variable declaration from its headers

# Type size_t is defined in stdlib.h
cdef extern from "stdlib.h":
  #ctypedef int size_t
  # The correct correspondence between size_t and a basic type is *long*
  # instead of int, because they are the same size even for 64-bit platforms
  # F. Altet 2003-01-08
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)
  double atof(char *nptr)

cdef extern from "time.h":
  ctypedef int time_t

# The next has been substituted by equivalents in Python, so that these
# functions could be accessible in Windows systems
# Thanks to Shack Toms for this!
# F. Altet 2004-10-01
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

# Structs and functions from numarray
cdef extern from "numarray/numarray.h":

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

# Structs and types from HDF5
cdef extern from "hdf5.h":
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

# Functions from HDF5
cdef extern from "H5public.h":
  hid_t  H5Fcreate(char *filename, unsigned int flags,
                   hid_t create_plist, hid_t access_plist)

  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id)

  herr_t H5Fclose (hid_t file_id)

  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope )

  hid_t  H5Dopen (hid_t file_id, char *name)

  herr_t H5Dclose (hid_t dset_id)

  herr_t H5Dread (hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf)

  hid_t H5Dget_type (hid_t dset_id)

  hid_t H5Dget_space (hid_t dset_id)

  herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id, hid_t plist_id,
                         void *buf)

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

  herr_t H5Gmove2( hid_t src_loc_id, char *src_name,
                   hid_t dst_loc_id, char *dst_name )

  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum )

  herr_t H5check_version(unsigned majnum, unsigned minnum,
          unsigned relnum )

  herr_t H5Adelete(hid_t loc_id, char *name )

  int H5Aget_num_attrs(hid_t loc_id)

  size_t H5Aget_name(hid_t attr_id, size_t buf_size, char *buf )

  hid_t H5Aopen_idx(hid_t loc_id, unsigned int idx )

  herr_t H5Aread(hid_t attr_id, hid_t mem_type_id, void *buf )

  herr_t H5Aclose(hid_t attr_id)

  herr_t H5Tclose(hid_t type_id)

  herr_t H5Tget_sign(hid_t type_id)

  #hid_t H5Pcreate(H5P_class_t type )  # Wrong in documentation!
  hid_t H5Pcreate(hid_t plist_id)
  herr_t H5Pclose(hid_t plist_id)

  herr_t H5Pset_cache(hid_t plist_id, int mdc_nelmts, int rdcc_nelmts,
                      size_t rdcc_nbytes, double rdcc_w0 )

  herr_t H5Pset_sieve_buf_size(hid_t fapl_id, hsize_t size)

  herr_t H5Pset_fapl_log(hid_t fapl_id, char *logfile,
                         unsigned int flags, size_t buf_size)

  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,
                             hsize_t start[], hsize_t _stride[],
                             hsize_t count[], hsize_t _block[])

  hid_t H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[])

  int H5Sget_simple_extent_ndims(hid_t space_id)

  int H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[], hsize_t maxdims[])

  herr_t H5Sclose(hid_t space_id)

  # Functions for enumeration handling:
  hid_t  H5Tget_super(hid_t type)
  H5T_class_t H5Tget_class(hid_t type_id)
  int    H5Tget_nmembers(hid_t type_id)
  hid_t  H5Tget_member_type(hid_t type_id, int membno)
  char*  H5Tget_member_name(hid_t type_id, int membno)
  herr_t H5Tget_member_value(hid_t type_id, int membno, void *value)

# Functions from HDF5 HL Lite
cdef extern from "H5ATTR.h":

  herr_t H5ATTRget_attribute_ndims( hid_t loc_id, char *attr_name, int *rank )

  herr_t H5ATTRget_attribute_info( hid_t loc_id, char *attr_name,
                                   hsize_t *dims, H5T_class_t *class_id,
                                   size_t *type_size, hid_t *type_id)

  herr_t H5ATTRset_attribute_string( hid_t loc_id, char *attr_name,
                                     char *attr_data )

  herr_t H5ATTRset_attribute_string_CAarray( hid_t loc_id, char *attr_name,
                                             size_t rank, hsize_t *dims,
                                             int itemsize, void *data )

  herr_t H5ATTRset_attribute_numerical_NAarray( hid_t loc_id, char *attr_name,
                                                size_t rank, hsize_t *dims,
                                                hid_t type_id, void *data )

  herr_t H5ATTRget_attribute( hid_t loc_id, char *attr_name,
                              hid_t mem_type_id, void *data )

  herr_t H5ATTRget_attribute_string( hid_t loc_id, char *attr_name,
                                     char **attr_value)

  herr_t H5ATTRget_attribute_string_CAarray( hid_t obj_id, char *attr_name,
                                             char *data )

  herr_t H5ATTR_find_attribute(hid_t loc_id, char *attr_name )


# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h":

  herr_t H5ARRAYmake( hid_t loc_id, char *dset_name, char *class_,
                      char *title, char *flavor, char *obversion,
                      int rank, hsize_t *dims, int extdim,
                      hid_t type_id, hsize_t *dims_chunk, void *fill_data,
                      int complevel, char  *complib, int shuffle,
                      int fletcher32, void *data)

  herr_t H5ARRAYappend_records( hid_t dataset_id, hid_t type_id,
                                int rank, hsize_t *dims_orig,
                                hsize_t *dims_new, int extdim, void *data )

  herr_t H5ARRAYwrite_records( hid_t dataset_id, hid_t type_id,
                               int rank, hsize_t *start, hsize_t *step,
                               hsize_t *count, void *data )

  herr_t H5ARRAYtruncate( hid_t dataset_id, int extdim, hsize_t size)

  herr_t H5ARRAYread( hid_t dataset_id, hid_t type_id,
                      hsize_t start,  hsize_t nrows, hsize_t step,
                      int extdim, void *data )

  herr_t H5ARRAYreadSlice( hid_t dataset_id, hid_t type_id,
                           hsize_t *start, hsize_t *stop,
                           hsize_t *step, void *data )

  herr_t H5ARRAYreadIndex( hid_t dataset_id, hid_t type_id, int notequal,
                           hsize_t *start, hsize_t *stop, hsize_t *step,
                           void *data )

  herr_t H5ARRAYget_ndims( hid_t dataset_id, hid_t type_id, int *rank )

  herr_t H5ARRAYget_info( hid_t dataset_id, hid_t type_id, hsize_t *dims,
                          hsize_t *maxdims, hid_t *super_type_id,
                          H5T_class_t *super_class_id, char *byteorder)

  herr_t H5ARRAYget_chunksize(hid_t dataset_id, int rank, hsize_t *dims_chunk)

# Functions for optimized operations for ARRAY
cdef extern from "H5ARRAY-opt.h":

  herr_t H5ARRAYOread_readSlice( hid_t dataset_id,
                                 hid_t space_id,
                                 hid_t type_id,
                                 hsize_t irow,
                                 hsize_t start,
                                 hsize_t stop,
                                 void *data )

# Functions for VLEN Arrays
cdef extern from "H5VLARRAY.h":

  herr_t H5VLARRAYmake( hid_t loc_id, char *dset_name, char *class_,
                        char *title, char *flavor, char *obversion,
                        int rank, int scalar, hsize_t *dims, hid_t type_id,
                        hsize_t chunk_size, void *fill_data, int complevel,
                        char *complib, int shuffle, int flecther32,
                        void *data)

  herr_t H5VLARRAYappend_records( hid_t dataset_id, hid_t type_id,
                                  int nobjects, hsize_t nrecords,
                                  void *data )

  herr_t H5VLARRAYmodify_records( hid_t dataset_id, hid_t type_id,
                                  hsize_t nrow, int nobjects,
                                  void *data )

  herr_t H5VLARRAYget_ndims( hid_t dataset_id, hid_t type_id, int *rank )

  herr_t H5VLARRAYget_info( hid_t dataset_id, hid_t type_id,
                            hsize_t *nrecords, hsize_t *base_dims,
                            hid_t *base_type_id, char *base_byteorder)


# Function to compute the HDF5 type from a numarray enum type
cdef extern from "arraytypes.h":

  hid_t convArrayType(int fmt, size_t size, char *byteorder)
  size_t getArrayType(hid_t type_id, int *fmt)


# Helper routines
cdef extern from "utils.h":
  herr_t set_cache_size(hid_t file_id, size_t cache_size)
  object Giterate(hid_t parent_id, hid_t loc_id, char *name)
  object Aiterate(hid_t loc_id)
  object H5UIget_info(hid_t loc_id, char *name, char *byteorder)


#-----------------------------------------------------------------------------

# Local variables

# CharArray type
CharType = numarray.records.CharType

NATypeToHDF5AtomicType = {
                            numarray.Int8      : H5T_NATIVE_SCHAR,
                            numarray.Int16     : H5T_NATIVE_SHORT,
                            numarray.Int32     : H5T_NATIVE_INT,
                            numarray.Int64     : H5T_NATIVE_LLONG,

                            numarray.UInt8     : H5T_NATIVE_UCHAR,
                            numarray.UInt16    : H5T_NATIVE_USHORT,
                            numarray.UInt32    : H5T_NATIVE_UINT,
                            numarray.UInt64    : H5T_NATIVE_ULLONG,

                            numarray.Float32   : H5T_NATIVE_FLOAT,
                            numarray.Float64   : H5T_NATIVE_DOUBLE
                        }

# Conversion from numarray int codes to strings
naEnumToNASType = {
  tBool:'Bool',  # Boolean type added
  tInt8:'Int8',    tUInt8:'UInt8',
  tInt16:'Int16',  tUInt16:'UInt16',
  tInt32:'Int32',  tUInt32:'UInt32',
  tInt64:'Int64',  tUInt64:'UInt64',
  tFloat32:'Float32',  tFloat64:'Float64',
  tComplex32:'Complex32',  tComplex64:'Complex64',
  # Special cases:
  ord('a'):'CharType',  # For strings.
  ord('t'):'Time32',  ord('T'):'Time64',  # For times.
  ord('e'):'Enum'}  # For enumerations.


#----------------------------------------------------------------------------

# Initialization code

# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()

#---------------------------------------------------------------------------

# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef hid_t   access_plist
  cdef char    *name


  def __new__(self, char *name, char *mode, char *title,
              object trTable, char *root, object filters,
              size_t metadataCacheSize, size_t nodeCacheSize):
    # Create a new file using default properties
    self.name = name
    self.mode = pymode = mode

    # These fields can be seen from Python.
    self._v_new = None  # this will be computed later
    """Is this file going to be created from scratch?"""
    self._isPTFile = True  # assume a PyTables file by default
    """Does this HDF5 file have a PyTables format?"""

    # After the following check we can be quite sure
    # that the file or directory exists and permissions are right.
    checkFileAccess(name, mode)

    assert pymode in ('r', 'r+', 'a', 'w'), \
           "an invalid mode string ``%s`` " \
           "passed the ``checkFileAccess()`` test; " \
           "please report this to the authors" % pymode

    # Should a new file be created?
    exists = os.path.exists(name)
    self._v_new = new = not (
      pymode in ('r', 'r+') or (pymode == 'a' and exists))

    # After the following check we know that the file exists
    # and it is an HDF5 file, or maybe a PyTables file.
    if not new:
      if not isPyTablesFile(name):
        if isHDF5File(name):
          # HDF5 but not PyTables.
          warnings.warn("file ``%s`` exists and it is an HDF5 file, " \
                        "but it does not have a PyTables format; " \
                        "I will try to do my best to guess what's there " \
                        "using HDF5 metadata" % name)
          self._isPTFile = False
        else:
          # The file is not even HDF5.
          raise IOError(
            "file ``%s`` exists but it is not an HDF5 file" % name)
      # Else the file is an ordinary PyTables file.

    if pymode == 'r':
      # Just a test for disabling metadata caching.
      ## access_plist = H5Pcreate(H5P_FILE_ACCESS)
      ## H5Pset_cache(access_plist, 0, 0, 0, 0.0)
      ## H5Pset_sieve_buf_size(access_plist, 0)
      ##self.file_id = H5Fopen(name, H5F_ACC_RDONLY, access_plist)
      self.file_id = H5Fopen(name, H5F_ACC_RDONLY, H5P_DEFAULT)
    elif pymode == 'r+':
      self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
    elif pymode == 'a':
      if exists:
        # A test for logging.
        ## access_plist = H5Pcreate(H5P_FILE_ACCESS)
        ## H5Pset_cache(access_plist, 0, 0, 0, 0.0)
        ## H5Pset_sieve_buf_size(access_plist, 0)
        ## H5Pset_fapl_log (access_plist, "test.log", H5FD_LOG_LOC_WRITE, 0)
        ## self.file_id = H5Fopen(name, H5F_ACC_RDWR, access_plist)
        self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
      else:
        self.file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
    elif pymode == 'w':
      self.file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)

    # Set the cache size (only for HDF5 1.8.x)
    set_cache_size(self.file_id, metadataCacheSize)

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
        raise HDF5ExtError("Problems closing the file '%s'" % self.name)


# Helper function for quickly fetch an attribute string
def get_attribute_string_or_none(node_id, attr_name):
  """
  Returns a string attribute if it exists in node_id.

  It returns ``None`` in case it don't exists (or there have been problems
  reading it).
  """

  cdef char *attr_value
  cdef object retvalue

  attr_value = NULL
  retvalue = None   # Default value
  if H5ATTR_find_attribute(node_id, attr_name):
    ret = H5ATTRget_attribute_string(node_id, attr_name, &attr_value)
    if ret < 0: return None
    retvalue = attr_value
    # Important to release attr_value, because it has been malloc'ed!
    if attr_value: free(<void *>attr_value)
  return retvalue


cdef class AttributeSet:
  cdef hid_t   dataset_id
  cdef char    *name

  def _g_new(self, node):
    # Initialize the C attributes of Node object
    self.name =  PyString_AsString(node._v_hdf5name)
    # The dataset id of the node
    self.dataset_id = node._v_objectID

  def __g_listAttr(self):
    "Return a tuple with the attribute list"
    a = Aiterate(self.dataset_id)
    #Py_DECREF(a)  # makes a core dump
    return a

  # The next is a re-implementation of Aiterate but in pure Pyrex
  # However, it seems to leak exactly the same as its C counterpart
  # This should be further revised :-/
  # F. Altet 2005/09/30
  def _g_listAttr(self):
    "Return a tuple with the attribute list"
    cdef int nattrs, i
    cdef hid_t attr_id
    cdef char attr_name[256]

    lattrs = []
    nattrs = H5Aget_num_attrs(self.dataset_id)
    for i from 0 <= i < nattrs:
      attr_id = H5Aopen_idx(self.dataset_id, i)
      H5Aget_name(attr_id, 256, attr_name)
      H5Aclose(attr_id)
      lattrs.append(attr_name)
    return lattrs

  # Get a system attribute (they should be only strings)
  def _g_getSysAttr(self, char *attrname):
    return get_attribute_string_or_none(self.dataset_id, attrname)

  # Set a system attribute (they should be only strings)
  def _g_setAttrStr(self, char *attrname, char *attrvalue):
    cdef int ret

    ret = H5ATTRset_attribute_string(self.dataset_id, attrname, attrvalue)
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                         (attrname, self._v_node))

  # Get attributes
  def _g_getAttr(self, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef char  *attrvaluechar
    cdef short  attrvalueshort
    cdef long  attrvaluelong
    cdef float  attrvaluefloat
    cdef double  attrvaluedouble
    cdef long long attrvaluelonglong
    cdef object retvalue
    cdef hid_t mem_type
    cdef int rank
    cdef int ret, i
    cdef int enumtype
    cdef void *rbuf
    cdef long buflen
    cdef hid_t type_id
    cdef H5T_sign_t sign #H5T_SGN_ERROR (-1), H5T_SGN_NONE (0), H5T_SGN_2 (1)
    cdef char *dsetname

    dsetname = self.name

    # Check if attribute exists
    if H5ATTR_find_attribute(self.dataset_id, attrname) <= 0:
      # If the attribute does not exists, return None
      # and do not even warn the user
      return None

    ret = H5ATTRget_attribute_ndims(self.dataset_id, attrname, &rank )
    if ret < 0:
      raise HDF5ExtError("Can't get ndims on attribute %s in node %s." %
                             (attrname, dsetname))

    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    else:
      dims = NULL

    ret = H5ATTRget_attribute_info(self.dataset_id, attrname, dims,
                                   &class_id, &type_size, &type_id)
    if ret < 0:
      raise HDF5ExtError("Can't get info on attribute %s in node %s." %
                               (attrname, dsetname))
    # Get the attribute type
    type_size = getArrayType(type_id, &enumtype)
    # type_size is of type size_t (unsigned long), so cast it to a long first
    if <long>type_size < 0:
      # This class is not supported. Instead of raising a TypeError,
      # return a string explaining the problem. This will allow to continue
      # browsing native HDF5 files, while informing the user about the problem.
      #raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id
      H5Tclose(type_id)    # Release resources
      return "***Attribute error: HDF5 class %d not supported. Sorry!***" % class_id
    sign = H5Tget_sign(type_id)
    H5Tclose(type_id)    # Release resources

    # Get the array shape
    shape = []
    for i from 0 <= i < rank:
      # The <int> cast avoids returning a Long integer
      shape.append(<int>dims[i])
    shape = tuple(shape)

    retvalue = None
    dtype = naEnumToNAType[enumtype]
    if class_id in (H5T_INTEGER, H5T_FLOAT):
      retvalue = numarray.array(None, type=dtype, shape=shape)
    elif class_id == H5T_STRING:
      retvalue = strings.array(None, itemsize=type_size, shape=shape)

    if retvalue is not None:
      # Get the pointer to the buffer data area
      buflen = NA_getBufferPtrAndSize(retvalue._data, 1, &rbuf)

    if class_id == H5T_INTEGER:
      if sign == H5T_SGN_2:
        if type_size == 1:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_SCHAR, rbuf)
        if type_size == 2:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_SHORT, rbuf)
        if type_size == 4:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_INT, rbuf)
        if type_size == 8:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_LLONG, rbuf)
      elif sign == H5T_SGN_NONE:
        if type_size == 1:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_UCHAR, rbuf)
        if type_size == 2:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_USHORT, rbuf)
        if type_size == 4:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_UINT, rbuf)
        if type_size == 8:
          ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                    H5T_NATIVE_ULLONG, rbuf)
      else:
        warnings.warn("""\
Type of attribute '%s' in node '%s' is not supported. Sorry about that!"""
                      % (attrname, dsetname))
        return None

    elif class_id == H5T_FLOAT:
      if type_size == 4:
        ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                  H5T_NATIVE_FLOAT, rbuf)
      if type_size == 8:
        ret = H5ATTRget_attribute(self.dataset_id, attrname,
                                  H5T_NATIVE_DOUBLE, rbuf)

    elif class_id == H5T_STRING:
      # Scalar string attributes are returned as Python strings, while
      # multi-dimensional ones are returned as character arrays.
      if rank == 0:
        ret = H5ATTRget_attribute_string(self.dataset_id, attrname, &attrvaluechar)
        retvalue = attrvaluechar
        # Important to release attrvaluechar, because it has been malloc'ed!
        if attrvaluechar: free(<void *>attrvaluechar)
      else:
        ret = H5ATTRget_attribute_string_CAarray(self.dataset_id, attrname,
                                                 <char *> rbuf)

    else:
      warnings.warn("""\
Type of attribute '%s' in node '%s' is not supported. Sorry about that!"""
                    % (attrname, dsetname))
      return None

    if dims:
      free(<void *> dims)

    # Check the return value of H5ATTRget_attribute_* call
    if ret < 0:
      raise HDF5ExtError("Attribute %s exists in node %s, but can't get it."\
                         % (attrname, dsetname))

    node = self._v_node

    # Check for multimensional attributes (if file.format_version > "1.4")
    if hasattr(node._v_file, "format_version"):
      format_version = node._v_file.format_version
    else:
      format_version = None

    if format_version is not None:
      if format_version < "1.4":
        if rank > 0 and shape[0] > 1:
          warnings.warn("""\
Multi-dimensional attribute '%s' in node '%s' is not supported in file format version %s.
Loaded anyway."""
                        % (attrname, dsetname, format_version))

        elif class_id == H5T_INTEGER:
          # return as 'int' built-in type
          if shape == ():
            retvalue = int(retvalue[()])
          else:
            retvalue = int(retvalue[0])

        elif class_id == H5T_FLOAT:
          # return as 'float' built-in type
          if shape == ():
            retvalue = float(retvalue[()])
          else:
            retvalue = float(retvalue[0])

        # Just if one wants to convert a scalar into a Python scalar
        # instead of an numarray scalar. But I think it is better and more
        # consistent a numarray scalar.
#       elif format_version >= "1.4" and rank == 0:
#         if class_id == H5T_INTEGER or class_id == H5T_FLOAT:
#           retvalue = retvalue[()]

    return retvalue

  def _g_setAttr(self, char *name, object value):
    cdef int ret
    cdef int valint
    cdef double valdouble
    cdef hid_t type_id
    cdef size_t rank
    cdef hsize_t *dims
    cdef void* data
    cdef long buflen
    cdef int i
    cdef int itemsize

    node = self._v_node

    # Check for scalar attributes (if file.format_version < "1.4")
    if hasattr(node._v_file, "format_version"):
      format_version = node._v_file.format_version
    else:
      format_version = None

    if format_version is not None and format_version < "1.4" and \
       (isinstance(value, int) or isinstance(value, float)):
      value = numarray.asarray(value)

    ret = 0
    # Append this attribute on disk
    if isinstance(value, str):
      ret = H5ATTRset_attribute_string(self.dataset_id, name, value)
    elif isinstance(value, strings.CharArray):
      itemsize = value.itemsize()
      if itemsize > 0:
        rank = value.rank
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
        for i from 0 <= i < rank:
          dims[i] = value.shape[i]
        buflen = NA_getBufferPtrAndSize(value._data, 1, &data)
        ret = H5ATTRset_attribute_string_CAarray(
          self.dataset_id, name, rank, dims, itemsize, data)
        free(<void *>dims)
      else:
        # HDF5 does not support strings with itemsize = 0. This kind of
        # strings will appear only in numarray strings.
        # Convert this object to a null-terminated string
        # (binary pickles are not supported at this moment)
        pickledvalue = cPickle.dumps(value, 0)
        self._g_setAttrStr(name, pickledvalue)
    elif isinstance(value, numarray.numarraycore.NumArray):
      if value.type() in NATypeToHDF5AtomicType.keys():
        type_id = NATypeToHDF5AtomicType[value.type()]

        rank = value.rank
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))

        for i from 0 <= i < rank:
          dims[i] = value.shape[i]

        buflen = NA_getBufferPtrAndSize(value._data, 1, &data)
        ret = H5ATTRset_attribute_numerical_NAarray(self.dataset_id, name,
                                                    rank, dims, type_id, data)
        free(<void *>dims)
      else:   # One should add complex support for numarrays
        pickledvalue = cPickle.dumps(value, 0)
        self._g_setAttrStr(name, pickledvalue)
    else:
      # Convert this object to a null-terminated string
      # (binary pickles are not supported at this moment)
      pickledvalue = cPickle.dumps(value, 0)
      self._g_setAttrStr(name, pickledvalue)

    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                         (name, node))

  def _g_remove(self, attrname):
    cdef int ret
    ret = H5Adelete(self.dataset_id, attrname)
    if ret < 0:
      raise HDF5ExtError("Attribute '%s' exists in node '%s', but cannot be deleted." \
                         % (attrname, self.name))

  def __dealloc__(self):
    self.dataset_id = 0


cdef class Node:
  # Instance variables declared in .pxd

  def _g_new(self, where, name, init):
    self.name = strdup(name)
    """The name of this node in its parent group."""
    self.parent_id = where._v_objectID
    """The identifier of the parent group."""

  def _g_delete(self):
    cdef int ret

    # Delete this node
    ret = H5Gunlink(self.parent_id, self.name)
    if ret < 0:
      raise HDF5ExtError("problems deleting the node ``%s``" % self.name)
    return ret

  def __dealloc__(self):
    free(<void *>self.name)


cdef class Group(Node):
  cdef hid_t   group_id

  def _g_create(self):
    cdef hid_t ret

    # Create a new group
    ret = H5Gcreate(self.parent_id, self.name, 0)
    if ret < 0:
      raise HDF5ExtError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_open(self):
    cdef hid_t ret

    ret = H5Gopen(self.parent_id, self.name)
    if ret < 0:
      raise HDF5ExtError("Can't open the group: '%s'." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_listGroup(self):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(self.parent_id, self._v_objectID, self.name)

  def _g_getGChildAttr(self, char *group_name, char *attr_name):
    """
    Return an attribute of a child `Group`.

    If the attribute does not exist, ``None`` is returned.
    """

    cdef hid_t gchild_id
    cdef object retvalue

    # Open the group
    retvalue = None  # Default value
    gchild_id = H5Gopen(self.group_id, group_name)
    retvalue = get_attribute_string_or_none(gchild_id, attr_name)
    # Close child group
    H5Gclose(gchild_id)

    return retvalue

  def _g_getLChildAttr(self, char *leaf_name, char *attr_name):
    """
    Return an attribute of a child `Leaf`.

    If the attribute does not exist, ``None`` is returned.
    """

    cdef hid_t leaf_id
    cdef object retvalue

    # Open the dataset
    leaf_id = H5Dopen(self.group_id, leaf_name)
    retvalue = get_attribute_string_or_none(leaf_id, attr_name)
    # Close the dataset
    H5Dclose(leaf_id)
    return retvalue

  def _g_flushGroup(self):
    # Close the group
    H5Fflush(self.group_id, H5F_SCOPE_GLOBAL)

  def _g_closeGroup(self):
    cdef int ret

    ret = H5Gclose(self.group_id)
    if ret < 0:
      raise HDF5ExtError("Problems closing the Group %s" % self.name )
    self.group_id = 0  # indicate that this group is closed

  def _g_moveNode(self, hid_t oldparent, char *oldname,
                  hid_t newparent, char *newname,
                  char *oldpathname, char *newpathname):
    cdef int ret

    ret = H5Gmove2(oldparent, oldname, newparent, newname)
    if ret < 0:
      raise HDF5ExtError("Problems moving the node %s to %s" %
                         (oldpathname, newpathname) )
    return ret



cdef class Leaf(Node):
  # Instance variables declared in .pxd

  def _g_new(self, where, name, init):
    if init:
      # Put this info to 0 just when the class is initialized
      self.dataset_id = -1
      self.type_id = -1
      self.base_type_id = -1
    super(Leaf, self)._g_new(where, name, init)

  def _g_flush(self):
    # Flush the dataset (in fact, the entire buffers in file!)
    if self.dataset_id >= 0:
        H5Fflush(self.dataset_id, H5F_SCOPE_GLOBAL)

  def _g_close(self):
    # Close dataset in HDF5 space
    # Release resources
    if self.type_id >= 0:
      H5Tclose(self.type_id)
    if self.base_type_id >= 0:
      H5Tclose(self.base_type_id)
    if self.dataset_id >= 0:
      H5Dclose(self.dataset_id)


cdef class Array(Leaf):
  # Instance variables declared in .pxd counterpart

  def _createArray(self, object naarr, char *title):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef long buflen
    cdef int enumtype, itemsize, offset
    cdef char *byteorder
    cdef char *flavor, *complib, *version, *class_
    cdef object type

    if isinstance(naarr, strings.CharArray):
      type = CharType
      enumtype = naTypeToNAEnum[CharType]
    else:
      type = naarr._type
      try:
        enumtype = naTypeToNAEnum[naarr._type]
      except KeyError:
        raise TypeError, \
      """Type class '%s' not supported right now. Sorry about that.
      """ % repr(naarr._type)

    # String types different from Numarray types are still not allowed
    # in regular Arrays.
    stype = str(type)

    itemsize = naarr._itemsize
    byteorder = PyString_AsString(self.byteorder)
    self.type_id = convArrayType(enumtype, itemsize, byteorder)
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
    self.rank = len(self.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < self.rank:
      self.dims[i] = naarr.shape[i]

    # Save the array
    flavor = PyString_AsString(self.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    self.dataset_id = H5ARRAYmake(self.parent_id, self.name, class_, title,
                                  flavor, version, self.rank, self.dims,
                                  self.extdim, self.type_id, NULL, NULL,
                                  self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    return (self.dataset_id, type, stype)

  def _createEArray(self, char *title):
    cdef int i, enumtype
    cdef herr_t ret
    cdef void *rbuf
    cdef char *byteorder
    cdef char *flavor, *complib, *version, *class_
    cdef void *fill_value
    cdef int itemsize

    atom = self.atom
    itemsize = atom.itemsize

    try:
      # Since Time columns have no Numarray type of their own,
      # a special case is made for them.
      stype = atom.stype
      if stype == 'Time32':
        enumtype = ord('t')
      elif stype == 'Time64':
        enumtype = ord('T')
      else:
        enumtype = naTypeToNAEnum[self.type]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported right now. Sorry about that.
            """ % repr(self.type)

    if stype == 'Enum':
      self.type_id = enumToHDF5(atom, self.byteorder)
    else:
      byteorder = PyString_AsString(self.byteorder)
      self.type_id = convArrayType(enumtype, itemsize, byteorder)
      if self.type_id < 0:
        raise TypeError, \
          """type '%s' is not supported right now. Sorry about that.""" \
      % self.type

    fill_value = NULL

    self.rank = len(self.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if self._v_chunksize:
      self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      self.dims[i] = self.shape[i]
      if self._v_chunksize:
        self.dims_chunk[i] = self._v_chunksize[i]

    rbuf = NULL   # The data pointer. We don't have data to save initially
    # Manually convert some string values that can't be done automatically
    flavor = PyString_AsString(atom.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    fill_value = <void *>malloc(<size_t> itemsize)
    if(fill_value):
      for i from  0 <= i < itemsize:
        (<char *>fill_value)[i] = 0
    else:
      raise HDF5ExtError("Unable to allocate memory for fill_value.")
    # Create the EArray
    self.dataset_id = H5ARRAYmake(self.parent_id, self.name, class_, title,
                                  flavor, version, self.rank, self.dims,
                                  self.extdim, self.type_id, self.dims_chunk,
                                  fill_value, self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)
    # Release resources
    if(fill_value): free(fill_value)

    return self.dataset_id


  def _loadEnum(self):
    """_loadEnum() -> (Enum, naType)
    Load enumerated type associated with this array.

    This method loads the HDF5 enumerated type associated with this
    array.  It returns an `Enum` instance built from that, and the
    Numarray type used to encode it.
    """

    cdef hid_t enumId

    enumId = getTypeEnum(self.type_id)

    # Get the Enum and Numarray types and close the HDF5 type.
    try:
      return enumFromHDF5(enumId)
    finally:
      # (Yes, the ``finally`` clause *is* executed.)
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")


  def _openArray(self):
    cdef object shape
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef H5T_sign_t sign
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i, enumtype
    cdef int extdim
    cdef char *flavor
    cdef hid_t base_type_id
    cdef herr_t ret

    # Open the dataset (and keep it open)
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems opening dataset %s" % self.name)
    # Get the datatype handle (and keep it open)
    self.type_id = H5Dget_type(self.dataset_id)
    if self.type_id < 0:
      raise HDF5ExtError("Problems getting type id for dataset %s" % self.name)

    # Get the rank for this array object
    if H5ARRAYget_ndims(self.dataset_id, self.type_id, &self.rank) < 0:
      raise HDF5ExtError("Problems getting ndims!")
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    self.maxdims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type (of base class)
    ret = H5ARRAYget_info(self.dataset_id, self.type_id,
                          self.dims, self.maxdims,
                          &base_type_id, &class_id, byteorder)
    if ret < 0:
      raise HDF5ExtError("Unable to get array info.")

    self.extdim = -1  # default is non-chunked Array
    # Get the extendeable dimension (if any)
    for i from 0 <= i < self.rank:
      if self.maxdims[i] == -1:
        self.extdim = i
        break

    flavor = "numarray"   # Default value
    if self._v_file._isPTFile:
      H5ATTRget_attribute_string(self.dataset_id, "FLAVOR", &flavor)
    self.flavor = flavor  # Gives class visibility to flavor

    # Allocate space for the dimension chunking info
    self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if ( (H5ARRAYget_chunksize(self.dataset_id, self.rank,
                               self.dims_chunk)) < 0):
      #H5ARRAYget_chunksize frees dims_chunk
      self.dims_chunk = NULL
      if self.extdim >= 0 or self.__class__.__name__ == 'CArray':
        raise HDF5ExtError, "Problems getting the chunksizes!"
    # Get the array type
    type_size = getArrayType(base_type_id, &enumtype)
    if type_size < 0:
      raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id

    H5Tclose(base_type_id)    # Release resources

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
      shape.append(self.dims[i])
      if self.dims_chunk:
        chunksizes.append(<int>self.dims_chunk[i])
    shape = tuple(shape)
    chunksizes = tuple(chunksizes)

    type = naEnumToNAType.get(enumtype, None)
    return (self.dataset_id, type, naEnumToNASType[enumtype],
            shape, type_size, byteorder, chunksizes)

  def _convertTypes(self, object naarr, int sense):
    """Converts Time64 elements in 'naarr' between Numarray and HDF5 formats.

    Numarray to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to Numarray conversion is performed.
    The conversion is done in place, i.e. 'naarr' is modified.
    """

    # This should be generalised to support other type conversions.
    if self.stype == 'Time64':
      convertTime64(naarr, len(naarr), sense)

  def _append(self, object naarr):
    cdef int ret
    cdef hsize_t *dims_arr
    cdef void *rbuf
    cdef long offset
    cdef int buflen
    cdef object shape
    cdef int extdim

    # Allocate space for the dimension axis info
    dims_arr = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < self.rank:
        dims_arr[i] = naarr.shape[i]

    # Get the pointer to the buffer data area
    # Both methods do the same
    buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)
    # Correct the start of the buffer with the _byteoffset
    offset = naarr._byteoffset
    rbuf = <void *>(<char *>rbuf + offset)

    # Convert some Numarray types to HDF5 before storing.
    self._convertTypes(naarr, 0)

    # Append the records:
    extdim = self.extdim
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYappend_records(self.dataset_id, self.type_id, self.rank,
                                self.dims, dims_arr, extdim, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems appending the elements")

    free(dims_arr)
    # Update the new dimensionality
    shape = list(self.shape)
    shape[self.extdim] = self.dims[self.extdim]
    self.shape = tuple(shape)
    self.nrows = self.dims[self.extdim]

  def _modify(self, object startl, object stepl, object countl,
              object naarr):
    cdef int ret
    cdef void *rbuf, *temp
    cdef hsize_t *start, *step, *count
    cdef long buflen, offset

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)
    # Correct the start of the buffer with the _byteoffset
    offset = naarr._byteoffset
    rbuf = <void *>(<char *>rbuf + offset)

    # Get the start, step and count values
    buflen = NA_getBufferPtrAndSize(startl._data, 1, <void **>&start)
    buflen = NA_getBufferPtrAndSize(stepl._data, 1, <void **>&step)
    buflen = NA_getBufferPtrAndSize(countl._data, 1, <void **>&count)

    # Convert some Numarray types to HDF5 before storing.
    self._convertTypes(naarr, 0)

    # Modify the elements:
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYwrite_records(self.dataset_id, self.type_id, self.rank,
                               start, step, count, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Internal error modifying the elements (H5ARRAYwrite_records returned errorcode -%i)"%(-ret))

  def _truncateArray(self, hsize_t size):
    cdef hsize_t extdim
    cdef hsize_t ret

    extdim = self.extdim
    if size >= self.shape[extdim]:
      return self.shape[extdim]

    ret = H5ARRAYtruncate(self.dataset_id, extdim, size)
    if ret < 0:
      raise HDF5ExtError("Problems truncating the EArray node.")

    # Update the new dimensionality
    shape = list(self.shape)
    shape[self.extdim] = size
    self.shape = tuple(shape)
    self.nrows = size

    return

  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step,
                 object naarr):
    cdef herr_t ret
    cdef void *rbuf
    cdef long buflen
    cdef hsize_t nrows
    cdef int extdim

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)

    # Number of rows to read
    nrows = ((stop - start - 1) / step) + 1  # (stop-start)/step  do not work
    if hasattr(self, "extdim"):
      extdim = self.extdim
    else:
      exdim = -1
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYread(self.dataset_id, self.type_id, start, nrows, step,
                      extdim, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Convert some HDF5 types to Numarray after reading.
    self._convertTypes(naarr, 1)

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
    ret = H5ARRAYreadSlice(self.dataset_id, self.type_id,
                           <hsize_t *>startlb, <hsize_t *>stoplb,
                           <hsize_t *>steplb, rbuflb)
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Convert some HDF5 types to Numarray after reading.
    self._convertTypes(bufferl, 1)

    return

  def __dealloc__(self):
    #print "Destroying object Array in Extension"
    free(<void *>self.dims)
    if self.maxdims:
      free(<void *>self.maxdims)
    if self.dims_chunk:
      free(self.dims_chunk)


cdef class VLArray(Leaf):
  # Instance variables
  cdef int     rank
  cdef hsize_t *dims
  cdef hsize_t nrecords
  cdef int     scalar


  def _createArray(self, char *title):
    cdef int i, enumtype
    cdef herr_t ret
    cdef void *rbuf
    cdef char *byteorder
    cdef char *flavor, *complib, *version, *class_

    atom = self.atom
    type_ = atom.type
    stype = atom.stype
    try:
      # Since Time columns have no Numarray type of their own,
      # a special case is made for them.
      if stype == 'Time32':
        enumtype = ord('t')
      elif stype == 'Time64':
        enumtype = ord('T')
      else:
        enumtype = naTypeToNAEnum[type_]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported right now. Sorry about that.
            """ % repr(type_)

    if stype == 'Enum':
      self.base_type_id = enumToHDF5(atom, self.byteorder)
    else:
      byteorder = PyString_AsString(self.byteorder)
      # Get the HDF5 type id
      self.base_type_id = convArrayType(enumtype, atom.itemsize, byteorder)
      if self.base_type_id < 0:
        raise TypeError, \
          """type '%s' is not supported right now. Sorry about that.""" \
      % type_

    # Allocate space for the dimension axis info
    if isinstance(atom.shape, int):
      self.rank = 1
      self.scalar = 1
    else:
      self.rank = len(atom.shape)
      self.scalar = 0

    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      if isinstance(atom.shape, int):
        self.dims[i] = atom.shape
      else:
        self.dims[i] = atom.shape[i]

    rbuf = NULL   # We don't have data to save initially

    # Manually convert some string values that can't be done automatically
    flavor = PyString_AsString(atom.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Create the vlarray
    self.dataset_id = H5VLARRAYmake(self.parent_id, self.name, class_, title,
                                    flavor, version, self.rank, self.scalar,
                                    self.dims, self.base_type_id,
                                    self._v_chunksize, rbuf,
                                    self.filters.complevel, complib,
                                    self.filters.shuffle,
                                    self.filters.fletcher32,
                                    rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the VLArray.")
    self.nrecords = 0  # Initialize the number of records saved

    # Get the datatype handle (and keep it open)
    self.type_id = H5Dget_type(self.dataset_id)

    return self.dataset_id


  def _loadEnum(self):
    """_loadEnum() -> (Enum, naType)
    Load enumerated type associated with this array.

    This method loads the HDF5 enumerated type associated with this
    array.  It returns an `Enum` instance built from that, and the
    Numarray type used to encode it.
    """

    cdef hid_t typeId, rowTypeId, enumId

    # Get the enumerated type.
    enumId = getTypeEnum(self.base_type_id)

    # Get the Enum and Numarray types and close the HDF5 type.
    try:
      return enumFromHDF5(enumId)
    finally:
      # (Yes, the ``finally`` clause *is* executed.)
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")


  def _openArray(self):
    cdef object shape
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i, enumtype
    cdef herr_t ret
    cdef hsize_t nrecords[1]
    cdef char *flavor

    # Open the dataset (and keep it open)
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems opening dataset %s" % self.name)
    # Get the datatype handle (and keep it open)
    self.type_id = H5Dget_type(self.dataset_id)
    if self.type_id < 0:
      raise HDF5ExtError("Problems getting type id for dataset %s" % self.name)

    # Get the rank for the atom in the array object
    ret = H5VLARRAYget_ndims(self.dataset_id, self.type_id, &self.rank)
    # Allocate space for the dimension axis info
    if self.rank:
      self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    else:
      self.dims = NULL;
    # Get info on dimensions, class and type (of base class)
    H5VLARRAYget_info(self.dataset_id, self.type_id, nrecords,
                      self.dims, &self.base_type_id, byteorder)
    flavor = "numarray"  # Default value
    if self._v_file._isPTFile:
      H5ATTRget_attribute_string(self.dataset_id, "FLAVOR", &flavor)
    self.flavor = flavor  # Gives class visibility to flavor
    self.byteorder = byteorder  # Gives class visibility to byteorder

    # Get the array type
    self._basesize = getArrayType(self.base_type_id, &enumtype)
    if self._basesize < 0:
      raise TypeError, "The HDF5 class of object does not seem VLEN. Sorry!"

    # Get the type of the atomic type
    self._atomictype = naEnumToNAType.get(enumtype, None)
    self._atomicstype = naEnumToNASType[enumtype]
    # Get the size and shape of the atomic type
    self._atomicsize = self._basesize
    if self.rank:
      shape = []
      for i from 0 <= i < self.rank:
        shape.append(self.dims[i])
        self._atomicsize = self._atomicsize * self.dims[i]
      shape = tuple(shape)
    else:
      # rank zero means a scalar
      shape = 1

    self._atomicshape = shape
    self.nrecords = nrecords[0]  # Initialize the number of records saved
    return (self.dataset_id, nrecords[0])

  def _convertTypes(self, object naarr, int sense):
    """Converts Time64 elements in 'naarr' between Numarray and HDF5 formats.

    Numarray to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to Numarray conversion is performed.
    The conversion is done in place, i.e. 'naarr' is modified.
    """

    # This should be generalised to support other type conversions.
    if self._atomicstype == 'Time64':
      convertTime64(naarr, len(naarr), sense)

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

      # Convert some Numarray types to HDF5 before storing.
      self._convertTypes(naarr, 0)
    else:
      rbuf = NULL

    # Append the records:
    Py_BEGIN_ALLOW_THREADS
    ret = H5VLARRAYappend_records(self.dataset_id, self.type_id,
                                  nobjects, self.nrecords, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems appending the records.")

    self.nrecords = self.nrecords + 1

  def _modify(self, hsize_t nrow, object naarr, int nobjects):
    cdef int ret
    cdef void *rbuf
    cdef long buflen, offset

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(naarr._data, 1, &rbuf)
    # Correct the start of the buffer with the _byteoffset
    offset = naarr._byteoffset
    rbuf = <void *>(<char *>rbuf + offset)

    if nobjects:
      # Convert some Numarray types to HDF5 before storing.
      self._convertTypes(naarr, 0)

    # Append the records:
    Py_BEGIN_ALLOW_THREADS
    ret = H5VLARRAYmodify_records(self.dataset_id, self.type_id,
                                  nrow, nobjects, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems modifying the record.")

    return nobjects

  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step):
    cdef herr_t ret
    cdef hvl_t *rdata
    cdef size_t vllen
    cdef object rbuf, naarr, shape, datalist
    cdef int i
    cdef hsize_t nrows
    cdef hid_t space_id
    cdef hid_t mem_space_id

    # Compute the number of rows to read
    nrows = ((stop - start - 1) / step) + 1  # (stop-start)/step  do not work
    if start + nrows > self.nrows:
      raise HDF5ExtError("Asking for a range of rows exceeding the available ones!.")

    # Now, read the chunk of rows
    Py_BEGIN_ALLOW_THREADS
    # Allocate the necessary memory for keeping the row handlers
    rdata = <hvl_t *>malloc(<size_t>nrows*sizeof(hvl_t))
    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle */
    mem_space_id = H5Screate_simple(1, &nrows, NULL)
    # Select the data to be read
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start, &step, &nrows, NULL)
    # Do the actual read
    ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                  H5P_DEFAULT, rdata)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("VLArray._readArray: Problems reading the array data.")

    datalist = []
    for i from 0 <= i < nrows:
      # Number of atoms in row
      vllen = rdata[i].len
      # Get the pointer to the buffer data area
      if vllen > 0:
        # Create a buffer to keep this info. It is important to do a
        # copy, because we will dispose the buffer memory later on by
        # calling the H5Dvlen_reclaim. PyString_FromStringAndSize do this.
        rbuf = PyString_FromStringAndSize(<char *>rdata[i].p,
                                          vllen*self._atomicsize)
      else:
        # Case where there is info with zero lentgh
        rbuf = None
      # Compute the shape for the read array
      if (isinstance(self._atomicshape, tuple)):
        shape = list(self._atomicshape)
        shape.insert(0, vllen)  # put the length at the beginning of the shape
      elif self._atomicshape > 1:
        shape = (vllen, self._atomicshape)
      else:
        # Case of scalars (self._atomicshape == 1)
        shape = (vllen,)
      if str(self._atomictype) == "CharType":
        naarr = strings.array(rbuf, itemsize=self._basesize, shape=shape)
      else:
        naarr = numarray.array(rbuf, type=self._atomictype, shape=shape)
        # Set the same byteorder than on-disk
        naarr._byteorder = self.byteorder
      # Convert some HDF5 types to Numarray after reading.
      self._convertTypes(naarr, 1)
      # Append this array to the output list
      datalist.append(naarr)

    # Release resources
    # Reclaim all the (nested) VL data
    ret = H5Dvlen_reclaim(self.type_id, mem_space_id, H5P_DEFAULT, rdata)
    if ret < 0:
      raise HDF5ExtError("VLArray._readArray: error freeing the data buffer.")
    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)
    # Free the amount of row pointers to VL row data
    free(rdata)

    return datalist

  def __dealloc__(self):
    if self.dims:
      free(<void *>self.dims)


cdef class UnImplemented(Leaf):

  def _openUnImplemented(self):
    cdef object shape
    cdef char byteorder[16]  # "non-relevant" fits easily here

    # Get info on dimensions
    shape = H5UIget_info(self.parent_id, self.name, byteorder)
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    return (shape, byteorder, self.dataset_id)

  def _g_close(self):
    H5Dclose(self.dataset_id)

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
## End:
