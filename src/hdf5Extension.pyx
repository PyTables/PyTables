#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/hdf5Extension.pyx,v $
#       $Id: hdf5Extension.pyx,v 1.49 2003/06/04 11:14:56 falted Exp $
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

__version__ = "$Revision: 1.49 $"


import sys, os
import numarray as num
import ndarray
import chararray
import recarray2 as recarray

# For defining the long long type
cdef extern from "type-longlong.h":
  cdef enum:
    LL_TYPE
    MY_MSC

# C funtions and variable declaration from its headers

# Type size_t is defined in stdlib.h
cdef extern from "stdlib.h":
  ctypedef int size_t
  void *malloc(size_t size)
  void free(void *ptr)

# Funtions for printng in C
cdef extern from "stdio.h":
  int sprintf(char *str,  char *format, ...)
  int snprintf(char *str, size_t size, char *format, ...)

cdef extern from "string.h":
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)

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
  object PyInt_FromLong(long ival)
  
  # To access strings
  object PyString_FromStringAndSize(char *s, int len)
  char *PyString_AsString(object string)
  object PyString_FromString(char *)

  ctypedef class PyStringObject [type PyString_Type]:
    cdef char *ob_sval
    cdef int  ob_size

  ctypedef class PyTupleObject [type PyTuple_Type]:
    cdef object ob_item
    cdef int    ob_size

  # To access to Memory (Buffer) objects presents in numarray
  object PyBuffer_FromMemory(void *ptr, int size)
  object PyBuffer_New(int size)
  int PyObject_AsReadBuffer(object, void **rbuf, int *len)
  int PyObject_AsWriteBuffer(object, void **rbuf, int *len)

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
  
  struct PyArray_Descr:
     int type_num, elsize
     char type
        
  ctypedef class PyArrayObject [type PyArray_Type]:
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
  void import_array()
    
# The Numeric API requires this function to be called before
# using any Numeric facilities in an extension module.
import_array()

# CharArray type
CharType = recarray.CharType

# Conversion tables from/to classes to the numarray enum types
toenum = {num.Int8:tInt8,       num.UInt8:tUInt8,
          num.Int16:tInt16,     num.UInt16:tUInt16,
          num.Int32:tInt32,     num.UInt32:tUInt32,
          num.Int64:tInt64,     num.UInt64:tUInt64,
          num.Float32:tFloat32, num.Float64:tFloat64,
          CharType:97   # ascii(97) --> 'a' # Special case (to be corrected)
          }

toclass = {tInt8:num.Int8,       tUInt8:num.UInt8,
           tInt16:num.Int16,     tUInt16:num.UInt16,
           tInt32:num.Int32,     tUInt32:num.UInt32,
           tInt64:num.Int64,     tUInt64:num.UInt64,
           tFloat32:num.Float32, tFloat64:num.Float64,
           97:CharType   # ascii(97) --> 'a' # Special case (to be corrected)
          }

# Define the CharType code as a constant
cdef enum:
  CHARTYPE = 97

# Functions from numarray API
cdef extern from "numarray/libnumarray.h":
  PyArrayObject NA_InputArray (object, NumarrayType, int)
  PyArrayObject NA_OutputArray (object, NumarrayType, int)
  PyArrayObject NA_IoArray (object, NumarrayType, int)
  PyArrayObject PyArray_FromDims(int nd, int *d, int type)
  PyArrayObject NA_Empty(int nd, int *d, NumarrayType type)
  object        NA_updateDataPtr(object)
  object        NA_getPythonScalar(object, long)
  object        NA_setFromPythonScalar  (object, int, object)

  object PyArray_ContiguousFromObject(object op, int type,
                                      int min_dim, int max_dim)
# Functions from HDF5
cdef extern from "hdf5.h":
  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT
  int H5P_DEFAULT, H5S_ALL
  int H5T_NATIVE_CHAR, H5T_NATIVE_INT, H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE

  ctypedef int hid_t  # In H5Ipublic.h
  ctypedef int herr_t
  ctypedef int htri_t
  ctypedef long long hsize_t    # How to declare that in a compatible MSVC way?
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

  cdef enum H5T_sign_t:
    H5T_SGN_ERROR        = -1,  #error                                      */
    H5T_SGN_NONE         = 0,   #this is an unsigned type                   */
    H5T_SGN_2            = 1,   #two's complement                           */
    H5T_NSGN             = 2    #this must be last!                         */

  cdef enum H5G_link_t:
    H5G_LINK_ERROR      = -1,
    H5G_LINK_HARD       = 0,
    H5G_LINK_SOFT       = 1

                
# Functions from HDF5
cdef extern from *:
  hid_t  H5Fcreate(char *filename, unsigned int flags,
                   hid_t create_plist, hid_t access_plist)
  
  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id)
                
  herr_t H5Fclose (hid_t file_id)

  htri_t H5Fis_hdf5(char *name)
  
  hid_t  H5Dopen (hid_t file_id, char *name)
  
  herr_t H5Dclose (hid_t dset_id)
  
  herr_t H5Dread (hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf)

  hid_t  H5Gcreate(hid_t loc_id, char *name, size_t size_hint )

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

  herr_t H5LTget_attribute_string( hid_t loc_id, char *obj_name,
                                   char *attr_name, char *attr_data )

  herr_t H5LT_find_attribute( hid_t loc_id, char *attr_name )


# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h":  
  
  herr_t H5ARRAYmake( hid_t loc_id, char *dset_name, char *title,
                      char *flavor, char *obversion, int atomictype,
                      int rank, hsize_t *dims, hid_t type_id,
                      void *data)

  herr_t H5ARRAYread( hid_t loc_id, char *dset_name,
                         void *data )

  herr_t H5ARRAYget_ndims ( hid_t loc_id, char *dset_name, int *rank )
  
  herr_t H5ARRAYget_info( hid_t loc_id, char *dset_name,
                          hsize_t *dims, H5T_class_t *class_id,
                          H5T_sign_t *sign, char *byteorder,
                          size_t *type_size )


# Funtion to compute the HDF5 type from a numarray enum type
cdef extern from "arraytypes.h":
    
  hid_t convArrayType(int fmt, size_t size, char *byteorder)
  int getArrayType(H5T_class_t class_id, size_t type_size,
                   H5T_sign_t sign, int *format)
                   
# I define this constant, but I should not, because it should be defined in
# the HDF5 library, but having problems importing it
cdef enum:
  MAX_FIELDS = 255

# Maximum size for strings
cdef enum:
  MAX_CHARS = 256

cdef extern from "H5TB.h":

  herr_t H5TBmake_table( char *table_title, hid_t loc_id, 
                         char *dset_name, hsize_t nfields,
                         hsize_t nrecords, size_t type_size,
                         char *field_names[], size_t *field_offset,
                         hid_t *field_types, hsize_t chunk_size,
                         void *fill_data, int compress, char *complib,
                         void *data )
                         
  herr_t H5TBappend_records ( hid_t loc_id, char *dset_name,
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
                         
  herr_t H5TBread_records ( hid_t loc_id, char *table_name,
                            hsize_t start, hsize_t nrecords, size_t type_size,
                            size_t *field_offset, void *data )

  herr_t H5TBread_fields_name ( hid_t loc_id, char *table_name,
                                char *field_names, hsize_t start,
                                hsize_t nrecords, size_t type_size,
                                size_t *field_offset, void *data )

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

  herr_t H5TBOclose_read( hid_t *dataset_id,
                          hid_t *space_id,
                          hid_t *mem_type_id )


  # These are maintained here just in case I want to use them in the future.
  # F.Alted 2003/04/20 

  herr_t H5TBOopen_append( hid_t loc_id, 
                           char *dset_name,
                           hsize_t nfields,
                           size_t type_size,
                           size_t *field_offset )
  
  herr_t H5TBOappend_records( hsize_t nrecords,
                              hsize_t nrecords_orig,
                              void *data )

  herr_t H5TBOclose_append( )

# Declarations from PyTables local functions

# Funtion to compute the offset of a struct format
cdef extern from "calcoffset.h":
  
  int calcoffset(char *fmt, size_t *offsets)
  
  int calctypes(char *fmt, hid_t *types, size_t *size_types)

# Funtion to get info from fields in a table
cdef extern from "getfieldfmt.h":
  herr_t getfieldfmt(hid_t loc_id, char *table_name, char *fmt)

# Helper routines
cdef extern from "utils.h":
  object _getTablesVersion()
  object createNamesTuple(char *buffer[], int nelements)
  object createDimsTuple(int dimensions[], int nelements)
  object Giterate(hid_t loc_id, char *name)
  object Aiterate(hid_t loc_id)
  H5T_class_t getHDF5ClassID(hid_t loc_id, char *name)

# ZLIB library
# We don't want to require the zlib headers installed
#cdef extern from "zlib.h":
#  char *zlibVersion()
  
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

def isLibAvailable(char *name):
  "Tell if an optional library is available or not"
    
  if (strcmp(name, "zlib") == 0):
    #return (1,zlibVersion(),None)   # Should be always available
    return (1,0,0)   # Should be always available
  if (strcmp(name, "lzo") == 0):
    if lzo_version:
      (lzo_version_string, lzo_version_date) = getLZOVersionInfo()
      return (lzo_version, lzo_version_string, lzo_version_date)
    else:
      return (0, None, None)
  elif (strcmp(name, "ucl") == 0):
    if ucl_version:
      (ucl_version_string, ucl_version_date) = getUCLVersionInfo()
      return (ucl_version, ucl_version_string, ucl_version_date)
    else:
      return (0, None, None)
  else:
    return (0, None, None)
    
def whichClass( hid_t loc_id, char *name):
  cdef H5T_class_t class_id

  class_id = getHDF5ClassID(loc_id, name)
  # Check if this a dataset of supported classtype for ARRAY
  if ((class_id == H5T_ARRAY)   or
      (class_id == H5T_INTEGER) or
      (class_id == H5T_FLOAT)   or
      (class_id == H5T_STRING)):
    return "Array"
  elif class_id == H5T_COMPOUND:
    return "Table"

  # Fallback 
  return "UNSUPPORTED"

def isHDF5(char *filename):
  """Determines whether a file is in the HDF5 format.

  When successful, returns a positive value, for TRUE, or 0 (zero),
  for FALSE. Otherwise returns a negative value.

  To this function to work, it needs a closed file.

  """
  
  return H5Fis_hdf5(filename)

def isPyTablesFile(char *filename):
  """Determines whether a file is in the PyTables format.

  When successful, returns the format version string, for TRUE, or 0
  (zero), for FALSE. Otherwise returns a negative value.

  To this function to work, it needs a closed file.

  """
  
  cdef hid_t root_id
  cdef herr_t ret
  cdef char attr_out[256]

  isptf = 0
  if os.path.isfile(filename) and H5Fis_hdf5(filename) > 0:
    # The file exists and is HDF5, that's ok
    # Open it in read-only mode
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT)
    # Open the root group
    root_id =  H5Gopen(file_id, "/")
    # Check if attribute exists
    if H5LT_find_attribute(root_id, 'PYTABLES_FORMAT_VERSION'):
      # Read the format_version attribute
      ret = H5LT_get_attribute_disk(root_id, 'PYTABLES_FORMAT_VERSION',
                                    attr_out)
      if ret >= 0:
        isptf = attr_out
      else:
        isptf = ret
    # Close root group
    H5Gclose(root_id)
    # Close the file
    H5Fclose(file_id)

  return isptf

def getHDF5Version():
  """Get the underlying HDF5 library version"""
  
  cdef unsigned majnum, minnum, relnum
  cdef char buffer[MAX_CHARS]
  cdef int ret
  
  ret = H5get_libversion(&majnum, &minnum, &relnum )
  if ret < 0:
    raise RuntimeError("Problems getting the HDF5 library version.")
  snprintf(buffer, MAX_CHARS, "%d.%d.%d", majnum, minnum, relnum )
  
  return buffer

def getExtVersion():
  """Return this extension CVS version"""
  
  # We need to do that here because
  # the cvsid really gives the CVS version of the generated C file (because
  # it is also in CVS!."""
  # But the $Id will be processed whenever a cvs commit is issued.
  # So, if you make a cvs commit *before* a .c generation *and*
  # you don't modify anymore the .pyx source file, you will get a cvsid
  # for the C file, not the Pyrex one!. The solution is not trivial!.
  return "$Id: hdf5Extension.pyx,v 1.49 2003/06/04 11:14:56 falted Exp $ "

def getPyTablesVersion():
  """Return this extension version."""
  
  #return PYTABLES_VERSION
  return _getTablesVersion()

# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef char    *name

  def __new__(self, char *name, char *mode, char *title,
              int new, object trTable):
    # Create a new file using default properties
    # Improve this to check if the file exists or not before
    self.name = name
    self.mode = mode
    if (strcmp(mode, "r") == 0 or strcmp(mode, "r+") == 0):
      if (os.path.isfile(name) and H5Fis_hdf5(name) > 0):
        # The file exists and is HDF5, that's ok
        #print "File %s exists... That's ok!" % name
        if strcmp(mode, "r") == 0:
          self.file_id = H5Fopen(name, H5F_ACC_RDONLY, H5P_DEFAULT)
        elif strcmp(mode, "r+") == 0:
          self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
      else:
        raise RuntimeError("File \'%s\' doesn't exist or is not a HDF5 file." \
                           % self.name )
    elif strcmp(mode, "a") == 0:
      if os.path.isfile(name):
        if H5Fis_hdf5(name) > 0:
          self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
        else:
          raise RuntimeError("File \'%s\' exist but is not a HDF5 file." % \
                             self.name )
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

  def _closeFile(self):
    # Close the table file
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
    self.parent_id = node._v_parent._v_groupId
    
  def _g_listAttr(self):
    cdef object attrlist
    cdef hid_t loc_id

    if isinstance(self.node, Group):
      # Return a tuple with the attribute list
      attrlist = Aiterate(self.node._v_groupId)
    else:
      # Get the dataset ID (the Leaf objects are always closed)
      loc_id = H5Dopen(self.parent_id, self.name)
      if loc_id < 0:
        raise RuntimeError("Cannot open the dataset '%s'" % self.name)
      attrlist = Aiterate(loc_id)
      # Close this dataset
      ret = H5Dclose(loc_id)
      if ret < 0:
        raise RuntimeError("Cannot close the dataset '%s'" % self.name)

    return attrlist

  def _g_setAttrStr(self, char *attrname, char *attrvalue):
    cdef int ret

    ret = H5LTset_attribute_string(self.parent_id, self.name,
                                   attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Can't set attribute '%s' in node:\n %s." % 
                         (attrname, self.node))

  def _g_getAttrStr(self, char *attrname):
    cdef object attrvalue
    cdef hid_t loc_id

    if isinstance(self.node, Group):
      attrvalue = self._g_getNodeAttrStr(self.parent_id, self.node._v_groupId,
                                         self.name, attrname)
    else:
      # Get the dataset ID
      loc_id = H5Dopen(self.parent_id, self.name)
      if loc_id < 0:
        raise RuntimeError("Cannot open the dataset '%s' in node '%s'" % \
                           (self.name, self._v_parent._v_name))

      attrvalue = self._g_getNodeAttrStr(self.parent_id, loc_id,
                                         self.name, attrname)
      # Close this dataset
      ret = H5Dclose(loc_id)
      if ret < 0:
        raise RuntimeError("Cannot close the dataset '%s'" % self.name)

    return attrvalue

  def _g_getChildAttrStr(self, char *dsetname, char *attrname):
    cdef object attrvalue
    cdef hid_t loc_id

    # Get the dataset ID
    loc_id = H5Dopen(self.node._v_groupId, dsetname)
    if loc_id < 0:
      raise RuntimeError("Cannot open the child '%s' of node '%s'" % \
                         (dsetname, self.name))

    attrvalue = self._g_getNodeAttrStr(self.node._v_groupId, loc_id,
                                       dsetname, attrname)
    # Close this dataset
    ret = H5Dclose(loc_id)
    if ret < 0:
      raise RuntimeError("Cannot close the dataset '%s'" % dsetname)

    return attrvalue

  # Get attributes (only supports string attributes right now)
  def _g_getNodeAttrStr(self, hid_t parent_id, hid_t loc_id,
                        char *dsetname, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef char *attrvalue
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

    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))

    ret = H5LTget_attribute_info(parent_id, dsetname, attrname,
                                 dims, &class_id, &type_size)
    if ret < 0:
        raise RuntimeError("Can't get info on attribute %s in node %s." %
                               (attrname, dsetname))

    if rank == 0:
      attrvalue = <char *>malloc(type_size * sizeof(char))
    else:
      elements = dim[0]
      for i from  0 < i < rank:
        nelements = nelements * dim[i]
      attrvalue = <char *>malloc(type_size * nelements * sizeof(char))

    ret = H5LTget_attribute_string(parent_id, dsetname,
                                    attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Attribute %s exists in node %s, but can't get it." \
                         % (attrname, dsetname))
                            
    return attrvalue

  def _g_remove(self, attrname):
    cdef int ret
    cdef hid_t loc_id
    
    if isinstance(self.node, Group):
      ret = H5Adelete(self.node._v_groupId, attrname ) 
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


cdef class Group:
  cdef hid_t   group_id
  cdef hid_t   parent_id
  cdef char    *name

  def _g_new(self, where, name):
    # Initialize the C attributes of Group object
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_groupId
    
  def _g_createGroup(self):
    cdef hid_t ret
    
    # Create a new group
    ret = H5Gcreate(self.parent_id, self.name, 0)
    if ret < 0:
      raise RuntimeError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_openGroup(self, hid_t loc_id, char *name):
    cdef hid_t ret
    
    # Open a existing group
    self.name = strdup(name)
    ret = H5Gopen(loc_id, self.name)
    if ret < 0:
      raise RuntimeError("Can't open the group %s." % self.name)
    self.group_id = ret
    #ret = H5Gclose(self.group_id)
    self.parent_id = loc_id
    return self.group_id

  def _g_listGroup(self, hid_t loc_id, char *name):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(loc_id, name)

  def _g_closeGroup(self):
    cdef int ret
    
    #print "Closing the HDF5 Group", self.name
    ret = H5Gclose(self.group_id)
    if ret < 0:
      raise RuntimeError("Problems closing the Group %s" % self.name )
    self.group_id = 0  # indicate that this group is closed

  def _g_renameNode(self, char *oldname, char *newname):
    cdef int ret

    #print "Renaming the HDF5 Node", oldname, "to", newname
    ret = H5Gmove(self.group_id, oldname, newname)
    if ret < 0:
      raise RuntimeError("Problems renaming the node %s" % oldname )
    return ret

  def _g_deleteGroup(self):
    cdef int ret

    # Delete this group
    #print "Removing the HDF5 Group", self.name
    ret = H5Gunlink(self.parent_id, self.name)
    if ret < 0:
      raise RuntimeError("Problems deleting the Group %s" % self.name )
    return ret

  def _g_deleteLeaf(self, char *dsetname):
    cdef int ret

    # Delete the leaf child
    #print "Removing the HDF5 Leaf", dsetname
    ret = H5Gunlink(self.group_id, dsetname)
    if ret < 0:
      raise RuntimeError("Problems deleting the Leaf %s" % dsetname )
    return ret

  def __dealloc__(self):
    cdef int ret
    
    #print "Destroying object Group in Extension"
    if self.group_id:
      #print "Closing the HDF5 Group", self.name," because user didn't do that!."
      ret = H5Gclose(self.group_id)
      if ret < 0:
        raise RuntimeError("Problems closing the Group %s" % self.name )

cdef class Table:
  # instance variables
  cdef size_t  rowsize
  cdef size_t  field_offset[MAX_FIELDS]
  cdef size_t  field_sizes[MAX_FIELDS]
  cdef hsize_t nfields
  cdef void    *rbuf
  cdef hsize_t totalrecords
  cdef hid_t   parent_id, loc_id
  cdef char    *name, *xtitle
  cdef char    *fmt
  cdef char    *field_names[MAX_FIELDS]
  cdef int     compress
  cdef char    *complib
  cdef hid_t   dataset_id, space_id, mem_type_id

  def _g_new(self, where, name):
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_groupId

  def _createTable(self, title, complib):
    cdef int nvar, offset
    cdef int i, nrecords, ret, buflen
    cdef hid_t fieldtypes[MAX_FIELDS]
    cdef void *fill_data, *data

    # Get a pointer to the table format
    self.fmt = PyString_AsString(self._v_fmt)
      
    # Assign the field_names pointers to the Tuple colnames strings
    i = 0
    for name in self.colnames:
      # The next works thanks to Pyrex magic
      self.field_names[i] = name
      i = i + 1
    # End old new

    # Compute the offsets
    nvar = calcoffset(self.fmt, self.field_offset)
    if (nvar > MAX_FIELDS):
        raise IndexError("A maximum of %d fields on tables is allowed" % \
                         MAX_FIELDS)
    self.nfields = nvar

    # Compute the field type sizes
    self.rowsize = calctypes(self.fmt, fieldtypes, self.field_sizes)
    
    # test if there is data to be saved initially
    if hasattr(self, "_v_recarray"):
      self.totalrecords = self.nrows
      ret = PyObject_AsWriteBuffer(self._v_recarray._data, &data, &buflen)
      if ret < 0:
        raise RuntimeError("Problems getting the pointer to the buffer")
      # Correct the offset in the buffer
      offset = self._v_recarray._byteoffset
      data = <void *>(<char *>data + offset)
    else:
      self.totalrecords = 0
      data = NULL

    # The next is settable if we have default values
    fill_data = NULL

    self.xtitle=strdup(title)
    self.complib=strdup(complib)
    ret = H5TBmake_table(self.xtitle, self.parent_id, self.name,
                         nvar, self.nrows, self.rowsize, self.field_names,
                         self.field_offset, fieldtypes, self._v_chunksize,
                         fill_data, self._v_compress, self.complib, data)
    if ret < 0:
      raise RuntimeError("Problems creating the table")
    
  def _append_records(self, object recarr, int nrecords):
    cdef int ret, buflen
    cdef void *rbuf

    # Get the pointer to the buffer data area
    ret = PyObject_AsWriteBuffer(recarr._data, &rbuf, &buflen)    
    if ret < 0:
      raise RuntimeError("Problems getting the pointer to the buffer")
    
    # Append the records:
    ret = H5TBappend_records(self.parent_id, self.name, nrecords,
                   self.rowsize, self.field_offset, rbuf)
    if ret < 0:
      raise RuntimeError("Problems appending the records")

    self.totalrecords = self.totalrecords + nrecords

  def _open_append(self, object recarr):
    cdef int buflen, ret

    # Get the pointer to the buffer data area
    ret = PyObject_AsWriteBuffer(recarr._data, &self.rbuf, &buflen)
    if ret < 0:
      raise RuntimeError("Problems getting the pointer to the buffer.")

    # Readout to the buffer
    if ( H5TBOopen_append(self.parent_id, self.name, self.nfields,
                          self.rowsize, self.field_offset) < 0 ):
      raise RuntimeError("Problems opening table for append.")

  def _append_records2(self, object recarr, int nrecords):
    cdef int ret,
    cdef void *rbuf

    self._open_append(recarr)
    
    # Append the records:
    ret = H5TBOappend_records(nrecords, self.totalrecords, self.rbuf)
    if ret < 0:
      raise RuntimeError("Problems appending the records.")

    self.totalrecords = self.totalrecords + nrecords
    self._close_append()
    
  def _close_append(self):

    # Close the table for append
    if ( H5TBOclose_append() < 0 ):
      raise RuntimeError("Problems closing table for append.")
    
  def _getTableInfo(self):
    "Get info from a table on disk. This method is standalone."
    cdef int     i, ret
    cdef hsize_t nrecords, nfields
    cdef hsize_t dims[1] # Tables are one-dimensional
    cdef H5T_class_t class_id
    cdef object  read_buffer, names_tuple
    cdef size_t  rowsize
    cdef size_t  field_offsets[MAX_FIELDS]
    cdef hid_t   fieldtypes[MAX_FIELDS]
    #cdef size_t  field_sizes[MAX_FIELDS]
    #cdef char    **field_names
    cdef char    fmt[255]

    # Get info about the table dataset
    ret = H5LTget_dataset_info(self.parent_id, self.name,
                               dims, &class_id, &rowsize)
    if ret < 0:
      raise RuntimeError("Problems getting table dataset info")
    self.rowsize = rowsize

    # First, know how many records (& fields) has the table
    ret = H5TBget_table_info(self.parent_id, self.name,
                             &nfields, &nrecords)
    if ret < 0:
      raise RuntimeError("Problems getting table info")
    self.nfields = nfields
    self.totalrecords = nrecords
    
    # Allocate space for the variable names
    #self.field_names = <char **>malloc(nfields * sizeof(char *))
    for i from  0 <= i < nfields:
      # Strings could not be larger than 255
      self.field_names[i] = <char *>malloc(MAX_CHARS * sizeof(char))  

    ret = H5TBget_field_info(self.parent_id, self.name,
                             self.field_names, self.field_sizes,
                             self.field_offset, &rowsize)
    if ret < 0:
      raise RuntimeError("Problems getting field info")
    ret = getfieldfmt(self.parent_id, self.name, fmt)
    if ret < 0:
      raise RuntimeError("Problems getting field format")
    self.fmt = fmt
    
    # Create a python tuple with the fields names
    names_tuple = []
    for i in range(nfields):
      names_tuple.append(self.field_names[i])
    names_tuple = tuple(names_tuple)

    # Return the buffer as a Python String
    return (nrecords, names_tuple, fmt)

  def _read_records_orig(self, hsize_t start, hsize_t nrecords,
                         object recarr):
    cdef herr_t ret
    cdef int buflen, ret2

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    # Get the pointer to the buffer data area
    ret2 = PyObject_AsWriteBuffer(recarr._data, &self.rbuf, &buflen)

    # Readout to the buffer
    ret = H5TBread_records(self.parent_id, self.name,
                           start, nrecords, self.rowsize,
                           self.field_offset, self.rbuf )
    if ret < 0:
      raise RuntimeError("Problems reading records.")

    return nrecords

  def _open_read(self, object recarr):
    cdef int buflen

    # Get the pointer to the buffer data area
    if ( PyObject_AsWriteBuffer(recarr._data, &self.rbuf, &buflen) < 0 ):
      raise RuntimeError("Problems getting the pointer to the buffer")

    # Readout to the buffer
    if ( H5TBOopen_read(&self.dataset_id, &self.space_id, &self.mem_type_id,
                        self.parent_id, self.name, self.nfields,
                        self.field_names, self.rowsize,
                        self.field_offset) < 0 ):
      raise RuntimeError("Problems opening table for read.")

  def _read_records(self, hsize_t start, hsize_t nrecords):

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    if ( H5TBOread_records(&self.dataset_id, &self.space_id,
                           &self.mem_type_id, start,
                           nrecords, self.rbuf) < 0 ):
      raise RuntimeError("Problems reading records.")

    return nrecords

  def _close_read(self):

    if ( H5TBOclose_read(&self.dataset_id, &self.space_id,
                         &self.mem_type_id) < 0 ):
      raise RuntimeError("Problems closing table for read.")
    
  def _read_field_name_orig(self, char *field_name, hsize_t start,
                       hsize_t nrecords, object recarr):
    cdef herr_t ret
    cdef void *rbuf
    cdef int buflen, ret2, i, fieldpos

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    for i in range(self.nfields):
      if strcmp(self.field_names[i], field_name) == 0:
        fieldpos = i
        
    # Get the pointer to the buffer data area
    ret2 = PyObject_AsWriteBuffer(recarr._data, &rbuf, &buflen)    

    # Readout to the buffer
    ret = H5TBread_fields_name(self.parent_id, self.name, field_name,
                               start, nrecords, self.field_sizes[fieldpos],
                               self.field_offset, rbuf )

    if ret < 0:
      raise RuntimeError("Problems reading records.")

    return nrecords

  def _read_field_name(self, char *field_name, hsize_t start,
                       hsize_t nrecords):
    cdef herr_t ret
    cdef int i, fieldpos

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    for i in range(self.nfields):
      if strcmp(self.field_names[i], field_name) == 0:
        fieldpos = i
        
    # Readout to the buffer
    ret = H5TBread_fields_name(self.parent_id, self.name, field_name,
                               start, nrecords, self.field_sizes[fieldpos],
                               self.field_offset, self.rbuf )

    if ret < 0:
      raise RuntimeError("Problems reading records.")

    return nrecords

  def __dealloc__(self):
    cdef int ret
    #print "Destroying object Table in Extension"
    free(<void *>self.name)

cdef class Row:
  """Row Class

  This class hosts accessors to a recarray row. The fields on a
  recarray can be accessed both as items (__getitem__/__setitem__),
  i.e. following the "map" protocol.
    
  """

  cdef object _fields, _recarray, _table, _saveBufferedRows, _indexes
  cdef int _row, _nrowinbuf, _nrow, _unsavednrows, _maxTuples, _strides, _opt
  cdef int start, stop, step, nextelement, nrowsinbuf
  cdef int *_dimensions, *_enumtypes

  def __new__(self, input, table):
    cdef int nfields, i
    
    self._recarray = input
    self._table = table
    #self.__dict__["_fields"] = input._fields ## Not allowed in pyrex!
    self._fields = input._fields
    self._unsavednrows = 0
    self._row = 0
    self._opt = 0
    self._nrow = 0
    self._strides = input._strides[0]
    nfields = input._nfields
    # Create a dictionary with the index columns of the recarray
    # and other tables
    i = 0
    self._indexes = {}
    self._dimensions = <int *>malloc(nfields * sizeof(int))
    self._enumtypes = <int *>malloc(nfields * sizeof(int))
    for field in input._names:
      self._indexes[field] = i
      self._dimensions[i] = input._repeats[i]
      self._enumtypes[i] = toenum[input._fmt[i]]
      i = i + 1
    self._maxTuples = table._v_maxTuples
    self._saveBufferedRows = table._saveBufferedRows

  def _initLoop(self, start, stop, step, nrowsinbuf):
    self.start = start
    self.stop = stop
    self.step = step
    self.nextelement = start
    self.nrowsinbuf = nrowsinbuf
    self._opt=1

  def __call__(self):
    """ return the row for this record object and update counters"""
    self._row = self._row + 1
    self._nrow = self._nrowinbuf + self._row
    return self

  def _getRow(self):
    """ return the row for this record object and update counters"""
    self._row = self._row + self.step
    self._nrow = self._nrowinbuf + self._row
    #print "Delivering row:", self._nrow, "// Buffer row:", self._row
    return self

  def _setBaseRow(self, start, startb):
    """ set the global row number and reset the local buffer row counter """
    self._nrowinbuf = start
    self._row = startb - self.step
    self._opt = 0

  def nrow(self):
    """ get the global row number for this table """
    return self._nrow

  def append(self):
    """Append the "row" object to the output buffer.
    
    "row" has to be a recarray2.Row object 

    """
    self._row = self._row + 1 # update the current buffer read counter
    self._unsavednrows = self._unsavednrows + 1
    # When the buffer is full, flush it
    if self._unsavednrows == self._maxTuples:
      self._saveBufferedRows()

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

  def __getitem__original(self, fieldName):
    try:
      return self._fields[fieldName][self._row]
      #return 40  # Just for testing purposes
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  # This method is twice as faster than __getattr__ because there is
  # not a lookup in the local dictionary
  def __getitem__(self, fieldName):
    cdef int index, offset

    # Optimization follows for the case that the field dimension is
    # == 1, i.e. columns elements are scalars, and the column is not
    # of CharType. This code accelerates the access to column
    # elements a 20%

    try:

      # Get the column index. This is very fast!
      index = self._indexes[fieldName]

      if (self._enumtypes[index] <> CHARTYPE and self._dimensions[index] == 1):
        # return 40   # Just for tests purposes
         #print "self._row -->", self._row, fieldName, self._strides
        #print "self._fields[fieldName] -->", self._fields[fieldName]
        # if not NA_updateDataPtr(self._fields[fieldName]):
        #  return None
        # This optimization sucks when using numarray 0.4!
        offset = self._row * self._strides
        return NA_getPythonScalar(self._fields[fieldName], offset)
        # return self._fields[fieldName][self._row]
      elif (self._enumtypes[index] == CHARTYPE):
        # CharType columns can only be unidimensional charrays right now,
        # so the elements has to be strings, so a copy() is not applicable here
        # But this should be addressed when multidimensional recarrays
        # were supported
        # Call the universal indexing function
        return self._fields[fieldName][self._row]
      else:  # Case when dimensions > 1 and not CharType
        # Call the universal indexing function
        # Make a copy of the (multi) dimensional array
        # so that the user does not have to do that!
        arr = self._fields[fieldName][self._row].copy()
        return arr
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  # This is slightly faster (around 3%) than __setattr__
  def __setitem__(self, fieldName, value):
    try:
      self._fields[fieldName][self._unsavednrows] = value
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  # This "optimization" sucks when using numarray 0.4 and 0.5!
  def __setitem__optim(self, fieldName, value):
    cdef int index, offset

    try:

      # Get the column index. This is very fast!
      index = self._indexes[fieldName]

      if (self._enumtypes[index] <> CHARTYPE and self._dimensions[index] == 1):
        # This optimization sucks when using numarray 0.4 and 0.5!
        offset = self._unsavednrows * self._strides
        #print "self._row -->", self._row, fieldName, self._strides
        #print "self._fields[fieldName] -->", self._fields[fieldName]
        # if not NA_updateDataPtr(self._fields[fieldName]):
        #  return None
        NA_setFromPythonScalar(self._fields[fieldName], offset, value)
        #self._fields[fieldName][self._unsavednrows] = value
      else:
        self._fields[fieldName][self._unsavednrows] = value
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  def __str__(self):
    """ represent the record as an string """
        
    outlist = []
    for name in self._recarray._names:
      outlist.append(`self._fields[name][self._row]`)
            #outlist.append(`self._recarray.field(name)[self._row]`)
    return "(" + ", ".join(outlist) + ")"

  def __repr__(self):
    """ represent the record as an string """

    return str(self)

  def _all(self):
    """ represent the record as a list """
    
    outlist = []
    for name in self._fields:
      outlist.append(self._fields[name][self._row])
      #outlist.append(self._recarray.field(name)[self._row])
    return outlist

  # Moved out of scope
  def _g_dealloc__(self):
    print "Deleting Row object"
    pass


cdef class Array:
  # Instance variables
  cdef hid_t   parent_id
  cdef char    *name
  cdef int     rank
  cdef hsize_t *dims
  cdef object  type
  cdef int     enumtype

  def _g_new(self, where, name):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_groupId

  def _createArray(self, object arr, char *title,
                  char *flavor, char *obversion, int atomictype):
    cdef int i
    cdef herr_t ret
    cdef hid_t type_id
    cdef void *rbuf
    cdef int buflen, ret2
    cdef object array, strcache
    cdef int itemsize, offset
    cdef char *tmp, *byteorder

    if isinstance(arr, num.NumArray):
      self.type = arr._type
      try:
        self.enumtype = toenum[arr._type]
      except KeyError:
        raise TypeError, \
      """Type class '%s' not supported rigth now. Sorry about that.
      """ % repr(arr._type)
      # Convert the array object to a an object with a well-behaved buffer
      #array = <object>NA_InputArray(arr, self.enumtype, C_ARRAY)
      # Do a copy of the array in case is not contiguous
      # We can deal with the non-aligned and byteswapped cases
      if not arr.iscontiguous():
        #array = arr.copy()
        # Change again the byteorder so as to keep the original one
        # (copy() resets the byteorder to that of the host machine)
        #if arr._byteorder <> array._byteorder:
        #  array._byteswap()
        # The next code is more efficient as it doesn't reverse the byteorder
        # twice (if byteorder is different than this of the machine).
        array = ndarray.NDArray.copy(arr)
        array._byteorder = arr._byteorder
        array._type = arr._type
      else:
        array = arr

      itemsize = array.type().bytes
      # The next is a trick to avoid a warning in Pyrex
      strcache = arr._byteorder
      byteorder = strcache
    elif isinstance(arr, chararray.CharArray):
      self.type = CharType
      #self.enumtype = 'a'
      self.enumtype = toenum[CharType]
      # Get a contiguous chararray object (well-behaved buffer)
      array = arr.contiguous()
      itemsize = array._itemsize
      # In CharArrays byteorder does not matter, but we need one
      # to pass it as convArrayType parameter
      strcache = sys.byteorder
      byteorder = strcache
      
    type_id = convArrayType(self.enumtype, itemsize, byteorder)
    if type_id < 0:
      raise TypeError, \
        """type '%s' is not supported right now. Sorry about that.""" \
    % self.type

    # Get the pointer to the buffer data area
    # PyObject_AsWriteBuffer cannot be used when buffers come from
    # Numeric objects. Using the Read version only leads to a
    # warning in compilation time.
    ret2 = PyObject_AsReadBuffer(array._data, &rbuf, &buflen)
    # Correct the start of the buffer with the _byteoffset
    offset = array._byteoffset
    rbuf = <void *>(<char *>rbuf + offset)

    if ret2 < 0:
      raise RuntimeError("Problems getting the buffer area.")

    # Allocate space for the dimension axis info
    self.rank = len(array.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < self.rank:
        self.dims[i] = array.shape[i]

    # Save the array
    ret = H5ARRAYmake(self.parent_id, self.name, title,
                      flavor, obversion, atomictype, self.rank,
                      self.dims, type_id, rbuf)
    if ret < 0:
      raise RuntimeError("Problems saving the array.")

    return self.type
    
  def _openArray(self):
    cdef object shape
    cdef size_t type_size
    cdef H5T_class_t class_id
    cdef H5T_sign_t sign
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i
    cdef herr_t ret

    # Get the rank for this array object
    ret = H5ARRAYget_ndims(self.parent_id, self.name, &self.rank)
    #ret = H5LTget_dataset_ndims(self.parent_id, self.name, &self.rank)
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type size
    ret = H5ARRAYget_info(self.parent_id, self.name, self.dims,
                             &class_id, &sign, byteorder, &type_size)

    # Get the array type
    ret = getArrayType(class_id, type_size,
                       sign, &self.enumtype)
    if ret < 0:
      raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id

    # We had problems when creating Tuples directly with Pyrex!.
    # A bug report has been sent to Greg and here is his answer:
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
    for i in range(self.rank):
      shape.append(self.dims[i])
    shape = tuple(shape)

    return (toclass[self.enumtype], shape, type_size, byteorder)
  
  def _readArray(self, object buf):
    cdef herr_t ret
    cdef void *rbuf
    cdef int buflen, ret2

    # Get the pointer to the buffer data area
    ret2 = PyObject_AsWriteBuffer(buf, &rbuf, &buflen)
    if ret2 < 0:
      raise RuntimeError("Problems getting the buffer area.")

    ret = H5ARRAYread(self.parent_id, self.name, rbuf)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    return 

  def __dealloc__(self):
    cdef int ret
    #print "Destroying object Array in Extension"
    free(<void *>self.dims)
    free(<void *>self.name)
