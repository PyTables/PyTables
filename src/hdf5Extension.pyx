#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/hdf5Extension.pyx,v $
#       $Id: hdf5Extension.pyx,v 1.25 2003/02/24 15:57:46 falted Exp $
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

__version__ = "$Revision: 1.25 $"


import sys, os.path
import numarray as num
import chararray
import recarray2 as recarray

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

# Functions from numarray API
cdef extern from "numarray/libnumarray.h":
  PyArrayObject NA_InputArray (object, NumarrayType, int)
  PyArrayObject NA_OutputArray (object, NumarrayType, int)
  PyArrayObject NA_IoArray (object, NumarrayType, int)
  PyArrayObject PyArray_FromDims(int nd, int *d, int type)
  PyArrayObject NA_Empty(int nd, int *d, NumarrayType type)
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

  herr_t H5get_libversion(unsigned *majnum, unsigned *minnum,
                          unsigned *relnum )

  herr_t H5check_version(unsigned majnum, unsigned minnum,
          unsigned relnum )
     
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
                      void *data )

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
                         void *fill_data, int compress,
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
  H5T_class_t getHDF5ClassID(hid_t loc_id, char *name)

# utility funtions (these can be directly invoked from Python)

def whichClass( hid_t loc_id, char *name):
  cdef H5T_class_t class_id

  class_id = getHDF5ClassID(loc_id, name)
  # Check if this a dataset of supported classtype for ARRAY
  if ((class_id == H5T_ARRAY)   or
      (class_id == H5T_INTEGER) or
      (class_id == H5T_FLOAT)   or
      (class_id == H5T_STRING)):
    return "ARRAY"
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
  #print "Release Numbers:", majnum, minnum, relnum
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
  return "$Id: hdf5Extension.pyx,v 1.25 2003/02/24 15:57:46 falted Exp $ "

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
  def getFileId(self):
    return self.file_id

  def closeFile(self):
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


cdef class Group:
  cdef hid_t   group_id
  cdef hid_t   parent_id
  cdef char    *name

  def _f_new(self, where, name):
    # Initialize the C attributes of Group object
    self.name = strdup(name)
    # The parent group id for this object
    self.parent_id = where._v_groupId
    
  def _f_createGroup(self):
    cdef hid_t ret
    
    # Create a new group
    ret = H5Gcreate(self.parent_id, self.name, 0)
    if ret < 0:
      raise RuntimeError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _f_openGroup(self, hid_t loc_id, char *name):
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

  def _f_listGroup(self, hid_t loc_id, char *name):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(loc_id, name)

  def _f_getLeafAttrStr(self, char *dsetname, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef char *attrvalue
    cdef int rank
    cdef int ret, i

    # Get the dataset
    loc_id = H5Dopen(self.group_id, dsetname)
    if loc_id < 0:
      raise RuntimeError("Cannot open the dataset")

    # Check if attribute exists
    if H5LT_find_attribute(loc_id, attrname) <= 0:
        return None
      
    ret = H5LTget_attribute_ndims(self.group_id, dsetname, attrname, &rank )
    if ret < 0:
      raise RuntimeError("Can't get ndims on attribute %s in dset %s." %
                             (attrname, dsetname))

    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))

    ret = H5LTget_attribute_info(self.group_id, dsetname, attrname,
                                 dims, &class_id, &type_size)
    if ret < 0:
        raise RuntimeError("Can't get info on attribute %s in dset %s." %
                               (attrname, dsetname))

    if rank == 0:
      attrvalue = <char *>malloc(type_size * sizeof(char))
    else:
      elements = dim[0]
      for i from  0 < i < rank:
        nelements = nelements * dim[i]
      attrvalue = <char *>malloc(type_size * nelements * sizeof(char))

    ret = H5LTget_attribute_string(self.group_id, dsetname,
                                    attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Attribute %s exists in dset %s, but can't get it." \
                         % (attrname, dsetname))
                            
    # Close this dataset
    ret = H5Dclose(loc_id)
    if ret < 0:
      raise RuntimeError("Cannot close the dataset")

    return attrvalue

  # Get attributes (only supports string attributes right now)
  def _f_getGroupAttrStr(self, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef char *attrvalue
    cdef int rank
    cdef int ret, i
        
    # Check if attribute exists
    if H5LT_find_attribute(self.group_id, attrname) <= 0:
        return None

    ret = H5LTget_attribute_ndims(self.parent_id, self.name, attrname, &rank )
    if ret < 0:
      raise RuntimeError("Can't get ndims on attribute %s in group %s." %
                             (attrname, self.name))

    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))

    ret = H5LTget_attribute_info(self.parent_id, self.name, attrname,
                                 dims, &class_id, &type_size)
    if ret < 0:
        raise RuntimeError("Can't get info on attribute %s in group %s." %
                               (attrname, self.name))

    if rank == 0:
      attrvalue = <char *>malloc(type_size * sizeof(char))
    else:
      elements = dim[0]
      for i from  0 < i < rank:
        nelements = nelements * dim[i]
      attrvalue = <char *>malloc(type_size * nelements * sizeof(char))

    ret = H5LTget_attribute_string(self.parent_id, self.name,
                                    attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Attribute %s exists in group %s, but can't get it." \
                         % (attrname, self.name))
                            
    return attrvalue

  def _f_setGroupAttrStr(self, char *attrname, char *attrvalue):
    cdef int ret
      
    ret = H5LTset_attribute_string(self.parent_id, self.name,
                                   attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Can't set attribute %s in group %s." % 
                             (self.attrname, self.name))

  def _f_closeGroup(self):
    cdef int ret
    
    #print "Closing the HDF5 Group", self.name
    ret = H5Gclose(self.group_id)
    if ret < 0:
      raise RuntimeError("Problems closing the Group %s" % self.name )
    self.group_id = 0  # indicate that this group is closed

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
  cdef hsize_t totalrecords
  cdef hid_t   group_id, loc_id
  cdef char    *name, *xtitle
  cdef char    *fmt
  cdef char    *field_names[MAX_FIELDS]
  cdef int     compress

  def _f_new(self, where, name):
    self.name = strdup(name)
    # The parent group id for this object
    self.group_id = where._v_groupId

  def createTable(self, title):
    cdef int nvar
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

    # Compute the field type sizes
    self.rowsize = calctypes(self.fmt, fieldtypes, self.field_sizes)
    
    # test if there is data to be saved initially
    if hasattr(self, "_v_recarray"):
      self.totalrecords = self.nrows
      ret = PyObject_AsWriteBuffer(self._v_recarray._data, &data, &buflen)
      if ret < 0:
        raise RuntimeError("Problems getting the pointer to the buffer")
    else:
      self.totalrecords = 0
      data = NULL

    # The next is settable if we have default values
    fill_data = NULL

    self.xtitle=strdup(title)
    ret = H5TBmake_table(self.xtitle, self.group_id, self.name,
                         nvar, self.nrows, self.rowsize, self.field_names,
                         self.field_offset, fieldtypes, self._v_chunksize,
                         fill_data, self._v_compress, data)
    if ret < 0:
      raise RuntimeError("Problems creating the table")
    
    
  def append_records0(self, PyStringObject records, int nrecords):
    cdef int ret

    #print "About to save %d records..." % nrecords
    # Append the records:
    ret = H5TBappend_records(self.group_id, self.name, nrecords,
                   self.rowsize, self.field_offset, <void *>records.ob_sval)
    if ret < 0:
      raise RuntimeError("Problems appending the records")
    #print "After saving %d records..." % nrecords
    self.totalrecords = self.totalrecords + nrecords

  # append_records0 and append_records perfom similar speed (!)
  def append_records(self, object recarr, int nrecords):
    cdef int ret, buflen
    cdef void *rbuf

    #print "About to save %d records..." % nrecords
    # Get the pointer to the buffer data area
    ret = PyObject_AsWriteBuffer(recarr._data, &rbuf, &buflen)    
    if ret < 0:
      raise RuntimeError("Problems getting the pointer to the buffer")
    
    # Append the records:
    ret = H5TBappend_records(self.group_id, self.name, nrecords,
                   self.rowsize, self.field_offset, rbuf)
    if ret < 0:
      raise RuntimeError("Problems appending the records")
    #print "After saving %d records..." % nrecords
    self.totalrecords = self.totalrecords + nrecords

  def getTableInfo(self):
    "Get info from a table on disk. This method is standalone."
    cdef int     i, ret
    cdef hsize_t nrecords, nfields
    cdef hsize_t dims[1] # Tables are one-dimensional
    cdef H5T_class_t class_id
    cdef object  read_buffer, names_tuple
    cdef size_t  rowsize
    cdef size_t  field_offsets[MAX_FIELDS]
    cdef hid_t   fieldtypes[MAX_FIELDS]
    cdef size_t  field_sizes[MAX_FIELDS]
    cdef char    **field_names
    cdef char    fmt[255]

    # Get info about the table dataset
    ret = H5LTget_dataset_info(self.group_id, self.name,
                               dims, &class_id, &rowsize)
    if ret < 0:
      raise RuntimeError("Problems getting table dataset info")
    self.rowsize = rowsize

    # First, know how many records (& fields) has the table
    ret = H5TBget_table_info(self.group_id, self.name,
                             &nfields, &nrecords)
    if ret < 0:
      raise RuntimeError("Problems getting table info")
    self.nfields = nfields
    self.totalrecords = nrecords
    
    # Allocate space for the variable names
    field_names = <char **>malloc(nfields * sizeof(char *))
    for i from  0 <= i < nfields:
      # Strings could not be larger than 255
      field_names[i] = <char *>malloc(MAX_CHARS * sizeof(char))  

    ret = H5TBget_field_info(self.group_id, self.name,
                             field_names, field_sizes,
                             self.field_offset, &rowsize)
    if ret < 0:
      raise RuntimeError("Problems getting field info")
    ret = getfieldfmt(self.group_id, self.name, fmt)
    if ret < 0:
      raise RuntimeError("Problems getting field format")
    self.fmt = fmt
    
    # Create a python tuple with the fields names
    names_tuple = []
    for i in range(nfields):
      names_tuple.append(field_names[i])
    names_tuple = tuple(names_tuple)

    # Return the buffer as a Python String
    return (nrecords, names_tuple, fmt)

  def read_records(self, hsize_t start, hsize_t nrecords,
                   object recarr):
    cdef herr_t ret
    cdef void *rbuf
    cdef int buflen, ret2

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    # Get the pointer to the buffer data area
    ret2 = PyObject_AsWriteBuffer(recarr._data, &rbuf, &buflen)    

    # Readout to the buffer
    ret = H5TBread_records(self.group_id, self.name,
                           start, nrecords, self.rowsize,
                           self.field_offset, rbuf )
    if ret < 0:
      raise RuntimeError("Problems reading records.")

    return nrecords

  def __dealloc__(self):
    cdef int ret
    #print "Destroying object Table in Extension"
    free(<void *>self.name)

cdef class Row:
  cdef object _fields, _array
  cdef int _row, _nbuf, _nrow

  """Row Class

  This class hosts accessors to a recarray row.
    
  """

  def __new__(self, input):

    self._array = input
    #self.__dict__["_fields"] = input._fields ## Not allowed in pyrex!
    self._fields = input._fields
    self._row = 0

  def __call__(self):
    """ return the row for this record object """

    self._row = self._row + 1
    self._nrow = self._nbuf + self._row
    return self

  def setNBuf(self, nbuf):
    """ set the row for this record object """
    self._nbuf = nbuf

  def nrow(self):
    """ set the row for this record object """
    return self._nrow

  def setRow(self, row):
    """ set the row for this record object """
    self._row = row

  def incRow(self):
    """ set the row for this record object """
    self._row = self._row + 1

  # This is twice as faster than __getattr__ because no lookup in local
  # dictionary
  def getField(self, fieldName):
    try:
      return self._fields[fieldName][self._row]
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  def __getattr__(self, fieldName):
    """ get the field data of the record"""

    # In case that the value is an array, the user should be responsible to
    # copy it if he wants to keep it.
    try:
      return self._fields[fieldName][self._row]
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  def setField(self, fieldName, value):
    try:
      self._fields[fieldName][self._row] = value
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type,value))

  def __setattr__(self, fieldName, value):
    """ set the field data of the record"""

    try:
      self._fields[fieldName][self._row] = value
    except:
      (type, value, traceback) = sys.exc_info()
      raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
            (fieldName, "Error was: \"%s: %s\"" % (type, value))

  def __str__(self):
    """ represent the record as an string """
        
    outlist = []
    for name in self._array._names:
      outlist.append(`self._fields[name][self._row]`)
            #outlist.append(`self._array.field(name)[self._row]`)
    return "(" + ", ".join(outlist) + ")"

  def _all(self):
    """ represent the record as a list """
    
    outlist = []
    for name in self._fields:
      outlist.append(self._fields[name][self._row])
      #outlist.append(self._array.field(name)[self._row])
    return outlist

  # Moved out of scope
  def _f_dealloc__(self):
    print "Deleting Row object"
    pass


cdef class Array:
  # Instance variables
  cdef hid_t   group_id
  cdef char    *name
  cdef int     rank
  cdef hsize_t *dims
  cdef object  type
  cdef int     enumtype

  def _f_new(self, where, name):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup(name)
    # The parent group id for this object
    self.group_id = where._v_groupId

  def createArray(self, object arr, char *title,
                  char *flavor, char *obversion, int atomictype):
    cdef int i
    cdef herr_t ret
    cdef hid_t type_id
    cdef void *rbuf
    cdef int buflen, ret2
    cdef object array, strcache
    cdef int itemsize
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
        array = arr.copy()
        # Change again the byteorder so as to keep the original one
        # (copy() resets the byteorder to that of the host machine)
        if arr._byteorder <> array._byteorder:
          array._byteswap()
      else:
        array = arr

      itemsize = array.type().bytes
      # The next is a trick to avoid a warning in Pyrex
      strcache = arr._byteorder
      byteorder = strcache
    elif isinstance(arr, chararray.CharArray):
      self.type = CharType
      self.enumtype = 'a'
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
    ret2 = PyObject_AsReadBuffer(array._data, &rbuf, &buflen)
    # PyObject_AsWriteBuffer cannot be used when buffers come from
    # Numeric objects. Using the Read version only leads to a
    # warning in compilation time
    if ret2 < 0:
      raise RuntimeError("Problems getting the buffer area.")

    # Allocate space for the dimension axis info
    self.rank = len(array.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < self.rank:
        self.dims[i] = array.shape[i]

    # Save the array
    ret = H5ARRAYmake(self.group_id, self.name, title,
                      flavor, obversion, atomictype, self.rank,
                      self.dims, type_id, rbuf)
    if ret < 0:
      raise RuntimeError("Problems saving the array.")

    return self.type
    
  def openArray(self):
    cdef object shape
    cdef size_t type_size
    cdef H5T_class_t class_id
    cdef H5T_sign_t sign
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i
    cdef herr_t ret

    # Get the rank for this array object
    ret = H5ARRAYget_ndims(self.group_id, self.name, &self.rank)
    #ret = H5LTget_dataset_ndims(self.group_id, self.name, &self.rank)
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type size
    ret = H5ARRAYget_info(self.group_id, self.name, self.dims,
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
  
  def readArray(self, object buf):
    cdef herr_t ret
    cdef void *rbuf
    cdef int buflen, ret2

    # Get the pointer to the buffer data area
    ret2 = PyObject_AsWriteBuffer(buf, &rbuf, &buflen)
    if ret2 < 0:
      raise RuntimeError("Problems getting the buffer area.")

    ret = H5ARRAYread(self.group_id, self.name, rbuf)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    return 

  def __dealloc__(self):
    cdef int ret
    #print "Destroying object Array in Extension"
    free(<void *>self.dims)
    free(<void *>self.name)
