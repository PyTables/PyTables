#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/hdf5Extension.pyx,v $
#       $Id: hdf5Extension.pyx,v 1.4 2002/11/10 13:31:50 falted Exp $
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

__version__ = "$Revision: 1.4 $"


import os.path

# C funtions and variable declaration from its headers

# Type size_t is defined in stdlib.h
cdef extern from "stdlib.h":
  ctypedef int size_t
  void *malloc(size_t size)

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
    
# Some declarations for Numeric objects and functions
cdef extern from "Numeric/arrayobject.h":
  
    struct PyArray_Descr:
        int type_num, elsize
        char type
        
    ctypedef class PyArrayObject [type PyArray_Type]:
        cdef char *data
        cdef int nd
        cdef int *dimensions, *strides
        cdef object base
        cdef PyArray_Descr *descr
        cdef int flags
    
    void import_array()
    
# The Numeric API requires this function to be called before
# using any Numeric facilities in an extension module.
import_array()

# Functions from Numeric
cdef extern from *:
  object PyArray_FromDims(int nd, int *d, int type)

cdef extern from "hdf5.h":
  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT
  int H5P_DEFAULT, H5S_ALL
  int H5T_NATIVE_CHAR, H5T_NATIVE_INT, H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE

  ctypedef int hid_t  # In H5Ipublic.h
  ctypedef int herr_t
  ctypedef int htri_t
  ctypedef long long hsize_t
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

  herr_t H5LTmake_array( hid_t loc_id, char *dset_name, char *title,
                         int rank, hsize_t *dims, hid_t type_id,
                         void *buffer )

  herr_t H5LTmake_dataset( hid_t loc_id, char *dset_name, int rank,
                           hsize_t *dims, hid_t type_id, void *buffer )
  
  herr_t H5LTread_dataset( hid_t loc_id, char *dset_name,
                           hid_t type_id, void *buffer )
                           
  herr_t H5LTread_array( hid_t loc_id, char *dset_name,
                         void *buffer )

  herr_t H5LTget_dataset_ndims ( hid_t loc_id, char *dset_name, int *rank )
  
  herr_t H5LTget_array_ndims ( hid_t loc_id, char *dset_name, int *rank )
  
  herr_t H5LTget_dataset_info ( hid_t loc_id, char *dset_name,
                                hsize_t *dims, H5T_class_t *class_id,
                                size_t *type_size )

  herr_t H5LTget_dataset_info_mod( hid_t loc_id, char *dset_name,
                                   hsize_t *dims, H5T_class_t *class_id,
                                   H5T_sign_t *sign, size_t *type_size )

  herr_t H5LTget_array_info( hid_t loc_id, char *dset_name,
                             hsize_t *dims, H5T_class_t *class_id,
                             H5T_sign_t *sign, size_t *type_size )

  herr_t H5LTget_attribute( hid_t loc_id, char *attr_name, void *attr_out )
          
  herr_t H5LTget_attribute_ndims( hid_t loc_id, char *attr_name, int *rank )
  
  herr_t H5LTget_attribute_info( hid_t loc_id, char *attr_name,
                                 hsize_t *dims, H5T_class_t *class_id,
                                 size_t *type_size )

  herr_t H5LTset_attribute_string( hid_t loc_id, char *obj_name,
                                   char *attr_name, char *attr_data )

  herr_t H5LTfind_attribute( hid_t loc_id, char *attr_name )
  
  
# Funtion to compute the HDF5 type from a Numarray typecode
cdef extern from "arraytypes.h":
    
  hid_t convArrayType(char fmt)
  
  int getArrayType(H5T_class_t class_id, size_t type_size,
                   H5T_sign_t sign, char *format)
                   
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

# utility funtions (these can be directly invoked from Python)

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
    if H5LTfind_attribute(root_id, 'PYTABLES_FORMAT_VERSION'):
      # Read the format_version attribute
      ret = H5LTget_attribute(root_id, 'PYTABLES_FORMAT_VERSION', attr_out)
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
  return "$Id: hdf5Extension.pyx,v 1.4 2002/11/10 13:31:50 falted Exp $ "

def getPyTablesVersion():
  """Return this extension version."""
  
  #return PYTABLES_VERSION
  return _getTablesVersion()

# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef char    *name

  def __new__(self, char *name, char *mode, char *title, int new):
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
  def _f_dealloc__(self):
    cdef int ret

    if self.file_id:
      print "Closing the HDF5 file", name," because user didn't do that!."
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
    
  def _f_createGroup(self, hid_t loc_id, char *name):
    cdef hid_t ret
    
    # Create a new group
    self.name = strdup(name)
    ret = H5Gcreate(loc_id, name, 0)
    if ret < 0:
      raise RuntimeError("Can't create the group %s." % self.name)
    self.group_id = ret
    self.parent_id = loc_id
    return self.group_id

  def _f_openGroup(self, hid_t loc_id, char *name):
    cdef hid_t ret
    
    # Open a existing group
    self.name = strdup(name)
    ret = H5Gopen(loc_id, name)
    if ret < 0:
      raise RuntimeError("Can't open the group %s." % self.name)
    self.group_id = ret
    self.parent_id = loc_id
    return self.group_id

  def _f_listGroup(self, hid_t loc_id, char *name):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(loc_id, name)

  # Get attributes (only supports string attributes right now)
  def _f_getGroupAttrStr(self, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id 
    cdef size_t type_size
    cdef char *attrvalue
    cdef int rank
    cdef int ret, i
        
    ret = H5LTget_attribute_ndims(self.group_id, attrname, &rank )
    if ret < 0:
      raise RuntimeError("Can't get ndims on attribute %s in group %s." % 
                             (attrname, self.name))
    #print "Attribute rank -->", rank

    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
        dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    
    ret = H5LTget_attribute_info(self.group_id, attrname,
                                 dims, &class_id, &type_size)
    if ret < 0:
        raise RuntimeError("Can't get info on attribute %s in group %s." % 
                               (self.attrname, self.name))
        
    if rank == 0:
      attrvalue = <char *>malloc(type_size * sizeof(char))
    else:
      elements = dim[0]
      for i from  0 < i < rank:
        nelements = nelements * dim[i]
      attrvalue = <char *>malloc(type_size * nelements * sizeof(char))
    
    ret = H5LTget_attribute(self.group_id, attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Can't get attribute %s in group %s." % 
                             (attrname, self.name))
                            
    return PyString_FromString(attrvalue)

  def _f_setGroupAttrStr(self, char *attrname, char *attrvalue):
    cdef int ret
      
    ret = H5LTset_attribute_string(self.parent_id, self.name,
                                   attrname, attrvalue)
    if ret < 0:
      raise RuntimeError("Can't set attribute %s in group %s." % 
                             (self.attrname, self.name))

  def _f_getDsetAttr(self, char *dsetname, char *attrname):
    cdef int ret
    cdef char attrvalue[MAX_CHARS]

    # Get the dataset
    loc_id = H5Dopen(self.group_id, dsetname)
    if loc_id < 0:
      raise RuntimeError("Cannot open the dataset")

    # Check if attribute exists
    if H5LTfind_attribute(loc_id, attrname):
      # Get the table TITLE attribute
      ret = H5LTget_attribute(loc_id, attrname, attrvalue)
      if ret < 0:
        raise RuntimeError("Cannot get '%s' attribute on dataset" % attrname)
    else:
      raise RuntimeError("Cannot get '%s' attribute on dataset" % attrname)

    # Close this dataset
    ret = H5Dclose(loc_id)
    if ret < 0:
      raise RuntimeError("Cannot close the dataset")

    return PyString_FromString(attrvalue)

cdef class Table:
  # instance variables
  cdef size_t  dst_size
  cdef size_t  field_offset[MAX_FIELDS]
  cdef size_t  field_sizes[MAX_FIELDS]
  cdef hsize_t nfields
  cdef hsize_t totalrecords
  cdef hid_t   group_id, loc_id
  cdef char    *tablename, title[MAX_CHARS]
  cdef char    *fmt
  cdef char    *field_names[MAX_FIELDS]
  cdef int     compress

  def _f_new(self, where, name):
    self.tablename = strdup(name)
    # The parent group id for this object
    self.group_id = where._v_groupId

  def openTable(self):
    """ Open the table, and read the table TITLE attribute."""
    cdef int ret
    
    # We need to do that just to obtain the loc_id for this dataset, and
    # H5LT doesn't provide a call to get it. Why?
    self.loc_id = H5Dopen(self.group_id, self.tablename)
    if self.loc_id < 0:
      raise RuntimeError("Problems opening the table")
  
    # Get the table TITLE attribute
    ret = H5LTget_attribute(self.loc_id, "TITLE", self.title)
    if ret < 0:
      raise RuntimeError("Problems getting table TITLE attribute")
  
  def closeTable(self):
    cdef herr_t  ret

    ret = H5Dclose(self.loc_id)
    if ret < 0:
      raise RuntimeError("Problems closing the table")
  
  # Function reachable from Python
  def createTable(self, PyTupleObject varnames, char *fmt,
                  char *title, int compress,
                  int rowsize, hsize_t chunksize):
    cdef int     nvar
    cdef int     i, nrecords, ret
    cdef hid_t   fieldtypes[MAX_FIELDS]
    cdef char    *fill_data, *data

    self.fmt = fmt
    self.compress = compress
    strncpy(self.title, title, MAX_CHARS)
    # End properly this string
    self.title[MAX_CHARS-1] = '\0'
  
    # Get the number of fields
    self.nfields = varnames.ob_size
    if (self.nfields > MAX_FIELDS):
        raise IndexError("A maximum of %d fields on tables is allowed" % \
                         MAX_FIELDS)
      
    # Assign the field_names pointers to the Tuple varnames strings
    i = 0
    for name in varnames:
      # The next works thanks to Pyrex magic
      self.field_names[i] = name
      i = i + 1
    # End old new

    # Compute the offsets
    nvar = calcoffset(self.fmt, self.field_offset)

    # Compute the field type sizes
    rowsize = calctypes(self.fmt, fieldtypes, self.field_sizes)
    self.dst_size = rowsize

    if nvar <> self.nfields:
      print 'nvar: ', nvar, 'nfields: ', self.nfields
      raise RuntimeError("Format and varnames differ in variable number")

    # Create the table
    nrecords = 0     # Don't save any records now
    fill_data = NULL
    data = NULL
    ret = H5TBmake_table(self.title, self.group_id, self.tablename,
                         nvar, nrecords, rowsize, self.field_names,
                         self.field_offset, fieldtypes, chunksize,
                         fill_data, self.compress, data)
    if ret < 0:
      raise RuntimeError("Problems creating the table")
    
    # Initialize the total number of records for this table
    self.totalrecords = 0
    
    # We need to assign loc_id a value to close the table afterwards
    self.openTable()  

  def append_record(self, PyStringObject record):
    cdef int ret, len
    cdef char *str

    str = record.ob_sval
    len = record.ob_size
    # Append a record:
    ret = H5TBappend_records(self.group_id, self.tablename, 1, self.dst_size,
                             self.field_offset, <void *>str)  
    if ret < 0:
      raise RuntimeError("Problems appending the record")
    self.totalrecords = self.totalrecords + 1

  def append_records(self, PyStringObject records, int nrecords):
    cdef int ret

    #print "About to save %d records..." % nrecords
    # Append the records:
    ret = H5TBappend_records(self.group_id, self.tablename, nrecords,
                   self.dst_size, self.field_offset, <void *>records.ob_sval)
    if ret < 0:
      raise RuntimeError("Problems appending the records")
    #print "After saving %d records..." % nrecords
    self.totalrecords = self.totalrecords + nrecords

  def getTitle(self):
    "Get the attribute TITLE from table. The maximum TITLE size is MAX_CHARS."
    return PyString_FromString(self.title)

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
    ret = H5LTget_dataset_info(self.group_id, self.tablename,
                               dims, &class_id, &rowsize)
    if ret < 0:
      raise RuntimeError("Problems getting table dataset info")
    self.dst_size = rowsize

    # First, know how many records (& fields) has the table
    ret = H5TBget_table_info(self.group_id, self.tablename,
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

    ret = H5TBget_field_info(self.group_id, self.tablename,
                             field_names, field_sizes,
                             self.field_offset, &rowsize)
    if ret < 0:
      raise RuntimeError("Problems getting field info")
    ret = getfieldfmt(self.group_id, self.tablename, fmt)
    if ret < 0:
      raise RuntimeError("Problems getting field format")
    #print "Calculated format ==> ", fmt
    self.fmt = fmt
    
    # Create a python tuple with the fields names
    names_tuple = createNamesTuple(field_names, nfields)

    # Return the buffer as a Python String
    return (nrecords, names_tuple, fmt)

  def read_records(self, hsize_t start, hsize_t nrecords,
                   PyStringObject buffer):
    cdef herr_t   ret

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    # Readout to the buffer
    ret = H5TBread_records(self.group_id, self.tablename,
                           start, nrecords, self.dst_size,
                           self.field_offset, <void *>buffer.ob_sval )
    if ret < 0:
      raise RuntimeError("Problems reading records.")

    return nrecords

cdef class Array:
  # Instance variables
  cdef hid_t   group_id
  cdef char    *name, title[MAX_CHARS]
  cdef char    fmt
  cdef hsize_t *dims
  cdef hid_t   type_id
  cdef int     *dimensions
  cdef H5T_class_t class_id
  cdef H5T_sign_t sign
  cdef size_t  type_size
  cdef char    typecode
  cdef int     rank

  def _f_new(self, where, name):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup(name)
    # The parent group id for this object
    self.group_id = where._v_groupId

  def createArray(self, PyArrayObject array, char *title):
    cdef int i
    cdef herr_t ret

    # Save the title as a class variable
    strncpy(self.title, title, MAX_CHARS)
    # End properly this string
    self.title[MAX_CHARS-1] = '\0'

    # Get some important parameters
    self.rank = array.nd
    self.typecode = array.descr.type

    # Get the HDF5 type_id for this typecode
    self.type_id = convArrayType(self.typecode)
    if self.type_id < 0:
      raise TypeError, \
        """typecode '%c' is not supported right now. Sorry about that.""" \
    % self.typecode

    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(array.nd * sizeof(hsize_t))
    self.dimensions = <int *>malloc(array.nd * sizeof(int))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < array.nd:
        self.dims[i] = array.dimensions[i]
        self.dimensions[i] = array.dimensions[i]
                               
    # Save the array
    ret = H5LTmake_array(self.group_id, self.name, self.title, array.nd,
                         self.dims, self.type_id, array.data)
    if ret < 0:
      raise RuntimeError("Problems saving the array.")

  def openArray(self):
    cdef hid_t loc_id
    cdef object shape
    cdef int i
    cdef herr_t ret
    cdef char typecodestr[2]

    # Get the rank for this array object
    ret = H5LTget_array_ndims(self.group_id, self.name, &self.rank)
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type size
    ret = H5LTget_array_info(self.group_id, self.name, self.dims,
                             &self.class_id, &self.sign, &self.type_size)
    
    # Copy the dimensions in a integer array
    # but first, book memory for the dimensions array
    self.dimensions = <int *>malloc(self.rank * sizeof(int))
    for i from  0 <= i < self.rank:
      self.dimensions[i] = <int>self.dims[i]

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
    # So, finally I've decided to code the shape tuple creation in C
    shape = createDimsTuple(self.dimensions, self.rank)
    
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    
    # Get info on dimensions, class and type size
    ret = H5LTget_array_info(self.group_id, self.name, self.dims,
                             &self.class_id, &self.sign, &self.type_size)
    ret = getArrayType(self.class_id, self.type_size,
                       self.sign, &self.typecode)
    # Get the type_id from the array typecode
    self.type_id = convArrayType(self.typecode)
    
    # Put the title in the self.title attribute
    self.getTitle()

    # Construct a typecode array from typecode character
    typecodestr[0] = self.typecode
    typecodestr[1] = '\0'
    
    # When returning the shape, the problems seems to appear
    return (PyString_FromString(typecodestr), shape)
  
  def getTitle(self):
    "Get the attribute TITLE. The maximum TITLE size is MAX_CHARS."

    # First open the data set
    loc_id = H5Dopen(self.group_id, self.name)
    if loc_id < 0:
      raise RuntimeError("Problems opening the array.")
    # Get the TITLE attribute
    ret = H5LTget_attribute(loc_id, "TITLE", &self.title)
    if ret < 0:
      raise RuntimeError("Problems getting array TITLE attribute.")
    # Close the dataset
    ret = H5Dclose(loc_id)
    if ret < 0:
      raise RuntimeError("Problems closing the array.")

    return ret

  def getArrayTitle(self):
    "Return the attribute self.title"
    return PyString_FromString(self.title)

  def readArray(self):
    cdef herr_t ret
    cdef PyArrayObject array

    # Create the array to fill it up later
    array = PyArray_FromDims(self.rank, self.dimensions, self.typecode)
    
    ret = H5LTread_array(self.group_id, self.name, <void *>array.data)
    if ret < 0:
      raise RuntimeError("Problems reading the array data.")

    # Return the new Numeric array
    return array

