#  Ei!, emacs, this is -*-Python-*- mode
#  This is a test to access the HDF5 Lite
#  trough Pyrex!
#
# F. Alted 27 / August / 2002

cvsid = "$Id: hdf5Extension.pyx,v 1.1 2002/10/02 17:20:06 falted Exp $"

# This does not work. I think it's a Pyrex problem.
#cdef extern from "version.h":
#  cdef enum:
#    PYTABLES_VERSION

import os.path

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
                           hsize_t *dims, hid_t type_id, void *buffer )
  
  herr_t H5LTread_dataset( hid_t loc_id, char *dset_name,
		           hid_t type_id, void *buffer )

  herr_t H5LTget_dataset_info ( hid_t loc_id, char *dset_name,
                                hsize_t *dims, H5T_class_t *class_id,
                                size_t *type_size )

  herr_t H5LTget_attribute( hid_t loc_id, char *attr_name, void *attr_out )

  herr_t H5LTset_attribute_string( hid_t loc_id, char *obj_name,
                                   char *attr_name, char *attr_data )
  
# I define this constant, but I should not cause it should be defined in
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
  object Giterate(hid_t loc_id, char *name)


# utility funtions
def isHDF5(char *filename):
  return H5Fis_hdf5(filename)

def getHDF5Version():
  """Get the underlying HDF5 library version."""
  cdef unsigned majnum, minnum, relnum
  cdef char buffer[MAX_CHARS]
  cdef int ret
  
  ret = H5get_libversion(&majnum, &minnum, &relnum )
  if ret < 0:
    raise RuntimeError("Problems getting the HDF5 library version.")
  #print "Release Numbers:", majnum, minnum, relnum
  snprintf(buffer, MAX_CHARS, "%d.%d.%d", majnum, minnum, relnum )
  return buffer

def getExtCVSVersion():
  """ To guess this extension CVS version."""
  #We need to do that here because
  # the cvsid really gives the CVS version of the generated C file (because
  #it is also in CVS!."""
  # But this does not work!. The $Id will be processed whenever a cvs commit
  # is made. So, if you make a cvs commit *before* a .c generation *and*
  # you don't modify anymore the .pyx source file, you will get a cvsid
  # for the C file, not the Pyrex one!. The solution is not trivial!.
  return "$Id: hdf5Extension.pyx,v 1.1 2002/10/02 17:20:06 falted Exp $ "

def getTablesVersion():
  """Returns this extension version."""
  #return PYTABLES_VERSION
  return _getTablesVersion()

cdef class File:
  cdef hid_t   file_id
  cdef char    *filename

  def __new__(self, char *filename, char *mode):
    # Create a new file using default properties
    # Improve this to check if the file exists or not before
    self.filename = filename
    self.mode = mode
    if (strcmp(mode, "r") == 0 or strcmp(mode, "r+") == 0):
      if (os.path.isfile(filename) and H5Fis_hdf5(filename) > 0):
        # The file exists and is HDF5, that's ok
        #print "File %s exists... That's ok!" % filename
        if strcmp(mode, "r") == 0:
          self.file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT)
        elif strcmp(mode, "r+") == 0:
          self.file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT)
      else:
        raise RuntimeError("File %s doesn't exist or not a HDF5 file." % self.filename )
    elif strcmp(mode, "a") == 0:
      if os.path.isfile(filename):
        if H5Fis_hdf5(filename) > 0:
          self.file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT)
        else:
          raise RuntimeError("File %s exist but is not a HDF5 file." % self.filename )
      else:
        self.file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT)
    elif strcmp(mode, "w") == 0:
      self.file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT)
    else:
      raise RuntimeError("Invalid mode %s for opening a file." % self.mode )

  # Accessor definitions
  def getFileId(self):
    return self.file_id

  def closeFile(self):
    # Close the table file
    H5Fclose( self.file_id )
    self.file_id = 0    # Means file closed

  def __dealloc__(self):
    cdef int ret

    if self.file_id:
      print "Closing the HDF5 file", filename," because user didn't do that!."
      ret = H5Fclose(self.file_id)
      if ret < 0:
        raise RuntimeError("Problems closing the file %s" % self.filename )


cdef class Group:
  cdef hid_t   group_id
  cdef char    *name

  def __new__(self):
    # Initialize the C attributes of Group object (Very important!)
    self.name = strdup("")
    self.group_id = 0
    
  def _f_createGroup(self, hid_t loc_id, char *name):
    cdef hid_t ret
    
    # Create a new group
    self.name = strdup(name)
    ret = H5Gcreate(loc_id, name, 0)
    if ret < 0:
      raise RuntimeError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _f_openGroup(self, hid_t loc_id, char *name):
    cdef hid_t ret
    
    # Open a existing group
    self.name = strdup(name)
    ret = H5Gopen(loc_id, name)
    if ret < 0:
      raise RuntimeError("Can't open the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _f_listGroup(self, hid_t loc_id, char *name):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(loc_id, name)

  def _f_getGroupId(self):
    return self.group_id

  def _f_getGroupName(self):
    return self.name

  def __dealloc__(self):
    cdef int ret
    
    if strcmp(self.name, "") <> 0:
      #print "Closing a group (%s) in dealloc." % self.name
      if self.group_id:
        ret = H5Gclose(self.group_id)
        if ret < 0:
          raise RuntimeError("Problems closing the group %s" % self.name )
    else:
      # Un comment this when adding coding for clean up
      #print "Deallocating a non-used group in dealloc."
      pass


cdef class Table:

  # Class variables
  cdef size_t  dst_size
  cdef size_t  field_offset[MAX_FIELDS]
  cdef size_t  field_sizes[MAX_FIELDS]
  cdef hsize_t nfields
  cdef hsize_t totalrecords
  cdef hid_t   group_id, loc_id
  cdef char    *tablename, tabletitle[MAX_CHARS]
  cdef char    *fmt
  cdef char    **field_names
  cdef int     compress

  def __new__(self, where, name, root_id):
    # where is not needed
    self.tablename = strdup(name)
    # This parameter is not needed here
    #self.root_id = root_id
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
    ret = H5LTget_attribute(self.loc_id, "TITLE", self.tabletitle)
    if ret < 0:
      raise RuntimeError("Problems getting table TITLE attribute")
  
  def closeTable(self):
    cdef herr_t  ret

    ret = H5Dclose(self.loc_id)
    if ret < 0:
      raise RuntimeError("Problems closing the table")
  
  # Function reachable from Python
  def createTable(self, PyTupleObject varnames, char *fmt,
                  char *tableTitle, int compress,
                  int rowsize, hsize_t chunksize):
    cdef int     nvar
    cdef int     i, nrecords, ret
    cdef hid_t   fieldtypes[MAX_FIELDS]
    cdef char    *fill_data, *data

    #self.group_id = group_id
    self.fmt = fmt
    self.compress = compress
    #self.tabletitle = strdup(tableTitle)
    strncpy(self.tabletitle, tableTitle, MAX_CHARS)
  
    # Get the number of fields
    self.nfields = varnames.ob_size
    #nfields = PyTuple_Size(varnames)  # This is equivalent to the one before
    #print "nfields ==> ", self.nfields
    # Allocate space for the variable names
    self.field_names = <char **>malloc(self.nfields * sizeof(char *))
    # Assign this field_names pointers to the Tuple varnames strings
    i = 0
    for name in varnames:
      #print "Name", i, "--> ", name
      if (i > MAX_FIELDS):
        raise IndexError("Maximum of %d fields on struct allowed" % MAX_FIELDS)
      self.field_names[i] = name
      i = i + 1
    # End old new

    # Compute the offsets
    #print "fmt -->", self.fmt
    nvar = calcoffset(self.fmt, self.field_offset)

    # Compute the field type sizes
    rowsize = calctypes(self.fmt, fieldtypes, self.field_sizes)
    self.dst_size = rowsize

    if nvar <> self.nfields:
      print 'nvar: ', nvar, 'nfields: ', self.nfields
      raise RuntimeError("Format and varnames differ in variable number")

    # Create the table
    #print "Dataset --> ", self.tablename
    #print "Data size --> ", self.dst_size
    #print "Group id --> ", self.group_id

    # Don't fill data
    nrecords = 0     # Don't save any records now
    fill_data = NULL
    data = NULL
    ret = H5TBmake_table(self.tabletitle, self.group_id, self.tablename, nvar,
                   nrecords, rowsize, <char **>self.field_names,
                   self.field_offset, fieldtypes, chunksize, fill_data,
                   self.compress, data)
    if ret < 0:
      raise RuntimeError("Problems creating the table")
    # Initialize the total number of records for this table
    self.totalrecords = 0
    # We don't need to fill up the loc_id
    #self.openTable()  

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

  def getTableTitle(self):
    "Get the attribute TITLE from table. The maximum TITLE size is MAX_CHARS."    
    return PyString_FromString(self.tabletitle)

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
    #print "Dimension dataset ==> ", dims[0]
    #print "Dataset Class ID ==> ", class_id
    #print "Row size ==> ", rowsize

    # First, know how many records (& fields) has the table
    ret = H5TBget_table_info(self.group_id, self.tablename,
                             &nfields, &nrecords)
    if ret < 0:
      raise RuntimeError("Problems getting table info")
    self.nfields = nfields
    self.totalrecords = nrecords
    #print "Table Name ==> ", self.tablename
    #print "Nfields ==> ", nfields
    #print "Nrecords ==> ", nrecords
    
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

