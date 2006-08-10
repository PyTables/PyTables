#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: June 17, 2005
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Here is where Table and Row extension types live.

Classes (type extensions):

    Table
    Row

Functions:



Misc variables:

    __version__
"""

import numarray

import tables.hdf5Extension
from tables.exceptions import HDF5ExtError
from tables.utilsExtension import createNestedType, \
     getNestedType, convertTime64, space2null, getTypeEnum, enumFromHDF5
from tables.numexpr import evaluate  ##XXX


from definitions cimport \
     import_libnumarray, NA_getPythonScalar, NA_setFromPythonScalar, \
     NA_getBufferPtrAndSize, Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS

__version__ = "$Revision$"


#-----------------------------------------------------------------

# Define the CharType code as a constant
cdef enum:
  CHARTYPE = 97  # 97 == ord('a')

# Standard C functions.
cdef extern from "stdlib.h":
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "string.h":
  char *strcpy(char *dest, char *src)
  char *strncpy(char *dest, char *src, size_t n)
  int strcmp(char *s1, char *s2)
  char *strdup(char *s)
  void *memcpy(void *dest, void *src, size_t n)

# Python API functions.
cdef extern from "Python.h":
  char *PyString_AsString(object string)

# HDF5 API.
cdef extern from "hdf5.h":
  # types
  ctypedef int hid_t
  ctypedef int herr_t
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

  # HDF5 layouts
  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    #raw data is very small                     */
    H5D_CONTIGUOUS      = 1,    #the default                                */
    H5D_CHUNKED         = 2,    #slow and fancy                             */
    H5D_NLAYOUTS        = 3     #this one must be last!                     */

  # The difference between a single file and a set of mounted files
  cdef enum H5F_scope_t:
    H5F_SCOPE_LOCAL     = 0,    # specified file handle only
    H5F_SCOPE_GLOBAL    = 1,    # entire virtual file
    H5F_SCOPE_DOWN      = 2     # for internal use only

  # For deleting a table
  herr_t H5Gunlink (hid_t file_id, char *name)

  # For flushing
  herr_t H5Fflush(hid_t object_id, H5F_scope_t scope)

  # Functions for dealing with datasets
  hid_t  H5Dopen(hid_t file_id, char *name)
  herr_t H5Dclose(hid_t dset_id)
  herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
                  hid_t file_space_id, hid_t plist_id, void *buf)
  hid_t H5Dget_type(hid_t dset_id)
  hid_t H5Dget_space(hid_t dset_id)

  # For getting the layout of a dataset
  hid_t H5Dget_create_plist(hid_t dataset_id)
  H5D_layout_t H5Pget_layout(hid_t plist)
  herr_t H5Pclose(hid_t plist)

  # Functions for dealing with dataspaces
  int H5Sget_simple_extent_ndims(hid_t space_id)

  int H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[],
                                hsize_t maxdims[])

  herr_t H5Sclose(hid_t space_id)

  # Functions for dealing with datatypes
  size_t H5Tget_size(hid_t type_id)
  hid_t  H5Tcreate(H5T_class_t type, size_t size)
  hid_t  H5Tcopy(hid_t type_id)
  herr_t H5Tclose(hid_t type_id)
  herr_t H5Tget_sign(hid_t type_id)


cdef extern from "H5TB-opt.h":

  herr_t H5TBOmake_table( char *table_title, hid_t loc_id, char *dset_name,
                          char *version, char *class_,
                          hid_t mem_type_id, hsize_t nrecords,
                          hsize_t chunk_size, int compress,
                          char *complib, int shuffle, int fletcher32,
                          void *data )

  herr_t H5TBOread_records( hid_t dataset_id, hid_t mem_type_id,
                            hsize_t start, hsize_t nrecords, void *data )

  herr_t H5TBOread_elements( hid_t dataset_id, hid_t mem_type_id,
                             hsize_t nrecords, void *coords, void *data )

  herr_t H5TBOappend_records( hid_t dataset_id, hid_t mem_type_id,
                              hsize_t nrecords, hsize_t nrecords_orig,
                              void *data )

  herr_t H5TBOwrite_records ( hid_t dataset_id, hid_t mem_type_id,
                              hsize_t start, hsize_t nrecords,
                              hsize_t step, void *data )

  herr_t H5TBOwrite_elements( hid_t dataset_id, hid_t mem_type_id,
                              hsize_t nrecords, void *coords, void *data )

  herr_t H5TBOdelete_records( hid_t   dataset_id, hid_t   mem_type_id,
                              hsize_t ntotal_records, size_t  src_size,
                              hsize_t start, hsize_t nrecords,
                              hsize_t maxtuples )

# Functions from HDF5 HL Lite
cdef extern from "H5ATTR.h":

  herr_t H5ATTRset_attribute_string( hid_t loc_id, char *attr_name,
                                     char *attr_data )

  herr_t H5ATTR_set_attribute_numerical( hid_t loc_id, char *attr_name,
                                         hid_t type_id, void *data )

#----------------------------------------------------------------------------

# Initialization code

# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()

#-------------------------------------------------------------

# XXX This should inherit from `tables.hdf5Extension.Leaf`,
# XXX but I don't know the Pyrex machinery to make it work.
# XXX ivb(2005-07-21)

cdef class Table:  # XXX extends Leaf
  # instance variables
  cdef void    *rbuf, *wbuf
  cdef hsize_t totalrecords
  cdef char    *name  # XXX from Node
  cdef hid_t   parent_id  # XXX from Node
  cdef hid_t   dataset_id, type_id, disk_type_id

  def _g_new(self, where, name, init):
    self.name = strdup(name)  # XXX from Node._g_new()
    # The parent group id for this object
    self.parent_id = where._v_objectID  # XXX from Node._g_new()
    if init:
      self.dataset_id = -1
      self.type_id = -1
      self.disk_type_id = -1

  def _g_delete(self):  # XXX Should inherit from Node
    cdef int ret

    # Delete this node
    ret = H5Gunlink(self.parent_id, self.name)
    if ret < 0:
      raise HDF5ExtError("problems deleting the node ``%s``" % self.name)
    return ret

  def _createTable(self, char *title, char *complib, char *obversion):
    cdef int     offset
    cdef int     ret
    cdef long    buflen
    cdef hid_t   oid
    cdef void    *data
    cdef hsize_t nrecords
    cdef char    *class_
    cdef object  i, fieldname, name

    # Compute the complete compound datatype based on the table description
    self.type_id = createNestedType(self.description, self.byteorder)
    # The on-disk type should be the same than in-memory
    self.disk_type_id = H5Tcopy(self.type_id)

    # test if there is data to be saved initially
    if self._v_recarray is not None:
      self.totalrecords = self.nrows
      buflen = NA_getBufferPtrAndSize(self._v_recarray._data, 1, &data)
      # Correct the offset in the buffer
      offset = self._v_recarray._byteoffset
      data = <void *>(<char *>data + offset)
    else:
      self.totalrecords = 0
      data = NULL

    class_ = PyString_AsString(self._c_classId)
    self.dataset_id = H5TBOmake_table(title, self.parent_id, self.name,
                                      obversion, class_, self.type_id,
                                      self.nrows, self._v_chunksize,
                                      self.filters.complevel, complib,
                                      self.filters.shuffle,
                                      self.filters.fletcher32,
                                      data)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the table")

    # Set the conforming table attributes
    # Attach the CLASS attribute
    ret = H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_)
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in table:\n %s." %
                         ("CLASS", self.name))
    # Attach the VERSION attribute
    ret = H5ATTRset_attribute_string(self.dataset_id, "VERSION", obversion)
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in table:\n %s." %
                         ("VERSION", self.name))
    # Attach the TITLE attribute
    ret = H5ATTRset_attribute_string(self.dataset_id, "TITLE", title)
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in table:\n %s." %
                         ("TITLE", self.name))
    # Attach the NROWS attribute
    nrecords = self.nrows
    ret = H5ATTR_set_attribute_numerical(self.dataset_id, "NROWS",
                                         H5T_NATIVE_LLONG, &nrecords )
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in table:\n %s." %
                         ("NROWS", self.name))

    # Attach the FIELD_N_NAME attributes
    # We write only the first level names
    i = 0
    for name in self.description._v_names:
      fieldname = "FIELD_%s_NAME" % i
      ret = H5ATTRset_attribute_string(self.dataset_id, fieldname, name)
      i = i + 1
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in table:\n %s." %
                         (fieldname, self.name))

    # If created in PyTables, the table is always chunked
    self._chunked = 1  # Accessible from python

    # Finally, return the object identifier.
    return self.dataset_id


  def _getInfo(self):
    "Get info from a table on disk."
    cdef hid_t   space_id
    cdef size_t  type_size
    cdef hsize_t dims[1]  # enough for unidimensional tables
    cdef hid_t   plist
    cdef H5D_layout_t layout

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    # Get the datatype on disk
    self.disk_type_id = H5Dget_type(self.dataset_id)
    # Get the number of rows
    space_id = H5Dget_space(self.dataset_id)
    H5Sget_simple_extent_dims(space_id, dims, NULL)
    self.totalrecords = dims[0]
    # Make totalrecords visible in python space
    self.nrows = self.totalrecords
    # Free resources
    H5Sclose(space_id)

    # Get the layout of the datatype
    plist = H5Dget_create_plist(self.dataset_id)
    layout = H5Pget_layout(plist)
    H5Pclose(plist)
    if layout == H5D_CHUNKED:
      self._chunked = 1
    else:
      self._chunked = 0

    # Get the type size
    type_size = H5Tget_size(self.disk_type_id)
    # Create the native data in-memory
    self.type_id = H5Tcreate(H5T_COMPOUND, type_size)
    # Fill-up the (nested) native type and description
    desc = getNestedType(self.disk_type_id, self.type_id, self)
    if desc == {}:
      raise HDF5ExtError("Problems getting desciption for table %s", self.name)

    # Return the object ID and the description
    return (self.dataset_id, desc)

  def _loadEnum(self, hid_t fieldTypeId):
    """_loadEnum(colname) -> (Enum, naType)
    Load enumerated type associated with `colname` column.

    This method loads the HDF5 enumerated type associated with
    `colname`.  It returns an `Enum` instance built from that, and the
    Numarray type used to encode it.
    """

    cdef hid_t enumId

    enumId = getTypeEnum(fieldTypeId)

    # Get the Enum and Numarray types and close the HDF5 type.
    try:
      return enumFromHDF5(enumId)
    finally:
      # (Yes, the ``finally`` clause *is* executed.)
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")

  def _convertTypes(self, object recarr, hsize_t nrecords, int sense):
    """Converts columns in 'recarr' between Numarray and HDF5 formats.

    Numarray to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to Numarray conversion is performed.
    The conversion is done in place, i.e. 'recarr' is modified.
    """

    # This should be generalised to support other type conversions.
    for t64cname in self._time64colnames:
      convertTime64(recarr.field(t64cname), nrecords, sense)

    # Only convert padding spaces into nulls if we have a "numpy" flavor
    if self.flavor == "numpy":
      for strcname in self._strcolnames:
        space2null(recarr.field(strcname), nrecords, sense)

  def _open_append(self, object recarr):
    cdef long buflen

    self._v_recarray = recarr
    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &self.wbuf)

  def _append_records(self, int nrecords):
    cdef int ret

    # Convert some Numarray types to HDF5 before storing.
    self._convertTypes(self._v_recarray, nrecords, 0)

    # release GIL (allow other threads to use the Python interpreter)
    Py_BEGIN_ALLOW_THREADS
    # Append the records:
    ret = H5TBOappend_records(self.dataset_id, self.type_id,
                              nrecords, self.totalrecords, self.wbuf)
    # acquire GIL (disallow other threads from using the Python interpreter)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems appending the records.")

    self.totalrecords = self.totalrecords + nrecords

  def _close_append(self):

    # Update the NROWS attribute
    if (H5ATTR_set_attribute_numerical(self.dataset_id, "NROWS",
                                       H5T_NATIVE_LLONG,
                                       &self.totalrecords)<0):
      raise HDF5ExtError("Problems setting the NROWS attribute.")

  def _update_records(self, hsize_t start, hsize_t stop,
                      hsize_t step, object recarr):
    cdef herr_t ret
    cdef void *rbuf
    cdef hsize_t nrecords, nrows

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &rbuf)

    # Compute the number of records to update
    nrecords = len(recarr)
    nrows = ((stop - start - 1) / step) + 1
    if nrecords > nrows:
      nrecords = nrows

    # Convert some Numarray types to HDF5 before storing.
    self._convertTypes(recarr, nrecords, 0)
    # Update the records:
    Py_BEGIN_ALLOW_THREADS
    ret = H5TBOwrite_records(self.dataset_id, self.type_id,
                             start, nrecords, step, rbuf )
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems updating the records.")

  def _update_elements(self, hsize_t nrecords, object elements,
                       object recarr):
    cdef herr_t ret
    cdef void *rbuf
    cdef long offset
    cdef void *coords

    # Get the chunk of the coords that correspond to a buffer
    buflen = NA_getBufferPtrAndSize(elements._data, 1, &coords)
    # Correct the offset
    offset = elements._byteoffset
    coords = <void *>(<char *>coords + offset)

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &rbuf)

    # Convert some Numarray types to HDF5 before storing.
    self._convertTypes(recarr, nrecords, 0)
    # Update the records:
    Py_BEGIN_ALLOW_THREADS
    ret = H5TBOwrite_elements(self.dataset_id, self.type_id,
                              nrecords, coords, rbuf )
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems updating the records.")

  def _read_records(self, hsize_t start, hsize_t nrecords, object recarr):
    cdef long buflen
    cdef void *rbuf
    cdef int ret

    # Correct the number of records to read, if needed
    if (start + nrecords) > self.totalrecords:
      nrecords = self.totalrecords - start

    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &rbuf)

    # Read the records from disk
    Py_BEGIN_ALLOW_THREADS
    ret = H5TBOread_records(self.dataset_id, self.type_id, start,
                            nrecords, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading records.")

    # Convert some HDF5 types to Numarray after reading.
    self._convertTypes(recarr, nrecords, 1)

    return nrecords

  def _read_elements(self, object recarr, object elements):
    cdef long buflen, offset
    cdef void *rbuf, *coords
    cdef hsize_t nrecords
    cdef int ret

    # Get the chunk of the coords that correspond to a buffer
    nrecords = len(elements)
    # Get the pointer to the buffer data area
    buflen = NA_getBufferPtrAndSize(recarr._data, 1, &rbuf)
    # Get the pointer to the buffer coords area
    buflen = NA_getBufferPtrAndSize(elements._data, 1, &coords)
    # Correct the offset
    offset = elements._byteoffset
    coords = <void *>(<char *>coords + offset)

    Py_BEGIN_ALLOW_THREADS
    ret = H5TBOread_elements(self.dataset_id, self.type_id,
                             nrecords, coords, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading records.")

    # Convert some HDF5 types to Numarray after reading.
    self._convertTypes(recarr, nrecords, 1)

    return nrecords

  def _remove_row(self, hsize_t nrow, hsize_t nrecords):
    cdef size_t rowsize

    # Protection against deleting too many rows
    if (nrow + nrecords > self.totalrecords):
      nrecords = self.totalrecords - nrow

    rowsize = self.rowsize
    # Using self.disk_type_id should be faster (i.e. less conversions)
    if (H5TBOdelete_records(self.dataset_id, self.disk_type_id,
                            self.totalrecords, rowsize, nrow, nrecords,
                            self._v_maxTuples) < 0):
      raise HDF5ExtError("Problems deleting records.")
      #print "Problems deleting records."
      # Return no removed records
      return 0
    else:
      self.totalrecords = self.totalrecords - nrecords
      # Attach the NROWS attribute
      H5ATTR_set_attribute_numerical(self.dataset_id, "NROWS",
                                     H5T_NATIVE_LLONG, &self.totalrecords)
      # Return the number of records removed
      return nrecords

  def  _get_type_id(self):
    "Accessor to type_id"
    return self.type_id

  def _g_flush(self):
    # Flush the dataset (in fact, the entire buffers in file!)
    if self.dataset_id >= 0:
        H5Fflush(self.dataset_id, H5F_SCOPE_GLOBAL)

  def _g_close(self):
    # Close dataset in HDF5 space
    # Release resources
    if self.type_id >= 0:
      H5Tclose(self.type_id)
    if self.disk_type_id >= 0:
      H5Tclose(self.disk_type_id)
    if self.dataset_id >= 0:
      H5Dclose(self.dataset_id)

  def __dealloc__(self):
    #print "Destroying object Table in Extension"
    free(<void *>self.name)  # XXX from Node


cdef class Row:
  """Row Class

  This class hosts accessors to a recarray row. The fields on a
  recarray can be accessed both as items (__getitem__/__setitem__),
  i.e. following the "map" protocol.

  """

  cdef long    _row, _unsaved_nrows, _mod_nrows
  cdef hsize_t start, stop, step, nextelement, _nrow
  cdef hsize_t nrowsinbuf, nrows, nrowsread, stopindex
  cdef int     bufcounter, counter, startb, stopb,  _all
  cdef int     *_scalar, *_enumtypes, exist_enum_cols
  cdef int     _riterator, _stride
  cdef int     whereCond2XXX, whereCond, indexed2XXX, indexed, indexChunk
  cdef int     ro_filemode, chunked
  cdef int     _bufferinfo_done
  cdef char    *colname
  cdef object  table
  cdef object  rbufRA, wbufRA
  cdef object  _wfields, _rfields, _indexes
  cdef object  indexValid, coords, bufcoords, index, indices
  cdef object  ops, opsValues
  cdef object  condstr, condvars, condcols  ##XXX
  cdef object  mod_elements, colenums

  #def __new__(self, Table table):
    # The MIPSPro C compiler on a SGI does not like to have an assignation
    # of a type Table to a type object. For now, as we do not have to call
    # C methods in Tables, I'll declare table as object.
    # F. Altet 2004-02-11
  def __new__(self, table):
    cdef int nfields, i
    # Location-dependent information.
    self.table = table
    self._unsaved_nrows = table._unsaved_nrows
    self._mod_nrows = 0
    self._row = self._unsaved_nrows
    self._nrow = 0   # Useful in mod_append read iterators
    self._riterator = 0
    self._bufferinfo_done = 0
    self.rbufRA = None
    self.wbufRA = None
    # Some variables from table will be cached here
    if table._v_file.mode == 'r':
      self.ro_filemode = 1
    else:
      self.ro_filemode = 0
    self.chunked = table._chunked
    self.colenums = table._colenums
    self.exist_enum_cols = len(self.colenums)
    self.nrowsinbuf = table._v_maxTuples

  def __call__(self, start=0, stop=0, step=1, coords=None, ncoords=0):
    """ return the row for this record object and update counters"""

    self._initLoop(start, stop, step, coords, ncoords)
    return iter(self)

  def __iter__(self):
    "Iterator that traverses all the data in the Table"

    return self

  cdef _newBuffer(self, write):
    "Create the recarray for I/O buffering"

    table = self.table

    if write:
      # Get the write buffer in table (it is unique, remember!)
      buff = self.wbufRA = table._v_wbuffer
      self._wfields = buff._fields
      # Initialize an array for keeping the modified elements
      # (just in case Row.update() would be used)
      self.mod_elements = numarray.array(None, shape=table._v_maxTuples,
                                         type="Int64")
    else:
      buff = self.rbufRA = table._newBuffer(init=0)
      self._rfields = buff._fields

    # Get info from this buffer
    self._getBufferInfo(buff)
    self.nrows = table.nrows  # This value may change

  cdef _getBufferInfo(self, buff):
    "Get info for __getitem__ and __setitem__ methods"

    if self._bufferinfo_done:
      # The info has been already collected. Giving up.
      return
    self._stride = buff._strides[0]
    nfields = buff._nfields
    # Create a dictionary with the index columns of the recarray
    # and other tables
    i = 0
    self._indexes = {}
    self._scalar = <int *>malloc(nfields * sizeof(int))
    self._enumtypes = <int *>malloc(nfields * sizeof(int))
    for field in buff._names:
      self._indexes[field] = i
      if buff._repeats[i] == 1:
        self._scalar[i] = 1
      else:
        self._scalar[i] = 0
      self._enumtypes[i] = tables.hdf5Extension.naTypeToNAEnum[buff._fmt[i]]
      i = i + 1
    self._bufferinfo_done = 1

  cdef _initLoop(self, hsize_t start, hsize_t stop, hsize_t step,
                     object coords, int ncoords):
    "Initialization for the __iter__ iterator"

    self._riterator = 1   # We are inside a read iterator
    self._newBuffer(False)   # Create a buffer for reading
    self.start = start
    self.stop = stop
    self.step = step
    self.coords = coords
    self.startb = 0
    self.nrowsread = start
    self._nrow = start - self.step
    self.whereCond2XXX = 0
    self.whereCond = 0
    self.indexed2XXX = 0
    self.indexed = 0

    table = self.table
    self.nrows = table.nrows   # Update the row counter
    ##XXX
    if table.whereCondition:
      self.whereCond2XXX = 1
      self.condstr, self.condvars = table.whereCondition
      self.condcols = condcols = []
      for (var, val) in self.condvars.items():
        if hasattr(val, 'pathname'):  # looks like a column
          condcols.append(var)
      table.whereCondition = None
    if table.whereIndex:
      self.indexed2XXX = 1
      self.index = table.cols._f_col(table.whereIndex).index
      self.indices = self.index.indices
      self.nrowsread = 0
      self.nextelement = 0
      table.whereIndex = None
    ##XXX
    # Do we have in-kernel selections?
    if (hasattr(table, "whereColname") and
        table.whereColname is not None):
      self.whereCond = 1
      self.colname = PyString_AsString(table.whereColname)
      # Is this column indexed and ready to use?
      if table.colindexed[self.colname] and ncoords >= 0:
        self.indexed = 1
        self.index = table.cols._f_col(self.colname).index
        self.indices = self.index.indices
        self.nrowsread = 0
        self.nextelement = 0
      # Copy the table conditions to local variable
      self.ops = table.ops[:]
      self.opsValues = table.opsValues[:]
      # Reset the table variable conditions
      table.ops = []
      table.opsValues = []
      table.whereColname = None

    if self.coords is not None:
      self.stopindex = len(coords)
      self.nrowsread = 0
      self.nextelement = 0
    elif self.indexed or self.indexed2XXX:
      self.stopindex = ncoords

  def __next__(self):
    "next() method for __iter__() that is called on each iteration"
    if self.indexed or self.coords is not None:
      return self.__next__indexed()
    elif self.indexed2XXX:
      return self.__next__indexed2XXX()
    elif self.whereCond2XXX:
      return self.__next__inKernel2XXX()
    elif self.whereCond:
      return self.__next__inKernel()
    else:
      return self.__next__general()

  cdef __next__indexed2XXX(self):
    """The version of next() for indexed columns or with user coordinates"""
    cdef int recout
    cdef long long stop
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
          # Optmized version of getCoords in Pyrex
          self.bufcoords = self.indices._getCoords(self.index,
                                                   self.nrowsread, stop)
          nrowsread = len(self.bufcoords)
          tmp = self.bufcoords
          # If a step was specified, select the strided elements first
          if len(tmp) > 0 and self.step > 1:
            tmp2=(tmp-self.start) % self.step
            tmp = tmp[tmp2.__eq__(0)]
          # Now, select those indices in the range start, stop:
          if len(tmp) > 0 and self.start > 0:
            # Pyrex can't use the tmp>=number notation when tmp is a numarray
            # object. Why?
            tmp = tmp[tmp.__ge__(self.start)]
          if len(tmp) > 0 and self.stop < self.nrows:
            tmp = tmp[tmp.__lt__(self.stop)]
          self.bufcoords = tmp
        self._row = -1
        if len(self.bufcoords) > 0:
          recout = self.table._read_elements(self.rbufRA, self.bufcoords)

          if self.whereCond2XXX:
            numexpr_locals = self.condvars.copy()
            # Replace references to columns with the proper array fragment.
            for colvar in self.condcols:
              col = self.condvars[colvar]
              numexpr_locals[colvar] = self._rfields[col.pathname]
            self.indexValid = evaluate(self.condstr, numexpr_locals, {})
          else:
            # No residual condition, all selected rows are valid.
            self.indexValid = numarray.ones(recout, 'Bool')

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
      # Return this row if it fullfills the residual condition
      if self.indexValid[self._row]:
        return self
    else:  ##XXX???
      # Re-initialize the possible cuts in columns
      self.indexed = 0
      if self.coords is None and not self.index.is_pro:
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
        self.finish_riterator()
      else:
        # Continue the iteration with the __next__inKernel2XXX() method
        self.start = nextelement
        self.startb = 0
        self.nrowsread = self.start
        self._nrow = self.start - self.step
        return self.__next__inKernel2XXX()

  cdef __next__indexed(self):
    """The version of next() for indexed columns or with user coordinates"""
    cdef int recout
    cdef long long stop
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
          # Optmized version of getCoords in Pyrex
          self.bufcoords = self.indices._getCoords(self.index,
                                                   self.nrowsread, stop)
          nrowsread = len(self.bufcoords)
          tmp = self.bufcoords
          # If a step was specified, select the strided elements first
          if len(tmp) > 0 and self.step > 1:
            tmp2=(tmp-self.start) % self.step
            tmp = tmp[tmp2.__eq__(0)]
          # Now, select those indices in the range start, stop:
          if len(tmp) > 0 and self.start > 0:
            # Pyrex can't use the tmp>=number notation when tmp is a numarray
            # object. Why?
            tmp = tmp[tmp.__ge__(self.start)]
          if len(tmp) > 0 and self.stop < self.nrows:
            tmp = tmp[tmp.__lt__(self.stop)]
          self.bufcoords = tmp
        self._row = -1
        if len(self.bufcoords) > 0:
          recout = self.table._read_elements(self.rbufRA, self.bufcoords)
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
      if self.coords is None and not self.index.is_pro:
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
        self.finish_riterator()
      else:
        # Continue the iteration with the __next__inKernel() method
        self.start = nextelement
        self.startb = 0
        self.nrowsread = self.start
        self._nrow = self.start - self.step
        return self.__next__inKernel()

  cdef __next__inKernel2XXX(self):
    """The version of next() in case of in-kernel conditions"""
    cdef int recout, correct
    cdef object numexpr_locals, colvar, col

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
        recout = self.table._read_records(self.nextelement, self.nrowsinbuf,
                                          self.rbufRA)
        self.nrowsread = self.nrowsread + recout
        self.indexChunk = -self.step

        numexpr_locals = self.condvars.copy()
        # Replace references to columns with the proper array fragment.
        for colvar in self.condcols:
          col = self.condvars[colvar]
          numexpr_locals[colvar] = self._rfields[col.pathname]
        self.indexValid = evaluate(self.condstr, numexpr_locals, {})

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
      self.finish_riterator()

  cdef __next__inKernel(self):
    """The version of next() in case of in-kernel conditions"""
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
        recout = self.table._read_records(self.nextelement, self.nrowsinbuf,
                                          self.rbufRA)
        self.nrowsread = self.nrowsread + recout
        self.indexChunk = -self.step
        # Iterate over the conditions
        ncond = 0
        for op in self.ops:
          opValue = self.opsValues[ncond]
          # Copying first to a non-strided array, reduces the speed
          # in a factor of 20%
          #field = self._rfields[self.colname].copy()
          if op == 1:
            indexValid1 = self._rfields[self.colname].__lt__(opValue)
          elif op == 2:
            indexValid1 = self._rfields[self.colname].__le__(opValue)
          elif op == 3:
            indexValid1 = self._rfields[self.colname].__gt__(opValue)
          elif op == 4:
            indexValid1 = self._rfields[self.colname].__ge__(opValue)
          elif op == 5:
            indexValid1 = self._rfields[self.colname].__eq__(opValue)
          elif op == 6:
            indexValid1 = self._rfields[self.colname].__ne__(opValue)
          # Consolidate the valid indexes
          if ncond == 0:
            self.indexValid = indexValid1
          else:
            self.indexValid = self.indexValid.__and__(indexValid1)
          ncond = ncond + 1
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
      self.finish_riterator()

  # This is the most general __next__ version, simple, but effective
  cdef __next__general(self):
    """The version of next() for the general cases"""
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
        recout = self.table._read_records(self.nrowsread, self.nrowsinbuf,
                                          self.rbufRA)
        self.nrowsread = self.nrowsread + recout

      self._row = self._row + self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = (self._row + self.step) % self.nrowsinbuf

      self.nextelement = self._nrow + self.step
      # Return this value
      return self
    else:
      self.finish_riterator()

  cdef finish_riterator(self):
    """Clean-up things after iterator has been done"""

    self._riterator = 0        # out of iterator
    if self._mod_nrows > 0:    # Check if there is some modified row
      self._flushModRows()       # Flush any possible modified row
    raise StopIteration        # end of iteration

  def _fillCol(self, result, start, stop, step, field):
    "Read a field from a table on disk and put the result in result"
    cdef hsize_t startr, stopr, i, j, istartb, istopb
    cdef hsize_t istart, istop, istep, inrowsinbuf, inextelement, inrowsread
    cdef object fields

    # We can't reuse existing buffers in this context
    self._initLoop(start, stop, step, None, 0)
    istart, istop, istep = (self.start, self.stop, self.step)
    inrowsinbuf, inextelement, inrowsread = (self.nrowsinbuf, istart, istart)
    istartb, startr = (self.startb, 0)
    # This is commented out until one knows what is happening here. See:
    # https://sourceforge.net/mailarchive/forum.php?thread_id=8428233&forum_id=13760
    # for more info
#     if field:
#       # If field is not None, select it
#       fields = self._rfields[field]
#     else:
#       # if don't, select all fields
#       fields = self.rbufRA
    fields = self.rbufRA
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
      inrowsread = inrowsread + self.table._read_records(i, inrowsinbuf,
                                                         self.rbufRA)
      # Assign the correct part to result
      # The bottleneck is in this assignment. Hope that the numarray
      # people might improve this in the short future
      # As above, see:
      # https://sourceforge.net/mailarchive/forum.php?thread_id=8428233&forum_id=13760
      #result[startr:stopr] = fields[istartb:istopb:istep]
      if field:
        result[startr:stopr] = fields.field(field)[istartb:istopb:istep]
      else:
        result[startr:stopr] = fields[istartb:istopb:istep]

      # Compute some indexes for the next iteration
      startr = stopr
      j = istartb + ((istopb - istartb - 1) / istep) * istep
      istartb = (j+istep) % inrowsinbuf
      inextelement = inextelement + istep
      i = i + inrowsinbuf
    self._riterator = 0  # out of iterator
    return

  # The nrow() method has been converted into a property, which is handier
  property nrow:
    "Makes current row visible from Python space."
    def __get__(self):
      return self._nrow

  def append(self):
    """Append self object to the output buffer."""

    if self.ro_filemode:
      raise IOError("Attempt to write over a file opened in read-only mode")

    if not self.chunked:
      raise HDF5ExtError("You cannot append rows to a non-chunked table.")

    if self._riterator:
      raise NotImplementedError("You cannot append rows when in middle of a table iterator. If what you want is updating records, use Row.update() instead.")

    self._unsaved_nrows = self._unsaved_nrows + 1
    self.table._unsaved_nrows = self._unsaved_nrows
    # When the buffer is full, flush it
    if self._unsaved_nrows == self.nrowsinbuf:
      table = self.table
      # Save the records on disk
      table._saveBufferedRows()
      # Reset the buffer unsaved counter
      self._unsaved_nrows = 0
      # self.table._unsaved_nrows *has* to be reset in table._saveBufferedRows
      # so that flush works well.

  def update(self):
    """Update current row copying it from the input buffer."""

    if self.ro_filemode:
      raise IOError("Attempt to write over a file opened in read-only mode")

    if self.wbufRA is None:
      # Get the array pointers for write buffers
      self._newBuffer(True)

    if not self._riterator:
      raise NotImplementedError("You are only allowed to update rows through the Row.update() method if you are in the middle of a table iterator.")

    # Add this row to the list of elements to be modified
    self.mod_elements[self._mod_nrows] = self._nrow
    # Copy the current buffer row in input to the output buffer
    self.wbufRA[self._mod_nrows] = self.rbufRA[self._row]
    # Increase the modified buffer count by one
    self._mod_nrows = self._mod_nrows + 1
    # When the buffer is full, flush it
    if self._mod_nrows == self.nrowsinbuf:
      self._flushModRows()

  def _flushModRows(self):
    """Flush any possible modified row using Row.update()"""

    table = self.table
    # Save the records on disk
    table._update_elements(self._mod_nrows, self.mod_elements, self.wbufRA)
    # Reset the counter of modified rows to 0
    self._mod_nrows = 0
    # Redo the indexes if needed. This could be optimized if we would
    # be able to track the modified columns.
    table._reIndex(table.colnames)

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
        # Get the column very vast
        offset = self._row * self._stride
        return NA_getPythonScalar(self._rfields[fieldName], offset)
      elif (self._enumtypes[index] == CHARTYPE and self._scalar[index]):
        # Case of a plain string in the cell
        # Call the universal indexing function
        return self._rfields[fieldName][self._row]
      else:  # Case when dimensions > 1
        # Call the universal indexing function
        # Make a copy of the (multi) dimensional array
        # so that the user does not have to do that!
        arr = self._rfields[fieldName][self._row].copy()
        return arr
    except KeyError:
      raise KeyError("no such column: %s" % (fieldName,))

  # This is slightly faster (around 3%) than __setattr__
  def __setitem__(self, fieldName, value):
    cdef int index
    cdef long offset

    if self.ro_filemode:
      raise IOError("attempt to write over a file opened in read-only mode")

    if self.wbufRA is None:
      # Get the array pointers for write buffers
      self._newBuffer(True)

    # Check validity of enumerated value.
    if self.exist_enum_cols:
      if fieldName in self.colenums:
        enum = self.colenums[fieldName]
        cenvals = numarray.array(value).flat
        for cenval in cenvals:
          enum(cenval)  # raises ``ValueError`` on invalid values

    if self._riterator:
      # We are in the middle of an iterator for reading. So the
      # user most probably wants to update this row.
      field = self._rfields[fieldName]
      offset = self._row
    else:
      field = self._wfields[fieldName]
      offset = self._unsaved_nrows
    try:
      #field[offset] = value
      # Optimization for scalar values. This can optimize the writes
      # between a 10% and 100%, depending on the number of columns modified
      # F. Altet 2005-10-25
      index = self._indexes[fieldName]
      if (self._enumtypes[index] <> CHARTYPE and
          (type(value) is not str) and  # to deal with row['intfield'] = 'xx'
          self._scalar[index]):
        NA_setFromPythonScalar(field, offset * self._stride, value)
      else:
        field[offset] = value
      ##### End of optimization for scalar values
    except KeyError:
      raise KeyError("no such column: %s" % (fieldName,))
    except TypeError:
      raise TypeError("invalid type for ``%s`` column: %s" % (fieldName,
                                                              type(value)))
    if not self._riterator:
      # Before write and read buffer got separated, we were able to write:
      # row['var1'] = '%04d' % (self.expectedrows - i)
      # row['var7'] = row['var1'][-1]
      # during table fillings. This is to allow this to continue happening.
      # F. Altet 2005-04-25
      self._rfields = self._wfields
      self._row = self._unsaved_nrows

  def __str__(self):
    """ represent the record as an string """

    # We need to do a cast for recognizing negative row numbers!
    if <signed long long>self._nrow < 0:
      return "Warning: Row iterator has not been initialized for table:\n  %s\n %s" % \
             (self.table, \
    "You will normally want to use this object in iterator contexts.")

    # Create the read buffers
    self._newBuffer(True)
    outlist = []
    # Special case where Row has not been initialized yet
    if self.rbufRA == None:
      return "Warning: Row iterator has not been initialized for table:\n  %s\n %s" % \
             (self.table, \
    "You will normally want to use to use this object in iterator contexts.")
    for name in self.rbufRA._names:
      outlist.append(`self._rfields[name][self._row]`)
    return "(" + ", ".join(outlist) + ")"

  def __repr__(self):
    """ represent the record as an string """
    return str(self)

  def __dealloc__(self):
    #print "Deleting Row object"
    if self._scalar:
      free(<void *>self._scalar)
    if self._enumtypes:
      free(<void *>self._enumtypes)

