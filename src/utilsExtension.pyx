#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: May 20, 2005
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Pyrex utilities for PyTables and HDF5 library.

"""

import sys
import warnings

try:
  import zlib
  zlib_imported = True
except ImportError:
  zlib_imported = False

import numarray
import numarray.records

from tables.exceptions import HDF5ExtError
from tables.enum import Enum
from tables.IsDescription import Description, Col, StringCol, EnumCol, TimeCol

from tables.utils import checkFileAccess

from definitions cimport import_libnumarray, NA_getBufferPtrAndSize, MAXDIM

__version__ = "$Revision$"


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

  # HDF5 constants
  int H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_EXCL
  int H5F_ACC_DEBUG, H5F_ACC_CREAT
  int H5P_DEFAULT, H5S_ALL
  int H5P_FILE_CREATE, H5P_FILE_ACCESS
  int H5FD_LOG_LOC_WRITE, H5FD_LOG_ALL
  int H5I_INVALID_HID

  # HDF5 types
  ctypedef int hid_t
  ctypedef int herr_t
  ctypedef long long hsize_t
  ctypedef int htri_t

  # HDF5 layouts
  ctypedef enum H5D_layout_t:
    H5D_LAYOUT_ERROR    = -1,
    H5D_COMPACT         = 0,    #raw data is very small                     */
    H5D_CONTIGUOUS      = 1,    #the default                                */
    H5D_CHUNKED         = 2,    #slow and fancy                             */
    H5D_NLAYOUTS        = 3     #this one must be last!                     */

  # HDF5 classes.
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

  # Types of integer sign schemes.
  cdef enum H5T_sign_t:
    H5T_SGN_ERROR        = -1,  # error
    H5T_SGN_NONE         = 0,   # this is an unsigned type
    H5T_SGN_2            = 1,   # two's complement
    H5T_NSGN             = 2    # this must be last!

  # The order to retrieve atomic native datatype */
  cdef enum H5T_direction_t:
    H5T_DIR_DEFAULT     = 0,    #default direction is inscendent
    H5T_DIR_ASCEND      = 1,    #in inscendent order
    H5T_DIR_DESCEND     = 2     #in descendent order

  # Native types.
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
    H5T_UNIX_D32BE
    H5T_UNIX_D64BE


  # HDF5 API functions.

  # Operations with files
  hid_t  H5Fopen(char *name, unsigned flags, hid_t access_id)
  herr_t H5Fclose (hid_t file_id)
  htri_t H5Fis_hdf5(char *name)

  # Operations with groups
  hid_t  H5Gopen(hid_t loc_id, char *name )
  herr_t H5Gclose(hid_t group_id)

  # Operations for datasets
  hid_t  H5Dopen (hid_t file_id, char *name)
  herr_t H5Dclose (hid_t dset_id)
  hid_t H5Dget_type (hid_t dset_id)

  # Operations defined on all data types.
  hid_t H5Tcreate(H5T_class_t type, size_t size)
  hid_t H5Tcopy(hid_t type_id)
  herr_t H5Tclose(hid_t type_id)

  # Operations defined on compound data types.
  int   H5Tget_nmembers(hid_t type_id)
  char  *H5Tget_member_name(hid_t type_id, unsigned membno)
  hid_t  H5Tget_member_type(hid_t type_id, unsigned membno)
  hid_t  H5Tget_native_type(hid_t type_id, H5T_direction_t direction)
  herr_t H5Tget_member_value(hid_t type_id, int membno, void *value)
  size_t H5Tget_size(hid_t type_id)
  H5T_class_t H5Tget_class(hid_t type_id)
  hid_t H5Tget_super(hid_t type)
  H5T_sign_t H5Tget_sign(hid_t type_id)
  int H5Tget_offset(hid_t type_id)
  herr_t H5Tinsert(hid_t parent_id, char *name, size_t offset,
                   hid_t member_id)

  # Operations defined on enumeration data types.
  hid_t H5Tenum_create(hid_t base_id)
  herr_t H5Tenum_insert(hid_t type, char *name, void *value)

  # Operations defined on array data types.
  hid_t H5Tarray_create(hid_t base_id, int ndims, hsize_t dims[], int perm[])
  int   H5Tget_array_ndims(hid_t type_id)
  int   H5Tget_array_dims(hid_t type_id, hsize_t dims[], int perm[])

  # Operations defined on string data types.
  htri_t H5Tis_variable_str(hid_t dtype_id)

  # Setting property values.
  herr_t H5Tset_size(hid_t type_id, size_t size)
  herr_t H5Tset_precision(hid_t type_id, size_t prec)


# CharArray type
CharType = numarray.records.CharType

# Conversion from Numarrray string types to HDF5 native types.
# Put here types that are susceptible of changing byteorder
naSTypeToH5Type = {
  'Int8': H5T_NATIVE_SCHAR,    'UInt8': H5T_NATIVE_UCHAR,
  'Int16': H5T_NATIVE_SHORT,   'UInt16': H5T_NATIVE_USHORT,
  'Int32': H5T_NATIVE_INT,     'UInt32': H5T_NATIVE_UINT,
  'Int64': H5T_NATIVE_LLONG,   'UInt64': H5T_NATIVE_ULLONG,
  'Float32': H5T_NATIVE_FLOAT, 'Float64': H5T_NATIVE_DOUBLE,
  'Time32': H5T_UNIX_D32BE,    'Time64': H5T_UNIX_D64BE }

# Special cases that cannot be directly mapped:
ptSpecialTypes = ['Bool', 'Complex32', 'Complex64', 'CharType', 'Enum']

cdef extern from "H5ATTR.h":

  herr_t H5ATTR_get_attribute_disk(hid_t loc_id, char *attr_name,
                                   void *attr_out)
  herr_t H5ATTR_find_attribute(hid_t loc_id, char *attr_name )


cdef extern from "H5ARRAY.h":

  herr_t H5ARRAYget_ndims( hid_t dataset_id, hid_t type_id, int *rank )

  herr_t H5ARRAYget_info( hid_t dataset_id, hid_t type_id, hsize_t *dims,
                          hsize_t *maxdims, hid_t *super_type_id,
                          H5T_class_t *super_class_id, char *byteorder)



# PyTables helper routines.
cdef extern from "utils.h":

  int getLibrary(char *libname)
  object _getTablesVersion()
  #object getZLIBVersionInfo()
  object getHDF5VersionInfo()
  hid_t  create_native_complex64(char *byteorder)
  hid_t  create_native_complex32(char *byteorder)
  int    is_complex(hid_t type_id)
  herr_t set_order(hid_t type_id, char *byteorder)
  herr_t get_order(hid_t type_id, char *byteorder)
  object get_filter_names( hid_t loc_id, char *dset_name)

  H5T_class_t getHDF5ClassID(hid_t loc_id, char *name, H5D_layout_t *layout,
                             hid_t *type_id, hid_t *dataset_id)

  # To access to the slice.indices functionality for long long ints
  hsize_t getIndicesExt(object s, hsize_t length,
                        hsize_t *start, hsize_t *stop, hsize_t *step,
                        hsize_t *slicelength)


# Type conversion routines
cdef extern from "typeconv.h":
  void conv_float64_timeval32(void *base,
                              unsigned long byteoffset,
                              unsigned long bytestride,
                              long long nrecords,
                              unsigned long nelements,
                              int sense)

  void conv_space_null(void *base,
                       unsigned long byteoffset,
                       unsigned long bytestride,
                       long long nrecords,
                       unsigned long nelements,
                       unsigned long itemsize,
                       int sense)

cdef extern from "arraytypes.h":
  # Function to compute the HDF5 type from a Numarray enum type.
  hid_t convArrayType(int fmt, size_t size, char *byteorder)
  size_t getArrayType(hid_t type_id, int *fmt)


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


# Names of HDF5 classes.
hdf5ClassToString = {
  H5T_NO_CLASS:   'H5T_NO_CLASS',
  H5T_INTEGER:    'H5T_INTEGER',
  H5T_FLOAT:      'H5T_FLOAT',
  H5T_TIME:       'H5T_TIME',
  H5T_STRING:     'H5T_STRING',
  H5T_BITFIELD:   'H5T_BITFIELD',
  H5T_OPAQUE:     'H5T_OPAQUE',
  H5T_COMPOUND:   'H5T_COMPOUND',
  H5T_REFERENCE:  'H5T_REFERENCE',
  H5T_ENUM:       'H5T_ENUM',
  H5T_VLEN:       'H5T_VLEN',
  H5T_ARRAY:      'H5T_ARRAY',
  H5T_NCLASSES:   'H5T_NCLASSES'}

# Conversion tables from/to classes to the numarray enum types
naEnumToNAType = {
  tBool: numarray.Bool,  # Boolean type added
  tInt8: numarray.Int8,    tUInt8: numarray.UInt8,
  tInt16: numarray.Int16,  tUInt16: numarray.UInt16,
  tInt32: numarray.Int32,  tUInt32: numarray.UInt32,
  tInt64: numarray.Int64,  tUInt64: numarray.UInt64,
  tFloat32: numarray.Float32,  tFloat64: numarray.Float64,
  tComplex32: numarray.Complex32,  tComplex64: numarray.Complex64,
  # Special cases:
  ord('a'): CharType,  # For strings.
  ord('t'): numarray.Int32,  ord('T'): numarray.Float64}  # For times.

naTypeToNAEnum = {
  numarray.Bool: tBool,
  numarray.Int8: tInt8,    numarray.UInt8: tUInt8,
  numarray.Int16: tInt16,  numarray.UInt16: tUInt16,
  numarray.Int32: tInt32,  numarray.UInt32: tUInt32,
  numarray.Int64: tInt64,  numarray.UInt64: tUInt64,
  numarray.Float32: tFloat32,  numarray.Float64: tFloat64,
  numarray.Complex32: tComplex32,  numarray.Complex64: tComplex64,
  # Special cases:
  numarray.records.CharType: ord('a')}  # For strings.


#----------------------------------------------------------------------------

# Initialization code

# The numarray API requires this function to be called before
# using any numarray facilities in an extension module.
import_libnumarray()

#---------------------------------------------------------------------

# Main functions

if sys.platform == "win32":
  # We need a different approach in Windows, because it compains when
  # trying to import the extension that is linked with a dynamic library
  # that is not installed in the system.

  # Initialize & register lzo
  if getLibrary("lzo2") == 0 or getLibrary("lzo1") == 0:
    import tables._comp_lzo
    lzo_version = tables._comp_lzo.register_()
  else:
    lzo_version = None

  # Initialize & register ucl
  if getLibrary("ucl1") == 0:
    import tables._comp_ucl
    ucl_version = tables._comp_ucl.register_()
  else:
    ucl_version = None

  # Initialize & register bzip2
  if getLibrary("bzip2") == 0:
    import tables._comp_bzip2
    bzip2_version = tables._comp_bzip2.register_()
  else:
    bzip2_version = None

else:  # Unix systems
  # Initialize & register lzo
  try:
    import tables._comp_lzo
    lzo_version = tables._comp_lzo.register_()
  except ImportError:
    lzo_version = None

  # Initialize & register ucl
  try:
    import tables._comp_ucl
    ucl_version = tables._comp_ucl.register_()
  except ImportError:
    ucl_version = None

  # Initialize & register bzip2
  try:
    import tables._comp_bzip2
    bzip2_version = tables._comp_bzip2.register_()
  except ImportError:
    bzip2_version = None

# utility funtions (these can be directly invoked from Python)

# def PyNextAfter(double x, double y):
#   return nextafter(x, y)

# def PyNextAfterF(float x, float y):
#   return nextafterf(x, y)

def isHDF5(char *filename):
  warnings.warn(DeprecationWarning("""\
``isHDF5()`` is deprecated; please use ``isHDF5File()``"""),
                stacklevel=2)
  return isHDF5File(filename)


def isHDF5File(char *filename):
  """isHDF5File(filename) -> bool
  Determine whether a file is in the HDF5 format.

  When successful, it returns a true value if the file is an HDF5
  file, false otherwise.  If there were problems identifying the file,
  an `HDF5ExtError` is raised.
  """

  # Check that the file exists and is readable.
  checkFileAccess(filename)

  ret = H5Fis_hdf5(filename)
  if ret < 0:
    raise HDF5ExtError("problems identifying file ``%s``" % (filename,))
  return ret > 0


def isPyTablesFile(char *filename):
  """isPyTablesFile(filename) -> true or false value
  Determine whether a file is in the PyTables format.

  When successful, it returns a true value if the file is a PyTables
  file, false otherwise.  The true value is the format version string
  of the file.  If there were problems identifying the file, an
  `HDF5ExtError` is raised.
  """

  cdef hid_t file_id

  isptf = "unknown"
  if isHDF5File(filename):
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

def getPyTablesVersion():
  """Return this extension version."""

  return _getTablesVersion()

def whichLibVersion(char *name):
  """whichLibVersion(name) -> version info
  Get version information about a C library.

  If the library indicated by `name` is available, this function returns
  a 3-tuple containing the major library version as an integer, its full
  version as a string, and the version date as a string.  If the library
  is not available, ``None`` is returned.

  The currently supported library names are ``hdf5``, ``zlib``, ``lzo``,
  ``ucl`` and ``bzip2``.  If another name is given, a ``ValueError`` is
  raised.
  """

  libnames = ('hdf5', 'zlib', 'lzo', 'ucl', 'bzip2')

  if strcmp(name, "hdf5") == 0:
    binver, strver = getHDF5VersionInfo()
    return (binver, strver, None)     # Should be always available
  elif strcmp(name, "zlib") == 0:
    if zlib_imported:
      return (1, zlib.ZLIB_VERSION, None)
  elif strcmp(name, "lzo") == 0:
    if lzo_version:
      (lzo_version_string, lzo_version_date) = lzo_version
      return (lzo_version, lzo_version_string, lzo_version_date)
  elif strcmp(name, "ucl") == 0:
    if ucl_version:
      (ucl_version_string, ucl_version_date) = ucl_version
      return (ucl_version, ucl_version_string, ucl_version_date)
  elif strcmp(name, "bzip2") == 0:
    if bzip2_version:
      (bzip2_version_string, bzip2_version_date) = bzip2_version
      return (bzip2_version, bzip2_version_string, bzip2_version_date)
  else:
    raise ValueError("""\
asked version of unsupported library ``%s``; \
supported library names are ``%s``""" % (name, libnames))

  # A supported library was specified, but no version is available.
  return None


def whichClass( hid_t loc_id, char *name):
  """Detects a class ID using heuristics.
  """
  cdef H5T_class_t  class_id
  cdef H5D_layout_t layout
  cdef hsize_t      nrecords_orig, nfields
  cdef char         *field_name1, *field_name2
  cdef int          i
  cdef hid_t        type_id, dataset_id, type_id2
  cdef object       classId
  cdef int          rank
  cdef hsize_t      *dims, *maxdims
  cdef char         byteorder[16]  # "non-relevant" fits easily here

  classId = "UNSUPPORTED"  # default value
  # Get The HDF5 class for the dattatype in this dataset
  class_id = getHDF5ClassID(loc_id, name, &layout, &type_id, &dataset_id)
  # Check if this a dataset of supported classtype for ARRAY
  if class_id == H5T_ARRAY:
    warnings.warn("""\
Dataset object '%s' contains unsupported H5T_ARRAY datatypes.""" % (name,))
  if  ((class_id == H5T_INTEGER)  or
       (class_id == H5T_FLOAT)    or
       (class_id == H5T_BITFIELD) or
       (class_id == H5T_TIME)     or
       (class_id == H5T_ENUM)     or
       (class_id == H5T_STRING)):
    if layout == H5D_CHUNKED:
      if H5ARRAYget_ndims(dataset_id, type_id, &rank) < 0:
        raise HDF5ExtError("Problems getting ndims!")
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      maxdims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      if H5ARRAYget_info(dataset_id, type_id, dims, maxdims,
                         &type_id2, &class_id, byteorder) < 0:
        raise HDF5ExtError("Unable to get array info.")
      else:
        H5Tclose(type_id2)
      classId = "CARRAY"
      # Check whether some dimension is enlargeable
      for i in range(rank):
        if maxdims[i] == -1:
          classId = "EARRAY"
          break
      free(<void *>dims)
      free(<void *>maxdims)
    else:
      classId = "ARRAY"

  if class_id == H5T_COMPOUND:
    # check whether the type is complex or not
    iscomplex = False
    nfields = H5Tget_nmembers(type_id)
    if nfields == 2:
      field_name1 = H5Tget_member_name(type_id, 0)
      field_name2 = H5Tget_member_name(type_id, 1)
      # The pair ("r", "i") is for PyTables. ("real", "imag") for Octave.
      if ( (strcmp(field_name1, "real") == 0 and
            strcmp(field_name2, "imag") == 0) or
           (strcmp(field_name1, "r") == 0 and
            strcmp(field_name2, "i") == 0) ):
        iscomplex = True
      free(<void *>fieldname1)
      free(<void *>fieldname2)
    if layout == H5D_CHUNKED:
      if iscomplex:
        classId = "CARRAY"
      else:
        classId = "TABLE"
    else:  # Not chunked case
      # Octave saves complex arrays as non-chunked tables
      # with two fields: "real" and "imag"
      # Francesc Altet 2005-04-29
      # Get number of records
      if iscomplex:
        classId = "ARRAY"  # It is probably an Octave complex array
      else:
        # Added to support non-chunked tables
        classId = "TABLE"  # A test for supporting non-growable tables

  if class_id == H5T_VLEN:
    if layout == H5D_CHUNKED:
      classId = "VLARRAY"

  # Release the datatype.
  H5Tclose(type_id)

  # Close the dataset.
  H5Dclose(dataset_id)

  # Fallback
  return classId


def getIndices(object s, hsize_t length):
  cdef hsize_t start, stop, step, slicelength

  if getIndicesExt(s, length, &start, &stop, &step, &slicelength) < 0:
    raise ValueError("Problems getting the indices on slice '%s'" % s)
  return (start, stop, step)


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
  if H5ATTR_find_attribute(root_id, attr_name):
    # Read the format_version attribute
    ret = H5ATTR_get_attribute_disk(root_id, attr_name, attr_value)
    if ret < 0:
      strcpy(attr_value, "unknown")

  # Close root group
  H5Gclose(root_id)

  return attr_value


def getFilters(parent_id, name):
  "Get a dictionary with the filter names and cd_values"
  return get_filter_names(parent_id, name)


# This is used by several <Leaf>._convertTypes() methods.
def convertTime64(object naarr, hsize_t nrecords, int sense):
  """Converts a NumArray of Time64 elements between Numarray and HDF5 formats.

  Numarray to HDF5 conversion is performed when 'sense' is 0.
  Otherwise, HDF5 to Numarray conversion is performed.
  The conversion is done in place, i.e. 'naarr' is modified.
  """

  cdef void *t64buf
  cdef long buflen, byteoffset, bytestride, nelements

  byteoffset = naarr._byteoffset
  bytestride = naarr._strides[0]  # supports multi-dimensional RecArray
  nelements = naarr.size() / len(naarr)
  buflen = NA_getBufferPtrAndSize(naarr._data, 1, &t64buf)

  conv_float64_timeval32(
    t64buf, byteoffset, bytestride, nrecords, nelements, sense)


def space2null(object naarr, hsize_t nrecords, int sense):
  """Converts a the space padding of CharArray object into null's.

  Numarray to HDF5 conversion is performed when 'sense' is 0.
  Otherwise, HDF5 to Numarray conversion is performed.
  The conversion is done in place, i.e. 'naarr' is modified.
  """

  cdef void *strbuf
  cdef long buflen, byteoffset, bytestride, nelements, itemsize

  byteoffset = naarr._byteoffset
  bytestride = naarr._strides[0]  # supports multi-dimensional RecArray
  nelements = naarr.size() / len(naarr)
  itemsize = naarr.itemsize()
  buflen = NA_getBufferPtrAndSize(naarr._data, 1, &strbuf)

  conv_space_null(
    strbuf, byteoffset, bytestride, nrecords, nelements, itemsize, sense)


def getLeafHDF5Type(hid_t parentId, char *name):
  """_getLeafHDF5Type(parentId, name) -> hid_t
  Get the HDF5 type of a leaf object.

  The leaf called `name` under the `parentId` HDF5 group must be
  closed, and it is guaranteed to be closed again when the function
  ends (be it successfully or with an `HDF5ExtError`.  The HDF5 type
  (be it simple or compound) of the leaf is returned.
  """

  cdef hid_t dsetId, typeId

  dsetId = H5Dopen(parentId, name)
  if dsetId < 0:
    raise HDF5ExtError("failed to open HDF5 dataset")

  try:
    typeId = H5Dget_type(dsetId)
    if typeId < 0:
      raise HDF5ExtError("failed to get type of HDF5 dataset")
    return typeId
  finally:
    # (Yes, the ``finally`` clause *is* executed.)
    if H5Dclose(dsetId) < 0:
      raise HDF5ExtError("failed to close HDF5 dataset")


def getTypeEnum(hid_t h5type):
  """_getTypeEnum(h5type) -> hid_t
  Get the HDF5 enumerated type of `h5type`.

  If `h5type` is an enumerated type, it is returned.  If it is a
  multi-dimensional type with an enumerated base type, this is returned.
  Else, a ``TypeError`` is raised.
  """

  cdef H5T_class_t typeClass
  cdef hid_t enumId, enumId2

  typeClass = H5Tget_class(h5type)
  if typeClass < 0:
    raise HDF5ExtError("failed to get class of HDF5 type")

  if typeClass == H5T_ENUM:
    # Get the native type (in order to do byteorder conversions automatically)
    enumId = H5Tget_native_type(h5type, H5T_DIR_DEFAULT)
  elif typeClass == H5T_ARRAY:
    # The field is multi-dimensional.
    enumId2 = H5Tget_super(h5type)
    if enumId2 < 0:
      raise HDF5ExtError("failed to get base type of HDF5 type")
    enumId = H5Tget_native_type(enumId2, H5T_DIR_DEFAULT)
    H5Tclose(enumId2)
  else:
    raise TypeError(
      "enumerated values can not be stored using the given type")
  return enumId


def enumFromHDF5(hid_t enumId):
  """_enumFromHDF5(enumId) -> (Enum, naType)
  Convert an HDF5 enumerated type to a PyTables one.

  This function takes an HDF5 enumerated type and returns an `Enum`
  instance built from that, and the Numarray type used to encode it.
  """

  cdef hid_t  baseId
  cdef int    nelems, naEnum, i
  cdef size_t typeSize
  cdef void  *valueAddr
  cdef char  *ename

  # Find the base type of the enumerated type.
  baseId = H5Tget_super(enumId)
  if baseId < 0:
    raise HDF5ExtError("failed to get base type of HDF5 enumerated type")

  # Get the corresponding Numarray type and create temporary value.
  if getArrayType(baseId, &naEnum) < 0:
    raise HDF5ExtError("failed to convert HDF5 base type to numarray type")
  if H5Tclose(baseId) < 0:
    raise HDF5ExtError("failed to close HDF5 base type")

  try:
    naType = naEnumToNAType[naEnum]
  except KeyError:
    raise NotImplementedError("""\
sorry, only scalar concrete values are supported for the moment""")
  if not isinstance(naType, numarray.IntegralType):
    raise NotImplementedError("""\
sorry, only integer concrete values are supported for the moment""")

  naValue = numarray.array((0,), type=naType)
  NA_getBufferPtrAndSize(naValue._data, 0, &valueAddr)

  # Get the name and value of each of the members
  # and put the pair in `enumDict`.
  enumDict = {}

  nelems = H5Tget_nmembers(enumId)
  if enumId < 0:
    raise HDF5ExtError(
      "failed to get element count of HDF5 enumerated type")

  for i from 0 <= i < nelems:
    ename = H5Tget_member_name(enumId, i)
    if ename == NULL:
      raise HDF5ExtError(
        "failed to get element name from HDF5 enumerated type")
    pyename = ename
    free(ename)

    if H5Tget_member_value(enumId, i, valueAddr) < 0:
      raise HDF5ExtError(
        "failed to get element value from HDF5 enumerated type")

    enumDict[pyename] = naValue[0]  # converted to Python scalar

  # Build an enumerated type from `enumDict` and return it.
  return Enum(enumDict), naType


def enumToHDF5(object enumCol, char *byteOrder):
  """enumToHDF5(enumCol, byteOrder) -> hid_t
  Convert a PyTables enumerated type to an HDF5 one.

  This function creates an HDF5 enumerated type from the information
  contained in `enumCol` (a ``Col`` object), with the specified
  `byteOrder` (a string).  The resulting HDF5 enumerated type is
  returned.
  """

  cdef int    naEnum
  cdef size_t itemSize
  cdef char  *name
  cdef hid_t  baseId, enumId
  cdef int    stride, i
  cdef void  *rbuffer, *valueAddr

  # Get the base HDF5 type and create the enumerated type.
  naEnum = naTypeToNAEnum[enumCol.type]
  itemSize = enumCol.itemsize
  baseId = convArrayType(naEnum, itemSize, byteOrder)
  if baseId < 0:
    raise HDF5ExtError("failed to convert Numarray base type to HDF5")

  try:
    enumId = H5Tenum_create(baseId)
    if enumId < 0:
      raise HDF5ExtError("failed to create HDF5 enumerated type")
  finally:
    if H5Tclose(baseId) < 0:
      raise HDF5ExtError("failed to close HDF5 base type")

  # Set the name and value of each of the members.
  naNames = enumCol._naNames
  naValues = enumCol._naValues
  stride = naValues._strides[0]
  NA_getBufferPtrAndSize(naValues._data, 1, &rbuffer)
  for i from 0 <= i < len(naNames):
    name = PyString_AsString(naNames[i])
    valueAddr = <void *>(<char *>rbuffer + stride * i)
    if H5Tenum_insert(enumId, name, valueAddr) < 0:
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")
      raise HDF5ExtError("failed to insert value into HDF5 enumerated type")

  # Return the new, open HDF5 enumerated type.
  return enumId


def conv2HDF5Type(object col, char *byteorder):
  cdef hid_t   tid
  cdef int     rank, atomic
  cdef hsize_t *dims

  if isinstance(col.shape, tuple):
    rank = len(col.shape)
  else:
    rank = 1
  if isinstance(col.shape, int):
    atomic = 1
    dims = NULL
  else:
    atomic = 0
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < rank:
      dims[i] = col.shape[i]
  # Create the column type
  if col.stype in naSTypeToH5Type:
    if atomic == 1:
      tid = H5Tcopy(naSTypeToH5Type[col.stype])
    else:
      tid = H5Tarray_create(naSTypeToH5Type[col.stype], rank, dims, NULL)
    # All types in naSTypeToH5Type needs to fix the byte order
    # but this may change in the future!
    set_order(tid, byteorder)
  elif col.stype in ptSpecialTypes:
    # Special cases
    if col.stype == 'Bool':
      tid = H5Tcopy(H5T_NATIVE_B8)
      H5Tset_precision(tid, 1)
    elif col.stype == 'Complex32':
      tid = create_native_complex32(byteorder)
    elif col.stype == 'Complex64':
      tid = create_native_complex64(byteorder)
    elif col.stype == 'CharType':
      tid = H5Tcopy(H5T_C_S1);
      H5Tset_size(tid, col.itemsize)  #.itemsize is the length of the arrays
    elif col.stype == 'Enum':
      tid = enumToHDF5(col, byteorder)
    if not atomic:
      tid2 = H5Tarray_create(tid, rank, dims, NULL)
      H5Tclose(tid)
      tid = tid2
  else:
    raise TypeError("Invalid type for column %s: %s" % (col._v_name, col.type))


  # Release resources
  if dims:
    free(dims)

  return tid

def createNestedType(object desc, char *byteorder):
  """Create a nested type based on a description and return an HDF5 type."""
  cdef hid_t   tid, tid2
  cdef herr_t  ret
  cdef size_t  offset, totalshape

  tid = H5Tcreate (H5T_COMPOUND, desc._v_totalsize)
  if tid < 0:
    return -1;

  offset = 0
  for k in desc._v_names:
    obj = desc._v_colObjects[k]
    if isinstance(obj, Description):
      tid2 = createNestedType(obj, byteorder)
    else:
      tid2 = conv2HDF5Type(obj, byteorder)
    ret = H5Tinsert(tid, k, offset, tid2)
    offset = offset + desc._v_totalsizes[k]
    # Release resources
    H5Tclose(tid2)

  return tid

def getRAType(hid_t type_id, int klass, size_t size):
  """Map the atomic type to a NumArray format.

  This follows the standard size and alignment.

  Return the string repr of type and the shape.
  """
  cdef H5T_sign_t  sign
  cdef hid_t       super_type_id
  cdef int         super_klass
  cdef size_t      super_size
  cdef object      stype, shape, shape2
  cdef hsize_t     dims[MAXDIM]

  # default shape
  shape = 1

  if klass == H5T_BITFIELD:
    stype = "b1"
  elif klass ==  H5T_INTEGER:
    # Get the sign
    sign = H5Tget_sign(type_id)
    if (sign):
      stype = "i%s" % (size)
    else:
      stype = "u%s" % (size)
  elif klass ==  H5T_FLOAT:
    stype = "f%s" % (size)
  elif klass ==  H5T_COMPOUND:
    # Here, this can only be a complex
    stype = "c%s" % (size)
  elif klass ==  H5T_STRING:
    if H5Tis_variable_str(type_id):
      raise TypeError("variable length strings are not supported yet")
    stype = "a%s" % (size)
  elif klass ==  H5T_TIME:
    stype = "t%s" % (size)
  elif klass ==  H5T_ENUM:
    stype = "e"
  elif klass ==  H5T_ARRAY:
    # Get the array base component
    super_type_id = H5Tget_super(type_id)
    # Get the class
    super_klass = H5Tget_class(super_type_id)
    # Get the size
    super_size = H5Tget_size(super_type_id)
    # Find the super member format
    stype, shape2 = getRAType(super_type_id, super_klass, super_size)
    # Get shape
    shape = []
    ndims = H5Tget_array_ndims(type_id)
    H5Tget_array_dims(type_id, dims, NULL)
    for i from 0 <= i < ndims:
      shape.append(<int>dims[i])  # cast to avoid long representation (i.e. 2L)
    shape = tuple(shape)
    # Release resources
    H5Tclose(super_type_id)
  else:
    # Other types are not supported yet
    raise TypeError("the HDF5 class ``%s`` is not supported yet"
                    % hdf5ClassToString[klass])

  return stype, shape

def _joinPath(object parent, object name):
  if parent == "":
    return name
  else:
    return parent + '/' + name

def getNestedType(hid_t type_id, hid_t native_type_id,
                  object table, object colpath=""):
  """Open a nested type and return a nested dictionary as description."""
  cdef hid_t   member_type_id, native_member_type_id
  cdef hsize_t nfields, dims[1]
  cdef size_t  itemsize, type_size
  cdef int     i, tsize
  cdef char    *colname
  cdef H5T_class_t  klass
  cdef object  format, stype, shape, desc, colobj, colpath2
  cdef char    byteorder[16], byteorder2[16]  # "non-relevant" fits easily here
  cdef object  sysbyteorder
  cdef herr_t  ret

  sysbyteorder = sys.byteorder  # a workaround against temporary Pyrex error
  strcpy(byteorder, sysbyteorder)  # default byteorder
  offset = 0
  desc = {}
  # Get the number of members
  nfields = H5Tget_nmembers(type_id)
  type_size = H5Tget_size(type_id)
  # Iterate thru the members
  for i from 0 <= i < nfields:
      # Get the member name
      colname = H5Tget_member_name(type_id, i)
      # Get the member type
      member_type_id = H5Tget_member_type(type_id, i)
      # Get the member size
      itemsize = H5Tget_size(member_type_id)
      # Get the HDF5 class
      klass = H5Tget_class(member_type_id)
      if klass == H5T_COMPOUND and not is_complex(member_type_id):
        colpath2 = _joinPath(colpath, colname)
        # Create the native data in-memory
        native_member_type_id = H5Tcreate(H5T_COMPOUND, itemsize)
        desc[colname] = getNestedType(member_type_id, native_member_type_id,
                                      table, colpath2)
        desc[colname]["_v_pos"] = i  # Remember the position
      else:
        # Get the member format
        try:
          colstype, colshape = getRAType(member_type_id, klass, itemsize)
        except TypeError, te:
          # Re-raise TypeError again with more info
          raise TypeError(
            ("table ``%s``, column ``%s``: %%s" % (table.name, colname))
            % te.args[0])
        # Get the native type
        if colstype in ["b1", "t4", "t8"]:
          # This types are not supported yet
          native_member_type_id = H5Tcopy(member_type_id)
        else:
          native_member_type_id = H5Tget_native_type(member_type_id,
                                                     H5T_DIR_DEFAULT)
        # Create the Col object.
        # Indexes will be treated later on, in Table._open()
        if colstype[0] == 'a':
          tsize = int(colstype[1:])
          colobj = StringCol(length = tsize, shape = colshape, pos = i)
        elif colstype == 'e':
          colpath2 = _joinPath(colpath, colname)
          (enum, naType) = table._loadEnum(native_member_type_id)
          # Take one of the names as the default in the enumeration.
          # Can this get harmful?  Keep an eye on user's reactions.
          # ivb -- 2005-05-12
          dflt = iter(enum).next()[0]
          colobj = EnumCol(enum, dflt, dtype = naType, shape = colshape,
                           pos = i)
        elif colstype[0] == 't':
          tsize = int(colstype[1:])
          colobj = TimeCol(itemsize = tsize, shape = colshape, pos = i)
        else:
          colobj = Col(dtype = colstype, shape = colshape, pos = i)
        desc[colname] = colobj
        # If *any* column has a different byteorder than sys, it is
        # changed here. This should be further refined for columns
        # with different byteorders, but this case is strange enough.
        ret = get_order(member_type_id, byteorder2)
        if ret > 0 and byteorder2 in ["big", "little"]:  # exclude non-relevant
          strcpy(byteorder, byteorder2)

      # Insert the native member
      H5Tinsert(native_type_id, colname, offset, native_member_type_id)
      # Update the offset
      offset = offset + itemsize
      # Release resources
      H5Tclose(native_member_type_id)
      H5Tclose(member_type_id)
      free(colname)

  # set the byteorder (just in top level)
  if colpath == "":
    desc["_v_byteorder"] = byteorder
  # return the Description object
  return desc

# This is almost 4x times faster than its Python counterpart in lrucache.py
cdef class _g_Node:
  """Record of a cached value. Not for public consumption."""
  cdef object key_, obj_
  cdef int    atime_
        
  def __init__(self, key, obj, timestamp):
    object.__init__(self)
    self.key_ = key
    self.obj_ = obj
    self.atime_ = timestamp
	    
  def __cmp__(self, _g_Node other):
    cdef int atime, atime2

    #return cmp(self.atime, other.atime)
    # This optimization makes the comparison more than twice as faster
    atime = self.atime_
    atime2 = other.atime_
    if atime < atime2:
      return -1
    elif atime > atime2:
      return 1
    else:
      return 0

  property atime:
    "The sorting attribute."

    def __get__(self):
      return self.atime_

    def __set__(self, value):
      self.atime_ = value

  property obj:
    "The object kept."

    def __get__(self):
      return self.obj_

    def __set__(self, value):
      self.obj_ = value

  property key:
    "The key for object."

    def __get__(self):
      return self.key_

  def __repr__(self):
    return "<%s %s => %s (accessed at %s)>" % \
           (self.__class__, self.key_, self.obj_, self.atime_)
