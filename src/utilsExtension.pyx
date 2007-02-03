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

import numpy

from tables.misc.enum import Enum
from tables.exceptions import HDF5ExtError
from tables.atom import Atom
from tables.description import Description, EnumCol, Col

from tables.utils import checkFileAccess

from definitions cimport import_array, ndarray, \
     malloc, free, strchr, strcpy, strncpy, strcmp, strdup, \
     PyString_AsString, PyString_FromString, \
     H5F_ACC_RDONLY, H5P_DEFAULT, H5D_CHUNKED, H5T_DIR_DEFAULT, \
     size_t, hid_t, herr_t, hsize_t, htri_t, \
     H5T_class_t, H5D_layout_t, H5T_sign_t, \
     H5Fopen, H5Fclose, H5Fis_hdf5, H5Gopen, H5Gclose, \
     H5Dopen, H5Dclose, H5Dget_type, H5Tcreate, H5Tcopy, H5Tclose, \
     H5Tget_nmembers, H5Tget_member_name, H5Tget_member_type, \
     H5Tget_native_type, H5Tget_member_value, H5Tget_size, \
     H5Tget_class, H5Tget_super, H5Tget_sign, H5Tget_offset, \
     H5Tinsert, H5Tenum_create, H5Tenum_insert, H5Tarray_create, \
     H5Tget_array_ndims, H5Tget_array_dims, H5Tis_variable_str, \
     H5Tset_size, H5Tset_precision, \
     H5ATTRget_attribute_string, H5ATTRfind_attribute, \
     H5ARRAYget_ndims, H5ARRAYget_info, \
     create_ieee_complex64, create_ieee_complex128, \
     convArrayType, getArrayType, get_order, set_order



# Include conversion tables & type
include "convtypetables.pxi"

__version__ = "$Revision$"


#----------------------------------------------------------------------

# External declarations


# PyTables helper routines.
cdef extern from "utils.h":

  int getLibrary(char *libname)
  object _getTablesVersion()
  #object getZLIBVersionInfo()
  object getHDF5VersionInfo()
  int    is_complex(hid_t type_id)
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



#----------------------------------------------------------------------
# Initialization code

# The NumPy API requires this function to be called before
# using any NumPy facilities in an extension module.
import_array()

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

  # Initialize & register bzip2
  try:
    import tables._comp_bzip2
    bzip2_version = tables._comp_bzip2.register_()
  except ImportError:
    bzip2_version = None


# End of initialization code
#---------------------------------------------------------------------


# Main functions

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

  isptf = None    # A PYTABLES_FORMAT_VERSION attribute was not found
  if isHDF5File(filename):
    # The file exists and is HDF5, that's ok
    # Open it in read-only mode
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT)
    isptf = read_f_attr(file_id, 'PYTABLES_FORMAT_VERSION')
    # Close the file
    H5Fclose(file_id)

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

  If the library indicated by `name` is available, this function returns a
  3-tuple containing the major library version as an integer, its full version
  as a string, and the version date as a string.  If the library is not
  available, ``None`` is returned.

  The currently supported library names are ``hdf5``, ``zlib``, ``lzo``, and
  ``bzip2``.  If another name is given, a ``ValueError`` is raised.
  """

  libnames = ('hdf5', 'zlib', 'lzo', 'bzip2')

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


def whichClass(hid_t loc_id, char *name):
  """Detects a class ID using heuristics.
  """
  cdef H5T_class_t  class_id
  cdef H5D_layout_t layout
  cdef hsize_t      nfields
  cdef char         *field_name1, *field_name2
  cdef int          i
  cdef hid_t        type_id, dataset_id, type_id2
  cdef object       classId
  cdef int          rank
  cdef hsize_t      *dims, *maxdims
  cdef char         byteorder[11]  # "irrelevant" fits easily here

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
        raise HDF5ExtError("Problems getting ndims.")
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
      free(<void *>field_name1)
      free(<void *>field_name2)
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


def getNestedField(recarray, fieldname):
  """
  Get the maybe nested field named `fieldname` from the `array`.

  The `fieldname` may be a simple field name or a nested field name
  with slah-separated components.
  """
  try:
    if strchr(fieldname, 47) != NULL:   # ord('/') == 47
      # It may be convenient to implement this way of descending nested
      # fields into the ``__getitem__()`` method of a subclass of
      # ``numpy.ndarray``.  -- ivb
      field = recarray
      for nfieldname in fieldname.split('/'):
        field = field[nfieldname]
    else:
      # Faster approach for non-nested columns
      field = recarray[fieldname]
  except KeyError:
    raise KeyError("no such column: %s" % (fieldname,))
  return field


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
  cdef char *attr_value
  cdef object retvalue

  # Open the root group
  root_id =  H5Gopen(file_id, "/")
  attr_value = NULL
  retvalue = None
  # Check if attribute exists
  if H5ATTRfind_attribute(root_id, attr_name):
    # Read the attr_name attribute
    ret = H5ATTRget_attribute_string(root_id, attr_name, &attr_value)
    if ret >= 0:
      retvalue = attr_value
    # Important to release attr_value, because it has been malloc'ed!
    if value: free(attr_value)

  # Close root group
  H5Gclose(root_id)

  if retvalue is not None:
    return numpy.string_(retvalue)
  else:
    return None


def getFilters(parent_id, name):
  "Get a dictionary with the filter names and cd_values"
  return get_filter_names(parent_id, name)


# This is used by several <Leaf>._convertTypes() methods.
def convertTime64(ndarray nparr, hsize_t nrecords, int sense):
  """Converts a NumPy of Time64 elements between NumPy and HDF5 formats.

  NumPy to HDF5 conversion is performed when 'sense' is 0.
  Otherwise, HDF5 to NumPy conversion is performed.
  The conversion is done in place, i.e. 'nparr' is modified.
  """

  cdef void *t64buf
  cdef long byteoffset, bytestride, nelements

  byteoffset = 0   # NumPy objects doesn't have an offset
  bytestride = nparr.strides[0]  # supports multi-dimensional recarray
  nelements = nparr.size / len(nparr)
  t64buf = nparr.data

  conv_float64_timeval32(
    t64buf, byteoffset, bytestride, nrecords, nelements, sense)


def getTypeEnum(hid_t h5type):
  """_getTypeEnum(h5type) -> hid_t
  Get the native HDF5 enumerated type of `h5type`.

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


def enumFromHDF5(hid_t enumId, char *byteorder):
  """enumFromHDF5(enumId) -> (Enum, npType)
  Convert an HDF5 enumerated type to a PyTables one.

  This function takes an HDF5 enumerated type and returns an `Enum`
  instance built from that, and the NumPy type used to encode it.
  """

  cdef hid_t  baseId
  cdef int    nelems, npenum, i
  cdef void   *rbuf
  cdef char   *ename
  cdef ndarray npvalue
  cdef object dtype, sctype

  # Find the base type of the enumerated type.
  baseId = H5Tget_super(enumId)
  if baseId < 0:
    raise HDF5ExtError("failed to get base type of HDF5 enumerated type")

  # Get the corresponding NumPy type and create temporary value.
  if getArrayType(baseId, &npenum) < 0:
    raise HDF5ExtError("failed to convert HDF5 base type to NumPy type")
  if H5Tclose(baseId) < 0:
    raise HDF5ExtError("failed to close HDF5 base type")

  try:
    sctype = NPCodeToType[npenum]
  except KeyError:
    raise NotImplementedError("""\
sorry, only scalar concrete values are supported at this moment""")

  # Get the dtype
  dtype = numpy.dtype(sctype)
  if dtype.kind not in ['i', 'u']:   # not an integer check
    raise NotImplementedError("""\
sorry, only integer concrete values are supported at this moment""")

  npvalue = numpy.array((0,), dtype=dtype)
  rbuf = npvalue.data

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

    if H5Tget_member_value(enumId, i, rbuf) < 0:
      raise HDF5ExtError(
        "failed to get element value from HDF5 enumerated type")

    enumDict[pyename] = npvalue[0]  # converted to NumPy scalar

  # Build an enumerated type from `enumDict` and return it.
  return Enum(enumDict), dtype


def enumToHDF5(object enumAtom, char *byteorder):
  """enumToHDF5(enumAtom, byteorder) -> hid_t
  Convert a PyTables enumerated type to an HDF5 one.

  This function creates an HDF5 enumerated type from the information
  contained in `enumAtom` (an ``Atom`` object), with the specified
  `byteorder` (a string).  The resulting HDF5 enumerated type is
  returned.
  """

  cdef int    npenum
  cdef size_t itemsize
  cdef char  *name
  cdef hid_t  baseId, enumId
  cdef long   bytestride, i
  cdef void  *rbuffer, *rbuf
  cdef ndarray npValues

  # Get the base HDF5 type and create the enumerated type.
  npenum = NPTypeToCode[enumAtom.dtype.base.type]
  itemsize = enumAtom.dtype.base.itemsize
  baseId = convArrayType(npenum, itemsize, byteorder)
  if baseId < 0:
    raise HDF5ExtError("failed to convert NumPy base type to HDF5")

  try:
    enumId = H5Tenum_create(baseId)
    if enumId < 0:
      raise HDF5ExtError("failed to create HDF5 enumerated type")
  finally:
    if H5Tclose(baseId) < 0:
      raise HDF5ExtError("failed to close HDF5 base type")

  # Set the name and value of each of the members.
  npNames = enumAtom._names
  npValues = enumAtom._values
  bytestride = npValues.strides[0]
  rbuffer = npValues.data
  for i from 0 <= i < len(npNames):
    name = PyString_AsString(npNames[i])
    rbuf = <void *>(<char *>rbuffer + bytestride * i)
    if H5Tenum_insert(enumId, name, rbuf) < 0:
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")
      raise HDF5ExtError("failed to insert value into HDF5 enumerated type")

  # Return the new, open HDF5 enumerated type.
  return enumId


def conv2HDF5Type(object col, char *byteorder):
  cdef hid_t   tid
  cdef int     rank, scalar
  cdef hsize_t *dims

  shape = col.dtype.shape
  if shape == ():
    scalar = True
    dims = NULL
  else:
    scalar = False
    rank = len(shape)
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info (and type!)
    for i from  0 <= i < rank:
      dims[i] = col.dtype.shape[i]
  # Create the column type
  if col.type in PTTypeToHDF5:
    if scalar:
      tid = H5Tcopy(PTTypeToHDF5[col.type])
    else:
      tid = H5Tarray_create(PTTypeToHDF5[col.type], rank, dims, NULL)
    # All types in PTTypeToHDF5 needs to fix the byte order
    # but this may change in the future!
    set_order(tid, byteorder)
  elif col.kind in PTSpecialKinds:
    # Special cases
    if col.kind == 'bool':
      tid = H5Tcopy(H5T_STD_B8)
      H5Tset_precision(tid, col.itemsize)
    elif col.type == 'complex64':
      tid = create_ieee_complex64(byteorder)
    elif col.type == 'complex128':
      tid = create_ieee_complex128(byteorder)
    elif col.kind == 'string':
      tid = H5Tcopy(H5T_C_S1);
      H5Tset_size(tid, col.itemsize)
    elif col.kind == 'enum':
      tid = enumToHDF5(col, byteorder)
    if not scalar:
      tid2 = H5Tarray_create(tid, rank, dims, NULL)
      H5Tclose(tid)
      tid = tid2
  else:
    raise TypeError("Invalid type for column %s: %s" % \
                    (col._v_name, col.type))

  # Release resources
  if dims:
    free(dims)

  return tid


def createNestedType(object desc, char *byteorder):
  """Create a nested type based on a description and return an HDF5 type."""
  cdef hid_t   tid, tid2
  cdef herr_t  ret
  cdef size_t  offset

  tid = H5Tcreate (H5T_COMPOUND, desc._v_dtype.itemsize)
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
    offset = offset + desc._v_dtype[k].itemsize
    # Release resources
    H5Tclose(tid2)

  return tid


def getRAType(hid_t type_id, int klass, size_t size):
  """Map the atomic type to a NumPy format.

  This follows the standard size and alignment.

  Return the string repr of type and the shape.
  """
  cdef H5T_sign_t  sign
  cdef hid_t       super_type_id
  cdef int         super_klass
  cdef size_t      super_size
  cdef object      stype, shape, shape2
  cdef hsize_t     *dims

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
    dims = <hsize_t *>malloc(ndims * sizeof(hsize_t))
    H5Tget_array_dims(type_id, dims, NULL)
    for i from 0 <= i < ndims:
      shape.append(<int>dims[i])  # cast to avoid long representation (i.e. 2L)
    shape = tuple(shape)
    # Release resources
    free(dims)
    H5Tclose(super_type_id)
  else:
    # Other types are not supported yet
    raise TypeError("the HDF5 class ``%s`` is not supported yet"
                    % HDF5ClassToString[klass])

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
  cdef char    byteorder[11], byteorder2[11]  # "irrelevant" fits easily here
  cdef herr_t  ret
  cdef object  sysbyteorder, desc, colobj, colpath2, typeclassname, typeclass

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
        desc[colname], _ = getNestedType(member_type_id, native_member_type_id,
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
          # These types are not supported yet
          native_member_type_id = H5Tcopy(member_type_id)
        else:
          native_member_type_id = H5Tget_native_type(member_type_id,
                                                     H5T_DIR_DEFAULT)
        # Create the Col object.
        # Indexes will be treated later on, in Table._open()
        if colstype == 'e':
          (enum, nptype) = table._g_loadEnum(native_member_type_id)
          # Take one of the names as the default in the enumeration.
          dflt = iter(enum).next()[0]
          base = Atom.from_dtype(nptype)
          colobj = EnumCol(enum, dflt, base, shape=colshape, pos=i)
        elif colstype[0] in 'at':
          kind = {'a': 'string', 't': 'time'}[colstype[0]]
          tsize = int(colstype[1:])
          colobj = Col.from_kind(kind, tsize, shape=colshape, pos=i)
        else:
          sctype = numpy.sctypeDict[colstype]
          colobj = Col.from_sctype(sctype, shape=colshape, pos=i)
        desc[colname] = colobj
        # If *any* column has a different byteorder than sys, the byteorder
        # attribute for the entire table is changed here. This should be
        # further refined for columns with different byteorders, but this case
        # is strange enough.
        ret = get_order(member_type_id, byteorder2)
        if ret > 0 and byteorder2 in ["big", "little"]:  # exclude 'irrelevant'
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
  # return the Description object and the size of the compound type
  return desc, offset



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
