########################################################################
#
#       License: BSD
#       Created: May 20, 2005
#       Author:  Francesc Alted - faltet@pytables.com
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

from tables.description import Description, Col
from tables.misc.enum import Enum
from tables.exceptions import HDF5ExtError
from tables.atom import Atom, EnumAtom

from tables.utils import checkFileAccess

from definitions cimport import_array, ndarray, \
     malloc, free, strchr, strcpy, strncpy, strcmp, strdup, \
     PyString_AsString, PyString_FromString, \
     H5F_ACC_RDONLY, H5P_DEFAULT, H5D_CHUNKED, H5T_DIR_DEFAULT, \
     size_t, hid_t, herr_t, hsize_t, hssize_t, htri_t, \
     H5T_class_t, H5D_layout_t, H5T_sign_t, \
     H5Fopen, H5Fclose, H5Fis_hdf5, H5Gopen, H5Gclose, \
     H5Dopen, H5Dclose, H5Dget_type, \
     H5Tcreate, H5Tcopy, H5Tclose, \
     H5Tget_nmembers, H5Tget_member_name, H5Tget_member_type, \
     H5Tget_member_value, H5Tget_size, H5Tget_native_type, \
     H5Tget_class, H5Tget_super, H5Tget_sign, H5Tget_offset, \
     H5Tinsert, H5Tenum_create, H5Tenum_insert, H5Tarray_create, \
     H5Tget_array_ndims, H5Tget_array_dims, H5Tis_variable_str, \
     H5Tset_size, H5Tset_precision, H5Tpack, \
     H5ATTRget_attribute_string, H5ATTRfind_attribute, \
     H5ARRAYget_ndims, H5ARRAYget_info, \
     create_ieee_complex64, create_ieee_complex128, \
     get_order, set_order, is_complex, \
     get_len_of_range, NPY_INT64, npy_int64, dtype, \
     PyArray_DescrFromType, PyArray_Scalar, \
     register_blosc



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
  object get_filter_names( hid_t loc_id, char *dset_name)

  H5T_class_t getHDF5ClassID(hid_t loc_id, char *name, H5D_layout_t *layout,
                             hid_t *type_id, hid_t *dataset_id)

  # To access to the slice.indices functionality for long long ints
  hssize_t getIndicesExt(object s, hsize_t length,
                         hssize_t *start, hssize_t *stop, hssize_t *step,
                         hsize_t *slicelength)


# Functions from Blosc
cdef extern from "blosc.h":
  int blosc_set_nthreads(int nthreads)



#----------------------------------------------------------------------
# Initialization code

# The NumPy API requires this function to be called before
# using any NumPy facilities in an extension module.
import_array()

cdef register_blosc_():
  cdef char *version_string, *version_date

  register_blosc(&version_string, &version_date)
  version = (version_string, version_date)
  free(version_string)
  free(version_date)
  return version


# Blosc is always accessible
blosc_version = register_blosc_()
blosc_version_string, blosc_version_date = blosc_version


# Important: Blosc calls that modifies global variables in Blosc must be
# called from the same extension where Blosc is registered in HDF5.
def setBloscMaxThreads(nthreads):
  """Set the maximum number of threads that Blosc can use.

  This actually overrides the `MAX_THREADS` setting in
  ``tables/parameters.py``, so the new value will be effective until this
  function is called again or a new file with a different `MAX_THREADS` value
  is specified.

  Returns the previous setting for maximum threads.
  """
  return blosc_set_nthreads(nthreads)


if sys.platform == "win32":
  # We need a different approach in Windows, because it complains when
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

# Helper functions

cdef hsize_t *malloc_dims(object pdims):
  """Return a malloced hsize_t dims from a python pdims."""
  cdef int i, rank
  cdef hsize_t *dims

  dims = NULL
  rank = len(pdims)
  if rank > 0:
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    for i from 0 <= i < rank:
      dims[i] = pdims[i]
  return dims


# This is a re-implementation of a working H5Tget_native_type for nested
# compound types.  I should report the flaw to THG as soon as possible.
# F. Alted 2009-08-19
cdef hid_t get_nested_native_type(hid_t type_id):
  """Get a native nested type of an HDF5 type.

  In addition, it also recursively remove possible padding on type_id, i.e. it
  acts as a combination of H5Tget_native_type and H5Tpack."""
  cdef hid_t   tid, tid2
  cdef hid_t   member_type_id, native_type_id
  cdef hsize_t nfields
  cdef H5T_class_t class_id
  cdef size_t  offset, itemsize, itemsize1
  cdef char    *colname
  cdef int     i

  # Get the itemsize
  itemsize1 = H5Tget_size(type_id)
  # Build a new type container
  tid = H5Tcreate(H5T_COMPOUND, itemsize1)

  offset = 0
  # Get the number of members
  nfields = H5Tget_nmembers(type_id)
  # Iterate thru the members
  for i from 0 <= i < nfields:
    # Get the member name
    colname = H5Tget_member_name(type_id, i)
    # Get the member type
    member_type_id = H5Tget_member_type(type_id, i)
    # Get the HDF5 class
    class_id = H5Tget_class(member_type_id)
    if class_id == H5T_COMPOUND:
      native_tid = get_nested_native_type(member_type_id)
    else:
      native_tid = H5Tget_native_type(member_type_id, H5T_DIR_DEFAULT)
    H5Tinsert(tid, colname, offset, native_tid)
    itemsize = H5Tget_size(native_tid)
    offset = offset + itemsize
    # Release resources
    H5Tclose(native_tid)
    H5Tclose(member_type_id)
    free(colname)

  # Correct the type size in case the memory type size is less
  # than the type in-disk (probably due to reading native HDF5
  # files written with tools allowing field padding)
  if H5Tget_size(tid) > offset:
    H5Tset_size(tid, offset)

  return tid


# This routine is more complex than required because HDF5 1.6.x does
# not implement support for H5Tget_native_type with some types, like
# H5T_BITFIELD and probably others.  When 1.8.x would be a requisite,
# this can be simplified.
cdef hid_t get_native_type(hid_t type_id):
  """Get the native type of a HDF5 type."""
  cdef H5T_class_t class_id
  cdef hid_t native_type_id, super_type_id
  cdef char *sys_byteorder

  class_id = H5Tget_class(type_id)
  if class_id == H5T_COMPOUND:
    # XXX It turns out that HDF5 does not correctly implement
    # H5Tget_native_type on nested compounds types.  I should
    # report this to THG.
    #
    # *Note*: the next call *combines* the effect of H5Tget_native_type and
    # H5Tpack, and both effects are needed.  Have this in mind if you
    # ever wants to replace get_nested_native_type by native HDF5 calls.
    # F. Alted  2009-08-19
    return get_nested_native_type(type_id)

  if class_id in (H5T_ARRAY, H5T_VLEN):
    # Get the array base component
    super_type_id = H5Tget_super(type_id)
    # Get the class
    class_id = H5Tget_class(super_type_id)
    H5Tclose(super_type_id)

  if class_id in (H5T_INTEGER, H5T_FLOAT, H5T_ENUM):
    native_type_id = H5Tget_native_type(type_id, H5T_DIR_DEFAULT)
  else:
    # Fixing the byteorder for other types shouldn't be needed.
    # More in particular, H5T_TIME is not managed yet by HDF5 and so this
    # has to be managed explicitely inside the PyTables extensions.
    # Regarding H5T_BITFIELD, well, I'm not sure if changing the byteorder
    # of this is a good idea at all.
    native_type_id = H5Tcopy(type_id)

  return native_type_id


def encode_filename(object filename):
  """Return the encoded filename in the filesystem encoding."""
  if type(filename) is unicode:
    encoding = sys.getfilesystemencoding()
    encname = filename.encode(encoding)
  else:
    encname = filename
  return encname


# Main functions

def isHDF5File(object filename):
  """isHDF5File(filename) -> bool
  Determine whether a file is in the HDF5 format.

  When successful, it returns a true value if the file is an HDF5
  file, false otherwise.  If there were problems identifying the file,
  an `HDF5ExtError` is raised.
  """

  # Encode the filename in case it is unicode
  encname = encode_filename(filename)

  # Check that the file exists and is readable.
  checkFileAccess(encname)

  ret = H5Fis_hdf5(encname)
  if ret < 0:
    raise HDF5ExtError("problems identifying file ``%s``" % (filename,))
  return ret > 0


def isPyTablesFile(object filename):
  """isPyTablesFile(filename) -> true or false value
  Determine whether a file is in the PyTables format.

  When successful, it returns the format version string if the file is a
  PyTables file, `None` otherwise.  If there were problems identifying the
  file, an `HDF5ExtError` is raised.
  """

  cdef hid_t file_id

  isptf = None    # A PYTABLES_FORMAT_VERSION attribute was not found
  if isHDF5File(filename):
    # Encode the filename in case it is unicode
    encname = encode_filename(filename)
    # The file exists and is HDF5, that's ok
    # Open it in read-only mode
    file_id = H5Fopen(encname, H5F_ACC_RDONLY, H5P_DEFAULT)
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

  libnames = ('hdf5', 'zlib', 'lzo', 'bzip2', 'blosc')

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
  elif strcmp(name, "blosc") == 0:
    return (blosc_version, blosc_version_string, blosc_version_date)
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
  cdef hid_t        type_id, dataset_id
  cdef object       classId
  cdef int          rank
  cdef hsize_t      *dims, *maxdims
  cdef char         byteorder[11]  # "irrelevant" fits easily here

  classId = "UNSUPPORTED"  # default value
  # Get The HDF5 class for the datatype in this dataset
  class_id = getHDF5ClassID(loc_id, name, &layout, &type_id, &dataset_id)
  # Check if this a dataset of supported classtype for ARRAY
  if  ((class_id == H5T_INTEGER)  or
       (class_id == H5T_FLOAT)    or
       (class_id == H5T_BITFIELD) or
       (class_id == H5T_TIME)     or
       (class_id == H5T_ENUM)     or
       (class_id == H5T_STRING)   or
       (class_id == H5T_ARRAY)):
    if layout == H5D_CHUNKED:
      if H5ARRAYget_ndims(dataset_id, &rank) < 0:
        raise HDF5ExtError("Problems getting ndims.")
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      maxdims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      if H5ARRAYget_info(dataset_id, type_id, dims, maxdims,
                         &class_id, byteorder) < 0:
        raise HDF5ExtError("Unable to get array info.")
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
      # Francesc Alted 2005-04-29
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
  Get the maybe nested field named `fieldname` from the `recarray`.

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
      # Faster method for non-nested columns
      field = recarray[fieldname]
  except KeyError:
    raise KeyError("no such column: %s" % (fieldname,))
  return field


def getIndices(object start, object stop, object step, hsize_t length):
  cdef hssize_t o_start, o_stop, o_step
  cdef hsize_t slicelength
  cdef object s

  # In order to convert possible numpy.integer values to long ones
  if start is not None: start = long(start)
  if stop is not None: stop = long(stop)
  if step is not None: step = long(step)
  s = slice(start, stop, step)
  if getIndicesExt(s, length, &o_start, &o_stop, &o_step, &slicelength) < 0:
    raise ValueError("Problems getting the indices on slice '%s'" % s)
  return (o_start, o_stop, o_step)


def read_f_attr(hid_t file_id, char *attr_name):
  """Read PyTables file attributes (i.e. in root group).

  Returns the value of the `attr_name` attribute in root group, or `None` if
  it does not exist.  This call cannot fail.
  """

  cdef herr_t ret
  cdef char *attr_value
  cdef object retvalue

  attr_value = NULL
  retvalue = None
  # Check if attribute exists
  if H5ATTRfind_attribute(file_id, attr_name):
    # Read the attr_name attribute
    ret = H5ATTRget_attribute_string(file_id, attr_name, &attr_value)
    if ret >= 0:
      retvalue = attr_value
    # Important to release attr_value, because it has been malloc'ed!
    if attr_value:
      free(attr_value)

  if retvalue is not None:
    return numpy.string_(retvalue)
  else:
    return None


def getFilters(parent_id, name):
  "Get a dictionary with the filter names and cd_values"
  return get_filter_names(parent_id, name)


# This is used by several <Leaf>._convertTypes() methods.
def getTypeEnum(hid_t h5type):
  """_getTypeEnum(h5type) -> hid_t
  Get the native HDF5 enumerated type of `h5type`.

  If `h5type` is an enumerated type, it is returned.  If it is a
  variable-length type with an enumerated base type, this is returned.  If it
  is a multi-dimensional type with an enumerated base type, this is returned.
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
  elif typeClass in (H5T_ARRAY, H5T_VLEN):
    # The field is multi-dimensional or variable length.
    enumId2 = H5Tget_super(h5type)
    enumId = getTypeEnum(enumId2)
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
  cdef object dtype

  # Find the base type of the enumerated type, and get the atom
  baseId = H5Tget_super(enumId)
  atom = AtomFromHDF5Type(baseId)
  H5Tclose(baseId)
  if atom.kind not in ('int', 'uint'):
    raise NotImplementedError("""\
sorry, only integer concrete values are supported at this moment""")

  dtype = atom.dtype
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
    pyename = str(ename)
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

  cdef char  *name
  cdef hid_t  baseId, enumId
  cdef long   bytestride, i
  cdef void  *rbuffer, *rbuf
  cdef ndarray npValues
  cdef object baseAtom

  # Get the base HDF5 type and create the enumerated type.
  baseAtom = Atom.from_dtype(enumAtom.dtype.base)
  baseId = AtomToHDF5Type(baseAtom, byteorder)

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


def AtomToHDF5Type(atom, char *byteorder):
  cdef hid_t   tid
  cdef hsize_t *dims

  # Create the base HDF5 type
  if atom.type in PTTypeToHDF5:
    tid = H5Tcopy(PTTypeToHDF5[atom.type])
    # Fix the byteorder
    if atom.kind != 'time':
      set_order(tid, byteorder)
  elif atom.kind in PTSpecialKinds:
    # Special cases (the byteorder doesn't need to be fixed afterwards)
    if atom.type == 'complex64':
      tid = create_ieee_complex64(byteorder)
    elif atom.type == 'complex128':
      tid = create_ieee_complex128(byteorder)
    elif atom.kind == 'string':
      tid = H5Tcopy(H5T_C_S1);
      H5Tset_size(tid, atom.itemsize)
    elif atom.kind == 'bool':
      tid = H5Tcopy(H5T_STD_B8);
    elif atom.kind == 'enum':
      tid = enumToHDF5(atom, byteorder)
  else:
    raise TypeError("Invalid type for atom %s" % (atom,))
  # Create an H5T_ARRAY in case of non-scalar atoms
  if atom.shape != ():
    dims = malloc_dims(atom.shape)
    tid2 = H5Tarray_create(tid, len(atom.shape), dims, NULL)
    free(dims)
    H5Tclose(tid)
    tid = tid2

  return tid


def loadEnum(hid_t type_id):
  """loadEnum() -> (Enum, npType)
  Load the enumerated HDF5 type associated with this type_id.

  It returns an `Enum` instance built from that, and the
  NumPy type used to encode it.
  """

  cdef hid_t enumId
  cdef char  byteorder[11]  # "irrelevant" fits well here

  # Get the enumerated type
  enumId = getTypeEnum(type_id)
  # Get the byteorder
  get_order(type_id, byteorder)
  # Get the Enum and NumPy types and close the HDF5 type.
  try:
    return enumFromHDF5(enumId, byteorder)
  finally:
    # (Yes, the ``finally`` clause *is* executed.)
    if H5Tclose(enumId) < 0:
      raise HDF5ExtError("failed to close HDF5 enumerated type")


def HDF5ToNPNestedType(hid_t type_id):
  """Given a HDF5 `type_id`, return a dtype string representation of it."""
  cdef hid_t   member_type_id
  cdef hsize_t nfields
  cdef int     i
  cdef char    *colname
  cdef H5T_class_t class_id
  cdef object  desc

  desc = {}
  # Get the number of members
  nfields = H5Tget_nmembers(type_id)
  # Iterate thru the members
  for i from 0 <= i < nfields:
    # Get the member name
    colname = H5Tget_member_name(type_id, i)
    # Get the member type
    member_type_id = H5Tget_member_type(type_id, i)
    # Get the HDF5 class
    class_id = H5Tget_class(member_type_id)
    if class_id == H5T_COMPOUND and not is_complex(member_type_id):
      desc[colname] = HDF5ToNPNestedType(member_type_id)
      desc[colname]["_v_pos"] = i  # Remember the position
    else:
      atom = AtomFromHDF5Type(member_type_id, pure_numpy_types=True)
      desc[colname] = Col.from_atom(atom, pos=i)

    # Release resources
    H5Tclose(member_type_id)
    free(colname)

  return desc


def HDF5ToNPExtType(hid_t type_id, pure_numpy_types=True, atom=False):
  """Map the atomic HDF5 type to a string repr of NumPy extended codes.

  If `pure_numpy_types` is true, detected HDF5 types that does not match pure
  NumPy types will raise a ``TypeError`` exception.  If not, HDF5 types like
  TIME, VLEN or ENUM are passed through.

  If `atom` is true, the resulting repr is meant for atoms.  If not, the
  result is meant for attributes.

  Returns the string repr of type and its shape.  The exception is for
  compounds types, that returns a NumPy dtype and shape instead.
  """
  cdef H5T_sign_t  sign
  cdef hid_t       super_type_id, native_type_id
  cdef H5T_class_t class_id, super_class_id
  cdef size_t      itemsize, super_itemsize
  cdef object      stype, shape, shape2
  cdef hsize_t     *dims

  # default shape
  shape = ()
  # Get the HDF5 class
  class_id = H5Tget_class(type_id)
  # Get the itemsize
  itemsize = H5Tget_size(type_id)

  if class_id == H5T_BITFIELD:
    stype = "b1"
  elif class_id == H5T_INTEGER:
    # Get the sign
    sign = H5Tget_sign(type_id)
    if (sign > 0):
      stype = "i%s" % (itemsize)
    else:
      stype = "u%s" % (itemsize)
  elif class_id == H5T_FLOAT:
    stype = "f%s" % (itemsize)
  elif class_id ==  H5T_COMPOUND:
    if is_complex(type_id):
      stype = "c%s" % (itemsize)
    else:
      if atom:
        raise TypeError("the HDF5 class ``%s`` is not supported yet"
                        % HDF5ClassToString[class_id])
      # Recursively remove possible padding on type_id.
      # H5Tpack(type_id)
      # H5Tpack has problems with nested compund types that were solved
      # in HDF5 1.8.2 (or perhaps 1.8.3).  Use the next better.
      native_type_id = get_nested_native_type(type_id)
      desc = Description(HDF5ToNPNestedType(native_type_id))
      # stype here is not exactly a string, but the NumPy dtype factory
      # will deal with this.
      stype = desc._v_dtype
      H5Tclose(native_type_id)
  elif class_id == H5T_STRING:
    if H5Tis_variable_str(type_id):
      raise TypeError("variable length strings are not supported yet")
    stype = "S%s" % (itemsize)
  elif class_id == H5T_TIME:
    if pure_numpy_types:
      raise TypeError("the HDF5 class ``%s`` is not supported yet"
                      % HDF5ClassToString[class_id])
    stype = "t%s" % (itemsize)
  elif class_id == H5T_ENUM:
    if pure_numpy_types:
      raise TypeError("the HDF5 class ``%s`` is not supported yet"
                      % HDF5ClassToString[class_id])
    stype = "e"
  elif class_id == H5T_VLEN:
    if pure_numpy_types:
      raise TypeError("the HDF5 class ``%s`` is not supported yet"
                      % HDF5ClassToString[class_id])
    # Get the variable length base component
    super_type_id = H5Tget_super(type_id)
    # Find the super member format
    stype, shape = HDF5ToNPExtType(super_type_id, pure_numpy_types)
    # Release resources
    H5Tclose(super_type_id)
  elif class_id == H5T_ARRAY:
    # Get the array base component
    super_type_id = H5Tget_super(type_id)
    # Get the class
    super_class_id = H5Tget_class(super_type_id)
    # Get the itemsize
    super_itemsize = H5Tget_size(super_type_id)
    # Find the super member format
    stype, shape2 = HDF5ToNPExtType(super_type_id, pure_numpy_types)
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
                    % HDF5ClassToString[class_id])

  return stype, shape


def AtomFromHDF5Type(hid_t type_id, pure_numpy_types=False):
  """Get an atom from a type_id.

  See `HDF5ToNPExtType` for an explanation of the `pure_numpy_types`
  parameter.
  """
  cdef object stype, shape, atom_, sctype, tsize, kind
  cdef object dflt, base, enum, nptype

  stype, shape = HDF5ToNPExtType(type_id, pure_numpy_types, atom=True)
  # Create the Atom
  if stype == 'e':
    (enum, nptype) = loadEnum(type_id)
    # Take one of the names as the default in the enumeration.
    dflt = iter(enum).next()[0]
    base = Atom.from_dtype(nptype)
    atom_ = EnumAtom(enum, dflt, base, shape=shape)
  else:
    kind = NPExtPrefixesToPTKinds[stype[0]]
    tsize = int(stype[1:])
    atom_ = Atom.from_kind(kind, tsize, shape=shape)

  return atom_


def createNestedType(object desc, char *byteorder):
  """Create a nested type based on a description and return an HDF5 type."""
  cdef hid_t   tid, tid2
  cdef herr_t  ret
  cdef size_t  offset

  tid = H5Tcreate(H5T_COMPOUND, desc._v_itemsize)
  if tid < 0:
    return -1;

  offset = 0
  for k in desc._v_names:
    obj = desc._v_colObjects[k]
    if isinstance(obj, Description):
      tid2 = createNestedType(obj, byteorder)
    else:
      tid2 = AtomToHDF5Type(obj, byteorder)
    ret = H5Tinsert(tid, k, offset, tid2)
    offset = offset + desc._v_dtype[k].itemsize
    # Release resources
    H5Tclose(tid2)

  return tid


cdef class lrange:
  """
  Iterate over long ranges.

  This is similar to ``xrange()``, but it allows 64-bit arguments on all
  platforms.  The results of the iteration are sequentially yielded in
  the form of ``numpy.int64`` values, but getting random individual
  items is not supported.

  Because of the Python 32-bit limitation on object lengths, the
  ``length`` attribute (which is also a ``numpy.int64`` value) should be
  used instead of the ``len()`` syntax.

  Default ``start`` and ``step`` arguments are supported in the same way
  as in ``xrange()``.  When the standard ``[x]range()`` Python objects
  support 64-bit arguments, this iterator will be deprecated.
  """
  cdef npy_int64 start, stop, step, next
  cdef dtype int64  # caches the ``numpy.int64`` type

  property length:  # no __len__ since the result would get truncated
    """
    Get the number of elements in this iteration.

    This should be used instead of ``len()`` because the latter
    truncates the real length to a 32-bit signed value.
    """
    def __get__(self):
      cdef npy_int64 rlen
      rlen = get_len_of_range(self.start, self.stop, self.step)
      return PyArray_Scalar(&rlen, self.int64, None)

  def __cinit__(self, *args):
    cdef int nargs
    cdef object start, stop, step

    nargs = len(args)
    if nargs == 1:
      start = 0
      stop = args[0]
      step = 1
    elif nargs == 2:
      start = args[0]
      stop = args[1]
      step = 1
    elif nargs == 3:
      start = args[0]
      stop = args[1]
      step = args[2]
    else:
      raise TypeError("expected 1-3 arguments, got %d" % nargs)

    if step == 0:
      raise ValueError("``step`` argument can not be zero")
    self.start = start
    self.stop = stop
    self.step = step
    self.next = start
    self.int64 = PyArray_DescrFromType(NPY_INT64)

  def __iter__(self):
    return self

  def __next__(self):
    cdef object current
    if ( (self.step > 0 and self.next >= self.stop)
         or (self.step < 0 and self.next <= self.stop) ):
      raise StopIteration
    current = PyArray_Scalar(&self.next, self.int64, None)
    self.next = self.next + self.step
    return current


## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
