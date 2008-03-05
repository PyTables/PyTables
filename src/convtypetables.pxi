#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: Sep 15, 2006
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id: convtypetables.pxi 1808 2006-09-15 10:33:19Z faltet $
#
########################################################################

# """Tables for type conversion between PyTables, NumPy & HDF5"""

import sys

# Definitions that are platform-independent
from definitions cimport \
     NPY_BOOL, NPY_STRING, \
     NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, \
     NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64, \
     NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128, \
     H5T_C_S1, \
     H5T_NO_CLASS, H5T_INTEGER, H5T_FLOAT, H5T_TIME, H5T_STRING, \
     H5T_BITFIELD, H5T_OPAQUE, H5T_COMPOUND, H5T_REFERENCE, \
     H5T_ENUM, H5T_VLEN, H5T_ARRAY, H5T_class_t, \
     H5T_ORDER_LE, H5T_ORDER_BE, H5Tget_order, \
     npy_intp

from definitions cimport \
     H5T_ORDER_BE, \
     H5T_STD_B8LE, H5T_UNIX_D32LE, H5T_UNIX_D64LE, \
     H5T_STD_I8LE, H5T_STD_I16LE, H5T_STD_I32LE, H5T_STD_I64LE, \
     H5T_STD_U8LE, H5T_STD_U16LE, H5T_STD_U32LE, H5T_STD_U64LE, \
     H5T_IEEE_F32LE, H5T_IEEE_F64LE, \
     H5T_STD_B8BE, H5T_UNIX_D32BE, H5T_UNIX_D64BE, \
     H5T_STD_I8BE, H5T_STD_I16BE, H5T_STD_I32BE, H5T_STD_I64BE, \
     H5T_STD_U8BE, H5T_STD_U16BE, H5T_STD_U32BE, H5T_STD_U64BE, \
     H5T_IEEE_F32BE, H5T_IEEE_F64BE

# Platform-dependent types
if sys.byteorder == "little":
  platform_byteorder = H5T_ORDER_LE
  # Standard types, independent of the byteorder
  H5T_STD_B8   = H5T_STD_B8LE
  H5T_STD_I8   = H5T_STD_I8LE
  H5T_STD_I16  = H5T_STD_I16LE
  H5T_STD_I32  = H5T_STD_I32LE
  H5T_STD_I64  = H5T_STD_I64LE
  H5T_STD_U8   = H5T_STD_U8LE
  H5T_STD_U16  = H5T_STD_U16LE
  H5T_STD_U32  = H5T_STD_U32LE
  H5T_STD_U64  = H5T_STD_U64LE
  H5T_IEEE_F32 = H5T_IEEE_F32LE
  H5T_IEEE_F64 = H5T_IEEE_F64LE
  H5T_UNIX_D32  = H5T_UNIX_D32LE
  H5T_UNIX_D64  = H5T_UNIX_D64LE
else:  # sys.byteorder == "big"
  platform_byteorder = H5T_ORDER_BE
  # Standard types, independent of the byteorder
  H5T_STD_B8   = H5T_STD_B8BE
  H5T_STD_I8   = H5T_STD_I8BE
  H5T_STD_I16  = H5T_STD_I16BE
  H5T_STD_I32  = H5T_STD_I32BE
  H5T_STD_I64  = H5T_STD_I64BE
  H5T_STD_U8   = H5T_STD_U8BE
  H5T_STD_U16  = H5T_STD_U16BE
  H5T_STD_U32  = H5T_STD_U32BE
  H5T_STD_U64  = H5T_STD_U64BE
  H5T_IEEE_F32 = H5T_IEEE_F32BE
  H5T_IEEE_F64 = H5T_IEEE_F64BE
  H5T_UNIX_D32  = H5T_UNIX_D32BE
  H5T_UNIX_D64  = H5T_UNIX_D64BE


#----------------------------------------------------------------------------

# Conversion from PyTables string types to HDF5 native types
# List only types that are susceptible of changing byteorder
# (complex & enumerated types are special and should not be listed here)
PTTypeToHDF5 = {
  'int8'   : H5T_STD_I8,   'uint8'  : H5T_STD_U8,
  'int16'  : H5T_STD_I16,  'uint16' : H5T_STD_U16,
  'int32'  : H5T_STD_I32,  'uint32' : H5T_STD_U32,
  'int64'  : H5T_STD_I64,  'uint64' : H5T_STD_U64,
  'float32': H5T_IEEE_F32, 'float64': H5T_IEEE_F64,
  'time32' : H5T_UNIX_D32, 'time64' : H5T_UNIX_D64,
  }

# Special cases whose byteorder cannot be directly changed
PTSpecialKinds = ['complex', 'string', 'enum', 'bool']

# Conversion table from NumPy extended codes prefixes to PyTables kinds
NPExtPrefixesToPTKinds = {
  "S": "string",
  "b": "bool",
  "i": "int",
  "u": "uint",
  "f": "float",
  "c": "complex",
  "t": "time",
  "e": "enum",
  }

# Names of HDF5 classes
HDF5ClassToString = {
  H5T_NO_CLASS  : 'H5T_NO_CLASS',
  H5T_INTEGER   : 'H5T_INTEGER',
  H5T_FLOAT     : 'H5T_FLOAT',
  H5T_TIME      : 'H5T_TIME',
  H5T_STRING    : 'H5T_STRING',
  H5T_BITFIELD  : 'H5T_BITFIELD',
  H5T_OPAQUE    : 'H5T_OPAQUE',
  H5T_COMPOUND  : 'H5T_COMPOUND',
  H5T_REFERENCE : 'H5T_REFERENCE',
  H5T_ENUM      : 'H5T_ENUM',
  H5T_VLEN      : 'H5T_VLEN',
  H5T_ARRAY     : 'H5T_ARRAY',
  }


# Helper routines. These are here so as to easy the including in .pyx files.
# The next functions are not directly related with this file. If the list
# below starts to grow, they should be moved to its own .pxi file.

cdef hsize_t *malloc_dims(object pdims):
  """Returns a malloced hsize_t dims from a python pdims."""
  cdef int i, rank
  cdef hsize_t *dims

  dims = NULL
  rank = len(pdims)
  if rank > 0:
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    for i from 0 <= i < rank:
      dims[i] = pdims[i]
  return dims


cdef hid_t get_native_type(hid_t type_id):
  """Get the native type of a HDF5 type."""
  cdef H5T_class_t class_id
  cdef hid_t native_type_id, super_type_id
  cdef char *sys_byteorder

  class_id = H5Tget_class(type_id)
  if class_id in (H5T_ARRAY, H5T_VLEN):
    # Get the array base component
    super_type_id = H5Tget_super(type_id)
    # Get the class
    class_id = H5Tget_class(super_type_id)
    H5Tclose(super_type_id)
  if class_id in (H5T_INTEGER, H5T_FLOAT, H5T_COMPOUND, H5T_ENUM):
    native_type_id = H5Tget_native_type(type_id, H5T_DIR_DEFAULT)
  else:
    # Fixing the byteorder for other types shouldn't be needed.
    # More in particular, H5T_TIME is not managed yet by HDF5 and so this
    # has to be managed explicitely inside the PyTables extensions.
    # Regarding H5T_BITFIELD, well, I'm not sure if changing the byteorder
    # of this is a good idea at all.
    native_type_id = H5Tcopy(type_id)
  if native_type_id < 0:
    raise HDF5ExtError("Problems getting type id for class %s" % class_id)
  return native_type_id


## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
