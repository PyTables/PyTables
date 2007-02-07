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

"""Tables for type conversion between PyTables, NumPy & HDF5
"""

import sys

# Definitions that are platform-independent
from definitions cimport \
     NPY_BOOL, NPY_STRING, \
     NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, \
     NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64, \
     NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128, \
     H5T_C_S1, H5T_UNIX_D32BE, H5T_UNIX_D64BE, \
     H5T_NO_CLASS, H5T_INTEGER, H5T_FLOAT, H5T_TIME, H5T_STRING, \
     H5T_BITFIELD, H5T_OPAQUE, H5T_COMPOUND, H5T_REFERENCE, \
     H5T_ENUM, H5T_VLEN, H5T_ARRAY, H5T_class_t, \
     npy_intp

# Platform-dependent types
if sys.byteorder == "little":

  from definitions cimport \
       H5T_STD_B8LE, \
       H5T_STD_I8LE, H5T_STD_I16LE, H5T_STD_I32LE, H5T_STD_I64LE, \
       H5T_STD_U8LE, H5T_STD_U16LE, H5T_STD_U32LE, H5T_STD_U64LE, \
       H5T_IEEE_F32LE, H5T_IEEE_F64LE

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

else:  # sys.byteorder == "big"

  from definitions cimport \
       H5T_STD_B8BE, \
       H5T_STD_I8BE, H5T_STD_I16BE, H5T_STD_I32BE, H5T_STD_I64BE, \
       H5T_STD_U8BE, H5T_STD_U16BE, H5T_STD_U32BE, H5T_STD_U64BE, \
       H5T_IEEE_F32BE, H5T_IEEE_F64BE

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


#----------------------------------------------------------------------------

# Conversion from PyTables string types to HDF5 native types
# List only types that are susceptible of changing byteorder
# (complex & enumerated types are special and should not be listed here)
PTTypeToHDF5 = {
  'bool'   : H5T_STD_B8,
  'int8'   : H5T_STD_I8,   'uint8'  : H5T_STD_U8,
  'int16'  : H5T_STD_I16,  'uint16' : H5T_STD_U16,
  'int32'  : H5T_STD_I32,  'uint32' : H5T_STD_U32,
  'int64'  : H5T_STD_I64,  'uint64' : H5T_STD_U64,
  'float32': H5T_IEEE_F32, 'float64': H5T_IEEE_F64,
  # time datatypes cannot be distinguished if they are LE and BE
  # so, we (arbitrarily) always choose BE byteorder
  'time32' : H5T_UNIX_D32BE, 'time64' : H5T_UNIX_D64BE,
  }

# Special cases whose byteorder cannot be directly changed
PTSpecialKinds = ['complex', 'string', 'enum']


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

# The next conversion tables doesn't seem to be needed anymore.
# I'll comment them out and will eventually be removed.

# # Conversion table from NumPy extended codes to NumPy type classes
# NPExtToType = {
#   NPY_BOOL:      numpy.bool_,     NPY_STRING:     numpy.string_,
#   NPY_INT8:      numpy.int8,      NPY_UINT8:      numpy.uint8,
#   NPY_INT16:     numpy.int16,     NPY_UINT16:     numpy.uint16,
#   NPY_INT32:     numpy.int32,     NPY_UINT32:     numpy.uint32,
#   NPY_INT64:     numpy.int64,     NPY_UINT64:     numpy.uint64,
#   NPY_FLOAT32:   numpy.float32,   NPY_FLOAT64:    numpy.float64,
#   NPY_COMPLEX64: numpy.complex64, NPY_COMPLEX128: numpy.complex128,
#   # Special cases:
#   ord('t'): numpy.int32,          ord('T'):       numpy.float64,
# ##  ord('e'):      'Enum',  # fake type (the actual type can be different)
#   }


# # # Conversion table from NumPy type classes to NumPy type codes
# NPTypeToCode = {
#   numpy.bool_:     NPY_BOOL,      numpy.string_:    NPY_STRING,
#   numpy.int8:      NPY_INT8,      numpy.uint8:      NPY_UINT8,
#   numpy.int16:     NPY_INT16,     numpy.uint16:     NPY_UINT16,
#   numpy.int32:     NPY_INT32,     numpy.uint32:     NPY_UINT32,
#   numpy.int64:     NPY_INT64,     numpy.uint64:     NPY_UINT64,
#   numpy.float32:   NPY_FLOAT32,   numpy.float64:    NPY_FLOAT64,
#   numpy.complex64: NPY_COMPLEX64, numpy.complex128: NPY_COMPLEX128,
#   }


# # Conversion from NumPy extended codes to PyTables string types
# NPExtToPTType = {
#   NPY_BOOL:      'bool',      NPY_STRING:     'string',
#   NPY_INT8:      'int8',      NPY_UINT8:      'uint8',
#   NPY_INT16:     'int16',     NPY_UINT16:     'uint16',
#   NPY_INT32:     'int32',     NPY_UINT32:     'uint32',
#   NPY_INT64:     'int64',     NPY_UINT64:     'uint64',
#   NPY_FLOAT32:   'float32',   NPY_FLOAT64:    'float64',
#   NPY_COMPLEX64: 'complex64', NPY_COMPLEX128: 'complex128',
#   # Extended codes:
#   ord('t'):      'time32',    ord('T'):       'time64',
#   ord('e'):      'enum',
#   }


# # Conversion from PyTables string types to NumPy extended codes
# PTTypeToNPExt = {}
# for key, value in NPExtToPTType.items():
#   PTTypeToNPExt[value] = key


# The next functions are not directly related with this file
# If the list below starts to grow, they should be moved to its own
# .pxi file.
cdef hsize_t *malloc_dims(object pdims):
  "Returns a malloced hsize_t dims from a python pdims."
  cdef int i, rank
  cdef hsize_t *dims

  dims = NULL
  rank = len(pdims)
  if rank > 0:
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    for i from 0 <= i < rank:
      dims[i] = pdims[i]
  return dims


cdef hsize_t *npy_malloc_dims(int rank, npy_intp *pdims):
  "Returns a malloced hsize_t dims from a npy_intp *pdims."
  cdef int i
  cdef hsize_t *dims

  dims = NULL
  if rank > 0:
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    for i from 0 <= i < rank:
      dims[i] = pdims[i]
  return dims


cdef hid_t get_native_type(hid_t type_id):
  "Get the native type of a HDF5 type"
  cdef H5T_class_t class_id
  cdef hid_t native_type_id, super_type_id
  cdef char *sys_byteorder

  class_id = H5Tget_class(type_id)
  if class_id == H5T_ARRAY:
    # Get the array base component
    super_type_id = H5Tget_super(type_id)
    # Get the class
    class_id = H5Tget_class(super_type_id)
    H5Tclose(super_type_id)    
  if class_id in (H5T_INTEGER, H5T_FLOAT, H5T_COMPOUND, H5T_ENUM):
    native_type_id = H5Tget_native_type(type_id, H5T_DIR_DEFAULT)
  elif class_id in (H5T_BITFIELD, H5T_TIME):
    # These types are not supported yet by H5Tget_native_type
    native_type_id = H5Tcopy(type_id)
    sys_byteorder = PyString_AsString(sys.byteorder)
    if set_order(native_type_id, sys_byteorder) < 0:
      raise HDF5ExtError(
        "problems setting the byteorder for type of class: %s" % class_id)
  else:
    # Fixing the byteorder for these types shouldn't be needed
    native_type_id = H5Tcopy(type_id)
  if native_type_id < 0:
    raise HDF5ExtError("Problems getting type id for class %s" % class_id)
  return native_type_id


# Helper routines. These are here so as to easy the including in .pyx files.
## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
