#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: Sep 15, 2006
#       Author:  Francesc Alted - faltet@pytables.com
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


## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
