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

from definitions cimport \
  NPY_BOOL, NPY_STRING, \
  NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, \
  NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64, \
  NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128, \
  H5T_C_S1, H5T_NATIVE_B8, \
  H5T_NATIVE_SCHAR, H5T_NATIVE_SHORT, H5T_NATIVE_INT, H5T_NATIVE_LLONG,\
  H5T_NATIVE_UCHAR, H5T_NATIVE_USHORT, H5T_NATIVE_UINT, H5T_NATIVE_ULLONG, \
  H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE, H5T_UNIX_D32BE, H5T_UNIX_D64BE, \
  H5T_NO_CLASS, H5T_INTEGER, H5T_FLOAT, H5T_TIME, H5T_STRING, H5T_BITFIELD, \
  H5T_OPAQUE, H5T_COMPOUND, H5T_REFERENCE, H5T_ENUM, H5T_VLEN, H5T_ARRAY


# Conversion from NumPy codes to native HDF5 types (for attributes)
NPCodeToHDF5 = {
  NPY_INT8      : H5T_NATIVE_SCHAR,
  NPY_INT16     : H5T_NATIVE_SHORT,
  NPY_INT32     : H5T_NATIVE_INT,
  NPY_INT64     : H5T_NATIVE_LLONG,
  NPY_UINT8     : H5T_NATIVE_UCHAR,
  NPY_UINT16    : H5T_NATIVE_USHORT,
  NPY_UINT32    : H5T_NATIVE_UINT,
  NPY_UINT64    : H5T_NATIVE_ULLONG,
  NPY_FLOAT32   : H5T_NATIVE_FLOAT,
  NPY_FLOAT64   : H5T_NATIVE_DOUBLE
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


# Conversion table from NumPy codes to NumPy type classes
NPCodeToType = {
  NPY_BOOL:      numpy.bool_,     NPY_STRING:     numpy.string_,
  NPY_INT8:      numpy.int8,      NPY_UINT8:      numpy.uint8,
  NPY_INT16:     numpy.int16,     NPY_UINT16:     numpy.uint16,
  NPY_INT32:     numpy.int32,     NPY_UINT32:     numpy.uint32,
  NPY_INT64:     numpy.int64,     NPY_UINT64:     numpy.uint64,
  NPY_FLOAT32:   numpy.float32,   NPY_FLOAT64:    numpy.float64,
  NPY_COMPLEX64: numpy.complex64, NPY_COMPLEX128: numpy.complex128,
  # Special cases:
  ord('t'): numpy.int32,          ord('T'):       numpy.float64,
  }


# Conversion table from NumPy type classes to NumPy type codes
NPTypeToCode = {
  numpy.bool_:     NPY_BOOL,      numpy.string_:    NPY_STRING,
  numpy.int8:      NPY_INT8,      numpy.uint8:      NPY_UINT8,
  numpy.int16:     NPY_INT16,     numpy.uint16:     NPY_UINT16,
  numpy.int32:     NPY_INT32,     numpy.uint32:     NPY_UINT32,
  numpy.int64:     NPY_INT64,     numpy.uint64:     NPY_UINT64,
  numpy.float32:   NPY_FLOAT32,   numpy.float64:    NPY_FLOAT64,
  numpy.complex64: NPY_COMPLEX64, numpy.complex128: NPY_COMPLEX128,
  }


# Conversion from NumPy codes to PyTables string types
NPCodeToPTType = {
  NPY_BOOL:      'Bool',      NPY_STRING:     'CharType',
  NPY_INT8:      'Int8',      NPY_UINT8:      'UInt8',
  NPY_INT16:     'Int16',     NPY_UINT16:     'UInt16',
  NPY_INT32:     'Int32',     NPY_UINT32:     'UInt32',
  NPY_INT64:     'Int64',     NPY_UINT64:     'UInt64',
  NPY_FLOAT32:   'Float32',   NPY_FLOAT64:    'Float64',
  NPY_COMPLEX64: 'Complex32', NPY_COMPLEX128: 'Complex64',
  # Special cases:
  ord('t'):      'Time32',    ord('T'):       'Time64',
  ord('e'):      'Enum',
  }

# Conversion from PyTables string types to HDF5 native types
# List only types that are susceptible of changing byteorder
PTTypeToHDF5 = {
  'Int8':    H5T_NATIVE_SCHAR,  'UInt8':   H5T_NATIVE_UCHAR,
  'Int16':   H5T_NATIVE_SHORT,  'UInt16':  H5T_NATIVE_USHORT,
  'Int32':   H5T_NATIVE_INT,    'UInt32':  H5T_NATIVE_UINT,
  'Int64':   H5T_NATIVE_LLONG,  'UInt64':  H5T_NATIVE_ULLONG,
  'Float32': H5T_NATIVE_FLOAT,  'Float64': H5T_NATIVE_DOUBLE,
  'Time32':  H5T_UNIX_D32BE,    'Time64':  H5T_UNIX_D64BE }

# Special cases that cannot be directly mapped:
PTSpecialTypes = ['Bool', 'Complex32', 'Complex64', 'CharType', 'Enum']

