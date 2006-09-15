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
  NPY_BYTE, NPY_SHORT, NPY_INT, NPY_LONGLONG, \
  NPY_UBYTE, NPY_USHORT, NPY_UINT, NPY_ULONGLONG, \
  NPY_BOOL, NPY_STRING, NPY_FLOAT, NPY_DOUBLE, NPY_CFLOAT, NPY_CDOUBLE, \
  H5T_NATIVE_SCHAR, H5T_NATIVE_SHORT, H5T_NATIVE_INT, H5T_NATIVE_LLONG,\
  H5T_NATIVE_UCHAR, H5T_NATIVE_USHORT, H5T_NATIVE_UINT, H5T_NATIVE_ULLONG, \
  H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE, H5T_UNIX_D32BE, H5T_UNIX_D64BE, \
  H5T_NO_CLASS, H5T_INTEGER, H5T_FLOAT, H5T_TIME, H5T_STRING, H5T_BITFIELD, \
  H5T_OPAQUE, H5T_COMPOUND, H5T_REFERENCE, H5T_ENUM, H5T_VLEN, H5T_ARRAY


# Conversion from NumPy codes to native HDF5 types
NPCodeToHDF5 = {
  NPY_BYTE      : H5T_NATIVE_SCHAR,
  NPY_SHORT     : H5T_NATIVE_SHORT,
  NPY_INT       : H5T_NATIVE_INT,
  NPY_LONGLONG  : H5T_NATIVE_LLONG,
  NPY_UBYTE     : H5T_NATIVE_UCHAR,
  NPY_USHORT    : H5T_NATIVE_USHORT,
  NPY_UINT      : H5T_NATIVE_UINT,
  NPY_ULONGLONG : H5T_NATIVE_ULLONG,
  NPY_FLOAT     : H5T_NATIVE_FLOAT,
  NPY_DOUBLE    : H5T_NATIVE_DOUBLE
  }


# Names of HDF5 classes
HDF5ClassToString = {
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
  }


# Conversion table from NumPy codes to NumPy type classes
NPCodeToType = {
  NPY_BOOL:      numpy.bool_,     NPY_STRING:     numpy.string_,
  NPY_BYTE:      numpy.int8,      NPY_UBYTE:      numpy.uint8,
  NPY_SHORT:     numpy.int16,     NPY_USHORT:     numpy.uint16,
  NPY_INT:       numpy.int32,     NPY_UINT:       numpy.uint32,
  NPY_LONGLONG:  numpy.int64,     NPY_ULONGLONG:  numpy.uint64,
  NPY_FLOAT:     numpy.float32,   NPY_DOUBLE:     numpy.float64,
  NPY_CFLOAT:    numpy.complex64, NPY_CDOUBLE:    numpy.complex128,
  # Special cases:
  ord('t'): numpy.int32,          ord('T'):       numpy.float64,
  }


# Conversion table from NumPy type classes to NumPy type codes
NPTypeToCode = {
  numpy.bool_:     NPY_BOOL,      numpy.string_:    NPY_STRING,
  numpy.int8:      NPY_BYTE,      numpy.uint8:      NPY_UBYTE,
  numpy.int16:     NPY_SHORT,     numpy.uint16:     NPY_USHORT,
  numpy.int32:     NPY_INT,       numpy.uint32:     NPY_UINT,
  numpy.int64:     NPY_LONGLONG,  numpy.uint64:     NPY_ULONGLONG,
  numpy.float32:   NPY_FLOAT,     numpy.float64:    NPY_DOUBLE,
  numpy.complex64: NPY_CFLOAT,    numpy.complex128: NPY_CDOUBLE,
  }


# Conversion from NumPy codes to PyTables string types
NPCodeToPTType = {
  NPY_BOOL:     'Bool',      NPY_STRING:    'CharType',
  NPY_BYTE:     'Int8',      NPY_UBYTE:     'UInt8',
  NPY_SHORT:    'Int16',     NPY_USHORT:    'UInt16',
  NPY_INT:      'Int32',     NPY_UINT:      'UInt32',
  NPY_LONGLONG: 'Int64',     NPY_ULONGLONG: 'UInt64',
  NPY_FLOAT:    'Float32',   NPY_DOUBLE:    'Float64',
  NPY_CFLOAT:   'Complex32', NPY_CDOUBLE:   'Complex64',
  # Special cases:
  ord('t'):     'Time32',    ord('T'):      'Time64',
  ord('e'):     'Enum',
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
ptSpecialTypes = ['Bool', 'Complex32', 'Complex64', 'CharType', 'Enum']

