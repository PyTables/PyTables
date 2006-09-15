
#############################################################
# Tables for type conversion between PyTables, NumPy & HDF5 #
#############################################################


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
  NPY_FLOAT32   : H5T_NATIVE_FLOAT,
  NPY_FLOAT64   : H5T_NATIVE_DOUBLE
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
  H5T_NCLASSES:   'H5T_NCLASSES'
  }


# Conversion table from numpy codes to numpy type classes
NPCodeToType = {
  NPY_BOOL:      numpy.bool,      NPY_STRING:     numpy.string,
  NPY_INT8:      numpy.int8,      NPY_UINT8:      numpy.uint8,
  NPY_INT16:     numpy.int16,     NPY_UINT16:     numpy.uint16,
  NPY_INT32:     numpy.int32,     NPY_UINT32:     numpy.uint32,
  NPY_INT64:     numpy.int64,     NPY_UINT64:     numpy.uint64,
  NPY_FLOAT32:   numpy.float32,   NPY_FLOAT64:    numpy.float64,
  NPY_COMPLEX64: numpy.complex64, NPY_COMPLEX128: numpy.complex128,
  # Special cases:
  ord('t'): numpy.Int32,          ord('T'):       numpy.Float64,
  }


# Conversion table from numpy type classes to numpy type codes
NPTypeToCode = {
  numpy.bool:      NPY_BOOL,      numpy.string:     NPY_STRING,
  numpy.int8:      NPY_INT8,      numpy.uint8:      NPY_UINT8,
  numpy.int16:     NPY_INT16,     numpy.uint16:     NPY_UINT16,
  numpy.int32:     NPY_INT32,     numpy.uint32:     NPY_UINT32,
  numpy.int64:     NPY_INT64,     numpy.uint64:     NPY_UINT64,
  numpy.float32:   NPY_FLOAT32,   numpy.float64:    NPY_FLOAT64,
  numpy.complex64: NPY_COMPLEX64, numpy.complex128: NPY_COMPLEX128,
  }

# Conversion from numpy codes to pytables string types
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

# Conversion from pytables string types to HDF5 native types
# List only types that are susceptible of changing byteorder
#naSTypeToH5Type = {
PTTypeToHDF5 = {
  'Int8':    H5T_NATIVE_SCHAR,  'UInt8':   H5T_NATIVE_UCHAR,
  'Int16':   H5T_NATIVE_SHORT,  'UInt16':  H5T_NATIVE_USHORT,
  'Int32':   H5T_NATIVE_INT,    'UInt32':  H5T_NATIVE_UINT,
  'Int64':   H5T_NATIVE_LLONG,  'UInt64':  H5T_NATIVE_ULLONG,
  'Float32': H5T_NATIVE_FLOAT,  'Float64': H5T_NATIVE_DOUBLE,
  'Time32':  H5T_UNIX_D32BE,    'Time64':  H5T_UNIX_D64BE }

# Special cases that cannot be directly mapped:
ptSpecialTypes = ['Bool', 'Complex32', 'Complex64', 'CharType', 'Enum']


