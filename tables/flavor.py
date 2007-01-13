"""
Utilities for handling different array flavors in PyTables.

:Author: Ivan Vilata i Balaguer
:Contact: ivilata at carabos dot com
:License: BSD
:Created: December 28, 2004
:Revision: $Id$

Variables
=========

`__docformat`__
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
`internal_flavor`
    The flavor used internally by PyTables.
`all_flavors`
    List of all flavors available to PyTables.
`description_map`
    Maps flavors to short descriptions of their supported objects.
`identifier_map`
    Maps flavors to functions that can identify their objects.

    The function associated with a given flavor will return a true
    value if the object passed to it can be identified as being of
    that flavor.

    See the `flavor_of()` function for a friendlier interface to
    flavor identification.

`converter_map`
    Maps (source, destination) flavor pairs to converter functions.

    Converter functions get an array of the source flavor and return
    an array of the destination flavor.

    See the `array_of_flavor()` and `flavor_to_flavor()` functions for
    friendlier interfaces to flavor conversion.
"""

# Imports
# =======
import warnings

from tables.exceptions import FlavorError, FlavorWarning


# Public variables
# ================
__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""

internal_flavor = 'numpy'
"""The flavor used internally by PyTables."""

# This is very slightly slower than a set for a small number of values
# in terms of (infrequent) lookup time, but allows `flavor_of()`
# (which may be called much more frequently) to check for flavors in
# order, beginning with the most common one.
all_flavors = []  # filled as flavors are registered
"""List of all flavors available to PyTables."""

description_map = {}  # filled as flavors are registered
"""Maps flavors to short descriptions of their supported objects."""

identifier_map = {}  # filled as flavors are registered
"""
Maps flavors to functions that can identify their objects.

The function associated with a given flavor will return a true value
if the object passed to it can be identified as being of that flavor.

See the `flavor_of()` function for a friendlier interface to flavor
identification.
"""

converter_map = {}  # filled as flavors are registered
"""
Maps (source, destination) flavor pairs to converter functions.

Converter functions get an array of the source flavor and return an
array of the destination flavor.

See the `array_of_flavor()` and `flavor_to_flavor()` functions for
friendlier interfaces to flavor conversion.
"""


# Public functions
# ================
def check_flavor(flavor):
    """Raise a ``FlavorError`` if the `flavor` is not valid."""
    if flavor not in all_flavors:
        available_flavs = ", ".join(flav for flav in all_flavors)
        raise FlavorError(
            "flavor ``%s`` is unsupported or unavailable; "
            "available flavors in this system are: %s"
            % (flavor, available_flavs) )

def array_of_flavor2(array, src_flavor, dst_flavor):
    """
    Get a version of the given `array` in a different flavor.

    The input `array` must be of the given `src_flavor`, and the
    returned array will be of the indicated `dst_flavor`.  Both
    flavors may be the same, but it is not guaranteed that the
    returned array will be the same object as the input one in this
    case.

    If the conversion is not supported, a ``FlavorError`` is raised.
    """
    convkey = (src_flavor, dst_flavor)
    if convkey not in converter_map:
        raise FlavorError( "conversion from flavor ``%s`` to flavor ``%s`` "
                           "is unsupported or unavailable in this system"
                           % (src_flavor, dst_flavor) )
    convfunc = converter_map[convkey]
    return convfunc(array)

def flavor_to_flavor(array, src_flavor, dst_flavor):
    """
    Get a version of the given `array` in a different flavor.

    The input `array` must be of the given `src_flavor`, and the
    returned array will be of the indicated `dst_flavor` (see below
    for an exception to this).  Both flavors may be the same, but it
    is not guaranteed that the returned array will be the same object
    as the input one in this case.

    If the conversion is not supported, a `FlavorWarning` is issued
    and the input `array` is returned as is.
    """
    try:
        return array_of_flavor2(array, src_flavor, dst_flavor)
    except FlavorError, fe:
        warnings.warn( "%s; returning an object of the ``%s`` flavor instead"
                       % (fe.args[0], src_flavor), FlavorWarning )
        return array

def internal_to_flavor(array, dst_flavor):
    """
    Get a version of the given `array` in a different `dst_flavor`.

    The input `array` must be of the internal flavor, and the returned
    array will be of the given `dst_flavor`.  See `flavor_to_flavor()`
    for more information.
    """
    return flavor_to_flavor(array, internal_flavor, dst_flavor)

def array_as_internal(array, src_flavor):
    """
    Get a version of the given `array` in the internal flavor.

    The input `array` must be of the given `src_flavor`, and the
    returned array will be of the internal flavor.

    If the conversion is not supported, a ``FlavorError`` is raised.
    """
    return array_of_flavor2(array, src_flavor, internal_flavor)

def flavor_of(array):
    """
    Identify the flavor of a given `array`.

    If the `array` can not be matched with any flavor, a ``TypeError``
    is raised.
    """
    for flavor in all_flavors:
        if identifier_map[flavor](array):
            return flavor
    type_name = type(array).__name__
    supported_descs = "; ".join(description_map[fl] for fl in all_flavors)
    raise TypeError(
        "objects of type ``%s`` are not supported in this context, sorry; "
        "supported objects are: %s." % (type_name, supported_descs) )

def array_of_flavor(array, dst_flavor):
    """
    Get a version of the given `array` in a different `dst_flavor`.

    The flavor of the input `array` is guessed, and the returned array
    will be of the given `dst_flavor`.

    If the conversion is not supported, a ``FlavorError`` is raised.
    """
    return array_of_flavor2(array, flavor_of(array), dst_flavor)


# Flavor registration
# ===================
#
# The order in which flavors appear in `all_flavors` determines the
# order in which they will be tested for by `flavor_of()`, so place
# most frequent flavors first.
import numpy
all_flavors.append('numpy')  # this is the internal flavor

all_flavors.append('python')  # this is always supported

try:
    import numarray
    import numarray.generic
    import numarray.strings
    import numarray.records
except ImportError:
    pass
else:
    all_flavors.append('numarray')

try:
    import Numeric
except ImportError:
    pass
else:
    all_flavors.append('numeric')

def _register_descriptions():
    """Register descriptions of *available* flavors."""
    for flavor in all_flavors:
        description_map[flavor] = eval('_%s_desc' % flavor)

def _register_identifiers():
    """Register identifier functions of *available* flavors."""
    for flavor in all_flavors:
        identifier_map[flavor] = eval('_is_%s' % flavor)

def _register_converters():
    """Register converter functions between *available* flavors."""
    def identity(array):
        return array
    for src_flavor in all_flavors:
        for dst_flavor in all_flavors:
            # Converters with the same source and destination flavor
            # are used when available, since they may perform some
            # optimizations on the resulting array (e.g. making it
            # contiguous).  Otherwise, an identity function is used.
            convfunc = None
            try:
                convfunc = eval('_conv_%s_to_%s' % (src_flavor, dst_flavor))
            except NameError:
                if src_flavor == dst_flavor:
                    convfunc = identity
            if convfunc:
                converter_map[(src_flavor, dst_flavor)] = convfunc


# Implementation of flavors
# =========================
_python_desc = ( "homogeneous list or tuple, "
                 "integer, float, complex or string" )
def _is_python(array):
    if ( 'numpy' in all_flavors and numpy.isscalar(array)
         and not isinstance(array, numpy.void) ):  # NumPy record
        return True  # NumPy scalars will be treated as Python objects
    return isinstance(array, (tuple, list, int, float, complex, str))

_numpy_desc = "NumPy array or record"
def _is_numpy(array):
    return isinstance(array, (numpy.ndarray, numpy.void))

_numarray_desc = "numarray array or record"
def _is_numarray(array):
    na_array_or_record = (numarray.generic.NDArray, numarray.records.Record)
    return isinstance(array, na_array_or_record)

_numeric_desc = "Numeric array"
def _is_numeric(array):
    return isinstance(array, Numeric.ArrayType)

def _numpy_contiguous(convfunc):
    """Decorate `convfunc` to return a *contiguous* NumPy array."""
    def conv_to_numpy(array):
        nparr = convfunc(array)
        if hasattr(nparr, 'flags') and not nparr.flags.contiguous:
            nparr = nparr.copy()  # copying the array makes it contiguous
        return nparr
    conv_to_numpy.__name__ = convfunc.__name__
    conv_to_numpy.__doc__ = convfunc.__doc__
    return conv_to_numpy

@_numpy_contiguous
def _conv_numpy_to_numpy(array):  # just to make the array contiguous
    return array

@_numpy_contiguous
def _conv_numarray_to_numpy(array):
    # Homogeneous arrays.
    if isinstance(array, (numarray.NumArray, numarray.strings.CharArray)):
        return numpy.asarray(array)  # use the array protocol

    # Heterogeneous arrays and records.
    record = None
    if isinstance(array, numarray.records.Record):
        # Get a RecArray from a record
        record = array
        row = record.row
        array = record.array[row:row+1]
    if type(array) is numarray.records.RecArray:
        # Create a NestedRecArray array from the RecArray to easy the
        # conversion. This is sub-optimal and should be replaced by a
        # faster way to convert a plain RecArray into a numpy recarray.
        # F. Altet 2006-06-19
        array = nra.array(array)
    nparray = numpy.ndarray( buffer=array._data, shape=array.shape,
                             dtype=array.array_descr,
                             offset=array._byteoffset )
    if record:
        return nparray[row]  # get the NumPy record
    return nparray

@_numpy_contiguous
def _conv_numeric_to_numpy(array):
    return numpy.asarray(array)  # use the array protocol

@_numpy_contiguous
def _conv_python_to_numpy(array):
    return numpy.array(array)

if 'numeric' in all_flavors:
    _numtype_from_nptype = {
        numpy.bool_: Numeric.UInt8,
        numpy.int8: Numeric.Int8,
        numpy.int16: Numeric.Int16,
        numpy.int32: Numeric.Int32,
        numpy.uint8: Numeric.UInt8,
        numpy.uint16: Numeric.UInt16,
        numpy.uint32: Numeric.UInt32,
        numpy.float32: Numeric.Float32,
        numpy.float64: Numeric.Float64,
        numpy.complex64: Numeric.Complex32,
        numpy.complex128: Numeric.Complex64 }

def _conv_numpy_to_numeric(array):
    kind = array.dtype.kind
    if kind == 'V':
        raise FlavorError( "the ``numeric`` flavor does not support "
                           "heterogeneous arrays" )

    # It seems that the array protocol in Numeric does leak.  See
    # http://comments.gmane.org/gmane.comp.python.numeric.general/12563
    # for more info on this issue.
    ##if kind != 'S':
    ##    return Numeric.asarray(array)  # use the array protocol

    shape = array.shape
    if kind == 'S':
        if array.itemsize > 1:
            # Numeric does not support character arrays with elements
            # with a size > 1.  Simulate with an additional dimension.
            shape = shape + (array.itemsize,)
        typecode = 'c'
    else:
        # See the above note about the Numeric leak.
        typecode = _numtype_from_nptype[array.dtype.type]
    # Convert to a contiguous buffer (``tostring()`` is very efficient).
    arrstr = array.tostring()
    array = Numeric.fromstring(arrstr, typecode)
    array = Numeric.reshape(array, shape)
    return array

if 'numarray' in all_flavors:
    from tables import nra

def _conv_numpy_to_numarray(array):
    kind = array.dtype.kind
    if kind == 'S':  # homogeneous string array
        # We can't use the array protocol to do this conversion
        if array.shape == ():
            array = array.item()
        return numarray.strings.array( buffer=array, shape=array.shape,
                                       itemsize=array.itemsize, padc='\x00' )
    if kind != 'V':  # homogeneous array
        # This works for regular homogeneous arrays and even for rank-0 arrays
        # Using asarray gives problems in some tests (I don't know exactly why)
        ##return numarray.asarray(array)  # use the array protocol
        return numarray.array(array)  # use the array protocol (with copy)

    # For the remaining heterogeneous arrays and records, leave the
    # task to ``nra``.
    return nra.fromnumpy(array)

def _conv_numpy_to_python(array):
    if array.shape != ():
        # Lists are the default for returning multidimensional objects
        array = array.tolist()
    else:
        # 0-dim or scalar case
        array = array.item()
    return array

# Now register everything related with *available* flavors.
_register_descriptions()
_register_identifiers()
_register_converters()


# Main part
# =========
def _test():
    """Run ``doctest`` on this module."""
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
