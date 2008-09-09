"""
Support for arrays of nested records.

This module provides the `NestedRecArray` and `NestedRecord` classes,
which can be used to handle arrays of nested records in a way which is
compatible with ``numarray.records``.

Nested record arrays are made up by a sequence of nested records.  A
nested record is made up of a set of non-nested and nested fields, each
of them having a different name.  Non-nested fields have homogeneous
n-dimensional values (where n >= 1), while nested fields consist of a
set of fields (sub-fields), each of them with a different name.
Sub-fields can also be nested.

Several utility functions are provided for creating nested record
arrays.

Note: Despite the migration of PyTables to NumPy, this module is still
necessary in case the user still wants a nested RecArray wich is based
in numarray instead of NumPy.
"""

import sys
import types
import re

import numarray
import numarray.strings
import numarray.records

try:
    import numpy
    numpy_imported = True
except ImportError:
    numpy_imported = False

from tables.nra.attributeaccess import AttributeAccess
from tables.nra import nriterators

__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""



# formats regular expression
# allows multidimension spec with a tuple syntax in front of the letter code
# '(2,3)f4' and ' (  2 ,  3  )  f4  ' are equally allowed
format_re = re.compile(r'(?P<repeat> *[(]?[ ,0-9]*[)]? *)(?P<dtype>[A-Za-z0-9.]*)'
)

cbyteorder = { 'little': '<', 'big': '>' }

revfmt = {'Int8':'i1',  'UInt8':'u1',
          'Int16':'i2', 'UInt16':'u2',
          'Int32':'i4', 'UInt32':'u4',
          'Int64':'i8', 'UInt64':'u8',
          'Float32':'f4', 'Float64':'f8',
          'Complex32':'c8', 'Complex64':'c16',
          'Bool':'b1'}


def _isThereStructure(formats, descr, buffer):
    """
    Check if buffer structure is given.  It must be given in order to
    disambiguate possible ambiguities.

    For an explanation of argument meanings see the `array()` function.
    """

    if not (formats or descr):
        if buffer is None:
            raise ValueError("""``formats`` or ``descr`` arguments """
                """must be given if ``buffer`` is ``None``""")
        else:
            raise NotImplementedError("""unable to infer the buffer """
                """structure; it must be supplied with ``formats`` or """
                """``descr`` arguments""")


def _onlyOneSyntax(descr, formats, names):
    """
    Ensure that buffer structure is specified using either `descr` or
    `formats`.

    For an explanation of argument meanings see the `array()` function.
    """

    if descr and (formats or names):
        raise  ValueError("""only one syntax can be used to specify """
            """the buffer structure; please use either ``descr`` or """
            """``formats`` and ``names``""")


def _checkFormats(formats):
    """
    Check the format of the `formats` list.

    For an explanation of argument meanings see the `array()` function.
    """

    # Formats description must be a list or a tuple
    if not (isinstance(formats, list) or isinstance(formats, tuple)):
        raise TypeError("""``formats`` argument must be a list or a tuple""")

    # Formats elements must be strings or sequences
    for item in nriterators.flattenFormats(formats, check=True):
        if item is None:
            raise TypeError("""elements of the ``formats`` list must """
                """be strings or sequences""")


def _checkNames(names):
    """
    Check the format of the `names` list.

    For an explanation of argument meanings see the `array()` function.
    """

    # Names description must be a list or a tuple
    if not (isinstance(names, list) or isinstance(names, tuple)):
        raise TypeError("""``names`` argument must be a list""")

    # Names elements must be strings or 2-tuples
    # (flattenNames will issue a TypeError in case this is not true)
    colnames = nriterators.flattenNames(names)

    # The names used in the names list should not contain the '/' string
    for item in nriterators.getSubNames(names):
        if '/' in item:
            raise ValueError(
                """field names cannot contain the ``/`` character""")

    # For every level of the names structure names must be unique
    nriterators.checkNamesUniqueness(names)


def _checkDescr(descr):
    """
    Check the format of the `descr` list.

    For an explanation of argument meanings see the `array()` function.
    """

    # descr must be a list
    if not isinstance(descr, list):
        raise TypeError("""the descr argument must be a list!""")

    # descr must be a list of 2-tuples
    for item in nriterators.flattenDescr(descr, check=True):
        if item is None:
            raise TypeError(
                """elements of the `descr` list must be 2-tuples!""")


def _checkFieldsInDescr(descr):
    """
    Check that field names do not contain the ``/`` character.

    The checking is done on the most deeply nested field names.  For an
    explanation of argument meanings see the `array()` function.
    """

    names = [item for item in nriterators.getNamesFromDescr(descr)]
    _checkNames(names)


def makeDescr(formats, names=None):
    """
    Create a ``descr`` list for the array.

    If no `names` are passed fields are automatically named as ``c1``,
    ``c2``...
    """

    return [item for item in nriterators.getDescr(names, formats)]


def makeFormats(descr):
    """Create a ``formats`` list for the array."""

    return [item for item in nriterators.getFormatsFromDescr(descr)]


def makeNames(descr):
    """Create a ``names`` list for the array."""

    return [item for item in nriterators.getNamesFromDescr(descr)]


def _checkBufferStructure(structure, buffer):
    """
    Check the `buffer` structure using the given `structure`.

    The checking is done after flattening both the `structure` and the
    `buffer`.  `structure` is the descr list that describes the buffer
    structure.  buffer` has its usual meaning in this module.
    """

    for row in buffer:
        for item in nriterators.zipBufferDescr(row, structure):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError("""row structure doesn't match that """
                    """provided by the format specification""")
            if not isinstance(item[1], str):
                raise TypeError("""field descriptors must be strings""")


def _matchFormats_orig(seq1, seq2):
    """Check if two flat formats lists are equivalent."""

    # Lists must have the same length
    if len(seq1) != len(seq2):
        raise ValueError("""buffer structure doesn't match that """
            """provided by the format specification""")

    # Elements in the same position must describe the same format
    for (f1, f2) in zip(seq1, seq2):
        ra1 = numarray.records.array(buffer=None, formats = [f1])
        ra2 = numarray.records.array(buffer=None, formats = [f2])
        if ra1._formats != ra2._formats:
            raise ValueError("""buffer formats don't match those """
                """provided by the format specification""")

# This should be quite faster than _matchFormats_orig.
# F. Alted 2006-01-18
def _matchFormats(seq1, seq2):
    """Check if two flat formats lists are equivalent."""

    # Lists must have the same length
    if len(seq1) != len(seq2):
        raise ValueError("""buffer structure doesn't match that """
            """provided by the format specification""")

    # Elements in the same position must describe the same format
    for (f1, f2) in zip(seq1, seq2):
        (repeat1, dtype1) = format_re.match(f1.strip()).groups()
        (repeat2, dtype2) = format_re.match(f2.strip()).groups()
        dtype1 = dtype1.strip()
        dtype2 = dtype2.strip()
        if dtype1 in revfmt: dtype1 = revfmt[dtype1]
        if dtype2 in revfmt: dtype2 = revfmt[dtype2]
        if repeat1 in ['','1']: dtype1 = '1'+dtype1
        if repeat2 in ['','1']: dtype2 = '1'+dtype2
        if dtype1 != dtype2:
            raise ValueError("""buffer formats don't match those """
                """provided by the format specification""")

def _renameFields(recarray, newFieldNames):
    """Rename the fields of a recarray"""
    pass  # To be completed

def _narrowRecArray(recarray, startField, newFieldNames):
    """
    Take a set of contiguous columns from a ``RecArray``.

    This function creates and returns a new ``RecArray`` by taking a
    number of contiguous columns from `recarray`, starting by field
    `startField`.  The new columns take their names from the
    `newFieldNames` list, which also determines the number of fields to
    take.  The resulting array shares its data with `recarray`.
    """

    iStartField = recarray._names.index(startField)
    iEndField = iStartField + len(newFieldNames)
    byteOffset = recarray.field(iStartField)._byteoffset
    return numarray.records.RecArray(
        recarray._data, recarray._formats[iStartField:iEndField],
        shape=recarray._shape, names=newFieldNames, byteoffset=byteOffset,
        bytestride=recarray._bytestride, byteorder=recarray._byteorder,
        aligned=recarray._rec_aligned)


def array(buffer=None, formats=None, shape=0, names=None,
          byteorder=sys.byteorder, aligned=0, descr=None):
    """
    Create a new instance of a `NestedRecArray`.

    This function can be used to build a new array of nested records.
    The new array is returned as a result.

    The function works much like ``numarray.records.array()``, with some
    differences:

    1. In addition to flat buffers and regular sequences of non-nested
       elements, the `buffer` argument can take regular sequences where
       each element has a structure nested to an arbitrary depth.  Of
       course, all elements in a non-flat buffer must have the same
       format.

    2. The `formats` argument only supports sequences of strings and
       other sequences.  Each string defines the shape and type of a
       non-nested field.  Each sequence contains the formats of the
       sub-fields of a nested field.

       The structure of this argument must match that of the elements in
       `buffer`.  This argument may have a recursive structure.

    3. The `names` argument only supports lists of strings and 2-tuples.
       Each string defines the name of a non-nested field.  Each 2-tuple
       contains the name of a nested field and a list describing the
       names of its sub-fields.

       The structure of this argument must match that of the elements in
       `buffer`.  This argument may have a recursive structure.

    The `descr` argument is a new-style description of the structure of
    the `buffer`.  It is intended to replace the `formats` and `names`
    arguments, so they can not be used at the same time [#descr]_.

    The `descr` argument is a list of 2-tuples, each of them describing
    a field.  The first value in a tuple is the *name* of the field,
    while the second one is a description of its *structure*.  If the
    second value is a string, it defines the format (shape and type) of
    a non-nested field.  Else, it is a list of 2-tuples describing the
    sub-fields of a nested field.

    If `descr` is ``None`` (or omitted), the whole structure of the
    array is tried to be inferred from that of the `buffer`, and
    automatic names (``c1``, ``c2`` etc. on each nested field) are
    assigned to all fields.

    The `descr` argument may have a recursive structure.

    Please note that names used in `names` or `descr` should *not*
    contain the string ``'/'``, since it is used as the field/sub-field
    separator by `NestedRecArray.asRecArray()`.  If the separator is
    found in a name, a ``ValueError`` is raised.

    .. [#descr] The syntax of `descr` is based on that of the
       ``__array_descr__`` attribute in the proposed standard
       `N-dimensional array interface`__.

    __ http://numeric.scipy.org/array_interface.html


    When to use `descr` or `formats`
    ================================

    Since `descr` requires both the name and structure of fields to
    always be specified, the `formats` argument comes more handy when
    one does not want to explicitly specify names.  However it is not
    allowed to use the `names` argument without the `formats` one.  This
    is due to the fact that automatic inferrence of the `buffer`
    structure is not implemented.  When fully specifying names and
    structure, the `descr` argument is preferred over `formats` and
    `names` for the sake of code legibility and conciseness.


    Examples
    ========

    The following examples will help to clarify the words above.  In
    them, an array of two elements is created.  Each element has three
    fields: a 64-bit integer (``id``), a bi-dimensional 32-bit floating
    point (``pos``) and a nested field (``info``); the nested field has
    two sub-fields: a two-character string (``name``) and a 64-bit
    complex (``value``).

    Example 1
    ---------

    In this example the array is created by specifying both its contents
    and its structure, so the structure of the used arguments must be
    coherent.

    This is how the array would be created in the old-style way,
    i.e. using the `formats` and `names` arguments:

    >>> nra = array(
    ...     [(1, (0.5, 1.0), ('a1', 1j)), (2, (0, 0), ('a2', 1+.1j))],
    ...     names=['id', 'pos', ('info', ['name', 'value'])],
    ...     formats=['Int64', '(2,)Float32', ['a2', 'Complex64']])

    And this is how the array would be created in the new-style way,
    i.e. using the `descr` argument:

    >>> nra = array(
    ...     [(1, (0.5, 1.0), ('a1', 1j)), (2, (0, 0), ('a2', 1+.1j))],
    ...     descr=[('id', 'Int64'), ('pos', '(2,)Float32'),
    ...            ('info', [('name', 'a2'), ('value', 'Complex64')])])

    Note how `formats` and `descr` mimic the structure of each element
    in `buffer`.

    Example 2
    ---------

    Now the array is created from a flat string representing the data in
    memory.  Names will be automatically assigned.  For that to work,
    the resulting array shape and record format must be fully specified.

    >>> datastring = binary_representation_of_data
    >>> nra = array(
    ...     datastring, shape=2,
    ...     formats=['Int64', '(2,)Float32', ['a2', 'Complex64']])

    Byte ordering and alignment is assumed to be that of the host
    machine, since it has not been explicitly stated via the `byteorder`
    and `aligned` arguments.
    """

    # Check if buffer structure is specified using descr OR formats (and,
    # optionally, names)
    _onlyOneSyntax(descr, formats, names)

    # Create or check the descr format
    if descr is None:
        if formats is not None:
            descr = makeDescr(formats, names)
        # `buffer` can still be some object which describes its own structure,
        # so `descr`/`formats` are not yet required.
    else:
        _checkDescr(descr)
        _checkFieldsInDescr(descr)
    # After this, if it exists a formats, it will always exists a descr

    # This is to keep compatibility with numarray.records.array function
    if isinstance(formats, str):
        formats = formats.split(',')
    if isinstance(names, str):
        names = names.split(',')

    # First, check for easily convertible objects (NRA, NA and NumPy
    # objects)
    # F. Alted 2006-01-20
    if isinstance(buffer, NestedRecArray):
        buffer = buffer.copy()    # Always return a copy of the data
        # Return as soon as possible is not descr, formats or names specified
        if (descr is None and formats is None and names is None):
            return buffer
        # Check that descriptions are consistent
        if descr is not None:
            fmts = [item for item in nriterators.flattenFormats(makeFormats(descr))]
            # Check just the formats, not the names
            if _matchFormats(fmts, buffer._formats):
                raise ValueError, \
"""buffer structure doesn't match that provided by the format
    specification."""
            # New description is compatible. Assign it to the NRA
            buffer.descr = descr
            # Also the names (they may have been changed in new description)
            if names is None:
                names = makeNames(descr)
            # Assignements in both the NRA and flatArray are necessary
            buffer._names =  [i for i in nriterators.flattenNames(names)]
            buffer._flatArray._names = [i for i in nriterators.flattenNames(names)]
        return buffer

    if isinstance(buffer, numarray.records.RecArray):
        buffer = buffer.copy()    # Always return a copy of the data
        # Return as soon as possible is not descr, formats or names specified
        if (descr is None and formats is None and names is None):
            descr = makeDescr(buffer._formats, buffer._names)
            # Return the nested recarray
            return NestedRecArray(buffer, descr)

        # Check that descriptions are consistent
        if formats is not None and _matchFormats(formats, buffer._formats):
            raise ValueError, \
"""buffer structure doesn't match that provided by the format
    specification."""
        if names is not None:
            buffer._names = names
        # Also, the names may have been changed in new description
        elif descr is not None:
            buffer._names = makeNames(descr)
        if descr is None:
            descr = makeDescr(buffer._formats, buffer._names)
        # Refresh the cache of fields (just in case the names has changed)
        buffer._fields = buffer._get_fields()
        # Return the nested recarray
        return NestedRecArray(buffer, descr)

    # Check for numpy ndarrays, recarrays, records or scalar records
    if numpy_imported and isinstance(buffer, (numpy.ndarray, numpy.void)):
        buffer = buffer.copy()    # Always return a copy of the data
        # Try to convert into a nestedrecarray
        try:
            nra = fromnumpy(buffer)
        except Exception, exc:  #XXX
            raise ValueError, \
"""buffer parameter of type numpy cannot be converted into a NestedRecArray
object. The error was: <%s>""" % (exc,)

        # Check that descriptions are consistent
        if descr is not None:
            fmt1 = [i for i in nriterators.flattenFormats(makeFormats(nra.descr))]
            fmt2 = [i for i in nriterators.flattenFormats(makeFormats(descr))]
            if _matchFormats(fmt1, fmt2):
                raise ValueError, \
"""buffer structure doesn't match that provided by the format specification."""
        return nra

    # Check if a buffer structure is given. It must be given in order to
    # disambiguate possible ambiguities
    _isThereStructure(formats, descr, buffer)

    # Check the formats format
    if formats is None:
        formats = makeFormats(descr)
    _checkFormats(formats)

    # Check the names format
    if names is None:
        names = makeNames(descr)
    _checkNames(names)

    # Flatten the structure descriptors
    flatFormats = [item for item in nriterators.flattenFormats(formats)]
    flatNames = [item for item in nriterators.flattenNames(names)]

    # Check the buffer structure (order matters!)
    if (isinstance(buffer, types.ListType) or
        isinstance(buffer, types.TupleType)):
        if (isinstance(buffer[0], numarray.NumArray) or
            isinstance(buffer[0], numarray.strings.CharArray)):
            return fromarrays(buffer, formats=formats,
                              shape=shape, names=names,
                              byteorder=byteorder, aligned=aligned)
    elif buffer:
        _checkBufferStructure( descr, buffer)

    # Flatten the buffer (if any)
    if buffer is None:
        flatBuffer = None
    else:
        # Buffer is a list of sequences. Every sublist represents a row
        # of the array
        flatBuffer = \
            [tuple([v for (v, f) in nriterators.zipBufferDescr(row, descr)])
            for row in buffer]

    # Create a flat recarray
    flatArray = numarray.records.array(
        flatBuffer, flatFormats, shape, flatNames, byteorder, aligned)

    # Create the nested recarray
    return NestedRecArray(flatArray, descr)


def _checkArrayList(arrayList):
    """
    Check the type of the arraylist argument of fromarrays.

    For an explanation of argument meanings see the `array()` function.
    """

    # The argument must be a list or a tuple
    if not (isinstance(arrayList, list) or isinstance(arrayList, tuple)):
        raise TypeError("""``arrayList`` argument must be a list or a tuple""")

def fromarrays(arrayList, formats=None, names=None, shape=0,
               byteorder=sys.byteorder, aligned=0, descr=None):
    """
    Create a new instance of a `NestedRecArray` from `arrayList` arrays.

    This function can be used to build a new array of nested records
    from a list of arrays, one for each field.  The new array is
    returned as a result.

    The function works much like ``numarray.records.fromarrays()``, but
    `arrayList` may also contain nested fields, i.e. sequences of other
    arrays (nested or not).  All non-nested arrays appearing in
    `arrayList` must have the same length.

    The rest of arguments work as explained in `array()`.


    Example
    =======

    Let us build the sample array used in the examples of `array()`.  In
    the old way:

    >>> nra = fromarrays(
    ...     [[1, 2], [(0.5, 1.0), (0, 0)], [['a1', 'a2'], [1j, 1+.1j]]],
    ...     names=['id', 'pos', ('info', ['name', 'value'])],
    ...     formats=['Int64', '(2,)Float32', ['a2', 'Complex64']])

    In the new way:

    >>> nra = fromarrays(
    ...     [[1, 2], [(0.5, 1.0), (0, 0)], [['a1', 'a2'], [1j, 1+.1j]]],
    ...     descr=[('id', 'Int64'), ('pos', '(2,)Float32'),
    ...            ('info', [('name', 'a2'), ('value', 'Complex64')])])

    Note how `formats` and `descr` mimic the structure of the whole
    `arrayList`.
    """

    _checkArrayList(arrayList)

    # Check if a buffer structure is given. It must be given in order to
    # disambiguate possible ambiguities
    _isThereStructure(formats, descr, arrayList)

    # Check if buffer structure is specified using descr OR formats (and,
    # optionally, names)
    _onlyOneSyntax(descr, formats, names)

    # This is to keep compatibility with numarray.records.array function
    if isinstance(formats, str):
        formats = formats.split(',')
    if isinstance(names, str):
        names = names.split(',')

    # Check the descr format
    # Check for '/' in descr
    if descr is None:
        descr = makeDescr(formats, names)
    _checkDescr(descr)
    _checkFieldsInDescr(descr)

    # Check the formats format
    if formats is None:
        formats = makeFormats(descr)
    _checkFormats(formats)

    # Check the names format
    if names is None:
        names = makeNames(descr)
    _checkNames(names)

    # Flatten the structure descriptors
    flatFormats = [item for item in nriterators.flattenFormats(formats)]
    flatNames = [item for item in nriterators.flattenNames(names)]

    # Create a regular recarray from the arrays list
    flatArrayList = []
    nriterators.flattenArraysList(arrayList, descr, flatArrayList)
    ra = numarray.records.fromarrays(flatArrayList, formats=flatFormats,
        names=flatNames, shape=shape, byteorder=byteorder, aligned=aligned)

    # Create the nested recarray
    nra = NestedRecArray(ra, descr)

    return nra


def fromnumpy(array):
    """
    Create a new instance of a `RecArray` from NumPy `array`.

    If nested records are present, a `NestedRecArray` is returned.  The
    input `array` must be a NumPy array or record (it is not checked).
    """
    # Convert the original description based in the array protocol in
    # something that can be understood by the NestedRecArray
    # constructor.
    descr = [i for i in convertFromAPDescr(array.dtype.descr)]
    # Flat the description
    flatDescr = [i for i in nriterators.flattenDescr(descr)]
    # Flat the structure descriptors
    flatFormats = [i for i in nriterators.getFormatsFromDescr(flatDescr)]
    flatNames = [i for i in nriterators.getNamesFromDescr(flatDescr)]
    # Create a regular RecArray
    if array.shape == ():
        shape = 1     # Scalar case. Shape = 1 will provide an adequate buffer.
    else:
        shape = array.shape
    rarray = numarray.records.array(
        array.data, formats=flatFormats, names=flatNames,
        shape=shape, byteorder=sys.byteorder,
        aligned=False)  # aligned RecArrays are not supported yet

    # A ``NestedRecArray`` is only needed if there are nested fields.
    # This check has been disabled because a NestedRecArray does offer
    # more features that the users of PyTables 1.x are used to, most
    # specially, the __getitem__(fieldname) special method.
#     if '/' in ''.join(flatNames):
#         return NestedRecArray(rarray, descr)
#     return rarray
    return NestedRecArray(rarray, descr)


def convertToAPDescr(descr, byteorder):
    """
    Convert a NRA `descr` descriptor into a true array protocol description.
    """

    # Build a descriptor compliant with array protocol
    # (http://numeric.scipy.org/array_interface.html)
    i = nriterators.getIter(descr)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item[1], str):
                # parse the formats into repeats and formats
                try:
                    (_repeat, _dtype) = format_re.match(item[1].strip()).groups()
                except TypeError, AttributeError:
                    raise ValueError('format %s is not recognized' %  _fmt[i])
                _dtype = _dtype.strip()
                # String type needs special treatment
                if _dtype[0] in ('a', 'S'):
                    _dtype = '|S'+_dtype[1:]
                else:
                    if _dtype in revfmt:
                        # _dtype is in long format
                        _dtype = cbyteorder[byteorder]+revfmt[_dtype]
                    elif _dtype in revfmt.values():
                        # _dtype is already in short format
                        _dtype = cbyteorder[byteorder]+_dtype
                    else:
                        # This should never happen
                        raise ValueError, \
                              "Fatal error: format %s not recognized." % (_dtype)
                # Return the column
                if _repeat in ['','1','()']:
                    # scalar case
                    yield (item[0], _dtype)
                else:
                    yield (item[0], _dtype, eval(_repeat))
            else:
                l = []
                for j in convertToAPDescr(item[1], byteorder):
                    l.append(j)
                yield (item[0], l)
            item = i.next()
    except StopIteration:
        pass


def convertFromAPDescr(array_descr):
    """
    Convert a true description of the array protocol in one suited for NRA.
    """

    # Get an iterator for the array descrition
    i = nriterators.getIter(array_descr)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item[1], str):
                _dtype = item[1].strip()
                _dtype = _dtype[1:]  # remove the byteorder
                if _dtype in revfmt.values():
                    _dtype = _dtype
                elif _dtype[0] == 'S':
                    # String type needs special treatment
                    _dtype = 'a'+_dtype[1:]
                elif _dtype[0] == "V":
                    raise NotImplementedError, """ \
Padding fields are not supported yet. Try to provide objects without padding fields.
"""
                elif _dtype[0] == "U":
                    raise NotImplementedError, """ \
Unicode fields are not supported yet. Try to provide objects without unicode fields.
"""
                else:
                    # All the other fields are not supported
                    raise ValueError, \
                          "Fatal error: format %s not supported." % (_dtype)
                # Return the column
                if len(item) <= 2:
                    # scalar case
                    yield (item[0], _dtype)
                else:
                    shape = item[2]
                    yield (item[0], "%s%s" % (shape, _dtype))
            else:
                l = []
                for j in convertFromAPDescr(item[1]):
                    l.append(j)
                yield (item[0], l)
            item = i.next()
    except StopIteration:
        pass


class NestedRecArray(numarray.records.RecArray):

    """
    Array of nested records.

    This is a generalization of the ``numarray.records.RecArray`` class.
    It supports nested fields and records via the `NestedRecord` class.

    This class is compatible with ``RecArray``.  However, part of its
    behaviour has been extended to support nested fields:

    1. Getting a single item from an array will return a `NestedRecord`,
       a special kind of ``Record`` with support for nested structures.

    2. Getting a range of items will return another `NestedRecArray`
       instead of an ordinary ``RecArray``.

    3. Getting a whole field may return a `NestedRecArray` instead of a
       ``NumArray`` or ``CharArray``, if the field is nested.

    Fields and sub-fields can be accessed using both the `field()`
    method and the ``fields`` interface, which allows accessing fields
    as Python attributes: ``nrec = nrarr.fields.f1.fields.subf1[4]``.
    The `field()` method supports the ``'/'`` separator to access
    sub-fields.

    Nested record arrays can be converted to ordinary record arrays by
    using the `asRecArray()` method.

    Finally, the constructor of this class is not intended to be used
    directly by users.  Instead, use one of the creation functions
    (`array()`, `fromarrays()` or the others).
    """

    def __init__(self, recarray, descr):
        super(NestedRecArray, self).__init__(
            recarray._data, recarray._formats, shape=recarray._shape,
            names=recarray._names, byteoffset=recarray._byteoffset,
            bytestride=recarray._bytestride,
            byteorder=recarray._byteorder, aligned=recarray._rec_aligned)
        # ``_strides`` is not properly copied from the original array,
        # so, the copy must be made by hand. :[
        self._strides = recarray._strides

        self._flatArray = recarray
        self.descr = descr
        # Create a true compliant array protocol description
        self.array_descr = [d for d in convertToAPDescr(descr, self._byteorder)]

        self.fields = AttributeAccess(self, 'field')  # XXXX

        """
        Provides attribute access to fields.

        For instance, accessing ``recarray.fields.x`` is equivalent to
        ``recarray.field('x')``, and ``recarray.fields.x.fields.y`` is
        equivalent to ``recarray.field('x/y')``.  This functionality is
        mainly intended for interactive usage from the Python console.
        """


    def __str__(self):
        """Return a string representation of the nested record array."""

        psData = {}
        psData['psClassName'] = self.__class__.__name__
        psData['psElems'] = ',\n'.join([str(elem) for elem in self])

        return '''\
%(psClassName)s[
%(psElems)s
]''' % psData


    def __repr__(self):
        """
        Return the canonical string representation of the nested record
        array.
        """

        rsData = {}
        rsData['rsElems'] = '[%s]' % ',\n'.join([str(elem) for elem in self])
        rsData['rsDescr'] = str(self.descr)
        rsData['rsShape'] = str(self.shape[0])

        return '''\
array(
%(rsElems)s,
descr=%(rsDescr)s,
shape=%(rsShape)s)''' % rsData


    def __getitem__(self, key):
        if not isinstance(key, slice):
            # The `key` must be a single index.
            # Let `self._getitem()` do the job.
            return super(NestedRecArray, self).__getitem__(key)

        # The `key` is a slice.
        # Delegate selection to flat array and build a nested one from that.
        return NestedRecArray(self._flatArray[key], self.descr)


    def _getitem(self, offset):
        flatArray = self._flatArray
        row = (offset - flatArray._byteoffset) / flatArray._strides[0]
        return NestedRecord(self, row)


    def __setitem__(self, key, value):
        _RecArray = numarray.records.RecArray  # Should't it be NestedRecArray?
        if isinstance(key, slice) and not isinstance(value, _RecArray):
            # Conversion of the value to an array will need a little help
            # until structure inference is supported.
            value = array(value, descr=self.descr)
        #super(NestedRecArray, self).__setitem__(key, value)
        # Call the setitem method with the flatArray mate instead.
        # It's extremely important doing this because the shape can
        # be temporarily modified during the assign process, and self
        # and self._flatArray may end having different shapes, which
        # gives problems (specially with numarray > 1.1.1)
        # F. Alted 2005-06-09
        self._flatArray.__setitem__(key, value)


    # It seems like this method is never called, because __setitem__ calls
    # the flatArray (RecArray object) __setitem__
    # F. Alted 2005-06-09
#     def _setitem(self, offset, value):
#         row = (offset - self._byteoffset) / self._strides[0]
#         for i in range(0, self._nfields):
#             self._flatArray.field(self._names[i])[row] = \
#                 value.field(self._names[i])


    def __add__(self, other):
        """Add two NestedRecArray objects in a row wise manner."""

        if isinstance(other, NestedRecArray):
            return NestedRecArray(self._flatArray + other._flatArray,
                self.descr)
        else:
            # Assume other is a RecArray
            return NestedRecArray(self._flatArray + other, self.descr)


    def field(self, fieldName):
        """
        Get field data as an array.

        `fieldName` can be the name or the index of a field in the
        record array.  If it is not nested, a ``NumArray`` or
        ``CharArray`` object representing the values in that field is
        returned.  Else, a `NestedRecArray` object is returned.

        `fieldName` can be used to provide the name of sub-fields.  In
        that case, it will consist of several field name components
        separated by the string ``'/'``.  For instance, if there is a
        nested field named ``x`` with a sub-field named ``y``, the last
        one can be accesed by using ``'x/y'`` as the value of
        `fieldName`.
        """

        # fieldName can be an integer, get the corresponding name
        if isinstance(fieldName, int):
            fieldName = self.descr[fieldName][0]

        # The descr list of the field whose content is being extracted
        fieldDescr = [
            item for item in nriterators.getFieldDescr(fieldName, self.descr)]
        if fieldDescr == []:
            raise ValueError("there is no field named ``%s``" % (fieldName,))
        fieldDescr = fieldDescr[0][1]

        # Case 1) non nested fields (bottom level)
        if isinstance(fieldDescr, str):
            # The field content is returned as numarray or chararray
            return self._flatArray.field(fieldName)

        # Case 2) nested fields (both top and intermediate levels)
        # We need fully qualified names to access the flat array fields
        fieldNames = [
            name for name in nriterators.getNamesFromDescr(fieldDescr)]
        flatNames = [
            name for name in nriterators.flattenNames(fieldNames)]

        # This is the flattened name of the original first bottom field.
        startField = '%s/%s' % (fieldName, flatNames[0])
        # Get the requested fields from the flat array and build a nested one.
        newFlatArray = _narrowRecArray(self._flatArray, startField, flatNames)
        return NestedRecArray(newFlatArray, fieldDescr)


    def asRecArray(self, copy=True):
        """
        Convert a nested array to a non-nested equivalent array.

        This function creates a new vanilla ``RecArray`` instance
        equivalent to this one by *flattening* its fields.  Only
        bottom-level fields are included in the array.  Sub-fields are
        named by prepending the names of their parent fields up to the
        top-level fields, using ``'/'`` as a separator.

        By default the data area of the array is copied into the new one,
        but a pointer to the data area can be returned if the copy
        argument is set to False.

        Example
        -------

        Let us take the following nested array:

        >>> nra = array([(1, (0, 0), ('a1', 1j)), (2, (0, 0), ('a2', 2j))],
        ...             names=['id', 'pos', ('info', ['name', 'value'])],
        ...             formats=['Int64', '(2,)Float32', ['a2', 'Complex64']])

        Calling ``nra.asRecArray()`` would return the same array as
        calling:

        >>> ra = numarray.records.array(
        ...     [(1, (0, 0), 'a1', 1j), (2, (0, 0), 'a2', 2j)],
        ...     names=['id', 'pos', 'info/name', 'info/value'],
        ...     formats=['Int64', '(2,)Float32', 'a2', 'Complex64'])

        Please note that the shape of multi-dimensional fields is kept.
        """

        if copy:
            return self._flatArray.copy()
        else:
            return self._flatArray


    def copy(self):
        return NestedRecArray(self._flatArray.copy(), self.descr)



class NestedRecord(numarray.records.Record):

    """
    Nested record.

    This is a generalization of the ``numarray.records.Record`` class to
    support nested fields.  It represents a record in a `NestedRecArray`
    or an isolated record.  In the second case, its names are
    automatically set to ``c1``, ``c2`` etc. on each nested field.

    This class is compatible with ``Record``.  However, getting a field
    may return a `NestedRecord` instead of a Python scalar, ``NumArray``
    or ``CharArray``, if the field is nested.

    Fields and sub-fields can be accessed using both the `field()`
    method and the ``fields`` interface, which allows accessing fields
    as Python attributes: ``nfld = nrec.fields.f1.fields.subf1[4]``.
    The `field()` method supports the ``'/'`` separator to access
    sub-fields.

    Nested records can be converted to ordinary records by using the
    `asRecord()` method.
    """

    def __init__(self, input, row=0):
        numarray.records.Record.__init__(self, input._flatArray, row)
        self.array = input

        self.fields = AttributeAccess(self, 'field')
        """
        Provides attribute access to fields.

        For instance, accessing ``record.fields.x`` is equivalent to
        ``record.field('x')``, and ``record.fields.x.fields.y`` is
        equivalent to ``record.field('x/y')``.  This functionality is
        mainly intended for interactive usage from the Python console.
        """


    def __str__(self):
        """Return a string representation of the nested record."""

        # This is only defined to avoid falling back to ``Record.__str__()``.
        return repr(self)


    def __repr__(self):
        """Return the canonical string representation of the nested record."""

        nra = self.array
        row = self.row

        fieldNames = [fieldName for (fieldName, fieldFormat) in nra.descr]

        field_rsValues = []
        for fieldName in fieldNames:
            rsFieldValue = repr(nra.field(fieldName)[row])
            field_rsValues.append(rsFieldValue)
        rsFieldValues = '(%s)' % ', '.join(field_rsValues)
        return rsFieldValues


    def field(self, fieldName):
        """
        Get field data.

        If the named field (`fieldName`, a string) is not nested, a
        Python scalar, ``NumArray`` or ``CharArray`` object with the
        value of that field is returned.  Else, a `NestedRecord` object
        is returned.

        `fieldName` can be used to provide the name of sub-fields.  In
        that case, it will consist of several field name components
        separated by the string ``'/'``.  For instance, if there is a
        nested field named ``x`` with a sub-field named ``y``, the last
        one can be accesed by using ``'x/y'`` as the value of
        `fieldName`.
        """
        return self.array.field(fieldName)[self.row]


    def asRecord(self):
        """
        Convert a nested record to a non-nested equivalent record.

        This function creates a new vanilla ``Record`` instance
        equivalent to this one by *flattening* its fields.  Only
        bottom-level fields are included in the array.

        The *whole array* to which the record belongs is copied. If you
        want to repeatedly access nested records as flat records you
        should consider converting the whole nested array into a flat
        one and access its records normally.

        Example
        -------

        Let us take the following nested record:

        >>> nr = NestedRecord([1, (0, 0), ('a1', 1j)])

        Calling ``nr.asRecord()`` would return the same record as
        calling:

        >>> r = numarray.records.Record([1, (0, 0), 'a1', 1j])

        Please note that the shape of multi-dimensional fields is kept.
        """
        return self.array.asRecArray()[self.row]


    def __len__(self):
        """Get the number of fields in this record."""
        return len(self.array.descr)


    def __getitem__(self, fieldName):
        """Get the value of the field `fieldName`."""
        return self.field(fieldName)


    def __setitem__(self, fieldName, value):
        """Set the `value` of the field `fieldName`."""
        self.setfield(fieldName, value)


    def copy(self):
        """
        Make a copy of this record.

        Only one row of the nested recarray is copied. This is useful in
        some corner cases, for instance

        (nra[0], nra[-1]) = (nra[-1], nra[0])

        doesn't work but

        (nra[0], nra[-1]) = (nra[-1].copy(), nra[0].copy())

        works just fine.

        No data are shared between the copy and the source.
        """
        nra = NestedRecArray(self.array[self.row:self.row + 1].asRecArray(),
            self.array.descr)
        return NestedRecord(nra, 0)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
