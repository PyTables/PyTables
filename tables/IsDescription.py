########################################################################
#
#       License:        BSD
#       Created:        September 21, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/IsDescription.py,v $
#       $Id$
#
########################################################################

"""Classes and metaclasses for defining user data columns for Table objects.

Classes:


    Col, StringCol, BoolCol, Int8Col, UInt8Col, Int16Col, UInt16Col,
    Int32Col, UInt32Col, Int64Col, UInt64Col,
    Float32Col, Float64Col, Complex32Col, Complex64Col,
    Time32Col, Time64Col, EnumCol
    metaIsDescription
    IsDescription

Functions:

Misc variables:

    __version__

"""

import warnings
import sys
import operator
import copy

import numpy

from tables.enum import Enum
from tables.utils import checkNameValidity


__version__ = "$Revision$"


def normalize_shape(shape):
    """Check that the shape is safe to be used and return it as a tuple"""

    if type(shape) not in (int, long, tuple, list):
        raise ValueError("Illegal shape object: %s" % (shape,))

    if type(shape) in (int, long):
        if shape < 1:
            raise ValueError(
                "Shape value must be greater than 0: %s" % (shape,))
        elif shape == 1:
            shape = ()  # Equivalent to 1, but it is NumPy convention
        else:
            # To prevent confusions between 2 and (2,):
            # the meaning is the same
            shape = (shape,)
    else:
        shape = tuple(shape)

    # HDF5 does not support ranks greater than 32
    if len(shape) > 32:
        raise ValueError(
            "Shapes with rank > 32 are not supported: %s" % (shape,))

    return shape


def get_shape_itemsize_str(shape):
    """Get the shape and itemsize for a String in a generic Col object"""

    if shape == ():
        shape = [1]
    else:
        shape = list(shape)
    itemsize = shape.pop()
    shape = tuple(shape)
    return shape, itemsize


def checkIndexable(dtype):
    """Raise `TypeError` if the data type `dtype` is not indexable."""
    if dtype.kind == 'c':
        raise TypeError("complex columns can not be indexed")
    if dtype.shape != ():
        raise TypeError("only scalar columns can be indexed")



class Col(object):
    """Defines a general column that supports all NumPy data types.

    The ``dtype`` argument will accept any NumPy dtype, NumPy scalar
    type or PyTables datatype (string). However, a multidimensional
    NumPy dtype will not be accepted. If you want to declare
    multidimensional columns, use an scalar dtype and pass the
    dimensions in the ``shape`` argument.
    
    """

    # This class should become abstract somewhere in the future,
    # with no methods beyond __init__() or __repr__().
    # An auxiliary function returning an object of the adequate class
    # should be used instead.
    # So, the following _set*() methods should be removed in the future.
    # ivilata(2004-12-17)

    def _setPosition(self, pos):
        "Sets the '_v_pos' attribute."
        self._v_pos = pos


    def _setType(self, type_, shape):
        "Sets the 'dtype' and 'ptype' attributes."

        # Try to convert the type_ into a dtype (in order to accept
        # dtype string representations as input as well)
        try:
            type_ = numpy.dtype(type_)
        except TypeError:
            pass
        
        shape = normalize_shape(shape)

        # Check if type_ is a pytables type
        if type(type_) == str:
            if type_ == 'String':
                shape, itemsize = get_shape_itemsize_str(shape)
                self.dtype = numpy.dtype(("S%s"%itemsize, shape))
                self.ptype = 'String'
            elif type_ in numpy.sctypeNA or type_ in numpy.sctypeDict:
                self.dtype = numpy.dtype((type_, shape))
                self.ptype = numpy.sctypeNA[self.dtype.base.type]
            elif type_ == 'Time32':
                self.dtype = numpy.dtype((numpy.int32, shape))
                self.ptype = 'Time32'
            elif type_ == 'Time64':
                self.dtype = numpy.dtype((numpy.float64, shape))
                self.ptype = 'Time64'
            else:
                raise TypeError,  "Illegal type: %s" % (type_,)
        # Check if type_ is a numpy dtype
        elif type(type_) == numpy.dtype:
            if type_.shape != ():
                # If dtype is not an scalar, add the dtype dimensions to shape
                lshape = list(shape)
                lshape.extend(type_.shape)
                shape = tuple(lshape)
                # ...and set the dtype to the scalar one
                type_ = type_.base
            if type_.kind == "S":
                # type_ is already a dtype, so it is not necessary to compute
                # the string itemsize from the shape
                self.dtype = numpy.dtype((type_, shape))
                self.ptype = 'String'
            elif type_.kind in ['b', 'i', 'u', 'f', 'c']:
                self.dtype = numpy.dtype((type_, shape))
                self.ptype = numpy.sctypeNA[type_.base.type]
            else:
                raise TypeError, "Illegal type: %s" % (type_,)
        # Chek if type_ is a numpy scalar
        elif type(type_) == type:
            if type_ == numpy.string_:
                shape, itemsize = get_shape_itemsize_str(shape)
                self.dtype = numpy.dtype(("S%s"%itemsize, shape))
                self.ptype = 'String'
            elif type_ in numpy.sctypeNA or type_ in numpy.sctypeDict:
                self.dtype = numpy.dtype((type_, shape))
                self.ptype = numpy.sctypeNA[self.dtype.base.type]
            else:
                raise TypeError, "Illegal type: %s" % (type_,)
        else:
            raise TypeError, "Illegal type: %s" % (type_,)


    def _setDefault(self, dflt):
        "Sets the 'dflt' attribute."
        # Create NumPy objects as defaults
        # This is better in order to serialize them as attributes
        if dflt is None:
            if self.dtype.base.type == numpy.string_:
                dflt = ""
            else:
                dflt = 0
        self.dflt = numpy.array(dflt, dtype=self.dtype.base)
        # 0-dim arrays will be representented as NumPy scalars
        # (PyTables attribute convention)
        if self.dflt.dtype.shape == ():
            self.dflt = self.dflt[()]


    def _setIndex(self, indexed):
        if indexed:
            checkIndexable(self.dtype)
        self.indexed = indexed


    def __init__(self, dtype='Float64', shape=1, dflt=None, pos=None,
                 indexed=False):
        self._setType(dtype, shape)
        self._setDefault(dflt)
        self._setIndex(indexed)
        self._setPosition(pos)


    def __repr__(self):
        shape = self.dtype.shape
        return "Col(dtype=%r, shape=%s, dflt=%s, pos=%s, indexed=%s)" % (
            self.ptype, shape, self.dflt, self._v_pos, self.indexed)



class StringCol(Col):
    "Defines a string column."

    def _setType(self, length, shape):
        self.dtype = numpy.dtype(("S%s"%length, shape))
        self.ptype = "String"


    def __init__(self, length=None, dflt=None, shape=1, pos=None,
                 indexed=False):

        # Some more work needed for constructor call idiosyncrasies:
        # 'itemsize' is deduced from the default value if not specified.
        if length is None and dflt:
            length = len(dflt)  # 'dflt' has already been checked
            # NumPy explicitely forbids 0-sized arrays
            if length == 0:
                length = 1

        if length is None:
            raise ValueError("""\
You must specify at least a length or a default value
  where this length can be inferred from.""")

        # Set the basic attributes
        self._setType(length, shape)
        self._setDefault(dflt)
        self._setIndex(indexed)
        self._setPosition(pos)


    def __repr__(self):
        itemsize = self.dtype.base.itemsize
        return ("StringCol(length=%s, dflt=%r, shape=%s, pos=%s, indexed=%s)"
                % (itemsize, self.dflt, self.dtype.shape, self._v_pos,
                   self.indexed))



class BoolCol(Col):
    "Defines a boolean column."

    def __init__(self, dflt = False, shape = 1, pos = None, indexed = False):
        Col.__init__(self, dtype = 'Bool', dflt = dflt, shape = shape,
                     pos = pos, indexed = indexed)


    def __repr__(self):
        return "BoolCol(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)



class IntCol(Col):
    "Defines an integer column."

    def _setType(self, itemsize, sign, shape):
        if itemsize not in (1, 2, 4, 8):
            raise ValueError("""\
Integer itemsizes different from 1, 2, 4 or 8 are not supported""")

        if sign:
            self.dtype = numpy.dtype(('i%s'%itemsize, shape))
        else:
            self.dtype = numpy.dtype(('u%s'%itemsize, shape))

        self.ptype = numpy.sctypeNA[self.dtype.base.type]


    def __init__(self, dflt = 0, shape = 1, itemsize = 4, sign = 1,
                 pos = None, indexed = False):
        # This method is overridden to build item type from size and sign
        self._setType(itemsize, sign, shape)
        self._setDefault(dflt)
        self._setIndex(indexed)
        self._setPosition(pos)


    def __repr__(self):
        if numpy.array(0, self.dtype) - numpy.array(1, self.dtype) < 0:
            sign = True
        else:
            sign = False

        itemsize = self.dtype.base.itemsize
        return """\
IntCol(dflt=%s, shape=%s, itemsize=%s, sign=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.dtype.shape, itemsize, sign, self._v_pos,
            self.indexed)


class Int8Col(IntCol):
    "Description class for a signed integer of 8 bits."

    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = False):
        IntCol.__init__(self, dflt, itemsize = 1, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int8Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class UInt8Col(IntCol):
    "Description class for an unsigned integer of 8 bits."

    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 1, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt8Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class Int16Col(IntCol):
    "Description class for a signed integer of 16 bits."

    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 2, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int16Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class UInt16Col(IntCol):
    "Description class for an unsigned integer of 16 bits."

    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 2, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt16Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class Int32Col(IntCol):
    "Description class for a signed integer of 32 bits."

    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = False):
        IntCol.__init__(self, dflt , itemsize=4, shape=shape, sign=1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class UInt32Col(IntCol):
    "Description class for an unsigned integer of 32 bits."

    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 4, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class Int64Col(IntCol):
    "Description class for a signed integer of 64 bits."

    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 8, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class UInt64Col(IntCol):
    "Description class for an unsigned integer of 64 bits."

    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 8, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)



class FloatCol(Col):
    "Defines a float column."

    def _setType(self, itemsize, shape):
        if itemsize not in (4, 8):
            raise ValueError("""\
Float itemsizes different from 4 or 8 are not supported""")

        self.dtype = numpy.dtype(('f%s'%itemsize, shape))
        self.ptype = numpy.sctypeNA[self.dtype.base.type]

    def __init__(self, dflt = 0.0, shape = 1, itemsize = 8, pos = None,
                 indexed = False):
        # This method is overridden to build item type from size
        self._setType(itemsize, shape)
        self._setDefault(dflt)
        self._setIndex(indexed)
        self._setPosition(pos)

    def __repr__(self):
        itemsize = self.dtype.base.itemsize
        return """\
FloatCol(dflt=%s, shape=%s, itemsize=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.dtype.shape, itemsize, self._v_pos, self.indexed)


class Float32Col(FloatCol):
    "Description class for a floating point of 32 bits."

    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = False):
        FloatCol.__init__(self, dflt , shape = shape, itemsize = 4,
                          pos = pos, indexed = indexed)
    def __repr__(self):
        return "Float32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class Float64Col(FloatCol):
    "Description class for a floating point of 64 bits."

    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = False):
        FloatCol.__init__(self, dflt , shape = shape, itemsize = 8,
                          pos = pos, indexed = indexed)
    def __repr__(self):
        return "Float64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)



class ComplexCol(Col):
    "Defines a complex column."

    def _setType(self, itemsize, shape):
        if itemsize not in (8, 16):
            raise ValueError("""\
Complex itemsizes different from 8 or 16 are not supported""")

        self.dtype = numpy.dtype(('c%s'%itemsize, shape))
        self.ptype = numpy.sctypeNA[self.dtype.base.type]

    def __init__(self, dflt=(0.0+0.0j), shape=1, itemsize=16, pos=None):
        # This method is overridden to build item type from size
        self._setType(itemsize, shape)
        self._setDefault(dflt)
        self._setPosition(pos)
        self.indexed = False

    def __repr__(self):
        itemsize = self.dtype.base.itemsize
        return "ComplexCol(dflt=%s, shape=%s, itemsize=%s, pos=%s)" % (
            self.dflt, self.dtype.shape, itemsize, self._v_pos)


class Complex32Col(ComplexCol):
    "Description class for a complex of simple precision."

    def __init__(self, dflt = (0.0+0.0j), shape = 1, pos = None):
        ComplexCol.__init__(self, dflt, shape=shape, itemsize=8, pos=pos)
    def __repr__(self):
        return "Complex32Col(dflt=%s, shape=%s, pos=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos)


class Complex64Col(ComplexCol):
    "Description class for a complex of double precision."

    def __init__(self, dflt = (0.0+0.0j), shape = 1, pos = None):
        ComplexCol.__init__(self, dflt , shape=shape, itemsize=16, pos=pos)
    def __repr__(self):
        return "Complex64Col(dflt=%s, shape=%s, pos=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos)



class TimeCol(Col):
    "Defines a time column."

    # There are two distinct supported kinds of date:
    # the first is a 32 bit integer value (Time32Col)
    # and the second a 64 bit floating point value (Time64Col).
    # Both of them reflect the number of seconds since the Epoch.
    # This column has the property of being stored
    # using the HDF5 time datatypes.
    # ivb(2004-12-14)

    def _setType(self, itemsize, shape):
        if itemsize not in (4, 8):
            raise ValueError("""\
Time itemsizes different from 4 or 8 are not supported""")

        # Since Time columns have no NumPy type of their own,
        # a special case is made for them.
        if itemsize == 4:
            self.dtype = numpy.dtype((numpy.int32, shape))
            self.ptype = 'Time32'
        elif itemsize == 8:
            self.dtype = numpy.dtype((numpy.float64, shape))
            self.ptype = 'Time64'

    def __init__(self, dflt = 0, shape = 1, itemsize = 8, pos = None,
                 indexed = False):
        # This method is overridden to build item type from size
        self._setType(itemsize, shape)
        self._setDefault(dflt)
        self._setIndex(indexed)
        self._setPosition(pos)

    def __repr__(self):
        itemsize = self.dtype.base.itemsize
        return """\
TimeCol(dflt=%s, shape=%s, itemsize=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.dtype.shape, itemsize, self._v_pos, self.indexed)


class Time32Col(TimeCol):
    "Description class for an integer time of 32 bits."

    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = False):
        TimeCol.__init__(self, dflt , shape = shape, itemsize = 4,
                         pos = pos, indexed = indexed)
    def __repr__(self):
        return "Time32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class Time64Col(TimeCol):
    "Description class for a floating point time of 64 bits."

    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = False):
        TimeCol.__init__(self, dflt , shape = shape, itemsize = 8,
                         pos = pos, indexed = indexed)
    def __repr__(self):
        return "Time64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.dtype.shape, self._v_pos, self.indexed)


class EnumCol(Col):

    """
    Description of a column of an enumerated type.

    Instances of this class describe a table column which stores
    enumerated values.  Those values belong to an enumerated type,
    defined by the first argument (``enum``) in the constructor of
    `EnumCol`, which accepts the same kinds of arguments as `Enum`.  The
    enumerated type is stored in the ``enum`` attribute of the column.

    A default value must be specified as the second argument (``dflt``)
    in the constructor; it must be the *name* (a string) of one of the
    enumerated values in the enumerated type.  Once the column is
    created, the corresponding concrete value is stored in its ``dflt``
    attribute.  If the name does not match any value in the enumerated
    type, a ``KeyError`` is raised.

    A NumPy data type might be specified in order to determine the base
    type used for storing the values of enumerated values in memory and
    disk.  The data type must be able to represent each and every
    concrete value in the enumeration.  If it is not, a ``TypeError`` is
    raised.  The default base type is unsigned 32-bit integer, which is
    sufficient for most cases.

    The ``ptype`` attribute of enumerated columns is always ``'Enum'``,
    while the ``dtype`` attribute is the data type used for storing
    concrete values.

    The shape, position and indexed attributes of the column are treated
    as with other column description objects (see `Col`).

    Examples
    --------

    The next C ``enum`` construction::

      enum myEnum {
        T0,
        T1,
        T2
      };

    would correspond to the following PyTables declaration:

    >>> myEnumCol = EnumCol(['T0', 'T1', 'T2'], 'T0')

    Please note the ``dflt`` argument with a value of ``'T0'``.  Since
    the concrete value matching ``T0`` is unknown right now (we have not
    used explicit concrete values), using the name is the only option
    left for defining a default value for the column.

    The default representation of values in this enumerated column uses
    unsigned 32-bit integers, which surely wastes quite a lot of memory.
    Another size could be selected by using the ``dtype`` argument:

    >>> myEnumCol = EnumCol(['T0', 'T1', 'T2'], 'T0', dtype='UInt8')

    You can also define multidimensional arrays for data elements:

    >>> myEnumCol = EnumCol(
    ...    ['T0', 'T1', 'T2'], 'T0', dtype='UInt32', shape=(3,2))

    for 3x2 arrays of ``UInt32``.

    You should be able to index enumerated columns as well:

    >>> myEnumCol = EnumCol(['T0', 'T1', 'T2'], 'T0', indexed=True)

    This can only be applied, of course, to scalar data elements.
    """

    def __init__(self, enum, dflt, dtype='UInt32', shape=1, pos=None,
                 indexed=False):
        self._setEnum(enum)
        Col.__init__(self, dtype, shape, dflt, pos, indexed)


    def _setEnum(self, enum):
        if not isinstance(enum, Enum):
            enum = Enum(enum)

        self.enum = enum
        """The associated `Enum` instance."""

        values = [value for (name, value) in enum]
        try:
            asArray = numpy.array(values)

            # Check integer type of concrete values.
            if not asArray.dtype.kind in ['i', 'u']:
                raise NotImplementedError("""\
sorry, only integer concrete values are supported for the moment""")

            # Check scalar shape of concrete values.
            if len(asArray.shape) > 1:
                raise NotImplementedError("""\
sorry, only scalar concrete values are supported for the moment""")

        except ValueError:
            # Check common shape of concrete values.
            raise NotImplementedError("""\
sorry, only uniformly-shaped concrete values are supported for the moment""")

        except TypeError:
            # Check numeric type of concrete values.
            raise NotImplementedError("""\
sorry, only numeric concrete values are supported for the moment""")


    def _setType(self, type_, shape):
        # Check integer type of representation.
        if not numpy.dtype(type_).kind in ['i', 'u']:
            raise NotImplementedError("""\
sorry, only integer concrete values type are supported for the moment""")

        names = []
        values = []
        for (name, value) in self.enum:
            names.append(name)
            values.append(value)

        # Check that type can represent concrete values.
        encoded = numpy.array(values, type_)
        if values != encoded.tolist():
            raise TypeError("""\
type ``%s`` can not represent all concrete values in the enumeration"""
                            % type_)

        self._npNames = names
        """List of enumerated names."""
        self._npValues = encoded
        """List of enumerated concrete values."""
        self.dtype = numpy.dtype((type_, shape))
        self.ptype = 'Enum'


    def _setDefault(self, dflt):
        if not isinstance(dflt, basestring):
            raise TypeError(
                "name of default enumerated value is not a string: %r"
                % (dflt,))

        self.dflt = numpy.array(self.enum[dflt], dtype=self.dtype.base)


    def __repr__(self):
        underptype = numpy.sctypeNA[self.dtype.base.type]
        return ('EnumCol(%s, %r, dtype=\'%s\', shape=%s, pos=%s, indexed=%s)'
                % (self.enum, self.enum(self.dflt.tolist()),
                   underptype, self.dtype.shape, self._v_pos, self.indexed))



class Description(object):
    """
    Description of the structure of a table.

    An instance of this class is automatically bound to `Table` objects
    when they are created.  It provides a browseable representation of
    the structure of the table, made of non-nested (`Col`) and nested
    (`Description`) columns.  It also contains information that will
    allow you to build ``RecArray`` objects suited for the different
    columns in a table (be they nested or not).

    Columns under a description can be accessed as attributes of it.
    For instance, if ``desc`` is a ``Description`` instance with a colum
    named ``col1`` under it, the later can be accessed as ``desc.col1``.
    If ``col1`` is nested and contains a ``col2`` column, this can be
    accessed as ``desc.col1.col2``.

    Instance variables:

    _v_names
        The name of this description group. The name of the root group
        is '/'.

    _v_names
        A list of the names of the columns hanging directly from the
        associated table or nested column.  The order of the names
        matches the order of their respective columns in the containing
        table.

    _v_nestedNames
        A nested list of the names of all the columns under this table
        or nested column.  You can use this for the ``names`` argument
        of ``NestedRecArray`` factory functions.

    _v_nestedFormats
        A nested list of the NumPy string formats (and shapes) of all
        the columns under this table or nested column.  You can use this
        for the ``formats`` argument of ``NestedRecArray`` factory
        functions.

    _v_nestedDescr
        A nested list of pairs of ``(name, format)`` tuples for all the
        columns under this table or nested column.  You can use this for
        the ``descr`` argument of ``NestedRecArray`` factory functions.

    _v_dtypes
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective NumPy types.

    _v_ptypes
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective PyTables types.

    _v_dflts
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective default values.

    _v_colObjects
        A dictionary mapping the names of the columns hanging directly
        from the associated table or nested column to their respective
        descriptions (`Col` or `Description` instances).

    _v_nestedlvl
        The level of the associated table or nested column in the nested
        datatype.

    _v_is_nested
        Either the associated table has nested columns or not (Boolean).


    Public methods:

    _f_walk([type])
        Iterate over nested columns.

    """

    def __init__(self, classdict, nestedlvl=-1):

        # Do a shallow copy of classdict just in case this is going to
        # be shared by other instances
        #self.classdict = classdict.copy()
        # I think this is not necessary
        self.classdict = classdict
        keys = classdict.keys()
        newdict = self.__dict__
        newdict["_v_name"] = "/"   # The name for root descriptor
        newdict["_v_names"] = []
        newdict["_v_dtypes"] = {}
        newdict["_v_ptypes"] = {}
        newdict["_v_dflts"] = {}
        newdict["_v_colObjects"] = {}
        newdict["_v_is_nested"] = False
        nestedFormats = []
        nestedDType = []

        if not hasattr(newdict, "_v_nestedlvl"):
            newdict["_v_nestedlvl"] = nestedlvl + 1

        # __check_validity__ must be check out prior to the keys loop
        if "__check_validity__" in keys:
            check_validity = classdict["__check_validity__"]
        else:
            check_validity = 1   # Default value for name validity check

        # Check for special variables
        for k in keys[:]:
            object = classdict[k]
            if (k.startswith('__') or k.startswith('_v_')):
                if k in newdict:
                    print "Warning!"
                    # special methods &c: copy to newdict, warn about conflicts
                    warnings.warn("Can't set attr %r in description class %r" \
                                  % (k, self))
                else:
                    #print "Special variable!-->", k, classdict[k]
                    newdict[k] = classdict[k]
                    keys.remove(k)  # This variable is not needed anymore

            elif (type(object) == type(IsDescription) and
                issubclass(object, IsDescription)):
                #print "Nested object (type I)-->", k
                descr = object()
                # Doing a deepcopy is very important when one has nested
                # records in the form:
                #
                # class Nested(IsDescription):
                #     uid = IntCol()
                #
                # class B_Candidate(IsDescription):
                #     nested1 = Nested
                #     nested2 = Nested
                #
                # This makes that nested1 and nested2 point to the same
                # 'columns' dictionary, so that successive accesses to
                # the different columns are actually accessing to the
                # very same object.
                # F. Altet 2006-08-22
                columns = copy.deepcopy(object().columns)
                classdict[k] = Description(columns, self._v_nestedlvl)
            elif (type(object.__class__) == type(IsDescription) and
                issubclass(object.__class__, IsDescription)):
                #print "Nested object (type II)-->", k
                # Regarding the need of a deepcopy, see note above
                columns = copy.deepcopy(object.columns)
                classdict[k] = Description(columns, self._v_nestedlvl)
            elif isinstance(object, dict):
                #print "Nested object (type III)-->", k
                # Regarding the need of a deepcopy, see note above
                columns = copy.deepcopy(object)
                classdict[k] = Description(columns, self._v_nestedlvl)

        # Check if we have any ._v_pos position attribute
        for column in classdict.values():
            if hasattr(column, "_v_pos") and column._v_pos:
                keys.sort(self._g_cmpkeys)
                break
        else:
            # No ._v_pos was set
            # fall back to alphanumerical order
            keys.sort()

        pos = 0
        # Get properties for compound types
        for k in keys:
            # Class variables
            if check_validity:
                # Check for key name validity
                checkNameValidity(k)
            object = classdict[k]
            newdict[k] = object    # To allow natural naming
            if not (isinstance(object, Col) or
                    isinstance(object, Description)):
                raise TypeError, \
"""Passing an incorrect value to a table column. Expected a Col (or
  subclass) instance and got: "%s". Please, make use of the Col(), or
  descendant, constructor to properly initialize columns.
""" % object
            object._v_pos = pos  # Set the position of this object
            object._v_parent = self  # The parent description
            pos += 1
            newdict['_v_colObjects'][k] = object
            newdict['_v_names'].append(k)
            object.__dict__['_v_name'] = k
            if isinstance(object, Col):
                dtype = object.dtype
                newdict['_v_dtypes'][k] = dtype
                newdict['_v_ptypes'][k] = object.ptype
                newdict['_v_dflts'][k] = object.dflt
                baserecarrtype = dtype.base.str[1:]
                object.recarrtype = str(dtype.shape) + baserecarrtype
                nestedFormats.append(object.recarrtype)
                nestedDType.append((k, baserecarrtype, dtype.shape))
            else:  # A description
                nestedFormats.append(object._v_nestedFormats)
                nestedDType.append((k, object._v_dtype))

        # Assign the format list to _v_nestedFormats
        newdict['_v_nestedFormats'] = nestedFormats
        newdict['_v_dtype'] = numpy.dtype(nestedDType)
        if self._v_nestedlvl == 0:
            # Get recursively nested _v_nestedNames and _v_nestedDescr attrs
            self._g_setNestedNamesDescr()
            # Get pathnames for nested groups
            self._g_setPathNames()
            # Assign the byteorder (if not yet)
            if not hasattr(self, "_v_byteorder"):
                newdict["_v_byteorder"] = sys.byteorder

        # finally delegate the rest of the work to type.__new__
        return


    def _g_cmpkeys(self, key1, key2):
        """Helps .sort() to respect pos field in type definition"""
        # Do not try to order variables that starts with special
        # prefixes
        if ((key1.startswith('__') or key1.startswith('_v_')) and
            (key2.startswith('__') or key2.startswith('_v_'))):
            return 0
        # A variable that starts with a special prefix
        # is always greater than a normal variable
        elif (key1.startswith('__') or key1.startswith('_v_')):
            return 1
        elif (key2.startswith('__') or key2.startswith('_v_')):
            return -1
        pos1 = getattr(self.classdict[key1], "_v_pos", None)
        pos2 = getattr(self.classdict[key2], "_v_pos", None)
#         print "key1 -->", key1, pos1
#         print "key2 -->", key2, pos2
        # pos = None is always greater than a number
        if pos1 is None:
            return 1
        if pos2 is None:
            return -1
        if pos1 < pos2:
            return -1
        if pos1 == pos2:
            return 0
        if pos1 > pos2:
            return 1


    def _g_setNestedNamesDescr(self):
        """Computes the nested names and descriptions for nested datatypes.
        """
        names = self._v_names
        fmts = self._v_nestedFormats
        self._v_nestedNames = names[:]  # Important to do a copy!
        self._v_nestedDescr = [(names[i], fmts[i]) for i in range(len(names))]
        for i in range(len(names)):
            name = names[i]
            new_object = self._v_colObjects[name]
            if isinstance(new_object, Description):
                new_object._g_setNestedNamesDescr()
                # replace the column nested name by a correct tuple
                self._v_nestedNames[i] = (name, new_object._v_nestedNames)
                self._v_nestedDescr[i] = (name, new_object._v_nestedDescr)
                # set the _v_is_nested flag
                self._v_is_nested = True


    def _g_setPathNames(self):
        """Compute the pathnames for arbitrary nested descriptions.

        This method sets the ``_v_pathname`` and ``_v_pathnames``
        attributes of all the elements (both descriptions and columns)
        in this nested description.
        """

        def getColsInOrder(description):
            return [description._v_colObjects[colname]
                    for colname in description._v_names]

        def joinPaths(path1, path2):
            if not path1:
                return path2
            return '%s/%s' % (path1, path2)

        # The top of the stack always has a nested description
        # and a list of its child columns
        # (be they nested ``Description`` or non-nested ``Col`` objects).
        # In the end, the list contains only a list of column paths
        # under this one.
        #
        # For instance, given this top of the stack::
        #
        #   (<Description X>, [<Column A>, <Column B>])
        #
        # After computing the rest of the stack, the top is::
        #
        #   (<Description X>, ['a', 'a/m', 'a/n', ... , 'b', ...])

        stack = []

        # We start by pushing the top-level description
        # and its child columns.
        self._v_pathname = ''
        stack.append((self, getColsInOrder(self)))

        while stack:
            desc, cols = stack.pop()
            head = cols[0]

            # What's the first child in the list?
            if isinstance(head, Description):
                # A nested description.  We remove it from the list and
                # push it with its child columns.  This will be the next
                # handled description.
                head._v_pathname = joinPaths(desc._v_pathname, head._v_name)
                stack.append((desc, cols[1:]))  # alter the top
                stack.append((head, getColsInOrder(head)))  # new top
            elif isinstance(head, Col):
                # A non-nested column.  We simply remove it from the
                # list and append its name to it.
                head._v_pathname = joinPaths(desc._v_pathname, head._v_name)
                cols.append(head._v_name)  # alter the top
                stack.append((desc, cols[1:]))  # alter the top
            else:
                # Since paths and names are appended *to the end* of
                # children lists, a string signals that no more children
                # remain to be processed, so we are done with the
                # description at the top of the stack.
                assert isinstance(head, basestring)
                # Assign the computed set of descendent column paths.
                desc._v_pathnames = cols
                if len(stack) > 0:
                    # Compute the paths with respect to the parent node
                    # (including the path of the current description)
                    # and append them to its list.
                    descName = desc._v_name
                    colPaths = [joinPaths(descName, path) for path in cols]
                    colPaths.insert(0, descName)
                    parentCols = stack[-1][1]
                    parentCols.extend(colPaths)
                # (Nothing is pushed, we are done with this description.)


    def _f_walk(self, type="All"):
        """
        Iterate over nested columns.

        If `type` is ``'All'`` (the default), all column description
        objects (`Col` and `Description` instances) are returned in
        top-to-bottom order (pre-order).

        If `type` is ``'Col'`` or ``'Description'``, only column
        or descriptions of the specified type are returned.
        """

        if type not in ["All", "Col", "Description"]:
            raise ValueError("""\
type can only take the parameters 'All', 'Col' or 'Description'.""")

        stack = [self]
        while stack:
            object = stack.pop(0)  # pop at the front so as to ensure the order
            if type in ["All", "Description"]:
                yield object  # yield description
            names = object._v_names
            for i in range(len(names)):
                new_object = object._v_colObjects[names[i]]
                if isinstance(new_object, Description):
                    stack.append(new_object)
                else:
                    if type in ["All", "Col"]:
                        yield new_object  # yield column


    def __repr__(self):
        """ Gives a detailed Description column representation.
        """
        rep = [ '%s\"%s\": %r' %  \
                ("  "*self._v_nestedlvl, k, self._v_colObjects[k])
                for k in self._v_names]
        return '{\n  %s}' % (',\n  '.join(rep))


    def __str__(self):
        """ Gives a brief Description representation.
        """
        return 'Description(%s)' % self._v_nestedDescr



class metaIsDescription(type):
    "Helper metaclass to return the class variables as a dictionary "

    def __new__(cls, classname, bases, classdict):
        """ Return a new class with a "columns" attribute filled
        """

        newdict = {"columns":{},
                   }
        for k in classdict.keys():
            #if not (k.startswith('__') or k.startswith('_v_')):
            # We let pass _v_ variables to configure class behaviour
            if not (k.startswith('__')):
                newdict["columns"][k] = classdict[k]

        # Return a new class with the "columns" attribute filled
        return type.__new__(cls, classname, bases, newdict)



class IsDescription(object):
    """ For convenience: inheriting from IsDescription can be used to get
        the new metaclass (same as defining __metaclass__ yourself).
    """
    __metaclass__ = metaIsDescription



if __name__=="__main__":
    """Test code"""

    class Info(IsDescription):
        _v_pos = 2
        Name = UInt32Col()
        Value = Float64Col()

    class Test(IsDescription):
        """A description that has several columns"""
        x = Col("Int32", 2, 0, pos=0)
        y = FloatCol(1, shape=(2,3))
        z = UInt8Col(1)
        color = StringCol(2, " ")
        #color = UInt32Col(2)
        Info = Info()
        class info(IsDescription):
            _v_pos = 1
            name = UInt32Col()
            value = Float64Col(pos=0)
            y2 = FloatCol(1, shape=(2,3), pos=1)
            z2 = UInt8Col(1)
            class info2(IsDescription):
                y3 = FloatCol(1, shape=(2,3))
                z3 = UInt8Col(1)
                name = UInt32Col()
                value = Float64Col()
                class info3(IsDescription):
                    name = UInt32Col()
                    value = Float64Col()
                    y4 = FloatCol(1, shape=(2,3))
                    z4 = UInt8Col(1)

#     class Info(IsDescription):
#         _v_pos = 2
#         Name = StringCol(length=2)
#         Value = Complex64Col()

#     class Test(IsDescription):
#         """A description that has several columns"""
#         x = Col("Int32", 2, 0, pos=0)
#         y = FloatCol(1, shape=(2,3))
#         z = UInt8Col(1)
#         color = StringCol(2, " ")
#         Info = Info()
#         class info(IsDescription):
#             _v_pos = 1
#             name = StringCol(length=2)
#             value = Complex64Col(pos=0)
#             y2 = FloatCol(1, shape=(2,3), pos=1)
#             z2 = UInt8Col(1)
#             class info2(IsDescription):
#                 y3 = FloatCol(1, shape=(2,3))
#                 z3 = UInt8Col(1)
#                 name = StringCol(length=2)
#                 value = Complex64Col()
#                 class info3(IsDescription):
#                     name = StringCol(length=2)
#                     value = Complex64Col()
#                     y4 = FloatCol(1, shape=(2,3))
#                     z4 = UInt8Col(1)

    # example cases of class Test
    klass = Test()
    #klass = Info()
    desc = Description(klass.columns)
    print "Description representation (short) ==>", desc
    print "Description representation (long) ==>", repr(desc)
    print "Column names ==>", desc._v_names
    print "Column x ==>", desc.x
    print "Column Info ==>", desc.Info
    print "Column Info.value ==>", desc.Info.Value
    print "Nested column names  ==>", desc._v_nestedNames
    print "Defaults ==>", desc._v_dflts
    print "Nested Formats ==>", desc._v_nestedFormats
    print "Nested Descriptions ==>", desc._v_nestedDescr
    print "Nested Descriptions (info) ==>", desc.info._v_nestedDescr
    print "Total size ==>", desc._v_dtype.itemsize


    # check _f_walk
    for object in desc._f_walk():
        if isinstance(object, Description):
            print "******begin object*************",
            print "name -->", object._v_name
            #print "name -->", object._v_dtype.name
            #print "object childs-->", object._v_names
            #print "object nested childs-->", object._v_nestedNames
            print "totalsize-->", object._v_dtype.itemsize
        else:
            #pass
            print "leaf -->", object._v_name, object.dtype



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
