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

    metaIsDescription
    IsDescription

Functions:

Misc variables:

    __version__

"""

import warnings
import sys
import operator

import numarray as NA
import numarray.records as records
import numarray.strings as strings

from tables.enum import Enum
from tables.utils import checkNameValidity
import tables


__version__ = "$Revision$"


# Translation tables for numarray datatypes and recarray formats
recarrfmt = {
    'a':   records.CharType,
    'b1':  NA.Bool,
    'i1':  NA.Int8,
    'i2':  NA.Int16,
    'i4':  NA.Int32,
    'i8':  NA.Int64,
    'u1':  NA.UInt8,
    'u2':  NA.UInt16,
    'u4':  NA.UInt32,
    'u8':  NA.UInt64,
    'f4':  NA.Float32,
    'f8':  NA.Float64,
    'c8':  NA.Complex32,
    'c16': NA.Complex64,
    }

# the reverse translation table of the above
revrecarrfmt = {}
for key in recarrfmt.keys():
    revrecarrfmt[recarrfmt[key]]=key


class ShapeMixin:
    "Mix-in class for standard shape handling."

    def _setShape(self, shape):
        "Sets the 'shape' and 'itemsize' attributes. Uses 'self.type'."

        if type(shape) not in (int, long, tuple, list):
            raise ValueError("Illegal shape object: %s" % (shape,))

        # Turn shape into 1 or a properly-formed tuple
        if type(shape) in (int, long):
            if shape < 1:
                raise ValueError(
                    "Shape value must be greater than 0: %s" % (shape,))
            elif shape == 1:
                self.shape = shape
            else:
                # To prevent confusions between 2 and (2,):
                # the meaning is the same
                self.shape = (shape,)
        else:
            assert type(shape) in (tuple, list)
            # HDF5 does not support ranks greater than 32
            if len(shape) > 32:
                raise ValueError(
                    "Shapes with rank > 32 are not supported: %s" % (shape,))
            self.shape = tuple(shape)

        # Set itemsize
        self.itemsize = self.type.bytes


    def _setIndex(self, indexed):
        "Sets the 'indexed' attribute."
        if indexed and self.shape != 1:
            raise TypeError("only columns with shape 1 can be indexed")
        self.indexed = indexed



def setDefaultString(dflt):
    "Set a default value for strings. Valid for Col and StringCol instances."

    if dflt != None and not (type(dflt) in [str, list, tuple] or
                             type(dflt) == strings.CharArray):
        raise ValueError("Invalid default value: %s" % (dflt,))


class Col(ShapeMixin, object):
    "Defines a general column that supports all numarray data types."

    # This class should become abstract somewhere in the future,
    # with no methods beyond __init__() or __repr__().
    # An auxiliary function returning an object of the adequate class
    # should be used instead.
    # So, the following _set*() methods should be removed in the future.
    # ivilata(2004-12-17)

    def _setPosition(self, pos):
        "Sets the '_v_pos' attribute."
        self._v_pos = pos

    def _setType(self, type_):
        "Sets the 'type', 'recarrtype' and 'stype' attributes."
        if type_ in NA.typeDict:
            self.type = NA.typeDict[type_]
            self.stype = str(self.type)
        elif type_ == 'Time32':
            self.type = NA.Int32  # special case for times
            self.stype = type_
        elif type_ == 'Time64':
            self.type = NA.Float64  # special case for times
            self.stype = type_
        elif type_ == 'CharType' or isinstance(type_, records.Char):
            self.type = records.CharType  # special case for strings
            self.stype = str(self.type)
        else:
            raise TypeError, "Illegal type: %s" % (type_,)

        self.recarrtype = revrecarrfmt[self.type]

    def _setDefault(self, dflt):
        "Sets the 'dflt' attribute."
        # Create NumArray or CharArray objects as defaults
        # This is better in order to serialize then as attributes
        if self.type == records.CharType:
            # We should raise an error in case dflt is None, but that
            # makes several tests in test_create.py to fail
            # and I don't have time right now to look into this.
##            if dflt is None:
##                raise ValueError, \
##"You must set a default value for CharType when using the Col constructor."
##            elif dflt == "":
##                # Putting just to "" gives problems with numarray
##                dflt = "\x00"*self.itemsize
            if dflt is None or dflt == "":
                # Putting just to "" gives problems with numarray
                dflt = "\x00"
            # It is important to set the padding character to NULL in order
            # to avoid the '' string to become a ' ' after de-serializing.
            # See:
            # http://sourceforge.net/tracker/index.php?func=detail&aid=1304615&group_id=1369&atid=450446
            # for more info.
            self.dflt = strings.array(dflt, padc='\x00')
        else:
            if dflt is None:
                # We set the value to int(0), but it will be changed
                # to the appropriate type immediately after.
                dflt = 0
            self.dflt = NA.array(dflt, type=self.type)

    def _setIndex(self, indexed):
        if indexed and self.type in (NA.Complex32, NA.Complex64):
            raise TypeError("%r do not support indexation" % (self.type,))
        super(Col, self)._setIndex(indexed)

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

        # Override itemsize; strings still need some tampering with the shape
        #
        # This exposes Numeric BUG#1087158, since the comparison is True
        # for self.type = Float64, Int32, Complex64...
        # ivilata(2004-12-17)
        #
        if self.type is records.CharType:
            if type(shape) in (int, long):
                self.shape = 1
                self.itemsize = shape
            else:
                shape = list(self.shape)
                self.itemsize = shape.pop()
                if shape == ():
                    self.shape = 1
                elif len(shape) == 1:
                    #self.shape = shape[0]
                    # This is better for Atoms
                    self.shape = (shape[0],)
                else:
                    self.shape = tuple(shape)
            # In case of strings, this attribute is overwritten
            self.recarrtype = revrecarrfmt[self.type]+str(self.itemsize)

    def __init__(self, dtype = 'Float64', shape = 1, dflt = None, pos = None,
                 indexed = False):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(dtype)    -> type, recarrtype, stype
        2. _setDefault(dflt)  -> dflt
        3. _setShape(shape)   -> shape, itemsize
        4. _setIndex(indexed) -> indexed
        5. _setPosition(pos)  -> _v_pos
        """

        self._setType(dtype)
        self._setDefault(dflt)
        self._setShape(shape)
        self._setIndex(indexed)
        self._setPosition(pos)

    def __repr__(self):
        if self.type == 'CharType' or isinstance(self.type, records.Char):
            if self.shape == 1:
                shape = [self.itemsize]
            else:
                shape = list(self.shape)
                shape.append(self.itemsize)
            shape = tuple(shape)
        else:
            shape = self.shape

        return "Col(dtype=%r, shape=%s, dflt=%s, pos=%s, indexed=%s)" % (
            self.stype, shape, self.dflt, self._v_pos, self.indexed)



class StringCol(Col):
    "Defines a string column."

    def _setType(self, type_):
        self.type       = records.CharType
        self.stype      = str(records.CharType)

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

        # Set itemsize; forced to None to get it from the 'length' argument
        self.itemsize = None

    def _setDefault(self, dflt):
        "Sets the 'dflt' attribute (overwrites the method in Col)."
        if dflt is None or dflt == "":
            dflt = "\x00"*self.itemsize
        # It is important to set the padding character to NULL in order
        # to avoid the '' string to become a ' ' after de-serializing.
        # See:
        # http://sourceforge.net/tracker/index.php?func=detail&aid=1304615&group_id=1369&atid=450446
        # for more info.
        self.dflt = strings.array(dflt, padc='\x00')

    def __init__(self, length = None, dflt = None, shape = 1, pos = None,
                 indexed = False):

        # Some more work needed for constructor call idiosincracies:
        # 'itemsize' is deduced from the default value if not specified.
        if length is None and dflt:
            length = len(dflt)  # 'dflt' has already been checked
            # We explicitely forbid 0-sized arrays
            # (this creates too much headaches)
            if length == 0:
                length = 1

        if length is None:
            raise ValueError("""\
You must specify at least a length or a default value
  where this length can be inferred from.""")

        # This is set here just to be used in _setDefault()
        self.itemsize = length

        Col.__init__(
            self, dtype = 'CharType',
            dflt = dflt, shape = shape, pos = pos, indexed = indexed)
        # This needs to be set again (more indiosincracies :-()
        self.itemsize = length
        self.recarrtype = revrecarrfmt[self.type]+str(self.itemsize)

    def __repr__(self):
        return ("StringCol(length=%s, dflt=%r, shape=%s, pos=%s, indexed=%s)"
                % (self.itemsize, self.dflt, self.shape, self._v_pos,
                   self.indexed))


class BoolCol(Col):
    "Defines a boolean column."

    def _setType(self, type_):
        self.type       = NA.Bool
        self.stype      = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]

    def _setIndex(self, indexed):
        super(Col, self)._setIndex(indexed)

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = False, shape = 1, pos = None, indexed = False):
        Col.__init__(
            self, dtype = 'Bool',
            dflt = dflt, shape = shape, pos = pos, indexed = indexed)

    def __repr__(self):
        return "BoolCol(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)



class IntCol(Col):
    "Defines an integer column."

    def _setType(self, itemsize, sign):
        if itemsize not in (1, 2, 4, 8):
            raise ValueError("""\
Integer itemsizes different from 1, 2, 4 or 8 are not supported""")

        if itemsize == 1:
            if sign:
                self.type = NA.Int8
            else:
                self.type = NA.UInt8
        elif itemsize == 2:
            if sign:
                self.type = NA.Int16
            else:
                self.type = NA.UInt16
        elif itemsize == 4:
            if sign:
                self.type = NA.Int32
            else:
                self.type = NA.UInt32
        elif itemsize == 8:
            if sign:
                self.type = NA.Int64
            else:
                self.type = NA.UInt64

        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]

    def _setIndex(self, indexed):
        super(Col, self)._setIndex(indexed)

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = 0, shape = 1, itemsize = 4, sign = 1,
                 pos = None, indexed = False):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize, sign) -> type, recarrtype, stype
        2. _setDefault(dflt)        -> dflt
        3. _setShape(shape)         -> shape, itemsize
        4. _setIndex(indexed)       -> indexed
        5. _setPosition(pos)        -> pos
        """

        # This method is overridden to build item type from size and sign
        self._setType(itemsize, sign)
        self._setDefault(dflt)
        self._setShape(shape)
        self._setIndex(indexed)
        self._setPosition(pos)

    def __repr__(self):
        if NA.array(0, self.type)[()] - NA.array(1, self.type)[()] < 0:
            sign = 1
        else:
            sign = 0

        return """\
IntCol(dflt=%s, shape=%s, itemsize=%s, sign=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.shape, self.itemsize, sign, self._v_pos,
            self.indexed)

class Int8Col(IntCol):
    "Description class for a signed integer of 8 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int8
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = False):
        IntCol.__init__(self, dflt, itemsize = 1, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int8Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class UInt8Col(IntCol):
    "Description class for an unsigned integer of 8 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt8
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 1, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt8Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class Int16Col(IntCol):
    "Description class for a signed integer of 16 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int16
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 2, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int16Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class UInt16Col(IntCol):
    "Description class for an unsigned integer of 16 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt16
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 2, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt16Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class Int32Col(IntCol):
    "Description class for a signed integer of 32 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int32
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = False):
        IntCol.__init__(self, dflt , itemsize=4, shape=shape, sign=1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class UInt32Col(IntCol):
    "Description class for an unsigned integer of 32 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt32
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 4, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class Int64Col(IntCol):
    "Description class for a signed integer of 64 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int64
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 8, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class UInt64Col(IntCol):
    "Description class for an unsigned integer of 64 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt64
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=False):
        IntCol.__init__(self, dflt , itemsize = 8, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)



class FloatCol(Col):
    "Defines a float column."

    def _setType(self, itemsize):
        if itemsize not in (4, 8):
            raise ValueError("""\
Float itemsizes different from 4 or 8 are not supported""")

        if itemsize == 4:
            self.type = NA.Float32
        elif itemsize == 8:
            self.type = NA.Float64

        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]

    def _setIndex(self, indexed):
        super(Col, self)._setIndex(indexed)

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = 0.0, shape = 1, itemsize = 8, pos = None,
                 indexed = False):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize) -> type, recarrtype, stype
        2. _setDefault(dflt)  -> dflt
        3. _setShape(shape)   -> shape, itemsize
        4. _setIndex(indexed) -> indexed
        5. _setPosition(pos)  -> pos
        """

        # This method is overridden to build item type from size
        self._setType(itemsize)
        self._setDefault(dflt)
        self._setShape(shape)
        self._setIndex(indexed)
        self._setPosition(pos)

    def __repr__(self):
        return """\
FloatCol(dflt=%s, shape=%s, itemsize=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.shape, self.itemsize, self._v_pos, self.indexed)

class Float32Col(FloatCol):
    "Description class for a floating point of 32 bits."
    def _setType(self, itemsize):
        self.type = NA.Float32
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = False):
        FloatCol.__init__(self, dflt , shape = shape, itemsize = 4,
                          pos = pos, indexed = indexed)
    def __repr__(self):
        return "Float32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class Float64Col(FloatCol):
    "Description class for a floating point of 64 bits."
    def _setType(self, itemsize):
        self.type = NA.Float64
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = False):
        FloatCol.__init__(self, dflt , shape = shape, itemsize = 8,
                          pos = pos, indexed = indexed)
    def __repr__(self):
        return "Float64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)



class ComplexCol(Col):
    "Defines a complex column."

    def _setType(self, itemsize):
        if itemsize not in (8, 16):
            raise ValueError("""\
Complex itemsizes different from 8 or 16 are not supported""")

        if itemsize == 8:
            self.type = NA.Complex32
        elif itemsize == 16:
            self.type = NA.Complex64

        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = (0.0+0.0j), shape = 1, itemsize = 16, pos = None):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize) -> type, recarrtype, stype
        2. _setDefault(dflt)  -> dflt
        3. _setShape(shape)   -> shape, itemsize
        4. _setPosition(pos)  -> pos
        5.                    -> indexed
        """

        # This method is overridden to build item type from size
        self._setType(itemsize)
        self._setDefault(dflt)
        self._setShape(shape)
        self._setPosition(pos)
        self.indexed = False

    def __repr__(self):
        return "ComplexCol(dflt=%s, shape=%s, itemsize=%s, pos=%s)" % (
            self.dflt, self.shape, self.itemsize, self._v_pos)

class Complex32Col(ComplexCol):
    "Description class for a complex of simple precision."
    def _setType(self, itemsize):
        self.type = NA.Complex32
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = (0.0+0.0j), shape = 1, pos = None):
        ComplexCol.__init__(self, dflt, shape = shape, itemsize = 8, pos = pos)
    def __repr__(self):
        return "Complex32Col(dflt=%s, shape=%s, pos=%s)" % (
            self.dflt, self.shape, self._v_pos)

class Complex64Col(ComplexCol):
    "Description class for a complex of double precision."
    def _setType(self, itemsize):
        self.type = NA.Complex64
        self.stype = str(self.type)
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = (0.0+0.0j), shape = 1, pos = None):
        ComplexCol.__init__(self, dflt , shape = shape, itemsize = 16, pos = pos)
    def __repr__(self):
        return "Complex64Col(dflt=%s, shape=%s, pos=%s)" % (
            self.dflt, self.shape, self._v_pos)



class TimeCol(Col):
    "Defines a time column."

    # There are two distinct supported kinds of date:
    # the first is a 32 bit integer value (Time32Col)
    # and the second a 64 bit floating point value (Time64Col).
    # Both of them reflect the number of seconds since the Epoch.
    # This column has the property of being stored
    # using the HDF5 time datatypes.
    # ivb(2004-12-14)


    def _setType(self, itemsize):
        if itemsize not in (4, 8):
            raise ValueError("""\
Time itemsizes different from 4 or 8 are not supported""")

        # Since Time columns have no Numarray type of their own,
        # a special case is made for them.
        if itemsize == 4:
            self.type = NA.Int32
            self.stype = 'Time32'
        elif itemsize == 8:
            self.type = NA.Float64
            self.stype = 'Time64'

        self.recarrtype = revrecarrfmt[self.type]

    def _setIndex(self, indexed):
        super(Col, self)._setIndex(indexed)

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = 0, shape = 1, itemsize = 8, pos = None,
                 indexed = False):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize) -> type, recarrtype, stype
        2. _setDefault(dflt)  -> dflt
        3. _setShape(shape)   -> shape, itemsize
        4. _setIndex(indexed) -> indexed
        5. _setPosition(pos)  -> pos
        """

        # This method is overridden to build item type from size
        self._setType(itemsize)
        self._setDefault(dflt)
        self._setShape(shape)
        self._setIndex(indexed)
        self._setPosition(pos)

    def __repr__(self):
        return """\
TimeCol(dflt=%s, shape=%s, itemsize=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.shape, self.itemsize, self._v_pos, self.indexed)

class Time32Col(TimeCol):
    "Description class for an integer time of 32 bits."
    def _setType(self, itemsize):
        self.type = NA.Int32
        self.stype = 'Time32'
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = False):
        TimeCol.__init__(self, dflt , shape = shape, itemsize = 4,
                         pos = pos, indexed = indexed)
    def __repr__(self):
        return "Time32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)

class Time64Col(TimeCol):
    "Description class for a floating point time of 64 bits."
    def _setType(self, itemsize):
        self.type = NA.Float64
        self.stype = 'Time64'
        self.recarrtype = revrecarrfmt[self.type]
    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = False):
        TimeCol.__init__(self, dflt , shape = shape, itemsize = 8,
                         pos = pos, indexed = indexed)
    def __repr__(self):
        return "Time64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self._v_pos, self.indexed)


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

    A Numarray data type might be specified in order to determine the
    base type used for storing the values of enumerated values in memory
    and disk.  The data type must be able to represent each and every
    concrete value in the enumeration.  If it is not, a ``TypeError`` is
    raised.  The default base type is unsigned 32-bit integer, which is
    sufficient for most cases.

    The ``stype`` attribute of enumerated columns is always ``'Enum'``,
    while the ``type`` attribute is the data type used for storing
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
            asArray = NA.array(values)

            # Check integer type of concrete values.
            if not isinstance(asArray.type(), NA.IntegralType):
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


    def _setType(self, type_):
        type_ = NA.typeDict[type_]

        # Check integer type of representation.
        if not isinstance(type_, NA.IntegralType):
            raise NotImplementedError("""\
sorry, only integer concrete values type are supported for the moment""")

        names = []
        values = []
        for (name, value) in self.enum:
            names.append(name)
            values.append(value)

        # Check that type can represent concrete values.
        encoded = NA.array(values, type_)
        if values != encoded.tolist():
            raise TypeError("""\
type ``%s`` can not represent all concrete values in the enumeration"""
                            % type_)

        self._naNames = names
        """List of enumerated names."""

        self._naValues = encoded
        """List of enumerated concrete values."""

        self.type = type_
        self.stype = 'Enum'
        self.recarrtype = revrecarrfmt[type_]


    def _setDefault(self, dflt):
        if not isinstance(dflt, basestring):
            raise TypeError(
                "name of default enumerated value is not a string: %r"
                % (dflt,))

        # The defaults are now numarray objects. However,
        # for enumerated types, we still hold using python objects
        # because some issues with Enum.__call__(NumArray) calls.
        # F. Altet 2005-11-9
        #self.dflt = self.enum[dflt]
        self.dflt = NA.array(self.enum[dflt], type=self.type)


    def __repr__(self):
        return ('EnumCol(%s, %r, dtype=\'%s\', shape=%s, pos=%s, indexed=%s)'
                % (self.enum, self.enum(self.dflt[()]),
                   self.type, self.shape, self._v_pos, self.indexed))


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
        A list of the names of the columns hanging directly from the
        associated table or nested column.  The order of the names
        matches the order of their respective columns in the containing
        table.

    _v_nestedNames
        A nested list of the names of all the columns under this table
        or nested column.  You can use this for the ``names`` argument
        of ``NestedRecArray`` factory functions.

    _v_nestedFormats
        A nested list of the Numarray string formats (and shapes) of all
        the columns under this table or nested column.  You can use this
        for the ``formats`` argument of ``NestedRecArray`` factory
        functions.

    _v_nestedDescr
        A nested list of pairs of ``(name, format)`` tuples for all the
        columns under this table or nested column.  You can use this for
        the ``descr`` argument of ``NestedRecArray`` factory functions.

    _v_types
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective Numarray types.

    _v_stypes
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective string types.

    _v_dflts
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective default values.

    _v_colObjects
        A dictionary mapping the names of the columns hanging directly
        from the associated table or nested column to their respective
        descriptions (`Col` or `Description` instances).

    _v_shapes
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective shapes.

    _v_itemsizes
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective item size (in bytes).

    _v_nestedlvl
        The level of the associated table or nested column in the nested
        datatype.

    _v_is_nested
        Either the associated table has nested columns or not (Boolean).


    Public methods:

    _v_walk([type])
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
        newdict["_v_names"] = []
        newdict["_v_types"] = {}
        newdict["_v_stypes"] = {}
        newdict["_v_dflts"] = {}
        newdict["_v_colObjects"] = {}
        newdict["_v_shapes"] = {}
        newdict["_v_itemsizes"] = {}
        newdict["_v_totalsizes"] = {}
        newdict["_v_fmt"] = ""
        newdict["_v_is_nested"] = False
        nestedFormats = []

        if not hasattr(newdict, "_v_nestedlvl"):
            newdict["_v_nestedlvl"] = nestedlvl + 1

        if "_v_byteorder" in keys and self._v_nestedlvl > 0:
            raise KeyError, \
"You can only specify a byteorder in the root level of the description object, not in nested levels."
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
                classdict[k] = Description(descr.columns, self._v_nestedlvl)
            elif (type(object.__class__) == type(IsDescription) and
                issubclass(object.__class__, IsDescription)):
                #print "Nested object (type II)-->", k
                descr = object.__class__()
                classdict[k] = Description(descr.columns, self._v_nestedlvl)
            elif isinstance(object, dict):
                #print "Nested object (type III)-->", k
                classdict[k] = Description(object, self._v_nestedlvl)

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
                newdict['_v_types'][k] = object.type
                newdict['_v_stypes'][k] = object.stype
                newdict['_v_dflts'][k] = object.dflt
                nestedFormats.append(str(object.shape) + object.recarrtype)
                newdict['_v_shapes'][k] = object.shape
                newdict['_v_itemsizes'][k] = object.itemsize
                if isinstance(object.shape, tuple):
                    totalshape = reduce(operator.mul, object.shape)
                else:
                    totalshape = object.shape
                newdict['_v_totalsizes'][k] = totalshape * object.itemsize
            else:  # A description
                nestedFormats.append(object._v_nestedFormats)
                # multidimensional nested records not supported yet
                newdict['_v_shapes'][k] = 1
                itemsize = sum(object._v_itemsizes.values())
                newdict['_v_itemsizes'][k] = itemsize
                totalsize = sum(object._v_totalsizes.values())
                totalshape = 1
                newdict['_v_totalsizes'][k] = totalshape * totalsize

        # Compute the itemsize for self
        totalsize = sum(self._v_totalsizes.values())
        newdict['_v_totalsize'] = totalsize

        # Assign the format list to _v_nestedFormats
        newdict['_v_nestedFormats'] = nestedFormats
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

    def _v_walk(self, type="All"):
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
    print "Shapes ==>", desc._v_shapes
    print "Itemsizes ==>", desc._v_itemsizes
    print "Totalsizes ==>", desc._v_totalsizes
    print "Total size ==>", sum(desc._v_totalsizes.values()), desc._v_totalsize


    # check _v_walk
    for object in desc._v_walk():
        if isinstance(object, Description):
            print "******begin object*************",
            print "name -->", object._v_name
            #print "object childs-->", object._v_names
            #print "object nested childs-->", object._v_nestedNames
            print "totalsize-->", object._v_totalsize
        else:
            pass
            #print "leaf -->", object._v_name, object.type, object.shape, object.itemsize



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
