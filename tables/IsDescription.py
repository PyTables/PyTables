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
import struct
import sys

import numarray as NA
import numarray.records as records

from tables.utils import checkNameValidity



__version__ = "$Revision: 1.41 $"


# Map between the Numarray types and struct datatypes.
naTypeToStruct = {
    NA.Int8:'b',  NA.UInt8:'B',  NA.Int16:'h', NA.UInt16:'H',
    NA.Int32:'i', NA.UInt32:'I', NA.Int64:'q', NA.UInt64:'Q',
    NA.Float32:'f', NA.Float64:'d',
    NA.Complex32:'F', NA.Complex64:'D',  # added to support complex
    NA.Bool:'c', records.CharType:'s'}



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



class Col(ShapeMixin, object):
    "Defines a general column that supports all numarray data types."

    # This class should become abstract somewhere in the future,
    # with no methods beyond __init__() or __repr__().
    # An auxiliary function returning an object of the adequate class
    # should be used instead.
    # So, the following _set*() methods should be removed in the future.
    # ivilata(2004-12-17)

    def _setPosition(self, pos):
        "Sets the 'pos' attribute."
        self.pos = pos

    def _setType(self, type_):
        "Sets the 'type', 'recarrtype', 'stype' and 'rectype' attributes."
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

        self.recarrtype = records.revfmt[self.type]

        # Since Time columns have no Numarray type of their own,
        # a special case is made for them.
        if type_ == 'Time32':
            self.rectype = 't'  # special code for times
        elif type_ in ('Time', 'Time64'):
            self.rectype = 'T'  # special code for times
        else:
            self.rectype = naTypeToStruct[self.type]

    def _setDefault(self, dflt):
        "Sets the 'dflt' attribute."
        self.dflt = dflt

    def _setIndex(self, indexed):
        "Sets the 'indexed' attribute."
        if indexed and self.type in (NA.Complex32, NA.Complex64):
            raise TypeError("%r do not support indexation" % (self.type,))
        self.indexed = indexed

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

        # Override itemsize; strings still need some tampering with the shape
        #
        # This exposes NumPy BUG#1087158, since the comparison is True
        # for self.type = Float64, Int32, Complex64...
        # ivilata(2004-12-17)
        #
        ##if self.type == records.CharType:
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

    def __init__(self, dtype = 'Float64', shape = 1, dflt = None, pos = None,
                 indexed = 0):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(dtype)    -> type, recarrtype, stype, rectype
        2. _setDefault(dflt)  -> dflt
        3. _setShape(shape)   -> shape, itemsize
        4. _setIndex(indexed) -> indexed
        5. _setPosition(pos)  -> pos
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
            self.stype, shape, self.dflt, self.pos, self.indexed)



class BoolCol(Col):
    "Defines a boolean column."

    def _setType(self, type_):
        self.type       = NA.Bool
        self.stype      = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype    = naTypeToStruct[self.type]

    def _setIndex(self, indexed):
        self.indexed = indexed

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = False, shape = 1, pos = None, indexed = 0):
        Col.__init__(
            self, dtype = 'Bool',
            dflt = dflt, shape = shape, pos = pos, indexed = indexed)

    def __repr__(self):
        return "BoolCol(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)



class StringCol(Col):
    "Defines a string column."

    def _setType(self, type_):
        self.type       = records.CharType
        self.stype      = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype    = naTypeToStruct[self.type]

    def _setDefault(self, dflt):
        if dflt != None and not isinstance(dflt, str):
            raise ValueError("Invalid default value: %s" % (dflt,))
        self.dflt = dflt

    def _setIndex(self, indexed):
        self.indexed = indexed

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

        # Set itemsize; forced to None to get it from the 'length' argument
        self.itemsize = None

    def __init__(self, length = None, dflt = None, shape = 1, pos = None,
                 indexed = 0):
        Col.__init__(
            self, dtype = 'CharType',
            dflt = dflt, shape = shape, pos = pos, indexed = indexed)

        # Some more work needed for constructor call idiosincracies:
        # 'itemsize' is deduced from the default value if not specified.
        if length == None and dflt:
            length = len(dflt)  # 'dflt' has already been checked

        if not length:
            raise RuntimeError("""\
You must specify at least a length or a default value
  where this length can be inferred from.""")

        self.itemsize = length

    def __repr__(self):
        return ("StringCol(length=%s, dflt=%r, shape=%s, pos=%s, indexed=%s)"
                % (self.itemsize, self.dflt, self.shape, self.pos,
                   self.indexed))



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
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]

    def _setIndex(self, indexed):
        self.indexed = indexed

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = 0, shape = 1, itemsize = 4, sign = 1,
                 pos = None, indexed=0):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize, sign) -> type, recarrtype, stype, rectype
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
        if NA.array(0, self.type) - NA.array(1, self.type) < 0:
            sign = 1
        else:
            sign = 0

        return """\
IntCol(dflt=%s, shape=%s, itemsize=%s, sign=%s, pos=%s, indexed=%s)""" % (
            self.dflt, self.shape, self.itemsize, sign, self.pos, self.indexed)

class Int8Col(IntCol):
    "Description class for a signed integer of 8 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int8
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = 0):
        IntCol.__init__(self, dflt, itemsize = 1, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int8Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class UInt8Col(IntCol):
    "Description class for an unsigned integer of 8 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt8
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=0):
        IntCol.__init__(self, dflt , itemsize = 1, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt8Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class Int16Col(IntCol):
    "Description class for a signed integer of 16 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int16
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=0):
        IntCol.__init__(self, dflt , itemsize = 2, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int16Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class UInt16Col(IntCol):
    "Description class for an unsigned integer of 16 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt16
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=0):
        IntCol.__init__(self, dflt , itemsize = 2, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt16Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class Int32Col(IntCol):
    "Description class for a signed integer of 32 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int32
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = 0):
        IntCol.__init__(self, dflt , itemsize=4, shape=shape, sign=1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class UInt32Col(IntCol):
    "Description class for an unsigned integer of 32 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt32
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=0):
        IntCol.__init__(self, dflt , itemsize = 4, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class Int64Col(IntCol):
    "Description class for a signed integer of 64 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.Int64
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=0):
        IntCol.__init__(self, dflt , itemsize = 8, shape = shape, sign = 1,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "Int64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class UInt64Col(IntCol):
    "Description class for an unsigned integer of 64 bits."
    def _setType(self, itemsize, sign):
        self.type = NA.UInt64
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt=0, shape=1, pos=None, indexed=0):
        IntCol.__init__(self, dflt , itemsize = 8, shape = shape, sign = 0,
                        pos = pos, indexed = indexed)
    def __repr__(self):
        return "UInt64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)



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
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]

    def _setIndex(self, indexed):
        self.indexed = indexed

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = 0.0, shape = 1, itemsize = 8, pos = None,
                 indexed = 0):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize) -> type, recarrtype, stype, rectype
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
            self.dflt, self.shape, self.itemsize, self.pos, self.indexed)

class Float32Col(FloatCol):
    "Description class for a floating point of 32 bits."
    def _setType(self, itemsize):
        self.type = NA.Float32
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = 0):
        FloatCol.__init__(self, dflt , shape = shape, itemsize = 4,
                          pos = pos, indexed = indexed)
    def __repr__(self):
        return "Float32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class Float64Col(FloatCol):
    "Description class for a floating point of 64 bits."
    def _setType(self, itemsize):
        self.type = NA.Float64
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = 0):
        FloatCol.__init__(self, dflt , shape = shape, itemsize = 8,
                          pos = pos, indexed = indexed)
    def __repr__(self):
        return "Float64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)



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
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = (0.0+0.0j), shape = 1, itemsize = 16, pos = None):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize) -> type, recarrtype, stype, rectype
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
        self.indexed = 0

    def __repr__(self):
        return "ComplexCol(dflt=%s, shape=%s, itemsize=%s, pos=%s)" % (
            self.dflt, self.shape, self.itemsize, self.pos)

class Complex32Col(ComplexCol):
    "Description class for a complex of simple precision."
    def _setType(self, itemsize):
        self.type = NA.Complex32
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt = (0.0+0.0j), shape = 1, pos = None):
        ComplexCol.__init__(self, dflt, shape = shape, itemsize = 8, pos = pos)
    def __repr__(self):
        return "Complex32Col(dflt=%s, shape=%s, pos=%s)" % (
            self.dflt, self.shape, self.pos)

class Complex64Col(ComplexCol):
    "Description class for a complex of double precision."
    def _setType(self, itemsize):
        self.type = NA.Complex64
        self.stype = str(self.type)
        self.recarrtype = records.revfmt[self.type]
        self.rectype = naTypeToStruct[self.type]
    def __init__(self, dflt = (0.0+0.0j), shape = 1, pos = None):
        ComplexCol.__init__(self, dflt , shape = shape, itemsize = 16, pos = pos)
    def __repr__(self):
        return "Complex64Col(dflt=%s, shape=%s, pos=%s)" % (
            self.dflt, self.shape, self.pos)



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
            self.rectype = 't'  # special code for times
        elif itemsize == 8:
            self.type = NA.Float64
            self.stype = 'Time64'
            self.rectype = 'T'  # special code for times

        self.recarrtype = records.revfmt[self.type]

    def _setIndex(self, indexed):
        self.indexed = indexed

    def _setShape(self, shape):
        super(Col, self)._setShape(shape)

    def __init__(self, dflt = 0, shape = 1, itemsize = 8, pos = None,
                 indexed = 0):
        """Calls _set*() in this precise order,
        setting the indicated attributes:

        1. _setType(itemsize) -> type, recarrtype, stype, rectype
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
            self.dflt, self.shape, self.itemsize, self.pos, self.indexed)

class Time32Col(TimeCol):
    "Description class for an integer time of 32 bits."
    def _setType(self, itemsize):
        self.type = NA.Int32
        self.stype = 'Time32'
        self.recarrtype = records.revfmt[self.type]
        self.rectype = 't'  # special code for times
    def __init__(self, dflt = 0, shape = 1, pos = None, indexed = 0):
        TimeCol.__init__(self, dflt , shape = shape, itemsize = 4,
                         pos = pos, indexed = indexed)
    def __repr__(self):
        return "Time32Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)

class Time64Col(TimeCol):
    "Description class for a floating point time of 64 bits."
    def _setType(self, itemsize):
        self.type = NA.Float64
        self.stype = 'Time64'
        self.recarrtype = records.revfmt[self.type]
        self.rectype = 'T'  # special code for times
    def __init__(self, dflt = 0.0, shape = 1, pos = None, indexed = 0):
        TimeCol.__init__(self, dflt , shape = shape, itemsize = 8,
                         pos = pos, indexed = indexed)
    def __repr__(self):
        return "Time64Col(dflt=%s, shape=%s, pos=%s, indexed=%s)" % (
            self.dflt, self.shape, self.pos, self.indexed)



class Description(object):
    "Regular class to keep table description metadata"

    def __init__(self, classdict):

        self.classdict = classdict
        keys = classdict.keys()
        newdict = self.__dict__
        newdict["__names__"] = []
        newdict["__types__"] = {}
        newdict["__stypes__"] = {}
        newdict["__dflts__"] = {}
        newdict["_v_ColObjects"] = {}
        newdict["_v_shapes"] = {}
        newdict["_v_itemsizes"] = {}
        newdict["_v_fmt"] = ""
        # Check if we have any .pos position attribute
        for column in classdict.values():
            if hasattr(column, "pos") and column.pos:
                keys.sort(self.cmpkeys)
                break
        else:
            # No .pos was set
            # fall back to alphanumerical order
            keys.sort()
        recarrfmt = []
        if "__check_validity__" in keys:
            check_validity = classdict["__check_validity__"]
        else:
            check_validity = 1   # Default value for name validity check
        for k in keys:
            if (k.startswith('__') or k.startswith('_v_')):
                if k in newdict:
                    # special methods &c: copy to newdict, warn about conflicts
                    warnings.warn("Can't set attr %r in coldescr-class %r" % (
                        k, classname))
                else:
                    #print "Special variable!:", k
                    newdict[k] = classdict[k]
            else:
                # Class variables
                if check_validity:
                    # Check for key name validity
                    checkNameValidity(k)
                object = classdict[k]
                if not isinstance(object, Col):
                    raise TypeError, \
"""Passing an incorrect value to a table column.
  Please, make use of the Col(), or descendant, constructor to
  properly initialize columns. Expected a Col (or subclass) instance
  and got: "%s"

""" % object
                newdict['__names__'].append(k)
                newdict['_v_ColObjects'][k] = object
                newdict['__types__'][k] = object.type
                newdict['__stypes__'][k] = object.stype
                if hasattr(object, 'dflt') and not object.dflt is None:
                    newdict['__dflts__'][k] = object.dflt
                else:
                    newdict['__dflts__'][k] = self.testtype(object)

                # Special case for strings: "aN"
                if object.recarrtype == "a":
                    # This needs to be fixed when calcoffset will support
                    # the recarray format, for ex: "(1,3)f4,3i4,(2,)a5,i2"
                    if type(object.shape) in (int,long):
                        # If shape is int type, it is always 1
                        shape = object.itemsize
                    else:
                        shape = list(object.shape)
                        shape.append(object.itemsize)
                        shape = tuple(shape)
                        
                    newdict['_v_fmt'] +=  str(shape) + object.rectype
                    newdict['_v_shapes'][k] = object.shape
                    newdict['_v_itemsizes'][k] = object.itemsize
                    recarrfmt.append(str(object.shape) + \
                                     object.recarrtype + str(object.itemsize))
                else:
                    newdict['_v_fmt'] += str(object.shape) + object.rectype
                    recarrfmt.append(str(object.shape) + object.recarrtype)
                    newdict['_v_shapes'][k] = object.shape
                    newdict['_v_itemsizes'][k] = object.itemsize

        # Set up the alignment
        if newdict.has_key('_v_align'):
            newdict['_v_fmt'] = newdict['_v_align'] + newdict['_v_fmt']
        else:
            newdict['_v_fmt'] = "=" + newdict['_v_fmt']  # Standard align
        # Assign the formats list to _v_recarrfmt
        newdict['_v_recarrfmt'] = recarrfmt
        # finally delegate the rest of the work to type.__new__
        return

    def __repr__(self):
        """ Gives a Table column representation
        """
        rep = [ '\"%s\": %r' %  \
                (k, self._v_ColObjects[k])
                for k in self.__names__]
        return '{\n    %s }' % (',\n    '.join(rep))

    def __str__(self):
        """ Gives a Table representation for printing purposes
        """
        rep = [ '%s(%r%r)' %  \
                (k, self.__types__[k], self._v_shapes[k])
                for k in self.__names__ ]
        return '[%s]' % (', '.join(rep))

    def _close(self):
        self._v_ColObjects.clear()
        del self.__dict__["_v_ColObjects"]
        self._v_itemsizes.clear()
        self._v_shapes.clear()
        self.__dflts__.clear()
        self.__stypes__.clear()
        self.__types__.clear()
        self.__dict__.clear()
        return

    def testtype(self, object):
        """Test if datatype is valid and returns a default value for
        each one.
        """
        datatype = object.rectype
        if datatype in ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q'):
            dfltvalue = int(0)
        elif datatype in ('f', 'd'):
            dfltvalue = float(0)
        elif datatype in ('F', 'D'):
            dfltvalue = complex(0)
        elif datatype in ('c',):
#             dfltvalue = str(" ")
            dfltvalue = int(0)
        elif datatype in ('t',):  # integer time value
            dfltvalue = int(0)
        elif datatype in ('T',):  # floating point time value
            dfltvalue = float(0)
        # Add more code to check for validity on string type!
        elif datatype.find("s") != -1:
            dfltvalue = str("")
        else:
            raise TypeError, "DataType \'%s\' not supported!." \
                  % datatype
        return dfltvalue

    def cmpkeys(self, key1, key2):
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
        pos1 = self.classdict[key1].pos
        pos2 = self.classdict[key2].pos
        # pos = None is always greater than a number
        if pos1 == None:
            return 1
        if pos2 == None:
            return -1
        if pos1 < pos2:
            return -1
        if pos1 == pos2:
            return 0
        if pos1 > pos2:
            return 1


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
    
    class Test(IsDescription):
        """A description that has several columns"""
        x = Col("Int32", 2, 0)
        y = FloatCol(1, shape=(2,3))
        z = UInt8Col(1)
        color = StringCol(2, " ")

    # example cases of class Test
    klass = Test()
    rec = Description(klass.columns)
    print "rec value ==>", rec
    print "Column names ==>", rec.__names__
    print "Format for this table ==>", rec._v_fmt
    print "recarray Format for this table ==>", rec._v_recarrfmt
