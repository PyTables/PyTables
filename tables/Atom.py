########################################################################
#
#       License: BSD
#       Created: December 16, 2004
#       Author:  Ivan Vilata i Balaguer - reverse:com.carabos@ivilata
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Atom.py,v $
#       $Id$
#
########################################################################

"""Here are defined some declarative classes for VLArray components

See *Atom docstrings for more info.

Classes:

    Atom, ObjectAtom, VLStringAtom, StringAtom, BoolAtom,
    IntAtom, Int8Atom, UInt8Atom, Int16Atom, UInt16Atom,
    TimeAtom, Time32Atom, Time64Atom

Functions:

   checkflavor

Misc variables:

    __version__


"""

__version__ = "$Revision: 1.2 $"


import types
import numarray
import numarray.records as records
#import IsDescription # to access BaseCol without polluting public namespace
from IsDescription import \
     Col, BoolCol, StringCol, IntCol, FloatCol, ComplexCol, TimeCol



def checkflavor(flavor, dtype):
    #if dtype == "CharType" or isinstance(dtype, records.Char):
    if str(dtype) == "CharType":
        if flavor in ["CharArray", "String"]:
            return flavor
        else:
            raise ValueError, \
"""flavor of type '%s' must be one of the "CharArray" or "String" values, and you tried to set it to "%s".
"""  % (dtype, flavor)
    else:
        if flavor in ["NumArray", "Numeric", "Tuple", "List"]:
            return flavor
        else:
            raise ValueError, \
"""flavor of type '%s' must be one of the "NumArray", "Numeric", "Tuple" or "List" values, and you tried to set it to "%s".
"""  % (dtype, flavor)



# Class to support variable length strings as components of VLArray
# It supports UNICODE strings as well.
class VLStringAtom(IntCol):
    """ Define an atom of type Variable Length String """
    def __init__(self):
        # This special strings will be represented by unsigned bytes
        IntCol.__init__(self, itemsize=1, shape=1, sign=0)
        self.flavor = "VLString"

    def __repr__(self):
        return "VLString()"

    def atomsize(self):
        " Compute the item size of the VLStringAtom "
        # Always return 1 because strings are saved in UTF-8 format
        return 1


class ObjectAtom(IntCol):
    """ Define an atom of type Object """
    def __init__(self):
        IntCol.__init__(self, shape=1, itemsize=1, sign=0)
        self.flavor = "Object"

    def __repr__(self):
        return "Object()"

    def atomsize(self):
        " Compute the item size of the Object "
        # Always return 1 because strings are saved in UInt8 format
        return 1



#class Atom(IsDescription.BaseCol):
class Atom(Col):
    """ Define an Atomic object to be used in VLArray objects """

    def __init__(self, dtype="Float64", shape=1, flavor="NumArray"):
        Col.__init__(self, dtype, shape)
        self.flavor = checkflavor(flavor, self.type)

    def __repr__(self):
        if self.type == "CharType" or isinstance(self.type, records.Char):
            if self.shape == 1:
                shape = [self.itemsize]
            else:
                shape = list(self.shape)
                shape.append(self.itemsize)
            shape = tuple(shape)
        else:
            shape = self.shape

        return "Atom(dtype=%r, shape=%s, flavor=%r)" % (
            self.stype, shape, self.flavor)

    def atomsize(self):
        " Compute the size of the atom type "
        atomicsize = self.itemsize
        if isinstance(self.shape, types.TupleType):
            for i in self.shape:
                if i > 0:  # To deal with EArray Atoms
                    atomicsize *= i
        else:
            atomicsize *= self.shape
        return atomicsize


class StringAtom(StringCol, Atom):
    """ Define an atom of type String """
    def __init__(self, shape=1, length=None, flavor="CharArray"):
        StringCol.__init__(self, length=length, shape=shape)
        self.flavor = checkflavor(flavor, self.type)
    def __repr__(self):
        return "StringAtom(shape=%s, length=%s, flavor=%r)" % (
            self.shape, self.itemsize, self.flavor)


class BoolAtom(BoolCol, Atom):
    """ Define an atom of type Bool """
    def __init__(self, shape=1, flavor="NumArray"):
        BoolCol.__init__(self, shape=shape)
        self.flavor = checkflavor(flavor, self.type)
    def __repr__(self):
        return "BoolAtom(shape=%s, flavor=%r)" % (self.shape, self.flavor)


class IntAtom(IntCol, Atom):
    """ Define an atom of type Integer """
    def __init__(self, shape=1, itemsize=4, sign=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=itemsize, sign=sign)
        self.flavor = checkflavor(flavor, self.type)
    def __repr__(self):
        if numarray.array(0, self.type) - numarray.array(1, self.type) < 0:
            sign = 1
        else:
            sign = 0
        return "IntAtom(shape=%s, itemsize=%s, sign=%s, flavor=%r)" % (
            self.shape, self.itemsize, sign, self.flavor)

class Int8Atom(IntAtom):
    """ Define an atom of type Int8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=1, sign=1, flavor=flavor)
    def __repr__(self):
        return "Int8Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class UInt8Atom(IntAtom):
    """ Define an atom of type UInt8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=1, sign=0, flavor=flavor)
    def __repr__(self):
        return "UInt8Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class Int16Atom(IntAtom):
    """ Define an atom of type Int16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=2, sign=1, flavor=flavor)
    def __repr__(self):
        return "Int16Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class UInt16Atom(IntAtom):
    """ Define an atom of type UInt16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=2, sign=0, flavor=flavor)
    def __repr__(self):
        return "UInt16Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class Int32Atom(IntAtom):
    """ Define an atom of type Int32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=4, sign=1, flavor=flavor)
    def __repr__(self):
        return "Int32Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class UInt32Atom(IntAtom):
    """ Define an atom of type UInt32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=4, sign=0, flavor=flavor)
    def __repr__(self):
        return "UInt32Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class Int64Atom(IntAtom):
    """ Define an atom of type Int64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=8, sign=1, flavor=flavor)
    def __repr__(self):
        return "Int64Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class UInt64Atom(IntAtom):
    """ Define an atom of type UInt64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntAtom.__init__(self, shape=shape, itemsize=8, sign=0, flavor=flavor)
    def __repr__(self):
        return "UInt64Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)


class FloatAtom(FloatCol, Atom):
    """ Define an atom of type Float """
    def __init__(self, shape=1, itemsize=8, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = checkflavor(flavor, self.type)
    def __repr__(self):
        return "FloatAtom(shape=%s, itemsize=%s, flavor=%r)" % (
            self.shape, self.itemsize, self.flavor)

class Float32Atom(FloatAtom):
    """ Define an atom of type Float32 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatAtom.__init__(self, shape=shape, itemsize=4, flavor=flavor)
    def __repr__(self):
        return "Float32Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class Float64Atom(FloatAtom):
    """ Define an atom of type Float64 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatAtom.__init__(self, shape=shape, itemsize=8, flavor=flavor)
    def __repr__(self):
        return "Float64Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)


class ComplexAtom(ComplexCol, Atom):
    """ Define an atom of type Complex """
    def __init__(self, shape=1, itemsize=16, flavor="NumArray"):
        ComplexCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = checkflavor(flavor, self.type)
    def __repr__(self):
        return "ComplexAtom(shape=%s, itemsize=%s, flavor=%r)" % (
            self.shape, self.itemsize, self.flavor)

class Complex32Atom(ComplexAtom):
    """ Define an atom of type Complex32 """
    def __init__(self, shape=1, flavor="NumArray"):
        ComplexAtom.__init__(self, shape=shape, itemsize=8, flavor=flavor)
    def __repr__(self):
        return "Complex32Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class Complex64Atom(ComplexAtom):
    """ Define an atom of type Complex64 """
    def __init__(self, shape=1, flavor="NumArray"):
        ComplexAtom.__init__(self, shape=shape, itemsize=16, flavor=flavor)
    def __repr__(self):
        return "Complex64Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)


class TimeAtom(TimeCol, Atom):
    """ Define an atom of type Time """
    def __init__(self, shape=1, itemsize=8, flavor="NumArray"):
        TimeCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = checkflavor(flavor, self.type)
    def __repr__(self):
        return "TimeAtom(shape=%s, itemsize=%s, flavor=%r)" % (
            self.shape, self.itemsize, self.flavor)

class Time32Atom(TimeAtom):
    """ Define an atom of type Time32 """
    def __init__(self, shape=1, flavor="NumArray"):
        TimeAtom.__init__(self, shape=shape, itemsize=4, flavor=flavor)
    def __repr__(self):
        return "Time32Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)

class Time64Atom(TimeAtom):
    """ Define an atom of type Time64 """
    def __init__(self, shape=1, flavor="NumArray"):
        TimeAtom.__init__(self, shape=shape, itemsize=8, flavor=flavor)
    def __repr__(self):
        return "Time64Atom(shape=%s, flavor=%r)" % (self.shape, self.flavor)
