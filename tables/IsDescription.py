########################################################################
#
#       License:        BSD
#       Created:        September 21, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/IsDescription.py,v $
#       $Id: IsDescription.py,v 1.13 2003/07/15 18:52:48 falted Exp $
#
########################################################################

"""Classes and metaclasses for defining user data columns for Table objects.

See the metaIsDescription for a deep explanation on how exactly this works.

Classes:

    metaIsDescription
    IsDescription

Functions:

Misc variables:

    __version__

"""

__version__ = "$Revision: 1.13 $"


import warnings
import struct
import types
import sys

import numarray as NA
#import recarray
import numarray.records as records
#import recarray2

from utils import checkNameValidity

# Map between the numarray types and struct datatypes
tostructfmt = {NA.Int8:'b', NA.UInt8:'B',
               NA.Int16:'h', NA.UInt16:'H',
               NA.Int32:'i', NA.UInt32:'I',
               NA.Int64:'q', NA.UInt64:'Q',
               NA.Float32:'f', NA.Float64:'d',
               records.CharType:'s', 
               }

# translation table from the Struct data types to numarray types
fromstructfmt = {'b':NA.Int8, 'B':NA.UInt8,
                 'h':NA.Int16, 'H':NA.UInt16,
                 'i':NA.Int32, 'I':NA.UInt32,
                 'q':NA.Int64, 'Q':NA.UInt64,
                 'f':NA.Float32, 'd':NA.Float64,
                 's':records.CharType,
              }

class Col:
    """ Define a numerical column """
    def __init__(self, dtype="Float64", shape=1, dflt=None, pos = None):

        self.pos = pos

        assert shape != None and shape != 0 and shape != (0,), \
               "None or zero-valued shapes are not supported '%s'" % `shape`

        if type(shape) in [types.IntType, types.LongType]:
            self.shape = shape
        elif type(shape) in [types.ListType, types.TupleType]:
            # HDF5 does not support ranks greater than 32
            assert len(shape) <= 32, \
               "Shapes with rank > 32 are not supported '%s'" % `shape`
            self.shape = tuple(shape)
        else:
            raise ValueError, "Illegal shape object: '%s'" % `shape`

        self.dflt = dflt

        if dtype in NA.typeDict:
            self.type = NA.typeDict[dtype]
            #self.recarrtype = recarray2.revfmt[self.type]
            self.recarrtype = records.revfmt[self.type]
            self.itemsize = self.type.bytes
        elif dtype == "CharType" or isinstance(dtype, records.Char):
            # Special case for Strings
            self.type = records.CharType
            if type(shape) in [types.IntType, types.LongType]:
                self.shape = 1
                self.itemsize = shape
            else:
                shape = list(self.shape)
                self.itemsize = shape.pop()
                if shape == ():
                    self.shape = 1
                elif len(shape) == 1:
                    self.shape = shape[0]
                else:
                    self.shape = tuple(shape)
                    
            self.recarrtype = records.revfmt[self.type]
        else:
            raise TypeError, "Illegal type: %s" % `dtype`

        self.rectype = tostructfmt[self.type]

    def __str__(self):
        out = "type: " + str(self.type) + " \\\\ shape: " + str(self.shape)
        return out

    def __repr__(self):
        out = "\n  type: " + str(self.type) + \
              "\n  shape: " +  str(self.shape) + \
              "\n  itemsize: " +  str(self.itemsize) + \
              "\n  position: " +  str(self.pos) + \
              "\n"
        return out

    # Moved out of scope
    def _f_del__(self):
        print "Deleting Col object"


class StringCol(Col):
    """ Define a string column """
    
    def __init__(self, dflt=None, itemsize=1, shape=1, pos = None):

        self.pos = pos

        assert shape != None and shape != 0 and shape != (0,), \
               "None or zero-valued shapes are not supported '%s'" % `shape`
        
        if type(shape) in [types.IntType, types.LongType]:
            self.shape = shape
        elif type(shape) in [types.ListType, types.TupleType]:
            self.shape = tuple(shape)
        else: raise ValueError, "Illegal shape %s" % `shape`

        self.dflt = dflt

        self.type = records.CharType
        self.itemsize = itemsize
        self.recarrtype = records.revfmt[self.type]
        self.rectype = tostructfmt[self.type]

    
class IntCol(Col):
    """ Define an integer column """
    def __init__(self, dflt=0, itemsize=4, shape=1, sign=1, pos=None):

        self.pos = pos

        assert shape != None and shape != 0 and shape != (0,), \
               "None or zero-valued shapes are not supported '%s'" % `shape`

        assert itemsize in [1, 2, 4, 8], \
               "Integer itemsizes different from 1,2,4 or 8 are not supported"
        
        if shape != None and shape != 0 and shape != (0,):
            if type(shape) in [types.IntType, types.LongType]:
                self.shape = shape
            elif type(shape) in [types.ListType, types.TupleType]:
                self.shape = tuple(shape)
            else: raise ValueError, "Illegal shape %s" % `shape`

        self.dflt = dflt

        self.itemsize = itemsize
        if itemsize == 1:
            if sign:
                self.type = "Int8"
            else:
                self.type = "UInt8"
        elif itemsize == 2:
            if sign:
                self.type = "Int16"
            else:
                self.type = "UInt16"
        elif itemsize == 4:
            if sign:
                self.type = "Int32"
            else:
                self.type = "UInt32"
        elif itemsize == 8:
            if sign:
                self.type = "Int64"
            else:
                self.type = "UInt64"
                
        self.recarrtype = records.revfmt[self.type]
        self.rectype = tostructfmt[self.type]

    
class FloatCol(Col):
    """ Define a float column """
    def __init__(self, dflt=0.0, itemsize=8, shape=1, pos=None):

        self.pos = pos

        assert shape != None and shape != 0 and shape != (0,), \
               "None or zero-valued shapes are not supported '%s'" % `shape`

        assert itemsize in [4,8], \
               "Float itemsizes different from 4 and 8 are not supported"
        
        if shape != None and shape != 0 and shape != (0,):
            if type(shape) in [types.IntType, types.LongType]:
                self.shape = shape
            elif type(shape) in [types.ListType, types.TupleType]:
                self.shape = tuple(shape)
            else: raise ValueError, "Illegal shape %s" % `shape`

        self.dflt = dflt

        self.itemsize = itemsize
        if itemsize == 4:
            self.type = "Float32"
        elif itemsize == 8:
            self.type = "Float64"
            
                
        self.recarrtype = records.revfmt[self.type]
        self.rectype = tostructfmt[self.type]

    
class metaIsDescription(type):
    """
    
    metaclass for Table "Col"umn "Descr"iption: implicitly defines
    __slots__, __init__ __repr__ and some others from variables bound
    in class scope.

    An instance of metaIsDescription (a class whose metaclass is
    metaIsDescription) defines only class-scope variables (and
    possibly special methods, but NOT __init__ and __repr__!).
    metaIsDescription removes those variables from class scope,
    snuggles them instead as items in a class-scope dict named
    __dflts__, and puts in the class a __slots__ listing those
    variables' names, an __init__ that takes as optional keyword
    arguments each of them (using the values in __dflts__ as defaults
    for missing ones), and a __repr__ that shows the repr of each
    attribute that differs from its default value (the output of
    __repr__ can be passed to __eval__ to make an equal instance, as
    per the usual convention in the matter).

    Author:
    
    This metaclass is loosely based on an example from Alex Martelli
    (http://mail.python.org/pipermail/python-list/2002-July/112007.html)
    However, I've modified things quite a bit, and the ultimate goal
    has changed.

    """

    def __new__(cls, classname, bases, classdict):
        """ Everything needs to be done in __new__, since type.__new__ is
            where __slots__ are taken into account.
        """

        # define as local functions the __init__ and __repr__ that we'll
        # use in the new class
        def __init__(self, **kw):
            """ Simplistic __init__: first set all attributes to default
                values, then override those explicitly passed in kw.
            """
            # Initialize this values
            for k in self.__dflts__: setattr(self, k, self.__dflts__[k])
            # Initialize the values passed as keyword parameters
            for k in kw:
                setattr(self, k, kw[k])

	def __repr__(self):
            """ Gives a Table representation ready to be passed to eval
            """
            rep = [ '\"%s\": Col(\"%r\", %r)' %  \
                    (k, self.__types__[k], self._v_shapes[k])
                    for k in self.__slots__ ]
            return '{ %s }' % (',\n  '.join(rep))
	
        def __str__(self):
            """ Gives a Table representation for printing purposes
            """
            rep = [ '%s(%r%r)' %  \
                    (k, self.__types__[k], self._v_shapes[k])
                    for k in self.__slots__ ]
            return '[%s]' % (', '.join(rep))

        # Moved out of scope
        def _f_del__(self):
            print "Deleting IsDescription object"

        def testtype(object):
            """Test if datatype is valid and returns a default value for
            each one.
            """
            datatype = object.rectype
            if datatype in ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q'):
                dfltvalue = int(0)
            elif datatype in ('f', 'd'):
                dfltvalue = float(0)
            elif datatype in ('c',):
                dfltvalue = str(" ")
            # Add more code to check for validity on string type!
            elif datatype.find("s") != -1:
                dfltvalue = str("")
            else:
                raise TypeError, "DataType \'%s\' not supported!." \
                      % datatype
            return dfltvalue
        
        # Build the newdict that we'll use as dict for the new class.
        # Warning!. You have to list here all attributes and methods
        # you want see exported to the new Description class.

        newdict = { '__slots__':[], '__types__':{}, '__dflts__':{},
                    '__init__':__init__, '__repr__':__repr__,
                    '__str__':__str__,
                    '_v_fmt': "", '_v_recarrfmt': "",
                    "_v_shapes":{}, "_v_itemsizes":{},
                    '_v_formats':[],
                    }
        

        def cmpkeys(key1, key2):
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
            pos1 = classdict[key1].pos
            pos2 = classdict[key2].pos
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

        keys = classdict.keys()
        # Check if we have any .pos position attribute
        for column in classdict.values():
            if hasattr(column, "pos") and column.pos:
                keys.sort(cmpkeys)
                break
        else:
            # No .pos was set
            # fall back to alphanumerical order
            keys.sort()
            
        recarrfmt = []
        for k in keys:
            if (k.startswith('__') or k.startswith('_v_')):
                if k in newdict:
                    # special methods &c: copy to newdict, warn about conflicts
                    warnings.warn("Can't set attr %r in coldescr-class %r" % (
                        k, classname))
                else:
                    # Beware, in this case, we don't allow fields with
                    # prefix "_v_". This is reserved to pass special
                    # variables to the new class.
                    #print "Special variable!:", k
                    newdict[k] = classdict[k]
            else:
                # class variables, store name in __slots__ and name and
                # value as an item in __dflts__

                # Check for key name validity
                checkNameValidity(k)
                object = classdict[k]
                if not isinstance(object, Col):
                    raise TypeError, \
"""Passing an incorrect value to a table column.

  Please, make use of the Col() constructor to properly initialize
  columns. Expected a Col instance and got: "%s"

""" % object
                newdict['__slots__'].append(k)
                newdict['__types__'][k] = object.type
                if hasattr(object, 'dflt') and not object.dflt is None:
                    newdict['__dflts__'][k] = object.dflt
                else:
                    newdict['__dflts__'][k] = testtype(object)

                # Special case for strings: "aN"
                if object.recarrtype == "a":
                    # This needs to be fixed when calcoffset will support
                    # the recarray format, for ex: "(1,3)f4,3i4,(2,)a5,i2"
                    if type(object.shape) in [types.IntType, types.LongType]:
                        if object.shape == 1:
                            shape = object.itemsize
                        else:
                            shape = (object.shape, object.itemsize)
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
        #print "fmt -->", newdict['_v_fmt']
        if newdict.has_key('_v_align'):
            newdict['_v_fmt'] = newdict['_v_align'] + newdict['_v_fmt']
        else:
            newdict['_v_fmt'] = "=" + newdict['_v_fmt']  # Standard align
        # Strip the last comma from _v_recarrfmt
        newdict['_v_recarrfmt'] = ','.join(recarrfmt)
        # finally delegate the rest of the work to type.__new__
        return type.__new__(cls, classname, bases, newdict)


class IsDescription(object):
    """ For convenience: inheriting from IsDescription can be used to get
        the new metaclass (same as defining __metaclass__ yourself).
    """
    __metaclass__ = metaIsDescription


if __name__=="__main__":
    """Here is code to benchmark the differents methods to pack/unpack.

    Use it to experiment new methods to accelerate the pack/unpack
    process.
    
    """
    
    class Description(IsDescription):
        """A description that has several columns.

        Represent the here as class variables, whose values are their
        types. The metaIsDescription class will take care the user
        won't add any new variables and that their type is correct.

        """
        #color = '3s'
        x = Col("Int32", 2, 0)
        y = Col("Float64", 1, 1)
        z = Col("UInt8", 1, 1)
        color = Col("CharType", 2, " ")

    # example cases of class Point
    rec = Description()  # Default values
    print "rec value ==>", rec
    print "Slots ==>", rec.__slots__
    print "Format for this table ==>", rec._v_fmt
    print "recarray Format for this table ==>", rec._v_recarrfmt
