########################################################################
#
#       License:        BSD
#       Created:        September 21, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Attic/IsRecord.py,v $
#       $Id: IsRecord.py,v 1.6 2003/02/03 10:13:08 falted Exp $
#
########################################################################

"""Classes and metaclasses for defining user data records.

See the metaIsRecord for a deep explanation on how exactly this works.

Classes:

    metaIsRecord
    IsRecord

Functions:

Misc variables:

    __version__

"""

__version__ = "$Revision: 1.6 $"


import warnings
import struct
import types
import sys

import numarray as NA
import recarray2 as recarray

from utils import checkNameValidity

# Map between the numarray types and struct datatypes
tostructfmt = {NA.Int8:'b', NA.UInt8:'B',
               NA.Int16:'h', NA.UInt16:'H',
               NA.Int32:'i', NA.UInt32:'I',
               NA.Int64:'q', NA.UInt64:'Q',
               NA.Float32:'f', NA.Float64:'d',
               recarray.CharType:'s', 
               }

# translation table from the Struct data types to numarray types
fromstructfmt = {'b':NA.Int8, 'B':NA.UInt8,
                 'h':NA.Int16, 'H':NA.UInt16,
                 'i':NA.Int32, 'I':NA.UInt32,
                 'q':NA.Int64, 'Q':NA.UInt64,
                 'f':NA.Float32, 'd':NA.Float64,
                 's':recarray.CharType,
              }

class defineType:

    def __init__(self, dtype="Float64", length=1, dflt=None):

        if length != None:
            if type(length) in [types.IntType, types.LongType]:
                self.length = length
            else: raise ValueError, "Illegal length %s" % `length`

        self.dflt = dflt

        if dtype in NA.typeDict:
            self.type = NA.typeDict[dtype]
        elif dtype == "CharType" or isinstance(dtype, recarray.Char):
            self.type = recarray.CharType
        else:
            raise TypeError, "Illegal type %s" % `dtype`

        self.rectype = tostructfmt[self.type]
        self.recarrtype = recarray.revfmt[self.type]
    
class metaIsRecord(type):
    """
    metaclass for "Record": implicitly defines __slots__, __init__
    __repr__ and some others from variables bound in class scope.

    An instance of metaIsRecord (a class whose metaclass is metaIsRecord)
    defines only class-scope variables (and possibly special methods, but
    NOT __init__ and __repr__!).  metaIsRecord removes those variables from
    class scope, snuggles them instead as items in a class-scope dict named
    __dflts__, and puts in the class a __slots__ listing those variables'
    names, an __init__ that takes as optional keyword arguments each of
    them (using the values in __dflts__ as defaults for missing ones), and
    a __repr__ that shows the repr of each attribute that differs from its
    default value (the output of __repr__ can be passed to __eval__ to make
    an equal instance, as per the usual convention in the matter).

    Author:
    
    This metaclass is largely based on an example from Alex Martelli
    (http://mail.python.org/pipermail/python-list/2002-July/112007.html)
    I've modified things quite a bit, but the spirit is the same, more
    or less.

    Usage:

class Record(IsRecord):
    x = 'l'
    y = 'd'
    color = '3s'

q = Record()
print q

p = Record(x=4, y=3.4, color = "pwpwp")
print p

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
            #print "dflts ==>", self.__dflts__
            for k in self.__dflts__: setattr(self, k, self.__dflts__[k])

            # Initialize the values passed as keyword parameters
            for k in kw:
                setattr(self, k, kw[k])

	def __repr__(self):
            """ Gives a record representation ready to be passed to eval
            """
            rep = [ '%s=%r' % (k, getattr(self, k)) for k in self.__slots__ ]
            return '%s(%s)' % (classname, ', '.join(rep))

	
        def __str__(self):
            """ Gives a record representation for printing purposes
            """
            rep = [ '  %s=%r' % (k, getattr(self, k)) for k in self.__slots__ ]
            return '%s(\n%s)' % (classname, ', \n'.join(rep))

        def testtype(object):
            """Test if datatype is valid and returns a default value for
            each one.
            """
            #datatype = tostructfmt[object.type]
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
        # you want see exported to the new Record class.
        
        newdict = { '__slots__':[], '__types__':{}, '__dflts__':{},
                    '__init__':__init__, '__repr__':__repr__,
                    '__str__':__str__, '_v_record':None,
                    '_v_fmt': "",
                    '_v_recarrfmt': "", '_v_formats':[],
                    }

        keys = classdict.keys()
        keys.sort() # Sort the keys to establish an order
        
        newdict['_v_fmt'] = "=" # Force the "standard" alignment (no align)
        recarrfmt = []
        for k in keys:
            if (k.startswith('__') or k.startswith('_v_')):
                if k in newdict:
                    # special methods &c: copy to newdict, warn about conflicts
                    warnings.warn("Can't set attr %r in record-class %r" % (
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
                newdict['__slots__'].append(k)
                object = classdict[k]
                newdict['__types__'][k] = object.type
                if object.dflt:
                    newdict['__dflts__'][k] = object.dflt
                else:
                    newdict['__dflts__'][k] = testtype(object)

                newdict['_v_fmt'] += str(object.length) + object.rectype
                recarrfmt.append(str(object.length) + object.recarrtype)
                # Formats in numarray notation
                newdict['_v_formats'].append(object.type)

        # Strip the last comma from _v_recarrfmt
        newdict['_v_recarrfmt'] = ','.join(recarrfmt)
        # finally delegate the rest of the work to type.__new__
        return type.__new__(cls, classname, bases, newdict)


class IsRecord(object):
    """ For convenience: inheriting from IsRecord can be used to get
        the new metaclass (same as defining __metaclass__ yourself).
    """
    __metaclass__ = metaIsRecord


if __name__=="__main__":
    """Here is code to benchmark the differents methods to pack/unpack.

    Use it to experiment new methods to accelerate the pack/unpack
    process.
    
    """
    
    class Record(IsRecord):
        """A record that has several columns.

        Represent the here as class variables, whose values are their
        types. The metaIsRecord class will take care the user won't
        add any new variables and that their type is correct.

        """
        #color = '3s'
        x = defineType("Int32", 2, 0)
        y = defineType("Float64", 1, 1)
        z = defineType("UInt8", 1, 1)
        color = defineType("CharType", 2, " ")

    # example cases of class Point
    rec = Record()  # Default values
    print "rec value ==>", rec
    print "Slots ==>", rec.__slots__
    print "Format for this table ==>", rec._v_fmt
    print "recarray Format for this table ==>", rec._v_recarrfmt
