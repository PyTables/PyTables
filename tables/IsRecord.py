########################################################################
#
#       Copyright:      LGPL
#       Created:        September 21, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Attic/IsRecord.py,v $
#       $Id: IsRecord.py,v 1.3 2002/11/07 17:52:35 falted Exp $
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

__version__ = "$Revision: 1.3 $"


import warnings
import struct
import sys

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
            for k in self.__dflts__: setattr(self, k, self.__dflts__[k])

            # Initialize the values passed as keyword parameters
            for k in kw:
                setattr(self, k, kw[k])

            # Convert this to a struct
            values = [ getattr(self, k) for k in self.__slots__ ]
            #print "About to convert values ==>", values
            #print "with format ==>", self._v_fmt
            #buffer = struct.pack(self._v_fmt, *values)
                
            #print "And the result is ==>", buffer
            
        
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

        def __call__(self, *args, **kw):
            """ Method to pack by default. We choose _f_pack."""
            if args and not kw:
                return struct.pack(self._v_fmt, *args)
            elif kw and not args:
                return self._f_pack(**kw)
            elif not args and not kw:
                return self._f_pack()
            else:
                raise RuntimeError, \
                  "Mix of variable-length args and keyword is not supported!"
                
        def _f_raiseValueError(self):
            """Helper function to indicate an error in struct.pack and
            provide detailed information on the record object.
            """
            (type, value, traceback) = sys.exc_info()
            record = []
            for k in self.__slots__:
                record.append((k, self.__types__[k], (getattr(self, k))))
            raise ValueError, \
             "Error packing record object: \n %s\n Error was: %s" % \
             (record, value)           
            
        def _f_pack(self, **kw):
            """A method to pack values.

            Notes:
            
            - Some keyword parameters allowed
            - Record updated
            - 3.79 s for 100.000 records  (all keywords set)
            - 2.98 s for 100.000 records  (no keywords set)
            
            This is the most flexible method, and if called with no
            keywords is reasonably fast.
            
            """
            # Initialize the values passed as keyword parameters
            for k in kw:
                setattr(self, k, kw[k])
            # Convert this to a struct
            values = [ getattr(self, k) for k in self.__slots__ ]
            try:
                buffer = struct.pack(self._v_fmt, *values)
            except struct.error:
                self._f_raiseValueError()

            return buffer

        def _f_pack2(self):
            """Method to pack values (2)

            Notes:
            
            - No parameters allowed
            - Record updated
            - 2.74 s for 100.000 records (with slots)
            - 2.83 s for 100.000 records (without slots)
            
            """
            # Convert slot elements to a struct
            values = [ getattr(self, k) for k in self.__slots__ ]
            try:
                buffer = struct.pack(self._v_fmt, *values)
            except struct.error:
                self._f_raiseValueError()
                
            return buffer

        def _f_packFast(self, **kw):
            """ Method to pack 3

            Notes:
            
            - All slots should be specified as keyworks
            - Record not updated!
            - 2.39 s for 100.000 rec when saveFast(var1 = var1,var2 = var2,...)
            - 2.22 s for 100.000 rec when saveFast(var1 = 12.3,var2="1", ...)
            
            """
            # Initialize the values passed as keyword parameters
            values = [ kw[k] for k in self.__slots__ ] # Get the ordered values
            return struct.pack(self._v_fmt, *values)

        def _f_packFastest(self, *values):
            """ Method to pack 4
            - Positional (alphanumeric order) params. All should be specified
            - Record not updated!
            - 0.99 s for 100.000 rec when saveFastest(var1, var2, ...)
            - 0.88 s for 100.000 rec when saveFastest(12.3, "1", ...)
            This is the fastest method to pack values, but we must be very
            careful on the order we put the Record elements!.
            """
            return struct.pack(self._v_fmt, *values)

        def _f_unpack(self, buffer):
            """ Method to unpack and set attributes.

            Notes:
            
            - Record updated
            - 3.43 s for 100.000 records
            
            """
            # Maybe we can get this faster if we found a way to setting
            # the slots without calling setattr
            # Anyway, this might be solved when numarray will get integrated
            # to do the I/O.
            tupla = struct.unpack(self._v_fmt, buffer)
            i = 0
            for k in self.__slots__:
                setattr(self, k, tupla[i])
                i += 1

        def _f_unpack2(self, buffer):
            """ Another method to unpack.

            Notes:
            
            - Record updated
            - 3.72 s for 100.000 records
            
            """
            i = 0
            for value in struct.unpack(self._v_fmt, buffer):
                setattr(self, self.__slots__[i], value)
                i += 1

        def testtype(datatype):
            """Test if datatype is valid and returns a default value for
            each one.
            """
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
		    '__str__':__str__,
                    '__call__':__call__, '_v_fmt': "",
                    '_f_raiseValueError':_f_raiseValueError, 
                    '_f_pack':_f_pack, '_f_pack2':_f_pack2,
                    '_f_packFast':_f_packFast, '_f_packFastest':_f_packFastest,
                    '_f_unpack':_f_unpack, '_f_unpack2':_f_unpack2,}

        keys = classdict.keys()
        keys.sort() # Sort the keys to establish an order
        for k in keys:
            if (k.startswith('__') or k.startswith('_v_') 
                or k.startswith('_f_')):
                if k in newdict:
                    # special methods &c: copy to newdict, warn about conflicts
                    warnings.warn("Can't set attr %r in record-class %r" % (
                        k, classname))
                else:
                    newdict[k] = classdict[k]
            else:
                # class variables, store name in __slots__ and name and
                # value as an item in __dflts__
                newdict['__slots__'].append(k)
                datatype = classdict[k]
                dfltvalue = testtype(datatype)
                newdict['__types__'][k] = datatype
                newdict['__dflts__'][k] = dfltvalue
                newdict['_v_fmt'] += datatype

        # Set up the alignment (if exists)
        if newdict.has_key('_v_align'):
            newdict['_v_fmt'] = newdict['_v_align'] + newdict['_v_fmt']
        # Execute this only when the real instance is created
        # (not the parent class)
        if newdict.has_key('_v_fmt'):
            fmt = newdict['_v_fmt']
            #print "Format for this table ==>", newdict['_v_fmt']
            # Check for validity of this format string
            try:
                struct.calcsize(fmt)
            except struct.error:
                raise TypeError, "This type format (%s) is not supported!." % \
                      fmt

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
        x = 'l'
        y = 'd'
        color = '3s'

    # example uses of class Point
    q = Record()  # Default values
    print q

    rec = Record(x=4, y=3.4, color = "pwpwp")
    print rec
    for i in range(0*100000):
        # Some keyword parameters allowed (all set)
        # 3.79 s for 100.000 records (with slots) (3.45 with -O)
        rec._f_pack(color = "pwpwp", y = 23.5, x = 5)
                                      
    for i in range(0*100000):
        # Some keyword parameters allowed (none set)
        # 2.98 s for 100.000 records 
        rec.x = 5
        rec.y = 23.4
        rec.color = "pwpwp"
        rec._f_pack()
                                      
    for i in xrange(100000):
        # No parameters allowed
        # 2.74 s for 100.000 records (with slots)
        # 2.83 s for 100.000 records (without slots)
        rec.x = 5
        rec.y = 23.4
        rec.color = "pwpwp"
        rec._f_pack2()

    for i in range(0*100000):
        # All slots should be specified  (p not modified)
        # 2.39 s for 100.000 rec
        x = 5
        y = 23.4
        color = "pwpwp"
        rec._f_packFast(color = color, x = x, y = y)
        
    for i in range(0*100000):
        # All slots should be specified (II) (p not modified)
        # 2.22 s for 100.000 rec
        rec._f_packFast(color = "pwpwp", x = 5, y = 24.5)

    for i in range(0*100000):
        # Positional (alphanumeric order) params (p not modified)
        # 0.99 s for 100.000 rec (fastest)
        x = 5
        y = 23.4
        color = "pwpwp"
        rec._f_packFastest(color, x, y)     

    for i in range(0*100000):
        # Positional (alphanumeric order) params (II) (p not modified)
        # 0.88 s for 100.000 rec (fastest)
        rec._f_packFastest("pwpwp", 5, 23.5)     
                                 
    print "rec value ==>", rec
    print "packed struct ==> (%s)" % rec._f_pack()
    
    print "Slots ==>", rec.__slots__
    print "Format for this table ==>", rec._v_fmt
