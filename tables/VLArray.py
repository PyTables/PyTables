########################################################################
#
#       License: BSD
#       Created: November 12, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/VLArray.py,v $
#       $Id: VLArray.py,v 1.2 2003/11/25 11:26:26 falted Exp $
#
########################################################################

"""Here is defined the VLArray class.

See VLArray class docstring for more info.

Classes:

    VLArray

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.2 $"

# default version for VLARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
import cPickle
import numarray
import numarray.strings as strings
import numarray.records as records
from Leaf import Leaf
#from utils import calcBufferSize
import hdf5Extension
from IsDescription import Col, BoolCol, StringCol, IntCol, FloatCol
from utils import processRange

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

# Class to support variable length strings as components of VLArray
# It supports UNICODE strings as well.
class VLString(IntCol):
    """ Define an atom of type Variable Length String """
    def __init__(self):
        # This special strings will be represented by unsigned bytes
        IntCol.__init__(self, itemsize=1, shape=1, sign=0)
        self.flavor = "VLString"

    def __repr__(self):
        out = "VLString()"
        return out
            

class Atom(Col):
    """ Define an Atomic object to be used in VLArray objects """

    def __init__(self, dtype="Float64", shape=1, flavor="NumArray"):
        Col.__init__(self, dtype, shape)

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

        out = "Atom(type='" +  str(self.type) + "'" + \
              ", shape=" +  str(shape) + \
              ", flavor=" +  str(self.flavor) + \
              ")"
        return out
    
class StringAtom(StringCol, Atom):
    """ Define an atom of type String """
    def __init__(self, shape=1, length=None, flavor="CharArray"):
        StringCol.__init__(self, length=length, shape=shape)
        self.flavor = flavor
        
class BoolAtom(BoolCol, Atom):
    """ Define an atom of type Bool """
    def __init__(self, shape=1, flavor="NumArray"):
        BoolCol.__init__(self, shape=shape)
        self.flavor = flavor

class IntAtom(IntCol, Atom):
    """ Define an atom of type Integer """
    def __init__(self, shape=1, itemsize=4, sign=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=itemsize, sign=sign)
        self.flavor = flavor

class Int8Atom(IntCol, Atom):
    """ Define an atom of type Int8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=1)
        self.flavor = flavor

class UInt8Atom(IntCol, Atom):
    """ Define an atom of type UInt8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=0)
        self.flavor = flavor

class Int16Atom(IntCol, Atom):
    """ Define an atom of type Int16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=2, sign=1)
        self.flavor = flavor

class UInt16Atom(IntCol, Atom):
    """ Define an atom of type UInt16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=2, sign=0)
        self.flavor = flavor

class Int32Atom(IntCol, Atom):
    """ Define an atom of type Int32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=4, sign=1)
        self.flavor = flavor

class UInt32Atom(IntCol, Atom):
    """ Define an atom of type UInt32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=4, sign=0)
        self.flavor = flavor

class Int64Atom(IntCol, Atom):
    """ Define an atom of type Int64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=8, sign=1)
        self.flavor = flavor

class UInt64Atom(IntCol, Atom):
    """ Define an atom of type UInt64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=8, sign=0)
        self.flavor = flavor

class FloatAtom(FloatCol, Atom):
    """ Define an atom of type Float """
    def __init__(self, shape=1, itemsize=8, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = flavor

class Float32Atom(FloatCol, Atom):
    """ Define an atom of type Float32 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=4)
        self.flavor = flavor

class Float64Atom(FloatCol, Atom):
    """ Define an atom of type Float64 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=8)
        self.flavor = flavor

class ObjectAtom(IntCol, Atom):
    """ Define an atom of type Object """
    def __init__(self, shape=1, flavor="Object"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=0)
        self.flavor = flavor

        
def calcChunkSize(expectedsizeinMB, compress):
    """Computes the optimum value for the chunksize"""
    if expectedsizeinMB <= 100:
        # Values for files less than 100 KB of size
        chunksize = 1024
    elif (expectedsizeinMB > 100 and
          expectedsizeinMB <= 1000):
        # Values for files less than 1 MB of size
        chunksize = 2048
    elif (expectedsizeinMB > 1000 and
          expectedsizeinMB <= 20 * 1000):
        # Values for sizes between 1 MB and 20 MB
        chunksize = 4096
    elif (expectedsizeinMB > 20 * 1000 and
          expectedsizeinMB <= 200 * 1000):
        # Values for sizes between 20 MB and 200 MB
        chunksize = 8192
    else:  # Greater than 200 MB
        chunksize = 16384

    # Correction for compression.
    if compress:
        chunksize = 1024   # This seems optimal for compression

    return chunksize


class VLArray(Leaf, hdf5Extension.VLArray, object):
    """Represent a variable length (ragged) array in HDF5 file.

    It enables to create new datasets on-disk from Numeric, numarray,
    lists, tuples, strings or scalars, or open existing ones. The
    datasets are made of records that are made of a variable length
    number of atomic objects (which has to have always the same
    shape).

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

      Common to all leaves:
        close()
        flush()
        getAttr(attrname)
        rename(newname)
        remove()
        setAttr(attrname, attrvalue)
        
      Specific of VLArray:
        read()

    Instance variables:

      Common to all leaves:
        name -- the leaf node name
        hdf5name -- the HDF5 leaf node name
        title -- the leaf title
        shape -- the leaf shape
        byteorder -- the byteorder of the leaf
        
      Specific of VLArray:
        atom -- the class instance choosed for the atomic object

    """
    
    def __init__(self, atom=None, title = "",
                 compress = 0, complib = "zlib", shuffle = 0,
                 expectedsizeinMB = 1.0):
        """Create the instance Array.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and
            flavor of the atomic objects to be saved.
        
        title -- Sets a TITLE attribute on the HDF5 array entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default. A value greater than 0 implies enlargeable
            Arrays (see above).

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        shuffle -- Whether or not to use the shuffle filter in HDF5. This
            is normally used to improve the compression ratio.

        expectedsizeinMB -- An user estimate about the size (in MB) in
            the final VLArray object. If not provided, the default
            value is 1 MB.  If you plan to create both much smaller or
            much bigger Arrays try providing a guess; this will
            optimize the HDF5 B-Tree creation and management process
            time and the amount of memory used.

        """
        self.new_title = title
        if shuffle and not compress:
            # Shuffling and not compressing makes not sense
            shuffle = 0
        self._v_compress = compress
        self._v_complib = complib
        self._v_shuffle = shuffle
        self._v_expectedsizeinMB = expectedsizeinMB
        # Check if we have to create a new object or read their contents
        # from disk
        if atom is not None:
            self.atom = atom
            self._v_new = 1
        else:
            self._v_new = 0

    def _create(self):
        """Create a variable length array (ragged array)."""
        global obversion

        self._v_version = obversion
        # Only support for creating objects in system byteorder
        self.byteorder  = sys.byteorder
        # Special case for Strings (the last value is the string size)
        dtype = self.atom.type
        shape = self.atom.shape
        if dtype == "CharType" or isinstance(dtype, records.Char):
            self.atom.type = records.CharType

        # Compute the optimal chunksize
        self._v_chunksize = calcChunkSize(self._v_expectedsizeinMB,
                                          self._v_compress)
        self.nrows = 0     # No rows in creation time
        self.shape = (0,)
        self._createArray(self.new_title)

    def _convertIntoNA(self, arr):
        "Convert a generic object into a numarray object"
        # Check for Numeric objects
        if (isinstance(arr, numarray.NumArray) or
            isinstance(arr, strings.CharArray)):
            naarr = arr
        elif (Numeric_imported and type(arr) == type(Numeric.array(1))):
            if arr.typecode() == "c":
                # To emulate as close as possible Numeric character arrays,
                # itemsize for chararrays will be always 1
                if arr.iscontiguous():
                    # This the fastest way to convert from Numeric to numarray
                    # because no data copy is involved
                    naarr = strings.array(buffer(arr),
                                          itemsize=1,
                                          shape=arr.shape)
                else:
                    # Here we absolutely need a copy so as to obtain a buffer.
                    # Perhaps this can be avoided or optimized by using
                    # the tolist() method, but this should be tested.
                    naarr = strings.array(buffer(arr.copy()),
                                          itemsize=1,
                                          shape=arr.shape)
            else:
                if arr.iscontiguous():
                    # This the fastest way to convert from Numeric to numarray
                    # because no data copy is involved
                    naarr = numarray.array(buffer(arr),
                                           type=arr.typecode(),
                                           shape=arr.shape)
                else:
                    # Here we absolutely need a copy in order
                    # to obtain a buffer.
                    # Perhaps this can be avoided or optimized by using
                    # the tolist() method, but this should be tested.
                    naarr = numarray.array(buffer(arr.copy()),
                                           type=arr.typecode(),
                                           shape=arr.shape)                    

        else:
            # Test if arr can be converted to a numarray object of the
            # correct type
            try:
                naarr = numarray.array(arr, type=self.atom.type)
            # If not, test with a chararray
            except TypeError:
                try:
                    naarr = strings.array(arr)
                # If still doesn't, issues an error
                except:
                    raise TypeError, \
"""The object '%s' can't be converted into a numarray object of type '%s'. Sorry, but this object is not supported in this context.""" % (arr, self.atom.type)

        # We always want a contiguous buffer
        # (no matter if has an offset or not; that will be corrected later)
        if (not naarr.iscontiguous()):
            # Do a copy of the array in case is not contiguous
            naarr = numarray.NDArray.copy(naarr)

        # Test that this object is shape and type compliant
        # deal with scalars or strings
        if not hasattr(naarr[0], "shape"):
            shape = 1
        else:
            shape = naarr[0].shape
        if shape <> self.atom.shape:
            raise TypeError, \
"""The object '%s' is not composed of elements with shape '%s'.""" % (arr, self.atom.shape)
        if not hasattr(naarr, "type"):  # To deal with string objects
            datatype = records.CharType
            # Made an additional check for strings
            if self.atom.itemsize <> naarr.itemsize():
                raise TypeError, \
"""The object '%s' has not a base string size of '%s'.""" % \
(arr, self.atom.itemsize)
        else:
            datatype = naarr.type()
        if datatype <> self.atom.type:
            raise TypeError, \
"""The object '%s' is not composed of elements of type '%s'.""" % \
(arr, self.atom.type)

        return naarr
            
    def append(self, *objects):
        """Append the object to this enlargeable object"""

        # To make append([1,0,1]) equivalent to append(1,0,1)
        if len(objects) == 1:
            # Correction for only one parameter passed
            object = objects[0]
        else:
            if self.atom.flavor == "VLString":
                raise ValueError, \
"""The append method only accepts one parameter for 'VLString' data type."""
            else:
                object = objects

        # Convert the object to a numarray object
        if self.atom.flavor == "Object":
            # Special case for a generic object
            # (to be pickled and saved as an array of unsigned bytes)
            object = numarray.array(cPickle.dumps(object), type=numarray.UInt8)
        elif self.atom.flavor == "VLString":
            # Special case for a generic object
            # (to be pickled and saved as an array of unsigned bytes)
            if not (isinstance(object, types.StringType) or
                    isinstance(object, types.UnicodeType)):
                raise TypeError, \
"""The object "%s" is not of type String or Unicode.""" % (str(object))
            try:
                object = object.encode('utf-8')
            except:
                (type, value, traceback) = sys.exc_info()
                raise ValueError, "Problems when converting the object '%s' to the encoding 'utf-8'. The error was: %s" % (object, value)
            object = numarray.array(object, type=numarray.UInt8)
        naarr = self._convertIntoNA(object)
        if self._append(naarr) > 0:
            self.nrows += 1
            self.shape = (self.nrows,)
            # Return the last entry in object
            return self.nrows
        else:
            return -1

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.atomictype, self.nrows, self.atomicshape, self.byteorder) = \
                        self._openArray()
        self.shape = (self.nrows,)
        # self.attrs is not available yet
        #self.flavor = self.getAttr("FLAVOR")

    def iterrows(self, start=None, stop=None, step=None):
        """Iterator over all the rows or a range"""

        return self.__call__(start, stop, step)

    def __call__(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.
        
        It returns the same iterator than
        Table.iterrows(start, stop, step).
        It is, therefore, a shorter way to call it.
        """

        (self.start, self.stop, self.step) = \
                     processRange(self.nrows, start, stop, step)
        self._initLoop()
        return self
        
    def __iter__(self):
        """Iterate over all the rows."""

        if not hasattr(self, "_init"):
            # If the iterator is called directly, assign default variables
            self.start = 0
            self.stop = self.nrows
            self.step = 1
            # and initialize the loop
            self._initLoop()
        return self

    def _initLoop(self):
        "Initialization for the __iter__ iterator"

        self._nrowsinbuf = 100    # Maybe enough for most applications
        self.nrowsread = self.start
        self.startb = self.start
        self._row = -1   # Sentinel
        self._init = 1    # Sentinel

    def next(self):
        "next() method for __iter__() that is called on each iteration"
        if self.nrowsread >= self.stop:
            del self._init
            raise StopIteration        # end of iteration
        else:
            # Read a chunk of rows
            if self._row > self._nrowsinbuf or self._row < 0:
                self.stopb = self.startb+self.step*self._nrowsinbuf
                self.listarr = self.read(self.startb, self.stopb, self.step)
                #print "listarr-->", self.listarr
                self._row = -1
                self.startb = self.stopb
            self._row += 1
            self.nrowsread += self.step
            #print "_row-->", self._row
            return self.listarr[self._row]

    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=None):
        """Read the array from disk and return it as numarray."""

        (start, stop, step) = processRange(self.nrows, start, stop, step)

        if start == stop:
            listarr = []
        else:
            listarr = self._readArray(start, stop, step)
            
        #self.flavor = self.getAttr("FLAVOR")
        self.flavor = self.attrs.FLAVOR
        outlistarr = []
        for arr in listarr:
            # Convert to Numeric, tuple or list if needed
            if self.flavor == "Numeric":
                if Numeric_imported:
                    # This works for both numeric and chararrays
                    # arr=Numeric.array(arr, typecode=arr.typecode())
                    # The next is 10 times faster (for tolist(),
                    # we should check for tostring()!)
                    if repr(self.atomictype) == "CharType":
                        arrstr = arr.tostring()
                        arr=Numeric.reshape(Numeric.array(arrstr), arr.shape)
                    else:
                        # tolist() method creates a list with a sane byteorder
                        if arr.shape <> ():
                            arr=Numeric.array(arr.tolist(), typecode=arr.typecode())
                        else:
                            # This works for rank-0 arrays
                            # (but is slower for big arrays)
                            arr=Numeric.array(arr, typecode=arr.typecode())

                else:
                    # Warn the user
                    warnings.warn( \
    """The object on-disk is type Numeric, but Numeric is not installed locally.
      Returning a numarray object instead!.""")
            elif self.flavor == "Tuple":
                arr = tuple(arr.tolist())
            elif self.flavor == "List":
                arr = arr.tolist()
            elif self.flavor == "Int":
                arr = int(arr)
            elif self.flavor == "Float":
                arr = float(arr)
            elif self.flavor == "String":
                arr = arr.tostring()
            elif self.flavor == "VLString":
                arr = arr.tostring().decode('utf-8')
            elif self.flavor == "Object":
                arr = cPickle.loads(arr.tostring())

            outlistarr.append(arr)
        return outlistarr
        
    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return "%s\n  type = %r\n  shape = %r\n  flavor = %r\n  byteorder = %r" % \
               (self, self.atom.type, self.shape, self.attrs.FLAVOR, self.byteorder)
