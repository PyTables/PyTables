########################################################################
#
#       License: BSD
#       Created: November 12, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/VLArray.py,v $
#       $Id: VLArray.py,v 1.14 2003/12/28 23:23:25 falted Exp $
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

__version__ = "$Revision: 1.14 $"

# default version for VLARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
import cPickle
import numarray
import numarray.strings as strings
import numarray.records as records
from Leaf import Leaf
import hdf5Extension
from IsDescription import Col, BoolCol, StringCol, IntCol, FloatCol
from utils import processRange, processRangeRead, convertIntoNA

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

def checkflavor(flavor):
    if flavor in ["NumArray", "CharArray", "Numeric", "Tuple", "List",
                  "String", "Object"]:
        return flavor
    else:
        raise ValueError, \
"""flavor must be one of the "NumArray", "CharArray", "Numeric", "Tuple",
 "List", "String" or "Object" values, and you try to set it to "%s".
"""  % (flavor)

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


class Atom(Col):
    """ Define an Atomic object to be used in VLArray objects """

    def __init__(self, dtype="Float64", shape=1, flavor="NumArray"):
        Col.__init__(self, dtype, shape)
        self.flavor = checkflavor(flavor)

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

        out = "Atom(type=" +  str(self.type) + \
              ", shape=" +  str(shape) + \
              ", flavor=" + "'" + str(self.flavor) + "'" + \
              ")"
        return out

    def atomsize(self):
        " Compute the size of the atom type "
        atomicsize = self.itemsize
        if isinstance(self.shape, types.TupleType):
            for i in self.shape:
                atomicsize *= i
        else:
            atomicsize *= self.shape
        return atomicsize

    
class StringAtom(Atom, StringCol):
    """ Define an atom of type String """
    def __init__(self, shape=1, length=None, flavor="CharArray"):
        StringCol.__init__(self, length=length, shape=shape)
        self.flavor = checkflavor(flavor)
        
class BoolAtom(Atom, BoolCol):
    """ Define an atom of type Bool """
    def __init__(self, shape=1, flavor="NumArray"):
        BoolCol.__init__(self, shape=shape)
        self.flavor = checkflavor(flavor)

class IntAtom(Atom, IntCol):
    """ Define an atom of type Integer """
    def __init__(self, shape=1, itemsize=4, sign=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=itemsize, sign=sign)
        self.flavor = checkflavor(flavor)

class Int8Atom(Atom, IntCol):
    """ Define an atom of type Int8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=1)
        self.flavor = checkflavor(flavor)

class UInt8Atom(Atom, IntCol):
    """ Define an atom of type UInt8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=0)
        self.flavor = checkflavor(flavor)

class Int16Atom(Atom, IntCol):
    """ Define an atom of type Int16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=2, sign=1)
        self.flavor = checkflavor(flavor)

class UInt16Atom(Atom, IntCol):
    """ Define an atom of type UInt16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=2, sign=0)
        self.flavor = checkflavor(flavor)

class Int32Atom(Atom, IntCol):
    """ Define an atom of type Int32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=4, sign=1)
        self.flavor = checkflavor(flavor)

class UInt32Atom(IntCol, Atom):
    """ Define an atom of type UInt32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=4, sign=0)
        self.flavor = checkflavor(flavor)

class Int64Atom(Atom, IntCol):
    """ Define an atom of type Int64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=8, sign=1)
        self.flavor = checkflavor(flavor)

class UInt64Atom(Atom, IntCol):
    """ Define an atom of type UInt64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=8, sign=0)
        self.flavor = checkflavor(flavor)

class FloatAtom(Atom, FloatCol):
    """ Define an atom of type Float """
    def __init__(self, shape=1, itemsize=8, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = checkflavor(flavor)

class Float32Atom(Atom, FloatCol):
    """ Define an atom of type Float32 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=4)
        self.flavor = checkflavor(flavor)

class Float64Atom(Atom, FloatCol):
    """ Define an atom of type Float64 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=8)
        self.flavor = checkflavor(flavor)

        
def calcChunkSize(expectedsizeinMB, complevel):
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
    if complevel:
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

        append(*objects)
        read(start, stop, step)
        iterrows(start, stop, step)

    Instance variables:

        atom -- the class instance choosed for the atomic object
        nrow -- On iterators, this is the index of the row currently
            dealed with.
        nrows -- The total number of rows
            

    """
    
    def __init__(self, atom=None, title = "",
                 compress = 0, complib = "zlib", shuffle = 1,
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
        self._v_expectedsizeinMB = expectedsizeinMB
        self._v_maxTuples = 100    # Maybe enough for most applications
        # Check if we have to create a new object or read their contents
        # from disk
        if atom is not None:
            self.atom = atom
            self._v_new = 1
            self._g_setComprAttr(compress, complib, shuffle)
        else:
            self._v_new = 0

    def _create(self):
        """Create a variable length array (ragged array)."""
        global obversion

        self._v_version = obversion
        # Only support for creating objects in system byteorder
        self.byteorder  = sys.byteorder
        dtype = self.atom.type
        shape = self.atom.shape
        if dtype == "CharType" or isinstance(dtype, records.Char):
            self.atom.type = records.CharType

        self._atomictype = self.atom.type
        self._atomicshape = self.atom.shape
        self._atomicsize = self.atom.atomsize()
        self._basesize = self.atom.itemsize
        self.flavor = self.atom.flavor

        # Compute the optimal chunksize
        self._v_chunksize = calcChunkSize(self._v_expectedsizeinMB,
                                          self.complevel)
        self.nrows = 0     # No rows in creation time
        self.shape = (0,)
        self._createArray(self.new_title)

    
    def _checkTypeShape(self, naarr):
        # Test that this object is shape and type compliant
        # Check for type
        if not hasattr(naarr, "type"):  # To deal with string objects
            datatype = records.CharType
            # Made an additional check for strings
            if self.atom.itemsize <> naarr.itemsize():
                raise TypeError, \
"""The object '%s' has not a base string size of '%s'.""" % \
(naarr, self.atom.itemsize)
        else:
            datatype = naarr.type()
        if str(datatype) <> str(self.atom.type):
            raise TypeError, \
"""The object '%s' is not composed of elements of type '%s'.""" % \
(naarr, self.atom.type)

        if len(naarr):
            if hasattr(naarr, "shape") and naarr.shape == self.atom.shape:
                # Case of only one element
                shape = self.atom.shape
                self._nobjects = 1
            else:
                # See if naarr is made of elements with the correct shape
                if not hasattr(naarr[0], "shape"):
                    shape = 1
                else:
                    shape = naarr[0].shape
                self._nobjects = len(naarr)
                if shape <> self.atom.shape:
                    raise TypeError, \
"""The object '%s' is composed of elements with shape '%s', not '%s'.""" % \
    (naarr, shape, self.atom.shape)
        else:
            self._nobjects = 0
        # Ok. all conditions are meet. Return the numarray object
        return naarr
            
    def append(self, *objects):
        """Append the objects to this enlargeable object"""

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
        # Prepare the object to convert it into a numarray object
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
            
        naarr = convertIntoNA(object, self.atom.type)
        self._checkTypeShape(naarr)
        if self._append(naarr, self._nobjects) > 0:
            self.nrows += 1
            self.shape = (self.nrows,)
            # Return the last entry in object
            return self.nrows
        else:
            return -1

    def _open(self):
        """Get the metadata info for an array in file."""
        self.nrows = self._openArray()
        self.shape = (self.nrows,)
        if self.flavor == "VLString":
            self.atom = VLStringAtom()
        else:
            if str(self._atomictype) == "CharType":
                self.atom = StringAtom(shape=self._atomicshape,
                                       length=self._basesize,
                                       flavor=self.flavor)
            else:
                self.atom = Atom(self._atomictype, self._atomicshape,
                                 self.flavor)
        # Get info about existing filters
        self.complevel, self.complib, self.shuffle = self._g_getFilters()

    def iterrows(self, start=None, stop=None, step=None):
        """Iterator over all the rows or a range"""

        return self.__call__(start, stop, step)

    def __call__(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.
        
        It returns the same iterator than
        Table.iterrows(start, stop, step).
        It is, therefore, a shorter way to call it.
        """

        (self._start, self._stop, self._step) = \
                     processRangeRead(self.nrows, start, stop, step)
        self._initLoop()
        return self
        
    def __iter__(self):
        """Iterate over all the rows."""

        if not hasattr(self, "_init"):
            # If the iterator is called directly, assign default variables
            self._start = 0
            self._stop = self.nrows
            self._step = 1
            # and initialize the loop
            self._initLoop()
        return self

    def _initLoop(self):
        "Initialization for the __iter__ iterator"

        self._nrowsread = self._start
        self._startb = self._start
        self._row = -1   # Sentinel
        self._init = 1    # Sentinel
        self.nrow = self._start - self._step    # row number

    def next(self):
        "next() method for __iter__() that is called on each iteration"
        if self._nrowsread >= self._stop:
            del self._init
            raise StopIteration        # end of iteration
        else:
            # Read a chunk of rows
            if self._row+1 >= self._v_maxTuples or self._row < 0:
                self._stopb = self._startb+self._step*self._v_maxTuples
                self.listarr = self.read(self._startb, self._stopb, self._step)
                self._row = -1
                self._startb = self._stopb
            self._row += 1
            self.nrow += self._step
            self._nrowsread += self._step
            return self.listarr[self._row]

    def _convToFlavor(self, arr):
        "convert the numarray parameter to to correct flavor"

        # Convert to Numeric, tuple or list if needed
        if self.flavor == "Numeric":
            if Numeric_imported:
                # This works for both numeric and chararrays
                # arr=Numeric.array(arr, typecode=arr.typecode())
                # The next is 10 times faster (for tolist(),
                # we should check for tostring()!)
                if str(self._atomictype) == "CharType":
                    arrstr = arr.tostring()
                    arr=Numeric.reshape(Numeric.array(arrstr), arr.shape)
                else:
                    if str(arr.type()) == "Bool":
                        # Typecode boolean does not exist on Numeric
                        typecode = "1"
                    else:
                        typecode = arr.typecode()                        
                    # tolist() method creates a list with a sane byteorder
                    if arr.shape <> ():
                        arr=Numeric.array(arr.tolist(), typecode)
                    else:
                        # This works for rank-0 arrays
                        # (but is slower for big arrays)
                        arr=Numeric.array(arr, typecode)

            else:
                # Warn the user
                warnings.warn( \
"""The object on-disk is type Numeric, but Numeric is not installed locally.
  Returning a numarray object instead!.""")
        elif self.flavor == "Tuple":
            arr = tuple(arr.tolist())
        elif self.flavor == "List":
            arr = arr.tolist()
        elif self.flavor == "String":
            arr = arr.tolist()
        elif self.flavor == "VLString":
            arr = arr.tostring().decode('utf-8')
        elif self.flavor == "Object":
            arr = cPickle.loads(arr.tostring())

        # Check for unitary length lists or tuples
        # it is better to disable it, as it is more consistent to return
        # unitary values as an additional dimension than removing it
#         if len(arr) == 1:
#             arr = arr[0]

        return arr

    def __getitem__(self, key):
        """Returns a table row, table slice or table column.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key"is an integer, the corresponding table row is returned
        as a RecArray.Record object. If "key" is a slice, the row
        slice determined by key is returned as a RecArray object.
        Finally, if "key" is a string, it is interpreted as a column
        name in the table, and, if it exists, it is read and returned
        as a NumArray or CharArray object (whatever is appropriate).

"""

        if isinstance(key, types.IntType):
            return self.read(key, key+1, 1)[0]
        elif isinstance(key, types.SliceType):
            return self.read(key.start, key.stop, key.step)
        else:
            raise ValueError, "Non-valid index or slice: %s" % \
                  key
        
    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=None):
        """Read the array from disk and return it as numarray."""

        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)

        if start == stop:
            listarr = []
        else:
            listarr = self._readArray(start, stop, step)

        if self.flavor <> "NumArray":
            # Convert the list to the right flavor
            outlistarr = [self._convToFlavor(arr) for arr in listarr ]
        else:
            # NumArray flavor does not need additional conversion
            outlistarr = listarr

        # Check for unitary length lists or tuples
        if len(outlistarr) == 1:
            outlistarr = outlistarr[0]
            
        return outlistarr

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom = %r
  nrows = %s
  flavor = %r
  byteorder = %r""" % (self, self.atom, self.nrows,
                       self.flavor, self.byteorder)
