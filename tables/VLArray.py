########################################################################
#
#       License: BSD
#       Created: November 12, 2003
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/VLArray.py,v $
#       $Id: VLArray.py,v 1.30 2004/09/17 11:51:48 falted Exp $
#
########################################################################

"""Here is defined the VLArray class and declarative classes for its components

See VLArray class docstring and *Atom docstrings for more info.

Classes:

    Atom, ObjectAtom, VLStringAtom, StringAtom, BoolAtom,
    IntAtom, Int8Atom, UInt8Atom, Int16Atom, UInt16Atom,
    VLArray

Functions:

   checkflavor

Misc variables:

    __version__


"""

__version__ = "$Revision: 1.30 $"

# default version for VLARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
import cPickle
import numarray
import numarray.strings as strings
import numarray.records as records
from Leaf import Leaf
import hdf5Extension
#import IsDescription # to access BaseCol without polluting public namespace
from IsDescription import Col, BoolCol, StringCol, IntCol, FloatCol, ComplexCol
from utils import processRange, processRangeRead, convertIntoNA

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

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
                if i > 0:  # To deal with EArray Atoms
                    atomicsize *= i
        else:
            atomicsize *= self.shape
        return atomicsize

    
class StringAtom(Atom, StringCol):
    """ Define an atom of type String """
    def __init__(self, shape=1, length=None, flavor="CharArray"):
        StringCol.__init__(self, length=length, shape=shape)
        self.flavor = checkflavor(flavor, self.type)
        
class BoolAtom(Atom, BoolCol):
    """ Define an atom of type Bool """
    def __init__(self, shape=1, flavor="NumArray"):
        BoolCol.__init__(self, shape=shape)
        self.flavor = checkflavor(flavor, self.type)

class IntAtom(Atom, IntCol):
    """ Define an atom of type Integer """
    def __init__(self, shape=1, itemsize=4, sign=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=itemsize, sign=sign)
        self.flavor = checkflavor(flavor, self.type)

class Int8Atom(Atom, IntCol):
    """ Define an atom of type Int8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=1)
        self.flavor = checkflavor(flavor, self.type)

class UInt8Atom(Atom, IntCol):
    """ Define an atom of type UInt8 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=1, sign=0)
        self.flavor = checkflavor(flavor, self.type)

class Int16Atom(Atom, IntCol):
    """ Define an atom of type Int16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=2, sign=1)
        self.flavor = checkflavor(flavor, self.type)

class UInt16Atom(Atom, IntCol):
    """ Define an atom of type UInt16 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=2, sign=0)
        self.flavor = checkflavor(flavor, self.type)

class Int32Atom(Atom, IntCol):
    """ Define an atom of type Int32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=4, sign=1)
        self.flavor = checkflavor(flavor, self.type)

class UInt32Atom(IntCol, Atom):
    """ Define an atom of type UInt32 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=4, sign=0)
        self.flavor = checkflavor(flavor, self.type)

class Int64Atom(Atom, IntCol):
    """ Define an atom of type Int64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=8, sign=1)
        self.flavor = checkflavor(flavor, self.type)

class UInt64Atom(Atom, IntCol):
    """ Define an atom of type UInt64 """
    def __init__(self, shape=1, flavor="NumArray"):
        IntCol.__init__(self, shape=shape, itemsize=8, sign=0)
        self.flavor = checkflavor(flavor, self.type)

class FloatAtom(Atom, FloatCol):
    """ Define an atom of type Float """
    def __init__(self, shape=1, itemsize=8, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = checkflavor(flavor, self.type)

class Float32Atom(Atom, FloatCol):
    """ Define an atom of type Float32 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=4)
        self.flavor = checkflavor(flavor, self.type)

class Float64Atom(Atom, FloatCol):
    """ Define an atom of type Float64 """
    def __init__(self, shape=1, flavor="NumArray"):
        FloatCol.__init__(self, shape=shape, itemsize=8)
        self.flavor = checkflavor(flavor, self.type)

class ComplexAtom(Atom, ComplexCol):
    """ Define an atom of type Complex """
    def __init__(self, shape=1, itemsize=16, flavor="NumArray"):
        ComplexCol.__init__(self, shape=shape, itemsize=itemsize)
        self.flavor = checkflavor(flavor, self.type)

class Complex32Atom(Atom, ComplexCol):
    """ Define an atom of type Complex32 """
    def __init__(self, shape=1, flavor="NumArray"):
        ComplexCol.__init__(self, shape=shape, itemsize=8)
        self.flavor = checkflavor(flavor, self.type)

class Complex64Atom(Atom, ComplexCol):
    """ Define an atom of type Complex64 """
    def __init__(self, shape=1, flavor="NumArray"):
        ComplexCol.__init__(self, shape=shape, itemsize=16)
        self.flavor = checkflavor(flavor, self.type)

        
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
        __iter__()
        iterrows(start, stop, step)
        __getitem__(slice)

    Instance variables:

        atom -- the class instance choosed for the atomic object
        nrow -- On iterators, this is the index of the row currently
            dealed with.
        nrows -- The total number of rows

    """
    
    def __init__(self, atom=None, title = "",
                 filters = None, expectedsizeinMB = 1.0):
        """Create the instance Array.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and
            flavor of the atomic objects to be saved.
        
        title -- Sets a TITLE attribute on the HDF5 array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedsizeinMB -- An user estimate about the size (in MB) in
            the final VLArray object. If not provided, the default
            value is 1 MB.  If you plan to create both much smaller or
            much bigger Arrays try providing a guess; this will
            optimize the HDF5 B-Tree creation and management process
            time and the amount of memory used.

        """
        self._v_new_title = title
        self._v_new_filters = filters
        self._v_expectedsizeinMB = expectedsizeinMB
        self._v_maxTuples = 100    # Maybe enough for most applications
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
        dtype = self.atom.type
        shape = self.atom.shape
        if dtype == "CharType" or isinstance(dtype, records.Char):
            self.atom.type = records.CharType
        # Check for zero dims in atom shape (not allowed in VLArrays)
        zerodims = numarray.sum(numarray.array(shape) == 0)
        if zerodims > 0:
            raise ValueError, \
                  "When creating VLArrays, none of the dimensions of the Atom instance can be zero."

        self._atomictype = self.atom.type
        self._atomicshape = self.atom.shape
        self._atomicsize = self.atom.atomsize()
        self._basesize = self.atom.itemsize
        self.flavor = self.atom.flavor

        # Compute the optimal chunksize
        self._v_chunksize = calcChunkSize(self._v_expectedsizeinMB,
                                          self.filters.complevel)
        self.nrows = 0     # No rows in creation time
        self.shape = (0,)
        self._createArray(self._v_new_title)
            
    def _checkShape(self, naarr):
        # Check for zero dimensionality array
        zerodims = numarray.sum(numarray.array(naarr.shape) == 0)
        if zerodims > 0:
            # No objects to be added
            return 0
        shape = naarr.shape
        atom_shape = self.atom.shape
        shapelen = len(naarr.shape)
        if isinstance(atom_shape, types.TupleType):
            atomshapelen = len(self.atom.shape)
        else:
            atom_shape = (self.atom.shape,)
            atomshapelen = 1
        diflen = shapelen - atomshapelen
        if shape == atom_shape:
            nobjects = 1
        elif (diflen == 1 and shape[diflen:] == atom_shape):
            # Check if the leading dimensions are all ones
            #if shape[:diflen-1] == (1,)*(diflen-1):
            #    nobjects = shape[diflen-1]
            #    shape = shape[diflen:]
            # It's better to accept only inputs with the exact dimensionality
            # i.e. a dimensionality only 1 element larger than atom
            nobjects = shape[0]
            shape = shape[1:]
        elif atom_shape == (1,) and shapelen == 1:
            # Case where shape = (N,) and shape_atom = 1 or (1,)
            nobjects = shape[0]
        else:
            raise ValueError, \
"""The object '%s' is composed of elements with shape '%s', which is not compatible with the atom shape ('%s').""" % \
(naarr, shape, atom_shape)
        return nobjects
            
    def append(self, *objects):
        """Append the objects to this enlargeable object"""

        # To make append([1,0,1]) equivalent to append(1,0,1)
        if len(objects) == 0:
            object = None
        elif len(objects) == 1:
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

        if len(objects) > 0:
            naarr = convertIntoNA(object, self.atom)
            nobjects = self._checkShape(naarr)
        else:
            nobjects = 0
            naarr = None
        if self._append(naarr, nobjects) > 0:
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
        # First, check the special cases VLString and Object types
        if self.flavor == "VLString":
            self.atom = VLStringAtom()
        elif self.flavor == "Object":
            self.atom = ObjectAtom()
        else:
            if str(self._atomictype) == "CharType":
                self.atom = StringAtom(shape=self._atomicshape,
                                       length=self._basesize,
                                       flavor=self.flavor)
            else:
                self.atom = Atom(self._atomictype, self._atomicshape,
                                 self.flavor)

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
            # We have to check for an empty array because of a
            # possible bug in HDF5 that claims that a dataset
            # has one record when in fact, it is empty
            if len(arr) == 0:
                arr = []
            else:
                arr = cPickle.loads(arr.tostring())
            # The next should also do the job
#             try:
#                 arr = cPickle.loads(arr.tostring())
#             except:
#                 arr = []

        return arr

    def __getitem__(self, key):
        """Returns a table row, table slice or table column.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key"is an integer, the corresponding row is returned. If
        "key" is a slice, the row slice determined by key is returned.

"""

        if isinstance(key, types.IntType):
            if key >= self.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            (start, stop, step) = processRange(self.nrows, key, key+1, 1)
            return self.read(start, stop, step, slice_specified=0)
        elif isinstance(key, types.SliceType):
            (start, stop, step) = processRange(self.nrows,
                                               key.start, key.stop, key.step)
            return self.read(start, stop, step, slice_specified=1)
        else:
            raise ValueError, "Non-valid index or slice: %s" % \
                  key
        
    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=1, slice_specified=0):
        """Read the array from disk and return it as a self.flavor object."""

#         if stop <> None:
#             stop_specified = 1
        start, stop, step = processRangeRead(self.nrows, start, stop, step)

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
        #if len(outlistarr) == 1 and not stop_specified:
        if len(outlistarr) == 1 and not slice_specified:
            outlistarr = outlistarr[0]
            
        return outlistarr

    def _g_copy(self, group, name, start, stop, step, title, filters):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new VLArray object
        object = VLArray(self.atom, title=title,
                         filters=filters,
                         expectedsizeinMB=self._v_expectedsizeinMB)
        setattr(group, name, object)
        # Now, fill the new vlarray with values from the old one
        # This is not buffered because we cannot forsee the length
        # of each record. So, the safest would be a copy row by row.
        # In the future, some analysis can be done in order to buffer
        # the copy process.
        nrowsinbuf = 1
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        # Optimized version (no conversions, no type and shape checks, etc...)
        nrowscopied = 0
        nbytes = 0
        atomsize = self.atom.atomsize()
        for start2 in xrange(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop 
            naarr = self._readArray(start=start2, stop=stop2, step=step)[0]
            nobjects = naarr.shape[0]
            object._append(naarr, nobjects)
            nbytes += nobjects*atomsize
            nrowscopied +=1
        object.nrows = nrowscopied
        object.shape = (nrowscopied,)
        return (object, nbytes)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom = %r
  nrows = %s
  flavor = %r
  byteorder = %r""" % (self, self.atom, self.nrows,
                       self.flavor, self.byteorder)
