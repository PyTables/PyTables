# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

########################################################################
#
#       License: BSD
#       Created: November 12, 2003
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/VLArray.py,v $
#       $Id$
#
########################################################################

"""Here is defined the VLArray class

See VLArray class docstring for more info.

Classes:

    VLArray

Functions:


Misc variables:

    __version__


"""

import sys
import warnings
import cPickle

import numarray
import numarray.records as records

try:
    import Numeric
    Numeric_imported = True
except ImportError:
    Numeric_imported = False

import tables.hdf5Extension as hdf5Extension
from tables.utils import processRangeRead, convertIntoNA
from tables.Atom import Atom, ObjectAtom, VLStringAtom, StringAtom
from tables.Leaf import Leaf



__version__ = "$Revision: 1.42 $"

# default version for VLARRAY objects
#obversion = "1.0"    # initial version
#obversion = "1.0"    # add support for complex datatypes
obversion = "1.1"    # This adds support for time datatypes.



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


class VLArray(hdf5Extension.VLArray, Leaf):
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
        __setitem__(slice, value)

    Instance variables:

        atom -- the class instance choosed for the atomic object
        nrow -- On iterators, this is the index of the row currently
            dealed with.
        nrows -- The total number of rows

    """

    # Class identifier.
    _c_classId = 'VLARRAY'


    # <undo-redo support>
    _c_canUndoCreate = True  # Can creation/copying be undone and redone?
    _c_canUndoRemove = True  # Can removal be undone and redone?
    _c_canUndoMove   = True  # Can movement/renaming be undone and redone?
    # </undo-redo support>


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

        # All this will eventually end up in the node constructor.

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
        self._atomicstype = self.atom.stype
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
        if isinstance(atom_shape, tuple):
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
        assert not self._v_file.mode == "r", "Attempt to write over a file opened in read-only mode"

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
            if type(object) not in (str,unicode):
                raise TypeError, \
"""The object "%s" is not of type String or Unicode.""" % (str(object))
            try:
                object = object.encode('utf-8')
            except UnicodeError:
                (typerr, value, traceback) = sys.exc_info()
                raise ValueError, "Problems when converting the object '%s' to the encoding 'utf-8'. The error was: %s" % (object, value)
            object = numarray.array(object, type=numarray.UInt8)

        if len(objects) > 0:
            # The object needs to be copied to make the operation safe
            # to in-place conversion.
            copy = self._atomicstype in ['Time64']
            naarr = convertIntoNA(object, self.atom, copy)
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

        # All this will eventually end up in the node constructor.

        self.nrows = self._openArray()
        self.shape = (self.nrows,)
        # First, check the special cases VLString and Object types
        if self.flavor == "VLString":
            self.atom = VLStringAtom()
        elif self.flavor == "Object":
            self.atom = ObjectAtom()
        else:
            if str(self._atomicstype) == 'CharType':
                self.atom = StringAtom(shape=self._atomicshape,
                                       length=self._basesize,
                                       flavor=self.flavor)
            else:
                self.atom = Atom(self._atomicstype, self._atomicshape,
                                 self.flavor)

    def iterrows(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.

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
                    shape = arr.shape
		    arr=Numeric.fromstring(arr._data, typecode)
		    arr.shape = shape
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
#             except cPicke.UnpicklingError:
#                 arr = []

        return arr

    def __getitem__(self, key):
        """Returns a vlarray row or slice.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key"is an integer, the corresponding row is returned. If
        "key" is a slice, the row slice determined by key is returned.

        """

        if type(key) in (int,long):
            if key >= self.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            return self.read(key)[0]
        elif isinstance(key, slice):
            return self.read(key.start, key.stop, key.step)
        else:
            raise IndexError, "Non-valid index or slice: %s" % \
                  key
        
    def __setitem__(self, keys, value):
        """Updates a vlarray row "keys" by setting it to "value".

        If "keys" is an integer, it refers to the number of row to be
        modified.

        If "keys" is a tuple, the first element refers to the row
        to be modified, and the second element to the range (so, it
        can be an integer or an slice) of the row that will be
        updated.

        Note: When updating VLStrings (codification UTF-8) or Objects,
        there is a problem: we can only update values with *exactly*
        the same bytes than in the original row. With UTF-8 encoding
        this is problematic because, for instance, 'c' takes 1 byte,
        but 'ç' takes at least two (!). Perhaps another codification
        does not have this problem, I don't know. With objects, the
        same happens, because cPickle applied on an instance (for
        example) does not guarantee to return the same number of bytes
        than over other instance, even of the same class than the
        former. This effectively limits the number of objects than can
        be updated in VLArrays, most specially VLStrings and Objects
        as has been said before.

        """

        assert not self._v_file.mode == "r", "Attempt to write over a file opened in read-only mode"

        if not isinstance(keys, tuple):
            keys = (keys, None)
        if len(keys) > 2:
            raise IndexError, "You cannot specify more than two dimensions"
        nrow, rng = keys
        # Process the first index
        if type(nrow) not in (int,long):
            raise IndexError, "The first dimension only can be an integer"
        if nrow >= self.nrows:
            raise IndexError, "First index out of range"
        if nrow < 0:
            # To support negative values
            nrow += self.nrows
        # Process the second index
        if type(rng) in (int,long):
            start = rng; stop = start+1; step = 1
        elif isinstance(rng, slice):
            start, stop, step = rng.start, rng.stop, rng.step
        elif rng is None:
            start, stop, step = None, None, None
        else:
            raise IndexError, "Non-valid second index or slice: %s" % rng
        
        object = value
        # Prepare the object to convert it into a numarray object
        if self.atom.flavor == "Object":
            # Special case for a generic object
            # (to be pickled and saved as an array of unsigned bytes)
            object = numarray.array(cPickle.dumps(object), type=numarray.UInt8)
        elif self.atom.flavor == "VLString":
            # Special case for a generic object
            # (to be pickled and saved as an array of unsigned bytes)
            if type(object) not in (str,unicode):
                raise TypeError, \
"""The object "%s" is not of type String or Unicode.""" % (str(object))
            try:
                object = object.encode('utf-8')
            except UnicodeError:
                (typerr, value, traceback) = sys.exc_info()
                raise ValueError, "Problems when converting the object '%s' to the encoding 'utf-8'. The error was: %s" % (object, value)
            object = numarray.array(object, type=numarray.UInt8)

        value = convertIntoNA(object, self.atom)
        nobjects = self._checkShape(value)

        #nobjects = len(value)

        # Get the previous value
        #naarr = self.read(nrow, nrow+1, 1, slice_specified=0)
        naarr = self._readArray(nrow, nrow+1, 1)[0]
        nobjects = len(naarr)
        if len(value) > nobjects:
            raise ValueError, \
"Length of value (%s) is larger than number of elements in row (%s)" % \
(len(value), nobjects)
        # Assign the value to it
        try:
            naarr[slice(start, stop, step)] = value
        except:  #XXX
            (typerr, value2, traceback) = sys.exc_info()
            raise ValueError, \
"Value parameter:\n'%r'\ncannot be converted into an array object compliant vlarray[%s] row: \n'%r'\nThe error was: <%s>" % \
        (value, keys, naarr[slice(start, stop, step)], value2)

        if naarr.size() > 0:
            self._modify(nrow, naarr, nobjects)

    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=1):
        """Read the array from disk and return it as a self.flavor object."""

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
        return outlistarr

    def _g_copyWithStats(self, group, name, start, stop, step, title, filters):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new VLArray object
        object = self._v_file.createVLArray(
            group, name, self.atom, title=title, filters=filters,
            expectedsizeinMB=self._v_expectedsizeinMB, _log = False)
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
