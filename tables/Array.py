########################################################################
#
#       License: BSD
#       Created: October 10, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Array.py,v $
#       $Id: Array.py,v 1.45 2003/12/16 12:59:03 falted Exp $
#
########################################################################

"""Here is defined the Array class.

See Array class docstring for more info.

Classes:

    Array

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.45 $"

# default version for ARRAY objects
#obversion = "1.0"    # initial version
obversion = "2.0"    # Added an optional EXTDIM attribute


import types, warnings, sys
from Leaf import Leaf
from utils import calcBufferSize, processRange
import hdf5Extension
import numarray
import numarray.strings as strings
import numarray.records as records

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

class Array(Leaf, hdf5Extension.Array, object):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from Numeric, numarray,
    lists, tuples, strings or scalars, or open existing ones.

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
        
      Specific of Array:
        read(start, stop, step)
        iterrows(start, stop, step)

    Instance variables:

      Common to all leaves:
        name -- the leaf node name
        hdf5name -- the HDF5 leaf node name
        title -- the leaf title
        shape -- the leaf shape
        byteorder -- the byteorder of the leaf
        
      Specific of Array:
      
        type -- The type class for the array.

        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        
        flavor -- The object type of this object (Numarray, Numeric, List,
            Tuple, String, Int of Float).
            
        nrows -- The value of the first dimension of Array.
            
        nrow -- On iterators, this is the index of the current row.

    """
    
    def __init__(self, object = None, title = ""):
        """Create the instance Array.

        Keyword arguments:

        object -- The (regular) object to be saved. It can be any of
            NumArray, CharArray, Numeric, List, Tuple, String, Int of
            Float types, provided that they are regular (i.e. they are
            not like [[1,2],2]).

        title -- Sets a TITLE attribute on the HDF5 array entity.

        """
        self.new_title = title
        self._v_compress = 0  # An Array can not be compressed
        self._v_complib = "zlib"  # Some default value
        self._v_shuffle = 0   # Values of an Array cannot be shuffled
        self.extdim = -1   # An Array object is not enlargeable
        # Check if we have to create a new object or read their contents
        # from disk
        if object is not None:
            self._v_new = 1
            self.object = object
        else:
            self._v_new = 0

    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        global obversion

        self._v_version = obversion
        naarr, self.flavor = self._convertIntoNA(self.object)
        if naarr.shape:
            self._v_expectednrows = naarr.shape[0]
        else:
            self._v_expectednrows = 1  # Scalar case
        if (isinstance(naarr, strings.CharArray)):
            self.byteorder = "non-relevant" 
        else:
            self.byteorder  = naarr._byteorder

        # Compute some values for buffering and I/O parameters
        # Compute the rowsize for each element
        self.rowsize = naarr.itemsize()
        for i in naarr.shape:
            if i>0:
                self.rowsize *= i
            else:
                raise ValueError, "An Array object cannot be empty."

        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
                            calcBufferSize(self.rowsize, self._v_expectednrows,
                                           self._v_compress)

        self.shape = naarr.shape
        if naarr.shape:
            self.nrows = naarr.shape[0]
        else:
            self.nrows = 1    # Scalar case
        self.itemsize = naarr.itemsize()
        self.type = self._createArray(naarr, self.new_title)

    def _convertIntoNA(self, object):
        "Convert a generic object into a numarray object"
        arr = object
        # Check for Numeric objects
        if isinstance(arr, numarray.NumArray):
            flavor = "NumArray"
            naarr = arr
            byteorder = arr._byteorder
        elif (Numeric_imported and type(arr) == type(Numeric.array(1))):
            flavor = "Numeric"
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

        elif (isinstance(arr, strings.CharArray)):
            flavor = "CharArray"
            naarr = arr
        elif (isinstance(arr, types.TupleType) or
              isinstance(arr, types.ListType)):
            # Test if can convert to numarray object
            try:
                naarr = numarray.array(arr)
            # If not, test with a chararray
            except TypeError:
                try:
                    naarr = strings.array(arr)
                # If still doesn't, issues an error
                except:
                    raise ValueError, \
"""The object '%s' can't be converted to a numerical or character array.
  Sorry, but this object is not supported.""" % (arr)
            if isinstance(arr, types.TupleType):
                flavor = "Tuple"
            else:
                flavor = "List"
        elif isinstance(arr, types.IntType):
            naarr = numarray.array(arr)
            flavor = "Int"
        elif isinstance(arr, types.FloatType):
            naarr = numarray.array(arr)

            flavor = "Float"
        elif isinstance(arr, types.StringType):
            naarr = strings.array(arr)
            flavor = "String"
        else:
            raise ValueError, \
"""The object '%s' is not in the list of supported objects (NumArray, CharArray, Numeric, homogeneous list or homogeneous tuple, int, float or str). Sorry, but this object is not supported.""" % (arr)

        # We always want a contiguous buffer
        # (no matter if has an offset or not; that will be corrected later)
        if (not naarr.iscontiguous()):
            # Do a copy of the array in case is not contiguous
            naarr = numarray.NDArray.copy(naarr)

        return naarr, flavor

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize, self.byteorder) = \
                        self._openArray()
        # Compute the rowsize for each element
        self.rowsize = self.itemsize
        for i in range(len(self.shape)):
            self.rowsize *= self.shape[i]
        # Assign a value to nrows in case we are a non-enlargeable object
        if self.shape:
            self.nrows = self.shape[0]
        else:
            self.nrows = 1   # Scalar case
        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
                   calcBufferSize(self.rowsize, self.nrows, self._v_compress)

    def iterrows(self, start=None, stop=None, step=None):
        """Iterator over all the rows or a range"""

        return self.__call__(start, stop, step)

    def __call__(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.
        
        It returns the same iterator than
        Table.iterrows(start, stop, step).
        It is, therefore, a shorter way to call it.
        """

        try:
            (self._start, self._stop, self._step) = \
                          processRange(self.nrows, start, stop, step)
        except IndexError:
            # If problems with indexes, silently return the null tuple
            return ()
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
            #print "start, stop, step:", self._start, self._stop, self._step
            # Read a chunk of rows
            if self._row+1 >= self._v_maxTuples or self._row < 0:
                self._stopb = self._startb+self._step*self._v_maxTuples
                # Protection for reading more elements than needed
                if self._stopb > self._stop:
                    self._stopb = self._stop
                self.listarr = self.read(self._startb, self._stopb, self._step)
                # Swap the axes to easy the return of elements
                if self.extdim > 0:
                    if self.flavor == "Numeric":
                        if Numeric_imported:
                            self.listarr = Numeric.swapaxes(self.listarr,
                                                            self.extdim, 0)
                        else:
                            # Warn the user
                            warnings.warn( \
"""The object on-disk has Numeric flavor, but Numeric is not installed locally. Returning a numarray object instead!.""")
                            # Default to numarray
                            self.listarr = swapaxes(self.listarr,
                                                    self.extdim, 0)
                    else:
                        self.listarr.swapaxes(self.extdim, 0)
                self._row = -1
                self._startb = self._stopb
            self._row += 1
            self.nrow += self._step
            self._nrowsread += self._step
            return self.listarr[self._row]

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
            (start, stop, step) = (key, key+1, 1)
            ret = self.read(start, stop, step)
        elif isinstance(key, types.SliceType):
            (start, stop, step) = processRange(self.nrows, key.start, key.stop, key.step)
            ret = self.read(start, stop, step)
        else:
            raise ValueError, "Non-valid index or slice: %s" % \
                  key

        if len(range(start, stop, step)) == 1 and self.extdim > 0:
            ret.swapaxes(self.extdim, 0)
            return ret[0]
        else:
            return ret
        
    def _convToFlavor(self, arr):
        "Convert the numarray parameter to the correct flavor"

        # Convert to Numeric, tuple or list if needed
        if self.flavor == "Numeric":
            if Numeric_imported:
                # This works for both numeric and chararrays
                # arr=Numeric.array(arr, typecode=arr.typecode())
                # The next is 10 times faster (for tolist(),
                # we should check for tostring()!)
                if repr(self.type) == "CharType":
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
"""The object on-disk has Numeric flavor, but Numeric is not installed locally. Returning a numarray object instead!.""")
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

        return arr
        
    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=None):
        """Read the array from disk and return it as a "flavor" object."""

        if self.extdim < 0:
            extdim = 0
        else:
            extdim = self.extdim

        (start, stop, step) = processRange(self.nrows, start, stop, step)
        rowstoread = ((stop - start - 1) / step) + 1
        shape = list(self.shape)
        if shape:
            shape[extdim] = rowstoread
            shape = tuple(shape)
        if repr(self.type) == "CharType":
            arr = strings.array(None, itemsize=self.itemsize,
                                  shape=shape)
        else:
            arr = numarray.array(buffer=None,
                                 type=self.type,
                                 shape=shape)
            # Set the same byteorder than on-disk
            arr._byteorder = self.byteorder
        # Protection against reading empty arrays
        zerodim = 0
        for i in range(len(shape)):
            if shape[i] == 0:
                zerodim = 1

        if not zerodim:
            # Arrays that have non-zero dimensionality
            self._readArray(start, stop, step, arr._data)
            
        if self.flavor <> "NumArray":
            return self._convToFlavor(arr)
        else:
            return arr
        
    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  flavor = %r
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize,
                       self.nrows, self.flavor, self.byteorder)
