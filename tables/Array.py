########################################################################
#
#       License: BSD
#       Created: October 10, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Array.py,v $
#       $Id: Array.py,v 1.55 2004/01/27 20:28:34 falted Exp $
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

__version__ = "$Revision: 1.55 $"

# default version for ARRAY objects
#obversion = "1.0"    # initial version
obversion = "2.0"    # Added an optional EXTDIM attribute


import types, warnings, sys
from Leaf import Leaf, Filters
from utils import calcBufferSize, processRange, processRangeRead
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

        read(start, stop, step)
        iterrows(start, stop, step)

    Instance variables:

        type -- The type class for the array.
        itemsize -- The size of the atomic items. Specially useful for
            CharArray objects.
        flavor -- The object type of this object ("NumArray", "CharArray",
            "Numeric", "List", "Tuple", "String", "Int" or "Float").
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
        # Assign some filter values by default
        self.filters = Filters()
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
                                           self.filters.complevel)

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
        if isinstance(arr, numarray.NumArray):
            flavor = "NumArray"
            naarr = arr
            self.byteorder  = naarr._byteorder
        elif isinstance(arr, strings.CharArray):
            flavor = "CharArray"
            naarr = arr
            self.byteorder = "non-relevant" 
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
                   calcBufferSize(self.rowsize, self.nrows,
                                  self.filters.complevel)

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
                          processRangeRead(self.nrows, start, stop, step)
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
            if self.listarr.shape:
                return self.listarr[self._row]
            else:
                return self.listarr    # Scalar case

    def __getitem__(self, keys):
        """Returns an Array row or slice.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key"is an integer, the corresponding row is returned. If
        "key" is a slice, the row slice determined by key is returned.

        """

        if self.shape == ():
            # Scalar case
            raise IndexError, "You cannot read scalar Arrays through indexing. Try using the read() method better."

        maxlen = len(self.shape)
        shape = (maxlen,)
        startl = numarray.array(None, shape=shape, type=numarray.Int64)
        stopl = numarray.array(None, shape=shape, type=numarray.Int64)
        stepl = numarray.array(None, shape=shape, type=numarray.Int64)
        stop_None = numarray.zeros(shape=shape, type=numarray.Int64)
        if not isinstance(keys, types.TupleType):
            keys = (keys,)
        nkeys = len(keys)
        dim = 0
        for key in keys:
            ellipsis = 0  # Sentinel
            if dim >= maxlen:
                raise IndexError, "Too many indices for object '%s'" % \
                      self._v_pathname
            if isinstance(key, types.EllipsisType):
                ellipsis = 1
                for diml in range(dim, len(self.shape) - (nkeys - dim) + 1):
                    startl[dim] = 0
                    stopl[dim] = self.shape[diml]
                    stepl[dim] = 1
                    dim += 1
            elif isinstance(key, types.IntType):
                start, stop, step = processRange(self.shape[dim],
                                                 key, key+1, 1)
                stop_None[dim] = 1
            elif isinstance(key, types.SliceType):
                start, stop, step = processRange(self.shape[dim],
                                                 key.start, key.stop, key.step)
            else:
                raise ValueError, "Non-valid index or slice: %s" % \
                      key
            if not ellipsis:
                startl[dim] = start
                stopl[dim] = stop
                stepl[dim] = step
                dim += 1
            
        # Complete the other dimensions, if needed
        if dim < len(self.shape):
            for diml in range(dim, len(self.shape)):
                startl[dim] = 0
                stopl[dim] = self.shape[diml]
                stepl[dim] = 1
                dim += 1

        return self._readSlice(startl, stopl, stepl, stop_None)

    # Accessor for the _readArray method in superclass
    def _readSlice(self, startl, stopl, stepl, stop_None):

        if self.extdim < 0:
            extdim = 0
        else:
            extdim = self.extdim

        shape = []
        for dim in range(len(self.shape)):
            new_dim = ((stopl[dim] - startl[dim] - 1) / stepl[dim]) + 1
            if not (new_dim == 1 and stop_None[dim]):
                # Append dimension
                shape.append(new_dim)
            
        if repr(self.type) == "CharType":
            arr = strings.array(None, itemsize=self.itemsize, shape=shape)
        else:
            arr = numarray.zeros(type=self.type, shape=shape)
            # Set the same byteorder than on-disk
            arr._byteorder = self.byteorder
        # Protection against reading empty arrays
        zerodim = 0
        for i in range(len(shape)):
            if shape[i] == 0:
                zerodim = 1

        if not zerodim:
            # Arrays that have non-zero dimensionality
            self._g_readSlice(startl, stopl, stepl, arr._data)
            pass
            
        if self.flavor <> "NumArray":
            return self._convToFlavor(arr)
        else:
            return arr
        
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
        """Read the array from disk and return it as a self.flavor object."""

        if self.extdim < 0:
            extdim = 0
        else:
            extdim = self.extdim

        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        rowstoread = ((stop - start - 1) / step) + 1
        shape = list(self.shape)
        if shape:
            shape[extdim] = rowstoread
            shape = tuple(shape)
        if repr(self.type) == "CharType":
            arr = strings.array(None, itemsize=self.itemsize, shape=shape)
        else:
            arr = numarray.array(buffer=None, type=self.type, shape=shape)
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
            
        if self.flavor in ["NumArray", "CharArray"]:
            return arr  # No further conversions needed
        else:
            return self._convToFlavor(arr)
        
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
