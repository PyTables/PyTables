########################################################################
#
#       License: BSD
#       Created: October 10, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Here is defined the Array class.

See Array class docstring for more info.

Classes:

    Array
    ImageArray

Functions:


Misc variables:

    __version__


"""

import types, warnings, sys

import numpy

try:
    import Numeric
    Numeric_imported = True
except ImportError:
    Numeric_imported = False

try:
    import numarray
    import numarray.strings
    numarray_imported = True
except ImportError:
    numarray_imported = False

import tables.hdf5Extension as hdf5Extension
from tables.utils import calcBufferSize, processRange, processRangeRead, \
                         convToFlavor, is_idx
from tables.Leaf import Leaf, Filters


__version__ = "$Revision$"


# default version for ARRAY objects
#obversion = "1.0"    # initial version
#obversion = "2.0"    # Added an optional EXTDIM attribute
#obversion = "2.1"    # Added support for complex datatypes
#obversion = "2.2"    # This adds support for time datatypes.
obversion = "2.3"    # This adds support for enumerated datatypes.

# The byteorders in NumPy
byteorders = {'>': 'big',
              '<': 'little',
              '=': sys.byteorder,
              '|': 'non-relevant'}


class Array(hdf5Extension.Array, Leaf):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from NumPy, numarray,
    Numeric, lists, tuples, strings or scalars, or open existing ones.

    All NumPy, Numeric and numarray typecodes are supported.

    Methods:

        read(start, stop, step)
        iterrows(start, stop, step)

    Instance variables:

        flavor -- The object representation of this array.
        nrows -- The length of the first dimension of the array.
        nrow -- On iterators, this is the index of the current row.
        type -- The type class of the represented array.
        stype -- The string type of the represented array.
        itemsize -- The size of the base items.

    """

    # Class identifier.
    _c_classId = 'ARRAY'


    def _g_calcBufferSize(self, expectedfsizeinKb):
        """Compute and optimum buffer size.

        The logic to do that is based purely in experiments playing
        with different buffer sizes, chunksize and compression
        flag. It is obvious that using big buffers optimize the I/O
        speed. This might (should) be further optimized doing more
        experiments.

        This only is important for CArray and EArray classes, and it
        is keeped here just because it's an easy way to share it.

        """

        # Increasing the bufmultfactor would enable a good compression
        # ratio (up to an extend), but it would affect to reading
        # performance. Be careful when touching this
        # F. Altet 2004-11-10
        #bufmultfactor = int(1000 * 5) # Conservative value
        bufmultfactor = int(1000 * 10) # Medium value
        #bufmultfactor = int(1000 * 20)  # Agressive value
        #bufmultfactor = int(1000 * 50) # Very Aggresive value

        if expectedfsizeinKb <= 100:
            # Values for files less than 100 KB of size
            buffersize = 5 * bufmultfactor
        elif (expectedfsizeinKb > 100 and
            expectedfsizeinKb <= 1000):
            # Values for files less than 1 MB of size
            buffersize = 10 * bufmultfactor
        elif (expectedfsizeinKb > 1000 and
              expectedfsizeinKb <= 20 * 1000):
            # Values for sizes between 1 MB and 20 MB
            buffersize = 20  * bufmultfactor
        elif (expectedfsizeinKb > 20 * 1000 and
              expectedfsizeinKb <= 200 * 1000):
            # Values for sizes between 20 MB and 200 MB
            buffersize = 40 * bufmultfactor
        elif (expectedfsizeinKb > 200 * 1000 and
              expectedfsizeinKb <= 2000 * 1000):
            # Values for sizes between 200 MB and 2 GB
            buffersize = 50 * bufmultfactor
        else:  # Greater than 2 GB
            buffersize = 60 * bufmultfactor

        return buffersize


    def __init__(self, parentNode, name,
                 object=None, title="",
                 _log=True):
        """Create the instance Array.

        Keyword arguments:

        object -- The (regular) object to be saved. It can be any of
            numpy, numarray, numeric, list, tuple, string, integer of
            floating point types, provided that they are regular
            (i.e. they are not like [[1,2],2]).

        title -- Sets a TITLE attribute on the HDF5 array entity.
        """

        self._v_version = None
        """The object version of this array."""

        self._v_new = new = object is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""

        self._object = object
        """
        The object to be stored in the array.  It can be any of
        ``numpy``, ``numarray``, ``numeric``, list, tuple, string,
        integer of floating point types, provided that they are
        regular (i.e. they are not like ``[[1, 2], 2]``).
        """
        self._v_expectedrows = None
        """The expected number of rows to be stored in the array."""
        self.byteorder = None
        """
        The endianness of data in memory ('big', 'little' or
        'non-relevant').
        """
        self.rowsize = None
        """The size in bytes of each row in the array."""
        self._v_maxTuples = None
        """The maximum number of rows that are read on each chunk iterator."""
        self._v_chunksize = None
        """
        The HDF5 chunk size for ``CArray``, ``EArray`` and ``VLArray``
        objects.
        """
        self._v_convert = True
        """Whether the ``Array`` object must be converted or not."""
        self.shape = None
        """The shape of the stored array."""
        self._enum = None
        """The enumerated type containing the values in this array."""

        # Miscellaneous iteration rubbish.
        self._start = None
        """Starting row for the current iteration."""
        self._stop = None
        """Stopping row for the current iteration."""
        self._step = None
        """Step size for the current iteration."""
        self._nrowsread = None
        """Number of rows read up to the current state of iteration."""
        self._startb = None
        """Starting row for current buffer."""
        self._stopb = None
        """Stopping row for current buffer. """
        self._row = None
        """Current row in iterators (sentinel)."""
        self._init = False
        """Whether we are in the middle of an iteration or not (sentinel)."""
        self.listarr = None
        """Current buffer in iterators."""

        # Documented (*public*) attributes.
        self.flavor = None
        """The object representation of this array.  It can be any of
        'numpy', 'numarray', 'numeric' or 'python' values."""
        self.nrows = None
        """The length of the first dimension of the array."""
        self.nrow = None
        """On iterators, this is the index of the current row."""
        self.type = None
        """The type class of the represented array."""
        self.stype = None
        """The string type of the represented array."""
        self.itemsize = None
        """
        The size of the base items (specially useful for ``CharType``
        objects).
        """
        self.extdim = -1   # ordinary arrays are not enlargeable
        """The index of the enlargeable dimension."""

        # Ordinary arrays have no filters: leaf is created with default ones.
        super(Array, self).__init__(parentNode, name, new, Filters(), _log)


    def _g_create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""

        self._v_version = obversion
        try:
            nparr, self.flavor = self._convertIntoNP(self._object)
        except:  #XXX
            # Problems converting data. Close the node and re-raise exception.
            #print "Problems converting input object:", str(self._object)
            self.close(flush=0)
            raise

        # Decrease the number of references to the object
        self._object = None

        if nparr.shape:
            self._v_expectedrows = nparr.shape[0]
        else:
            self._v_expectedrows = 1  # Scalar case

        # Set the byteorder in PyTables format
        self.byteorder = byteorders[nparr.dtype.byteorder]

        # Compute some values for buffering and I/O parameters
        # Compute the rowsize for each element
        self.rowsize = nparr.itemsize
        for i in nparr.shape:
            if i>0:
                self.rowsize *= i
            else:
                raise ValueError, "An Array object cannot have zero-dimensions."

        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
                            calcBufferSize(self.rowsize, self._v_expectedrows)

        # If shape has not been assigned yet, assign it.
        if self.shape is None:
            self.shape = nparr.shape
        if self.shape != ():
            self.nrows = nparr.shape[0]
        else:
            self.nrows = 1    # Scalar case
        self.itemsize = nparr.itemsize
        try:
            (self._v_objectID, self.type, self.stype) = (
                self._createArray(nparr, self._v_new_title))
            return self._v_objectID
        except:  #XXX
            # Problems creating the Array on disk. Close node and re-raise.
            self.close(flush=0)
            raise


    def _g_open(self):
        """Get the metadata info for an array in file."""

        (self._v_objectID, self.type, self.stype, self.shape,
         self.itemsize, self.byteorder, self._v_chunksize) = self._openArray()

        # Get enumeration from disk.
        if self.stype == 'Enum':
            (self._enum, self.type) = self._loadEnum()
        # Compute the rowsize for each element
        self.rowsize = self.itemsize
        for i in xrange(len(self.shape)):
            self.rowsize *= self.shape[i]
        # Assign a value to nrows in case we are a non-enlargeable object
        if self.shape:
            self.nrows = self.shape[0]
        else:
            self.nrows = 1L   # Scalar case

        # Compute the maxTuples for the buffers in iterators
        (self._v_maxTuples, chunksize) = \
                            calcBufferSize(self.rowsize, self.nrows)

        return self._v_objectID


    def _convertIntoNP(self, arr):
        "Convert a generic object into a NumPy object"

        if type(arr) == numpy.ndarray and arr.dtype.type <> numpy.void:
            flavor = "numpy"
            # Do a copy of the array in case it is not contiguous
            nparr = numpy.asarray(arr)
        elif (numarray_imported and
              type(arr) in (numarray.NumArray, numarray.strings.CharArray)):
            flavor = "numarray"
            # Do a copy of the array in case it is not contiguous
            nparr = numpy.asarray(arr)
        elif Numeric_imported and type(arr) == Numeric.ArrayType:
            flavor = "numeric"
            nparr = numpy.asarray(arr)
        elif type(arr) in (tuple, list, int, float, complex, str):
            flavor = "python"
            # Test if this can be converted into a NumPy object
            try:
                nparr = numpy.array(arr)
            except:  #XXX
                raise TypeError, \
"""The object '%s' can't be converted into a numerical or character array.
Sorry, but this object is not supported.""" % (arr)
        else:
            raise TypeError, \
"""The object '%s' is not in the list of supported objects: numpy,
numarray, numeric, homogeneous list or tuple, int, float, complex or str.
Sorry, but this object is not supported.""" % (arr)

        return nparr, flavor


    def getEnum(self):
        """
        Get the enumerated type associated with this array.

        If this array is of an enumerated type, the corresponding `Enum`
        instance is returned.  If it is not of an enumerated type, a
        ``TypeError`` is raised.
        """

        if self.stype != 'Enum':
            raise TypeError("array ``%s`` is not of an enumerated type"
                            % self._v_pathname)

        return self._enum


    def iterrows(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.

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

        if not self._init:
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
        self._init = True  # Sentinel
        self.nrow = self._start - self._step    # row number

    def next(self):
        "next() method for __iter__() that is called on each iteration"
        if self._nrowsread >= self._stop:
            self._init = False
            raise StopIteration        # end of iteration
        else:
            # Read a chunk of rows
            if self._row+1 >= self._v_maxTuples or self._row < 0:
                self._stopb = self._startb+self._step*self._v_maxTuples
                # Protection for reading more elements than needed
                if self._stopb > self._stop:
                    self._stopb = self._stop
                self.listarr = self.read(self._startb, self._stopb, self._step)
                # Swap the axes to easy the return of elements
                if self.extdim > 0:
                    if self.flavor == "numarray":
                        if numarray_imported:
                            self.listarr = numarray.swapaxes(self.listarr,
                                                             self.extdim, 0)
                        else:
                            # Warn the user
                            warnings.warn( \
"""The object on-disk has numarray flavor, but numarray is not installed locally. Returning a NumPy object instead!.""")
                            # Default to NumPy
                            self.listarr = numpy.swapaxes(self.listarr,
                                                          self.extdim, 0)
                    elif self.flavor == "numeric":
                        if Numeric_imported:
                            self.listarr = Numeric.swapaxes(self.listarr,
                                                            self.extdim, 0)
                        else:
                            # Warn the user
                            warnings.warn( \
"""The object on-disk has numeric flavor, but Numeric is not installed locally. Returning a NumPy object instead!.""")
                            # Default to NumPy
                            self.listarr = numpy.swapaxes(self.listarr,
                                                          self.extdim, 0)
                    else:
                        self.listarr = numpy.swapaxes(self.listarr,
                                                      self.extdim, 0)
                self._row = -1
                self._startb = self._stopb
            self._row += 1
            self.nrow += self._step
            self._nrowsread += self._step
            # Fixes bug #968132
            #if self.listarr.shape:
            if self.shape:
                return self.listarr[self._row]
            else:
                return self.listarr    # Scalar case


    def _interpret_indexing(self, keys):
        """Internal routine used by __getitem__ and __setitem__"""

        maxlen = len(self.shape)
        shape = (maxlen,)
        startl = numpy.empty(shape=shape, dtype=numpy.int64)
        stopl = numpy.empty(shape=shape, dtype=numpy.int64)
        stepl = numpy.empty(shape=shape, dtype=numpy.int64)
        stop_None = numpy.zeros(shape=shape, dtype=numpy.int64)
        if not isinstance(keys, tuple):
            keys = (keys,)
        nkeys = len(keys)
        dim = 0
        # Here is some problem when dealing with [...,...] params
        # but this is a bit weird way to pass parameters anyway
        for key in keys:
            ellipsis = 0  # Sentinel
            if isinstance(key, types.EllipsisType):
                ellipsis = 1
                for diml in xrange(dim, len(self.shape) - (nkeys - dim) + 1):
                    startl[dim] = 0
                    stopl[dim] = self.shape[diml]
                    stepl[dim] = 1
                    dim += 1
            elif dim >= maxlen:
                raise IndexError, "Too many indices for object '%s'" % \
                      self._v_pathname
            elif is_idx(key):
                # Protection for index out of range
                if key >= self.shape[dim]:
                    raise IndexError, "Index out of range"
                if key < 0:
                    # To support negative values (Fixes bug #968149)
                    key += self.shape[dim]
                start, stop, step = processRange(self.shape[dim],
                                                 key, key+1, 1)
                stop_None[dim] = 1
            elif isinstance(key, slice):
                start, stop, step = processRange(self.shape[dim],
                                                 key.start, key.stop, key.step)
            else:
                raise TypeError, "Non-valid index or slice: %s" % \
                      key
            if not ellipsis:
                startl[dim] = start
                stopl[dim] = stop
                stepl[dim] = step
                dim += 1

        # Complete the other dimensions, if needed
        if dim < len(self.shape):
            for diml in xrange(dim, len(self.shape)):
                startl[dim] = 0
                stopl[dim] = self.shape[diml]
                stepl[dim] = 1
                dim += 1

        # Compute the shape for the container properly. Fixes #1288792
        shape = []
        for dim in xrange(len(self.shape)):
            # The negative division operates differently with python scalars
            # and numpy scalars (which are similar to C conventions). See:
            # http://www.python.org/doc/faq/programming.html#why-does-22-10-return-3
            # and
            # http://www.peterbe.com/Integer-division-in-programming-languages
            # for more info on this issue.
            # I've finally decided to rely on the len(xrange) function.
            # F. Altet 2006-09-25
            #new_dim = ((stopl[dim] - startl[dim] - 1) / stepl[dim]) + 1
            new_dim = len(xrange(startl[dim], stopl[dim], stepl[dim]))
            if not (new_dim == 1 and stop_None[dim]):
            #if not stop_None[dim]:
                # Append dimension
                shape.append(new_dim)

        return startl, stopl, stepl, shape


    def __getitem__(self, keys):
        """Returns an Array element, row or extended slice.

        It takes different actions depending on the type of the "keys"
        parameter:

        If "keys" is an integer, the corresponding row is returned. If
        "keys" is a slice, the row slice determined by key is returned.

        """

        startl, stopl, stepl, shape = self._interpret_indexing(keys)
        return self._readSlice(startl, stopl, stepl, shape)


    def __setitem__(self, keys, value):
        """Sets an Array element, row or extended slice.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key" is an integer, the corresponding row is assigned to
        value.

        If "key" is a slice, the row slice determined by it is
        assigned to "value". If needed, this "value" is broadcasted to
        fit in the desired range. If the slice to be updated exceeds
        the actual shape of the array, only the values in the existing
        range are updated, i.e. the index error will be silently
        ignored. If "value" is a multidimensional object, then its
        shape must be compatible with the slice specified in "key",
        otherwhise, a ValueError will be issued.

        """

        startl, stopl, stepl, shape = self._interpret_indexing(keys)
        countl = ((stopl - startl - 1) / stepl) + 1
        # Create an array compliant with the specified slice
        if self.stype == "CharType":
            narr = numpy.empty(dtype="S%s"%self.itemsize,
            # Here shape=shape should be enough, but it makes some
            # tests to fail. This should be analyzed more carefully.
            # F. Altet 2005-09-12
                                 shape=countl)
                                 #shape=shape)
        else:
            narr = numpy.empty(shape=shape, dtype=self.type)

        # Assign the value to it
        try:
            #narr[...] = value
            # The next is a workaround for allowing downcasting even with
            # objects that are not NumPy, but that follow the array protocol.
            # For more info, see:
            # http://projects.scipy.org/scipy/numpy/ticket/299
            if self.stype == "CharType":
                narr[...] = value
            else:
                narr[...] = numpy.asarray(value)
        except Exception, exc:  #XXX
            raise ValueError, \
"""value parameter '%s' cannot be converted into an array object compliant with %s:
'%r'
The error was: <%s>""" % (value, self.__class__.__name__, self, exc)

        if narr.size:
            self._modify(startl, stepl, countl, narr)


    # Accessor for the _readArray method in superclass
    def _readSlice(self, startl, stopl, stepl, shape):
        if self.stype == "CharType":
            arr = numpy.empty(dtype="S%s"%self.itemsize, shape=shape)
        else:
            arr = numpy.empty(dtype=self.type, shape=shape)

        # Protection against reading empty arrays
        if 0 not in shape:
            # Arrays that have non-zero dimensionality
            self._g_readSlice(startl, stopl, stepl, arr)

        if not self._v_convert:
            return arr

        if self.flavor == 'numpy':
            if arr.shape == ():  # Scalar case
                return arr[()]  # Return a numpy scalar
            else:             # No conversion needed
                return arr
        # Fixes #968131
        elif arr.shape == ():  # Scalar case
            return arr.item()  # return the python value
        else:
            return convToFlavor(self, arr)


    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=None):
        """Read the array from disk and return it as a self.flavor object."""

        if self.extdim < 0:
            extdim = 0
        else:
            extdim = self.extdim

        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        #rowstoread = ((stop - start - 1) / step) + 1
        rowstoread = len(xrange(start, stop, step))
        shape = list(self.shape)
        if shape:
            shape[extdim] = rowstoread
            shape = tuple(shape)
        if self.stype == "CharType":
            arr = numpy.empty(dtype="S%s"%self.itemsize, shape=shape)
        else:
            arr = numpy.empty(dtype=self.type, shape=shape)
            # Set the correct byteorder for this array
            arr.dtype = arr.dtype.newbyteorder(self.byteorder)

        # Protection against reading empty arrays
        if 0 not in shape:
            # Arrays that have non-zero dimensionality
            self._readArray(start, stop, step, arr)

        if self.flavor == "numpy":
            # No conversion needed
            return arr
        # Fixes #968131
        elif arr.shape == ():  # Scalar case
            return arr.item()  # return the python value
        else:
            return convToFlavor(self, arr)


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf"
        # Get the slice of the array
        # (non-buffered version)
        if self.shape:
            arr = self[start:stop:step]
        else:
            arr = self[()]
        # Build the new Array object
        object = Array(group, name, arr, title=title, _log=_log)
        nbytes = self.itemsize
        for i in self.shape:
            nbytes*=i

        return (object, nbytes)


    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  type = %r
  stype = %r
  shape = %s
  itemsize = %s
  nrows = %s
  flavor = %r
  byteorder = %r""" % (self, self.type, self.stype, self.shape, self.itemsize,
                       self.nrows, self.flavor, self.byteorder)



class ImageArray(Array):

    """
    Array containing an image.

    This class has no additional behaviour or functionality compared
    to that of an ordinary array.  It simply enables the user to open
    an ``IMAGE`` HDF5 node as a normal `Array` node in PyTables.
    """

    # Class identifier.
    _c_classId = 'IMAGE'
