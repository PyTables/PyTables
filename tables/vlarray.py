########################################################################
#
#       License: BSD
#       Created: November 12, 2003
#       Author:  Francesc Alted - faltet@pytables.com
#
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

import numpy

from tables import hdf5Extension
from tables.utilsExtension import lrange
from tables.utils import convertToNPAtom, convertToNPAtom2, idx2long, \
     correct_byteorder, SizeType, is_idx, lazyattr


from tables.atom import (
    ObjectAtom, VLStringAtom, VLUnicodeAtom, EnumAtom, Atom, split_type )
from tables.flavor import internal_to_flavor
from tables.leaf import Leaf, calc_chunksize


__version__ = "$Revision$"

# default version for VLARRAY objects
#obversion = "1.0"    # initial version
#obversion = "1.0"    # add support for complex datatypes
#obversion = "1.1"    # This adds support for time datatypes.
#obversion = "1.2"    # This adds support for enumerated datatypes.
obversion = "1.3"     # Introduced 'PSEUDOATOM' attribute.

class VLArray(hdf5Extension.VLArray, Leaf):
    """
    This class represents variable length (ragged) arrays in an HDF5 file.

    Instances of this class represent array objects in the object tree
    with the property that their rows can have a *variable* number of
    homogeneous elements, called *atoms*.  Like `Table` datasets,
    variable length arrays can have only one dimension, and the
    elements (atoms) of their rows can be fully multidimensional.
    `VLArray` objects do also support compression.

    When reading a range of rows from a `VLArray`, you will *always*
    get a Python list of objects of the current flavor (each of them
    for a row), which may have different lengths.

    This class provides methods to write or read data to or from
    variable length array objects in the file.  Note that it also
    inherits all the public attributes and methods that `Leaf` already
    provides.

    Public instance variables
    -------------------------

    atom
        An `Atom` instance representing the *type* and *shape* of the
        atomic objects to be saved.  You may use a *pseudo-atom* for
        storing a serialized object or variable length string per row.
    flavor
        The type of data object read from this leaf.

        Please note that when reading several rows of `VLArray` data,
        the flavor only applies to the *components* of the returned
        Python list, not to the list itself.
    nrow
        On iterators, this is the index of the current row.
    extdim
        The index of the enlargeable dimension (always 0 for vlarrays).

    Public methods
    --------------

    append(sequence)
        Add a ``sequence`` of data to the end of the dataset.
    getEnum()
        Get the enumerated type associated with this array.
    iterrows([start][, stop][, step])
        Iterate over the rows of the array.
    next()
        Get the next element of the array during an iteration.
    read([start][, stop][, step])
        Get data in the array as a list of objects of the current
        flavor.

    Special methods
    ---------------

    The following methods automatically trigger actions when a
    `VLArray` instance is accessed in a special way
    (e.g. ``vlarray[2:5]`` will be equivalent to a call to
    ``vlarray.__getitem__(slice(2, 5, None))``).

    __getitem__(key)
        Get a row or a range of rows from the array.
    __iter__()
        Iterate over the rows of the array.
    __setitem__(key, value)
        Set a row in the array.

    Example of use
    --------------

    See below a small example of the use of the `VLArray` class.  The
    code is available in ``examples/vlarray1.py``::

        import tables
        from numpy import *

        # Create a VLArray:
        fileh = tables.openFile('vlarray1.h5', mode='w')
        vlarray = fileh.createVLArray(fileh.root, 'vlarray1',
                                      tables.Int32Atom(shape=()),
                                      \"ragged array of ints\",
                                      filters=tables.Filters(1))
        # Append some (variable length) rows:
        vlarray.append(array([5, 6]))
        vlarray.append(array([5, 6, 7]))
        vlarray.append([5, 6, 9, 8])

        # Now, read it through an iterator:
        print '-->', vlarray.title
        for x in vlarray:
            print '%s[%d]--> %s' % (vlarray.name, vlarray.nrow, x)

        # Now, do the same with native Python strings.
        vlarray2 = fileh.createVLArray(fileh.root, 'vlarray2',
                                      tables.StringAtom(itemsize=2),
                                      \"ragged array of strings\",
                                      filters=tables.Filters(1))
        vlarray2.flavor = 'python'
        # Append some (variable length) rows:
        print '-->', vlarray2.title
        vlarray2.append(['5', '66'])
        vlarray2.append(['5', '6', '77'])
        vlarray2.append(['5', '6', '9', '88'])

        # Now, read it through an iterator:
        for x in vlarray2:
            print '%s[%d]--> %s' % (vlarray2.name, vlarray2.nrow, x)

        # Close the file.
        fileh.close()

    The output for the previous script is something like::

        --> ragged array of ints
        vlarray1[0]--> [5 6]
        vlarray1[1]--> [5 6 7]
        vlarray1[2]--> [5 6 9 8]
        --> ragged array of strings
        vlarray2[0]--> ['5', '66']
        vlarray2[1]--> ['5', '6', '77']
        vlarray2[2]--> ['5', '6', '9', '88']
    """

    # Class identifier.
    _c_classId = 'VLARRAY'


    # Lazy read-only attributes
    # `````````````````````````
    @lazyattr
    def dtype(self):
        """The NumPy ``dtype`` that most closely matches this array."""
        return self.atom.dtype

    # Properties
    # ~~~~~~~~~~
    shape = property(
        lambda self: (self.nrows,), None, None,
        "The shape of the stored array.")


    # Other methods
    # ~~~~~~~~~~~~~
    def __init__( self, parentNode, name,
                  atom=None, title="",
                  filters=None, expectedsizeinMB=1.0,
                  chunkshape=None, byteorder=None,
                  _log=True ):
        """
        Create a `VLArray` instance.

        `atom`
            An `Atom` instance representing the *type* and *shape* of
            the atomic objects to be saved.

        `title`
            A description for this node (it sets the ``TITLE`` HDF5
            attribute on disk).

        `filters`
            An instance of the `Filters` class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        `expectedsizeinMB`
            An user estimate about the size (in MB) in the final
            `VLArray` object.  If not provided, the default value is 1
            MB.  If you plan to create either a much smaller or a much
            bigger `VLArray` try providing a guess; this will optimize
            the HDF5 B-Tree creation and management process time and
            the amount of memory used.

        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The dimensionality of `chunkshape` must
            be 1.  If ``None``, a sensible value is calculated (which
            is recommended).

        `byteorder`
            The byteorder of the data *on disk*, specified as 'little'
            or 'big'.  If this is not specified, the byteorder is that
            of the platform.
        """

        self._v_version = None
        """The object version of this array."""
        self._v_new = new = atom is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_new_filters = filters
        """New filter properties for this array."""
        self._v_expectedsizeinMB = expectedsizeinMB
        """The expected size of the array in MiB."""
        self._v_chunkshape = None
        """Private storage for the `chunkshape` property of Leaf."""

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
        self.atom = atom
        """
        An `Atom` instance representing the shape and type of the
        atomic objects to be saved.
        """
        self.nrow = None
        """On iterators, this is the index of the current row."""
        self.nrows = None
        """The total number of rows."""
        self.extdim = 0   # VLArray only have one dimension currently
        """The index of the enlargeable dimension (always 0 for vlarrays)."""

        # Check the chunkshape parameter
        if new and chunkshape is not None:
            if isinstance(chunkshape, (int, numpy.integer, long)):
                chunkshape = (chunkshape,)
            try:
                chunkshape = tuple(chunkshape)
            except TypeError:
                raise TypeError(
                    "`chunkshape` parameter must be an integer or sequence "
                    "and you passed a %s" % type(chunkshape) )
            if len(chunkshape) != 1:
                raise ValueError( "`chunkshape` rank (length) must be 1: %r"
                                  % (chunkshape,) )
            self._v_chunkshape = tuple(SizeType(s) for s in chunkshape)

        super(VLArray, self).__init__(parentNode, name, new, filters,
                                      byteorder, _log)

    def _g_postInitHook(self):
        super(VLArray, self)._g_postInitHook()
        self.nrowsinbuf = 100  # maybe enough for most applications


    # This is too specific for moving it into Leaf
    def _calc_chunkshape(self, expectedsizeinMB):
        """Calculate the size for the HDF5 chunk."""

        chunksize = calc_chunksize(expectedsizeinMB)

        # For computing the chunkshape for HDF5 VL types, we have to
        # choose the itemsize of the *each* element of the atom and
        # not the size of the entire atom.  I don't know why this
        # should be like this, perhaps I should report this to the
        # HDF5 list.
        # F. Alted 2006-11-23
        #elemsize = self.atom.atomsize()
        elemsize = self._basesize
        # Set the chunkshape
        chunkshape = chunksize // elemsize
        # Safeguard against itemsizes being extremely large
        if chunkshape == 0:
            chunkshape = 1
        return (SizeType(chunkshape),)


    def _g_create(self):
        """Create a variable length array (ragged array)."""

        atom = self.atom
        self._v_version = obversion
        # Check for zero dims in atom shape (not allowed in VLArrays)
        zerodims = numpy.sum(numpy.array(atom.shape) == 0)
        if zerodims > 0:
            raise ValueError, \
"""When creating VLArrays, none of the dimensions of the Atom instance can
be zero."""

        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            self._atomicdtype = atom.base.dtype
            self._atomicsize = atom.base.size
            self._basesize = atom.base.itemsize
        else:
            self._atomicdtype = atom.dtype
            self._atomicsize = atom.size
            self._basesize = atom.itemsize
        self._atomictype = atom.type
        self._atomicshape = atom.shape

        # Compute the optimal chunkshape, if needed
        if self._v_chunkshape is None:
            self._v_chunkshape = self._calc_chunkshape(
                self._v_expectedsizeinMB)
        self.nrows = SizeType(0)     # No rows at creation time

        # Correct the byteorder if needed
        if self.byteorder is None:
            self.byteorder = correct_byteorder(atom.type, sys.byteorder)

        # After creating the vlarray, ``self._v_objectID`` needs to be
        # set because it is needed for setting attributes afterwards.
        self._v_objectID = self._createArray(self._v_new_title)

        # Add an attribute in case we have a pseudo-atom so that we
        # can retrieve the proper class after a re-opening operation.
        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            self.attrs.PSEUDOATOM = atom.kind

        return self._v_objectID


    def _g_open(self):
        """Get the metadata info for an array in file."""

        self._v_objectID, self.nrows, self._v_chunkshape, atom = \
                          self._openArray()

        # Check if the atom can be a PseudoAtom
        if "PSEUDOATOM" in self.attrs:
            kind = self.attrs.PSEUDOATOM
            if kind == 'vlstring':
                atom = VLStringAtom()
            elif kind == 'vlunicode':
                atom = VLUnicodeAtom()
            elif kind == 'object':
                atom = ObjectAtom()
            else:
                raise ValueError(
                    "pseudo-atom name ``%s`` not known." % kind)
        elif self._v_file.format_version[:1] == "1":
            flavor1x = self.attrs.FLAVOR
            if flavor1x == "VLString":
                atom = VLStringAtom()
            elif flavor1x == "Object":
                atom = ObjectAtom()

        self.atom = atom
        return self._v_objectID


    def _getnobjects(self, nparr):
        "Return the number of objects in a NumPy array."

        # Check for zero dimensionality array
        zerodims = numpy.sum(numpy.array(nparr.shape) == 0)
        if zerodims > 0:
            # No objects to be added
            return 0
        shape = nparr.shape
        atom_shape = self.atom.shape
        shapelen = len(nparr.shape)
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
(nparr, shape, atom_shape)
        return nobjects


    def getEnum(self):
        """
        Get the enumerated type associated with this array.

        If this array is of an enumerated type, the corresponding `Enum`
        instance is returned.  If it is not of an enumerated type, a
        ``TypeError`` is raised.
        """

        if self.atom.kind != 'enum':
            raise TypeError("array ``%s`` is not of an enumerated type"
                            % self._v_pathname)

        return self.atom.enum


    def append(self, sequence):
        """
        Add a `sequence` of data to the end of the dataset.

        This method appends the objects in the `sequence` to a *single
        row* in this array.  The type and shape of individual objects
        must be compliant with the atoms in the array.  In the case of
        serialized objects and variable length strings, the object or
        string to append is itself the `sequence`.
        """

        self._v_file._checkWritable()

        # Prepare the sequence to convert it into a NumPy object
        atom = self.atom
        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            sequence = atom.toarray(sequence)
            statom = atom.base
        else:
            try:  # fastest check in most cases
                len(sequence)
            except TypeError:
                raise TypeError("argument is not a sequence")
            statom = atom

        if len(sequence) > 0:
            # The sequence needs to be copied to make the operation safe
            # to in-place conversion.
            nparr = convertToNPAtom2(sequence, statom)
            nobjects = self._getnobjects(nparr)
        else:
            nobjects = 0
            nparr = None

        self._append(nparr, nobjects)
        self.nrows += 1


    def iterrows(self, start=None, stop=None, step=None):
        """
        Iterate over the rows of the array.

        This method returns an iterator yielding an object of the
        current flavor for each selected row in the array.

        If a range is not supplied, *all the rows* in the array are
        iterated upon --you can also use the `VLArray.__iter__()`
        special method for that purpose.  If you only want to iterate
        over a given *range of rows* in the array, you may use the
        `start`, `stop` and `step` parameters, which have the same
        meaning as in `VLArray.read()`.

        Example of use::

            for row in vlarray.iterrows(step=4):
                print '%s[%d]--> %s' % (vlarray.name, vlarray.nrow, row)
        """

        (self._start, self._stop, self._step) = \
                     self._processRangeRead(start, stop, step)
        self._initLoop()
        return self


    def __iter__(self):
        """
        Iterate over the rows of the array.

        This is equivalent to calling `VLArray.iterrows()` with default
        arguments, i.e. it iterates over *all the rows* in the array.

        Example of use::

            result = [row for row in vlarray]

        Which is equivalent to::

            result = [row for row in vlarray.iterrows()]
        """

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
        self.nrow = SizeType(self._start - self._step)    # row number


    def next(self):
        """
        Get the next element of the array during an iteration.

        The element is returned as a list of objects of the current
        flavor.
        """
        if self._nrowsread >= self._stop:
            self._init = False
            raise StopIteration        # end of iteration
        else:
            # Read a chunk of rows
            if self._row+1 >= self.nrowsinbuf or self._row < 0:
                self._stopb = self._startb+self._step*self.nrowsinbuf
                self.listarr = self.read(self._startb, self._stopb, self._step)
                self._row = -1
                self._startb = self._stopb
            self._row += 1
            self.nrow += self._step
            self._nrowsread += self._step
            return self.listarr[self._row]


    def __getitem__(self, key):
        """
        Get a row or a range of rows from the array.

        If `key` argument is an integer, the corresponding array row
        is returned as an object of the current flavor.  If `key` is a
        slice, the range of rows determined by it is returned as a
        list of objects of the current flavor.

        In addition, NumPy-style point selections are supported.  In
        particular, if `key` is a list of row coordinates, the set of
        rows determined by it is returned.  Furthermore, if `key` is an
        array of boolean values, only the coordinates where `key` is
        ``True`` are returned.  Note that for the latter to work it is
        necessary that `key` list would contain exactly as many rows as
        the array has.

        Example of use::

            a_row = vlarray[4]
            a_list = vlarray[4:1000:2]
            a_list2 = vlarray[[0,2]]   # get list of coords
            a_list3 = vlarray[[0,-2]]  # negative values accepted
            a_list4 = vlarray[numpy.array([True,...,False])]  # array of bools
        """

        if is_idx(key):
            # Index out of range protection
            if key >= self.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            (start, stop, step) = self._processRange(key, key+1, 1)
            return self.read(start, stop, step)[0]
        elif isinstance(key, slice):
            start, stop, step = self._processRange(
                key.start, key.stop, key.step)
            return self.read(start, stop, step)
        # Try with a boolean or point selection
        elif type(key) in (list, tuple) or isinstance(key, numpy.ndarray):
            coords = self._pointSelection(key)
            return self._readCoordinates(coords)
        else:
            raise IndexError("Invalid index or slice: %r" % (key,))


    def _assign_values(self, coords, values):
        """Assign the `values` to the positions stated in `coords`."""

        for nrow, value in zip(coords, values):
            if nrow >= self.nrows:
                raise IndexError, "First index out of range"
            if nrow < 0:
                # To support negative values
                nrow += self.nrows
            object_ = value
            # Prepare the object to convert it into a NumPy object
            atom = self.atom
            if not hasattr(atom, 'size'):  # it is a pseudo-atom
                object_ = atom.toarray(object_)
                statom = atom.base
            else:
                statom = atom
            value = convertToNPAtom(object_, statom)
            nobjects = self._getnobjects(value)

            # Get the previous value
            nrow = idx2long(nrow)   # To convert any possible numpy scalar value
            nparr = self._readArray(nrow, nrow+1, 1)[0]
            nobjects = len(nparr)
            if len(value) > nobjects:
                raise ValueError, \
    "Length of value (%s) is larger than number of elements in row (%s)" % \
    (len(value), nobjects)
            try:
                nparr[:] = value
            except Exception, exc:  #XXX
                raise ValueError, \
    "Value parameter:\n'%r'\ncannot be converted into an array object compliant vlarray[%s] row: \n'%r'\nThe error was: <%s>" % \
    (value, nrow, nparr[:], exc)

            if nparr.size > 0:
                self._modify(nrow, nparr, nobjects)


    def __setitem__(self, key, value):
        """
        Set a row, or set of rows, in the array.

        It takes different actions depending on the type of the `key`
        parameter: if it is an integer, the corresponding table row is
        set to `value` (a record, list or tuple capable of being
        converted to the table field format).  If `key` is a slice, the
        row slice determined by it is set to `value` (a NumPy record
        array, ``NestedRecArray`` or list of rows).

        In addition, NumPy-style point selections are supported.  In
        particular, if `key` is a list of row coordinates, the set of
        rows determined by it is set to `value`.  Furthermore, if `key`
        is an array of boolean values, only the coordinates where `key`
        is ``True`` are set to values from `value`.  Note that for the
        latter to work it is necessary that `key` list would contain
        exactly as many rows as the table has.

        .. Note:: When updating the rows of a `VLArray` object which
           uses a pseudo-atom, there is a problem: you can only update
           values with *exactly* the same size in bytes than the
           original row.  This is very difficult to meet with
           ``object`` pseudo-atoms, because ``cPickle`` applied on a
           Python object does not guarantee to return the same number
           of bytes than over another object, even if they are of the
           same class.  This effectively limits the kinds of objects
           than can be updated in variable-length arrays.

        Example of use::

            vlarray[0] = vlarray[0] * 2 + 3
            vlarray[99] = arange(96) * 2 + 3
            # Negative values for the index are supported.
            vlarray[-99] = vlarray[5] * 2 + 3
            vlarray[1:30:2] = list_of_rows
            vlarray[[1,3]] = new_1_and_3_rows
        """

        self._v_file._checkWritable()

        if is_idx(key):
            # If key is not a sequence, convert to it
            coords = [key]
            value = [value]
        elif isinstance(key, slice):
            (start, stop, step) = self._processRange(
                key.start, key.stop, key.step )
            coords = range(start, stop, step)
        # Try with a boolean or point selection
        elif type(key) in (list, tuple) or isinstance(key, numpy.ndarray):
            coords = self._pointSelection(key)
        else:
            raise IndexError("Invalid index or slice: %r" % (key,))

        # Do the assignment row by row
        self._assign_values(coords, value)


    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=1):
        """
        Get data in the array as a list of objects of the current
        flavor.

        Please note that, as the lengths of the different rows are
        variable, the returned value is a *Python list* (not an array
        of the current flavor), with as many entries as specified rows
        in the range parameters.

        The `start`, `stop` and `step` parameters can be used to
        select only a *range of rows* in the array.  Their meanings
        are the same as in the built-in ``range()`` Python function,
        except that negative values of `step` are not allowed yet.
        Moreover, if only `start` is specified, then `stop` will be
        set to ``start+1``.  If you do not specify neither `start` nor
        `stop`, then *all the rows* in the array are selected.
        """

        start, stop, step = self._processRangeRead(start, stop, step)
        if start == stop:
            listarr = []
        else:
            listarr = self._readArray(start, stop, step)

        atom = self.atom
        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            outlistarr = [atom.fromarray(arr) for arr in listarr]
        else:
            # Convert the list to the right flavor
            flavor = self.flavor
            outlistarr = [internal_to_flavor(arr, flavor) for arr in listarr]
        return outlistarr


    def _readCoordinates(self, coords):
        """Read rows specified in `coords`."""
        rows = []
        for coord in coords:
            rows.append(self.read(coord)[0])
        return rows


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, chunkshape, _log, **kwargs):
        "Private part of Leaf.copy() for each kind of leaf"

        # Build the new VLArray object
        object = VLArray(
            group, name, self.atom, title=title, filters=filters,
            expectedsizeinMB=self._v_expectedsizeinMB, chunkshape=chunkshape,
            _log=_log)
        # Now, fill the new vlarray with values from the old one
        # This is not buffered because we cannot forsee the length
        # of each record. So, the safest would be a copy row by row.
        # In the future, some analysis can be done in order to buffer
        # the copy process.
        nrowsinbuf = 1
        (start, stop, step) = self._processRangeRead(start, stop, step)
        # Optimized version (no conversions, no type and shape checks, etc...)
        nrowscopied = SizeType(0)
        nbytes = 0
        if not hasattr(self.atom, 'size'):  # it is a pseudo-atom
            atomsize = self.atom.base.size
        else:
            atomsize = self.atom.size
        for start2 in lrange(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            nparr = self._readArray(start=start2, stop=stop2, step=step)[0]
            nobjects = nparr.shape[0]
            object._append(nparr, nobjects)
            nbytes += nobjects*atomsize
            nrowscopied +=1
        object.nrows = nrowscopied
        return (object, nbytes)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom = %r
  byteorder = %r
  nrows = %s
  flavor = %r""" % (self, self.atom, self.byteorder, self.nrows,
                    self.flavor)
