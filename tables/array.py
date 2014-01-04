# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: October 10, 2002
# Author: Francesc Alted - faltet@pytables.com
#
# $Id$
#
########################################################################

"""Here is defined the Array class."""

import sys

import numpy

from tables import hdf5extension
from tables.filters import Filters
from tables.flavor import flavor_of, array_as_internal, internal_to_flavor

from tables.utils import (is_idx, convert_to_np_atom2, SizeType, lazyattr,
                          byteorders, quantize)
from tables.leaf import Leaf

from tables._past import previous_api, previous_api_property

# default version for ARRAY objects
# obversion = "1.0"    # initial version
# obversion = "2.0"    # Added an optional EXTDIM attribute
# obversion = "2.1"    # Added support for complex datatypes
# obversion = "2.2"    # This adds support for time datatypes.
# obversion = "2.3"    # This adds support for enumerated datatypes.
obversion = "2.4"    # Numeric and numarray flavors are gone.


class Array(hdf5extension.Array, Leaf):
    """This class represents homogeneous datasets in an HDF5 file.

    This class provides methods to write or read data to or from array objects
    in the file. This class does not allow you neither to enlarge nor compress
    the datasets on disk; use the EArray class (see :ref:`EArrayClassDescr`) if
    you want enlargeable dataset support or compression features, or CArray
    (see :ref:`CArrayClassDescr`) if you just want compression.

    An interesting property of the Array class is that it remembers the
    *flavor* of the object that has been saved so that if you saved, for
    example, a list, you will get a list during readings afterwards; if you
    saved a NumPy array, you will get a NumPy object, and so forth.

    Note that this class inherits all the public attributes and methods that
    Leaf (see :ref:`LeafClassDescr`) already provides. However, as Array
    instances have no internal I/O buffers, it is not necessary to use the
    flush() method they inherit from Leaf in order to save their internal state
    to disk.  When a writing method call returns, all the data is already on
    disk.

    Parameters
    ----------
    parentnode
        The parent :class:`Group` object.

        .. versionchanged:: 3.0
           Renamed from *parentNode* to *parentnode*

    name : str
        The name of this node in its parent group.
    obj
        The array or scalar to be saved.  Accepted types are NumPy
        arrays and scalars as well as native Python sequences and
        scalars, provided that values are regular (i.e. they are not
        like ``[[1,2],2]``) and homogeneous (i.e. all the elements are
        of the same type).

        .. versionchanged:: 3.0
           Renamed form *object* into *obj*.
    title
        A description for this node (it sets the ``TITLE`` HDF5 attribute on
        disk).
    byteorder
        The byteorder of the data *on disk*, specified as 'little' or 'big'.
        If this is not specified, the byteorder is that of the given `object`.

    """

    # Class identifier.
    _c_classid = 'ARRAY'

    _c_classId = previous_api_property('_c_classid')
    _v_objectId = previous_api_property('_v_objectid')

    # Lazy read-only attributes
    # `````````````````````````
    @lazyattr
    def dtype(self):
        """The NumPy ``dtype`` that most closely matches this array."""

        return self.atom.dtype

    # Properties
    # ~~~~~~~~~~
    def _getnrows(self):
        if self.shape == ():
            return SizeType(1)  # scalar case
        else:
            return self.shape[self.maindim]
    nrows = property(
        _getnrows, None, None,
        "The number of rows in the array.")

    def _getrowsize(self):
        maindim = self.maindim
        rowsize = self.atom.size
        for i, dim in enumerate(self.shape):
            if i != maindim:
                rowsize *= dim
        return rowsize
    rowsize = property(
        _getrowsize, None, None,
        "The size of the rows in bytes in dimensions orthogonal to *maindim*.")

    size_in_memory = property(
        lambda self: self.nrows * self.rowsize, None, None,
        """The size of this array's data in bytes when it is fully loaded into
        memory.""")

    # Other methods
    # ~~~~~~~~~~~~~
    def __init__(self, parentnode, name,
                 obj=None, title="",
                 byteorder=None, _log=True, _atom=None):

        self._v_version = None
        """The object version of this array."""
        self._v_new = new = obj is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._obj = obj
        """The object to be stored in the array.  It can be any of numpy,
        list, tuple, string, integer of floating point types, provided
        that they are regular (i.e. they are not like ``[[1, 2], 2]``).

        .. versionchanged:: 3.0
           Renamed form *_object* into *_obj*.

        """

        self._v_convert = True
        """Whether the ``Array`` object must be converted or not."""

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
        self.atom = _atom
        """An Atom (see :ref:`AtomClassDescr`) instance representing the *type*
        and *shape* of the atomic objects to be saved.
        """
        self.shape = None
        """The shape of the stored array."""
        self.nrow = None
        """On iterators, this is the index of the current row."""
        self.extdim = -1   # ordinary arrays are not enlargeable
        """The index of the enlargeable dimension."""

        # Ordinary arrays have no filters: leaf is created with default ones.
        super(Array, self).__init__(parentnode, name, new, Filters(),
                                    byteorder, _log)

    def _g_create(self):
        """Save a new array in file."""

        self._v_version = obversion
        try:
            # `Leaf._g_post_init_hook()` should be setting the flavor on disk.
            self._flavor = flavor = flavor_of(self._obj)
            nparr = array_as_internal(self._obj, flavor)
        except:  # XXX
            # Problems converting data. Close the node and re-raise exception.
            self.close(flush=0)
            raise

        # Raise an error in case of unsupported object
        if nparr.dtype.kind in ['V', 'U', 'O']:  # in void, unicode, object
            raise TypeError("Array objects cannot currently deal with void, "
                            "unicode or object arrays")

        # Decrease the number of references to the object
        self._obj = None

        # Fix the byteorder of data
        nparr = self._g_fix_byteorder_data(nparr, nparr.dtype.byteorder)

        # Create the array on-disk
        try:
            # ``self._v_objectid`` needs to be set because would be
            # needed for setting attributes in some descendants later
            # on
            (self._v_objectid, self.shape, self.atom) = self._create_array(
                nparr, self._v_new_title, self.atom)
        except:  # XXX
            # Problems creating the Array on disk. Close node and re-raise.
            self.close(flush=0)
            raise

        # Compute the optimal buffer size
        self.nrowsinbuf = self._calc_nrowsinbuf()
        # Arrays don't have chunkshapes (so, set it to None)
        self._v_chunkshape = None

        return self._v_objectid

    def _g_open(self):
        """Get the metadata info for an array in file."""

        (oid, self.atom, self.shape, self._v_chunkshape) = self._open_array()

        self.nrowsinbuf = self._calc_nrowsinbuf()

        return oid

    def get_enum(self):
        """Get the enumerated type associated with this array.

        If this array is of an enumerated type, the corresponding Enum instance
        (see :ref:`EnumClassDescr`) is returned. If it is not of an enumerated
        type, a TypeError is raised.

        """

        if self.atom.kind != 'enum':
            raise TypeError("array ``%s`` is not of an enumerated type"
                            % self._v_pathname)

        return self.atom.enum

    getEnum = previous_api(get_enum)

    def iterrows(self, start=None, stop=None, step=None):
        """Iterate over the rows of the array.

        This method returns an iterator yielding an object of the current
        flavor for each selected row in the array.  The returned rows are taken
        from the *main dimension*.

        If a range is not supplied, *all the rows* in the array are iterated
        upon - you can also use the :meth:`Array.__iter__` special method for
        that purpose.  If you only want to iterate over a given *range of rows*
        in the array, you may use the start, stop and step parameters.

        Examples
        --------

        ::

            result = [row for row in arrayInstance.iterrows(step=4)]

        .. versionchanged:: 3.0
           If the *start* parameter is provided and *stop* is None then the
           array is iterated from *start* to the last line.
           In PyTables < 3.0 only one element was returned.

        """

        try:
            (self._start, self._stop, self._step) = self._process_range(
                start, stop, step)
        except IndexError:
            # If problems with indexes, silently return the null tuple
            return ()
        self._init_loop()
        return self

    def __iter__(self):
        """Iterate over the rows of the array.

        This is equivalent to calling :meth:`Array.iterrows` with default
        arguments, i.e. it iterates over *all the rows* in the array.

        Examples
        --------

        ::

            result = [row[2] for row in array]

        Which is equivalent to::

            result = [row[2] for row in array.iterrows()]

        """

        if not self._init:
            # If the iterator is called directly, assign default variables
            self._start = 0
            self._stop = self.nrows
            self._step = 1
            # and initialize the loop
            self._init_loop()
        return self

    def _init_loop(self):
        """Initialization for the __iter__ iterator"""

        self._nrowsread = self._start
        self._startb = self._start
        self._row = -1   # Sentinel
        self._init = True  # Sentinel
        self.nrow = SizeType(self._start - self._step)    # row number

    _initLoop = previous_api(_init_loop)

    def next(self):
        """Get the next element of the array during an iteration.

        The element is returned as an object of the current flavor.

        """

        # this could probably be sped up for long iterations by reusing the
        # listarr buffer
        if self._nrowsread >= self._stop:
            self._init = False
            self.listarr = None        # fixes issue #308
            raise StopIteration        # end of iteration
        else:
            # Read a chunk of rows
            if self._row + 1 >= self.nrowsinbuf or self._row < 0:
                self._stopb = self._startb + self._step * self.nrowsinbuf
                # Protection for reading more elements than needed
                if self._stopb > self._stop:
                    self._stopb = self._stop
                listarr = self._read(self._startb, self._stopb, self._step)
                # Swap the axes to easy the return of elements
                if self.extdim > 0:
                    listarr = listarr.swapaxes(self.extdim, 0)
                self.listarr = internal_to_flavor(listarr, self.flavor)
                self._row = -1
                self._startb = self._stopb
            self._row += 1
            self.nrow += self._step
            self._nrowsread += self._step
            # Fixes bug #968132
            # if self.listarr.shape:
            if self.shape:
                return self.listarr[self._row]
            else:
                return self.listarr    # Scalar case

    def _interpret_indexing(self, keys):
        """Internal routine used by __getitem__ and __setitem__"""

        maxlen = len(self.shape)
        shape = (maxlen,)
        startl = numpy.empty(shape=shape, dtype=SizeType)
        stopl = numpy.empty(shape=shape, dtype=SizeType)
        stepl = numpy.empty(shape=shape, dtype=SizeType)
        stop_None = numpy.zeros(shape=shape, dtype=SizeType)
        if not isinstance(keys, tuple):
            keys = (keys,)
        nkeys = len(keys)
        dim = 0
        # Here is some problem when dealing with [...,...] params
        # but this is a bit weird way to pass parameters anyway
        for key in keys:
            ellipsis = 0  # Sentinel
            if isinstance(key, type(Ellipsis)):
                ellipsis = 1
                for diml in xrange(dim, len(self.shape) - (nkeys - dim) + 1):
                    startl[dim] = 0
                    stopl[dim] = self.shape[diml]
                    stepl[dim] = 1
                    dim += 1
            elif dim >= maxlen:
                raise IndexError("Too many indices for object '%s'" %
                                 self._v_pathname)
            elif is_idx(key):
                # Protection for index out of range
                if key >= self.shape[dim]:
                    raise IndexError("Index out of range")
                if key < 0:
                    # To support negative values (Fixes bug #968149)
                    key += self.shape[dim]
                start, stop, step = self._process_range(
                    key, key + 1, 1, dim=dim)
                stop_None[dim] = 1
            elif isinstance(key, slice):
                start, stop, step = self._process_range(
                    key.start, key.stop, key.step, dim=dim)
            else:
                raise TypeError("Non-valid index or slice: %s" % key)
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
            # F. Alted 2006-09-25
            # Switch to `lrange` to allow long ranges (see #99).
            # use xrange, since it supports large integers as of Python 2.6
            # see github #181
            new_dim = len(xrange(startl[dim], stopl[dim], stepl[dim]))
            if not (new_dim == 1 and stop_None[dim]):
                shape.append(new_dim)

        return startl, stopl, stepl, shape

    def _fancy_selection(self, args):
        """Performs a NumPy-style fancy selection in `self`.

        Implements advanced NumPy-style selection operations in
        addition to the standard slice-and-int behavior.

        Indexing arguments may be ints, slices or lists of indices.

        Note: This is a backport from the h5py project.

        """

        # Internal functions

        def validate_number(num, length):
            """Validate a list member for the given axis length"""

            try:
                num = long(num)
            except TypeError:
                raise TypeError("Illegal index: %r" % num)
            if num > length - 1:
                raise IndexError("Index out of bounds: %d" % num)

        def expand_ellipsis(args, rank):
            """Expand ellipsis objects and fill in missing axes."""

            n_el = sum(1 for arg in args if arg is Ellipsis)
            if n_el > 1:
                raise IndexError("Only one ellipsis may be used.")
            elif n_el == 0 and len(args) != rank:
                args = args + (Ellipsis,)

            final_args = []
            n_args = len(args)
            for idx, arg in enumerate(args):
                if arg is Ellipsis:
                    final_args.extend((slice(None),) * (rank - n_args + 1))
                else:
                    final_args.append(arg)

            if len(final_args) > rank:
                raise IndexError("Too many indices.")

            return final_args

        def translate_slice(exp, length):
            """Given a slice object, return a 3-tuple (start, count, step)

            This is for for use with the hyperslab selection routines.

            """

            start, stop, step = exp.start, exp.stop, exp.step
            if start is None:
                start = 0
            else:
                start = long(start)
            if stop is None:
                stop = length
            else:
                stop = long(stop)
            if step is None:
                step = 1
            else:
                step = long(step)

            if step < 1:
                raise IndexError("Step must be >= 1 (got %d)" % step)
            if stop == start:
                raise IndexError("Zero-length selections are not allowed")
            if stop < start:
                raise IndexError("Reverse-order selections are not allowed")
            if start < 0:
                start = length + start
            if stop < 0:
                stop = length + stop

            if not 0 <= start <= (length - 1):
                raise IndexError(
                    "Start index %s out of range (0-%d)" % (start, length - 1))
            if not 1 <= stop <= length:
                raise IndexError(
                    "Stop index %s out of range (1-%d)" % (stop, length))

            count = (stop - start) // step
            if (stop - start) % step != 0:
                count += 1

            if start + count > length:
                raise IndexError(
                    "Selection out of bounds (%d; axis has %d)" %
                    (start + count, length))

            return start, count, step

        # Main code for _fancy_selection
        mshape = []
        selection = []

        if not isinstance(args, tuple):
            args = (args,)

        args = expand_ellipsis(args, len(self.shape))

        list_seen = False
        reorder = None
        for idx, (exp, length) in enumerate(zip(args, self.shape)):
            if isinstance(exp, slice):
                start, count, step = translate_slice(exp, length)
                selection.append((start, count, step, idx, "AND"))
                mshape.append(count)
            else:
                try:
                    exp = list(exp)
                except TypeError:
                    exp = [exp]  # Handle scalar index as a list of length 1
                    mshape.append(0)  # Keep track of scalar index for NumPy
                else:
                    mshape.append(len(exp))
                if len(exp) == 0:
                    raise IndexError(
                        "Empty selections are not allowed (axis %d)" % idx)
                elif len(exp) > 1:
                    if list_seen:
                        raise IndexError(
                            "Only one selection list is allowed")
                    else:
                        list_seen = True
                nexp = numpy.asarray(exp, dtype="i8")
                # Convert negative values
                nexp = numpy.where(nexp < 0, length + nexp, nexp)
                # Check whether the list is ordered or not
                # (only one unordered list is allowed)
                if not len(nexp) == len(numpy.unique(nexp)):
                    raise IndexError(
                        "Selection lists cannot have repeated values")
                neworder = nexp.argsort()
                if not numpy.alltrue(neworder == numpy.arange(len(exp))):
                    if reorder is not None:
                        raise IndexError(
                            "Only one selection list can be unordered")
                    corrected_idx = sum(1 for x in mshape if x != 0) - 1
                    reorder = (corrected_idx, neworder)
                    nexp = nexp[neworder]
                for select_idx in xrange(len(nexp) + 1):
                    # This crazy piece of code performs a list selection
                    # using HDF5 hyperslabs.
                    # For each index, perform a "NOTB" selection on every
                    # portion of *this axis* which falls *outside* the list
                    # selection.  For this to work, the input array MUST be
                    # monotonically increasing.
                    if select_idx < len(nexp):
                        validate_number(nexp[select_idx], length)
                    if select_idx == 0:
                        start = 0
                        count = nexp[0]
                    elif select_idx == len(nexp):
                        start = nexp[-1] + 1
                        count = length - start
                    else:
                        start = nexp[select_idx - 1] + 1
                        count = nexp[select_idx] - start
                    if count > 0:
                        selection.append((start, count, 1, idx, "NOTB"))

        mshape = tuple(x for x in mshape if x != 0)
        return selection, reorder, mshape

    _fancySelection = previous_api(_fancy_selection)

    def __getitem__(self, key):
        """Get a row, a range of rows or a slice from the array.

        The set of tokens allowed for the key is the same as that for extended
        slicing in Python (including the Ellipsis or ... token).  The result is
        an object of the current flavor; its shape depends on the kind of slice
        used as key and the shape of the array itself.

        Furthermore, NumPy-style fancy indexing, where a list of indices in a
        certain axis is specified, is also supported.  Note that only one list
        per selection is supported right now.  Finally, NumPy-style point and
        boolean selections are supported as well.

        Examples
        --------

        ::

            array1 = array[4]                       # simple selection
            array2 = array[4:1000:2]                # slice selection
            array3 = array[1, ..., ::2, 1:4, 4:]    # general slice selection
            array4 = array[1, [1,5,10], ..., -1]    # fancy selection
            array5 = array[np.where(array[:] > 4)]  # point selection
            array6 = array[array[:] > 4]            # boolean selection

        """

        self._g_check_open()

        try:
            # First, try with a regular selection
            startl, stopl, stepl, shape = self._interpret_indexing(key)
            arr = self._read_slice(startl, stopl, stepl, shape)
        except TypeError:
            # Then, try with a point-wise selection
            try:
                coords = self._point_selection(key)
                arr = self._read_coords(coords)
            except TypeError:
                # Finally, try with a fancy selection
                selection, reorder, shape = self._fancy_selection(key)
                arr = self._read_selection(selection, reorder, shape)

        if self.flavor == "numpy" or not self._v_convert:
            return arr

        return internal_to_flavor(arr, self.flavor)

    def __setitem__(self, key, value):
        """Set a row, a range of rows or a slice in the array.

        It takes different actions depending on the type of the key parameter:
        if it is an integer, the corresponding array row is set to value (the
        value is broadcast when needed).  If key is a slice, the row slice
        determined by it is set to value (as usual, if the slice to be updated
        exceeds the actual shape of the array, only the values in the existing
        range are updated).

        If value is a multidimensional object, then its shape must be
        compatible with the shape determined by key, otherwise, a ValueError
        will be raised.

        Furthermore, NumPy-style fancy indexing, where a list of indices in a
        certain axis is specified, is also supported.  Note that only one list
        per selection is supported right now.  Finally, NumPy-style point and
        boolean selections are supported as well.

        Examples
        --------

        ::

            a1[0] = 333        # assign an integer to a Integer Array row
            a2[0] = 'b'        # assign a string to a string Array row
            a3[1:4] = 5        # broadcast 5 to slice 1:4
            a4[1:4:2] = 'xXx'  # broadcast 'xXx' to slice 1:4:2

            # General slice update (a5.shape = (4,3,2,8,5,10).
            a5[1, ..., ::2, 1:4, 4:] = numpy.arange(1728, shape=(4,3,2,4,3,6))
            a6[1, [1,5,10], ..., -1] = arr    # fancy selection
            a7[np.where(a6[:] > 4)] = 4       # point selection + broadcast
            a8[arr > 4] = arr2                # boolean selection

        """

        self._g_check_open()

        # Create an array compliant with the specified slice
        nparr = convert_to_np_atom2(value, self.atom)
        if nparr.size == 0:
            return

        # truncate data if least_significant_digit filter is set
        # TODO: add the least_significant_digit attribute to the array on disk
        if (self.filters.least_significant_digit is not None and not numpy.issubdtype(nparr.dtype, int)):
            nparr = quantize(nparr, self.filters.least_significant_digit)

        try:
            startl, stopl, stepl, shape = self._interpret_indexing(key)
            self._write_slice(startl, stopl, stepl, shape, nparr)
        except TypeError:
            # Then, try with a point-wise selection
            try:
                coords = self._point_selection(key)
                self._write_coords(coords, nparr)
            except TypeError:
                selection, reorder, shape = self._fancy_selection(key)
                self._write_selection(selection, reorder, shape, nparr)

    def _check_shape(self, nparr, slice_shape):
        """Test that nparr shape is consistent with underlying object.

        If not, try creating a new nparr object, using broadcasting if
        necessary.

        """

        if nparr.shape != (slice_shape + self.atom.dtype.shape):
            # Create an array compliant with the specified shape
            narr = numpy.empty(shape=slice_shape, dtype=self.atom.dtype)

            # Assign the value to it. It will raise a ValueError exception
            # if the objects cannot be broadcast to a single shape.
            narr[...] = nparr
            return narr
        else:
            return nparr

    _checkShape = previous_api(_check_shape)

    def _read_slice(self, startl, stopl, stepl, shape):
        """Read a slice based on `startl`, `stopl` and `stepl`."""

        nparr = numpy.empty(dtype=self.atom.dtype, shape=shape)
        # Protection against reading empty arrays
        if 0 not in shape:
            # Arrays that have non-zero dimensionality
            self._g_read_slice(startl, stopl, stepl, nparr)
        # For zero-shaped arrays, return the scalar
        if nparr.shape == ():
            nparr = nparr[()]
        return nparr

    _readSlice = previous_api(_read_slice)

    def _read_coords(self, coords):
        """Read a set of points defined by `coords`."""

        nparr = numpy.empty(dtype=self.atom.dtype, shape=len(coords))
        if len(coords) > 0:
            self._g_read_coords(coords, nparr)
        # For zero-shaped arrays, return the scalar
        if nparr.shape == ():
            nparr = nparr[()]
        return nparr

    _readCoords = previous_api(_read_coords)

    def _read_selection(self, selection, reorder, shape):
        """Read a `selection`.  Reorder if necessary."""

        # Create the container for the slice
        nparr = numpy.empty(dtype=self.atom.dtype, shape=shape)
        # Arrays that have non-zero dimensionality
        self._g_read_selection(selection, nparr)
        # For zero-shaped arrays, return the scalar
        if nparr.shape == ():
            nparr = nparr[()]
        elif reorder is not None:
            # We need to reorder the array
            idx, neworder = reorder
            k = [slice(None)] * len(shape)
            k[idx] = neworder.argsort()
            # Apparently, a copy is not needed here, but doing it
            # for symmetry with the `_write_selection()` method.
            nparr = nparr[k].copy()
        return nparr

    _readSelection = previous_api(_read_selection)

    def _write_slice(self, startl, stopl, stepl, shape, nparr):
        """Write `nparr` in a slice based on `startl`, `stopl` and `stepl`."""

        nparr = self._check_shape(nparr, tuple(shape))
        countl = ((stopl - startl - 1) // stepl) + 1
        self._g_write_slice(startl, stepl, countl, nparr)

    _writeSlice = previous_api(_write_slice)

    def _write_coords(self, coords, nparr):
        """Write `nparr` values in points defined by `coords` coordinates."""

        if len(coords) > 0:
            nparr = self._check_shape(nparr, (len(coords),))
            self._g_write_coords(coords, nparr)

    _writeCoords = previous_api(_write_coords)

    def _write_selection(self, selection, reorder, shape, nparr):
        """Write `nparr` in `selection`.  Reorder if necessary."""

        nparr = self._check_shape(nparr, tuple(shape))
        # Check whether we should reorder the array
        if reorder is not None:
            idx, neworder = reorder
            k = [slice(None)] * len(shape)
            k[idx] = neworder
            # For a reason a don't understand well, we need a copy of
            # the reordered array
            nparr = nparr[k].copy()
        self._g_write_selection(selection, nparr)

    _writeSelection = previous_api(_write_selection)

    def _read(self, start, stop, step, out=None):
        """Read the array from disk without slice or flavor processing."""

        nrowstoread = len(xrange(start, stop, step))
        shape = list(self.shape)
        if shape:
            shape[self.maindim] = nrowstoread
        if out is None:
            arr = numpy.empty(dtype=self.atom.dtype, shape=shape)
        else:
            bytes_required = self.rowsize * nrowstoread
            # if buffer is too small, it will segfault
            if bytes_required != out.nbytes:
                raise ValueError(('output array size invalid, got {0} bytes, '
                                  'need {1} bytes').format(out.nbytes,
                                                           bytes_required))
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError('output array not C contiguous')
            arr = out
        # Protection against reading empty arrays
        if 0 not in shape:
            # Arrays that have non-zero dimensionality
            self._read_array(start, stop, step, arr)
        # data is always read in the system byteorder
        # if the out array's byteorder is different, do a byteswap
        if (out is not None and
                byteorders[arr.dtype.byteorder] != sys.byteorder):
            arr.byteswap(True)
        return arr

    def read(self, start=None, stop=None, step=None, out=None):
        """Get data in the array as an object of the current flavor.

        The start, stop and step parameters can be used to select only a
        *range of rows* in the array.  Their meanings are the same as in
        the built-in range() Python function, except that negative values
        of step are not allowed yet. Moreover, if only start is specified,
        then stop will be set to start + 1. If you do not specify neither
        start nor stop, then *all the rows* in the array are selected.

        The out parameter may be used to specify a NumPy array to receive
        the output data.  Note that the array must have the same size as
        the data selected with the other parameters.  Note that the array's
        datatype is not checked and no type casting is performed, so if it
        does not match the datatype on disk, the output will not be correct.
        Also, this parameter is only valid when the array's flavor is set
        to 'numpy'.  Otherwise, a TypeError will be raised.

        When data is read from disk in NumPy format, the output will be
        in the current system's byteorder, regardless of how it is stored
        on disk.
        The exception is when an output buffer is supplied, in which case
        the output will be in the byteorder of that output buffer.

        .. versionchanged:: 3.0
           Added the *out* parameter.

        """

        self._g_check_open()
        if out is not None and self.flavor != 'numpy':
            msg = ("Optional 'out' argument may only be supplied if array "
                   "flavor is 'numpy', currently is {0}").format(self.flavor)
            raise TypeError(msg)
        (start, stop, step) = self._process_range_read(start, stop, step)
        arr = self._read(start, stop, step, out)
        return internal_to_flavor(arr, self.flavor)

    def _g_copy_with_stats(self, group, name, start, stop, step,
                           title, filters, chunkshape, _log, **kwargs):
        """Private part of Leaf.copy() for each kind of leaf"""

        # Compute the correct indices.
        (start, stop, step) = self._process_range_read(start, stop, step)
        # Get the slice of the array
        # (non-buffered version)
        if self.shape:
            arr = self[start:stop:step]
        else:
            arr = self[()]
        # Build the new Array object.  Use the _atom reserved keyword
        # just in case the array is being copied from a native HDF5
        # with atomic types different from scalars.
        # For details, see #275 of trac.
        object_ = Array(group, name, arr, title=title, _log=_log,
                        _atom=self.atom)
        nbytes = numpy.prod(self.shape, dtype=SizeType) * self.atom.size

        return (object_, nbytes)

    _g_copyWithStats = previous_api(_g_copy_with_stats)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom := %r
  maindim := %r
  flavor := %r
  byteorder := %r
  chunkshape := %r""" % (self, self.atom, self.maindim,
                         self.flavor, self.byteorder,
                         self.chunkshape)


class ImageArray(Array):
    """Array containing an image.

    This class has no additional behaviour or functionality compared
    to that of an ordinary array.  It simply enables the user to open
    an ``IMAGE`` HDF5 node as a normal `Array` node in PyTables.

    """

    # Class identifier.
    _c_classid = 'IMAGE'

    _c_classId = previous_api_property('_c_classid')
