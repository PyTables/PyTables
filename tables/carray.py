# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: June 15, 2005
# Author: Antonio Valentino
# Modified by: Francesc Alted
#
# $Id$
#
########################################################################

"""Here is defined the CArray class."""

import sys

import numpy

from tables.atom import Atom
from tables.flavor import flavor_of, array_as_internal, internal_to_flavor
from tables.array import Array
from tables.utils import correct_byteorder, SizeType
from tables.utilsextension import atom_to_hdf5_type

from tables._past import previous_api, previous_api_property

# default version for CARRAY objects
# obversion = "1.0"    # Support for time & enumerated datatypes.
obversion = "1.1"    # Numeric and numarray flavors are gone.


class CArray(Array):
    """This class represents homogeneous datasets in an HDF5 file.

    The difference between a CArray and a normal Array (see
    :ref:`ArrayClassDescr`), from which it inherits, is that a CArray
    has a chunked layout and, as a consequence, it supports compression.
    You can use datasets of this class to easily save or load arrays to
    or from disk, with compression support included.

    CArray includes all the instance variables and methods of Array.
    Only those with different behavior are mentioned here.

    Parameters
    ----------
    parentnode
        The parent :class:`Group` object.

        .. versionchanged:: 3.0

            Renamed from *parentNode* to *parentnode*

    name : str
        The name of this node in its parent group.
    atom
       An `Atom` instance representing the *type* and *shape* of
       the atomic objects to be saved.

    shape
       The shape of the new array.

    title
       A description for this node (it sets the ``TITLE`` HDF5
       attribute on disk).

    filters
       An instance of the `Filters` class that provides
       information about the desired I/O filters to be applied
       during the life of this object.

    chunkshape
       The shape of the data chunk to be read or written in a
       single HDF5 I/O operation.  Filters are applied to those
       chunks of data.  The dimensionality of `chunkshape` must
       be the same as that of `shape`.  If ``None``, a sensible
       value is calculated (which is recommended).

    byteorder
        The byteorder of the data *on disk*, specified as 'little'
        or 'big'.  If this is not specified, the byteorder is that
        of the platform.

    Examples
    --------

    See below a small example of the use of the `CArray` class.
    The code is available in ``examples/carray1.py``::

        import numpy
        import tables

        fileName = 'carray1.h5'
        shape = (200, 300)
        atom = tables.UInt8Atom()
        filters = tables.Filters(complevel=5, complib='zlib')

        h5f = tables.open_file(fileName, 'w')
        ca = h5f.create_carray(h5f.root, 'carray', atom, shape,
                               filters=filters)

        # Fill a hyperslab in ``ca``.
        ca[10:60, 20:70] = numpy.ones((50, 50))
        h5f.close()

        # Re-open a read another hyperslab
        h5f = tables.open_file(fileName)
        print h5f
        print h5f.root.carray[8:12, 18:22]
        h5f.close()

    The output for the previous script is something like::

        carray1.h5 (File) ''
        Last modif.: 'Thu Apr 12 10:15:38 2007'
        Object Tree:
        / (RootGroup) ''
        /carray (CArray(200, 300), shuffle, zlib(5)) ''

        [[0 0 0 0]
         [0 0 0 0]
         [0 0 1 1]
         [0 0 1 1]]

    """

    # Class identifier.
    _c_classid = 'CARRAY'

    _c_classId = previous_api_property('_c_classid')

    # Properties
    # ~~~~~~~~~~
    # Special methods
    # ~~~~~~~~~~~~~~~
    def __init__(self, parentnode, name, obj=None,
                 atom=None, shape=None,
                 title="", filters=None,
                 chunkshape=None, byteorder=None,
                 _log=True):
        self._obj = obj
        """The object to be stored in the array.  It can be any of numpy,
        list, tuple, string, integer of floating point types, provided
        that they are regular (i.e. they are not like ``[[1, 2], 2]``).
        """
        self.atom = atom
        """An `Atom` instance representing the shape, type of the atomic
        objects to be saved.
        """
        self.shape = shape
        """The shape of the stored array."""
        self.extdim = -1  # `CArray` objects are not enlargeable by default
        """The index of the enlargeable dimension."""

        # Other private attributes
        self._v_version = None
        """The object version of this array."""
        self._v_new = new = (atom is not None) or (obj is not None)
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_convert = True
        """Whether the ``Array`` object must be converted or not."""
        self._v_chunkshape = chunkshape
        """Private storage for the `chunkshape` property of the leaf."""

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

        if new and obj is None:
            if not isinstance(atom, Atom):
                raise ValueError("atom parameter should be an instance of "
                                 "tables.Atom and you passed a %s." % type(atom))
            if shape is None:
                raise ValueError("you must specify a non-empty shape")
            try:
                shape = tuple(shape)
            except TypeError:
                raise TypeError("`shape` parameter must be a sequence "
                                "and you passed a %s" % type(shape))
            self.shape = tuple(SizeType(s) for s in shape)

        if new and chunkshape is not None:
            try:
                chunkshape = tuple(chunkshape)
            except TypeError:
                raise TypeError("`chunkshape` parameter must be a sequence "
                                "and you passed a %s" % type(chunkshape))
            if shape is not None and len(shape) != len(chunkshape):
                raise ValueError("the shape (%s) and chunkshape (%s) "
                                 "ranks must be equal." % (shape, chunkshape))
            elif min(chunkshape) < 1:
                raise ValueError("chunkshape parameter cannot have "
                                 "zero-dimensions.")
            self._v_chunkshape = tuple(SizeType(s) for s in chunkshape)

        # The `Array` class is not abstract enough! :(
        super(Array, self).__init__(parentnode, name, new, filters,
                                    byteorder, _log)

    def _g_create(self):
        """Create a new array in file (specific part)."""
        # Finish the common part of creation process
        if self.shape is None:
            return self._g_create_common(None)
        else:
            if min(self.shape) < 1:
                raise ValueError("shape parameter cannot have zero-dimensions.")
            return self._g_create_common(self.nrows)

    def _g_create_common(self, expectedrows):
        """Create a new array in file (common part)."""
        self._v_version = obversion
        if self._obj is not None:
            try:
                # `Leaf._g_post_init_hook()` should be setting the flavor on disk.
                self._flavor = flavor = flavor_of(self._obj)
                #flavor = flavor_of(self._obj)
                nparr = array_as_internal(self._obj, flavor)
            except:  # XXX
                # Problems converting data. Close the node and re-raise exception.
                self.close(flush=0)
                raise

            # Get the HDF5 type associated with this numpy type
            self.shape = nparr.shape
            if min(self.shape) < 1:
                raise ValueError("shape parameter cannot have zero-dimensions.")
            if self.atom is None or self.atom.shape == ():
                dtype_ = nparr.dtype.base
                self.atom = Atom.from_dtype(dtype_)
            else:
                self.shape = self.shape[:-len(self.atom.shape)]

        if expectedrows is None:
            expectedrows = self.nrows

        if self._v_chunkshape is None:
            # Compute the optimal chunk size
            self._v_chunkshape = self._calc_chunkshape(
                expectedrows, self.rowsize, self.atom.size)
        # Compute the optimal nrowsinbuf
        self.nrowsinbuf = self._calc_nrowsinbuf()
        # Correct the byteorder if needed
        if self.byteorder is None:
            self.byteorder = correct_byteorder(self.atom.type, sys.byteorder)

        self.disk_type_id = atom_to_hdf5_type(self.atom, self.byteorder)

        try:
            # ``self._v_objectid`` needs to be set because would be
            # needed for setting attributes in some descendants later
            # on
            self._v_objectid = self._create_carray(self._v_new_title)
        except:  # XXX
            # Problems creating the Array on disk. Close node and re-raise.
            self.close(flush=0)
            raise
        # copy values into array
        if self._obj is not None:
            self[...] = nparr
        self._obj = None  # deref obj
        return self._v_objectid

    def _g_copy_with_stats(self, group, name, start, stop, step,
                           title, filters, chunkshape, _log, **kwargs):
        """Private part of Leaf.copy() for each kind of leaf"""

        (start, stop, step) = self._process_range_read(start, stop, step)
        maindim = self.maindim
        shape = list(self.shape)
        shape[maindim] = len(xrange(start, stop, step))
        # Now, fill the new carray with values from source
        nrowsinbuf = self.nrowsinbuf
        # The slices parameter for self.__getitem__
        slices = [slice(0, dim, 1) for dim in self.shape]
        # This is a hack to prevent doing unnecessary conversions
        # when copying buffers
        self._v_convert = False
        # Build the new CArray object
        object = CArray(group, name, atom=self.atom, shape=shape,
                        title=title, filters=filters, chunkshape=chunkshape,
                        _log=_log)
        # Start the copy itself
        for start2 in xrange(start, stop, step * nrowsinbuf):
            # Save the records on disk
            stop2 = start2 + step * nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Set the proper slice in the main dimension
            slices[maindim] = slice(start2, stop2, step)
            start3 = (start2 - start) // step
            stop3 = start3 + nrowsinbuf
            if stop3 > shape[maindim]:
                stop3 = shape[maindim]
            # The next line should be generalised if, in the future,
            # maindim is designed to be different from 0 in CArrays.
            # See ticket #199.
            object[start3:stop3] = self.__getitem__(tuple(slices))
        # Activate the conversion again (default)
        self._v_convert = True
        nbytes = numpy.prod(self.shape, dtype=SizeType) * self.atom.size

        return (object, nbytes)

    _g_copyWithStats = previous_api(_g_copy_with_stats)
