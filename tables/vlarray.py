# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

########################################################################
#
#       License: BSD
#       Created: November 12, 2003
#       Author:  Francesc Altet - faltet@carabos.com
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
from tables.utils import convertToNPAtom, idx2long, byteorders, \
     correct_byteorder

from tables.atom import ObjectAtom, VLStringAtom, EnumAtom, Atom, split_type
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

    This class provides methods to write or read data to or from
    variable length array objects in the file.  Note that it also
    inherits all the public attributes and methods that `Leaf` already
    provides.

    Instance variables (specific of `VLArray`):

    `atom`
        An `Atom` instance representing the shape and type of the
        atomic objects to be saved.
    `nrows`
        The total number of rows of the array.
    `nrow`
        On iterators, this is the index of the current row.
    `shape`
        The shape of the array (expressed as ``(self.nrows,)``).
    """

    # Class identifier.
    _c_classId = 'VLARRAY'


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
        """Create the instance Array.

        Keyword arguments:

        `atom`
            An `Atom` instance representing the shape and type of the
            atomic objects to be saved.
        `title`
            Sets a ``TITLE`` attribute on the array entity.
        `filters`
            An instance of the `Filters` class that provides
            information about the desired I/O filters to be applied
            during the life of this object.
        `expectedsizeinMB`
            An user estimate about the size (in MB) in the final
            `VLArray` object.  If not provided, the default value is 1
            MB.  If you plan to create a much smaller or a much
            bigger `VLArray` try providing a guess; this will optimize
            the HDF5 B-Tree creation and management process time and
            the amount of memory used.
        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation. Filters are applied to those
            chunks of data. The dimensionality of `chunkshape` must be
            1. If ``None``, a sensible value is calculated (which is
            recommended).
        `byteorder` -- The byteorder of the data *on-disk*, specified
            as 'little' or 'big'. If this is not specified, the
            byteorder is that of the platform.
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

        self._v_nrowsinbuf = 100       # maybe enough for most applications
        """The maximum number of rows that are read on each chunk iterator."""
        self._v_chunkshape = None
        """The HDF5 chunk shape for ``VLArray`` objects."""
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
        self.atom = atom
        """
        An `Atom` instance representing the shape and type of the
        atomic objects to be saved.
        """
        self.nrow = None
        """On iterators, this is the index of the current row."""
        self.nrows = None
        """The total number of rows."""

        if new:
            if chunkshape is not None:
                if type(chunkshape) in (int, long):
                    chunkshape = (long(chunkshape),)
                if type(chunkshape) not in (tuple, list):
                    raise ValueError, """\
chunkshape parameter should be an int, tuple or list and you passed a %s.
""" % type(chunkshape)
                elif len(chunkshape) != 1:
                    raise ValueError, """\
the chunkshape (%s) rank must be equal to 1.""" % (chunkshape)
                else:
                    self._v_chunkshape = chunkshape

        super(VLArray, self).__init__(parentNode, name, new, filters,
                                      byteorder, _log)


    # This is too specific for moving it to Leaf
    def _calc_chunkshape(self, expectedsizeinMB):
        """Calculate the size for the HDF5 chunk."""

        chunksize = calc_chunksize(expectedsizeinMB)

        # For computing the chunkshape for HDF5 VL types, we have to
        # choose the itemsize of the *each* element of the atom and
        # not the size of the entire atom.  I don't know why this
        # should be like this, perhaps I should report this to the
        # HDF5 list.
        # F. Altet 2006-11-23
        #elemsize = self.atom.atomsize()
        elemsize = self._basesize
        # Set the chunkshape
        chunkshape = chunksize // elemsize
        # Safeguard against itemsizes being extremely large
        if chunkshape == 0:
            chunkshape = 1
        return (chunkshape,)


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
            self._atomicdtype = atom.base.dtype.base
            self._atomicsize = atom.base.size
            self._basesize = atom.base.itemsize
        else:
            self._atomicdtype = atom.dtype.base
            self._atomicsize = atom.size
            self._basesize = atom.itemsize
        self._atomictype = atom.type
        self._atomicshape = atom.shape

        # Compute the optimal chunkshape, if needed
        if self._v_chunkshape is None:
            self._v_chunkshape = self._calc_chunkshape(
                self._v_expectedsizeinMB)
        self.nrows = 0     # No rows at creation time

        # Correct the byteorder if needed
        if self.byteorder is None:
            self.byteorder = correct_byteorder(atom.type, sys.byteorder)

        # After creating the vlarray, ``self._v_objectID`` needs to be
        # set because it is needed for setting attributes afterwards.
        self._v_objectID = self._createArray(self._v_new_title)

        # Add an attribute in case we have a pseudo-atom so that we
        # can retrieve the proper class after a re-opening operation.
        if atom.kind in ('vlstring', 'object'):
            self.attrs.PSEUDOATOM = atom.kind

        return self._v_objectID


    def _g_open(self):
        """Get the metadata info for an array in file."""

        self._v_objectID, self.nrows, self._v_chunkshape = self._openArray()

        if "PSEUDOATOM" in self.attrs:
            kind = self.attrs.PSEUDOATOM
            if kind == 'vlstring':
                atom = VLStringAtom()
            elif kind == 'object':
                atom = ObjectAtom()
            else:
                raise ValueError(
                    "pseudo-atom name ``%s`` not known." % kind)
        else:
            kind, itemsize = split_type(self._atomictype)
            if kind == 'enum':
                dflt = iter(self._enum).next()[0]  # ignored, any of them is OK
                base = Atom.from_dtype(self._atomicdtype)
                atom = EnumAtom(self._enum, dflt, base,
                                shape=self._atomicshape)
            else:
                if itemsize is None:  # some types don't include precision
                    itemsize = self._atomicdtype.itemsize
                shape = self._atomicshape
                atom = Atom.from_kind(kind, itemsize, shape=shape)

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
        Append objects in the `sequence` to the array.

        This method appends the objects in the `sequence` to a *single
        row* in this array.  The type of individual objects must be
        compliant with the type of atoms in the array.  In the case of
        variable length strings, the very string to append is the
        `sequence`.

        Example of use (code available in ``examples/vlarray1.py``)::

            import tables
            from numpy import *   # or, from numarray import *

            # Create a VLArray:
            fileh = tables.openFile("vlarray1.h5", mode = "w")
            vlarray = fileh.createVLArray(
                fileh.root, 'vlarray1',
                tables.Int32Atom(), "ragged array of ints",
                filters=Filters(complevel=1))
            vlarray.flavor = 'Numeric'
            # Append some (variable length) rows:
            vlarray.append(array([5, 6]))
            vlarray.append(array([5, 6, 7]))
            vlarray.append([5, 6, 9, 8])

            # Now, read it through an iterator:
            for x in vlarray:
                print vlarray.name+"["+str(vlarray.nrow)+"]-->", x

            # Close the file
            fileh.close()

        The output of the previous program looks like this::

            vlarray1[0]--> [5 6]
            vlarray1[1]--> [5 6 7]
            vlarray1[2]--> [5 6 9 8]
        """

        self._v_file._checkWritable()

        try:  # fastest check in most cases
            len(sequence)
        except TypeError:
            raise TypeError("argument is not a sequence")
        else:
            object = sequence

        # Prepare the object to convert it into a NumPy object
        atom = self.atom
        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            object = atom.toarray(object)
            statom = atom.base
        else:
            statom = atom

        if len(object) > 0:
            # The object needs to be copied to make the operation safe
            # to in-place conversion.
            copy = self._atomictype in ['time64']
            nparr = convertToNPAtom(object, statom, copy)
            nobjects = self._getnobjects(nparr)
            # Finally, check the byteorder and change it if needed
            if (self.byteorder in ['little', 'big'] and
                byteorders[nparr.dtype.byteorder] != sys.byteorder):
                    # The byteorder needs to be fixed (a copy is made
                    # so that the original array is not modified)
                    nparr = nparr.byteswap()
        else:
            nobjects = 0
            nparr = None

        self._append(nparr, nobjects)
        self.nrows += 1


    def iterrows(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.

        """

        (self._start, self._stop, self._step) = \
                     self._processRangeRead(start, stop, step)
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
            if self._row+1 >= self._v_nrowsinbuf or self._row < 0:
                self._stopb = self._startb+self._step*self._v_nrowsinbuf
                self.listarr = self.read(self._startb, self._stopb, self._step)
                self._row = -1
                self._startb = self._stopb
            self._row += 1
            self.nrow += self._step
            self._nrowsread += self._step
            return self.listarr[self._row]


    def __getitem__(self, key):
        """Returns a vlarray row or slice.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key"is an integer, the corresponding row is returned. If
        "key" is a slice, the row slice determined by key is returned.

        """

        if type(key) in (int,long) or isinstance(key, numpy.integer):
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

        self._v_file._checkWritable()

        if not isinstance(keys, tuple):
            keys = (keys, None)
        if len(keys) > 2:
            raise IndexError, "You cannot specify more than two dimensions"
        nrow, rng = keys
        # Process the first index
        if not (type(nrow) in (int,long) or isinstance(nrow, numpy.integer)):
            raise IndexError, "The first dimension only can be an integer"
        if nrow >= self.nrows:
            raise IndexError, "First index out of range"
        if nrow < 0:
            # To support negative values
            nrow += self.nrows
        # Process the second index
        if type(rng) in (int,long) or isinstance(rng, numpy.integer):
            start = rng; stop = start+1; step = 1
        elif isinstance(rng, slice):
            start, stop, step = rng.start, rng.stop, rng.step
        elif rng is None:
            start, stop, step = None, None, None
        else:
            raise IndexError, "Non-valid second index or slice: %s" % rng

        object = value
        # Prepare the object to convert it into a NumPy object
        atom = self.atom
        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            object = atom.toarray(object)
            statom = atom.base
        else:
            statom = atom
        value = convertToNPAtom(object, statom)
        nobjects = self._getnobjects(value)

        # Get the previous value
        nrow = idx2long(nrow)   # To convert any possible numpy scalar value
        nparr = self._readArray(nrow, nrow+1, 1)[0]
        nobjects = len(nparr)
        if len(value) > nobjects:
            raise ValueError, \
"Length of value (%s) is larger than number of elements in row (%s)" % \
(len(value), nobjects)
        # Assign the value to it
        # The values can be numpy scalars. Convert them before building the slice.
        if start is not None: start = idx2long(start)
        if stop is not None: stop = idx2long(stop)
        if step is not None: step = idx2long(step)
        try:
            nparr[slice(start, stop, step)] = value
        except Exception, exc:  #XXX
            raise ValueError, \
"Value parameter:\n'%r'\ncannot be converted into an array object compliant vlarray[%s] row: \n'%r'\nThe error was: <%s>" % \
        (value, keys, nparr[slice(start, stop, step)], exc)

        if nparr.size > 0:
            self._modify(nrow, nparr, nobjects)


    # Accessor for the _readArray method in superclass
    def read(self, start=None, stop=None, step=1):
        """Read the array from disk and return it as a self.flavor object."""

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


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf"

        # Build the new VLArray object
        object = VLArray(
            group, name, self.atom, title=title, filters=filters,
            expectedsizeinMB=self._v_expectedsizeinMB, _log=_log)
        # Now, fill the new vlarray with values from the old one
        # This is not buffered because we cannot forsee the length
        # of each record. So, the safest would be a copy row by row.
        # In the future, some analysis can be done in order to buffer
        # the copy process.
        nrowsinbuf = 1
        (start, stop, step) = self._processRangeRead(start, stop, step)
        # Optimized version (no conversions, no type and shape checks, etc...)
        nrowscopied = 0
        nbytes = 0
        atomsize = self.atom.size
        for start2 in xrange(start, stop, step*nrowsinbuf):
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
