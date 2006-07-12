########################################################################
#
#       License: BSD
#       Created: June 15, 2005
#       Author:  Antonio Valentino
#       Modified by:  Francesc Altet
#
#       $Id$
#
########################################################################

"""Here is defined the CArray class.

See CArray class docstring for more info.

Classes:

    CArray

Functions:


Misc variables:

    __version__


"""

import sys, warnings

import numarray
import numarray.records as records

from tables.Atom import Atom
from tables.Array import Array
from tables.utils import processRangeRead


__version__ = "$Revision$"


# default version for CARRAY objects
obversion = "1.0"    # Support for time & enumerated datatypes.



class CArray(Array):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from Numeric, numarray,
    lists, tuples, strings or scalars, or open existing ones.

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

      Common to all Array's:
        read(start, stop, step)
        iterrows(start, stop, step)

    Instance variables:

      Common to all Array's:

        type -- The type class for the array.
        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        flavor -- The flavor of this object.
        nrow -- On iterators, this is the index of the row currently
            dealed with.


    """

    # Class identifier.
    _c_classId = 'CARRAY'


    # <undo-redo support>
    _c_canUndoCreate = True  # Can creation/copying be undone and redone?
    _c_canUndoRemove = True  # Can removal be undone and redone?
    _c_canUndoMove   = True  # Can movement/renaming be undone and redone?
    # </undo-redo support>


    # <properties>

    def _g_getnrows(self):
        if not self.shape:
            return 1  # scalar case
        else:
            return self.shape[0]

    nrows = property(
        _g_getnrows, None, None,
        "The length of the enlargeable dimension of the array.")

    _v_expectedrows = property(
        _g_getnrows, None, None,
        "The expected number of rows to be stored in the array.")

    # </properties>


    def __init__(self, parentNode, name,
                 shape=None, atom=None,
                 title="", filters=None,
                 log=True):
        """Create CArray instance.

        Keyword arguments:

        shape -- The shape of the chunked array to be saved.

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. One of the shape
            dimensions must be 0. The dimension being 0 means that the
            resulting EArray object can be extended along it.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        """

        if shape is not None and type(shape) not in (list, tuple):
            raise ValueError, """\
shape parameter should be either a tuple or a list and you passed a %s""" \
        % type(shape)

        if atom is not None and not isinstance(atom, Atom):
            raise ValueError, """\
atom parameter should be an instance of tables.Atom and you passed a %s""" \
        % type(atom)

        self._v_version = None
        """The object version of this array."""

        self._v_new = new = shape is not None and atom is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""

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
        """The HDF5 chunk size for ``CArray`` objects."""
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

        self.flavor = None
        """
        The object representation of this array.  It can be any of
        'numarray', 'numpy', 'numeric' or 'python' values.
        """
        self.type = None
        """The type class of the represented array."""
        self.stype = None
        """The string type of the represented array."""
        self.itemsize = None
        """The size of the base items."""
        self.extdim = -1  # `CArray` objects are not enlargeable
        """The index of the enlargeable dimension."""

        # Documented (*public*) attributes.
        self.atom = atom
        """
        An `Atom` instance representing the shape, type and flavor of
        the atomic objects to be saved.
        """

        if new:
            self.shape = tuple(shape)
        else:
            if shape is not None:
                warnings.warn("``atom`` is ``None``: ``shape`` ignored")
            if atom is not None:
                warnings.warn("``shape`` is ``None``: ``atom`` ignored")

        # The `Array` class is not abstract enough! :(
        super(Array, self).__init__(parentNode, name, new, filters, log)


    def _calcMaxTuples(self, atom, nrows, compress=None):
        """Calculate the maximum number of tuples."""

        # The buffer size
        expectedfsizeinKb = numarray.product(self.shape) * atom.itemsize / 1024
        buffersize = self._g_calcBufferSize(expectedfsizeinKb)

        # Max Tuples to fill the buffer
        maxTuples = buffersize // numarray.product(self.shape[1:])

        # Check if at least 1 tuple fits in buffer
        if maxTuples == 0:
            maxTuples = 1

        return maxTuples

    def _g_create(self):
        """Create a fresh array (i.e., not present on HDF5 file)."""

        if not isinstance(self.shape, tuple):
            raise TypeError(
                "the ``shape`` passed to the ``CArray`` constructor "
                "must be a tuple: %r" % (self.shape,))

        if not isinstance(self.atom, Atom):
            raise TypeError(
                "the ``atom`` passed to the ``CArray`` constructor "
                "must be a descendent of the ``Atom`` class: %r"
                % (self.atom,))

        if not isinstance(self.atom.shape, tuple):
            if isinstance(self.atom.shape, int):
                self.atom.shape = (self.atom.shape,)
            else:
                raise TypeError(
"""the shape of ``atom`` must be an int or tuple and you passed: %r""" % \
(self.atom.shape,))

        # Version, type, shape, flavor, byteorder
        self._v_version = obversion
        self.type = self.atom.type
        self.stype = self.atom.stype
        #self.shape = self.atom.shape
        self.flavor = self.atom.flavor
        if self.type == "CharType" or isinstance(self.type, records.Char):
            self.byteorder = "non-relevant"
        else:
            # Only support for creating objects in system byteorder
            self.byteorder  = sys.byteorder

        # Compute some values for buffering and I/O parameters
        # Compute the rowsize for each element
        self.rowsize = self.atom.itemsize
        for i in self.shape:
            if i>0:
                self.rowsize *= i
            else:
                raise ValueError, \
                      "A CArray object cannot have zero-dimensions."

        self.itemsize = self.atom.itemsize

        if min(self.atom.shape) < 1:
            raise ValueError, \
                  "Atom in CArray object cannot have zero-dimensions."

        self._v_chunksize = tuple(self.atom.shape)
        if len(self.shape) != len(self._v_chunksize):
            raise ValueError, "The CArray rank and atom rank must be equal:" \
                              " CArray.shape = %s, atom.shape = %s." % \
                                    (self.shape, self.atom.shape)

        # Compute the buffer chunksize
        self._v_maxTuples = self._calcMaxTuples(
            self.atom, self.nrows, self.filters.complevel)

        try:
            return self._createEArray(self._v_new_title)
        except:  #XXX
            # Problems creating the Array on disk. Close node and re-raise.
            self.close(flush=0)
            raise


    def _g_open(self):
        """Get the metadata info for an array in file."""

        (oid, self.type, self.stype, self.shape, self.itemsize, self.byteorder,
         self._v_chunksize) = self._openArray()

        # Post-condition
        assert self.extdim == -1, "extdim != -1: this should never happen!"
        assert numarray.product(self._v_chunksize) > 0, \
                "product(self._v_chunksize) > 0: this should never happen!"

        # Compute the rowsize for each element
        self.rowsize = self.itemsize
        if self._v_chunksize:
            for i in xrange(len(self._v_chunksize)):
                self.rowsize *= self._v_chunksize[i]
        else:
            for i in xrange(len(self.shape)):
                self.rowsize *= self.shape[i]

        # Compute the real shape for atom:
        shape = list(self._v_chunksize)
        if self.type == "CharType" or isinstance(self.type, records.Char):
            # Add the length of the array at the end of the shape for atom
            shape.append(self.itemsize)
        shape = tuple(shape)

        # Create the atom instance
        self.atom = Atom(dtype=self.stype, shape=shape,
                         flavor=self.flavor, warn=False)

        # Compute the maximum number of tuples
        self._v_maxTuples = self._calcMaxTuples(self.atom, self.nrows)

        return oid


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, log):
        "Private part of Leaf.copy() for each kind of leaf"
        # Now, fill the new carray with values from source
        nrowsinbuf = self._v_maxTuples
        # The slices parameter for self.__getitem__
        slices = [slice(0, dim, 1) for dim in self.shape]
        # This is a hack to prevent doing innecessary conversions
        # when copying buffers
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        self._v_convert = False
        shape = list(self.shape)
        shape[0] = ((stop - start - 1) / step) + 1
        # Build the new CArray object
        object = CArray(group, name, shape, atom=self.atom,
                          title=title, filters=filters, log=log)
        # Start the copy itself
        for start2 in range(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Set the proper slice in the first dimension
            slices[0] = slice(start2, stop2, step)
            start3 = (start2-start)/step
            stop3 = start3 + nrowsinbuf
            if stop3 > shape[0]:
                stop3 = shape[0]
            object[start3:stop3] = self.__getitem__(tuple(slices))
        # Activate the conversion again (default)
        self._v_convert = True
        nbytes = numarray.product(self.shape)*self.itemsize

        return (object, nbytes)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom = %r
  nrows = %s
  flavor = %r
  byteorder = %r""" % (self, self.atom, self.nrows, self.flavor, self.byteorder)
