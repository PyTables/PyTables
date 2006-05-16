########################################################################
#
#       License: BSD
#       Created: December 15, 2003
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Here is defined the EArray class.

See EArray class docstring for more info.

Classes:

    EArray

Functions:


Misc variables:

    __version__


"""

import sys

import numarray
import numarray.records as records

from tables.constants import EXPECTED_ROWS_EARRAY, CHUNKTIMES
from tables.utils import convertToNA, processRangeRead
from tables.Atom import Atom, EnumAtom
from tables.Array import Array


__version__ = "$Revision$"


# default version for EARRAY objects
#obversion = "1.0"    # initial version
#obversion = "1.1"    # support for complex datatypes
#obversion = "1.2"    # This adds support for time datatypes.
obversion = "1.3"    # This adds support for enumerated datatypes.



class EArray(Array):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from NumPy, Numeric and
    numarray packages, or open existing ones.

    All NumPy, Numeric and numarray typecodes are supported.

    Methods:

      Common to all Array's:
        read(start, stop, step)
        iterrows(start, stop, step)

      Specific of EArray:
        append(sequence)

    Instance variables:

      Common to all Array's:

        type -- The type class for the array.
        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        flavor -- The flavor of this object.
        nrow -- On iterators, this is the index of the row currently
            dealed with.

      Specific of EArray:

        extdim -- The enlargeable dimension.
        nrows -- The value of the enlargeable dimension.


    """

    # Class identifier.
    _c_classId = 'EARRAY'


    # <undo-redo support>
    _c_canUndoCreate = True  # Can creation/copying be undone and redone?
    _c_canUndoRemove = True  # Can removal be undone and redone?
    _c_canUndoMove   = True  # Can movement/renaming be undone and redone?
    # </undo-redo support>


    # <properties>
    def _g_getrowsize(self):
        return self.atom.atomsize()

    rowsize = property(_g_getrowsize, None, None,
                       "The size in bytes of each row in the array.")
    # </properties>

    def __init__(self, parentNode, name,
                 atom=None, title="",
                 filters=None, expectedrows=EXPECTED_ROWS_EARRAY,
                 log=True):
        """Create EArray instance.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. One of the shape
            dimensions must be 0. The dimension being 0 means that the
            resulting EArray object can be extended along it.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- In the case of enlargeable arrays this
            represents an user estimate about the number of row
            elements that will be added to the growable dimension in
            the EArray object. If you plan to create both much smaller
            or much bigger EArrays try providing a guess; this will
            optimize the HDF5 B-Tree creation and management process
            time and the amount of memory used.

        """

        # `Array` has some attributes that are lacking from `EArray`,
        # so the constructor of the former can not be used
        # and attributes must be defined all over again. :(

        self._v_version = None
        """The object version of this array."""

        self._v_new = new = atom is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""

        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the array."""
        self.byteorder = None
        """
        The endianness of data in memory ('big', 'little' or
        'non-relevant').
        """
        self._v_maxTuples = None
        """The maximum number of rows that are read on each chunk iterator."""
        self._v_chunksize = None
        """The HDF5 chunk size for ``EArray`` objects."""
        self._v_convert = True
        """Whether the *Array objects has to be converted or not."""
        self.shape = None
        """The shape of the stored array."""
        self._enum = None
        """The enumerated type containing the values in this array."""

        # Miscellaneous iteration rubbish.
        self.nrow = None
        """On iterators, this is the index of the current row."""
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
        'numarray', 'numpy', 'numeric' or 'python'.
        """
        self.type = None
        """The type class of the represented array."""
        self.stype = None
        """The string type of the represented array."""
        self.itemsize = None
        """The size of the base items."""

        # Documented (*public*) attributes.
        self.atom = atom
        """
        An `Atom` instance representing the shape, type and flavor of
        the atomic objects to be saved.  One of the dimensions of the
        shape is 0, meaning that the array can be extended along it.
        """
        self.extdim = None
        """
        The enlargeable dimension, i.e. the dimension this array can
        be extended along.
        """
        self.nrows = None
        """The length of the enlargeable dimension of the array."""

        # The `Array` class is not abstract enough! :(
        super(Array, self).__init__(parentNode, name, new, filters, log)


    def _calcTuplesAndChunks(self, atom, extdim, expectedrows, compress):
        """Calculate the maximun number of tuples and the HDF5 chunk size."""

        # The buffer size
        rowsize = atom.atomsize()
        expectedfsizeinKb = (expectedrows * rowsize) / 1024
        buffersize = self._g_calcBufferSize(expectedfsizeinKb)

        # Max Tuples to fill the buffer
        maxTuples = buffersize // (rowsize * CHUNKTIMES)
        chunksizes = list(atom.shape)
        # Check if at least 1 tuple fits in buffer
        if maxTuples >= 1:
            # Yes. So the chunk sizes for the non-extendeable dims will be
            # unchanged
            chunksizes[extdim] = maxTuples
        else:
            # No. reduce other dimensions until we get a proper chunksizes
            # shape
            chunksizes[extdim] = 1  # Only one row in extendeable dimension
            for j in range(len(chunksizes)):
                newrowsize = atom.itemsize
                for i in chunksizes[j+1:]:
                    newrowsize *= i
                maxTuples = buffersize // newrowsize
                if maxTuples >= 1:
                    break
                chunksizes[j] = 1
            # Compute the chunksizes correctly for this j index
            chunksize = maxTuples
            if j < len(chunksizes):
                # Only modify chunksizes[j] if needed
                if chunksize < chunksizes[j]:
                    chunksizes[j] = chunksize
            else:
                chunksizes[-1] = 1 # very large itemsizes!
        # Compute the correct maxTuples number
        newrowsize = atom.itemsize
        for i in chunksizes:
            newrowsize *= i
        maxTuples = buffersize // (newrowsize * CHUNKTIMES)
        # Safeguard against row sizes being extremely large
        if maxTuples == 0:
            maxTuples = 1
        return (maxTuples, chunksizes)


    def _g_create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""

        if not isinstance(self.atom, Atom):
            raise TypeError(
                "the object passed to the ``EArray`` constructor "
                "must be an instance of the ``Atom`` class")

        if not isinstance(self.atom.shape, tuple):
            raise TypeError(
                "the ``shape`` in the ``Atom`` instance "
                "must be a tuple for ``EArray``: %r"
                % (self.atom.shape,))

        # Version, type, shape, flavor, byteorder
        self._v_version = obversion
        self.type = self.atom.type
        self.stype = self.atom.stype
        self.shape = self.atom.shape
        self.flavor = self.atom.flavor
        if self.type == "CharType" or isinstance(self.type, records.Char):
            self.byteorder = "non-relevant"
        else:
            # Only support for creating objects in system byteorder
            self.byteorder  = sys.byteorder

        # extdim computation
        zerodims = numarray.sum(numarray.array(self.shape) == 0)
        if zerodims > 0:
            if zerodims == 1:
                self.extdim = list(self.shape).index(0)
            else:
                raise NotImplementedError, \
                      "Multiple enlargeable (0-)dimensions are not supported."
        else:
            raise ValueError, \
                  "When creating EArrays, you need to set one of the dimensions of the Atom instance to zero."

        # Compute some values for buffering and I/O parameters
        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = self._calcTuplesAndChunks(
            self.atom, self.extdim,
            self._v_expectedrows, self.filters.complevel)
        #print "chunksizes-->", self._v_chunksize
        self.nrows = 0   # No rows initially
        self.itemsize = self.atom.itemsize

        self._v_objectID = self._createEArray(self._v_new_title)
        return self._v_objectID


    def _g_open(self):
        """Get the metadata info for an array in file."""

        (self._v_objectID, type_, self.stype, self.shape,
         self.itemsize, self.byteorder, self._v_chunksize) = \
         self._openArray()  # sets `self.flavor`

        stype = self.stype
        flavor = self.flavor
        # Post-condition
        assert self.extdim >= 0, "extdim < 0: this should never happen!"
        # Compute the real shape for atom:
        shape = list(self.shape)
        shape[self.extdim] = 0
        if type_ == "CharType" or isinstance(type_, records.Char):
            # Add the length of the array at the end of the shape for atom
            shape.append(self.itemsize)
        shape = tuple(shape)
        # Create the atom instance and set definitive type
        if stype == 'Enum':
            (enum, type_) = self._loadEnum()
            self.atom = EnumAtom(enum, type_, shape, flavor, warn=False)
        else:
            self.atom = Atom(stype, shape, flavor, warn=False)
        self.type = type_
        # nrows in this instance
        self.nrows = self.shape[self.extdim]
        # Compute the optimal maxTuples
        (self._v_maxTuples, computedChunksize) = self._calcTuplesAndChunks(
            self.atom, self.extdim, self.nrows, self.filters.complevel)

        return self._v_objectID


    def getEnum(self):
        """
        Get the enumerated type associated with this array.

        If this array is of an enumerated type, the corresponding `Enum`
        instance is returned.  If it is not of an enumerated type, a
        ``TypeError`` is raised.
        """

        if self.atom.stype != 'Enum':
            raise TypeError("array ``%s`` is not of an enumerated type"
                            % self._v_pathname)

        return self.atom.enum


    def _checkTypeShape(self, naarr):
        "Test that naarr parameter is shape and type compliant"
        # Check the type
        if not hasattr(naarr, "type"):  # To deal with string objects
            datatype = records.CharType
            # Made an additional check for strings
            if naarr.itemsize() <> self.itemsize:
                raise TypeError, \
"""The object '%r' has not a base string size of '%s'.""" % \
(naarr, self.itemsize)
        else:
            datatype = naarr.type()
        #print "datatype, self.type:", datatype, self.type
        if str(datatype) <> str(self.type):
            raise TypeError, \
"""The object '%r' is not composed of elements of type '%s'.""" % \
(naarr, self.type)

        # The arrays conforms self expandibility?
        myshlen = len(self.shape)
        nashlen = len(naarr.shape)
        if myshlen != nashlen:
            raise ValueError("""\
the ranks of the appended object (%d) and the ``%s`` EArray (%d) differ"""
                             % (nashlen, self._v_pathname, myshlen))
        for i in range(myshlen):
            if i != self.extdim and self.shape[i] != naarr.shape[i]:
                raise ValueError("""\
the shapes of the appended object and the ``%s`` EArray \
differ in non-enlargeable dimension %d""" % (self._v_pathname, i))
        # Ok. all conditions are met. Return the numarray object
        return naarr

    def append(self, sequence):
        """Append the sequence to this (enlargeable) object"""

        if self._v_file.mode == 'r':
            raise IOError("attempt to write over a file opened in read-only mode")

        # The sequence needs to be copied to make the operation safe
        # to in-place conversion.
        copy = self.stype in ['Time64']
        # Convert the sequence into a numarray object
        naarr = convertToNA(sequence, self.atom, copy)
        # Check if it is correct type and shape
        naarr = self._checkTypeShape(naarr)
        self._append(naarr)

    def truncate(self, size):
        "Truncate the extendable dimension to at most size rows"

        if size <= 0:
            raise ValueError("`size` must be greater than 0")
        self._truncateArray(size)


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, log):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new EArray object
        object = EArray(
            group, name, atom=self.atom, title=title, filters=filters,
            expectedrows=self.nrows, log=log)
        # Now, fill the new earray with values from source
        nrowsinbuf = self._v_maxTuples
        # The slices parameter for self.__getitem__
        slices = [slice(0, dim, 1) for dim in self.shape]
        # This is a hack to prevent doing innecessary conversions
        # when copying buffers
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        self._v_convert = False
        # Start the copy itself
        for start2 in range(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Set the proper slice in the extensible dimension
            slices[self.extdim] = slice(start2, stop2, step)
            object._append(self.__getitem__(tuple(slices)))
        # Active the conversion again (default)
        self._v_convert = True
        nbytes = self.itemsize
        for i in self.shape:
            nbytes*=i

        return (object, nbytes)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom = %r
  nrows = %s
  extdim = %r
  flavor = %r
  byteorder = %r""" % (self, self.atom, self.nrows, self.extdim, self.flavor,
                       self.byteorder)
