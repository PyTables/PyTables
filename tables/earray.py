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

import numpy

from tables.parameters import EXPECTED_ROWS_EARRAY
from tables.utils import convertToNPAtom, convertToNPAtom2
from tables.atom import Atom, EnumAtom, split_type
from tables.leaf import Leaf
from tables.carray import CArray

__version__ = "$Revision$"


# default version for EARRAY objects
#obversion = "1.0"    # initial version
#obversion = "1.1"    # support for complex datatypes
#obversion = "1.2"    # This adds support for time datatypes.
obversion = "1.3"    # This adds support for enumerated datatypes.



class EArray(CArray):
    """
    This class represents extendible, homogeneous datasets in an HDF5 file.

    The main difference between an `EArray` and a `CArray`, from which
    it inherits, is that the former can be enlarged along one of its
    dimensions (the *enlargeable dimension*) using the `self.append()`
    method (multiple enlargeable dimensions might be supported in the
    future).  An `EArray` dataset can also be shrunken along its
    enlargeable dimension using the `self.truncate()` method.
    """

    # Class identifier.
    _c_classId = 'EARRAY'


    # Special methods
    # ~~~~~~~~~~~~~~~
    def __init__( self, parentNode, name,
                  atom=None, shape=None, title="",
                  filters=None, expectedrows=EXPECTED_ROWS_EARRAY,
                  chunkshape=None, byteorder=None,
                  _log=True ):
        """
        Create an `EArray` instance.

        Keyword arguments:

        `atom`
            An `Atom` instance representing the shape and type of the
            atomic objects to be saved.
        `shape`
            The shape of the array. One of the dimensions of the
            shape must be 0, meaning that the array can be extended
            along it.
        `title`
            Sets a ``TITLE`` attribute on the array entity.
        `filters`
            An instance of the `Filters` class that provides
            information about the desired I/O filters to be applied
            during the life of this object.
        `expectedrows`
            In the case of enlargeable arrays this represents an user
            estimate about the number of row elements that will be
            added to the growable dimension in the `EArray` object.
            If you plan to create either a much smaller or a much
            bigger `EArray` try providing a guess; this will optimize
            the HDF5 B-Tree creation and management process time and
            the amount of memory used.
        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The dimensionality of `chunkshape` must
            be the same as that of `shape` (beware: no dimension
            should be zero this time!).  If ``None``, a sensible value
            is calculated (which is recommended).
        `byteorder` -- The byteorder of the data *on-disk*, specified
            as 'little' or 'big'. If this is not specified, the
            byteorder is that of the platform.
        """
        # Specific of EArray
        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the array."""

        # Call the parent (CArray) init code
        super(EArray, self).__init__(parentNode, name, atom, shape, title,
                                     filters, chunkshape, byteorder, _log)


    # Public and private methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _g_create(self):
        """Create a new array in file (specific part)."""

        # Pre-conditions and extdim computation
        zerodims = numpy.sum(numpy.array(self.shape) == 0)
        if zerodims > 0:
            if zerodims == 1:
                self.extdim = list(self.shape).index(0)
            else:
                raise NotImplementedError(
                    "Multiple enlargeable (0-)dimensions are not "
                    "supported.")
        else:
            raise ValueError(
                "When creating EArrays, you need to set one of "
                "the dimensions of the Atom instance to zero.")

        # Finish the common part of the creation process
        return self._g_create_common(self._v_expectedrows)


    def _checkShapeAppend(self, nparr):
        "Test that nparr shape is consistent with underlying EArray."

        # The arrays conforms self expandibility?
        myshlen = len(self.shape)
        nashlen = len(nparr.shape)
        if myshlen != nashlen:
            raise ValueError("""\
the ranks of the appended object (%d) and the ``%s`` EArray (%d) differ"""
                             % (nashlen, self._v_pathname, myshlen))
        for i in range(myshlen):
            if i != self.extdim and self.shape[i] != nparr.shape[i]:
                raise ValueError("""\
the shapes of the appended object and the ``%s`` EArray \
differ in non-enlargeable dimension %d""" % (self._v_pathname, i))


    def append(self, sequence):
        """Append the sequence to this (enlargeable) object."""

        self._v_file._checkWritable()

        # Convert the sequence into a NumPy object
        nparr = convertToNPAtom2(sequence, self.atom)
        # Check if it has a consistent shape with underlying EArray
        self._checkShapeAppend(nparr)
        self._append(nparr)


    def truncate(self, size):
        "Truncate the extendable dimension to at most size rows."

        if size <= 0:
            raise ValueError("`size` must be greater than 0")
        self._truncateArray(size)


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf."
        # Build the new EArray object
        maindim = self.maindim
        shape = list(self.shape)
        shape[maindim] = 0
        # Build the new EArray object (do not specify the chunkshape so that
        # a sensible value would be calculated)
        object = EArray(
            group, name, atom=self.atom, shape=shape, title=title,
            filters=filters, expectedrows=self.nrows, _log=_log)
        # Now, fill the new earray with values from source
        nrowsinbuf = self._v_nrowsinbuf
        # The slices parameter for self.__getitem__
        slices = [slice(0, dim, 1) for dim in self.shape]
        # This is a hack to prevent doing unnecessary conversions
        # when copying buffers
        (start, stop, step) = self._processRangeRead(start, stop, step)
        self._v_convert = False
        # Start the copy itself
        for start2 in range(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Set the proper slice in the extensible dimension
            slices[maindim] = slice(start2, stop2, step)
            object._append(self.__getitem__(tuple(slices)))
        # Active the conversion again (default)
        self._v_convert = True
        nbytes = numpy.product(self.shape)*self.atom.itemsize

        return (object, nbytes)
