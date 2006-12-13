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

from tables.constants import EXPECTED_ROWS_EARRAY
from tables.utils import convertToNPAtom, processRangeRead, byteorders
from tables.Atom import Atom, EnumAtom, StringAtom, Time32Atom, Time64Atom
from tables.CArray import CArray

atom_mod = __import__("tables.Atom")

__version__ = "$Revision$"


# default version for EARRAY objects
#obversion = "1.0"    # initial version
#obversion = "1.1"    # support for complex datatypes
#obversion = "1.2"    # This adds support for time datatypes.
obversion = "1.3"    # This adds support for enumerated datatypes.



class EArray(CArray):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from NumPy, Numeric and
    numarray packages, or open existing ones.

    All NumPy, Numeric and numarray typecodes are supported.

    Methods (Specific of EArray):
        append(sequence)

    """

    # Class identifier.
    _c_classId = 'EARRAY'


    def __init__(self, parentNode, name, atom=None, shape=None,
                 title="", filters=None, expectedrows=EXPECTED_ROWS_EARRAY,
                 chunkshape=None, _log=True):
        """Create an EArray instance.

        Keyword arguments:

        atom -- An Atom object representing the type, shape and flavor
            of the atomic objects to be saved in the array.

        shape -- The shape of the array. One of the shape dimensions
            must be 0. The dimension being 0 means that the resulting
            EArray object can be extended along it.

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

        chunkshape -- The shape of the data chunk to be read or
            written as a single HDF5 I/O operation. The filters are
            applied to those chunks of data. Its dimensionality has to
            be the same as shape (beware: no dimension should be zero
            this time!).  If None, a sensible value is calculated
            (which is recommended).

        """

        # Specific of EArray
        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the array."""

        # Call the parent (CArray) init code
        super(EArray, self).__init__(parentNode, name, atom, shape, title,
                                     filters, chunkshape, _log)


    def _g_create(self):
        """Create a new EArray."""

        # Version, dtype, ptype, shape, flavor
        self._v_version = obversion
        # Create a scalar version of dtype
        #self.dtype = numpy.dtype((self.atom.dtype.base.type, ()))
        # Overwrite the dtype attribute of the atom to convert it to a scalar
        # (heck, I should find a more elegant way for dealing with this)
        #self.atom.dtype = self.dtype
        self.dtype = self.atom.dtype
        self.ptype = self.atom.ptype
        self.flavor = self.atom.flavor

        # extdim computation
        zerodims = numpy.sum(numpy.array(self.shape) == 0)
        if zerodims > 0:
            if zerodims == 1:
                self.extdim = list(self.shape).index(0)
            else:
                raise NotImplementedError, \
"Multiple enlargeable (0-)dimensions are not supported."
        else:
            raise ValueError, \
"""When creating EArrays, you need to set one of the dimensions of the Atom
instance to zero."""

        # Compute the optimal chunk size, if needed
        if self._v_chunkshape is None:
            self._v_chunkshape = self._calc_chunkshape(self._v_expectedrows,
                                                       self.rowsize)
        # Compute the optimal nrowsinbuf
        self._v_nrowsinbuf = self._calc_nrowsinbuf(self._v_chunkshape,
                                                   self.rowsize)

        self._v_objectID = self._createEArray(self._v_new_title)
        return self._v_objectID


    def _g_open(self):
        """Get the metadata info for an EArray in file."""

        (self._v_objectID, self.dtype, self.ptype, self.shape,
         self.flavor, self._v_chunkshape) = self._openArray()
        # Post-condition
        assert self.extdim >= 0, "extdim < 0: this should never happen!"

        ptype = self.ptype
        flavor = self.flavor

        # Create the atom instance and set definitive type
        if ptype == "String":
            length = self.dtype.itemsize
            self.atom = StringAtom(length=length, flavor=flavor, warn=False)
        elif ptype == 'Enum':
            (enum, self.dtype) = self._g_loadEnum()
            self.atom = EnumAtom(enum, self.dtype, flavor=flavor, warn=False)
        elif ptype == "Time32":
            self.atom = Time32Atom(flavor=flavor, warn=False)
        elif ptype == "Time64":
            self.atom = Time64Atom(flavor=flavor, warn=False)
        else:
            #self.atom = Atom(ptype, flavor=flavor, warn=False)
            # Make the atoms instantiate from a more specific classes
            # (this is better for representation -- repr() -- purposes)
            typeclassname = numpy.sctypeNA[numpy.sctypeDict[ptype]] + "Atom"
            typeclass = getattr(atom_mod, typeclassname)
            self.atom = typeclass(flavor=flavor, warn=False)

        # Compute the optimal nrowsinbuf
        self._v_nrowsinbuf = self._calc_nrowsinbuf(self._v_chunkshape,
                                                   self.rowsize)
        return self._v_objectID


    def _checkShape(self, nparr):
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

        # The sequence needs to be copied to make the operation safe
        # to in-place conversion.
        copy = self.ptype in ['Time64']
        # Convert the sequence into a NumPy object
        nparr = convertToNPAtom(sequence, self.atom, copy)
        # Check if it has a consistent shape with underlying EArray
        self._checkShape(nparr)
        # Finally, check the byteorder and change it if needed
        if (self.byteorder in ['little', 'big'] and
            byteorders[nparr.dtype.byteorder] != self.byteorder):
            # The byteorder needs to be fixed (a copy is made
            # so that the original array is not modified)
            nparr = nparr.byteswap()
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
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
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
        nbytes = numpy.product(self.shape)*self.itemsize

        return (object, nbytes)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__."""

        return """%s
  atom := %r
  shape := %r
  maindim := %r
  flavor := %r
  byteorder := %r""" % (self, self.atom, self.shape, self.maindim,
                        self.flavor, self.byteorder)
