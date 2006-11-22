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

import numpy

from tables.Atom import Atom, StringAtom
from tables.Array import Array
from tables.utils import processRangeRead, calcBufferSize


__version__ = "$Revision$"


# default version for CARRAY objects
obversion = "1.0"    # Support for time & enumerated datatypes.



class CArray(Array):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from NumPy, numarray,
    Numeric, lists, tuples, strings or scalars, or open existing ones.

    All NumPy datatype are supported.

    Methods (specific of CArray):

        No specific methods.

    Instance variables (specific of CArrays):

        nrow -- On iterators, this is the index of the row currently
            dealed with.


    """

    # Class identifier.
    _c_classId = 'CARRAY'


    # <properties>

    def _g_getnrows(self):
        if not self.shape:
            return 1  # scalar case
        else:
            return self.shape[0]

    nrows = property(
        _g_getnrows, None, None,
        "The length of the enlargeable dimension of the array.")

    # </properties>


    def _calcMaxTuples(self, nrows):
        """Calculate the maximum number of tuples for buffer."""

        # The buffer size
        expectedfsizeinKb = numpy.product(self.shape) * \
                            self.atom.atomsize() / 1024
        buffersize = calcBufferSize(expectedfsizeinKb)

        # Max Tuples to fill the buffer
        maxTuples = buffersize // numpy.product(self.shape[1:])

        # Check that buffer will have 1 tuple at least
        if maxTuples == 0:
            maxTuples = 1

        return maxTuples


    def __init__(self, parentNode, name, atom=None, shape=None,
                 title="", filters=None, chunkshape=None,
                 _log=True):
        """Create CArray instance.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved.

        shape -- The shape of the array.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        chunkshape -- The shape of the data chunk to be read or
            written as a single HDF5 I/O operation. The filters are
            applied to those chunks of data. Its dimensionality has to
            be the same as shape.

        """

        self._v_version = None
        """The object version of this array."""

        self._v_new = new = shape is not None and atom is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""

        self._v_maxTuples = None
        """The maximum number of rows that are read on each chunk iterator."""
        self._v_convert = True
        """Whether the ``Array`` object must be converted or not."""
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
        'numpy', 'numarray', 'numeric' or 'python' values.
        """
        self.dtype = None
        """The NumPy type of the represented array."""
        self.ptype = None
        """The PyTables type of the represented array."""
        self.extdim = -1  # `CArray` objects are not enlargeable
        """The index of the enlargeable dimension."""

        # Documented (*public*) attributes.
        self.atom = atom
        """
        An `Atom` instance representing the shape, type and flavor of
        the atomic objects to be saved.
        """

        if new:
            if shape is None or chunkshape is None:
                raise ValueError, """\
you must specify both shape & chunkshape for building a CArray"""

            if type(shape) not in (list, tuple):
                raise ValueError, """\
shape parameter should be either a tuple or a list and you passed a %s""" \
                % type(shape)

            if type(chunkshape) not in (list, tuple):
                raise ValueError, """\
chunkshape parameter should be either a tuple or a list and you passed a %s""" \
                % type(chunkshape)

            if not isinstance(atom, Atom):
                raise ValueError, """\
atom parameter should be an instance of tables.Atom and you passed a %s""" \
                % type(atom)

            self.shape = tuple(shape)
            """The shape of the stored array."""
            self._v_chunkshape = tuple(chunkshape)
            """The shape of the HDF5 chunk for ``CArray`` objects."""

        # The `Array` class is not abstract enough! :(
        super(Array, self).__init__(parentNode, name, new, filters, _log)


    def _g_create(self):
        """Create a fresh array (i.e., not present on HDF5 file)."""

        if not isinstance(self.atom, Atom):
            raise TypeError(
                "the ``atom`` passed to the ``CArray`` constructor "
                "must be a descendent of the ``Atom`` class: %r"
                % (self.atom,))

        # Version, types, flavor
        self._v_version = obversion
        self.dtype = self.atom.dtype
        self.ptype = self.atom.ptype
        self.flavor = self.atom.flavor

        # Different checks for shape and chunkshape
        if min(self.shape) < 1:
            raise ValueError, \
                  "Atom in CArray object cannot have zero-dimensions."
        if min(self._v_chunkshape) < 1:
            raise ValueError, \
                  "Atom in CArray object cannot have zero-dimensions."
        if len(self.shape) != len(self._v_chunkshape):
            raise ValueError, "The CArray rank and atom rank must be equal:" \
                              " CArray.shape = %s, atom.shape = %s." % \
                                    (self.shape, self.atom.shape)

        # Compute the buffer size for copying purposes
        self._v_maxTuples = self._calcMaxTuples(self.nrows)

        try:
            return self._createEArray(self._v_new_title)
        except:  #XXX
            # Problems creating the Array on disk. Close node and re-raise.
            self.close(flush=0)
            raise


    def _g_open(self):
        """Get the metadata info for an array in file."""

        (oid, self.dtype, self.ptype, self.shape,
         self.flavor, self._v_chunkshape) = self._openArray()

        # Post-condition
        assert self.extdim == -1, "extdim != -1: this should never happen!"
        assert numpy.product(self._v_chunkshape) > 0, \
                "product(self._v_chunkshape) > 0: this should never happen!"

        # Create the Atom instance
        if self.ptype == "String":
            self.atom = StringAtom(length=self.itemsize,
                                   flavor=self.flavor, warn=False)
        else:
            self.atom = Atom(dtype=self.ptype,
                             flavor=self.flavor, warn=False)

        # Compute the maximum number of tuples
        self._v_maxTuples = self._calcMaxTuples(self.nrows)

        return oid


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
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
        #shape[0] = ((stop - start - 1) / step) + 1
        shape[0] = len(xrange(start, stop, step))
        # Build the new CArray object
        object = CArray(group, name, atom=self.atom, shape=shape,
                        title=title, filters=filters,
                        chunkshape = self._v_chunkshape, _log=_log)
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
        nbytes = numpy.product(self.shape)*self.itemsize

        return (object, nbytes)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  atom := %r
  shape := %r
  flavor := %r
  byteorder := %r""" % (self, self.atom, self.shape,
                        self.flavor, self.byteorder)
