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
from tables.utils import processRangeRead


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


    # </properties>


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
            be the same as shape.  If not specified, a sensible value
            is calculated (this is the recommended action).

        """

        # Documented (*public*) attributes.
        self.dtype = None
        """The NumPy type of the represented array."""
        self.ptype = None
        """The PyTables type of the represented array."""
        self.atom = atom
        """
        An `Atom` instance representing the shape, type and flavor of
        the atomic objects to be saved.
        """
        self.shape = None
        """The shape of the stored array."""
        self.extdim = -1  # `CArray` objects are not enlargeable by default
        """The index of the enlargeable dimension."""
        self.flavor = None
        """
        The object representation of this array.  It can be any of
        'numpy', 'numarray', 'numeric' or 'python' values.
        """

        # Other private attributes
        self._v_version = None
        """The object version of this array."""
        self._v_new = new = atom is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_nrowsinbuf = None
        """The maximum number of rows that are read on each chunk iterator."""
        self._v_convert = True
        """Whether the ``Array`` object must be converted or not."""
        self._v_chunkshape = None
        """The HDF5 chunk size for ``CArray/EArray`` objects."""
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

        if new:
            if not isinstance(atom, Atom):
                raise ValueError, """\
atom parameter should be an instance of tables.Atom and you passed a %s.""" \
% type(atom)
            elif atom.shape not in ((), 1):
                raise NotImplementedError, """\
sorry, but multidimensional atoms are not supported in this context yet."""

            if shape is None:
                raise ValueError, """\
you must specify a non-empty shape."""
            elif type(shape) not in (list, tuple):
                raise ValueError, """\
shape parameter should be either a tuple or a list and you passed a %s.""" \
% type(shape)
            else:
                self.shape = tuple(shape)

            if chunkshape is not None:
                if type(chunkshape) not in (list, tuple):
                    raise ValueError, """\
chunkshape parameter should be either a tuple or a list and you passed a %s.
""" % type(chunkshape)
                elif len(shape) != len(chunkshape):
                    raise ValueError, """\
the shape (%s) and chunkshape (%s) ranks must be equal.""" \
% (shape, chunkshape)
                elif min(chunkshape) < 1:
                    raise ValueError, """ \
chunkshape parameter cannot have zero-dimensions."""
                else:
                    self._v_chunkshape = tuple(chunkshape)

        # The `Array` class is not abstract enough! :(
        super(Array, self).__init__(parentNode, name, new, filters, _log)


    def _g_create(self):
        """Create a new array in file."""

        # Different checks for shape and chunkshape
        if min(self.shape) < 1:
            raise ValueError, """\
shape parameter cannot have zero-dimensions."""

        # Version, types, flavor
        self._v_version = obversion
        self.dtype = self.atom.dtype
        self.ptype = self.atom.ptype
        self.flavor = self.atom.flavor

        if self._v_chunkshape is None:
            # Compute the optimal chunk size
            self._v_chunkshape = self._calc_chunkshape(self.nrows,
                                                       self.rowsize)
        # Compute the optimal nrowsinbuf
        self._v_nrowsinbuf = self._calc_nrowsinbuf(self._v_chunkshape,
                                                   self.rowsize)
            
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

        # Compute the optimal nrowsinbuf
        self._v_nrowsinbuf = self._calc_nrowsinbuf(self._v_chunkshape,
                                                   self.rowsize)
        return oid


    def getEnum(self):
        """
        Get the enumerated type associated with this array.

        If this array is of an enumerated type, the corresponding `Enum`
        instance is returned.  If it is not of an enumerated type, a
        ``TypeError`` is raised.
        """

        if self.atom.ptype != 'Enum':
            raise TypeError("array ``%s`` is not of an enumerated type"
                            % self._v_pathname)

        return self.atom.enum


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf"
        maindim = self.maindim
        shape = list(self.shape)
        shape[maindim] = len(xrange(start, stop, step))
        # Now, fill the new carray with values from source
        nrowsinbuf = self._v_nrowsinbuf
        # The slices parameter for self.__getitem__
        slices = [slice(0, dim, 1) for dim in self.shape]
        # This is a hack to prevent doing innecessary conversions
        # when copying buffers
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        self._v_convert = False
        # Build the new CArray object (do not specify the chunkshape so that
        # a sensible value would be calculated)
        object = CArray(group, name, atom=self.atom, shape=shape,
                        title=title, filters=filters, _log=_log)
        # Start the copy itself
        for start2 in range(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2 + step * nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Set the proper slice in the main dimension
            slices[maindim] = slice(start2, stop2, step)
            start3 = (start2-start)/step
            stop3 = start3 + nrowsinbuf
            if stop3 > shape[maindim]:
                stop3 = shape[maindim]
            # The next line should be generalised when maindim would be
            # different from 0
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
  maindim := %r
  flavor := %r
  byteorder := %r""" % (self, self.atom, self.shape, self.maindim,
                        self.flavor, self.byteorder)
