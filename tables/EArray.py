########################################################################
#
#       License: BSD
#       Created: December 15, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/EArray.py,v $
#       $Id: EArray.py,v 1.15 2004/02/09 18:54:11 falted Exp $
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

__version__ = "$Revision: 1.15 $"
# default version for EARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
from Array import Array
from VLArray import Atom
from utils import calcBufferSize, convertIntoNA, processRangeRead
import hdf5Extension
import numarray
import numarray.strings as strings
import numarray.records as records

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

class EArray(Array, hdf5Extension.Array, object):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from Numeric and
    numarray packages, or open existing ones.

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

      Common to all Array's:
        read(start, stop, step)
        iterrows(start, stop, step)

      Specific of EArray:
        append(object)
        
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
    
    def __init__(self, atom = None, title = "",
                 filters = None, expectedrows = 1000):
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
            the EArray object. If not provided, the default value is
            1000 rows. If you plan to create both much smaller or much
            bigger EArrays try providing a guess; this will optimize
            the HDF5 B-Tree creation and management process time and
            the amount of memory used.

        """
        self._v_new_title = title
        self._v_new_filters = filters
        self._v_expectedrows = expectedrows
        # Check if we have to create a new object or read their contents
        # from disk
        if atom is not None:
            self._v_new = 1
            self.atom = atom
        else:
            self._v_new = 0
            
    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        global obversion

        assert isinstance(self.atom, Atom), "The object passed to the EArray constructor must be a descendent of the Atom class."
        assert isinstance(self.atom.shape, types.TupleType), "The Atom shape has to be a tuple for EArrays, and you passed a '%s' object." % (self.atom.shape)
        # Version, type, shape, flavor, byteorder
        self._v_version = obversion
        self.type = self.atom.type
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
        # Compute the rowsize for each element
        self.rowsize = self.atom.atomsize()
        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
           calcBufferSize(self.rowsize, self._v_expectedrows,
                          self.filters.complevel)
        self.nrows = 0   # No rows initially
        self.itemsize = self.atom.itemsize
        self._createEArray(self._v_new_title)

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
        assert len(self.shape) == len(naarr.shape), \
"Sorry, the ranks of the EArray %r (%d) and object to be appended (%d) differ." % (self._v_pathname, len(self.shape), len(naarr.shape))
        for i in range(len(self.shape)):
            if i <> self.extdim:
                assert self.shape[i] == naarr.shape[i], \
"Sorry, shapes of EArray '%r' and object differ in non-enlargeable dimension (%d) " % (self._v_pathname, i) 
        # Ok. all conditions are met. Return the numarray object
        return naarr
            
    def append(self, object):
        """Append the object to this (enlargeable) object"""

        # Convert the object into a numarray object
        naarr = convertIntoNA(object, self.atom)
        # Check if it is correct type and shape
        naarr = self._checkTypeShape(naarr)
        self._append(naarr)

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize, self.byteorder) = \
                        self._openArray()
        # Post-condition
        assert self.extdim >= 0, "extdim < 0: this should never happen!"
        # Compute the real shape for atom:
        shape = list(self.shape)
        shape[self.extdim] = 0
        if self.type == "CharType" or isinstance(self.type, records.Char):
            # Add the length of the array at the end of the shape for atom
            shape.append(self.itemsize)
        shape = tuple(shape)
        # Create the atom instance
        self.atom = Atom(dtype=self.type, shape=shape,
                         flavor=self.attrs.FLAVOR)
        # Compute the rowsize for each element
        self.rowsize = self.atom.atomsize()
        # nrows in this instance
        self.nrows = self.shape[self.extdim]
        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
                  calcBufferSize(self.rowsize, self.nrows,
                                 self.filters.complevel)

    def _g_copy(self, group, name, start, stop, step, title, filters):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new EArray object
        object = EArray(atom=self.atom,
                        title=title,
                        filters=filters,
                        expectedrows=self.nrows)
        setattr(group, name, object)
        # Now, fill the new earray with values from source
        nrowsinbuf = self._v_maxTuples
        # The slices parameter for self.__getitem__
        slices = [slice(0, dim, 1) for dim in self.shape]
        # This is a hack to prevent doing innecessary conversions
        # when copying buffers
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        self._v_convert = 0
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
        self._v_convert = 1
        return object

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  extdim = %r
  flavor = %r
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.extdim, self.flavor, self.byteorder)
