########################################################################
#
#       License: BSD
#       Created: December 15, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/EArray.py,v $
#       $Id: EArray.py,v 1.6 2004/01/01 21:01:46 falted Exp $
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

__version__ = "$Revision: 1.6 $"
# default version for EARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
from Array import Array
from utils import calcBufferSize
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
    
    def __init__(self, object = None, title = "",
                 compress = 0, complib = "zlib",
                 shuffle = 0, expectednrows = 1000):
        """Create EArray instance.

        Keyword arguments:

        object -- An object describing the kind of objects that you
            can append to the EArray. It can be an instance of any of
            NumArray, CharArray or Numeric classes and one of its
            dimensions must be 0. The dimension being 0 means that the
            resulting EArray object can be extended along it. Multiple
            enlargeable dimensions are not supported right now.

        title -- Sets a TITLE attribute on the array entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default.

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        shuffle -- Whether or not to use the shuffle filter in the
            HDF5 library. This is normally used to improve the
            compression ratio. A value of 0 disables shuffling and it
            is the default.

        expectedrows -- In the case of enlargeable arrays this
            represents an user estimate about the number of row
            elements that will be added to the growable dimension in
            the EArray object. If not provided, the default value is
            1000 rows. If you plan to create both much smaller or much
            bigger EArrays try providing a guess; this will optimize
            the HDF5 B-Tree creation and management process time and
            the amount of memory used.

        """
        self.new_title = title
        self._v_expectednrows = expectednrows
        # Check if we have to create a new object or read their contents
        # from disk
        if object is not None:
            self._v_new = 1
            self.object = object
            self._g_setComprAttr(compress, complib, shuffle)
        else:
            self._v_new = 0
            
    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        global obversion

        self._v_version = obversion
        naarr, self.flavor = self._convertIntoNA(self.object)

        if (isinstance(naarr, strings.CharArray)):
            self.byteorder = "non-relevant" 
        else:
            self.byteorder  = naarr._byteorder

        # Check for null dimensions
        zerodims = numarray.sum(numarray.array(naarr.shape) == 0)
        if zerodims > 0:
            if zerodims == 1:
                extdim = list(naarr.shape).index(0)
                self.extdim = extdim
            else:
                raise NotImplementedError, \
                      "Multiple enlargeable (0-)dimensions are not supported."
        else:
            raise ValueError, \
                  "When creating EArrays, you need to set one of the dimensions of object to zero."

        # Compute some values for buffering and I/O parameters
        # Compute the rowsize for each element
        self.rowsize = naarr.itemsize()
        for i in naarr.shape:
            if i>0:
                self.rowsize *= i
        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
           calcBufferSize(self.rowsize, self._v_expectednrows, self.complevel)

        self.shape = naarr.shape
        self.nrows = naarr.shape[self.extdim]
        self.itemsize = naarr.itemsize()
        self.type = self._createArray(naarr, self.new_title)

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
"""Sorry, the ranks of the EArray '%r' and object to be appended differ
  (%d <> %d).""" % (self._v_pathname, len(self.shape), len(naarr.shape))
        for i in range(len(self.shape)):
            if i <> self.extdim:
                assert self.shape[i] == naarr.shape[i], \
"""Sorry, shapes of EArray '%r' and object differ in dimension %d (non-enlargeable)""" % (self._v_pathname, i) 
        # Ok. all conditions are met. Return the numarray object
        return naarr
            
    def append(self, object):
        """Append the object to this (enlargeable) object"""

        # Convert the object into a numarray object
        naarr, self.flavor = self._convertIntoNA(object)
        naarr = self._checkTypeShape(naarr)
        self._append(naarr)

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize, self.byteorder) = \
                        self._openArray()
        # Post-condition
        assert self.extdim >= 0, "extdim < 0: this should never happen!"
        # Compute the rowsize for each element
        self.rowsize = self.itemsize
        for i in range(len(self.shape)):
            if i <> self.extdim:
                self.rowsize *= self.shape[i]
            else:
                self.nrows = self.shape[i]
        # Get info about existing filters
        self.complevel, self.complib, self.shuffle = self._g_getFilters()
        # Compute the optimal chunksize
        (self._v_maxTuples, self._v_chunksize) = \
                  calcBufferSize(self.rowsize, self.nrows, self.complevel)

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
