########################################################################
#
#       License: BSD
#       Created: December 15, 2003
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/EArray.py,v $
#       $Id: EArray.py,v 1.19 2004/09/22 17:13:04 falted Exp $
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

__version__ = "$Revision: 1.19 $"
# default version for EARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
from Array import Array
from VLArray import Atom
from utils import convertIntoNA, processRangeRead
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

        assert isinstance(self.atom, Atom), "The object passed to the IndexArray constructor must be a descendent of the Atom class."
        assert isinstance(self.atom.shape, types.TupleType), "The Atom shape has to be a tuple for IndexArrays, and you passed a '%s' object." % (self.atom.shape)
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
        (self._v_buffersize, self._v_maxTuples, self._v_chunksize) = \
           self._calcBufferSize(self.atom, self.extdim, self._v_expectedrows,
                                self.filters.complevel)
        self.nrows = 0   # No rows initially
        self.itemsize = self.atom.itemsize
        self._createEArray("EARRAY", self._v_new_title)

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

    def _calcBufferSize(self, atom, extdim, expectedrows, compress):
        """Calculate the buffer size and the HDF5 chunk size.

        The logic to do that is based purely in experiments playing
        with different buffer sizes, chunksize and compression
        flag. It is obvious that using big buffers optimize the I/O
        speed when dealing with tables. This might (should) be further
        optimized doing more experiments.

        """

        rowsize = atom.atomsize()
        #bufmultfactor = int(1000 * 1.0)  # Optimum for Sorted objects
        # An 1.0 factor makes the lookup in sorted arrays to
        # decompress less, but, in exchange, the indexed dataset is
        # almost 5 times larger than with a 10.0 factor.
        # However, as the indexed arrays can be quite uncompressible
        # the size of the compressed sorted list is negligible when compared
        # against it.
        # The improvement in time (overall) is reduced (~5%)
        #bufmultfactor = int(1000 * 10.0) # Optimum for Index objects
        #bufmultfactor = int(1000 * 2) # Is a good choice too,
        # specially for very large tables and large available memory
        #bufmultfactor = int(1000 * 1) # Optimum for sorted object
        bufmultfactor = int(1000 * 1) # Optimum for sorted object
        
        rowsizeinfile = rowsize
        expectedfsizeinKb = (expectedrows * rowsizeinfile) / 1024

        if expectedfsizeinKb <= 100:
            # Values for files less than 100 KB of size
            buffersize = 5 * bufmultfactor
        elif (expectedfsizeinKb > 100 and
            expectedfsizeinKb <= 1000):
            # Values for files less than 1 MB of size
            buffersize = 20 * bufmultfactor
        elif (expectedfsizeinKb > 1000 and
              expectedfsizeinKb <= 20 * 1000):
            # Values for sizes between 1 MB and 20 MB
            buffersize = 40  * bufmultfactor
        elif (expectedfsizeinKb > 20 * 1000 and
              expectedfsizeinKb <= 200 * 1000):
            # Values for sizes between 20 MB and 200 MB
            buffersize = 50 * bufmultfactor
        else:  # Greater than 200 MB
            buffersize = 60 * bufmultfactor

        # Max Tuples to fill the buffer
        maxTuples = buffersize // rowsize
        chunksizes = list(atom.shape)
        # Check if at least 10 tuples fits in buffer
        if maxTuples > 10:
            # Yes. So the chunk sizes for the non-extendeable dims will be
            # unchanged
            chunksizes[extdim] = maxTuples // 10
        else:
            # No. reduce other dimensions until we get a proper chunksizes
            # shape
            chunksizes[extdim] = 1  # Only one row in extendeable dimension
            for j in range(len(chunksizes)):
                newrowsize = atom.itemsize
                for i in chunksizes[j+1:]:
                    newrowsize *= i
                maxTuples = buffersize // newrowsize
                if maxTuples > 10:
                    break
                chunksizes[j] = 1
            # Compute the chunksizes correctly for this j index
            chunksize = maxTuples // 10
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
        maxTuples = buffersize // newrowsize
        return (buffersize, maxTuples, chunksizes)

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize, self.byteorder,
         self._v_chunksize) = self._openArray()
        #print "chunksizes-->", self._v_chunksize
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
        # Compute the optimal maxTuples
        (self._v_buffersize, self._v_maxTuples, theoChunksize) = \
           self._calcBufferSize(self.atom, self.extdim, self.nrows,
                                self.filters.complevel)
        chunksize = self.atom.itemsize
        for i in self._v_chunksize:
            chunksize *= i
        self._v_maxTuples = self._v_buffersize // chunksize
        #print "maxTuples-->", self._v_maxTuples

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
        nbytes = self.itemsize
        for i in self.shape:
            nbytes*=i

        return (object, nbytes)

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
