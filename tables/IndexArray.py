########################################################################
#
#       License: BSD
#       Created: June 02, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/IndexArray.py,v $
#       $Id: IndexArray.py,v 1.4 2004/07/27 12:18:06 falted Exp $
#
########################################################################

"""Here is defined the IndexArray class.

See IndexArray class docstring for more info.

Classes:

    IndexArray

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.4 $"
# default version for IndexARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
from EArray import EArray
from VLArray import Atom, StringAtom
import hdf5Extension
import numarray
import numarray.strings as strings
import numarray.records as records

class IndexArray(EArray, hdf5Extension.IndexArray, object):
    """Represent the index (sorted or reverse index) dataset in HDF5 file.

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

      Common to all EArray's:
        read(start, stop, step)
        iterrows(start, stop, step)
        append(object)

        
    Instance variables:

      Common to all EArray's:

        type -- The type class for the array.
        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        flavor -- The flavor of this object.
        nrow -- On iterators, this is the index of the row currently
            dealed with.

      Specific of IndexArray:
        extdim -- The enlargeable dimension (always the first, or 0).
        nrows -- The number of slices in index.
        nelemslice -- The number of elements per slice.
        chunksize -- The HDF5 chunksize for the slice dimension (the 1).
            

    """
    
    def __init__(self, atom = None, title = "",
                 filters = None, expectedrows = 1000000):
        """Create an IndexArray instance.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. Only scalar atoms are
            supported.
        
        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- Represents an user estimate about the number
            of elements to index. If not provided, the default
            value is 1000000 slices.

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
            
    def _calcChunksize(self):
        """Calculate the HDF5 chunk size for index and sorted arrays.

        The logic to do that is based purely in experiments playing
        with different chunksizes and compression flag. It is obvious
        that using big chunks optimize the I/O speed, but if they are
        too large, the uncompressor takes too much time. This might
        (should) be further optimized doing more experiments.

        """

        expKrows = self._v_expectedrows / 1000000.  # Multiples of one million
        if expKrows < 0.01: # expected rows < 10 thousand
            nelemslice = 1000  # > 1/100th
            chunksize = 1000
        elif expKrows < 0.1: # expected rows < 100 thousand
            nelemslice = 10000  # > 1/10th
            chunksize = 1000
            #chunksize = 2000  # Experimental
        elif expKrows < 1: # expected rows < 1 milion
            nelemslice = 100000  # > 1/10th
            #chunksize = 1000
            chunksize = 5000  # Experimental
        elif expKrows < 10:  # expected rows < 10 milion
            nelemslice = 500000  # > 1/20th
            #chunksize = 1000
            chunksize = 5000  # Experimental (best)
            #chunksize = 10000  # Experimental
        elif expKrows < 100: # expected rows < 100 milions
            nelemslice = 1000000 # > 6/100th
            #chunksize = 2000
            chunksize = 10000  # Experimental
        elif expKrows < 1000: # expected rows < 1000 millions
#             nelemslice = 1000000 # > 1/1000th
#             chunksize = 1500
            nelemslice = 1500000 # > 1/1000th
            #chunksize = 4000
            chunksize = 10000   # Experimental
        else:  # expected rows > 1 billion
            #nelemslice = 1000000 # 1/1000  # Better for small machines
            #chunksize = 1000
#             nelemslice = 1500000 # 1.5/1000  # Better for medium machines
#             chunksize = 3000
            nelemslice = 2000000 # 2/1000  # Better for big machines
            #chunksize = 10000
            chunksize = 10000  # Experimental

        #chunksize *= 5  # Best value
        #chunksize = nelemslice // 150
        #print "nelemslice, chunksize -->", (nelemslice, chunksize)
        return (nelemslice, chunksize)

    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        global obversion

        assert isinstance(self.atom, Atom), "The object passed to the IndexArray constructor must be a descendent of the Atom class."
        assert self.atom.shape == 1, "Only scalar columns can be indexed."
        # Version, type, shape, flavor, byteorder
        self._v_version = obversion
        self.type = self.atom.type
        if self.type == "CharType" or isinstance(self.type, records.Char):
            self.byteorder = "non-relevant"
        else:
            # Only support for creating objects in system byteorder
            self.byteorder  = sys.byteorder
        # Compute the optimal chunksize
        (self.nelemslice, self.chunksize) = self._calcChunksize()
        self._v_chunksize = (1, self.chunksize)
        self.nrows = 0   # No rows initially
        self.itemsize = self.atom.itemsize
        self.rowsize = self.atom.atomsize() * self.nelemslice
        self.shape = (0, self.nelemslice)
        
        # extdim computation
        self.extdim = 0
        # Compute the optimal maxTuples
        # Ten chunks for each buffer would be enough for IndexArray objects
        # This is really necessary??
        self._v_maxTuples = 10  
        #print "dims-->", self.shape
        #print "chunk dims-->", self._v_chunksize
        # Create the IndexArray
        self._createEArray("INDEXARRAY", self._v_new_title)
            
    def append(self, arr):
        """Append the object to this (enlargeable) object"""
        arr.shape = (1, arr.shape[0])
        self._append(arr)

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize, self.byteorder,
         self._v_chunksize) = self._openArray()
        self.chunksize = self._v_chunksize[1]
        # Post-condition
        assert self.extdim == 0, "extdim != 0: this should never happen!"
        self.nelemslice = self.shape[1] 
        # Create the atom instance. Not for strings yet!
        if str(self.type) == "CharType":
            self.atom = StringAtom(shape=1, length=self.itemsize)
        else:
            self.atom = Atom(dtype=self.type, shape=1)
        # Compute the rowsize for each element
        self.rowsize = self.atom.atomsize() * self.nelemslice
        # nrows in this instance
        self.nrows = self.shape[0]
        # Compute the optimal maxTuples
        # Ten chunks for each buffer would be enough for IndexArray objects
        # This is really necessary??
        self._v_maxTuples = 10  

#     def searchBin(self, item):
#         """Do a binary search in this index for an item"""
#         ntotaliter = 0  # for counting the number of reads on each
#         inflimit = []; suplimit = []
#         bufsize = self._v_chunksize[1] # number of elements/chunksize
#         self._initSortedSlice(bufsize)
#         for i in xrange(self.nrows):
#             (result1, result2, niter) = self._searchBin(i, item)
#             inflimit.append(result1)
#             suplimit.append(result2)
#             ntotaliter = ntotaliter + niter
#         self._destroySortedSlice()
#         return (inflimit, suplimit, ntotaliter)

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  nelemslice = %s
  chunksize = %s
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.nelemslice, self.chunksize, self.byteorder)
