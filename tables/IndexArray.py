########################################################################
#
#       License: BSD
#       Created: June 02, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/IndexArray.py,v $
#       $Id: IndexArray.py,v 1.7 2004/08/10 07:48:51 falted Exp $
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

__version__ = "$Revision: 1.7 $"
# default version for IndexARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
from EArray import EArray
from VLArray import Atom, StringAtom
import hdf5Extension
import numarray
import numarray.strings as strings
import numarray.records as records

def calcChunksize(expectedrows, testmode=0):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimize the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized doing more experiments.

    """

    expKrows = expectedrows / 1000000.  # Multiples of one million

    if testmode:
        if expKrows < 0.0001: # expected rows < 1 hundred
            nelemslice = 10  # > 1/100th
            chunksize = 10
        elif expKrows < 0.001: # expected rows < 1 thousand
            nelemslice = 100  # > 1/10th
            chunksize = 50
        elif expKrows <= 0.01: # expected rows < 10 thousand
            nelemslice = 1000  # > 1/100th
            chunksize = 600
        else:
            raise ValueError, \
                  "expected rows cannot be larger than 10000 in test mode"
        #print "-->", (nelemslice, chunksize)
        return (nelemslice, chunksize)

    # expKrows < 0.01 is to few for indexing to represent a significant gain
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
#            nelemslice = 1500000 # > 1/1000th
        nelemslice = 1500000 # > 1/1000th
        #chunksize = 5000
        chunksize = 10000   # Experimental
    else:  # expected rows > 1 billion
        #nelemslice = 1000000 # 1/1000  # Better for small machines
        #chunksize = 1000
#             nelemslice = 1500000 # 1.5/1000  # Better for medium machines
#             chunksize = 3000
        nelemslice = 2000000 # 2/1000  # Better for big machines
        #chunksize = 10000
        chunksize = 10000  # Experimental

    #print "nelemslice, chunksize -->", (nelemslice, chunksize)
    return (nelemslice, chunksize)

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
    
    def __init__(self, parent = None, atom = None, title = "",
                 filters = None, expectedrows = 1000000,
                 testmode=0):
        """Create an IndexArray instance.

        Keyword arguments:

        parent -- The Index class from which this object will hang off

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
        self._v_parent = parent
        self._v_new_title = title
        self._v_new_filters = filters
        self._v_expectedrows = expectedrows
        self.testmode = testmode
        self.flavor = "NumArray"  # Needed by Array methods
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
        (self.nelemslice, self.chunksize) = \
                          calcChunksize(self._v_expectedrows,
                                        testmode=self.testmode)
        # The next is needed by hdf5Extension.Array._createEArray
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
        # Create the IndexArray
        self._createEArray("INDEXARRAY", self._v_new_title)
            
    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize,
         self.byteorder, chunksizes) = self._openArray()
        self.chunksize = chunksizes[1]  # Get the second dim
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

    def append(self, arr):
        """Append the object to this (enlargeable) object"""
        arr.shape = (1, arr.shape[0])
        self._append(arr)

    # This is coded in pyrex as well, but the improvement in speed is very
    # little. So, it's better to let _searchBin live here.
    def _searchBin(self, nrow, item):
        nelemslice = self.shape[1]
        hi = nelemslice   
        item1, item2 = item
        item1done = 0; item2done = 0
        chunksize = self.chunksize # Number of elements/chunksize
        niter = 1

        # First, look at the beginning of the slice (that could save lots of time)
        buffer = self._readSortedSlice(nrow, 0, chunksize)
        #print "buffer-->", buffer
        #buffer = xrange(0, chunksize)  # test  # 0.02 over 0.5 seg
        # Look for items at the beginning of sorted slices
        #print "item1, item2-->", repr(item1), repr(item2)
        result1 = self._bisect_left(buffer, item1, chunksize)
        if 0 <= result1 < chunksize:
            item1done = 1
        result2 = self._bisect_right(buffer, item2, chunksize)
        # print "item1done, item2done-->", item1done, item2done
        # print "result1, result2-->", result1, result2
        # print "chunksize-->", chunksize
        if 0 <= result2 < chunksize:
            item2done = 1
        if item1done and item2done:
            # print "done 1"
            return (result1, result2, niter)

        # Then, look for items at the end of the sorted slice
        buffer = self._readSortedSlice(nrow, hi-chunksize, hi)
        #buffer = xrange(hi-chunksize, hi)  # test
        niter += 1
        # print "item1done, item2done-->", item1done, item2done
        if not item1done:
            result1 = self._bisect_left(buffer, item1, chunksize)
            if 0 < result1 <= chunksize:
                item1done = 1
                result1 = hi - chunksize + result1
                # print "item1done, item2done-->", item1done, item2done
        if not item2done:
            result2 = self._bisect_right(buffer, item2, chunksize)
            if 0 < result2 <= chunksize:
                item2done = 1
                result2 = hi - chunksize + result2
        if item1done and item2done:
            # print "done 2"
            return (result1, result2, niter)
        # print "item1done, item2done-->", item1done, item2done
    
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        if not item1done:
            lo = 0
            hi = nelemslice
            beginning = 1
            result1 = 1  # a number different from 0
            while beginning and result1 != 0:
                (result1, beginning, iter) = \
                          self._interSearch_left(nrow, chunksize,
                                                 item1, lo, hi)
                tmpresult1 = result1
                niter = niter + iter
                if result1 == hi:  # The item is completely at right
                    break
                else:
                    hi = result1        # one chunk to the left
                    lo = hi - chunksize  
                    #print "lo, hi, beginning-->", lo, hi, beginning
            result1 = tmpresult1
        # Lookup in the middle of slice for item1
        if not item2done:
            lo = 0
            hi = nelemslice
            ending = 1
            result2 = 1  # a number different from 0
            while ending and result2 != nelemslice:
                (result2, ending, iter) = \
                          self._interSearch_right(nrow, chunksize,
                                                  item2, lo, hi)
                tmpresult2 = result2
                niter = niter + iter
                if result2 == lo:  # The item is completely at left
                    break
                else:
                    hi = result2 + chunksize      # one chunk to the right
                    lo = result2
                    #print "lo, hi, ending-->", lo, hi, ending
            result2 = tmpresult2
            niter = niter + iter
        #print "done 3"
        return (result1, result2, niter)

#     def searchBin(self, item):
#         """Do a binary search in this index for an item"""
#         ntotaliter = 0  # for counting the number of reads on each
#         inflimit = []; suplimit = []
#         bufsize = self.chunksize # number of elements/chunksize
#         self._initSortedSlice(bufsize)
#         for i in xrange(self.nrows):
#             (result1, result2, niter) = self._searchBin(i, item)
#             inflimit.append(result1)
#             suplimit.append(result2)
#             ntotaliter = ntotaliter + niter
#         self._destroySortedSlice()
#         return (inflimit, suplimit, ntotaliter)

    def _close(self):
        """Close this object and exit"""
        # First, flush the buffers:
        self.flush()
        # Delete back references
        del self._v_parent
        del self._v_file
        del self.type
        del self.atom
        del self.filters
        #self.__dict__.clear()

    def __str__(self):
        "A compact representation of this class"
        return "IndexArray(path=%s)" % \
               (self._v_parent._g_join(self.name))

    def __repr__(self):
        """A verbose representation of this class"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  nelemslice = %s
  chunksize = %s
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.nelemslice, self.chunksize, self.byteorder)
