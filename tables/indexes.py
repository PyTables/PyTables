########################################################################
#
#       License: BSD
#       Created: June 02, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /cvsroot/pytables/pytables/tables/indexes.py $
#       $Id$
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

__version__ = "$Revision$"
# default version for IndexARRAY objects
obversion = "1.0"    # initial version

import types, warnings, sys
from Array import Array
from EArray import EArray
from VLArray import Atom, StringAtom
import indexesExtension
import numarray
import numarray.strings as strings
import numarray.records as records
from bisect import bisect_left, bisect_right
from time import time


# The minimum row number in a column that can be indexed in tests
minRowIndex = 10

def calcChunksize(expectedrows, testmode=0):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimize the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized doing more experiments.

    """

    superblocksize = None
    blocksize = None

    if testmode:
        if expectedrows < minRowIndex*10:
            slicesize = 10
            chunksize = 5
        elif expectedrows < minRowIndex*100:
            slicesize = 100
            chunksize = 50
        elif expectedrows <= minRowIndex*1000:
            slicesize = 1000
            chunksize = 600
        else:
            raise ValueError, \
"expected rows cannot be larger than %s in test mode" % minRowIndex*1000
        if blocksize == None:
            blocksize = 2*slicesize
        if superblocksize == None:
            superblocksize = 2*blocksize
        print "superblocksize, blocksize, slicesize, chunksize:", \
              (superblocksize, blocksize, slicesize, chunksize)
        return (superblocksize, blocksize, slicesize, chunksize)

    expKrows = expectedrows / 1000000.  # Multiples of one million
    #print "expKrows:", expKrows

    # Hint: the slicesize should not pass 500 or 1000 thousand
    # That would make numarray to consume lots of memory for sorting
    # this slice.
    # In general, one should favor a small chunksize (100 ~ 1000) if one
    # wants to reduce the latency for indexed queries. However, keep in
    # mind that a very low value of chunksize for big datasets may
    # hurt the performance by requering the HDF5 to use a lot of memory
    # and CPU for its internal B-Tree
    if expKrows < 0.1: # expected rows < 100 thousand
        #chunksize = 1000            # best value
        #slicesize = 2*chunksize    #  "     "
        chunksize = 100              # best value
        slicesize = 100*chunksize    #  experimental
        #chunksize = 500             # experimental
        #slicesize = 2*chunksize    #      "
    elif expKrows < 1: # expected rows < 1 milion
        chunksize = 250             # experiment
        slicesize = 200*chunksize   # experiment
        #slicesize = 2*chunksize
        #chunksize = 2000           # experimental
        #slicesize = 2*chunksize   #    "
    elif expKrows < 10:  # expected rows < 10 milion
        #chunksize = 100   # Create a lot of slices. For tests purposes only!
        #slicesize = 2*chunksize
        #chunksize = 100   # Create a lot of slices. For tests purposes only!
        #slicesize = 10*chunksize
        #chunksize = 50
        #slicesize = 5000*chunksize
        # Above values lends to 1.04 ms (28 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 100
        #slicesize = 100*chunksize
        # Above values lends to 2.35 ms (21.3 MB indexes). No Compr
        # Above values lends to 2.88 ms (11.9 MB indexes). Compr
        #chunksize = 100
        #slicesize = 250*chunksize
        # Above values lends to 1.65 ms (21 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 100
        #slicesize = 1000*chunksize
        # Above values lends to 1.16 ms (23 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 100
        #slicesize = 2500*chunksize
        # Above values lends to 1.01 ms (26 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 100
        #slicesize = 5000*chunksize
        # Above values lends to 1.01 ms (31 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 200
        #slicesize = 1000*chunksize
        # Above values lends to 1.05 ms (24 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 200
        #slicesize = 2500*chunksize
        # Above values lends to 0.99 ms (30 MB indexes). No Compr
        # Above values lends to 1.27 ms (21 MB indexes). Compr
        #chunksize = 250         #  *** consumes very little memory ***
        #slicesize = 200*chunksize
        # Above values lends to 1.33 ms (21.0 MB indexes). No Compr
        # Above values lends to 1.74 ms (10.6 MB indexes). Compr
        #chunksize = 250   # *** good **** takes much less memory
        #slicesize = 400*chunksize  # Very good balance cpu/memory
        # Above values lends to XXX (0.96) ms (21.9 MB indexes). No Compr
        # Above values lends to XXX (1.29) ms (12.2 MB indexes). Compr
        #chunksize = 250   # *** good **** takes much less memory
        #slicesize = 500*chunksize  # Very good balance cpu/memory
        # Above values lends to 1.12 ms (22.4 MB indexes). No Compr
        # Above values lends to 1.46 ms (12.7 MB indexes). Compr
        chunksize = 250      # *** very good **** takes less memory
        slicesize = 1000*chunksize  # Good balance cpu/memory
        # Above values lends to 1.05 (0.87) ms (24.8 MB indexes). No Compr
        # Above values lends to 1.34 (1.15) ms (15.3 MB indexes). Compr
        #chunksize = 250     # *** optimal **** but takes memory
        #slicesize = 2000*chunksize
        # Above values lends to 0.99 (0.81) ms (30 MB indexes). No Compr
        # Above values lends to 1.28 (1.10) ms (20 MB indexes). Compr
        #chunksize = 256     # uses last row index. Use only as test
        #slicesize = 1000*chunksize
        # Above values lends to (0.99) ms (XXX MB indexes). No Compr
        # Above values lends to (1.29) ms (12.8 MB indexes). Compr
        #chunksize = 256     # uses last row index. Use only as test
        #slicesize = 1024*chunksize
        # Above values lends to (0.99) ms (20.6 MB indexes). No Compr
        # Above values lends to (1.29) ms (13.1 MB indexes). Compr
        #chunksize = 256     # uses last row index. Use only as test
        #slicesize = 2000*chunksize
        # Above values lends to 1.63 (0.97) ms (19.6 MB indexes). No Compr
        # Above values lends to 1.90 ms (14.8 MB indexes). Compr
        #chunksize = 300   # *** uses last row index ***
        #slicesize = 1000*chunksize
        # Above values lends to 1.47 ms (24 MB indexes). No Compr
        # Above values lends to XXX ms (XXX MB indexes). Compr
        #chunksize = 500
        #slicesize = 250*chunksize
        # Above values lends to 1.13 ms (21 MB indexes). No Compr
        # Above values lends to 5.2 ms (14 MB indexes). Compr
        #chunksize = 500
        #slicesize = 500*chunksize
        # Above values lends to 1.05 ms (24 MB indexes). No Compr
        # Above values lends to 5.2 ms (14 MB indexes). Compr
        #chunksize = 500
        #slicesize = 1000*chunksize
        # Above values lends to 1.00 ms (29.1 MB indexes). No Compr
        # Above values lends to 1.30 ms (19.6 MB indexes). Compr
        #chunksize = 1000
        #slicesize = 250*chunksize
        # Above values lends to XXX ms (24 MB indexes). No Compr
        # Above values lends to 1.11 ms (14 MB indexes). Compr
        #chunksize = 1000
        #slicesize = 500*chunksize
        # Above values lends to 1.02 ms (29 MB indexes). No Compr
        # Above values lends to  XXX ms (19 MB indexes). Compr
        #chunksize = 1000
        #slicesize = 100*chunksize
        # Above values lends to 1.25 ms (21.2 MB indexes). Compr
        # Above values lends to 1.78 ms (11.4 MB indexes). Compr
        #chunksize = 1000
        #slicesize = 250*chunksize
        # Above values lends to 5.0 ms
        #chunksize = 1000
        #slicesize = 1000*chunksize
        # Above values lends to 3.9 ms but double the time to index
        #chunksize = 1500   # *** all is kept in last row index ***
        #slicesize = 1000*chunksize
    elif expKrows < 100: # expected rows < 100 milions
        # From simulations it is evident that a small chunksize uses more CPU,
        # possibly due to lookups in HDF5 BTree. However, it reduces memory
        # usage, as well as I/O. Hence, in fast machines it would be effective
        # to favour relatively small chunksizes.
        #chunksize = 256  # simulations. High CPU usage. Much less I/O
        #slicesize = 32*chunksize     # simulations
        # Above values lends to 11.85 ms (198.4 MB indexes). No Compr
        #chunksize = 256  # simulations.
        #slicesize = 48*chunksize     # simulations
        # Above values lends to 8.55 ms (198.4 MB indexes). No Compr
        #chunksize = 256  # simulations. High CPU
        #slicesize = 64*chunksize     # simulations
        # Above values lends to 6.84 ms (198.6 MB indexes). No Compr
        # Above values lends to 9.55 ms (109.0 MB indexes). Compr
        #chunksize = 200  # simulations. High CPU usage. Much less I/O
        #slicesize = 50*chunksize     # simulations
        # Above values lends to 9.85 ms (201.7 MB indexes). No Compr
        #chunksize = 500              # simulations
        #slicesize = 20*chunksize     # simulations
        # Above values lends to 10.53 ms (195.9 MB indexes). No Compr
        #chunksize = 512              # simulations
        #slicesize = 20*chunksize     # simulations
        # Above values lends to 10.57 ms (194.8 MB indexes). No Compr
        # Above values lends to 16.86 ms (106.7 MB indexes). Compr
        #chunksize = 1000              # simulations. less I/O. fits on cache
        #slicesize = 10*chunksize     # simulations
        # Above values lends to 11.63 ms (193.0 MB indexes). No Compr
        #chunksize = 5000              # simulations.lots of I/O
        #slicesize = 2*chunksize     # simulations
        # Above values lends to 30.7 ms (191.5 MB indexes). No Compr
        #chunksize = 100              # experiment
        #slicesize = 5000*chunksize     # experiment
        # Above values lends to 2.61 ms (220 MB indexes). No Compr
        # Above values lends to 3.00 ms (141 MB indexes). Compr
        # This takes long time to index
        #chunksize = 250              # experiment  *** very good ***
        #slicesize = 2000*chunksize  # but takes long to time to index
        # Above values lends to 2.60 (1.43) ms (209 MB indexes). No Compr
        # Above values lends to 3.04 (1.87) ms (126 MB indexes). Compr
        #chunksize = 250              # experiment
        #slicesize = 1500*chunksize     # experiment
        # Above values lends to 3.30 ms (202 MB indexes). No Compr
        # Above values lends to 3.04 ms (126 MB indexes). Compr
        chunksize = 500              # ******best values*******
        slicesize = 1000*chunksize  # takes reasonable time to index
        # Above values lends to 2.66 (1.48) ms (205 MB indexes). No Compr
        # Above values lends to 3.16 (1.98) ms (120 MB indexes). Compr
        #chunksize = 500              # experiment
        #slicesize = 500*chunksize     # experiment
        # Above values lends to XXX ms (XX MB indexes). No Compr
        # Above values lends to 4.01 ms (111 MB indexes). Compr
        #chunksize = 750              # experiment
        #slicesize = 750*chunksize     # experiment
        # Above values lends to XXX (1.39*) ms (XX MB indexes). No Compr
        # Above values lends to 3.64 ms (114 MB indexes). Compr
        #chunksize = 1000
        #slicesize = 100*chunksize
        # Above values lends to XXX ms (XXX MB indexes). No Compr
        # Above values lends to 7.25 ms (102 MB indexes). Compr
        #chunksize = 1000              # experiment
        #slicesize = 500*chunksize     # experiment
        # Above values lends to XXX ms (XX MB indexes). No Compr
        # Above values lends to 3.47 ms (116 MB indexes). Compr
        #chunksize = 1500              # experiment
        #slicesize = 250*chunksize     # experiment
        # Above values lends to XXX ms (XX MB indexes). No Compr
        # Above values lends to 4.68 ms (108 MB indexes). Compr
        #chunksize = 1500              # experiment
        #slicesize = 500*chunksize     # experiment
        # Above values lends to XXX ms (XX MB indexes). No Compr
        # Above values lends to 3.78 ms (117 MB indexes). Compr
        #chunksize = 5000
        #slicesize = 200*chunksize
        # Above values lends to XXX ms (XX MB indexes). No Compr
        # Above values lends to 4.72 ms (121 MB indexes). Compr
    elif expKrows < 1000: # expected rows < 1000 millions
        #chunksize = 750              # experiment
        chunksize = 1000              # experiment
        slicesize = 1000*chunksize     # experiment
        blocksize = 1000*slicesize    # experiment
    elif expKrows < 10*1000: # expected rows < 10 (american) billions
        #chunksize = 1000              # experiment
        chunksize = 2000              # experiment
        slicesize = 1000*chunksize     # experiment
        blocksize = 1000*slicesize    # experiment
    elif expKrows < 100*1000: # expected rows < 100 (american) billions
        #chunksize = 1250              # experiment
        chunksize = 4000               # experiment
        slicesize = 1000*chunksize     # experiment
        blocksize = 10000*slicesize    # experiment
    else:  # expected rows >= 1 (american) trillion (perhaps by year 2010
           # this will be useful, who knows...)
        chunksize = 7500              # experiment
        slicesize = 1000*chunksize     # experiment
        blocksize = 100000*slicesize    # experiment

    # The defaults for blocksize & superblocksize
    if blocksize == None:
        blocksize = 1000*slicesize    # experiment
    if superblocksize == None:
        superblocksize = 1000*blocksize  # experiment

#     print "superblocksize, blocksize, slicesize, chunksize:", \
#           (superblocksize, blocksize, slicesize, chunksize)
    return (superblocksize, blocksize, slicesize, chunksize)


# Declarations for inheriting
class CacheArray(EArray, indexesExtension.CacheArray):
    """Container for keeping index caches of 1st and 2nd level."""

    # Class identifier.
    _c_classId = 'CACHEARRAY'


class LastRowArray(Array, indexesExtension.LastRowArray):
    """Container for keeping sorted and indices values of last rows of an index."""

    # Class identifier.
    _c_classId = 'LASTROWARRAY'


class IndexArray(EArray, indexesExtension.IndexArray):
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
        slicesize -- The number of elements per slice.
        chunksize -- The HDF5 chunksize for the slice dimension (the 1).

    """

    # Class identifier.
    _c_classId = 'INDEXARRAY'

    def __init__(self, parentNode, name,
                 atom=None, title="",
                 filters=None,
                 testmode=False,
                 expectedrows=0):
        """Create an IndexArray instance.

        Keyword arguments:

        parentNode -- The Index class from which this object will hang off.

        name -- The name of this node in its parent group (a string).

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. Only scalar atoms are
            supported.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- Represents an user estimate about the number
            of elements to index.

        """
        self.testmode = testmode
        """Enables test mode for index chunk size calculation."""
        self.nblocks = None
        """The number of blocks."""
        self.chunksize = None
        """The HDF5 chunksize for the slice dimension (the second)."""
        self.slicesize = None
        """The number of elements per slice."""
        self.superblocksize = None
        """The maximum number of elements that can be optimized."""
        self.blocksize = None
        """The maximum number of elements in a block."""
        self.bufferl = None
        """Buffer for reading chunks in sorted array in extension."""
        self.arrAbs = None
        """Buffer for reading indexes (absolute addresses) in extension."""
        self.coords = None
        """Buffer for reading coordenates (absolute addresses) in extension."""
        if atom is not None:
            (self.superblocksize, self.blocksize, self.slicesize, self.chunksize) = (
                calcChunksize(expectedrows, testmode))
        # Index creation is never logged.
        super(IndexArray, self).__init__(
            parentNode, name, atom, title, filters, expectedrows, log=False)


    def _g_create(self):
        objectId = super(IndexArray, self)._g_create()
        # The superblocksize & blocksize will be saved as (pickled) attributes
        self.attrs.superblocksize = self.superblocksize
        self.attrs.blocksize = self.blocksize
        assert self.extdim == 0, "computed extendable dimension is wrong"
        assert self.shape == (0, self.slicesize), "invalid shape"
        assert self._v_chunksize == (1, self.chunksize), "invalid chunk size"
        return objectId


    def _calcTuplesAndChunks(self, atom, extdim, expectedrows, compress):
        return (0, (1, self.chunksize))  # (_v_maxTuples, _v_chunksize)


    def _createEArray(self, title):
        # The shape of the index array needs to be fixed before creating it.
        # Admitted, this is a bit too much convoluted :-(
        self.shape = (0, self.slicesize)
        self._v_objectID = super(IndexArray, self)._createEArray(title)
        return self._v_objectID


    def _g_postInitHook(self):
        # initialize some index buffers
        self._startl = numarray.array(None, shape=(2,), type=numarray.Int64)
        self._stopl = numarray.array(None, shape=(2,), type=numarray.Int64)
        self._stepl = numarray.array([1,1], shape=(2,), type=numarray.Int64)
        # Set ``slicesize`` and ``chunksize`` when opening an existing node;
        # otherwise, they are already set.
        if not self._v_new:
            self.superblocksize = self.attrs.superblocksize
            self.blocksize = self.attrs.blocksize
            self.slicesize = self.shape[1]
            self.chunksize = self._v_chunksize[1]
        super(IndexArray, self)._g_postInitHook()


    def append(self, arr):
        """Append the object to this (enlargeable) object"""
        arr.shape = (1, arr.shape[0])
        self._append(arr)


    # This version of searchBin uses both rangeValues (1st level) and
    # bounds (2nd level) caches. This is more than 40% faster than the
    # version that only uses the 1st cache.
    def _searchBin(self, nrow, item):
        item1, item2 = item
        result1 = -1; result2 = -1
        hi = self.slicesize
        rangeValues = self._v_parent.rvcache
        #t1=time()
        # First, look at the beginning of the slice
        #begin, end = rangeValues[nrow]  # this is slower
        begin = rangeValues[nrow,0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
        if item2 < begin:
            result2 = 0
        if result1 >=0 and result2 >= 0:
            return (result1, result2)
        # Then, look for items at the end of the sorted slice
        end = rangeValues[nrow,1]
        if result1 < 0:
            if item1 > end:
                result1 = hi
        if result2 < 0:
            if item2 >= end:
                result2 = hi
        if result1 >= 0 and result2 >= 0:
            return (result1, result2)
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        chunksize = self.chunksize # Number of elements/chunksize
        nchunk = -1
        if self.bcache:
            bounds = self.boundscache[nrow]
        else:
            bounds = self._v_parent.bounds[nrow]
        if result1 < 0:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect_left(bounds, item1)
            chunk = self._readSortedSlice(nrow, chunksize*nchunk,
                                          chunksize*(nchunk+1))
            result1 = self._bisect_left(chunk, item1, chunksize)
            result1 += chunksize*nchunk
        # Lookup in the middle of slice for item2
        if result2 < 0:
            # Search the appropriate chunk in bounds cache
            nchunk2 = bisect_right(bounds, item2)
            if nchunk2 <> nchunk:
                chunk = self._readSortedSlice(nrow, chunksize*nchunk2,
                                              chunksize*(nchunk2+1))
            result2 = self._bisect_right(chunk, item2, chunksize)
            result2 += chunksize*nchunk2
        return (result1, result2)


    # This version of searchBin only uses the rangeValues (1st cache)
    def _g_searchBin(self, nrow, item, rangeValues):
        item1, item2 = item
        slicesize = self.shape[1]
        hi = slicesize
        item1done = 0; item2done = 0
        chunksize = self.chunksize # Number of elements/chunksize # change here
        niter = 0

        # First, look at the beginning of the slice
        # (that could save lots of time)
        #begin = self[nrow,0]
        #begin = self._v_parent.rangeValues[nrow,0]
        begin = rangeValues[nrow,0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
            item1done = 1
        if item2 < begin:
            result2 = 0
            item2done = 1
        if item1done and item2done:
            return (result1, result2, niter)

        # Then, look for items at the end of the sorted slice
        end = rangeValues[nrow,1]
        if not item1done:
            if item1 > end:
                result1 = hi
                item1done = 1
        if not item2done:
            if item2 >= end:
                result2 = hi
                item2done = 1
        if item1done and item2done:
            return (result1, result2, niter)

        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        if not item1done:
            lo = 0
            hi = slicesize
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
            result1 = tmpresult1
        # Lookup in the middle of slice for item2
        if not item2done:
            lo = 0
            hi = slicesize
            ending = 1
            result2 = 1  # a number different from 0
            while ending and result2 != slicesize:
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
            result2 = tmpresult2
            niter = niter + iter
        return (result1, result2, niter)


    # This version of searchBin does not use caches (1st or 2nd) at all
    # This is coded in pyrex as well, but the improvement in speed is very
    # little. So, it's better to let _searchBin live here.
    def _searchBinStd(self, nrow, item):
        slicesize = self.shape[1]
        hi = slicesize
        item1, item2 = item
        item1done = 0; item2done = 0
        chunksize = self.chunksize # Number of elements/chunksize # change here
        niter = 1

        # First, look at the beginning of the slice (that could save lots of time)
        buffer = self._readSortedSlice(nrow, 0, chunksize)
        #buffer = xrange(0, chunksize)  # test  # 0.02 over 0.5 seg
        # Look for items at the beginning of sorted slices
        result1 = self._bisect_left(buffer, item1, chunksize)
        if 0 <= result1 < chunksize:
            item1done = 1
        result2 = self._bisect_right(buffer, item2, chunksize)
        if 0 <= result2 < chunksize:
            item2done = 1
        if item1done and item2done:
            return (result1, result2)

        # Then, look for items at the end of the sorted slice
        buffer = self._readSortedSlice(nrow, hi-chunksize, hi)
        #buffer = xrange(hi-chunksize, hi)  # test

        niter += 1
        if not item1done:
            result1 = self._bisect_left(buffer, item1, chunksize)
            if 0 < result1 <= chunksize:
                item1done = 1
                result1 = hi - chunksize + result1
        if not item2done:
            result2 = self._bisect_right(buffer, item2, chunksize)
            if 0 < result2 <= chunksize:
                item2done = 1
                result2 = hi - chunksize + result2
        if item1done and item2done:
            return (result1, result2)

        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        if not item1done:
            lo = 0
            hi = slicesize
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
            result1 = tmpresult1
        # Lookup in the middle of slice for item2
        if not item2done:
            lo = 0
            hi = slicesize
            ending = 1
            result2 = 1  # a number different from 0
            while ending and result2 != slicesize:
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
            result2 = tmpresult2
            niter = niter + iter
        return (result1, result2)


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
  slicesize = %s
  chunksize = %s
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.slicesize, self.chunksize, self.byteorder)
