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
#obversion = "1.0"    # Initial version for PyTables 1.x series.
obversion = "2.0"    # Version of indexes in PyTables Pro 2.x series.
                     # Added new attributes for keeping optimization params.

import types
import warnings
import sys
from bisect import bisect_left, bisect_right
from time import time
import math

import numpy

from tables.node import NotLoggedMixin
from tables.array import Array
from tables.earray import EArray
from tables import indexesExtension

# Hints for chunk/slice/block/superblock computations:
# - The slicesize should not exceed 500 MB.  That would make the
#   sorting algorithms to consume up to 1 GB of memory.
# - In general, one should favor a small chunksize ( < 128 KB) if
#   one wants to reduce the latency for indexed queries. However,
#   keep in mind that a very low value of chunksize for big
#   datasets may hurt the performance by requering the HDF5 to use
#   a lot of memory and CPU for its internal B-Tree.

def csformula(nrows):
    """Return the fitted chunksize (a float value) for nrows."""
    # This formula has been computed using two points:
    # 2**12 = m * 2**(n + log10(10**6))
    # 2**15 = m * 2**(n + log10(10**9))
    # where 2**12 and 2**15 are reasonable values for chunksizes for indexes
    # with 10**6 and 10**9 elements respectively.
    # Yes, return a floating point number!
    return 64 * 2**math.log10(nrows)


def computechunksize(expectedrows):
    """Get the optimum chunksize based on expectedrows."""

    # Only four zones for varying chunksizes here:
    # 1. rows < 10**7
    # 2. 10**7 <= rows < 10**8
    # 3. 10**8 <= rows < 10**9
    # 2. rows > 10**9
    zone = int(math.log10(expectedrows))
    if zone < 6:
        zone = 6
    elif zone > 9:
        zone = 9
    nrows = 10**zone
    return int(csformula(nrows))


def computeslicesize(expectedrows, memlevel):
    """Get the optimum slicesize based on expectedrows and memorylevel."""

    # Protection against creating too small slices (there will be no
    # protection for creating too large ones!).
    if expectedrows < 10**6:
        expectedrows = 10**6
    # First, the optimum chunksize
    cs = csformula(expectedrows)
    # Now the slicesize
    ss = cs * memlevel**2 * 16  #XXX replace by MEMORY_FACTOR
    return int(ss)


def computeblocksize(expectedrows, compoundsize):
    """Calculate the optimum number of superblocks made from compounds blocks.

    This is useful for computing the sizes of both blocks and
    superblocks (using the PyTables terminology for blocks in indexes).
    """

    # Start the split when more than 3 compound blocks fits in expected rows
    nblocks = expectedrows/(compoundsize*3)
    if nblocks == 0:
        # Protection against large compoundsize blocks
        nblocks = expectedrows/compoundsize
        # Check again
        if nblocks == 0:
            nblocks = 1
    elif nblocks > 1000:
        # Protection against too large number of expected rows
        nblocks = 1000
    return compoundsize * nblocks


def calcChunksize(expectedrows, optlevel, testmode):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimizes the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized by doing more experiments.

    An additional refinement for the output parameters is introduced
    by specifying an optimization ``optlevel``.

    """

    debug = False
    #debug = True  # Uncomment this for debugging purposes

    # Decode memlevel & optlevel
    memlevel = optlevel // 10 + 1
    shufflelevel = optlevel - (memlevel-1) * 10
    if debug:
        print "memlevel, shufflelevel-->", memlevel, shufflelevel

    if testmode:
        if 0 <= shufflelevel < 9:
            boost = (shufflelevel % 3) * 2 + 1   # 1, 3, 5
        elif shufflelevel == 9:
            boost = 4
        chunksize = 3 * boost # a very small number here is useful
                              # for testing the optimitzation levels
                              # more exhaustively (2 is the bare minimum for
                              # tests to work!)
        # slicesize should be at least twice as bigger than chunksize
        slicesize = chunksize * boost * 2
        blocksize = 4*slicesize
        superblocksize = 4*blocksize
        sizes = (superblocksize, blocksize, slicesize, chunksize)
        if debug:
            print "superblocksize, blocksize, slicesize, chunksize:", \
                  sizes
        return sizes

    expMrows = expectedrows / 1000000.  # Multiples of one million

    chunksize = computechunksize(expectedrows)
    slicesize = computeslicesize(expectedrows, memlevel)
    blocksize = computeblocksize(expectedrows, slicesize)
    superblocksize = computeblocksize(expectedrows, blocksize)

    # The size for different blocks information
    sizes = (superblocksize, blocksize, slicesize, chunksize)
    if debug:
        print "superblocksize, blocksize, slicesize, chunksize:", sizes
    return sizes


# Declarations for inheriting
class CacheArray(NotLoggedMixin, EArray, indexesExtension.CacheArray):
    """Container for keeping index caches of 1st and 2nd level."""

    # Class identifier.
    _c_classId = 'CACHEARRAY'


class LastRowArray(NotLoggedMixin, Array, indexesExtension.LastRowArray):
    """Container for keeping sorted and indices values of last rows of an index."""

    # Class identifier.
    _c_classId = 'LASTROWARRAY'


class IndexArray(NotLoggedMixin, EArray, indexesExtension.IndexArray):
    """Represent the index (sorted or reverse index) dataset in HDF5 file.

    All NumPy typecodes are supported except for complex datatypes.

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
        nrow -- On iterators, this is the index of the row currently
            dealed with.

      Specific of IndexArray:
        extdim -- The enlargeable dimension (always the first, or 0).
        nrows -- The number of slices in index.
        slicesize -- The number of elements per slice.
        chunksize -- The HDF5 chunksize for the slice dimension (the second, or 1).

    """

    # Class identifier.
    _c_classId = 'INDEXARRAY'

    def __init__(self, parentNode, name,
                 atom=None, title="",
                 filters=None, optlevel=0,
                 testmode=False, expectedrows=None,
                 byteorder=None):
        """Create an IndexArray instance.

        Keyword arguments:

        parentNode -- The Index class from which this object will hang off.

        name -- The name of this node in its parent group (a string).

        atom -- An Atom object representing the shape and type of the
            atomic objects to be saved. Only scalar atoms are
            supported.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        optlevel -- The optimization level for the creation of this
            index.

        testmode -- Useful for selecting very small blocksizes in
            tests.

        expectedrows -- Represents an user estimate about the number
            of elements to index.

        byteorder -- The byteroder of the data on-disk.

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
        self.optlevel = optlevel
        """The optlevel passed to the optimize index function."""
        if atom is not None:
            sizes = calcChunksize(expectedrows, optlevel, testmode)
            (self.superblocksize, self.blocksize,
             self.slicesize, self.chunksize) = sizes
            # The shape and chunkshape needs to be fixed here
            shape = (0, self.slicesize)
            chunkshape = (1, self.chunksize)
        else:
            # The shape and chunkshape will be read from disk later on
            shape = None
            chunkshape = None

        super(IndexArray, self).__init__(
            parentNode, name, atom, shape, title, filters, expectedrows,
            chunkshape, byteorder)


    def _g_create(self):
        objectId = super(IndexArray, self)._g_create()
        assert self.extdim == 0, "computed extendable dimension is wrong"
        assert self.shape == (0, self.slicesize), "invalid shape"
        assert self._v_chunkshape == (1, self.chunksize), "invalid chunkshape"

        # The superblocksize & blocksize will be saved as attributes
        # (only necessary for sorted index)
        if self.name == "sorted":
            self.attrs.superblocksize = numpy.int64(self.superblocksize)
            self.attrs.blocksize = numpy.int64(self.blocksize)
            self.attrs.optlevel = numpy.int32(self.optlevel)
        return objectId


    def _g_postInitHook(self):
        # initialize some index buffers
        self._startl = numpy.empty(shape=(2,), dtype='int64')
        self._stopl = numpy.empty(shape=(2,), dtype='int64')
        self._stepl = numpy.array([1,1], dtype='int64')
        # Set ``slicesize`` and ``chunksize`` when opening an existing node;
        # otherwise, they are already set.
        if not self._v_new:
            if self.name == "sorted":
                self.slicesize = self.shape[1]
                self.chunksize = self._v_chunkshape[1]
                self.superblocksize = self.attrs.superblocksize
                self.blocksize = self.attrs.blocksize
                self.optlevel = self.attrs.optlevel
        super(IndexArray, self)._g_postInitHook()


    def append(self, arr):
        """Append the object to this (enlargeable) object"""

        extent = arr.shape[0]
        arr.shape = (1, extent)
        self._append(arr)
        arr.shape = (extent,)


    # This version of searchBin uses both ranges (1st level) and
    # bounds (2nd level) caches. It uses a cache for boundary rows,
    # but not for 'sorted' rows (this is only supported for the
    # 'optimized' types.
    def _searchBin(self, nrow, item):
        item1, item2 = item
        result1 = -1; result2 = -1
        hi = self.slicesize
        ranges = self._v_parent.rvcache
        boundscache = self.boundscache
        #t1=time()
        # First, look at the beginning of the slice
        #begin, end = ranges[nrow]  # this is slower
        begin = ranges[nrow,0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
        if item2 < begin:
            result2 = 0
        if result1 >=0 and result2 >= 0:
            return (result1, result2)
        # Then, look for items at the end of the sorted slice
        end = ranges[nrow,1]
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
        # Try to get the bounds row from the LRU cache
        nslot = boundscache.getslot(nrow)
        if nslot >= 0:
            # Cache hit. Use the row kept there.
            bounds = boundscache.getitem(nslot)
        else:
            # No luck with cached data. Read the row and put it in the cache.
            bounds = self._v_parent.bounds[nrow]
            size = bounds.size*bounds.itemsize
            boundscache.setitem(nrow, bounds, size)
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
