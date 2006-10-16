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

import types
import warnings
import sys
from bisect import bisect_left, bisect_right
from time import time

import numpy

from tables.Node import NotLoggedMixin
from tables.Array import Array
from tables.EArray import EArray
from tables.VLArray import Atom, StringAtom
import tables.indexesExtension as indexesExtension


# The minimum row number in a column that can be indexed in tests
minRowIndex = 10

def calcChunksize(expectedrows, optlevel=0, testmode=False):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimize the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized doing more experiments.

    An additional refinement for the output parameters is introduced
    by specifying an optimization ``optlevel``.

    """

    superblocksize, blocksize, slicesize, chunksize = (None, None, None, None)
    optstarts, optstops, optfull = (False, False, False)

    if testmode:
        if expectedrows < minRowIndex*10:
            chunksize = 5
            slicesize = 10
            if optlevel >= 3:
                optstarts = True
        elif expectedrows < minRowIndex*100:
            chunksize = 50
            slicesize = 100
            if optlevel >= 3:
                optstarts, optstops = (True, True)
        elif expectedrows <= minRowIndex*1000:
            chunksize = 500
            slicesize = 1000
            if optlevel >= 3:
                optfull = True
        else:
            raise ValueError, \
"expected rows cannot be larger than %s in test mode" % minRowIndex*1000
        if blocksize == None:
            blocksize = 4*slicesize
        if superblocksize == None:
            superblocksize = 4*blocksize
#         print "superblocksize, blocksize, slicesize, chunksize:", \
#               (superblocksize, blocksize, slicesize, chunksize)
        sizes = (superblocksize, blocksize, slicesize, chunksize)
        opts = (optstarts, optstops, optfull)
        return (sizes, opts)

    expKrows = expectedrows / 1000000.  # Multiples of one million
    #print "expKrows:", expKrows

    # Hint: the slicesize should not exceed 500 or 1000 thousand.
    # That would make NumPy to consume lots of memory for sorting
    # this slice.
    # In general, one should favor a small chunksize (100 ~ 1000) if one
    # wants to reduce the latency for indexed queries. However, keep in
    # mind that a very low value of chunksize for big datasets may
    # hurt the performance by requering the HDF5 to use a lot of memory
    # and CPU for its internal B-Tree
    if expKrows < 0.1: # expected rows < 100 thousand
        chunksize = 250
        slicesize = 100*chunksize
    elif expKrows < 1: # expected rows < 1 milion
        chunksize = 500
        slicesize = 200*chunksize
        if optlevel > 6:
            chunksize = 250
            slicesize = 200*chunksize
            optstarts = True
    elif expKrows < 10:  # expected rows < 10 milion
        if optlevel == 0:
            #chunksize = 1000
            #slicesize = 200*chunksize
            chunksize = 20  # test
            slicesize = 200*chunksize
        elif optlevel == 1:
            chunksize = 750
            slicesize = 300*chunksize
        elif 1 < optlevel <= 3:
            chunksize = 500
            slicesize = 500*chunksize
        elif 3 < optlevel <= 6:
            chunksize = 1000
            slicesize = 200*chunksize
            optstarts = True
        elif 6 < optlevel <= 8:
            chunksize = 500
            slicesize = 500*chunksize
            optstarts = True
            optstops = True
        elif optlevel >= 9:
#             chunksize = 500  # orig
#             slicesize = 1000*chunksize # orig
#             optfull = True
            chunksize = 20  # test
            slicesize = 200*chunksize
            optfull = True
        blocksize = 45*slicesize  # test
        superblocksize = 10*blocksize  # test
    elif expKrows < 100: # expected rows < 100 milions
        if optlevel == 0:
            chunksize = 2000
            slicesize = 250*chunksize
        elif optlevel == 1:
            chunksize = 1500
            slicesize = 350*chunksize
        elif optlevel == 2:
            chunksize = 1000
            slicesize = 500*chunksize
        elif optlevel == 3:
            chunksize = 5000
            slicesize = 100*chunksize
            optstarts = True
        elif optlevel == 4:
            chunksize = 4000
            slicesize = 150*chunksize
            optstarts = True
        elif optlevel == 5:
            chunksize = 2000
            slicesize = 250*chunksize
            optstarts = True
        elif optlevel == 6:
            chunksize = 1000
            slicesize = 500*chunksize
            optstarts = True
        elif optlevel == 7:
            chunksize = 2000
            slicesize = 250*chunksize
            optstarts = True
            optstops = True
        elif optlevel == 8:
            chunksize = 1000
            slicesize = 500*chunksize
            optstarts = True
            optstops = True
        elif optlevel >= 9:   # best effort
            chunksize = 1000
            slicesize = 1000*chunksize
            optfull = True
        blocksize = 15*slicesize
    elif expKrows < 1000: # expected rows < 1000 millions
        if optlevel == 0:
            chunksize = 3000
            slicesize = 250*chunksize
        elif optlevel == 1:
            chunksize = 2500
            slicesize = 350*chunksize
        elif optlevel == 2:
            chunksize = 2000
            slicesize = 500*chunksize
        elif optlevel == 3:
            chunksize = 5000
            slicesize = 200*chunksize
            optstarts = True
        elif optlevel == 4:
            chunksize = 4000
            slicesize = 250*chunksize
            optstarts = True
        elif optlevel == 5:
            chunksize = 3000
            slicesize = 350*chunksize
            optstarts = True
        elif optlevel == 6:
            chunksize = 4000
            slicesize = 250*chunksize
            optfull = True
        elif optlevel == 7:
            chunksize = 3000
            slicesize = 350*chunksize
            optfull = True
        elif optlevel == 8:
            chunksize = 2000
            slicesize = 500*chunksize
            optfull = True
        elif optlevel >= 9:   # best effort
            chunksize = 2000
            slicesize = 1000*chunksize
            optfull = True
        blocksize = 25*slicesize
    elif expKrows < 10*1000: # expected rows < 10 (american) billions
        if optlevel == 0:
            chunksize = 5000
            slicesize = 250*chunksize
        elif optlevel == 1:
            chunksize = 4000
            slicesize = 350*chunksize
        elif optlevel == 2:
            chunksize = 3000
            slicesize = 500*chunksize
        elif optlevel == 3:
            chunksize = 7500
            slicesize = 200*chunksize
            optstarts = True
        elif optlevel == 4:
            chunksize = 5000
            slicesize = 250*chunksize
            optstarts = True
        elif optlevel == 5:
            chunksize = 4000
            slicesize = 350*chunksize
            optstarts = True
        elif optlevel == 6:
            chunksize = 5000
            slicesize = 250*chunksize
            optfull = True
        elif optlevel == 7:
            chunksize = 4000
            slicesize = 350*chunksize
            optfull = True
        elif optlevel == 8:
            chunksize = 3000
            slicesize = 500*chunksize
            optfull = True
        elif optlevel >= 9:   # best effort
            chunksize = 2000
            slicesize = 1500*chunksize
            optfull = True
        blocksize = 35*slicesize
    elif expKrows < 100*1000: # expected rows < 100 (american) billions
        if optlevel == 0:
            chunksize = 7500
            slicesize = 250*chunksize
        elif optlevel == 1:
            chunksize = 6000
            slicesize = 350*chunksize
        elif optlevel == 2:
            chunksize = 5000
            slicesize = 500*chunksize
        elif optlevel == 3:
            chunksize = 10000
            slicesize = 200*chunksize
            optstarts = True
        elif optlevel == 4:
            chunksize = 7500
            slicesize = 250*chunksize
            optstarts = True
        elif optlevel == 5:
            chunksize = 5000
            slicesize = 350*chunksize
            optstarts = True
        elif optlevel == 6:
            chunksize = 5000
            slicesize = 350*chunksize
            optstarts = True
            optstops = True
        elif optlevel == 7:
            chunksize = 4000
            slicesize = 500*chunksize
            optstarts = True
            optstops = True
        elif optlevel == 8:
            chunksize = 4000
            slicesize = 500*chunksize
            optfull = True
        elif optlevel >= 9:   # best effort
            chunksize = 4000
            slicesize = 200*chunksize
            optfull = True
        blocksize = 45*slicesize
        superblocksize = 10*blocksize
    else:  # expected rows >= 1 (american) trillion (perhaps by year 2010
           # this will be useful, who knows...)
        if optlevel == 0:
            chunksize = 10000
            slicesize = 500*chunksize
        elif optlevel == 1:
            chunksize = 8500
            slicesize = 750*chunksize
        elif optlevel == 2:
            chunksize = 7500
            slicesize = 1000*chunksize
        elif optlevel == 3:
            chunksize = 12500
            slicesize = 400*chunksize
            optstarts = True
        elif optlevel == 4:
            chunksize = 8500
            slicesize = 500*chunksize
            optstarts = True
        elif optlevel == 5:
            chunksize = 6000
            slicesize = 750*chunksize
            optstarts = True
        elif optlevel == 6:
            chunksize = 8500
            slicesize = 500*chunksize
            optfull = True
        elif optlevel == 7:
            chunksize = 7500
            slicesize = 750*chunksize
            optfull = True
        elif optlevel == 8:
            chunksize = 6000
            slicesize = 1000*chunksize
            optfull = True
        elif optlevel >= 9:   # best effort
            chunksize = 5000
            slicesize = 1000*chunksize
            optfull = True
        blocksize = 50*slicesize
        superblocksize = 20*blocksize

    # The defaults for blocksize & superblocksize
    if blocksize == None:
        blocksize = 10*slicesize
    if superblocksize == None:
        superblocksize = 10*blocksize

#     print "superblocksize, blocksize, slicesize, chunksize:", \
#           (superblocksize, blocksize, slicesize, chunksize)
    # The size for different blocks information
    sizes = (superblocksize, blocksize, slicesize, chunksize)
    # The reordering optimization flags
    opts = (optstarts, optstops, optfull)
    return (sizes, opts)


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
                 optlevel=0,
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
        self.reord_opts = None
        """The reordering optimizations."""
        if atom is not None:
            sizes, reord_opts = calcChunksize(expectedrows, optlevel, testmode)
            (self.superblocksize, self.blocksize,
             self.slicesize, self.chunksize) = sizes
            self.reord_opts = reord_opts

        super(IndexArray, self).__init__(
            parentNode, name, atom, title, filters, expectedrows)


    def _g_create(self):
        objectId = super(IndexArray, self)._g_create()
        assert self.extdim == 0, "computed extendable dimension is wrong"
        assert self.shape == (0, self.slicesize), "invalid shape"
        assert self._v_chunksize == (1, self.chunksize), "invalid chunk size"
        # The superblocksize & blocksize will be saved as (pickled) attributes
        # (only necessary for sorted index)
        if self.name == "sorted":
            self.attrs.superblocksize = self.superblocksize
            self.attrs.blocksize = self.blocksize
            # The same goes for reordenation opts
            self.attrs.reord_opts = self.reord_opts
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
        self._startl = numpy.empty(shape=(2,), dtype='int64')
        self._stopl = numpy.empty(shape=(2,), dtype='int64')
        self._stepl = numpy.array([1,1], dtype='int64')
        # Set ``slicesize`` and ``chunksize`` when opening an existing node;
        # otherwise, they are already set.
        if not self._v_new:
            if self.name == "sorted":
                self.slicesize = self.shape[1]
                self.chunksize = self._v_chunksize[1]
                self.superblocksize = self.attrs.superblocksize
                self.blocksize = self.attrs.blocksize
                self.reord_opts = self.attrs.reord_opts
        super(IndexArray, self)._g_postInitHook()


    def append(self, arr):
        """Append the object to this (enlargeable) object"""

        extent = arr.shape[0]
        arr.shape = (1, extent)
        self._append(arr)
        arr.shape = (extent,)


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
        # XXX try to use a LRU cache here, if possible...
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


    def __str__(self):
        "A compact representation of this class"
        return "IndexArray(path=%s)" % \
               (self._v_parent._g_join(self.name))


    def __repr__(self):
        """A verbose representation of this class"""

        return """%s
  ptype = %r
  shape = %s
  itemsize = %s
  nrows = %s
  slicesize = %s
  chunksize = %s
  byteorder = %r""" % (self, self.ptype, self.shape, self.itemsize, self.nrows,
                       self.slicesize, self.chunksize, self.byteorder)
