#######################################################################
#
#       License: BSD
#       Created: June 08, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Here is defined the Index class.

See Index class docstring for more info.

Classes:

    Index

Functions:


Misc variables:

    __version__
    defaultAutoIndex
    defaultIndexFilters


"""

import sys
from bisect import bisect_left, bisect_right
from time import time, clock
import os, os.path
import tempfile
import math

import numpy

from tables.idxutils import (
    calcChunksize, calcoptlevels, get_reduction_level,
    show_stats, nextafter, infType )

from tables import indexesExtension
from tables import utilsExtension
from tables.attributeset import AttributeSet
from tables.node import NotLoggedMixin
from tables.atom import IntAtom, UIntAtom, Atom
from tables.earray import EArray
from tables.carray import CArray
from tables.leaf import Filters
from tables.indexes import CacheArray, LastRowArray, IndexArray
from tables.group import Group
from tables.path import joinPath
from tables.parameters import (
    LIMBOUNDS_MAX_SLOTS, LIMBOUNDS_MAX_SIZE,
    BOUNDS_MAX_SLOTS, BOUNDS_MAX_SIZE,
    SORTEDLR_MAX_SLOTS, SORTEDLR_MAX_SIZE,
    MAX_GROUP_WIDTH )
from tables.exceptions import PerformanceWarning
from tables.utils import lazyattr

from tables.lrucacheExtension import ObjectCache


__version__ = "$Revision: 1236 $"


# default version for INDEX objects
#obversion = "1.0"    # Version of indexes in PyTables 1.x series
#obversion = "2.0"    # Version of indexes in PyTables Pro 2.0 series
obversion = "2.1"    # Version of indexes in PyTables Pro 2.1 and up series


debug = False
#debug = True  # Uncomment this for printing sizes purposes
profile = False
#profile = True  # uncomment for profiling purposes only


# The default method for sorting
defsort = "quicksort"
#defsort = "mergesort"

# Default policy for automatically updating indexes after a table
# append operation, or automatically reindexing after an
# index-invalidating operation like removing or modifying table rows.
defaultAutoIndex = True
# Keep in sync with ``Table.autoIndex`` docstring.

# Default filters used to compress indexes.  This is quite fast and
# compression is pretty good.
defaultIndexFilters = Filters( complevel=1, complib='zlib',
                               shuffle=True, fletcher32=False )
# Keep in sync with ``Table.indexFilters`` docstring.

# The list of types for which an optimised search in Pyrex and C has
# been implemented. Always add here the name of a new optimised type.
opt_search_types = ("int8", "int16", "int32", "int64",
                    "uint8", "uint16", "uint32", "uint64",
                    "float32", "float64")

# The upper limit for uint32 ints
max32 = 2**32


class Index(NotLoggedMixin, indexesExtension.Index, Group):

    """
    Represents the index of a column in a table.

    This class is used to keep the indexing information for columns in a
    `Table` dataset.  It is actually a descendant of the `Group` class,
    with some added functionality.  An `Index` is always associated with
    one and only one column in the table.

    This class is mainly intended for internal use, but some of its
    attributes may be interesting for the programmer.

    Public instance variables
    -------------------------

    column
        The `Column` instance for the indexed column.
    dirty
        Whether the index is dirty or not. Dirty indexes are out of sync
        with column data, so are not usable.
    filters
        Filter properties for this index --see `Filters`.
    nelements
        The number of currently indexed rows for this column.
    """

    _c_classId = 'INDEX'


    # <properties>

    def _getdirty(self):
        if 'DIRTY' not in self._v_attrs:
            # If there is no ``DIRTY`` attribute, index should be clean.
            return False
        return self._v_attrs.DIRTY

    def _setdirty(self, dirty):
        wasdirty, isdirty = self.dirty, bool(dirty)
        self._v_attrs.DIRTY = dirty
        # If an *actual* change in dirtiness happens,
        # notify the condition cache by setting or removing a nail.
        conditionCache = self.column.table._conditionCache
        if not wasdirty and isdirty:
            conditionCache.nail()
        if wasdirty and not isdirty:
            conditionCache.unnail()

    dirty = property(
        _getdirty, _setdirty, None,
        """
        Whether the index is dirty or not.

        Dirty indexes are out of sync with column data, so they exist
        but they are not usable.
        """ )

    nblockssuperblock = property(
        lambda self: self.superblocksize / self.blocksize, None, None,
        "The number of blocks in a superblock.")

    nslicesblock = property(
        lambda self: self.blocksize / self.slicesize, None, None,
        "The number of slices in a block.")

    nchunkslice = property(
        lambda self: self.slicesize / self.chunksize, None, None,
        "The number of chunks in a slice.")

    def _g_nsuperblocks(self):
        # Last row should not be considered as a superblock
        nelements = self.nelements - self.nelementsILR
        nblocks = nelements / self.superblocksize
        if nelements % self.blocksize > 0:
            nblocks += 1
        return nblocks
    nsuperblocks = property(_g_nsuperblocks , None, None,
        "The total number of superblocks in index.")

    def _g_nblocks(self):
        # Last row should not be considered as a block
        nelements = self.nelements - self.nelementsILR
        nblocks = nelements / self.blocksize
        if nelements % self.blocksize > 0:
            nblocks += 1
        return nblocks
    nblocks = property(_g_nblocks , None, None,
        "The total number of blocks in index.")

    nslices = property(
        lambda self: self.nelements / self.slicesize, None, None,
        "The number of complete slices in index.")

    nchunks = property(
        lambda self: self.nelements / self.chunksize, None, None,
        "The number of complete chunks in index.")

    shape = property(
        lambda self: (self.nrows, self.slicesize), None, None,
        "The shape of this index (in slices and elements).")

    temp_required = property(
        lambda self: (self.indsize > 1 and
                      self.optlevel > 0 and
                      self.column.table.nrows > self.slicesize),
        None, None,
        "Whether a temporary file for indexes is required or not.")

    @lazyattr
    def nrowsinchunk(self):
        """The number of rows that fits in a *table* chunk."""
        return self.column.table.chunkshape[0]

    @lazyattr
    def lbucket(self):
        """Return the length of a bucket based index type."""
        # Avoid to set a too large lbucket size (mainly useful for tests)
        lbucket = min(self.nrowsinchunk, self.chunksize)
        if self.indsize == 1:
            # For ultra-light, we will never have to keep track of a
            # bucket outside of a slice.
            maxnb = 2**8
            if self.slicesize > maxnb*lbucket:
                lbucket = int(math.ceil(float(self.slicesize)/maxnb))
        elif self.indsize == 2:
            # For light, we will never have to keep track of a
            # bucket outside of a block.
            maxnb = 2**16
            if self.blocksize > maxnb*lbucket:
                lbucket = int(math.ceil(float(self.blocksize)/maxnb))
        else:
            # For medium and full indexes there should no need to
            # increase lbucket
            pass
        return lbucket


    # </properties>


    def __init__(self, parentNode, name,
                 atom=None, column=None, title="",
                 optlevel=None,
                 filters=None,
                 tmp_dir=None,
                 expectedrows=0,
                 byteorder=None,
                 blocksizes=None,
                 indsize=None,
                 new=True):
        """Create an Index instance.

        Keyword arguments:

        atom -- An Atom object representing the shape and type of the
            atomic objects to be saved. Only scalar atoms are
            supported.

        column -- The column object to be indexed

        title -- Sets a TITLE attribute of the Index entity.

        optlevel -- The desired optimization level for this index.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object. If not specified, the ZLIB
            & shuffle will be activated by default (i.e., they are not
            inherited from the parent, that is, the Table).

        tmp_dir -- The directory for the temporary files.

        expectedrows -- Represents an user estimate about the number
            of row slices that will be added to the growable dimension
            in the IndexArray object.

        byteorder -- The byteorder of the index datasets *on-disk*.

        blocksizes -- The four main sizes of the compound blocks in
            index datasets (a low level parameter).

        indsize -- Either the indices have a complete track of the row
            position (64-bit) or only in which chunk they are (32-bit,
            16-bit, 8-bit).  Its value can be 1 (8-bit), 2 (16-bit), 4
            (32-bit) or 8 (64-bit).

        """

        self._v_version = None
        """The object version of this index."""

        self.optlevel = optlevel
        """The optimization level for this index."""
        self.tmp_dir = tmp_dir
        """The directory for the temporary files."""
        self.expectedrows = expectedrows
        """The expected number of items of index arrays."""
        if byteorder in ["little", "big"]:
            self.byteorder = byteorder
        else:
            self.byteorder = sys.byteorder
        """The byteorder of the index datasets."""
        if atom is not None:
            self.dtype = atom.dtype.base
            self.type = atom.type
            """The datatypes to be stored by the sorted index array."""
            ############### Important note ###########################
            #The datatypes saved as index values are NumPy native
            #types, so we get rid of type metainfo like Time* or Enum*
            #that belongs to HDF5 types (actually, this metainfo is
            #not needed for sorting and looking-up purposes).
            ##########################################################
            assert indsize in (1, 2, 4, 8), "indsize should be 1, 2, 4 or 8!"

        self.column = column
        """The `Column` instance for the indexed column."""

        self.nrows = None
        """The total number of slices in the index."""
        self.nelements = None
        """The number of currently indexed row for this column."""
        self.blocksizes = blocksizes
        """The four main sizes of the compound blocks (if specified)."""
        self.indsize = indsize
        """The itemsize for the indices part of the index (1, 2, 4 or 8)."""
        self.dirtycache = True
        """Dirty cache (for ranges, bounds & sorted) flag."""
        self.superblocksize = None
        """Size of the superblock for this index."""
        self.blocksize = None
        """Size of the block for this index."""
        self.slicesize = None
        """Size of the slice for this index."""
        self.chunksize = None
        """Size of the chunk for this index."""
        self.tmpfilename = None
        """Filename for temporary bounds."""
        self.opt_search_types = opt_search_types
        """The types for which and optimized search has been implemented."""

        self.tprof = 0
        """Time counter for benchmarking purposes."""

        from tables.file import openFile
        self._openFile = openFile
        """The `openFile()` function, to avoid a circular import."""

        super(Index, self).__init__(
            parentNode, name, title, new, filters)


    def _g_postInitHook(self):
        if self._v_new:
            # The version for newly created indexes
            self._v_version = obversion
        super(Index, self)._g_postInitHook()

        # Index arrays must only be created for new indexes
        if not self._v_new:
            idxversion = self._v_version
            # Set-up some variables from info on disk and return
            attrs = self._v_attrs
            # Coerce NumPy scalars to Python scalars in order
            # to avoid undesired upcasting operations.
            self.superblocksize = long(attrs.superblocksize)
            self.blocksize = long(attrs.blocksize)
            self.slicesize = int(attrs.slicesize)
            self.chunksize = int(attrs.chunksize)
            self.blocksizes = (self.superblocksize, self.blocksize,
                               self.slicesize, self.chunksize)
            self.optlevel = int(attrs.optlevel)
            sorted = self.sorted
            indices = self.indices
            self.dtype = sorted.atom.dtype
            self.type = sorted.atom.type
            self.filters = attrs.FILTERS
            self.indsize = indices.atom.itemsize
            # Some sanity checks for slicesize, chunksize and indsize
            assert self.slicesize == indices.shape[1], "Wrong slicesize"
            assert self.chunksize == indices._v_chunkshape[1], "Wrong chunksize"
            assert self.indsize in (1, 2, 4, 8), "Wrong indices itemsize"
            if idxversion > "2.0":
                self.reduction = int(attrs.reduction)
                nelementsSLR = int(self.sortedLR.attrs.nelements)
                nelementsILR = int(self.indicesLR.attrs.nelements)
            else:
                self.reduction = 1
                nelementsILR = self.indicesLR[-1]
                nelementsSLR = nelementsILR
            self.nrows = sorted.nrows
            self.nelements = self.nrows * self.slicesize + nelementsILR
            self.nelementsSLR = nelementsSLR
            self.nelementsILR = nelementsILR
            if nelementsILR > 0:
                self.nrows += 1
            # Get the bounds as a cache (this has to remain here!)
            rchunksize = self.chunksize // self.reduction
            nboundsLR = (nelementsSLR - 1 ) // rchunksize
            if nboundsLR < 0:
                nboundsLR = 0 # correction for -1 bounds
            nboundsLR += 2 # bounds + begin + end
            # All bounds values (+begin+end) are at the end of sortedLR
            self.bebounds = self.sortedLR[nelementsSLR:nelementsSLR+nboundsLR]
            return

        # The index is new. Initialize the values
        self.nrows = 0
        self.nelements = 0
        self.nelementsSLR = 0
        self.nelementsILR = 0

        # The atom
        atom = Atom.from_dtype(self.dtype)

        # Set the filters for this object (they are *not* inherited)
        self.filters = filters = self._v_new_filters
        self.sfilters = sfilters = filters
        self.ifilters = ifilters = filters

        # Disable shuffle for an index with itemsize equal to 1 or a
        # string type because it doesn't represent any real advantage
        # for the compression process
        if filters.shuffle:
            # Filter for sorted values
            if atom.itemsize == 1 or atom.kind == "string":
                self.sfilters = sfilters = filters.copy(shuffle=False)
            # Filter for indices
            if self.indsize == 1:
                self.ifilters = ifilters = filters.copy(shuffle=False)

        # Compute the superblocksize, blocksize, slicesize and chunksize values
        # (in case these parameters haven't been passed to the constructor)
        if self.blocksizes is None:
            self.blocksizes = calcChunksize(
                self.expectedrows, self.optlevel, self.indsize)
        (self.superblocksize, self.blocksize,
         self.slicesize, self.chunksize) = self.blocksizes
        if debug:
            print "blocksizes:", self.blocksizes
        # Compute the reduction level
        self.reduction = get_reduction_level(
            self.indsize, self.optlevel, self.slicesize, self.chunksize)
        rchunksize = self.chunksize // self.reduction
        rslicesize = self.slicesize // self.reduction

        # Save them on disk as attributes
        self._v_attrs.superblocksize = numpy.uint64(self.superblocksize)
        self._v_attrs.blocksize = numpy.uint64(self.blocksize)
        self._v_attrs.slicesize = numpy.uint32(self.slicesize)
        self._v_attrs.chunksize = numpy.uint32(self.chunksize)
        # Save the optlevel as well
        self._v_attrs.optlevel = self.optlevel
        # Save the reduction level
        self._v_attrs.reduction = self.reduction

        # Create the IndexArray for sorted values
        sorted = IndexArray(self, 'sorted', atom, "Sorted Values",
                            sfilters, self.byteorder)

        # Create the IndexArray for index values
        IndexArray(self, 'indices', UIntAtom(itemsize=self.indsize),
                   "Number of chunk in table", ifilters, self.byteorder)

        # Create the cache for range values  (1st order cache)
        CacheArray(self, 'ranges', atom, (0,2), "Range Values", sfilters,
                   self.expectedrows//self.slicesize,
                   byteorder=self.byteorder)
        # median ranges
        EArray(self, 'mranges', atom, (0,), "Median ranges", sfilters,
               byteorder=self.byteorder, _log=False)

        # Create the cache for boundary values (2nd order cache)
        nbounds_inslice = (rslicesize-1)//rchunksize
        CacheArray(self, 'bounds', atom, (0, nbounds_inslice),
                   "Boundary Values", sfilters, self.nchunks,
                   (1, nbounds_inslice), byteorder=self.byteorder)

        # begin, end & median bounds (only for numerical types)
        EArray(self, 'abounds', atom, (0,), "Start bounds", sfilters,
               byteorder=self.byteorder, _log=False)
        EArray(self, 'zbounds', atom, (0,), "End bounds", sfilters,
               byteorder=self.byteorder, _log=False)
        EArray(self, 'mbounds', atom, (0,), "Median bounds", sfilters,
               byteorder=self.byteorder, _log=False)

        # Create the Array for last (sorted) row values + bounds
        shape = (rslicesize + 2 + nbounds_inslice,)
        sortedLR = LastRowArray(self, 'sortedLR', atom, shape,
                                "Last Row sorted values + bounds",
                                sfilters, (rchunksize,),
                                byteorder=self.byteorder)

        # Create the Array for the number of chunk in last row
        shape = (self.slicesize,)     # enough for indexes and length
        indicesLR = LastRowArray(self, 'indicesLR',
                                 UIntAtom(itemsize=self.indsize),
                                 shape, "Last Row number of chunk",
                                 sfilters, (self.chunksize,),
                                 byteorder=self.byteorder)

        # The number of elements in LR will be initialized here
        sortedLR.attrs.nelements = 0
        indicesLR.attrs.nelements = 0

        # All bounds values (+begin+end) are uninitialized in creation time
        self.bebounds = None

        # The starts and lengths initialization
        self.starts = numpy.empty(shape=self.nrows, dtype=numpy.int32)
        """Where the values fulfiling conditions starts for every slice."""
        self.lengths = numpy.empty(shape=self.nrows, dtype=numpy.int32)
        """Lengths of the values fulfilling conditions for every slice."""

        # Finally, create a temporary file for indexes if needed
        if self.temp_required:
            self.create_temp()


    def _g_updateDependent(self):
        super(Index, self)._g_updateDependent()
        self.column._updateIndexLocation(self)


    def initial_append(self, xarr, nrow, reduction):
        """Compute an initial indices arrays for data to be indexed."""
        if profile: tref = time()
        if profile: show_stats("Entering initial_append", tref)
        arr = xarr.pop()
        indsize = self.indsize
        slicesize = self.slicesize
        blocksize = self.blocksize
        if profile: show_stats("Before creating idx", tref)
        if indsize > 2:
            # As len(arr) < 2**31, we can choose uint32 for representing
            # indices.  In this way, we consume far less memory during
            # the keysort process.
            idx = numpy.arange(0, len(arr), dtype='uint32')
        else:
            idx = numpy.empty(len(arr), "uint%d"%(indsize*8))
            lbucket = self.lbucket
            # Fill the idx with the bucket indices
            offset = lbucket-((nrow*(slicesize%lbucket))%lbucket)
            idx[0:offset] = 0
            for i in xrange(offset, slicesize, lbucket):
                idx[i:i+lbucket] = (i+lbucket-1)/lbucket
            if indsize == 2:
                # Add a second offset in this case
                # First normalize the number of rows
                offset2 = (nrow%self.nslicesblock)*slicesize/lbucket
                idx += offset2
        # In-place sorting
        if profile: show_stats("Before keysort", tref)
        indexesExtension.keysort(arr, idx)
        larr = arr[-1]
        if reduction > 1:
            # It's important to do a copy() here in order to ensure that
            # sorted._append() will receive a contiguous array.
            if profile: show_stats("Before reduction", tref)
            reduc = arr[::reduction].copy()
            if profile: show_stats("After reduction", tref)
            arr = reduc
            if profile: show_stats("After arr <-- reduc", tref)
        if profile: show_stats("Entering initial_append", tref)
        return larr, arr, idx


    def final_idx(self, idx, offset):
        """Upcast idx to 64-bit and a possible additional downcast."""
        if profile: tref = time()
        if profile: show_stats("Entering final_idx", tref)
        # For medium (32-bit) and full (64-bit) indexes, all the
        # rows in tables should by directly reachable.
        # Do an upcast first in order to add the offset
        idx = idx.astype('uint64')
        idx += offset
        # Check if we have to do an additional downcast
        if self.indsize == 4:
            # The next partition is valid up to table sizes of
            # 2**30*2**20 = 2**50 bytes, that is, 1 Exabyte, which
            # should be a safe figure, at least for a while.
            idx /= self.lbucket
            # After the division, we can downsize the indexes to 'uint32'
            idx = idx.astype('uint32')
        if profile: show_stats("Exiting final_idx", tref)
        return idx


    def append(self, xarr, update=False):
        """Append the array to the index objects"""

        if profile: tref = time()
        if profile: show_stats("Entering append", tref)
        if not update and self.temp_required:
            where = self.tmp
            # The reduction will take place *after* the optimization process
            reduction = 1
        else:
            where = self
            reduction = self.reduction
        sorted = where.sorted; indices = where.indices
        ranges = where.ranges; mranges = where.mranges
        bounds = where.bounds; mbounds = where.mbounds
        abounds = where.abounds; zbounds = where.zbounds
        nrows = sorted.nrows  # before sorted.append()
        larr, arr, idx = self.initial_append(xarr, nrows, reduction)
        # Save the sorted array
        sorted.append(arr.reshape(1, arr.size))
        cs = self.chunksize/reduction;  ncs = self.nchunkslice
        # Save ranges & bounds
        ranges.append([[arr[0], larr]])
        bounds.append([arr[cs::cs]])
        abounds.append(arr[0::cs])
        zbounds.append(arr[cs-1::cs])
        # Compute the medians
        smedian = arr[cs/2::cs]
        mbounds.append(smedian)
        mranges.append([smedian[ncs/2]])
        if profile: show_stats("Before deleting arr & smedian", tref)
        del arr, smedian   # delete references
        if profile: show_stats("After deleting arr & smedian", tref)
        # Now that arr is gone, we can upcast the indices and add the offset
        if self.indsize >= 4:
            idx = self.final_idx(idx, nrows*self.slicesize)
        indices.append(idx.reshape(1, idx.size))
        if profile: show_stats("Before deleting idx", tref)
        del idx
        # Update counters after a successful append
        self.nrows = nrows + 1
        self.nelements = self.nrows * self.slicesize
        self.nelementsSLR = 0  # reset the counter of the last row index to 0
        self.nelementsILR = 0  # reset the counter of the last row index to 0
        self.dirtycache = True   # the cache is dirty now
        if profile: show_stats("Exiting append", tref)


    def appendLastRow(self, xarr):
        """Append the array to the last row index objects"""

        if profile: tref = time()
        if profile: show_stats("Entering appendLR", tref)
        # compute the elements in the last row sorted & bounds array
        nrows = self.nslices
        indicesLR = self.indicesLR
        sortedLR = self.sortedLR
        reduction = self.reduction
        larr, arr, idx = self.initial_append(xarr, nrows, reduction)
        nelementsSLR = len(arr)
        nelementsILR = len(idx)
        # Build the cache of bounds
        rchunksize = self.chunksize // reduction
        self.bebounds = numpy.concatenate((arr[::rchunksize], [larr]))
        # The number of elements will be saved as an attribute
        sortedLR.attrs.nelements = nelementsSLR
        indicesLR.attrs.nelements = nelementsILR
        # Save the number of elements, bounds and sorted values
        # at the end of the sorted array
        offset2 = len(self.bebounds)
        sortedLR[nelementsSLR:nelementsSLR+offset2] = self.bebounds
        sortedLR[:nelementsSLR] = arr
        del arr
        # Now that arr is gone, we can upcast the indices and add the offset
        if self.indsize >= 4:
            idx = self.final_idx(idx, nrows*self.slicesize)
        # Save the reverse index array
        indicesLR[:len(idx)] = idx
        del idx
        # Update counters after a successful append
        self.nrows = nrows + 1
        self.nelements = nrows * self.slicesize + nelementsILR
        self.nelementsILR = nelementsILR
        self.nelementsSLR = nelementsSLR
        self.dirtycache = True   # the cache is dirty now
        if profile: show_stats("Exiting appendLR", tref)


    def optimize(self, verbose=False):
        """Optimize an index so as to allow faster searches.

        verbose -- If True, messages about the progress of the
            optimization process are printed out.

        """

        if not self.temp_required:
            return

        if verbose == True:
            self.verbose = True
        else:
            self.verbose = debug

        # Initialize last_tover and last_nover
        self.last_tover = 0
        self.last_nover = 0

        if self.verbose:
            (nover, mult, tover) = self.compute_overlaps("init", self.verbose)

        # Compute the correct optimizations for current optim level
        opts = calcoptlevels(self.nblocks, self.optlevel, self.indsize)
        optmedian, optstarts, optstops, optfull = opts

        if debug:
            print "optvalues:", opts

        self.create_temp2()
        # Start the optimization process
        while True:
            if optfull:
                for niter in range(optfull):
                    if self.swap('chunks', 'median'): break
                    if self.nblocks > 1:
                        # Swap slices only in the case that we have
                        # several blocks
                        if self.swap('slices', 'median'): break
                        if self.swap('chunks','median'): break
                    if self.swap('chunks', 'start'): break
                    if self.swap('chunks', 'stop'): break
            else:
                if optmedian:
                    if self.swap('chunks', 'median'): break
                if optstarts:
                    if self.swap('chunks', 'start'): break
                if optstops:
                    if self.swap('chunks', 'stop'): break
            break  # If we reach this, exit the loop

        # Close and delete the temporal optimization index file
        self.cleanup_temp()
        return


    def swap(self, what, mode=None):
        "Swap chunks or slices using a certain bounds reference."

        # Thresholds for avoiding continuing the optimization
        # XXX TODO: These should be a function of the optimization level...
        thnover = 4        # minimum number of overlapping slices
        thmult = 0.01      # minimum ratio of multiplicity (a 1%)
        thtover = 0.001    # minimum overlaping index for slices (a .1%)
        if self.verbose:
            t1 = time();  c1 = clock()
        if what == "chunks":
            self.swap_chunks(mode)
        elif what == "slices":
            self.swap_slices(mode)
        if mode:
            message = "swap_%s(%s)" % (what, mode)
        else:
            message = "swap_%s" % (what,)
        (nover, mult, tover) = self.compute_overlaps(message, self.verbose)
        rmult = len(mult.nonzero()[0]) / float(len(mult))
        if self.verbose:
            t = round(time()-t1, 4);  c = round(clock()-c1, 4)
            print "time: %s. clock: %s" % (t, c)
        # Check that entropy is actually decreasing
        if what == "chunks" and self.last_tover > 0. and self.last_nover > 0:
            tover_var = (self.last_tover - tover) / self.last_tover
            nover_var = (self.last_nover - nover) / self.last_nover
            if tover_var < 0.05 and nover_var < 0.05:
                # Less than a 5% of improvement is too few
                return True
        self.last_tover = tover
        self.last_nover = nover
        # Check if some threshold has met
        if nover < thnover:
            return True
        if rmult < thmult:
            return True
        # Additional check for the overlap ratio
        if tover >= 0. and tover < thtover:
            return True
        return False


    def create_temp(self):
        "Create some temporary objects for slice sorting purposes."

        # The index will be dirty during the index optimization process
        self.dirty = True
        # Build the name of the temporary file
        fd, self.tmpfilename = tempfile.mkstemp(
            ".tmp", "pytables-" , self.tmp_dir)
        # Close the file descriptor so as to avoid leaks
        os.close(fd)
        # Create the proper PyTables file
        self.tmpfile = self._openFile(self.tmpfilename, "w")
        self.tmp = tmp = self.tmpfile.root
        cs = self.chunksize
        ss = self.slicesize
        sfilters = self.sfilters
        ifilters = self.ifilters
        # temporary sorted & indices arrays
        shape = (0, ss)
        atom = Atom.from_dtype(self.dtype)
        EArray(tmp, 'sorted', atom, shape,
               "Temporary sorted", sfilters, chunkshape=(1,cs))
        EArray(tmp, 'indices', UIntAtom(itemsize=self.indsize), shape,
               "Temporary indices", ifilters, chunkshape=(1,cs))
        # temporary bounds
        nbounds_inslice = (ss - 1) // cs
        shape = (0, nbounds_inslice)
        EArray(tmp, 'bounds', atom, shape, "Temp chunk bounds",
               sfilters, chunkshape=(cs, nbounds_inslice))
        shape = (0,)
        EArray(tmp, 'abounds', atom, shape, "Temp start bounds",
               sfilters, chunkshape=(cs,))
        EArray(tmp, 'zbounds', atom, shape, "Temp end bounds",
               sfilters, chunkshape=(cs,))
        EArray(tmp, 'mbounds', atom, shape, "Median bounds",
               sfilters, chunkshape=(cs,))
        # temporary ranges
        EArray(tmp, 'ranges', atom, (0, 2),
               "Temporary range values", sfilters, chunkshape=(cs,2))
        EArray(tmp, 'mranges', atom, (0,),
               "Median ranges", sfilters, chunkshape=(cs,))


    def create_temp2(self):
        "Create some temporary objects for slice sorting purposes."

        # The algorithms for doing the swap can be optimized so that
        # one should be necessary to create temporaries for keeping just
        # the contents of a single superblock.
        # F. Altet 2007-01-03
        cs = self.chunksize
        ss = self.slicesize
        sfilters = self.sfilters
        ifilters = self.ifilters
        # temporary sorted & indices arrays
        shape = (self.nslices, ss)
        atom = Atom.from_dtype(self.dtype)
        tmp = self.tmp
        CArray(tmp, 'sorted2', atom, shape,
               "Temporary sorted 2", sfilters, chunkshape=(1,cs))
        CArray(tmp, 'indices2', UIntAtom(itemsize=self.indsize), shape,
               "Temporary indices 2", ifilters, chunkshape=(1,cs))
        # temporary bounds
        nbounds_inslice = (ss - 1) // cs
        shape = (self.nslices, nbounds_inslice)
        CArray(tmp, 'bounds2', atom, shape, "Temp chunk bounds 2",
               sfilters, chunkshape=(cs, nbounds_inslice))
        shape = (self.nchunks,)
        CArray(tmp, 'abounds2', atom, shape, "Temp start bounds 2",
               sfilters, chunkshape=(cs,))
        CArray(tmp, 'zbounds2', atom, shape, "Temp end bounds 2",
               sfilters, chunkshape=(cs,))
        CArray(tmp, 'mbounds2', atom, shape, "Median bounds 2",
               sfilters, chunkshape=(cs,))
        # temporary ranges
        CArray(tmp, 'ranges2', atom, (self.nslices, 2),
               "Temporary range values 2", sfilters, chunkshape=(cs,2))
        CArray(tmp, 'mranges2', atom, (self.nslices,),
               "Median ranges 2", sfilters, chunkshape=(cs,))


    def cleanup_temp(self):
        "Copy the data and delete the temporaries for sorting purposes."

        if self.verbose:
            print "Copying temporary data..."
        # tmp -> index
        reduction = self.reduction
        cs = self.chunksize//reduction;  ncs = self.nchunkslice
        tmp = self.tmp
        for i in xrange(self.nslices):
            # Copy sorted & indices slices
            sorted = tmp.sorted[i][::reduction].copy()
            self.sorted.append(sorted.reshape(1, sorted.size))
            # Compute ranges
            self.ranges.append([[sorted[0], sorted[-1]]])
            # Compute chunk bounds
            self.bounds.append([sorted[cs::cs]])
            # Compute start, stop & median bounds and ranges
            self.abounds.append(sorted[0::cs])
            self.zbounds.append(sorted[cs-1::cs])
            smedian = sorted[cs/2::cs]
            self.mbounds.append(smedian)
            self.mranges.append([smedian[ncs/2]])
            del sorted, smedian   # delete references
            # Now that sorted is gone, we can copy the indices
            indices = tmp.indices[i]
            self.indices.append(indices.reshape(1, indices.size))

        if self.verbose:
            print "Deleting temporaries..."
        self.tmp = None
        self.tmpfile.close()
        os.remove(self.tmpfilename)
        self.tmpfilename = None

        # The optimization process has finished, and the index is ok now
        self.dirty = False
        # ...but the memory data cache is dirty now
        self.dirtycache = True


    def get_neworder(self, neworder, src_disk, tmp_disk,
                     nslices, offset, dtype):
        """Get sorted & indices values in new order."""
        cs = self.chunksize
        ncs = self.nchunkslice
        tmp = numpy.empty(shape=self.slicesize, dtype=dtype)
        for i in xrange(nslices):
            ns = offset + i;
            # Get slices in new order
            for j in xrange(ncs):
                idx = neworder[i*ncs+j]
                ins = idx / ncs;  inc = (idx - ins*ncs)*cs
                ins += offset
                nc = j * cs
                tmp[nc:nc+cs] = src_disk[ins,inc:inc+cs]
            tmp_disk[ns] = tmp


    def swap_chunks(self, mode="median"):
        "Swap & reorder the different chunks in a block."

        boundsnames = {'start':'abounds', 'stop':'zbounds', 'median':'mbounds'}
        tmp = self.tmp
        sorted = tmp.sorted;  indices = tmp.indices
        tmp_sorted = tmp.sorted2;  tmp_indices = tmp.indices2
        cs = self.chunksize
        ss = self.slicesize
        ncs = self.nchunkslice
        nsb = self.nslicesblock
        ncb = ncs * nsb
        ncb2 = ncb
        boundsobj = tmp._f_getChild(boundsnames[mode])
        for nblock in xrange(self.nblocks):
            # Protection for last block having less chunks than ncb
            remainingchunks = self.nchunks - nblock*ncb
            if remainingchunks < ncb:
                # To avoid reordering the chunks in last row (slice)
                # Implementing this suppose to complicate quite a bit
                # the code for index optimization and perhaps this is
                # not worth the effort.
                # F. Altet 2007-04-12
                ncb2 = (remainingchunks/ncs)*ncs
            if ncb2 <= 1:
                # if only zero or one chunks remains we are done
                break
            nslices = ncb2/ncs
            bounds = boundsobj[nblock*ncb:nblock*ncb+ncb2]
            sbounds_idx = bounds.argsort(kind=defsort)
            offset = nblock*nsb
            # Swap sorted and indices following the new order
            self.get_neworder(sbounds_idx, sorted, tmp_sorted,
                              nslices, offset, self.dtype)
            self.get_neworder(sbounds_idx, indices, tmp_indices,
                              nslices, offset, 'u%d' % self.indsize)
        # Reorder completely the index at slice level
        self.reorder_slices(tmp=True)


    def read_slice(self, where, nslice, buffer):
        """Read a slice from the `where` dataset and put it in `buffer`."""
        self.startl[:] = (nslice, 0)
        self.stopl[:] = (nslice+1, self.slicesize)
        where._g_readSlice(self.startl, self.stopl, self.stepl, buffer)


    def write_slice(self, where, nslice, buffer):
        """Write a `slice` to the `where` dataset with the `buffer` data."""
        self.startl[:] = (nslice, 0)
        self.stopl[:] = (nslice+1, self.slicesize)
        countl = self.stopl - self.startl   # (1, self.slicesize)
        where._modify(self.startl, self.stepl, countl, buffer)


    def reorder_slice(self, nslice, sorted, indices, ssorted, sindices,
                      tmp_sorted, tmp_indices):
        """Copy & reorder the slice in source to final destination."""
        ss = self.slicesize
        # Load the second part in buffers
        self.read_slice(tmp_sorted, nslice, ssorted[ss:])
        self.read_slice(tmp_indices, nslice, sindices[ss:])
        indexesExtension.keysort(ssorted, sindices)
        # Write the first part of the buffers to the regular leaves
        self.write_slice(sorted, nslice-1, ssorted[:ss])
        self.write_slice(indices, nslice-1, sindices[:ss])
        # Update caches
        self.update_caches(nslice-1, ssorted[:ss])
        # Shift the slice in the end to the beginning
        ssorted[:ss] = ssorted[ss:]; sindices[:ss] = sindices[ss:]


    def update_caches(self, nslice, ssorted):
        """Update the caches for faster lookups."""
        cs = self.chunksize
        ncs = self.nchunkslice
        tmp = self.tmp
        # update first & second cache bounds (ranges & bounds)
        tmp.ranges[nslice] = ssorted[[0,-1]]
        tmp.bounds[nslice] = ssorted[cs::cs]
        # update start & stop bounds
        tmp.abounds[nslice*ncs:(nslice+1)*ncs] = ssorted[0::cs]
        tmp.zbounds[nslice*ncs:(nslice+1)*ncs] = ssorted[cs-1::cs]
        # update median bounds
        smedian = ssorted[cs/2::cs]
        tmp.mbounds[nslice*ncs:(nslice+1)*ncs] = smedian
        tmp.mranges[nslice] = smedian[ncs/2]


    def reorder_slices(self, tmp):
        """Reorder completely the index at slice level.

        This method has to maintain the locality of elements in the
        ambit of ``blocks``, i.e. an element of a ``block`` cannot be
        sent to another ``block`` during this reordering.  This is
        *critical* for ``light`` indexes to be able to use this.

        This version of reorder_slices is optimized in that *two*
        complete slices are taken at a time (including the last row
        slice) so as to sort them.  Then, each new slice that is read is
        put at the end of this two-slice buffer, while the previous one
        is moved to the beginning of the buffer.  This is in order to
        better reduce the entropy of the regular part (i.e. all except
        the last row) of the index.

        A secondary effect of this is that it takes at least *twice* of
        memory than a previous version of reorder_slices() that only
        reorders on a slice-by-slice basis.  However, as this is more
        efficient than the old version, one can configure the slicesize
        to be smaller, so the memory consumption is barely similar.
        """

        tmp = self.tmp
        sorted = tmp.sorted; indices = tmp.indices
        if tmp:
            tmp_sorted = tmp.sorted2; tmp_indices = tmp.indices2
        else:
            tmp_sorted = tmp.sorted; tmp_indices = tmp.indices
        cs = self.chunksize
        ss = self.slicesize
        nsb = self.blocksize / self.slicesize
        nslices = self.nslices
        nblocks = self.nblocks
        nelementsSLR = self.nelementsSLR; nelementsILR = self.nelementsILR
        # Create the buffers for specifying the coordinates
        self.startl = numpy.empty(shape=2, dtype=numpy.uint64)
        self.stopl = numpy.empty(shape=2, dtype=numpy.uint64)
        self.stepl = numpy.ones(shape=2, dtype=numpy.uint64)
        # Create the buffer for reordering 2 slices at a time
        ssorted = numpy.empty(shape=ss*2, dtype=self.dtype)
        sindices = numpy.empty(shape=ss*2,
                               dtype=numpy.dtype('u%d' % self.indsize))

        # Iterate over each block.  No data should cross block
        # boundaries to avoid adressing problems with 16-bit indices.
        for nb in xrange(nblocks):
            # Bootstrap the process for reordering
            # Read the first slice in buffers
            nrow = nb * nsb
            self.read_slice(tmp_sorted, nrow, ssorted[:ss])
            self.read_slice(tmp_indices, nrow, sindices[:ss])

            # Loop over the remainding slices in block
            lrb = nrow + nsb
            if lrb > nslices:
                lrb = nslices
            nslice = nrow   # Just in case the loop behind executes nothing
            for nslice in xrange(nrow+1, lrb):
                self.reorder_slice(nslice, sorted, indices,
                                   ssorted, sindices,
                                   tmp_sorted, tmp_indices)

            # Write the first part of the buffers to the regular leaves
            self.write_slice(sorted, nslice, ssorted[:ss])
            self.write_slice(indices, nslice, sindices[:ss])
            # Update caches for this slice
            self.update_caches(nslice, ssorted[:ss])


    def swap_slices(self, mode="median"):
        "Swap slices in a superblock."

        tmp = self.tmp
        sorted = tmp.sorted
        indices = tmp.indices
        tmp_sorted = tmp.sorted2
        tmp_indices = tmp.indices2
        ncs = self.nchunkslice
        nss = self.superblocksize / self.slicesize
        nss2 = nss
        for sblock in xrange(self.nsuperblocks):
            # Protection for last superblock having less slices than nss
            remainingslices = self.nslices - sblock*nss
            if remainingslices < nss:
                nss2 = remainingslices
            if nss2 <= 1:
                break
            if mode == "start":
                ranges = tmp.ranges[sblock*nss:sblock*nss+nss2, 0]
            elif mode == "stop":
                ranges = tmp.ranges[sblock*nss:sblock*nss+nss2, 1]
            elif mode == "median":
                ranges = tmp.mranges[sblock*nss:sblock*nss+nss2]
            sranges_idx = ranges.argsort(kind=defsort)
            # Don't swap the superblock at all if one doesn't need to
            ndiff = (sranges_idx != numpy.arange(nss2)).sum()/2
            if ndiff*50 < nss2:
                # The number of slices to rearrange is less than 2.5%,
                # so skip the reordering of this superblock
                # (too expensive for such a little improvement)
                if self.verbose:
                    print "skipping reordering of superblock ->", sblock
                continue
            ns = sblock*nss2
            # Swap sorted and indices slices following the new order
            for i in xrange(nss2):
                idx = sranges_idx[i]
                # Swap sorted & indices slices
                oi = ns+i; oidx = ns+idx
                tmp_sorted[oi] = sorted[oidx]
                tmp_indices[oi] = indices[oidx]
                # Swap start, stop & median ranges
                tmp.ranges2[oi] = tmp.ranges[oidx]
                tmp.mranges2[oi] = tmp.mranges[oidx]
                # Swap chunk bounds
                tmp.bounds2[oi] = tmp.bounds[oidx]
                # Swap start, stop & median bounds
                j = oi*ncs; jn = (oi+1)*ncs
                xj = oidx*ncs; xjn = (oidx+1)*ncs
                tmp.abounds2[j:jn] = tmp.abounds[xj:xjn]
                tmp.zbounds2[j:jn] = tmp.zbounds[xj:xjn]
                tmp.mbounds2[j:jn] = tmp.mbounds[xj:xjn]
            # tmp -> originals
            for i in xrange(nss2):
                # Copy sorted & indices slices
                oi = ns+i
                sorted[oi] = tmp_sorted[oi]
                indices[oi] = tmp_indices[oi]
                # Copy start, stop & median ranges
                tmp.ranges[oi] = tmp.ranges2[oi]
                tmp.mranges[oi] = tmp.mranges2[oi]
                # Copy chunk bounds
                tmp.bounds[oi] = tmp.bounds2[oi]
                # Copy start, stop & median bounds
                j = oi*ncs; jn = (oi+1)*ncs
                tmp.abounds[j:jn] = tmp.abounds2[j:jn]
                tmp.zbounds[j:jn] = tmp.zbounds2[j:jn]
                tmp.mbounds[j:jn] = tmp.mbounds2[j:jn]


    def compute_overlaps(self, message, verbose):
        """Compute some statistics about overlaping of slices in index.

        It returns the following info:

        noverlaps -- The total number of slices that overlaps in index (int).
        multiplicity -- The number of times that a concrete slice overlaps
            with any other (array of ints).
        toverlap -- An ovelap index: the sum of the values in segment slices
            that overlaps divided by the entire range of values (float).
            This index is only computed for numerical types.
        """

        ranges = self.tmp.ranges[:]
        nslices = self.nslices
        if self.nelementsILR > 0:
            # Add the ranges corresponding to the last row
            rangeslr = numpy.array([self.bebounds[0], self.bebounds[-1]])
            ranges = numpy.concatenate((ranges, [rangeslr]))
            nslices += 1
        noverlaps = 0; soverlap = 0.; toverlap = -1.
        multiplicity = numpy.zeros(shape=nslices, dtype="int_")
        for i in xrange(nslices):
            for j in xrange(i+1, nslices):
                if ranges[i,1] > ranges[j,0]:
                    noverlaps += 1
                    multiplicity[j-i] += 1
                    if self.type != "string":
                        # Convert ranges into floats in order to allow
                        # doing operations with them without overflows
                        soverlap += float(ranges[i,1]) - float(ranges[j,0])

        # Return the overlap as the ratio between overlaps and entire range
        if self.type != "string":
            erange = float(ranges[-1,1]) - float(ranges[0,0])
            # Check that there is an effective range of values
            # Beware, erange can be negative in situations where
            # the values are suffering overflow. This can happen
            # specially on big signed integer values (on overflows,
            # the end value will become negative!).
            # Also, there is no way to compute overlap ratios for
            # non-numerical types. So, be careful and always check
            # that toverlap has a positive value (it must have been
            # initialized to -1. before) before using it.
            # F. Altet 2007-01-19
            if erange > 0:
                toverlap = soverlap / erange
        if verbose:
            print "overlaps (%s):" % message, noverlaps, toverlap
            print multiplicity
        return (noverlaps, multiplicity, toverlap)


    def restorecache(self):
        "Clean the limits cache and resize starts and lengths arrays"

        self.sorted.boundscache = ObjectCache(BOUNDS_MAX_SLOTS,
                                              BOUNDS_MAX_SIZE,
                                              'non-opt types bounds')
        """A cache for the bounds (2nd hash) data. Only used for
        non-optimized types searches."""
        self.limboundscache = ObjectCache(LIMBOUNDS_MAX_SLOTS,
                                          LIMBOUNDS_MAX_SIZE,
                                          'bounding limits')
        """A cache for bounding limits."""
        self.sortedLRcache = ObjectCache(SORTEDLR_MAX_SLOTS,
                                         SORTEDLR_MAX_SIZE,
                                         'last row chunks')
        """A cache for the last row chunks. Only used for searches in
        the last row, and mainly useful for small indexes."""
        self.starts = numpy.empty(shape=self.nrows, dtype=numpy.int32)
        self.lengths = numpy.empty(shape=self.nrows, dtype=numpy.int32)
        # Initialize the sorted array in extension
        self.sorted._initSortedSlice(self)
        self.dirtycache = False


    def search(self, item):
        """Do a binary search in this index for an item"""

        if self.dirtycache:
            self.restorecache()

        # An empty item means that the number of records is always
        # going to be empty, so we avoid further computation
        # (including looking up the limits cache).
        if not item:
            self.starts[:] = 0
            self.lengths[:] = 0
            return 0

        tlen = 0
        # Check whether the item tuple is in the limits cache or not
        nslot = self.limboundscache.getslot(item)
        if nslot >= 0:
            startlengths = self.limboundscache.getitem(nslot)
            # Reset the lengths array (not necessary for starts)
            self.lengths[:] = 0
            # Now, set the interesting rows
            for nrow in xrange(len(startlengths)):
                nrow2, start, length = startlengths[nrow]
                self.starts[nrow2] = start
                self.lengths[nrow2] = length
                tlen = tlen + length
            return tlen
        # The item is not in cache. Do the real lookup.
        sorted = self.sorted
        if self.nslices > 0:
            if self.type in self.opt_search_types:
                # The next are optimizations. However, they hide the
                # CPU functions consumptions from python profiles.
                # You may want to de-activate them during profiling.
                if self.type == "int32":
                    tlen = sorted._searchBinNA_i(*item)
                elif self.type == "int64":
                    tlen = sorted._searchBinNA_ll(*item)
                elif self.type == "float32":
                    tlen = sorted._searchBinNA_f(*item)
                elif self.type == "float64":
                    tlen = sorted._searchBinNA_d(*item)
                elif self.type == "uint32":
                    tlen = sorted._searchBinNA_ui(*item)
                elif self.type == "uint64":
                    tlen = sorted._searchBinNA_ull(*item)
                elif self.type == "int8":
                    tlen = sorted._searchBinNA_b(*item)
                elif self.type == "int16":
                    tlen = sorted._searchBinNA_s(*item)
                elif self.type == "uint8":
                    tlen = sorted._searchBinNA_ub(*item)
                elif self.type == "uint16":
                    tlen = sorted._searchBinNA_us(*item)
                else:
                    assert False, "This can't happen!"
            else:
                tlen = self.search_scalar(item, sorted)
        # Get possible remaining values in last row
        if self.nelementsSLR > 0:
            # Look for more indexes in the last row
            (start, stop) = self.searchLastRow(item)
            self.starts[-1] = start
            self.lengths[-1] = stop - start
            tlen += stop - start

        if self.limboundscache.couldenablecache():
            # Get a startlengths tuple and save it in cache.
            # This is quite slow, but it is a good way to compress
            # the bounds info. Moreover, the .couldenablecache()
            # is doing a good work so as to avoid computing this
            # when it is not necessary to do it.
            startlengths = []
            for nrow, length in enumerate(self.lengths):
                if length > 0:
                    startlengths.append((nrow, self.starts[nrow], length))
            # Compute the size of the recarray (aproximately)
            # The +1 at the end is important to avoid 0 lengths
            # (remember, the object headers take some space)
            size = len(startlengths) * 8 * 2 + 1
            # Put this startlengths list in cache
            self.limboundscache.setitem(item, startlengths, size)

        return tlen


    # This is an scalar version of search. It works with strings as well.
    def search_scalar(self, item, sorted):
        """Do a binary search in this index for an item."""
        tlen = 0
        # Do the lookup for values fullfilling the conditions
        for i in xrange(self.nslices):
            (start, stop) = sorted._searchBin(i, item)
            self.starts[i] = start
            self.lengths[i] = stop - start
            tlen += stop - start
        return tlen


    def searchLastRow(self, item):
        # Variable initialization
        item1, item2 = item
        bebounds = self.bebounds
        b0, b1 = bebounds[0], bebounds[-1]
        bounds = bebounds[1:-1]
        itemsize = self.dtype.itemsize
        sortedLRcache = self.sortedLRcache
        hi = self.nelementsSLR               # maximum number of elements
        rchunksize = self.chunksize // self.reduction

        nchunk = -1
        # Lookup for item1
        if item1 > b0:
            if item1 <= b1:
                # Search the appropriate chunk in bounds cache
                nchunk = bisect_left(bounds, item1)
                # Lookup for this chunk in cache
                nslot = sortedLRcache.getslot(nchunk)
                if nslot >= 0:
                    chunk = sortedLRcache.getitem(nslot)
                else:
                    begin = rchunksize*nchunk
                    end = rchunksize*(nchunk+1)
                    if end > hi:
                        end = hi
                    # Read the chunk from disk
                    chunk = self.sortedLR._readSortedSlice(
                        self.sorted, begin, end)
                    # Put it in cache.  It's important to *copy*
                    # the buffer, as it is reused in future reads!
                    # See bug #60 in xot.carabos.com
                    sortedLRcache.setitem(nchunk, chunk.copy(),
                                          (end-begin)*itemsize)
                start = bisect_left(chunk, item1)
                start += rchunksize*nchunk
            else:
                start = hi
        else:
            start = 0
        # Lookup for item2
        if item2 >= b0:
            if item2 < b1:
                # Search the appropriate chunk in bounds cache
                nchunk2 = bisect_right(bounds, item2)
                if nchunk2 != nchunk:
                    # Lookup for this chunk in cache
                    nslot = sortedLRcache.getslot(nchunk2)
                    if nslot >= 0:
                        chunk = sortedLRcache.getitem(nslot)
                    else:
                        begin = rchunksize*nchunk2
                        end = rchunksize*(nchunk2+1)
                        if end > hi:
                            end = hi
                        # Read the chunk from disk
                        chunk = self.sortedLR._readSortedSlice(
                            self.sorted, begin, end)
                        # Put it in cache.  It's important to *copy*
                        # the buffer, as it is reused in future reads!
                        # See bug #60 in xot.carabos.com
                        sortedLRcache.setitem(nchunk2, chunk.copy(),
                                              (end-begin)*itemsize)
                stop = bisect_right(chunk, item2)
                stop += rchunksize*nchunk2
            else:
                stop = hi
        else:
            stop = 0
        return (start, stop)


    def get_chunkmap(self):
        """Compute a map with the interesting chunks in index"""

        #t1 = time()
        ss = self.slicesize;  bs = self.blocksize
        nsb = self.nslicesblock;  nslices = self.nslices
        lbucket = self.lbucket;  indsize = self.indsize
        bucketsinblock = float(self.blocksize)/lbucket
        nchunks = long(math.ceil(float(self.nelements)/lbucket))
        chunkmap = numpy.zeros(shape=nchunks, dtype="bool")
        reduction = self.reduction
        starts = (self.starts-1)*reduction+1
        stops = (self.starts+self.lengths)*reduction
        starts[starts < 0] = 0    # All negative values set to zero
        indices = self.indices
        for nslice in xrange(self.nrows):
            start = starts[nslice];  stop = stops[nslice]
            if stop > start:
                idx = numpy.empty(shape=stop-start, dtype='u%d' % indsize)
                if nslice < nslices:
                    indices._readIndexSlice(nslice, start, stop, idx)
                else:
                    self.indicesLR._readIndexSlice(start, stop, idx)
                if indsize == 8:
                    idx /= lbucket
                elif indsize == 2:
                    # The chunkmap size cannot be never larger than 'int_'
                    idx = idx.astype("int_")
                    offset = long((nslice/nsb)*bucketsinblock)
                    idx += offset
                elif indsize == 1:
                    # The chunkmap size cannot be never larger than 'int_'
                    idx = idx.astype("int_")
                    offset = (nslice*ss)/lbucket
                    idx += offset
                chunkmap[idx] = True
        # The case lbucket < nrowsinchunk should only happen in tests
        nrowsinchunk = self.nrowsinchunk
        if lbucket != nrowsinchunk:
            # Map the 'coarse grain' chunkmap into the 'true' chunkmap
            nelements = self.nelements
            tnchunks = long(math.ceil(float(nelements)/nrowsinchunk))
            tchunkmap = numpy.zeros(shape=tnchunks, dtype="bool")
            ratio = float(lbucket)/nrowsinchunk
            idx = chunkmap.nonzero()[0]
            starts = (idx*ratio).astype('int_')
            stops = numpy.ceil((idx+1)*ratio).astype('int_')
            for i in range(len(idx)):
                tchunkmap[starts[i]:stops[i]] = True
            chunkmap = tchunkmap
        #self.tprof = round(time()-t1, 4)
        return chunkmap


    def getLookupRange(self, ops, limits, table):
        assert len(ops) in [1, 2]
        assert len(limits) in [1, 2]
        assert len(ops) == len(limits)

        column = self.column
        coldtype = column.dtype.base
        itemsize = coldtype.itemsize

        if len(limits) == 1:
            assert ops[0] in ['lt', 'le', 'eq', 'ge', 'gt']
            limit = limits[0]
            op = ops[0]
            if op == 'lt':
                range_ = (infType(coldtype, itemsize, sign=-1),
                          nextafter(limit, -1, coldtype, itemsize))
            elif op == 'le':
                range_ = (infType(coldtype, itemsize, sign=-1),
                          limit)
            elif op == 'gt':
                range_ = (nextafter(limit, +1, coldtype, itemsize),
                          infType(coldtype, itemsize, sign=+1))
            elif op == 'ge':
                range_ = (limit,
                          infType(coldtype, itemsize, sign=+1))
            elif op == 'eq':
                range_ = (limit, limit)

        elif len(limits) == 2:
            assert ops[0] in ('gt', 'ge') and ops[1] in ('lt', 'le')

            lower, upper = limits
            if lower > upper:
                # ``a <[=] x <[=] b`` is always false if ``a > b``.
                return ()

            if ops == ('gt', 'lt'):  # lower < col < upper
                range_ = (nextafter(lower, +1, coldtype, itemsize),
                          nextafter(upper, -1, coldtype, itemsize))
            elif ops == ('ge', 'lt'):  # lower <= col < upper
                range_ = (lower, nextafter(upper, -1, coldtype, itemsize))
            elif ops == ('gt', 'le'):  # lower < col <= upper
                range_ = (nextafter(lower, +1, coldtype, itemsize), upper)
            elif ops == ('ge', 'le'):  # lower <= col <= upper
                range_ = (lower, upper)

        return range_


    def _f_remove(self, recursive=False):
        """Remove this Index object"""

        # Index removal is always recursive,
        # no matter what `recursive` says.
        super(Index, self)._f_remove(True)


    def __str__(self):
        """This provides a more compact representation than __repr__"""
        return "Index(%s, type=%s, shape=%s, chunksize=%s)" % \
               (self.nelements, self.type, self.shape, self.sorted.chunksize)


    def __repr__(self):
        """This provides more metainfo than standard __repr__"""

        cpathname = self.column.table._v_pathname + ".cols." + self.column.name
        retstr = """%s (Index for column %s)
  nelements := %s
  chunksize := %s
  slicesize := %s
  blocksize := %s
  superblocksize := %s
  filters := %s
  dirty := %s
  byteorder := %r""" % (self._v_pathname, cpathname,
                        self.nelements,
                        self.chunksize, self.slicesize,
                        self.blocksize, self.superblocksize,
                        self.filters, self.dirty,
                        self.byteorder)
        retstr += "\n  sorted := %s" % self.sorted
        retstr += "\n  indices := %s" % self.indices
        retstr += "\n  ranges := %s" % self.ranges
        retstr += "\n  bounds := %s" % self.bounds
        retstr += "\n  sortedLR := %s" % self.sortedLR
        retstr += "\n  indicesLR := %s" % self.indicesLR
        return retstr



class IndexesDescG(NotLoggedMixin, Group):
    _c_classId = 'DINDEX'

    def _g_widthWarning(self):
        warnings.warn(
            "the number of indexed columns on a single description group "
            "is exceeding the recommended maximum (%d); "
            "be ready to see PyTables asking for *lots* of memory "
            "and possibly slow I/O" % MAX_GROUP_WIDTH, PerformanceWarning )


class IndexesTableG(NotLoggedMixin, Group):
    _c_classId = 'TINDEX'

    def _getauto(self):
        if 'AUTO_INDEX' not in self._v_attrs:
            return defaultAutoIndex
        return self._v_attrs.AUTO_INDEX
    def _setauto(self, auto):
        self._v_attrs.AUTO_INDEX = bool(auto)
    def _delauto(self):
        del self._v_attrs.AUTO_INDEX
    auto = property(_getauto, _setauto, _delauto)

    def _getfilters(self):
        if 'FILTERS' not in self._v_attrs:
            return defaultIndexFilters
        return self._v_attrs.FILTERS
    _setfilters = Group._g_setfilters
    _delfilters = Group._g_delfilters
    filters = property(_getfilters, _setfilters, _delfilters)

    def _g_postInitHook(self):
        super(IndexesTableG, self)._g_postInitHook()
        if self._v_new and defaultIndexFilters is not None:
            self._v_attrs._g__setattr('FILTERS', defaultIndexFilters)

    def _g_widthWarning(self):
        warnings.warn(
            "the number of indexed columns on a single table "
            "is exceeding the recommended maximum (%d); "
            "be ready to see PyTables asking for *lots* of memory "
            "and possibly slow I/O" % MAX_GROUP_WIDTH, PerformanceWarning )

    def _g_checkName(self, name):
        if not name.startswith('_i_'):
            raise ValueError(
                "names of index groups must start with ``_i_``: %s" % name )


class OldIndex(NotLoggedMixin, Group):
    """This is meant to hide indexes of PyTables 1.x files."""
    _c_classId = 'CINDEX'



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
