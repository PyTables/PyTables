########################################################################
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

import sys, os, subprocess

import bisect
from time import time, clock
import os, os.path
import tempfile
import sys

import numpy

from idxutils import calcChunksize, calcoptlevels, opts_pack, opts_unpack, \
     nextafter, infType, show_stats

from tables import indexesExtension
from tables import utilsExtension
from tables.attributeset import AttributeSet
from tables.node import NotLoggedMixin
from tables.atom import Int64Atom, Atom
from tables.earray import EArray
from tables.carray import CArray
from tables.leaf import Filters
from tables.indexes import CacheArray, LastRowArray, IndexArray
from tables.group import Group
from tables.path import joinPath
from tables.parameters import (
    LIMBOUNDS_MAX_SLOTS, LIMBOUNDS_MAX_SIZE,
    BOUNDS_MAX_SLOTS, BOUNDS_MAX_SIZE,
    MAX_GROUP_WIDTH )
from tables.exceptions import PerformanceWarning

from tables.lrucacheExtension import ObjectCache


__version__ = "$Revision: 1236 $"


# default version for INDEX objects
#obversion = "1.0"    # Version of indexes in PyTables 1.x series
obversion = "2.0"    # Version of indexes in PyTables Pro 2.x series


debug = False
#debug = True  # Uncomment this for printing sizes purposes
profile = False
#profile = True  # uncomment for profiling purposes only


# The default method for sorting
defsort = "quicksort"

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
opt_search_types = ("int32", "int64", "float32", "float64")

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
        Whether the index is dirty or not.

        Dirty indexes are out of sync with column data, so they exist
        but they are not usable.

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
        nblocks = self.nelements / self.superblocksize
        if self.nelements % self.superblocksize > 0:
            nblocks += 1
        return nblocks
    nsuperblocks = property(_g_nsuperblocks , None, None,
        "The total number of superblocks in index.")

    def _g_nblocks(self):
        nblocks = self.nelements / self.blocksize
        if self.nelements % self.blocksize > 0:
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

    # </properties>


    def __init__(self, parentNode, name,
                 atom=None, column=None,
                 title="", filters=None,
                 expectedrows=0,
                 byteorder=None,
                 blocksizes=None,
                 new=True):
        """Create an Index instance.

        Keyword arguments:

        atom -- An Atom object representing the shape and type of the
            atomic objects to be saved. Only scalar atoms are
            supported.

        column -- The column object to be indexed

        title -- Sets a TITLE attribute of the Index entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object. If not specified, the ZLIB
            & shuffle will be activated by default (i.e., they are not
            inherited from the parent, that is, the Table).

        expectedrows -- Represents an user estimate about the number
            of row slices that will be added to the growable dimension
            in the IndexArray object.

        byteorder -- The byteorder of the index datasets *on-disk*.

        blocksizes -- The four main sizes of the compound blocks in
            index datasets (a low level parameter).

        """

        self._v_version = None
        """The object version of this index."""

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
        self.column = column
        """The `Column` instance for the indexed column."""

        self.nrows = None
        """The total number of slices in the index."""
        self.nelements = None
        """The number of currently indexed row for this column."""
        self.blocksizes = blocksizes
        """The four main sizes of the compound blocks (if specified)."""
        self.opts = (False,)*4
        """The four optimization procedures applied."""
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
            # Set-up some variables from info on disk and return
            attrs = self._v_attrs
            self.superblocksize = attrs.superblocksize
            self.blocksize = attrs.blocksize
            self.slicesize = attrs.slicesize
            self.chunksize = attrs.chunksize
            self.blocksizes = (self.superblocksize, self.blocksize,
                               self.slicesize, self.chunksize)
            self.opts = opts_unpack(attrs.opts)
            sorted = self.sorted
            self.dtype = sorted.atom.dtype
            self.type = sorted.atom.type
            self.filters = sorted.filters
            # Some sanity checks for slicesize and chunksize
            assert self.slicesize == sorted.shape[1], "Wrong slicesize"
            assert self.chunksize == sorted._v_chunkshape[1], "Wrong chunksize"
            # The number of elements is at the end of the indices array
            nelementsLR = self.indicesLR[-1]
            self.nrows = sorted.nrows
            self.nelements = self.nrows * self.slicesize + nelementsLR
            self.nelementsLR = nelementsLR
            if nelementsLR > 0:
                self.nrows += 1
            # Get the bounds as a cache (this has to remain here!)
            nboundsLR = (nelementsLR - 1 ) // self.chunksize
            if nboundsLR < 0:
                nboundsLR = 0 # correction for -1 bounds
            nboundsLR += 2 # bounds + begin + end
            # All bounds values (+begin+end) are at the end of sortedLR
            self.bebounds = self.sortedLR[nelementsLR:nelementsLR+nboundsLR]
            return

        # The index is new. Initialize the values
        self.nrows = 0
        self.nelements = 0
        self.nelementsLR = 0

        # Set the filters for this object (they are *not* inherited)
        self.filters = filters = self._v_new_filters

        # Compute the superblocksize, blocksize, slicesize and chunksize values
        # (in case these parameters haven't been passed to the constructor)
        if self.blocksizes is None:
            self.blocksizes = calcChunksize(self.expectedrows)
        (self.superblocksize, self.blocksize,
         self.slicesize, self.chunksize) = self.blocksizes
        if debug:
            print "blocksizes:", self.blocksizes

        # Save them on disk as attributes
        self._v_attrs.superblocksize = numpy.int64(self.superblocksize)
        self._v_attrs.blocksize = numpy.int64(self.blocksize)
        self._v_attrs.slicesize = numpy.uint32(self.slicesize)
        self._v_attrs.chunksize = numpy.uint32(self.chunksize)

        # Save the optimization procedures (this attribute will be
        # overwritten in case an optimization is made later on)
        self._v_attrs.opts = opts_pack(self.opts)

        # Create the IndexArray for sorted values
        atom = Atom.from_dtype(self.dtype)
        sorted = IndexArray(self, 'sorted', atom, "Sorted Values",
                            filters, self.byteorder)

        # Create the IndexArray for index values
        IndexArray(self, 'indices', Int64Atom(), "Reverse Indices",
                   filters, self.byteorder)

        # Create the cache for range values  (1st order cache)
        CacheArray(self, 'ranges', atom, (0,2), "Range Values", filters,
                   self.expectedrows//self.slicesize,
                   byteorder=self.byteorder)
        # median ranges
        EArray(self, 'mranges', atom, (0,), "Median ranges", filters,
               byteorder=self.byteorder, _log=False)

        # Create the cache for boundary values (2nd order cache)
        nbounds_inslice = (self.slicesize - 1 ) // self.chunksize
        CacheArray(self, 'bounds', atom, (0, nbounds_inslice),
                   "Boundary Values", filters,
                   self.expectedrows//self.chunksize,
                   (1, nbounds_inslice), byteorder=self.byteorder)

        # begin, end & median bounds (only for numerical types)
        EArray(self, 'abounds', atom, (0,), "Start bounds",
               byteorder=self.byteorder, _log=False)
        EArray(self, 'zbounds', atom, (0,), "End bounds", filters,
               byteorder=self.byteorder, _log=False)
        EArray(self, 'mbounds', atom, (0,), "Median bounds", filters,
               byteorder=self.byteorder, _log=False)

        # Create the Array for last (sorted) row values + bounds
        shape = (self.slicesize + 2 + nbounds_inslice,)
        sortedLR = LastRowArray(self, 'sortedLR', atom, shape,
                                "Last Row sorted values + bounds",
                                filters, (self.chunksize,),
                                byteorder=self.byteorder)

        # Create the Array for reverse indexes in last row
        shape = (self.slicesize,)     # enough for indexes and length
        LastRowArray(self, 'indicesLR', Int64Atom(), shape,
                     "Last Row reverse indices",
                     filters, (self.chunksize,),
                     byteorder=self.byteorder)

        # All bounds values (+begin+end) are uninitialized in creation time
        self.bebounds = None

        # The starts and lengths initialization
        self.starts = numpy.empty(shape=self.nrows, dtype=numpy.int32)
        """Where the values fulfiling conditions starts for every slice."""
        self.lengths = numpy.empty(shape=self.nrows, dtype=numpy.int32)
        """Lengths of the values fulfilling conditions for every slice."""


    def _g_updateDependent(self):
        super(Index, self)._g_updateDependent()
        self.column._updateIndexLocation(self)


    def append(self, xarr):
        """Append the array to the index objects"""

        if profile: tref = time()
        if profile: show_stats("Entrant en append", tref)
        arr = xarr.pop()
        sorted = self.sorted
        offset = sorted.nrows * self.slicesize
        # As len(arr) < 2**32, we can choose uint32 for representing idx
        idx = numpy.arange(0, len(arr), dtype='uint32')
        # In-place sorting
        if profile: show_stats("Abans de keysort", tref)
        indexesExtension.keysort(arr, idx)
        # Save the sorted array
        sorted.append(arr)
        cs = self.chunksize
        ncs = self.nchunkslice
        self.ranges.append([arr[[0,-1]]])
        self.bounds.append([arr[cs::cs]])
        self.abounds.append(arr[0::cs])
        self.zbounds.append(arr[cs-1::cs])
        # Compute the medians
        smedian = arr[cs/2::cs]
        self.mbounds.append(smedian)
        self.mranges.append([smedian[ncs/2]])
        if profile: show_stats("Abans d'esborrar arr i smedian", tref)
        del arr, smedian   # delete references to arr
        # Append the indices
        if profile: show_stats("Abans d'apujar a 64 bits", tref)
        idx = idx.astype('int64')
        if profile: show_stats("Abans de sumar offset", tref)
        idx += offset
        if profile: show_stats("Abans de guardar indexos", tref)
        self.indices.append(idx)
        if profile: show_stats("Abans d'esborrar indexos", tref)
        del idx
        # Update nrows after a successful append
        self.nrows = sorted.nrows
        self.nelements = self.nrows * self.slicesize
        self.nelementsLR = 0  # reset the counter of the last row index to 0
        self.dirtycache = True   # the cache is dirty now
        if profile: show_stats("Eixint d'append", tref)


    def appendLastRow(self, xarr):
        """Append the array to the last row index objects"""

        if profile: tref = time()
        if profile: show_stats("Entrant a appendLR", tref)
        arr = xarr.pop()
        nelementsLR = len(arr)
        # compute the elements in the last row sorted & bounds array
        sorted = self.sorted
        indicesLR = self.indicesLR
        sortedLR = self.sortedLR
        offset = sorted.nrows * self.slicesize
        # As len(arr) < 2**32, we can choose uint32 for representing idx
        idx = numpy.arange(0, len(arr), dtype='uint32')
        # In-place sorting
        if profile: show_stats("Abans de keysort", tref)
        indexesExtension.keysort(arr, idx)
        # Build the cache of bounds
        self.bebounds = numpy.concatenate((arr[::self.chunksize],
                                           [arr[-1]]))
        # The number of elements is at the end of the indices array
        indicesLR[-1] = nelementsLR
        # Save the number of elements, bounds and sorted values
        # at the end of the sorted array
        offset2 = len(self.bebounds)
        sortedLR[nelementsLR:nelementsLR+offset2] = self.bebounds
        if profile: show_stats("Abans de guadar sorted", tref)
        sortedLR[:nelementsLR] = arr
        if profile: show_stats("Abans d'esborrar sorted", tref)
        del arr
        # Save the reverse index array
        if profile: show_stats("Abans d'apujar indexos", tref)
        idx = idx.astype('int64')
        if profile: show_stats("Abans de sumar offset", tref)
        idx += offset
        if profile: show_stats("Abans de guardar indexos", tref)
        indicesLR[:len(idx)] = idx
        if profile: show_stats("Abans d'esborrar indexos", tref)
        del idx
        # Update nelements after a successful append
        self.nrows = sorted.nrows + 1
        self.nelements = sorted.nrows * self.slicesize + nelementsLR
        self.nelementsLR = nelementsLR
        self.dirtycache = True   # the cache is dirty now
        if profile: show_stats("Eixint de appendLR", tref)


    def optimize(self, level, opts=None, testmode=False, verbose=False):
        """Optimize an index so as to allow faster searches.

        level -- The desired level of optimization for the index.

        opts -- A low level specification of the optimizations for the
            index. It is a tuple with the format ``(optmedian,
            optstarts, optstops, optfull)``.

        testmode -- If True, a optimization specific to be used in
            tests is used (basically, it does not depend on anything
            but the `level` argument). This is not considered if
            `opts` is specified.

        verbose -- If True, messages about the progress of the
            optimization process are printed out.

        """

        if verbose == True:
            self.verbose = True
        else:
            self.verbose = debug

        # Initialize last_tover and last_nover
        self.last_tover = 0
        self.last_nover = 0

        # Optimize only when we have more than one slice
        if self.nslices <= 1:
            if self.verbose:
                print "Less than 1 slice. Skipping optimization!"
            return

        if self.verbose:
            (nover, mult, tover) = self.compute_overlaps("init", self.verbose)

        # Compute the correct optimizations for optim level (if needed)
        if opts is None:
            opts = calcoptlevels(self.nblocks, level, testmode)
        optmedian, optstarts, optstops, optfull = opts
        if debug:
            print "optvalues:", opts
        # Overwrite the new optimizations in opts (a packed attribute)
        self._v_attrs.opts = opts_pack(opts)

        # Start the optimization process
        if optmedian or optstarts or optstops or optfull:
            create_tmp = True
            swap_done = True
        else:
            create_tmp = False
            swap_done = False
        while True:
            if create_tmp:
                if self.swap('create'):
                    swap_done = False  # No swap has been done!
                    break
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
        if create_tmp:
            self.cleanup_temps()
            # This should be called only if the overlaps are not zero
            # and in full optimization mode!
            (nover, mult, tover) = self.compute_overlaps("", False)
            if nover > 0 and level == 9:
                # Do a last pass by reordering the slices alone
                self.swap('reorder_slices')
            if swap_done:
                # the memory data cache is dirty now
                self.dirtycache = True
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
        if what == "create":
            self.create_temps()
        elif what == "chunks":
            self.swap_chunks(mode)
        elif what == "slices":
            self.swap_slices(mode)
        elif what == "reorder_slices":
            # Reorder completely the index at slice level
            self.reorder_slices(tmp=False)
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


    def create_temps(self):
        "Create some temporary objects for slice sorting purposes."

        # The algorithms for doing the swap can be optimized so that
        # one should be necessary to create temporaries for keeping just
        # the contents of a single superblock.
        # F. Altet 2007-01-03
        # Build the name of the temporary file
        dirname = os.path.dirname(self._v_file.filename)
        fd, self.tmpfilename = tempfile.mkstemp(".tmp", "pytables-", dirname)
        # Close the file descriptor so as to avoid leaks
        os.close(fd)
        # Create the proper PyTables file
        self.tmpfile = self._openFile(self.tmpfilename, "w")
        self.tmp = self.tmpfile.root
        cs = self.chunksize
        ss = self.slicesize
        filters = self.filters
        # temporary sorted & indices arrays
        shape = (self.nrows, ss)
        atom = Atom.from_dtype(self.dtype)
        CArray(self.tmp, 'sorted', atom, shape,
               "Temporary sorted", filters, chunkshape=(1,cs))
        CArray(self.tmp, 'indices', Int64Atom(), shape,
               "Temporary indices", filters, chunkshape=(1,cs))
        # temporary bounds
        nbounds_inslice = (ss - 1) // cs
        shape = (self.nslices, nbounds_inslice)
        CArray(self.tmp, 'bounds', atom, shape, "Temp chunk bounds",
               filters, chunkshape=(cs, nbounds_inslice))
        shape = (self.nchunks,)
        CArray(self.tmp, 'abounds', atom, shape, "Temp start bounds",
               filters, chunkshape=(cs,))
        CArray(self.tmp, 'zbounds', atom, shape, "Temp end bounds",
               filters, chunkshape=(cs,))
        CArray(self.tmp, 'mbounds', atom, shape, "Median bounds",
               filters, chunkshape=(cs,))
        # temporary ranges
        CArray(self.tmp, 'ranges', atom, (self.nslices, 2),
               "Temporary range values", filters, chunkshape=(cs,2))
        CArray(self.tmp, 'mranges', atom, (self.nslices,),
               "Median ranges", filters, chunkshape=(cs,))


    def cleanup_temps(self):
        "Delete the temporaries for sorting purposes."
        if self.verbose:
            print "Deleting temporaries..."
        self.tmp = None
        self.tmpfile.close()
        os.remove(self.tmpfilename)
        self.tmpfilename = None


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
        sorted = self.sorted
        indices = self.indices
        tmp_sorted = self.tmp.sorted
        tmp_indices = self.tmp.indices
        cs = self.chunksize
        ss = self.slicesize
        ncs = self.nchunkslice
        nsb = self.nslicesblock
        ncb = ncs * nsb
        ncb2 = ncb
        boundsobj = self._v_file.getNode(self, boundsnames[mode])
        for nblock in xrange(self.nblocks):
            # Protection for last block having less chunks than ncb
            remainingchunks = self.nchunks - nblock*ncb
            if remainingchunks < ncb:
                # To avoid reordering the chunks in last row (slice)
                # Implementing this suppose to complicate quite a bit
                # the code for index optimization and perhaps this is
                # not worth the effort.
                # What has finally been implemented is an algorithm
                # for reordering *two* slices at a time (including the
                # last row slice, see self.reorder_slices). This is
                # enough to make the last row to participate in the
                # whole index reordering (and hence, significantly
                # reducing the index entropy)
                # F. Altet 2007-04-12
                ncb2 = (remainingchunks/ncs)*ncs
            if ncb2 <= 1:
                # if only zero or one chunks remains we are done
                break
            nslices = ncb2/ncs
            bounds = boundsobj[nblock*ncb:nblock*ncb+ncb2]
            sbounds_idx = bounds.argsort(kind=defsort)
            # Don't swap the block at all if it doesn't need to
            ndiff = (sbounds_idx != numpy.arange(ncb2)).sum()/2
            if ndiff*20 < ncb2:
                # The number of chunks to rearrange is less than 5%,
                # so skip the reordering of this superblock
                # (too expensive for such a little improvement)
                if self.verbose:
                    print "skipping reordering of block-->", nblock, ndiff, ncb2
                continue
            # Swap sorted and indices following the new order
            offset = nblock*nsb
            self.get_neworder(sbounds_idx, sorted, tmp_sorted,
                              nslices, offset, self.dtype)
            self.get_neworder(sbounds_idx, indices, tmp_indices,
                              nslices, offset, 'int64')
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


    # Read version for LastRow
    def read_sliceLR(self, where, buffer):
        """Read a slice from the `where` dataset and put it in `buffer`."""
        startl = numpy.array([0], dtype=numpy.uint64)
        stopl = numpy.array([buffer.size], dtype=numpy.uint64)
        stepl = numpy.array([1], dtype=numpy.uint64)
        where._g_readSlice(startl, stopl, stepl, buffer)


    # Write version for LastRow
    def write_sliceLR(self, where, buffer):
        """Write a slice from the `where` dataset with the `buffer` data."""
        startl = numpy.array([0], dtype=numpy.uint64)
        countl = numpy.array([buffer.size], dtype=numpy.uint64)
        stepl = numpy.array([1], dtype=numpy.uint64)
        where._modify(startl, stepl, countl, buffer)


    def reorder_slice(self, nslice, sorted, indices, ssorted, sindices,
                      tmp_sorted, tmp_indices):
        """Copy & reorder the slice in source to final destination."""
        ss = self.slicesize
        # Load the second part in buffers
        self.read_slice(tmp_sorted, nslice, ssorted[ss:])
        self.read_slice(tmp_indices, nslice, sindices[ss:])
        indexesExtension.keysort(ssorted, sindices)
        # Write the first part of the buffers to the regular indices
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
        # update first & second cache bounds (ranges & bounds)
        self.ranges[nslice] = ssorted[[0,-1]]
        self.bounds[nslice] = ssorted[cs::cs]
        # update start & stop bounds
        self.abounds[nslice*ncs:(nslice+1)*ncs] = ssorted[0::cs]
        self.zbounds[nslice*ncs:(nslice+1)*ncs] = ssorted[cs-1::cs]
        # update median bounds
        smedian = ssorted[cs/2::cs]
        self.mbounds[nslice*ncs:(nslice+1)*ncs] = smedian
        self.mranges[nslice] = smedian[ncs/2]


    def reorder_slices(self, tmp):
        """Reorder completely the index at slice level (optim version).

        This version of reorder_slices is optimized in that *two*
        complete slices are taken at a time (including the last row
        slice) so as to sort them.  Then, each new slice that is read
        is put at the end of this two-slice buffer, while the previous
        one is moved to the beginning of the buffer. This is in order
        to make the last row to participate in the whole index
        reordering (and hence, significantly reducing the index
        entropy). Also, the new algorithm seems to be better at
        reducing the entropy of the regular part (i.e.  all except the
        last row) of the index.

        A secondary effect of this is that it takes at least *twice*
        of memory than regular reorder_slices(). However, as this is
        more efficient than the previous reorder_slices version (that
        used just one slice), one can configure the slicesize to be
        smaller.
        """

        sorted = self.sorted; indices = self.indices
        if tmp:
            tmp_sorted = self.tmp.sorted; tmp_indices = self.tmp.indices
        else:
            tmp_sorted = self.sorted; tmp_indices = self.indices
        cs = self.chunksize
        ss = self.slicesize
        nss = self.superblocksize / self.slicesize
        nelementsLR = self.nelementsLR
        # Create the buffers for specifying the coordinates
        self.startl = numpy.empty(shape=2, dtype=numpy.uint64)
        self.stopl = numpy.empty(shape=2, dtype=numpy.uint64)
        self.stepl = numpy.ones(shape=2, dtype=numpy.uint64)
        # Create the buffer for reordering 2 slices at a time
        ssorted = numpy.empty(shape=ss*2, dtype=self.dtype)
        sindices = numpy.empty(shape=ss*2, dtype=numpy.uint64)

        # Bootstrap the process for reordering
        # Read the first slice in buffers
        self.read_slice(tmp_sorted, 0, ssorted[:ss])
        self.read_slice(tmp_indices, 0, sindices[:ss])

        # Loop over the rest of slices in block
        for nslice in xrange(1, sorted.nrows):
            self.reorder_slice(nslice, sorted, indices, ssorted, sindices,
                               tmp_sorted, tmp_indices)

        # End the process (enrolling the lastrow if necessary)
        if nelementsLR > 0:
            sortedLR = self.sortedLR; indicesLR = self.indicesLR
            # Shrink the ssorted and sindices arrays to the minimum
            ssorted2 = ssorted[:ss+nelementsLR]; sortedlr = ssorted2[ss:]
            sindices2 = sindices[:ss+nelementsLR]; indiceslr = sindices2[ss:]
            # Read the last row info in the second part of the buffer
            self.read_sliceLR(sortedLR, sortedlr)
            self.read_sliceLR(indicesLR, indiceslr)
            indexesExtension.keysort(ssorted2, sindices2)
            # Write the second part of the buffers to the lastrow indices
            self.write_sliceLR(sortedLR, sortedlr)
            self.write_sliceLR(indicesLR, indiceslr)
            # Update the caches for last row
            bebounds = numpy.concatenate((sortedlr[::cs], [sortedlr[-1]]))
            sortedLR[nelementsLR:nelementsLR+len(bebounds)] = bebounds
            self.bebounds = bebounds
        # Write the first part of the buffers to the regular indices
        self.write_slice(sorted, nslice, ssorted[:ss])
        self.write_slice(indices, nslice, sindices[:ss])
        # Update caches for this slice
        self.update_caches(nslice, ssorted[:ss])


    def swap_slices(self, mode="median"):
        "Swap slices in a superblock."

        sorted = self.sorted
        indices = self.indices
        tmp_sorted = self.tmp.sorted
        tmp_indices = self.tmp.indices
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
                ranges = self.ranges[sblock*nss:sblock*nss+nss2, 0]
            elif mode == "stop":
                ranges = self.ranges[sblock*nss:sblock*nss+nss2, 1]
            elif mode == "median":
                ranges = self.mranges[sblock*nss:sblock*nss+nss2]
            sranges_idx = ranges.argsort(kind=defsort)
            # Don't swap the superblock at all if it doesn't need to
            ndiff = (sranges_idx != numpy.arange(nss2)).sum()/2
            if ndiff*50 < nss2:
                # The number of slices to rearrange is less than 2.5%,
                # so skip the reordering of this superblock
                # (too expensive for such a little improvement)
                if self.verbose:
                    print "skipping reordering of superblock-->", sblock
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
                self.tmp.ranges[oi] = self.ranges[oidx]
                self.tmp.mranges[oi] = self.mranges[oidx]
                # Swap chunk bounds
                self.tmp.bounds[oi] = self.bounds[oidx]
                # Swap start, stop & median bounds
                j = oi*ncs; jn = (oi+1)*ncs
                xj = oidx*ncs; xjn = (oidx+1)*ncs
                self.tmp.abounds[j:jn] = self.abounds[xj:xjn]
                self.tmp.zbounds[j:jn] = self.zbounds[xj:xjn]
                self.tmp.mbounds[j:jn] = self.mbounds[xj:xjn]
            # tmp --> originals
            for i in xrange(nss2):
                # Copy sorted & indices slices
                oi = ns+i
                sorted[oi] = tmp_sorted[oi]
                indices[oi] = tmp_indices[oi]
                # Copy start, stop & median ranges
                self.ranges[oi] = self.tmp.ranges[oi]
                self.mranges[oi] = self.tmp.mranges[oi]
                # Copy chunk bounds
                self.bounds[oi] = self.tmp.bounds[oi]
                # Copy start, stop & median bounds
                j = oi*ncs; jn = (oi+1)*ncs
                self.abounds[j:jn] = self.tmp.abounds[j:jn]
                self.zbounds[j:jn] = self.tmp.zbounds[j:jn]
                self.mbounds[j:jn] = self.tmp.mbounds[j:jn]


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

        ranges = self.ranges[:]
        nslices = self.nslices
        if self.nelementsLR > 0:
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
        self.starts = numpy.empty(shape=self.nrows, dtype = numpy.int32)
        self.lengths = numpy.empty(shape=self.nrows, dtype = numpy.int32)
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
        if sorted.nrows > 0:
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
                else:
                    assert False, "This can't happen!"
            else:
                tlen = self.search_scalar(item, sorted)
        # Get possible remaing values in last row
        if self.nelementsLR > 0:
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


    # This is an scalar version of search. It works well with strings as well.
    def search_scalar(self, item, sorted):
        """Do a binary search in this index for an item."""
        tlen = 0
        # Do the lookup for values fullfilling the conditions
        for i in xrange(sorted.nrows):
            (start, stop) = sorted._searchBin(i, item)
            self.starts[i] = start
            self.lengths[i] = stop - start
            tlen += stop - start
        return tlen


    def searchLastRow(self, item):
        item1, item2 = item
        item1done = 0; item2done = 0

        #t1=time()
        hi = self.nelementsLR               # maximum number of elements
        bebounds = self.bebounds
        assert hi == self.nelements - self.sorted.nrows * self.slicesize
        begin = bebounds[0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
            item1done = 1
        if item2 < begin:
            result2 = 0
            item2done = 1
        if item1done and item2done:
            return (result1, result2)
        # Then, look for items at the end of the sorted slice
        end = bebounds[-1]
        if not item1done:
            if item1 > end:
                result1 = hi
                item1done = 1
        if not item2done:
            if item2 >= end:
                result2 = hi
                item2done = 1
        if item1done and item2done:
            return (result1, result2)
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of the slice for item1
        bounds = bebounds[1:-1] # Get the bounds array w/out begin and end
        readSliceLR = self.sortedLR._readSortedSlice
        nchunk = -1
        if not item1done:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect.bisect_left(bounds, item1)
            begin = self.chunksize*nchunk
            end = self.chunksize*(nchunk+1)
            if end > hi:
                end = hi
            chunk = readSliceLR(self.sorted, begin, end)
            result1 = bisect.bisect_left(chunk, item1)
            result1 += self.chunksize*nchunk
        # Lookup in the middle of the slice for item2
        if not item2done:
            # Search the appropriate chunk in bounds cache
            nchunk2 = bisect.bisect_right(bounds, item2)
            if nchunk2 <> nchunk:
                begin = self.chunksize*nchunk2
                end = self.chunksize*(nchunk2+1)
                if end > hi:
                    end = hi
                chunk = readSliceLR(self.sorted, begin, end)
            result2 = bisect.bisect_right(chunk, item2)
            result2 += self.chunksize*nchunk2
        #t = time()-t1
        #print "time searching indices (last row):", round(t*1000, 3), "ms"
        return (result1, result2)


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
            assert ops[0] in ['gt', 'ge'] and ops[1] in ['lt', 'le']

            lower, upper = limits
            if lower > upper:
                # ``a <[=] x <[=] b`` is always false if ``a > b``.
                return ()

            if ops == ['gt', 'lt']:  # lower < col < upper
                range_ = (nextafter(lower, +1, coldtype, itemsize),
                          nextafter(upper, -1, coldtype, itemsize))
            elif ops == ['ge', 'lt']:  # lower <= col < upper
                range_ = (lower, nextafter(upper, -1, coldtype, itemsize))
            elif ops == ['gt', 'le']:  # lower < col <= upper
                range_ = (nextafter(lower, +1, coldtype, itemsize), upper)
            elif ops == ['ge', 'le']:  # lower <= col <= upper
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
