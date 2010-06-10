#######################################################################
#
#       License: See http://www.pytables.org/moin/PyTablesProPricing
#       Created: February 13, 2007
#       Author:  Ivan Vilata - ivan@selidor.net
#
#       $Id$
#
########################################################################

"""Here is defined the Table class (pro)."""

import warnings, math
import numpy

from tables.atom import Atom
from tables.exceptions import NoSuchNodeError
from tables.index import (
    defaultAutoIndex, Index, IndexesDescG, IndexesTableG )
from tables.lrucacheExtension import ObjectCache, NumCache
import numexpr
from tables._table_common import _indexNameOf, _indexPathnameOf

profile = False
#profile = True  # Uncomment for profiling
if profile:
    from time import time
    from tables.utils import show_stats


__version__ = "$Revision$"


class NailedDict(object):

    """A dictionary which ignores its items when it has nails on it."""

    def __init__(self, maxentries):
        self.maxentries = maxentries
        self._cache = {}
        self._nailcount = 0

    # Only a restricted set of dictionary methods are supported.  That
    # is why we buy instead of inherit.

    # The following are intended to be used by ``Table`` code changing
    # the set of usable indexes.

    def clear(self):
        self._cache.clear()
    def nail(self):
        self._nailcount += 1
    def unnail(self):
        self._nailcount -= 1

    # The following are intended to be used by ``Table`` code handling
    # conditions.

    def __contains__(self, key):
        if self._nailcount > 0:
            return False
        return key in self._cache

    def __getitem__(self, key):
        if self._nailcount > 0:
            raise KeyError(key)
        return self._cache[key]

    def get(self, key, default=None):
        if self._nailcount > 0:
            return default
        return self._cache.get(key, default)

    def __setitem__(self, key, value):
        if self._nailcount > 0:
            return
        cache = self._cache
        # Protection against growing the cache too much
        if len(cache) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = self.maxentries / 10
            for k in cache.keys()[:entries_to_remove]:
                del cache[k]
        cache[key] = value


def _table__setautoIndex(self, auto):
    auto = bool(auto)
    try:
        indexgroup = self._v_file._getNode(_indexPathnameOf(self))
    except NoSuchNodeError:
        indexgroup = createIndexesTable(self)
    indexgroup.auto = auto
    # Update the cache in table instance as well
    self._autoIndex = auto


# **************** WARNING! ***********************
# This function can be called during the destruction time of a table
# so measures have been taken so that it doesn't have to revive
# another node (which can fool the LRU cache). The solution devised
# has been to add a cache for autoIndex (Table._autoIndex), populate
# it in creation time of the cache (which is a safe period) and then
# update the cache whenever it changes.
# This solves the error when running test_indexes.py ManyNodesTestCase.
# F. Alted 2007-04-20
# **************************************************
def _table__getautoIndex(self):
    if self._autoIndex is None:
        try:
            indexgroup = self._v_file._getNode(_indexPathnameOf(self))
        except NoSuchNodeError:
            self._autoIndex = defaultAutoIndex  # update cache
            return self._autoIndex
        else:
            self._autoIndex = indexgroup.auto   # update cache
            return self._autoIndex
    else:
        # The value is in cache, return it
        return self._autoIndex

_table__autoIndex = property(
    _table__getautoIndex , _table__setautoIndex, None,
    """
    Automatically keep column indexes up to date?

    Setting this value states whether existing indexes should be
    automatically updated after an append operation or recomputed
    after an index-invalidating operation (i.e. removal and
    modification of rows).  The default is true.

    This value gets into effect whenever a column is altered.  If you
    don't have automatic indexing activated and you want to do an an
    immediate update use `Table.flushRowsToIndex()`; for an immediate
    reindexing of invalidated indexes, use `Table.reIndexDirty()`.

    This value is persistent.

    .. Note:: Column indexing is only available in PyTables Pro.
    """ )


def restorecache(self):
    # Define a cache for sparse table reads
    params = self._v_file.params
    chunksize = self._v_chunkshape[0]
    nslots = params['TABLE_MAX_SIZE'] / (chunksize * self._v_dtype.itemsize)
    self._chunkcache = NumCache((nslots, chunksize), self._v_dtype,
                                'table chunk cache')
    self._seqcache = ObjectCache(params['ITERSEQ_MAX_SLOTS'],
                                 params['ITERSEQ_MAX_SIZE'],
                                 'Iter sequence cache')
    self._dirtycache = False


def _table__whereIndexed(self, compiled, condition, condvars,
                         start, stop, step):
    if profile: tref = time()
    if profile: show_stats("Entering table_whereIndexed", tref)
    self._useIndex = True
    # Clean the table caches for indexed queries if needed
    if self._dirtycache:
        restorecache(self)

    # Get the values in expression that are not columns
    values = []
    for key, value in condvars.iteritems():
        if isinstance(value, numpy.ndarray):
            values.append((key, value.item()))
    # Build a key for the sequence cache
    seqkey = (condition, tuple(values), (start, stop, step))
    # Do a lookup in sequential cache for this query
    nslot = self._seqcache.getslot(seqkey)
    if nslot >= 0:
        # Get the row sequence from the cache
        seq = self._seqcache.getitem(nslot)
        if len(seq) == 0:
            return iter([])
        seq = numpy.array(seq, dtype='int64')
        # Correct the ranges in cached sequence
        if (start, stop, step) != (0, self.nrows, 1):
            seq = seq[(seq>=start)&(seq<stop)&((seq-start)%step==0)]
        return self.itersequence(seq)
    else:
        # No luck.  Set row sequence to empty.  It will be populated
        # in the iterator. If not possible, the slot entry will be
        # removed there.
        self._nslotseq = self._seqcache.setitem(seqkey, [], 1)

    # Compute the chunkmap for every index in indexed expression
    idxexprs = compiled.index_expressions
    strexpr = compiled.string_expression
    cmvars = {}
    tcoords = 0
    for i, idxexpr in enumerate(idxexprs):
        var, ops, lims = idxexpr
        col = condvars[var]
        index = col.index
        assert index is not None, "the chosen column is not indexed"
        assert not index.dirty, "the chosen column has a dirty index"

        # Get the number of rows that the indexed condition yields.
        range_ = index.getLookupRange(ops, lims)
        ncoords = index.search(range_)
        tcoords += ncoords
        if index.reduction == 1 and ncoords == 0:
            # No values from index condition, thus the chunkmap should be empty
            nrowsinchunk = self.chunkshape[0]
            nchunks = long(math.ceil(float(self.nrows)/nrowsinchunk))
            chunkmap = numpy.zeros(shape=nchunks, dtype="bool")
        else:
            # Get the chunkmap from the index
            chunkmap = index.get_chunkmap()
        # Assign the chunkmap to the cmvars dictionary
        cmvars["e%d"%i] = chunkmap

    if index.reduction == 1 and tcoords == 0:
        # No candidates found in any indexed expression component, so leave now
        return iter([])

    # Compute the final chunkmap
    chunkmap = numexpr.evaluate(strexpr, cmvars)
    # Method .any() is twice as faster than method .sum()
    if not chunkmap.any():
        # The chunkmap is empty
        return iter([])

    if profile: show_stats("Exiting table_whereIndexed", tref)
    return chunkmap


def createIndexesTable(table):
    itgroup = IndexesTableG(
        table._v_parent, _indexNameOf(table),
        "Indexes container for table "+table._v_pathname, new=True)
    return itgroup


def createIndexesDescr(igroup, dname, iname, filters):
    idgroup = IndexesDescG(
        igroup, iname,
        "Indexes container for sub-description "+dname,
        filters=filters, new=True)
    return idgroup


def _column__createIndex(self, optlevel, kind, filters, tmp_dir,
                         blocksizes, verbose):
    name = self.name
    table = self.table
    tableName = table._v_name
    dtype = self.dtype
    descr = self.descr
    index = self.index
    getNode = table._v_file._getNode

    # Warn if the index already exists
    if index:
        raise ValueError, \
"%s for column '%s' already exists. If you want to re-create it, please, try with reIndex() method better" % (str(index), str(self.pathname))

    # Check that the datatype is indexable.
    if dtype.str[1:] == 'u8':
        raise NotImplementedError(
            "indexing 64-bit unsigned integer columns "
            "is not supported yet, sorry" )
    if dtype.kind == 'c':
        raise TypeError("complex columns can not be indexed")
    if dtype.shape != ():
        raise TypeError("multidimensional columns can not be indexed")

    # Get the indexes group for table, and if not exists, create it
    try:
        itgroup = getNode(_indexPathnameOf(table))
    except NoSuchNodeError:
        itgroup = createIndexesTable(table)

    # Create the necessary intermediate groups for descriptors
    idgroup = itgroup
    dname = ""
    pathname = descr._v_pathname
    if pathname != '':
        inames = pathname.split('/')
        for iname in inames:
            if dname == '':
                dname = iname
            else:
                dname += '/'+iname
            try:
                idgroup = getNode('%s/%s' % (itgroup._v_pathname, dname))
            except NoSuchNodeError:
                idgroup = createIndexesDescr(idgroup, dname, iname, filters)

    # Create the atom
    assert dtype.shape == ()
    atom = Atom.from_dtype(numpy.dtype((dtype, (0,))))

    # Protection on tables larger than the expected rows (perhaps the
    # user forgot to pass this parameter to the Table constructor?)
    expectedrows = table._v_expectedrows
    if table.nrows > expectedrows:
        expectedrows = table.nrows

    # Create the index itself
    index = Index(
        idgroup, name, atom=atom,
        title="Index for %s column" % name,
        kind=kind,
        optlevel=optlevel,
        filters=filters,
        tmp_dir=tmp_dir,
        expectedrows=expectedrows,
        byteorder=table.byteorder,
        blocksizes=blocksizes)

    table._setColumnIndexing(self.pathname, True)

    # Feed the index with values
    slicesize = index.slicesize
    # Add rows to the index if necessary
    if table.nrows > 0:
        indexedrows = table._addRowsToIndex(
            self.pathname, 0, table.nrows, lastrow=True, update=False )
    else:
        indexedrows = 0
    index.dirty = False
    table._indexedrows = indexedrows
    table._unsaved_indexedrows = table.nrows - indexedrows

    # Optimize the index that has been already filled-up
    index.optimize(verbose=verbose)

    # We cannot do a flush here because when reindexing during a
    # flush, the indexes are created anew, and that creates a nested
    # call to flush().
    ##table.flush()

    return indexedrows
