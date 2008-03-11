#######################################################################
#
#       License: XXX
#       Created: February 13, 2007
#       Author:  Ivan Vilata - ivilata@carabos.com
#
#       $Id$
#
########################################################################

"""Here is defined the Table class (pro)."""

import warnings

import numpy

from tables.atom import Atom
from tables.exceptions import NoSuchNodeError
from tables.index import defaultAutoIndex, defaultIndexFilters, Index
from tables.leaf import Filters

from tables._table_common import _indexPathnameOf


__version__ = "$Revision$"


class NailedDict(object):

    """A dictionary which ignores its items when it has nails on it."""

    def __init__(self):
        self._cache = {}
        self._nailcount = 0

    # Only a restricted set of dictionary methods are supported.  That
    # is why we buy instead of inherit.

    # The following are intended to be used by ``Table`` code changing
    # the set of usable indexes.

    def clear(self):
        self._cache.clear()
    def nail(self):
        self._nailcount -= 1
    def unnail(self):
        self._nailcount += 1

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
        self._cache[key] = value


def _table__setautoIndex(self, auto):
    auto = bool(auto)
    try:
        indexgroup = self._v_file._getNode(_indexPathnameOf(self))
    except NoSuchNodeError:
        indexgroup = self._createIndexesTable()
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
# F. Altet 2007-04-20
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

def _table__setindexFilters(self, filters):
    warnings.warn(
        "``indexFilters`` property will soon be deprecated.  "
        "Please, do specify the filters in the ``filters`` "
        "argument of ``createIndex()`` method.",
        DeprecationWarning )
    if not isinstance(filters, Filters):
        raise TypeError("not an instance of ``Filters``: %r" % filters)
    try:
        indexgroup = self._v_file._getNode(_indexPathnameOf(self))
    except NoSuchNodeError:
        indexgroup = self._createIndexesTable()
    indexgroup.filters = filters

def _table__getindexFilters(self):
    try:
        indexgroup = self._v_file._getNode(_indexPathnameOf(self))
    except NoSuchNodeError:
        return defaultIndexFilters
    else:
        return indexgroup.filters

_table__indexFilters = property(
    _table__getindexFilters, _table__setindexFilters, None,
    """
    Filters used to compress indexes.

    Setting this value to a `Filters` instance determines the
    compression to be used for indexes.  Setting it to ``None``
    means that no filters will be used for indexes.  The default is
    zlib compression level 1 with shuffling.

    This value is used when creating new indexes or recomputing old
    ones.  To apply it to existing indexes, use `Table.reIndex()`.

    This value is persistent.

    .. Note:: Column indexing is only available in PyTables Pro.
    """ )

def _column__createIndex(self, optlevel, filters, tmp_dir,
                         blocksizes, indsize,
                         verbose):
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
        itgroup = table._createIndexesTable()

    # If no filters are specified, try the indexFilters property
    if filters is None:
        filters = table.indexFilters

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
                idgroup = table._createIndexesDescr(
                    idgroup, dname, iname, filters)

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
        idgroup, name, atom=atom, column=self,
        title="Index for %s column" % name,
        optlevel=optlevel,
        filters=filters,
        tmp_dir=tmp_dir,
        expectedrows=expectedrows,
        byteorder=table.byteorder,
        blocksizes=blocksizes,
        indsize=indsize)
    self._updateIndexLocation(index)

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

    # Finally, optimize the index that has been already filled-up
    index.optimize()

    return indexedrows
