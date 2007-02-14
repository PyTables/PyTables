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

from tables.parameters import (
    TABLE_MAX_SIZE, LIMDATA_MAX_SLOTS, LIMDATA_MAX_SIZE )
from tables.conditions import call_on_recarr
from tables.flavor import internal_to_flavor
from tables.lrucacheExtension import ObjectCache, NumCache
from tables.utilsExtension import getNestedField


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


def _table__restorecache(self):
    # Define a cache for sparse table reads
    maxslots = TABLE_MAX_SIZE / self.rowsize
    self._sparsecache = NumCache( shape=(maxslots, 1), itemsize=self.rowsize,
                                  name="sparse rows" )
    self._limdatacache = ObjectCache( LIMDATA_MAX_SLOTS, LIMDATA_MAX_SIZE,
                                      "data limits" )
    """A cache for data based on search limits and table colum."""

def _table__readWhere(self, splitted, condvars, field):
    idxvar = splitted.index_variable
    column = condvars[idxvar]
    index = column.index
    assert index is not None, "the chosen column is not indexed"
    assert not index.dirty, "the chosen column has a dirty index"

    # Clean the cache if needed
    if self._dirtycache:
        self._restorecache()

    # Get the coordinates to lookup
    range_ = index.getLookupRange(
        splitted.index_operators, splitted.index_limits, self )

    # Check whether the array is in the limdata cache or not.
    key = (column.name, range_)
    limdatacache = self._limdatacache
    nslot = limdatacache.getslot(key)
    if nslot >= 0:
        # Cache hit. Use the array kept there.
        recarr = limdatacache.getitem(nslot)
        nrecords = len(recarr)
    else:
        # No luck with cached data. Proceed with the regular search.
        nrecords = index.search(range_)
        # Create a buffer and read the values in.
        recarr = self._get_container(nrecords)
        if nrecords > 0:
            coords = index.indices._getCoords(index, 0, nrecords)
            recout = self._read_elements(recarr, coords)
        # Put this recarray in limdata cache.
        size = len(recarr) * self.rowsize + 1  # approx. size of array
        limdatacache.setitem(key, recarr, size)

    # Filter out rows not fulfilling the residual condition.
    rescond = splitted.residual_function
    if rescond and nrecords > 0:
        indexValid = call_on_recarr(
            rescond, splitted.residual_parameters,
            recarr, param2arg=condvars.__getitem__ )
        recarr = recarr[indexValid]

    if field:
        recarr = getNestedField(recarr, field)
    return internal_to_flavor(recarr, self.flavor)

def _table__getWhereList(self, splitted, condvars):
    idxvar = splitted.index_variable
    index = condvars[idxvar].index
    assert index is not None, "the chosen column is not indexed"
    assert not index.dirty, "the chosen column has a dirty index"

    # get the number of coords and set-up internal variables
    range_ = index.getLookupRange(
        splitted.index_operators, splitted.index_limits, self )
    ncoords = index.search(range_)
    if ncoords > 0:
        coords = index.indices._getCoords_sparse(index, ncoords)
        # Get a copy of the internal buffer to handle it to the user
        coords = coords.copy()
    else:
        #coords = numpy.empty(type=numpy.int64, shape=0)
        coords = self._getemptyarray("int64")

    # Filter out rows not fulfilling the residual condition.
    rescond = splitted.residual_function
    if rescond and ncoords > 0:
        indexValid = call_on_recarr(
            rescond, splitted.residual_parameters,
            recarr=self._readCoordinates(coords),
            param2arg=condvars.__getitem__ )
        coords = coords[indexValid]

    return coords
