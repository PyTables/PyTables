#######################################################################
#
#       License: XXX
#       Created: February 13, 2007
#       Author:  Ivan Vilata - ivilata@carabos.com
#
#       $Id$
#
########################################################################

"""Here is defined the Table class (index)."""

from tables.parameters import (
    TABLE_MAX_SIZE, LIMDATA_MAX_SLOTS, LIMDATA_MAX_SIZE )
from tables.conditions import call_on_recarr
from tables.flavor import internal_to_flavor
from tables.lrucacheExtension import ObjectCache, NumCache
from tables.utilsExtension import getNestedField


__version__ = "$Revision$"


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
