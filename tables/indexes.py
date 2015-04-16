# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: June 02, 2004
# Author:  Francesc Alted - faltet@pytables.com
#
# $Source: /cvsroot/pytables/pytables/tables/indexes.py $
# $Id$
#
########################################################################

"""Here is defined the IndexArray class."""

from bisect import bisect_left, bisect_right

from tables.node import NotLoggedMixin
from tables.carray import CArray
from tables.earray import EArray
from tables import indexesextension

from tables._past import previous_api, previous_api_property

# Declarations for inheriting


class CacheArray(NotLoggedMixin, EArray, indexesextension.CacheArray):
    """Container for keeping index caches of 1st and 2nd level."""

    # Class identifier.
    _c_classid = 'CACHEARRAY'

    _c_classId = previous_api_property('_c_classid')


class LastRowArray(NotLoggedMixin, CArray, indexesextension.LastRowArray):
    """Container for keeping sorted and indices values of last row of an
    index."""

    # Class identifier.
    _c_classid = 'LASTROWARRAY'

    _c_classId = previous_api_property('_c_classid')


class IndexArray(NotLoggedMixin, EArray, indexesextension.IndexArray):
    """Represent the index (sorted or reverse index) dataset in HDF5 file.

    All NumPy typecodes are supported except for complex datatypes.

    Parameters
    ----------
    parentnode
        The Index class from which this object will hang off.

        .. versionchanged:: 3.0
           Renamed from *parentNode* to *parentnode*.

    name : str
        The name of this node in its parent group.
    atom
        An Atom object representing the shape and type of the atomic objects to
        be saved. Only scalar atoms are supported.
    title
        Sets a TITLE attribute on the array entity.
    filters : Filters
        An instance of the Filters class that provides information about the
        desired I/O filters to be applied during the life of this object.
    byteorder
        The byteroder of the data on-disk.

    """

    # Class identifier.
    _c_classid = 'INDEXARRAY'

    _c_classId = previous_api_property('_c_classid')

    # Properties
    # ~~~~~~~~~~
    chunksize = property(
        lambda self: self.chunkshape[1], None, None,
        """The chunksize for this object.""")

    slicesize = property(
        lambda self: self.shape[1], None, None,
        """The slicesize for this object.""")

    # Other methods
    # ~~~~~~~~~~~~~
    def __init__(self, parentnode, name,
                 atom=None, title="",
                 filters=None, byteorder=None):
        """Create an IndexArray instance."""

        self._v_pathname = parentnode._g_join(name)
        if atom is not None:
            # The shape and chunkshape needs to be fixed here
            if name == "sorted":
                reduction = parentnode.reduction
                shape = (0, parentnode.slicesize // reduction)
                chunkshape = (1, parentnode.chunksize // reduction)
            else:
                shape = (0, parentnode.slicesize)
                chunkshape = (1, parentnode.chunksize)
        else:
            # The shape and chunkshape will be read from disk later on
            shape = None
            chunkshape = None

        super(IndexArray, self).__init__(
            parentnode, name, atom, shape, title, filters,
            chunkshape=chunkshape, byteorder=byteorder)

    # This version of searchBin uses both ranges (1st level) and
    # bounds (2nd level) caches. It uses a cache for boundary rows,
    # but not for 'sorted' rows (this is only supported for the
    # 'optimized' types).
    def _search_bin(self, nrow, item):
        item1, item2 = item
        result1 = -1
        result2 = -1
        hi = self.shape[1]
        ranges = self._v_parent.rvcache
        boundscache = self.boundscache
        # First, look at the beginning of the slice
        begin = ranges[nrow, 0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
        if item2 < begin:
            result2 = 0
        if result1 >= 0 and result2 >= 0:
            return (result1, result2)
        # Then, look for items at the end of the sorted slice
        end = ranges[nrow, 1]
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
        chunksize = self.chunksize  # Number of elements/chunksize
        nchunk = -1
        # Try to get the bounds row from the LRU cache
        nslot = boundscache.getslot(nrow)
        if nslot >= 0:
            # Cache hit. Use the row kept there.
            bounds = boundscache.getitem(nslot)
        else:
            # No luck with cached data. Read the row and put it in the cache.
            bounds = self._v_parent.bounds[nrow]
            size = bounds.size * bounds.itemsize
            boundscache.setitem(nrow, bounds, size)
        if result1 < 0:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect_left(bounds, item1)
            chunk = self._read_sorted_slice(nrow, chunksize * nchunk,
                                            chunksize * (nchunk + 1))
            result1 = indexesextension._bisect_left(chunk, item1, chunksize)
            result1 += chunksize * nchunk
        # Lookup in the middle of slice for item2
        if result2 < 0:
            # Search the appropriate chunk in bounds cache
            nchunk2 = bisect_right(bounds, item2)
            if nchunk2 != nchunk:
                chunk = self._read_sorted_slice(nrow, chunksize * nchunk2,
                                                chunksize * (nchunk2 + 1))
            result2 = indexesextension._bisect_right(chunk, item2, chunksize)
            result2 += chunksize * nchunk2
        return (result1, result2)

    _searchBin = previous_api(_search_bin)

    def __str__(self):
        "A compact representation of this class"
        return "IndexArray(path=%s)" % self._v_pathname

    def __repr__(self):
        """A verbose representation of this class."""

        return """%s
  atom = %r
  shape = %s
  nrows = %s
  chunksize = %s
  slicesize = %s
  byteorder = %r""" % (self, self.atom, self.shape, self.nrows,
                       self.chunksize, self.slicesize, self.byteorder)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
