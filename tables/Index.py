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

    IndexProps
    Index

Functions:


Misc variables:

    __version__


"""

import warnings
import math
import cPickle

import numarray

import tables.hdf5Extension as hdf5Extension
import tables.utilsExtension as utilsExtension
from tables.AttributeSet import AttributeSet
from tables.Atom import Atom
from tables.Leaf import Filters
from tables.IndexArray import IndexArray
from tables.Group import Group
from tables.utils import joinPath

__version__ = "$Revision: 1236 $"

# default version for INDEX objects
#obversion = "1.0"    # initial version
obversion = "2.0"    # indexes moved to a hidden directory

# Python implementations of NextAfter and NextAfterF
#
# These implementations exist because the standard function
# nextafterf is not available on Microsoft platforms.
#
# These implementations are based on the IEEE representation of
# floats and doubles.
# Author:  Shack Toms - shack@livedata.com
#
# Thanks to Shack Toms shack@livedata.com for NextAfter and NextAfterF
# implementations in Python. 2004-10-01

epsilon  = math.ldexp(1.0, -53) # smallest double such that 0.5+epsilon != 0.5
epsilonF = math.ldexp(1.0, -24) # smallest float such that 0.5+epsilonF != 0.5

maxFloat = float(2**1024 - 2**971)  # From the IEEE 754 standard
maxFloatF = float(2**128 - 2**104)  # From the IEEE 754 standard

minFloat  = math.ldexp(1.0, -1022) # min positive normalized double
minFloatF = math.ldexp(1.0, -126)  # min positive normalized float

smallEpsilon  = math.ldexp(1.0, -1074) # smallest increment for doubles < minFloat
smallEpsilonF = math.ldexp(1.0, -149)  # smallest increment for floats < minFloatF

infinity = math.ldexp(1.0, 1023) * 2
infinityF = math.ldexp(1.0, 128)
#Finf = float("inf")  # Infinite in the IEEE 754 standard (not avail in Win)

# A portable representation of NaN
# if sys.byteorder == "little":
#     testNaN = struct.unpack("d", '\x01\x00\x00\x00\x00\x00\xf0\x7f')[0]
# elif sys.byteorder == "big":
#     testNaN = struct.unpack("d", '\x7f\xf0\x00\x00\x00\x00\x00\x01')[0]
# else:
#     raise ValueError, "Byteorder '%s' not supported!" % sys.byteorder
# This one seems better
testNaN = infinity - infinity

# Utility functions
def infType(type, itemsize, sign=0):
    """Return a superior limit for maximum representable data type"""
    if str(type) != "CharType":
        if sign:
            return -infinity
        else:
            return infinity
    else:
        if sign:
            return "\x00"*itemsize
        else:
            return "\xff"*itemsize


# This check does not work for Python 2.2.x or 2.3.x (!)
def IsNaN(x):
    """a simple check for x is NaN, assumes x is float"""
    return x != x

def PyNextAfter(x, y):
    """returns the next float after x in the direction of y if possible, else returns x"""
    # if x or y is Nan, we don't do much
    if IsNaN(x) or IsNaN(y):
        return x

    # we can't progress if x == y
    if x == y:
        return x

    # similarly if x is infinity
    if x >= infinity or x <= -infinity:
        return x

    # return small numbers for x very close to 0.0
    if -minFloat < x < minFloat:
        if y > x:
            return x + smallEpsilon
        else:
            return x - smallEpsilon  # we know x != y

    # it looks like we have a normalized number
    # break x down into a mantissa and exponent
    m, e = math.frexp(x)

    # all the special cases have been handled
    if y > x:
        m += epsilon
    else:
        m -= epsilon

    return math.ldexp(m, e)

def PyNextAfterF(x, y):
    """returns the next IEEE single after x in the direction of y if possible, else returns x"""

    # if x or y is Nan, we don't do much
    if IsNaN(x) or IsNaN(y):
        return x

    # we can't progress if x == y
    if x == y:
        return x

    # similarly if x is infinity
    if x >= infinityF:
        return infinityF
    elif x <= -infinityF:
        return -infinityF

    # return small numbers for x very close to 0.0
    if -minFloatF < x < minFloatF:
        # since Python uses double internally, we
        # may have some extra precision to toss
        if x > 0.0:
            extra = x % smallEpsilonF
        elif x < 0.0:
            extra = x % -smallEpsilonF
        else:
            extra = 0.0
        if y > x:
            return x - extra + smallEpsilonF
        else:
            return x - extra - smallEpsilonF  # we know x != y

    # it looks like we have a normalized number
    # break x down into a mantissa and exponent
    m, e = math.frexp(x)

    # since Python uses double internally, we
    # may have some extra precision to toss
    if m > 0.0:
        extra = m % epsilonF
    else:  # we have already handled m == 0.0 case
        extra = m % -epsilonF

    # all the special cases have been handled
    if y > x:
        m += epsilonF - extra
    else:
        m -= epsilonF - extra

    return math.ldexp(m, e)


def CharTypeNextAfter(x, direction, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."
    # Pad the string with \x00 chars until itemsize completion
    padsize = itemsize - len(x)
    if padsize > 0:
        x += "\x00"*padsize
    xlist = list(x); xlist.reverse()
    i = 0
    if direction > 0:
        if xlist == "\xff"*itemsize:
            # Maximum value, return this
            return "".join(xlist)
        for xchar in xlist:
            if ord(xchar) < 0xff:
                xlist[i] = chr(ord(xchar)+1)
                break
            else:
                xlist[i] = "\x00"
            i += 1
    else:
        if xlist == "\x00"*itemsize:
            # Minimum value, return this
            return "".join(xlist)
        for xchar in xlist:
            if ord(xchar) > 0x00:
                xlist[i] = chr(ord(xchar)-1)
                break
            else:
                xlist[i] = "\xff"
            i += 1
    xlist.reverse()
    return "".join(xlist)


def nextafter(x, direction, type, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."

    if direction == 0:
        return x

    if str(type) == "CharType":
        return CharTypeNextAfter(x, direction, itemsize)
    elif isinstance(numarray.typeDict[type], numarray.IntegralType):
        if direction < 0:
            return x-1
        else:
            return x+1
    elif str(type) == "Float32":
        if direction < 0:
            return PyNextAfterF(x,x-1)
        else:
            return PyNextAfterF(x,x+1)
    elif str(type) == "Float64":
        if direction < 0:
            return PyNextAfter(x,x-1)
        else:
            return PyNextAfter(x,x+1)
    else:
        raise TypeError, "Type %s is not supported" % type


class IndexProps(object):
    """Container for index properties

    Instance variables:

        auto -- whether an existing index should be updated or not after a
            Table append operation
        reindex -- whether the table fields are to be re-indexed
            after an invalidating index operation (like Table.removeRows)
        filters -- the filter properties for the Table indexes

    """

    def __init__(self, auto=1, reindex=1, filters=None):
        """Create a new IndexProps instance

        Parameters:

        auto -- whether an existing index should be reindexed after a
            Table append operation. Defaults is reindexing.
        reindex -- whether the table fields are to be re-indexed
            after an invalidating index operation (like Table.removeRows).
            Default is reindexing.
        filters -- the filter properties. Default are ZLIB(1) and shuffle


            """
        if auto is None:
            auto = 1  # Default
        if reindex is None:
            reindex = 1  # Default
        assert auto in [0, 1], "'auto' can only take values 0 or 1"
        assert reindex in [0, 1], "'reindex' can only take values 0 or 1"
        self.auto = auto
        self.reindex = reindex
        if filters is None:
            self.filters = Filters(complevel=1, complib="zlib",
                                   shuffle=1, fletcher32=0)
        elif isinstance(filters, Filters):
            self.filters = filters
        else:
            raise TypeError, \
"If you pass a filters parameter, it should be a Filters instance."

    def __repr__(self):
        """The string reprsentation choosed for this object
        """
        descr = self.__class__.__name__
        descr += "(auto=%s" % (self.auto)
        descr += ", reindex=%s" % (self.reindex)
        descr += ", filters=%s" % (self.filters)
        return descr+")"

    def __str__(self):
        """The string reprsentation choosed for this object
        """

        return repr(self)

class Index(hdf5Extension.Index, Group):

    """Represent the index (sorted and reverse index) dataset in HDF5 file.

    It enables to create indexes of Columns of Table objects.

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

        search(start, stop, step, where)
        getCoords(startCoords, maxCoords)
        append(object)

    Instance variables:

        column -- The column object this index belongs to
        dirty -- Whether the index is dirty or not.
        nrows -- The number of slices in the index.
        nelemslice -- The number of elements per slice.
        nelements -- The number of indexed rows.
        shape -- The shape of this index (in slices and elements).
        filters -- The properties used to filter the stored items.
        sorted -- The IndexArray object with the sorted values information.
        indices -- The IndexArray object with the sorted indices information.

    """

    _c_classId = 'CINDEX'


    # <properties>

    dirty = property(
        lambda self: self.column.dirty, None, None,
        "Whether the index is dirty or not.")

    nrows = property(
        lambda self: self.sorted.nrows, None, None,
        "The number of slices in the index.")

    nelemslice = property(
        lambda self: self.sorted.nelemslice, None, None,
        "The number of elements per slice.")

    nelements = property(
        lambda self: self.sorted.nrows * self.sorted.nelemslice, None, None,
        "The number of indexed rows.")

    shape = property(
        lambda self: (self.sorted.nrows, self.sorted.nelemslice), None, None,
        "The shape of this index (in slices and elements).")

    filters = property(
        lambda self: self.sorted.filters, None, None,
        "The properties used to filter the stored items.")

    # </properties>


    def __init__(self, parentNode, name,
                 atom=None, column=None,
                 title="", filters=None,
                 expectedrows=0,
                 testmode=False, new=True):
        """Create an Index instance.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. Only scalar atoms are
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

        """

        self._v_version = None
        """The object version of this index."""

        self._v_expectedrows = expectedrows
        """The expected number of items of index arrays."""
        self.testmode = testmode
        """Enables test mode for index chunk size calculation."""
        self.atom = atom
        """The `Atom` instance matching to be stored by the index array."""
        self.column = column
        """The `Column` instance for the indexed column."""

        self.starts = None
        """Where the values fulfiling conditions starts for every slice."""
        self.lengths = None
        """Lengths of the values fulfilling conditions for every slice."""

        # Index creation is never logged.
        super(Index, self).__init__(
            parentNode, name, title, new, filters, log=False)

        # Set the version number of this object as an index, not a group.
        self._v_version = obversion


    def _g_postInitHook(self):
        super(Index, self)._g_postInitHook()

        # Index arrays must only be created for new indexes.
        if not self._v_new:
            return

        # Set the filters for this object (they are *not* inherited)
        filters = self._v_new_filters
        if filters is None:
            # If not filters has been passed in the constructor,
            # set a sensible default, using zlib compression and shuffling
            filters = Filters(complevel = 1, complib = "zlib",
                              shuffle = 1, fletcher32 = 0)

        # Create the IndexArray for sorted values
        IndexArray(self, 'sorted',
                   self.atom, "Sorted Values", filters,
                   self.testmode, self._v_expectedrows)

        # Create the IndexArray for index values
        IndexArray(self, 'indices',
                   Atom("Int32", shape=(0, 1)), "Reverse Indices", filters,
                   self.testmode, self._v_expectedrows)

        # Create the EArray for range values  (1st order cache)
        if str(self.atom.type) == "CharType":
            atom = StringAtom(shape=(0, 2), length=self.atom.itemsize,
                              flavor="CharArray")
        else:
            atom = Atom(self.atom.type, shape=(0,2), flavor="NumArray")
        CacheArray(self, 'ranges', atom, "Range Values",
                   Filters(complevel=0, shuffle=0),   # too small to use filters
                   self._v_expectedrows//self.nelemslice)

        # Create the EArray for boundary values (2nd order cache)
        nbounds = (self.nelemslice - 1 ) // self.chunksize
        if str(self.atom.type) == "CharType":
            atom = StringAtom(shape=(0, nbounds),
                              length=self.atom.itemsize,
                              flavor="CharArray")
        else:
            atom = Atom(self.atom.type, shape=(0, nbounds))
        CacheArray(self, 'bounds', atom, "Boundaries",
                   Filters(complevel=0, shuffle=0),   # too small to use filters
                   self._v_expectedrows//self.nelemslice)

        # Create the Array for last row values
        shape = 2 + nbounds + self.nelemslice
        if str(self.type) == "CharType":
            atom = strings.array(None, shape=shape, itemsize=self.itemsize)
        else:
            atom = numarray.array(None, shape=shape, type=self.type)
        LastRowArray(self, 'lrvb', atom, "Last Row Values + bounds")

        # Create the Array for reverse indexes in last row
        shape = self.nelemslice     # enough for indexes and length
        atom = numarray.zeros(shape=shape, type=numarray.Int32)
        LastRowArray(self, 'lrri', atom, "Last Row Reverse Indexes")

    def _g_updateDependent(self):
        super(Index, self)._g_updateDependent()
        self.column._updateIndexLocation(self)


    def append(self, arr):
        """Append the object to this (enlargeable) object"""

        # Save the sorted array
        if str(self.sorted.type) == "CharType":
            s=arr.argsort()
            # Caveat: this conversion is necessary for portability on
            # 64-bit systems because indexes are 64-bit long on these
            # platforms
            self.indices.append(numarray.array(s, type="Int32"))
            self.sorted.append(arr[s])
        else:
            #self.sorted.append(numarray.sort(arr))
            #self.indices.append(numarray.argsort(arr))
            # The next is a 10% faster, but the ideal solution would
            # be to find a funtion in numarray that returns both
            # sorted and argsorted all in one call
            s=numarray.argsort(arr)
            # Caveat: this conversion is necessary for portability on
            # 64-bit systems because indexes are 64-bit long on these
            # platforms
            self.indices.append(numarray.array(s, type="Int32"))
            self.sorted.append(arr[s])

    def search(self, item):
        """Do a binary search in this index for an item"""
        #t1=time.time()
        ntotaliter = 0; tlen = 0
        self.starts = []; self.lengths = []
        #self.irow = 0; self.len1 = 0; self.len2 = 0;  # useful for getCoords()
        self.sorted._initSortedSlice(self.sorted.chunksize)
        # Do the lookup for values fullfilling the conditions
        for i in xrange(self.sorted.nrows):
            (start, stop, niter) = self.sorted._searchBin(i, item)
            self.starts.append(start)
            self.lengths.append(stop - start)
            ntotaliter += niter
            tlen += stop - start
        self.sorted._destroySortedSlice()
        #print "time reading indices:", time.time()-t1
        #print "ntotaliter:", ntotaliter
        assert tlen >= 0, "Index.search(): Post-condition failed. Please, report this to the authors."
        return tlen

# This has been ported to Pyrex. However, with pyrex it has the same speed,
# so, it's better to stay here
    def getCoords(self, startCoords, maxCoords):
        """Get the coordinates of indices satisfiying the cuts.

        You must call the Index.search() method before in order to get
        good sense results.

        """
        #t1=time.time()
        len1 = 0; len2 = 0; relCoords = 0
        # Correction against asking too many elements
        nindexedrows = self.nelements
        if startCoords + maxCoords > nindexedrows:
            maxCoords = nindexedrows - startCoords
        for irow in xrange(self.sorted.nrows):
            leni = self.lengths[irow]; len2 += leni
            if (leni > 0 and len1 <= startCoords < len2):
                startl = self.starts[irow] + (startCoords-len1)
                # Read maxCoords as maximum
                stopl = startl + maxCoords
                # Correction if stopl exceeds the limits
                if stopl > self.starts[irow] + self.lengths[irow]:
                    stopl = self.starts[irow] + self.lengths[irow]
                self.indices._g_readIndex(irow, startl, stopl, relCoords)
                incr = stopl - startl
                relCoords += incr; startCoords += incr; maxCoords -= incr
                if maxCoords == 0:
                    break
            len1 += leni

        # I don't know if sorting the coordinates is better or not actually
        # Some careful tests must be carried out in order to do that
        #selections = self.indices.arrAbs[:relCoords]
        selections = numarray.sort(self.indices.arrAbs[:relCoords])
        #print "time getting coords:", time.time()-t1
        return selections

# This tries to be a version of getCoords that keeps track of visited rows
# in order to not re-visit them again. However, I didn't managed to make it
# work well. However, the improvement in speed should be not important
# in the majority of cases.
# Beware, the logic behind doing this is not trivial at all. You have been
# warned!. 2004-08-03
#     def getCoords_notwork(self, startCoords, maxCoords):
#         """Get the coordinates of indices satisfiying the cuts"""
#         relCoords = 0
#         # Correction against asking too many elements
#         nindexedrows = self.nelements
#         if startCoords + maxCoords > nindexedrows:
#             maxCoords = nindexedrows - startCoords
#         #for irow in xrange(self.irow, self.sorted.nrows):
#         while self.irow < self.sorted.nrows:
#             irow = self.irow
#             leni = self.lengths[irow]; self.len2 += leni
#             if (leni > 0 and self.len1 <= startCoords < self.len2):
#                 startl = self.starts[irow] + (startCoords-self.len1)
#                 # Read maxCoords as maximum
#                 stopl = startl + maxCoords
#                 # Correction if stopl exceeds the limits
#                 rowStop = self.starts[irow] + self.lengths[irow]
#                 if stopl >= rowStop:
#                     stopl = rowStop
#                     #self.irow += 1
#                 self.indices._g_readIndex(irow, startl, stopl, relCoords)
#                 incr = stopl - startl
#                 relCoords += incr
#                 maxCoords -= incr
#                 startCoords += incr
#                 self.len1 += incr
#                 if maxCoords == 0:
#                     break
#             #self.len1 += leni
#             self.irow += 1

#         # I don't know if sorting the coordinates is better or not actually
#         # Some careful tests must be carried out in order to do that
#         selections = numarray.sort(self.indices.arrAbs[:relCoords])
#         #selections = self.indices.arrAbs[:relCoords]
#         return selections

    def getLookupRange(self, column):
        #import time
        table = column.table
        # Get the coordinates for those values
        ilimit = table.opsValues
        ctype = column.type
        sctype = str(ctype)
        itemsize = table.colitemsizes[column.pathname]

        # Check that limits are compatible with type
        for limit in ilimit:
            # Check for strings
            if sctype == "CharType":
                if type(limit) is not str:
                    raise TypeError("""\
Bounds (or range limits) for string columns can only be strings.""")
                else:
                    continue

            nactype = numarray.typeDict[sctype]

            # Check for booleans
            if isinstance(nactype, numarray.BooleanType):
                if type(limit) not in (int, long, bool):
                    raise TypeError("""\
Bounds (or range limits) for bool columns can only be ints or booleans.""")
            # Check for ints
            elif isinstance(nactype, numarray.IntegralType):
                if type(limit) not in (int, long, float):
                    raise TypeError("""\
Bounds (or range limits) for integer columns can only be ints or floats.""")
            # Check for floats
            elif isinstance(nactype, numarray.FloatingType):
                if type(limit) not in (int, long, float):
                    raise TypeError("""\
Bounds (or range limits) for float columns can only be ints or floats.""")
            else:
                raise TypeError("""
Bounds (or range limits) can only be strings, bools, ints or floats.""")

        # Boolean types are a special case for searching
        if sctype == "Bool":
            if len(table.ops) == 1 and table.ops[0] == 5: # __eq__
                item = (ilimit[0], ilimit[0])
                ncoords = self.search(item)
                return ncoords
            else:
                raise NotImplementedError, \
                      "Only equality operator is supported for boolean columns."
        # Other types are treated here
        if len(ilimit) == 1:
            ilimit = ilimit[0]
            op = table.ops[0]
            if op == 1: # __lt__
                item = (infType(type=ctype, itemsize=itemsize, sign=-1),
                        nextafter(ilimit, -1, ctype, itemsize))
            elif op == 2: # __le__
                item = (infType(type=ctype, itemsize=itemsize, sign=-1),
                        ilimit)
            elif op == 3: # __gt__
                item = (nextafter(ilimit, +1, ctype, itemsize),
                        infType(type=ctype, itemsize=itemsize, sign=0))
            elif op == 4: # __ge__
                item = (ilimit,
                        infType(type=ctype, itemsize=itemsize, sign=0))
            elif op == 5: # __eq__
                item = (ilimit, ilimit)
            elif op == 6: # __ne__
                # I need to cope with this
                raise NotImplementedError, "'!=' or '<>' not supported yet"
        elif len(ilimit) == 2:
            item1, item2 = ilimit
            if item1 > item2:
                raise ValueError("""\
On 'val1 <{=} col <{=} val2' selections, \
val1 must be less or equal than val2""")
            op1, op2 = table.ops
            if op1 == 3 and op2 == 1:  # item1 < col < item2
                item = (nextafter(item1, +1, ctype, itemsize),
                        nextafter(item2, -1, ctype, itemsize))
            elif op1 == 4 and op2 == 1:  # item1 <= col < item2
                item = (item1, nextafter(item2, -1, ctype, itemsize))
            elif op1 == 3 and op2 == 2:  # item1 < col <= item2
                item = (nextafter(item1, +1, ctype, itemsize), item2)
            elif op1 == 4 and op2 == 2:  # item1 <= col <= item2
                item = (item1, item2)
            else:
                raise ValueError, \
"Combination of operators not supported. Use val1 <{=} col <{=} val2"

        #t1=time.time()
        ncoords = self.search(item)
        #print "time reading indices:", time.time()-t1
        return ncoords

    def _f_remove(self, recursive=False):
        """Remove this Index object"""

        if utilsExtension.whichLibVersion("hdf5")[1] == "1.6.3":
            warnings.warn("""\
You are using HDF5 version 1.6.3. It turns out that this precise
version has a bug that causes a seg fault when deleting a chunked
dataset. If you are getting such a seg fault immediately after this
message, please, get a patched version of HDF5 1.6.3, or, better,
get HDF5 1.6.4.""")

        # Index removal is always recursive,
        # no matter what `recursive` says.
        super(Index, self)._f_remove(True)


    def __str__(self):
        """This provides a more compact representation than __repr__"""
        return "Index(%s, shape=%s, chunksize=%s)" % \
               (self.nelements, self.shape, self.sorted.chunksize)

    def __repr__(self):
        """This provides more metainfo than standard __repr__"""

        cpathname = self.column.table._v_pathname + ".cols." + self.column.name
        return """%s (Index for column %s)
  type := %r
  nelements := %s
  shape := %s
  chunksize := %s
  byteorder := %r
  filters := %s
  dirty := %s
  sorted := %s
  indices := %s""" % (self._v_pathname, cpathname,
                     self.sorted.type, self.nelements, self.shape,
                     self.sorted.chunksize, self.sorted.byteorder,
                     self.filters, self.dirty, self.sorted, self.indices)
