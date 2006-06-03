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
import bisect
from time import time, clock

import numarray
from numarray import strings

import tables.indexesExtension as indexesExtension
import tables.utilsExtension as utilsExtension
from tables.AttributeSet import AttributeSet
from tables.Atom import Atom, StringAtom
from tables.Leaf import Filters
from tables.indexes import CacheArray, LastRowArray, IndexArray
from tables.Group import Group
from tables.utils import joinPath

__version__ = "$Revision: 1236 $"

# version of index format
#idx_version = "std"    # indexes for PyTables Standard
idx_version = "pro"     # indexes for PyTables Pro

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

# "infinity" for integral types
infinityIntegral = {"Int8": [-2**7, 2**7-1],
                    "UInt8": [0, 2**8-1],
                    "Int16": [-2**15, 2**15-1],
                    "UInt16": [0, 2**16-1],
                    "Int32": [-2**31, 2**31-1],
                    "UInt32": [0, 2**32-1],
                    "Int64": [-2**63, 2**63-1],
                    "UInt64": [0, 2**64-1]
                    }

# "infinity" for real types
infinityFloating = {"Float32": [-infinityF, infinityF],
                    "Float64": [-infinity, infinity],
                    }

# Utility functions
def infType(type, itemsize, sign=0):
    """Return a superior limit for maximum representable data type"""

    if str(type) == "CharType":
        if sign:
            return "\x00"*itemsize
        else:
            return "\xff"*itemsize
    elif isinstance(numarray.typeDict[type], numarray.FloatingType):
        return infinityFloating[type][1+sign]
    elif isinstance(numarray.typeDict[type], numarray.IntegralType):
        return infinityIntegral[type][1+sign]
    else:
        raise TypeError, "Type %s is not supported" % type

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


def IntTypeNextAfter(x, direction, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."

    # x is guaranteed to be either an int or a float
    if direction < 0:
        if type(x) is int:
            return x-1
        else:
            return int(PyNextAfter(x,x-1))
    else:
        if type(x) is int:
            return x+1
        else:
            return int(PyNextAfter(x,x+1))+1


def nextafter(x, direction, dtype, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."

    if direction == 0:
        return x

    if str(dtype) == "CharType":
        return CharTypeNextAfter(x, direction, itemsize)
    else:
        if type(x) not in (int, long, float):
            raise ValueError, "You need to pass integers or floats in this context."
        if isinstance(numarray.typeDict[dtype], numarray.IntegralType):
            return IntTypeNextAfter(x, direction, itemsize)
        elif str(dtype) == "Float32":
            if direction < 0:
                return PyNextAfterF(x,x-1)
            else:
                return PyNextAfterF(x,x+1)
        elif str(dtype) == "Float64":
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

class Index(indexesExtension.Index, Group):

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

    nelemslice = property(
        lambda self: self.sorted.nelemslice, None, None,
        "The number of elements per slice.")

    chunksize = property(
        lambda self: self.sorted.chunksize, None, None,
        "The number of elements per chunk on-disk.")

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
        if atom is not None:
            self.type = atom.type
            """The datatype to be stored by the sorted index array."""
            self.itemsize = atom.itemsize
            """The itemsize of the datatype to be stored by the index array."""
        self.column = column
        """The `Column` instance for the indexed column."""

        self.nrows = 0
        """The number of slices in the sorted/indices objects."""
        self.nelements = 0
        """The number of indexed elements in this index."""
        # The starts and lengths are initialized so that they can keep
        # info for last row procecessing at very least.
        self.starts = numarray.array(None, shape=1, type = numarray.Int32)
        """Where the values fulfiling conditions starts for every slice."""
        self.lengths = numarray.array(None, shape=1, type = numarray.Int32)
        """Lengths of the values fulfilling conditions for every slice."""

        self.cache = False   # no cache (ranges & bounds) is available initially

        # Set the version number of this object as an index, not a group.
        self._idx_version = idx_version

        # Index creation is never logged.
        super(Index, self).__init__(
            parentNode, name, title, new, filters, log=False)



    def _g_postInitHook(self):
        super(Index, self)._g_postInitHook()

        # Index arrays must only be created for new indexes
        if not self._v_new:
            # Set-up some variables from info on disk and return
            self.type = self.sorted.type
            self.itemsize = self.sorted.itemsize
            if self._idx_version == "pro":
                # The number of elements is at the end of the indices array
                nelementsLR = self.indicesLR[-1]
            else:
                nelementsLR = 0
            self.nrows = self.sorted.nrows
            self.nelements = self.nrows * self.nelemslice + nelementsLR
            self.nelementsLR = nelementsLR
            if nelementsLR > 0:
                self.nrows += 1
            self.shape = (self.nrows, self.nelemslice)
            # Get the bounds as a cache (this has to remain here!)
            nbounds = (nelementsLR -1 ) // self.chunksize
            if nbounds < 0:
                nbounds = 0 # correction for -1 bounds
            nbounds += 2 # bounds + begin + end
            # all bounds values (+begin+end) are at the beginning of lrvb
            self.bebounds = self.sortedLR[:nbounds]
            self.cache = False    # no cache (ranges & bounds) is available initially
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
        if str(self.type) == "CharType":
            atom = StringAtom(shape=(0, 2), length=self.itemsize,
                              flavor="numarray")
        else:
            atom = Atom(self.type, shape=(0,2), flavor="numarray")
        CacheArray(self, 'ranges', atom, "Range Values",
                   Filters(complevel=0, shuffle=0),   # too small to use filters
                   self._v_expectedrows//self.nelemslice)

        # Create the EArray for boundary values (2nd order cache)
        nbounds = (self.nelemslice - 1 ) // self.chunksize
        if str(self.type) == "CharType":
            atom = StringAtom(shape=(0, nbounds),
                              length=self.itemsize,
                              flavor="numarray")
        else:
            atom = Atom(self.type, shape=(0, nbounds))
        CacheArray(self, 'bounds', atom, "Boundary Values",
                   Filters(complevel=0, shuffle=0),   # too small to use filters
                   self._v_expectedrows//self.chunksize)

        # Create the Array for last (sorted) row values + bounds
        shape = 2 + nbounds + self.nelemslice
        if str(self.type) == "CharType":
            atom = strings.array(None, shape=shape, itemsize=self.itemsize)
        else:
            atom = numarray.array(None, shape=shape, type=self.type)
        LastRowArray(self, 'sortedLR', atom, "Last Row sorted values + bounds")

        # Create the Array for reverse indexes in last row
        shape = self.nelemslice     # enough for indexes and length
        atom = numarray.zeros(shape=shape, type=numarray.Int32)
        LastRowArray(self, 'indicesLR', atom, "Last Row reverse indices")


    def _g_updateDependent(self):
        super(Index, self)._g_updateDependent()
        self.column._updateIndexLocation(self)


    def append(self, arr):
        """Append the array to the index objects"""

        # Save the sorted array
        if str(self.sorted.type) == "CharType":
            s=arr.argsort()
        else:
            s=numarray.argsort(arr)
        # Caveat: this conversion is necessary for portability on
        # 64-bit systems because indexes are 64-bit long on these
        # platforms
        self.indices.append(numarray.array(s, type="Int32"))
        self.sorted.append(arr[s])
        #self.ranges.append([arr[s[[0,-1]]]])
        begend = [arr[s[[0,-1]]]]
        self.ranges.append(begend)
        self.bounds.append([arr[s[self.chunksize::self.chunksize]]])
        # Update nrows after a successful append
        self.nrows = self.sorted.nrows
        self.nelements = self.nrows * self.nelemslice
        self.shape = (self.nrows, self.nelemslice)
        self.nelementsLR = 0  # reset the counter of the last row index to 0
        self.cache = False   # the cache is dirty now


    def appendLastRow(self, arr, tnrows):
        """Append the array to the last row index objects"""

        # compute the elements in the last row sorted & bounds array
        nelementsLR = tnrows - self.sorted.nrows * self.nelemslice
        assert nelementsLR == len(arr), \
"The number of elements to append is incorrect!. Report this to the authors."
        # Sort the array
        if str(self.sorted.type) == "CharType":
            s=arr.argsort()
            # build the cache of bounds
            # this is a rather weird way of concatenating chararrays, I agree
            # We might risk loosing precision here....
            self.bebounds = arr[s[::self.chunksize]]
            self.bebounds.resize(self.bebounds.shape[0]+1)
            self.bebounds[-1] = arr[s[-1]]
        else:
            s=numarray.argsort(arr)
            # build the cache of bounds
            self.bebounds = numarray.concatenate([arr[s[::self.chunksize]],
                                                  arr[s[-1]]])
        # Save the reverse index array
        self.indicesLR[:len(arr)] = numarray.array(s, type="Int32")
        # The number of elements is at the end of the array
        self.indicesLR[-1] = nelementsLR
        # Save the number of elements, bounds and sorted values
        offset = len(self.bebounds)
        self.sortedLR[:offset] = self.bebounds
        self.sortedLR[offset:offset+len(arr)] = arr[s]
        # Update nelements after a successful append
        self.nrows = self.sorted.nrows + 1
        self.nelements = self.sorted.nrows * self.nelemslice + nelementsLR
        self.shape = (self.nrows, self.nelemslice)
        self.nelementsLR = nelementsLR


    #Aquest rutina vectoritzada no funciona com toca i arreglara-la no es trivial
    # user search_scalar en el seu lloc.
    def searchBinNA_vec(self, item1, item2):

        cs = self.chunksize
        nbounds = self.bounds.shape[1]
        rvc = self.rvcache
        nslice = self.nelemslice
        tlen = 0
        for nrow in xrange(self.sorted.nrows):
            nchunk = -1
            if self.sorted.bcache:
                ibounds = self.sorted.boundscache[nrow]
            else:
                ibounds = self.bounds[nrow]
#             start = (item1 <= rvc[nrow,0]) + (item1 > rvc[nrow,1]) * nslice
#             stop = (item2 < rvc[nrow,0]) + (item2 >= rvc[nrow,1]) * nslice
            start = (item1 < rvc[nrow,0]) + (item1 > rvc[nrow,1]) * nslice
            stop = (item2 < rvc[nrow,0]) + (item2 >= rvc[nrow,1]) * nslice
            #if start == 0:
            if start == 0 and item1 <= rvc[nrow,0]:
                #nchunk = bisect.bisect_left(ibounds, item1)
                nchunk = self.sorted._bisect_left(ibounds, item1, nbounds)
                chunk = self.sorted._readSortedSlice(nrow, cs*nchunk,
                                                     cs*(nchunk+1))
                start = self.sorted._bisect_left(chunk, item1, cs) + cs*nchunk
            if stop == 0:
                #nchunk2 = bisect.bisect_right(ibounds, item2)
                nchunk2 = self.sorted._bisect_right(ibounds, item2, nbounds)
                if nchunk2 <> nchunk:
                    # The chunk for item2 is different. Read the new chunk.
                    chunk = self.sorted._readSortedSlice(nrow, cs*nchunk2,
                                                         cs*(nchunk2+1))
                stop = self.sorted._bisect_right(chunk, item2, cs) + cs*nchunk2
            self.starts[nrow] = start
            len = stop - start
            self.lengths[nrow] = len
            tlen += len
        return tlen


    # This is an optimized version of search.
    # It does not work well with strings, because:
    # In [180]: a=strings.array(None, itemsize = 4, shape=1)
    # In [181]: a[0] = '0'
    # In [182]: a >= '0\x00\x00\x00\x01'
    # Out[182]: array([1], type=Bool)  # Incorrect
    # but...
    # In [183]: a[0] >= '0\x00\x00\x00\x01'
    # Out[183]: False  # correct
    #
    # While this is not a bug (see the padding policy for chararrays)
    # I think it would be much better to use '\0x00' as default padding
    #
    def search(self, item):
        """Do a binary search in this index for an item"""
        #t1 = time(); tcpu1 = clock()
        if str(self.type) == "CharType":
            tlen = self.search_scalar(item)
        else:
            # Do the lookup for values fullfilling the conditions
            if self.sorted.nrows > 0:
                if self._idx_version == "pro":
                    tlen = self.search_pro(item)
                else:
                    tlen = self.search_std(item)
            else:
                tlen = 0
        if self._idx_version == "pro" and self.nelementsLR > 0:
            # Look for more indexes in the last row
            (start, stop) = self.searchLastRow(item)
            self.starts[-1] = start
            self.lengths[-1] = stop - start
            tlen += stop - start
        #t = time()-t1
        #print "time searching indices (regular):", round(t*1000, 3), "ms"
        # CPU time not significant (problems with time slice)
        #print round((clock()-tcpu1)*1000*1000, 3), "us"
        return tlen


    def search_pro(self, item):
        """Do a binary search in this index for an item. pro version."""

        t1 = time()
        item1, item2 = item
        self.sorted._initSortedSlice(pro=1)
        # XXX Intentar tornar a activar l'optimitzacio
        if 0:
            tlen = self.searchBinNA_vec(item1, item2)
            #tlen = self.sorted._searchBinNA(item1, item2)
        else:
            # The next are optimizations. However, they hides the
            # CPU functions consumptions from python profiles
            # Activate only after development is done.
            if self.type == "Float64":
                # Both vectorial and scalar versions perform similar
                tlen = self.sorted._searchBinNA_d(item1, item2)
                # tlen = self.sorted._searchBinNA_d_vec(item1, item2)
            elif self.type == "Int32":
                # Buth vectorial and scalar versions perform similar
                tlen = self.sorted._searchBinNA_i(item1, item2)
                #tlen = self.search_scalar(item)
                # tlen = self.sorted._searchBinNA_i_vec(item1, item2)
            elif self.type == "Int64":
                # Buth vectorial and scalar versions perform similar
                tlen = self.sorted._searchBinNA_ll(item1, item2)
                # tlen = self.sorted._searchBinNA_ll_vec(item1, item2)
            else:
                tlen = self.search_scalar(item)
                #tlen = self.searchBinNA_vec(item1, item2)
        t = time()-t1
        #print "time searching indices (pro-main):", round(t*1000, 3), "ms"
        return tlen


    def search_std(self, item):
        """Do a binary search in this index for an item. std version."""

        t1 = time()
        self.sorted._initSortedSlice()
        for i in xrange(self.sorted.nrows):
            (start, stop) = self.sorted._searchBinStd(i, item)
            self.starts[i] = start
            self.lengths[i] = stop - start
        t = time()-t1
        #print "time searching indices (std-main):", round(t*1000, 3), "ms"
        return numarray.sum(self.lengths)


    # This is an scalar version of search. It works well with strings as well.
    def search_scalar(self, item):
        """Do a binary search in this index for an item."""
        tlen = 0
        # Do the lookup for values fullfilling the conditions
        if self._idx_version == "pro":
            self.sorted._initSortedSlice(pro=1)
            for i in xrange(self.sorted.nrows):
                (start, stop) = self.sorted._searchBin(i, item)
                self.starts[i] = start
                self.lengths[i] = stop - start
                tlen += stop - start
        else:
            self.sorted._initSortedSlice()
            for i in xrange(self.sorted.nrows):
                (start, stop) = self.sorted._searchBinStd(i, item)
                self.starts[i] = start
                self.lengths[i] = stop - start
                tlen += stop - start
        return tlen


    def searchLastRow(self, item):
        item1, item2 = item
        item1done = 0; item2done = 0

        t1=time()
        hi = hi2 = self.nelementsLR               # maximum number of elements
        bebounds = self.bebounds
        assert hi == self.nelements - self.sorted.nrows * self.nelemslice
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
        nbounds = len(bebounds)
        self.sorted._initSortedSlice(pro=1)
        readSliceLR = self.sortedLR._readSortedSlice
        nchunk = -1
        if not item1done:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect.bisect_left(bounds, item1)
            end = self.chunksize*(nchunk+1)
            if end > hi:
                end = hi
            chunk = readSliceLR(self.sorted, nbounds+self.chunksize*nchunk,
                                nbounds+end)
            if len(chunk) < hi:
                hi2 = len(chunk)
            result1 = bisect.bisect_left(chunk, item1, 0, hi2)
            result1 += self.chunksize*nchunk
        # Lookup in the middle of the slice for item2
        if not item2done:
            # Search the appropriate chunk in bounds cache
            nchunk2 = bisect.bisect_right(bounds, item2)
            if nchunk2 <> nchunk:
                end = self.chunksize*(nchunk2+1)
                if end > hi:
                    end = hi
                chunk = readSliceLR(self.sorted, nbounds+self.chunksize*nchunk2,
                                    nbounds+end)
                if len(chunk) < hi:
                    hi2 = len(chunk)
            result2 = bisect.bisect_right(chunk, item2, 0, hi2)
            result2 += self.chunksize*nchunk2
        t = time()-t1
        #print "time searching indices (last row):", round(t*1000, 3), "ms"
        return (result1, result2)


    # This version of getCoords reads the indexes in chunks.
    # Because of that, it can be used on iterators.
    # Version in pure python
    def getCoords(self, startCoords, nCoords):
        """Get the coordinates of indices satisfiying the cuts.

        You must call the Index.search() method before in order to get
        good sense results.

        This version is meant to be used in iterators.

        """
        #t1=time()
        len1 = 0; len2 = 0; relCoords = 0
        # Correction against asking too many elements
        nindexedrows = self.nelements
        if startCoords + nCoords > nindexedrows:
            nCoords = nindexedrows - startCoords
        # create buffers for indices
        self.indices._initIndexSlice(nCoords)
        for irow in xrange(self.nrows):
            leni = self.lengths[irow]; len2 += leni
            if (leni > 0 and len1 <= startCoords < len2):
                startl = self.starts[irow] + (startCoords-len1)
                # Read nCoords as maximum
                stopl = startl + nCoords
                # Correction if stopl exceeds the limits
                if stopl > self.starts[irow] + self.lengths[irow]:
                    stopl = self.starts[irow] + self.lengths[irow]
                if irow < self.sorted.nrows:
                    self.indices._readIndex(irow, startl, stopl, relCoords)
                else:
                    # Get indices for last row
                    offset = irow*self.nelemslice
                    stop = relCoords+(stopl-startl)
                    self.indices.arrAbs[relCoords:stop] = \
                         self.indicesLR[startl:stopl] + offset
                incr = stopl - startl
                relCoords += incr; startCoords += incr; nCoords -= incr
                if nCoords == 0:
                    break
            len1 += leni

        # I don't know if sorting the coordinates is better or not actually
        # Some careful tests must be carried out in order to do that
        selections = self.indices.arrAbs[:relCoords]
        # Preliminary results seems to show that sorting is not an advantage!
        #selections = numarray.sort(self.indices.arrAbs[:relCoords)]
        #t = time()-t1
        #print "time getting coords:",  round(t*1000, 3), "ms"
        return selections

# This tries to be a version of getCoords that keeps track of visited rows
# in order to not re-visit them again. However, I didn't managed to make it
# work well. However, the improvement in speed should be not important
# in most of cases.
# Beware, the logic behind doing this is not trivial at all. You have been
# warned!. 2004-08-03
#     def getCoords_notwork(self, startCoords, maxCoords):
#         """Get the coordinates of indices satisfiying the cuts"""
#         relCoords = 0
#         # Correction against asking too many elements
#         nindexedrows = self.nelemslice*self.nrows
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
#                 self.indices._readIndex(irow, startl, stopl, relCoords)
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


    # This version of getCoords reads all the indexes in one pass.
    # Because of that, it is not meant to be used on iterators.
    # This is the pure python version.
    def getCoords_sparse(self, ncoords):
        """Get the coordinates of indices satisfiying the cuts.

        You must call the Index.search() method before in order to get
        good sense results.

        This version is meant to be used for a complete read.

        """
        idc = self.indices
        # Initialize the index dataset
        idc._initIndexSlice(ncoords)
        # Create the sorted indices
        len1 = 0
        for irow in xrange(idc.nrows):
            for jrow in xrange(self.lengths[irow]):
                idc.coords[len1] = (irow, self.starts[irow]+jrow)
                len1 += 1

        # Given the sorted indices, get the real ones
        idc._readIndex_sparse(ncoords)

        # Finally, convert the values to full 64-bit addresses
        len1 = 0
        offset = idc.nelemslice * 1L
        for irow in xrange(idc.nrows):
            for jrow in xrange(self.lengths[irow]):
                idc.arrAbs[len1] = idc.arrRel[len1]+irow*offset
                len1 += 1

        # Get possible values in last slice
        if (self._idx_version == "pro" and self.nelementsLR > 0
            and self.lengths[idc.nrows] > 0):
            # Get indices for last row
            irow = idc.nrows
            offset = irow * self.nelemslice * 1L
            startl = self.starts[irow]
            stopl = startl + self.lengths[irow]
            idc.arrAbs[len1:ncoords] = self.indicesLR[startl:stopl] + offset

        selection = idc.arrAbs[:ncoords]
        return selection


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

        #t1=time()
        ncoords = self.search(item)
        #print "time reading indices:", time()-t1
        return ncoords


    def _f_remove(self, recursive=False):
        """Remove this Index object"""

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
        retstr = """%s (Index for column %s)
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
        if self._idx_version == "pro":
            retstr += "\n  ranges := %s" % self.ranges
            retstr += "\n  bounds := %s" % self.bounds
            retstr += "\n  sortedLR := %s" % self.sortedLR
            retstr += "\n  indicesLR := %s" % self.indicesLR
        return retstr
