########################################################################
#
#       License: BSD
#       Created: June 08, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /cvsroot/pytables/pytables/tables/Index.py,v $
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


"""

__version__ = "$Revision: 1.28 $"
# default version for INDEX objects
#obversion = "1.0"    # initial version
obversion = "1.1"    # optimization for very large columns and small selection
                     # groups

import cPickle
import warnings, sys
from IndexArray import IndexArray
from VLArray import Atom, StringAtom
from Array import Array
from EArray import EArray
from Leaf import Filters
from AttributeSet import AttributeSet
import hdf5Extension
#from hdf5Extension import PyNextAfter, PyNextAfterF
import numarray
from numarray import strings
from time import time
import math
import struct # we use this to define testNaN
import bisect

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

maxFloat=float(2**1024 - 2**971)  # From the IEEE 754 standard
maxFloatF=float(2**128 - 2**104)  # From the IEEE 754 standard

minFloat  = math.ldexp(1.0, -1022) # min positive normalized double
minFloatF = math.ldexp(1.0, -126)  # min positive normalized float

smallEpsilon  = math.ldexp(1.0, -1074) # smallest increment for doubles < minFloat
smallEpsilonF = math.ldexp(1.0, -149)  # smallest increment for floats < minFloatF

infinity = math.ldexp(1.0, 1023) * 2
infinityF = math.ldexp(1.0, 128)
#Finf=float("inf")  # Infinite in the IEEE 754 standard (not avail in Win)

# A portable representation of NaN
# if sys.byteorder == "little":
#     testNaN = struct.unpack("d", '\x01\x00\x00\x00\x00\x00\xf0\x7f')[0]
# elif sys.byteorder == "big":
#     testNaN = struct.unpack("d", '\x7f\xf0\x00\x00\x00\x00\x00\x01')[0]
# else:
#     raise RuntimeError, "Byteorder '%s' not supported!" % sys.byteorder
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
            raise ValueError, \
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

class Index(hdf5Extension.Group, hdf5Extension.Index, object):
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
        type -- The type class for the index.
        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        nrows -- The number of slices in index.
        nelements -- The total number of elements in the index.
        nelemslice -- The number of elements per slice.
        chunksize -- The HDF5 chunksize for each slice.
        filters -- The Filters instance for this object.
        dirty -- Whether the index is dirty or not.
        sorted -- The IndexArray object with the sorted values information.
        indices -- The IndexArray object with the sorted indices information.

    """

    def __init__(self, atom = None, where= None, name = None,
                 title = "", filters = None, expectedrows = 1000,
                 testmode = 0):
        """Create an Index instance.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. Only scalar atoms are
            supported.

        name -- The name for this Index object.

        where -- The indexed column who this instance pertains.

        title -- Sets a TITLE attribute of the Index entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object. If not specified, the ZLIB
            & shuffle will be activated by default (i.e., they are not
            inherited from the parent, that is, the Table).

        expectedrows -- Represents an user estimate about the number
            of row slices that will be added to the growable dimension
            in the IndexArray object. If not provided, the default
            value is 1000 slices.

        """
        self.name = name
        self._v_hdf5name = name
        self._v_new_title = title
        self._v_new_filters = filters
        self._v_expectedrows = expectedrows
        self.testmode = testmode
        self.column = where
        self._v_parent = where.table._v_parent  #Parent of table in object tree
        # Check whether we have to create a new object or read their contents
        # from disk
        if atom is not None:
            self._v_new = 1
            self.atom = atom
            self._create()
        else:
            self._v_new = 0
            self._open()

    def _g_join(self, name):
        """Helper method to correctly concatenate a name child object
        with the pathname of this group."""

        pathname = self._v_parent._g_join(self.name)
        if name == "/":
            # This case can happen when doing copies
            return pathname
        if pathname == "/":
            return "/" + name
        else:
            return pathname + "/" + name

    def _addAttrs(self, object, klassname):
        """ Add attributes to object """
        object.__dict__["_v_attrs"] = AttributeSet(object)
        object._v_attrs._g_setAttr('TITLE',  object._v_new_title)
        object._v_attrs._g_setAttr('CLASS', klassname)
        object._v_attrs._g_setAttr('VERSION', object._v_version)
        # Set the filters object
        if object._v_new_filters is None:
            # If not filters has been passed in the constructor,
            # set a sensible default, using zlib compression and shuffling
            filters = Filters(complevel = 1, complib = "zlib",
                              shuffle = 1, fletcher32 = 0)
            # Now, the filters are not inheritated. 2004-08-04
            #filters = object._v_parent._v_filters
        else:
            filters = object._v_new_filters
        filtersPickled = cPickle.dumps(filters, 0)
        object._v_attrs._g_setAttr('FILTERS', filtersPickled)
        # Add these attributes to the dictionary
        attrlist = ['TITLE','CLASS','VERSION','FILTERS']
        object._v_attrs._v_attrnames.extend(attrlist)
        object._v_attrs._v_attrnamessys.extend(attrlist)
        # Sort them
        object._v_attrs._v_attrnames.sort()
        object._v_attrs._v_attrnamessys.sort()
        return filters

    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        global obversion

        assert isinstance(self.atom, Atom), "The object passed to the IndexArray constructor must be a descendent of the Atom class."
        assert self.atom.shape == 1, "Only scalar columns can be indexed."
        # Version, type, shape, byteorder
        self._v_version = obversion
        self.type = self.atom.type
        self.title = self._v_new_title

        # Create the Index Group
        self._g_new(self._v_parent, self.name)
        self._v_objectID = self._g_createGroup()
        self.filters = self._addAttrs(self, "INDEX")

        # Create the IndexArray for sorted values
        object = IndexArray(self, self.atom, "Sorted Values",
                            self.filters, self._v_expectedrows,
                            self.testmode)
        object.name = object._v_name = object._v_hdf5name = "sortedArray"
        object._g_new(self, object.name)
        object.filters = self.filters
        object._create()
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        self._addAttrs(object, "IndexArray")
        self.sorted = object
        self.type = object.type
        self.shape = object.shape
        self.itemsize = object.itemsize
        self.chunksize = object.chunksize
        self.byteorder = object.byteorder
        # Create the IndexArray for index values
        object = IndexArray(self, Atom("Int32", shape=1), "Reverse Indices",
                            self.filters, self._v_expectedrows,
                            self.testmode)
        object.name = object._v_name = object._v_hdf5name = "revIndexArray"
        object._g_new(self, object.name)
        object.filters = self.filters
        object._create()
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        self.nrows = object.nrows
        self.nelemslice = object.nelemslice
        self.nelements = self.nrows * self.nelemslice
        self._addAttrs(object, "IndexArray")
        self.indices = object

        # Create the EArray for range values  (1st order cache)
        if str(self.type) == "CharType":
            atom = StringAtom(shape=(0, 2), length=self.itemsize,
                              flavor="CharArray")
        else:
            atom = Atom(self.type, shape=(0,2), flavor="NumArray")
        object = EArray(atom, "Range Values", self.filters,
                        self._v_expectedrows//self.nelemslice)
        object.name = object._v_name = object._v_hdf5name = "rangeValues"
        object._v_pathname = self._g_join(object._v_name)
        object._g_new(self, object.name)
        object.filters = self.filters
        object._create()
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        self.nrows = object.nrows
        self._addAttrs(object, "EArray")
        self.rangeValues = object

        # Create the EArray for boundary values (2nd order cache)
        nbounds = (self.nelemslice - 1 ) // self.chunksize
        if str(self.type) == "CharType":
            atom = StringAtom(shape=(0, nbounds),
                              length=self.itemsize,
                              flavor="CharArray")
        else:
            atom = Atom(self.type, shape=(0, nbounds))
        object = EArray(atom, "Boundaries", self.filters,
                        self._v_expectedrows//self.nelemslice)
        object.name = object._v_name = object._v_hdf5name = "bounds"
        object._v_pathname = self._g_join(object._v_name)
        object._g_new(self, object.name)
        object.filters = self.filters
        object._create()
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        self.nrows = object.nrows
        self._addAttrs(object, "EArray")
        self.bounds = object

        # Create the Array for last row values
        shape = 2 + nbounds + self.nelemslice
        if str(self.type) == "CharType":
            atom = strings.array(None, shape=shape, itemsize=self.itemsize)
        else:
            atom = numarray.array(None, shape=shape, type=self.type)
        object = Array(atom, "Last Row Values + bounds")
        object.name = object._v_name = object._v_hdf5name = "lrvb"
        object._v_pathname = self._g_join(object._v_name)
        object._g_new(self, object.name)
        object._v_parent = self
        object.filters = self.filters  # Needed by Array constructor
        object._create()
        object._v_file = self._v_parent._v_file
        self.nrows = object.nrows
        self._addAttrs(object, "Array")
        self.lrvb = object

        # Create the Array for reverse indexes in last row
        shape = self.nelemslice     # enough for indexes and length
        atom = numarray.zeros(shape=shape, type=numarray.Int32)
        object = Array(atom, "Last Row Reverse Indexes")
        object.name = object._v_name = object._v_hdf5name = "lrri"
        object._v_pathname = self._g_join(object._v_name)
        object._g_new(self, object.name)
        object._v_parent = self
        object.filters = self.filters  # Needed by Array constructor
        object._create()
        object._v_file = self._v_parent._v_file
        self.nrows = object.nrows
        self._addAttrs(object, "Array")
        self.lrri = object

    def _open(self):
        """Get the metadata info for an array in file."""
        self._g_new(self._v_parent, self.name)
        self._v_objectID = self._g_openIndex()
        self.__dict__["_v_attrs"] = AttributeSet(self)
        # Get the title and version attributes for this index
        self.title = self._v_attrs._g_getAttr("TITLE")
        self._v_version = self._v_attrs._g_getAttr("VERSION")
        # Open the IndexArray for sorted values
        object = IndexArray(parent=self)
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        object.name = object._v_name = object._v_hdf5name = "sortedArray"
        object._g_new(self, object._v_hdf5name)
        object.filters = object._g_getFilters()
        object._open()
        self.sorted = object
        self.type = object.type
        self.shape = object.shape
        self.itemsize = object.itemsize
        self.chunksize = object.chunksize
        self.byteorder = object.byteorder
        # Open the IndexArray for reverse Index values
        object = IndexArray(parent=self)
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        object.name = object._v_name = object._v_hdf5name = "revIndexArray"
        object._g_new(self, object._v_hdf5name)
        object.filters = object._g_getFilters()
        object._open()
        # these attrs should be the same for both sorted and indexed
        self.nrows = object.nrows
        self.nelemslice = object.nelemslice
        self.nelements = self.nrows * self.nelemslice
        self.filters = object.filters
        self.indices = object
        if self._v_version >= "1.1":
            # Open the rangeValues EArray
            object = EArray()
            object._v_parent = self
            object._v_file = self._v_parent._v_file
            object.name = object._v_name = object._v_hdf5name = "rangeValues"
            object._v_pathname = self._g_join(object._v_name)
            object._g_new(self, object._v_hdf5name)
            object.filters = object._g_getFilters()
            object._open()
            self.rangeValues = object
            #self.rvcache = self.rangeValues[:]  # rangeValues cache
            # Mark the rangeValues as non-vaild.
            # This optimizes the opening of files
            self.rvcache = None
            # Open the bounds EArray
            object = EArray()
            object._v_parent = self
            object._v_file = self._v_parent._v_file
            object.name = object._v_name = object._v_hdf5name = "bounds"
            object._v_pathname = self._g_join(object._v_name)
            object._g_new(self, object._v_hdf5name)
            object.filters = object._g_getFilters()
            object._open()
            self.bounds = object
            # Open the Last Row Values & Bounds Array
            object = Array()
            object._v_parent = self
            object._v_file = self._v_parent._v_file
            object.name = object._v_name = object._v_hdf5name = "lrvb"
            object._v_pathname = self._g_join(object._v_name)
            object._g_new(self, object._v_hdf5name)
            object.filters = object._g_getFilters()
            object._open()
            self.lrvb = object
            # Open the Last Row Reverse Index Array
            object = Array()
            object._v_parent = self
            object._v_file = self._v_parent._v_file
            object.name = object._v_name = object._v_hdf5name = "lrri"
            object._v_pathname = self._g_join(object._v_name)
            object._g_new(self, object._v_hdf5name)
            object.filters = object._g_getFilters()
            object._open()
            self.lrri = object
            self.nelementsLR = self.lrri[-1]
            #print "nelementsLR-->", self.nelementsLR
            if self.nelementsLR > 0:
                self.nrows += 1
                self.nelements += self.nelementsLR
                self.shape = (self.nrows, self.nelemslice)
                # Get the bounds as a cache
                chunksize = self.chunksize
                nbounds = (self.nelementsLR -1 ) // self.chunksize
                if nbounds < 0:
                    nbounds = 0 # correction for -1 bounds
                nbounds += 2 # bounds + begin + end
                # all bounds values (+begin+end) are at the beginning of lrvb
                self.bebounds = self.lrvb[:nbounds]


# The next does not seem to be necessary
        #dirty = self._v_attrs._g_getAttr("DIRTY")
#         dirty = getattr(self._v_attrs, "DIRTY", 0)
#         print "dirty-->", dirty
#         if dirty is not None:
#             # All the counters has to be reset
#             self.nrows = 0
#             self.nelements = 0
#             self.nelementsLR = 0
#             self.bebounds = None

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
        #self.rangeValues.append([arr[s[[0,-1]]]])
        begend = [arr[s[[0,-1]]]]
        self.rangeValues.append(begend)
        self.bounds.append([arr[s[self.chunksize::self.chunksize]]])
        # Update nrows after a successful append
        self.nrows = self.sorted.nrows
        self.nelements = self.nrows * self.nelemslice
        self.shape = (self.nrows, self.nelemslice)
        self.nelementsLR = 0  # reset the counter of the last row index to 0
        self.rvcache = None   # the rangeValues caches is dirty now
        # This takes some time is not worth the extra effort
        #self.rvcache = numarray.concatenate([self.rvcache,begend])

    def appendLastRow(self, arr, tnrows):
        """Append the array to the last row index objects"""

        # compute the elements in the last row vaules & bounds array
        nelementsLR = tnrows - self.sorted.nrows * self.nelemslice
        assert nelementsLR == len(arr), "The number of elements to append is incorrect!. Report this to the authors."
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
        self.lrri[:len(arr)] = numarray.array(s, type="Int32")
        self.lrri[-1] = nelementsLR   # The number of elements is at the end
        # Save the number of elements, bounds and sorted values
        offset = len(self.bebounds)
        self.lrvb[:offset] = self.bebounds
        self.lrvb[offset:offset+len(arr)] = arr[s]
        # Update nelements after a successful append
        self.nrows = self.sorted.nrows + 1
        self.nelements = self.sorted.nrows * self.nelemslice + nelementsLR
        self.shape = (self.nrows, self.nelemslice)
        self.nelementsLR = nelementsLR

    def _searchBinLastRow(self, item):
        item1, item2 = item
        item1done = 0; item2done = 0

        #t1=time()
        hi = self.nelementsLR               # maximum number of elements
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
            #print "done 1-->", time()-t1
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
            #print "done 2-->", time()-t1
            return (result1, result2)
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        bounds = bebounds[1:-1] # Get the bounds array w/out begin and end
        nbounds = len(bebounds)
        if not item1done:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect.bisect_left(bounds, item1)
            end = self.chunksize*(nchunk+1)
            if end > hi:
                end = hi
            chunk = self.lrvb[nbounds+self.chunksize*nchunk:nbounds+end]
            result1 = bisect.bisect_left(chunk, item1)
            result1 += self.chunksize*nchunk
        # Lookup in the middle of slice for item2
        if not item2done:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect.bisect_right(bounds, item2)
            end = self.chunksize*(nchunk+1)
            if end > hi:
                end = hi
            chunk = self.lrvb[nbounds+self.chunksize*nchunk:nbounds+end]
            result2 = bisect.bisect_right(chunk, item2)
            result2 += self.chunksize*nchunk
        #print "done 3-->", time()-t1
        return (result1, result2)

    def searchBinNA(self, nrow, item1, item2, result1, result2):

        ibounds = self.bounds[nrow]
        if result1[nrow] < 0:
            nchunk = bisect.bisect_left(ibounds, item1)
            chunk = self.sorted._readSortedSlice(nrow, self.chunksize*nchunk,
                                                 self.chunksize*(nchunk+1))
            result1[nrow] = self.sorted._bisect_left(chunk, item1, self.chunksize) + self.chunksize*nchunk
        if result2[nrow] < 0:
            nchunk = bisect.bisect_right(ibounds, item2)
            chunk = self.sorted._readSortedSlice(nrow, self.chunksize*nchunk,
                                                 self.chunksize*(nchunk+1))
            result2[nrow] = self.sorted._bisect_right(chunk, item2, self.chunksize) + self.chunksize*nchunk
        return

    # This is a vectorial version of search.
    # It does not work well with strings, because:
    # In [180]: a=strings.array(None, itemsize = 4, shape=1)
    # In [181]: a[0] = '0'
    # In [182]: a >= '0\x00\x00\x00\x01'
    # Out[182]: array([1], type=Bool)  # Incorrect
    # but...
    # In [183]: a[0] >= '0\x00\x00\x00\x01'
    # Out[183]: False  # correct
    # While this is not a bug (see the padding policy for chararrays)
    # I think it would be much better to use '\0x00' as default padding
    def search(self, item):
        """Do a binary search in this index for an item"""
        if str(self.type) == "CharType":
            return self.search_original(item)
        #t1=time()
        item1, item2 = item
        self.sorted._initSortedSlice(self.chunksize)
        # Internal Buffers
        self.starts = numarray.array(None,shape=(self.nrows,),
                                     type = numarray.Int32)
        self.lengths = numarray.array(None,shape=(self.nrows,),
                                      type = numarray.Int32)
        # Do the lookup for values fullfilling the conditions
        if self._v_version >= "1.1":
            if self.rvcache is None:
                self.rvcache = self.rangeValues[:]
            # Compute starts
            begin = self.rvcache[:,0]
            end = self.rvcache[:,1]
            r11 = (item1 <= begin)
            r12 = (item1 >  end)
            starts = (r11 + r12 * (self.nelemslice+1)) - 1
            # Compute stops
            r21 = (item2 < begin)
            r22 = (item2 >=  end)
            stops = (r21 + r22 * (self.nelemslice+1)) - 1
            # Get the remaining values
            for i in xrange(self.sorted.nrows):
                if starts[i] == -1 or stops[i] == -1:
                    self.searchBinNA(i, item1, item2, starts, stops)
            # compute lengths
            if self.nelementsLR:
                self.starts[:-1] = starts
                self.lengths[:-1] = stops - starts
            else:
                self.starts[:] = starts
                self.lengths[:] = stops - starts
        else:
            for i in xrange(self.sorted.nrows):
                (start, stop) = self.sorted._searchBin1_0(i, item)
                self.starts[i] = start
                self.lengths[i] = stop - start
        self.sorted._destroySortedSlice()
        if self._v_version >= "1.1" and self.nelementsLR:
            # Look for more indexes in the last row
            (start, stop) = self._searchBinLastRow(item)
            self.starts[-1] = start
            self.lengths[-1] = stop - start
        tlen = numarray.sum(self.lengths)
        #print "time reading indices: %6f" % round(time()-t1,6),
        #print "  selected:", tlen
        #assert tlen >= 0, "Index.search(): Post-condition failed. Please, report this to the authors."
        return tlen

    # This is an scalar version of search. It works well with strings as well.
    def search_original(self, item):
    #def search(self, item):
        """Do a binary search in this index for an item"""
        t1=time()
        tlen = 0
        self.sorted._initSortedSlice(self.chunksize)
        #self._v_version = "1.0"  # just for test speed comparisons
        # Internal Buffers
        self.starts = numarray.array(None,shape=(self.nrows,),
                                     type = numarray.Int32)
        self.lengths = numarray.array(None,shape=(self.nrows,),
                                      type = numarray.Int32)
        # Do the lookup for values fullfilling the conditions
        if self._v_version >= "1.1":
            if self.rvcache is None:
                self.rvcache = self.rangeValues[:]
            for i in xrange(self.sorted.nrows):
                (start, stop) = self.sorted._searchBin(i, item)
                self.starts[i] = start
                self.lengths[i] = stop - start
                tlen += stop - start
#            print "starts2-->", self.starts
        else:
            for i in xrange(self.sorted.nrows):
                (start, stop) = self.sorted._searchBin1_0(i, item)
                self.starts[i] = start
                self.lengths[i] = stop - start
                tlen += stop - start
        self.sorted._destroySortedSlice()
        if self._v_version >= "1.1" and self.nelementsLR > 0:
            # Look for more indexes in the last row
            (start, stop) = self._searchBinLastRow(item)
            self.starts[-1] = start
            self.lengths[-1] = stop - start
            tlen += stop - start
        #print "time reading indices: %6f" % round(time()-t1,6),
        #print "  selected:", tlen
        #print "starts, lengths-->", self.starts, self.lengths
        #assert tlen >= 0, "Index.search(): Post-condition failed. Please, report this to the authors."
        #return tlen
        return tlen

# This has been ported to Pyrex. However, with pyrex it has the same speed,
# so, it's better to stay here
    def getCoords(self, startCoords, maxCoords):
        """Get the coordinates of indices satisfiying the cuts.

        You must call the Index.search() method before in order to get
        good sense results.

        """
        t1=time()
        len1 = 0; len2 = 0; relCoords = 0
        # Correction against asking too many elements
        nindexedrows = self.nelements
        if startCoords + maxCoords > nindexedrows:
            maxCoords = nindexedrows - startCoords
        for irow in xrange(self.nrows):
            leni = self.lengths[irow]; len2 += leni
            if (leni > 0 and len1 <= startCoords < len2):
                startl = self.starts[irow] + (startCoords-len1)
                # Read maxCoords as maximum
                stopl = startl + maxCoords
                # Correction if stopl exceeds the limits
                if stopl > self.starts[irow] + self.lengths[irow]:
                    stopl = self.starts[irow] + self.lengths[irow]
                if irow < self.sorted.nrows:
                    self.indices._g_readIndex(irow, startl, stopl, relCoords)
                else:
                    # Get indices for last row
                    offset = irow*self.nelemslice
                    stop = relCoords+(stopl-startl)
                    self.indices.arrAbs[relCoords:stop] = \
                         self.lrri[startl:stopl] + offset
                incr = stopl - startl
                relCoords += incr; startCoords += incr; maxCoords -= incr
                if maxCoords == 0:
                    break
            len1 += leni

        # I don't know if sorting the coordinates is better or not actually
        # Some careful tests must be carried out in order to do that
        #selections = self.indices.arrAbs[:relCoords]
        selections = numarray.sort(self.indices.arrAbs[:relCoords])
        #print "time getting coords:", time()-t1
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
        # Get the coordenates for those values
        ilimit = table.opsValues
        ctype = column.type
        itemsize = table.colitemsizes[column.name]
        # Check that limits are compatible with type
        for limit in ilimit:
            # Check for strings
            if str(ctype) == "CharType":
                assert type(limit) == str, \
"Bounds (or range limits) for strings columns can only be strings."
            # Check for booleans
            elif isinstance(numarray.typeDict[str(ctype)],
                          numarray.BooleanType):
                assert type(limit) in (int,long,bool), \
"Bounds (or range limits) for bool columns can only be ints or booleans."
            # Check for ints
            elif isinstance(numarray.typeDict[str(ctype)],
                          numarray.IntegralType):
                assert (type(limit) in (int,long,float)), \
"Bounds (or range limits) for integer columns can only be ints or floats."
            # Check for floats
            elif isinstance(numarray.typeDict[str(ctype)],
                          numarray.FloatingType):
                assert (type(limit) in (int,long,float)), \
"Bounds (or range limits) for float columns can only be ints or floats."
            else:
                raise ValueError, \
"Bounds (or range limits) can only be strings, bools, ints or floats."

        # Boolean types are a special case for searching
        if str(ctype) == "Bool":
            if len(table.ops) == 1 and table.ops[0] == 5: # __eq__
                item = (ilimit[0], ilimit[0])
                ncoords = self.search(item)
                return ncoords
            else:
                raise NotImplementedError, \
                      "Only equality operator is suported for boolean columns."
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
            assert item1 <= item2, \
"On 'val1 <{=} col <{=} val2' selections, val1 must be less or equal than val2"
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
                raise SyntaxError, \
"Combination of operators not supported. Use val1 <{=} col <{=} val2"

        #t1=time.time()
        ncoords = self.search(item)
        #print "time searching indices:", time.time()-t1
        return ncoords

    def _g_remove(self):
        """Remove this Index object"""

        if hdf5Extension.whichLibVersion("hdf5")[1] == "1.6.3":
            warnings.warn( \
"""\nYou are using HDF5 version 1.6.3. It turns out that this precise
version has a bug that causes a seg fault when deleting a chunked
dataset. If you are getting such a seg fault immediately after this
message, please, get a patched version of HDF5 1.6.3.""", UserWarning)

        # Delete the associated IndexArrays
        #self.sorted._close()
        #self.indices._close()
        self.sorted.flush()
        self.indices.flush()
        # The next cannot be done because sortedArray and revIndexArray are
        # not regular Leafs
        #self.sorted.remove()
        #self.indices.remove()
        self._g_deleteLeaf(self.sorted.name)
        self._g_deleteLeaf(self.indices.name)
        self._g_deleteLeaf(self.bounds.name)
        self._g_deleteLeaf(self.lrvb.name)
        self._g_deleteLeaf(self.lrri.name)
        # Delete the caches
        self.rvcache = None
        self.bebounds = None
        self.nelementsLR = 0
        # delete the pointers to the objects
        self.sorted = None
        self.indices = None
        self.bounds = None
        self.lrvb = None
        self.lrri = None
        # close the Index Group
        self._f_close()
        # delete it (this is defined in hdf5Extension)
        self._g_deleteGroup()

    def _f_close(self):
        # close the indices
        if self.sorted:  # that might be already removed
            self.sorted._close()
        if self.indices: # that might be already removed
            self.indices._close()
        # Aco done problemes....
#         if self.bounds: # that might be already removed
#             self.bounds._close()
#         if self.lrvb: # that might be already removed
#             self.lrvb._close()
#         if self.lrri: # that might be already removed
#             self.lrri._close()
        # delete some references
        self.atom=None
        self.column=None
        self.filters=None
        self.indices = None
        self.sorted = None
        self.bounds = None
        self.lrvb = None
        self.lrri = None
        self._v_attrs = None
        self._v_parent = None
        # Close this group
        self._g_closeGroup()
        #self.__dict__.clear()

    def __str__(self):
        """This provides a more compact representation than __repr__"""
        return "Index(%s, shape=%s, chunksize=%s)" % \
               (self.nelements, self.shape, self.chunksize)

    def __repr__(self):
        """This provides more metainfo than standard __repr__"""

        cpathname = self.column.table._v_pathname + ".cols." + self.column.name
        pathname = self._v_parent._g_join(self.name)
        dirty = self.column.dirty
        #print "-->", self.sorted[:]
        #print "-->", self.rangeValues[:]
        retstr = """%s (Index for column %s)
  type := %r
  nelements := %s
  shape := %s
  chunksize := %s
  byteorder := %r
  filters := %s
  dirty := %s
  sorted := %s
  indices := %s""" % (pathname, cpathname,
                          self.type, self.nelements, self.shape,
                          self.chunksize, self.byteorder,
                          self.filters, dirty, self.sorted, self.indices)

        if self._v_version >= "1.1":
            retstr += "\n  rangeValues := %s" % self.rangeValues
            retstr += "\n  bounds := %s" % self.bounds
            retstr += "\n  lrvb := %s" % self.lrvb
            retstr += "\n  lrri := %s" % self.lrri
        return retstr
