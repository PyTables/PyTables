########################################################################
#
#       License: BSD
#       Created: June 08, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Index.py,v $
#       $Id: Index.py,v 1.20 2004/09/28 17:17:50 falted Exp $
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

__version__ = "$Revision: 1.20 $"
# default version for INDEX objects
obversion = "1.0"    # initial version

import cPickle
import types, warnings, sys
from IndexArray import IndexArray
from VLArray import Atom
from Leaf import Filters
from AttributeSet import AttributeSet
import hdf5Extension
from hdf5Extension import PyNextAfter, PyNextAfterF
import numarray
import time

maxFloat=float(2**1024 - 2**971)  # From the IEEE 754 standard
maxFloatF=float(2**128 - 2**104)  # From the IEEE 754 standard
Finf=float("inf")  # Infinite in the IEEE 754 standard

# Utility functions
def infType(type, itemsize, sign=0):
    """Return a superior limit for maximum representable data type"""
    if str(type) != "CharType":
        if sign:
            return -Finf
        else:
            return Finf
    else:
        if sign:
            return "\x00"*itemsize
        else:
            return "\xff"*itemsize

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

    if type in ["Int8", "UInt8","Int16", "UInt16",
                "Int32", "UInt32","Int64", "UInt64"]:
        if direction < 0:
            return x-1
        else:
            return x+1
    elif type == "Float32":
        if direction < 0:
            return PyNextAfterF(x,x-1)
        else:
            return PyNextAfterF(x,x+1)
    elif type == "Float64":
        if direction < 0:
            return PyNextAfter(x,x-1)
        else:
            return PyNextAfter(x,x+1)
    elif str(type) == "CharType":
        return CharTypeNextAfter(x, direction, itemsize)
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
            
    def _open(self):
        """Get the metadata info for an array in file."""
        self._g_new(self._v_parent, self.name)
        self._v_objectID = self._g_openIndex()
        self.__dict__["_v_attrs"] = AttributeSet(self)
        # Get the title, filters attributes for this index
        self.title = self._v_attrs._g_getAttr("TITLE")
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
            # Caveat: this is conversion necessary for portability on
            # 64-bit systems because indexes are 64-bit long on these
            # platforms
            self.indices.append(numarray.array(s, type="Int32"))
            self.sorted.append(arr[s])
        # Update nrows after a successful append
        self.nrows = self.sorted.nrows
        self.nelements = self.nrows * self.nelemslice
        self.shape = (self.nrows, self.nelemslice)
        
    def search(self, item, notequal):
        """Do a binary search in this index for an item"""
        #t1=time.time()
        ntotaliter = 0; tlen = 0
        self.starts = []; self.lengths = []
        #self.irow = 0; self.len1 = 0; self.len2 = 0;  # useful for getCoords()
        self.sorted._initSortedSlice(self.chunksize)
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
        assert tlen >= 0, "Post-condition failed. Please, report this to the authors."
        return tlen

# This has been passed to Pyrex. However, with pyrex it has the same speed,
# so, it's better to stay here
    def getCoords(self, startCoords, maxCoords):
        """Get the coordinates of indices satisfiying the cuts.

        You must call the Index.search() method before in order to get
        good sense results.

        """
        #t1=time.time()
        len1 = 0; len2 = 0; relCoords = 0
        # Correction against asking too many elements
        nindexedrows = self.nelemslice*self.nrows
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
        notequal = 0
        # Check that limits are compatible with type
        for limit in ilimit:
            # Check for strings
            if str(ctype) == "CharType":
                assert type(limit) == types.StringType, \
"Bounds (or range limits) for strings columns can only be strings."
            # Check for booleans
            elif isinstance(numarray.typeDict[str(ctype)],
                          numarray.BooleanType):
                assert (type(limit) == types.IntType or
                        type(limit) == types.BooleanType), \
"Bounds (or range limits) for bool columns can only be ints or booleans."
            # Check for ints
            elif isinstance(numarray.typeDict[str(ctype)],
                          numarray.IntegralType):
                assert (type(limit) == types.IntType or
                        type(limit) == types.FloatType), \
"Bounds (or range limits) for integer columns can only be ints or floats."
            # Check for floats
            elif isinstance(numarray.typeDict[str(ctype)],
                          numarray.FloatingType):
                assert (type(limit) == types.IntType or
                        type(limit) == types.FloatType), \
"Bounds (or range limits) for float columns can only be ints or floats."
            else:
                raise ValueError, \
"Bounds (or range limits) can only be strings, bools, ints or floats."
            
        # Boolean types are a special case for searching
        if str(ctype) == "Bool":
            if len(table.ops) == 1 and table.ops[0] == 5: # __eq__
                item = (ilimit[0], ilimit[0])
                ncoords = self.search(item, notequal=0)
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
                notequal = 1
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
        ncoords = self.search(item, notequal)
        #print "time reading indices:", time.time()-t1
        return ncoords

    def _g_remove(self):
        """Remove this Index object"""
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
        self.sorted = None
        self.indices = None
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
        # delete some references
        self.atom=None
        self.column=None
        self.filters=None
        self.indices = None
        self.sorted = None
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
        return """%s (Index for column %s)
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
