########################################################################
#
#       License: BSD
#       Created: June 08, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Index.py,v $
#       $Id: Index.py,v 1.11 2004/08/03 21:02:53 falted Exp $
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

__version__ = "$Revision: 1.11 $"
# default version for INDEX objects
obversion = "1.0"    # initial version

import cPickle
import types, warnings, sys
from IndexArray import IndexArray
from VLArray import Atom
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
    "Return the next representable neighbor of x in the apprppriate direction."

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

        type -- The type class for the array.
        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        flavor -- The flavor of this object.
        nrows -- The number of slices in index.
        nelemslice -- The number of elements per slice.
        chunksize -- The HDF5 chunksize for each slice.
        filters -- The Filters instance for this object.
            

    """

    def __init__(self, atom = None, where= None, name = None,
                 title = "", filters = None, expectedrows = 1000,
                 testmode = 0):
        """Create an IndexArray instance.

        Keyword arguments:

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. Only scalar atoms are
            supported.

        name -- The name for this Index object.

        where -- The indexed column who this instance pertains.
        
        title -- Sets a TITLE attribute of the Index entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

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

    def _addAttrs(self, object, klassname):
        """ Add attributes to object """
        object.__dict__["_v_attrs"] = AttributeSet(object)
        object._v_attrs._g_setAttr('TITLE',  object._v_new_title)
        object._v_attrs._g_setAttr('CLASS', klassname)
        object._v_attrs._g_setAttr('VERSION', object._v_version)
        # Set the filters object
        if object._v_new_filters is None:
            # If not filters has been passed in the constructor,
            filters = object._v_parent._v_filters
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
        # Version, type, shape, flavor, byteorder
        self._v_version = obversion
        self.type = self.atom.type
        self.title = self._v_new_title
        # Create the Index Group
        self._g_new(self._v_parent, self.name)
        self._v_objectID = self._g_createGroup()
        self.filters = self._addAttrs(self, "INDEX")
        # Create the IndexArray for sorted values
        object = IndexArray(self.atom, "Sorted Values",
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
        object = IndexArray(Atom("Int32", shape=1), "Reverse indices",
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
        self._addAttrs(object, "IndexArray")
        self.indices = object
            
    def _open(self):
        """Get the metadata info for an array in file."""
        self._g_new(self._v_parent, self.name)
        self._v_objectID = self._g_openIndex()
        # Get the title, filters attributes for this index
        self.title = self._g_getAttr("TITLE")
        self.filters = cPickle.loads(self._g_getAttr("FILTERS"))
        # Open the IndexArray for sorted values
        object = IndexArray()
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
        object = IndexArray()
        object._v_parent = self
        object._v_file = self._v_parent._v_file
        object.name = object._v_name = object._v_hdf5name = "revIndexArray"
        object._g_new(self, object._v_hdf5name)
        object.filters = object._g_getFilters()
        object._open()
        self.nrows = object.nrows
        self.nelemslice = object.nelemslice
        self.indices = object

    def append(self, arr):
        """Append the object to this (enlargeable) object"""

        # Save the sorted array
        if str(self.sorted.type) == "CharType":
            self.indices.append(arr.argsort())
            arr.sort()
            self.sorted.append(arr)
        else:
            self.sorted.append(numarray.sort(arr))
            self.indices.append(numarray.argsort(arr))
        # Update nrows after a successful append
        self.nrows = self.sorted.nrows
        
    def search(self, item, notequal):
        """Do a binary search in this index for an item"""
        #t1=time.time()
        ntotaliter = 0; tlen = 0
        self.starts = []; self.lengths = []
        self.sorted._initSortedSlice(self.chunksize)
        # Do the lookup for values fullfilling the conditions
        for i in xrange(self.sorted.nrows):
            (start, stop, niter) = self.sorted._searchBin(i, item)
            self.starts.append(start)
            self.lengths.append(stop - start)
            #print "start, stop-->", start, stop
            ntotaliter += niter
            tlen += stop - start
        self.sorted._destroySortedSlice()
        #print "time reading indices:", time.time()-t1
        #print "ntotaliter-->", ntotaliter
        return tlen

# This has been passed to Pyrex. However, with pyrex it has the same speed,
# so, it's better to stay here
    def getCoords(self, startCoords, maxCoords):
        """Get the coordinates of indices satisfiying the cuts"""
        len1 = 0; len2 = 0;
        relCoords = 0
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
                relCoords += stopl - startl
                break
            len1 += leni

        # I don't know if sorting the coordinates is better or not actually
        # Some careful tests must be carried out in order to do that
        selections = numarray.sort(self.indices.arrAbs[:relCoords])
        #selections = self.indices.arrAbs[:relCoords]
        return selections

# This tried to be a version of getCoords that would merge several
# selected ranges in different rows in one single selection return array
# However, I didn't managed to make it work well.
# Beware, the logic behind doing this is not trivial at all. You have been
# warned!. 2004-08-03

#     def getCoords_orig_modif(self, startCoords, maxCoords):
#         """Get the coordinates of indices satisfiying the cuts"""
#         #t1=time.time()
#         len1 = 0; len2 = 0;
#         stop = 0; relCoords = 0
#         # Correction against asking too many elements
#         nindexedrows = self.nelemslice*self.nrows
#         #print "nindexedrows-->", nindexedrows
#         if startCoords + maxCoords > nindexedrows:
#             maxCoords = nindexedrows - startCoords
#         print "startCoords, maxCoords-->(1)", startCoords, maxCoords
#         #print "lengths-->", self.lengths
#         irow=self.irow
#         #for irow in xrange(self.sorted.nrows):
#         while irow < self.sorted.nrows:
#             leni = self.lengths[irow]; len2 += leni
#             print "len1, leni, len2-->", len1, leni, len2
#             print "startCoords, maxCoords-->(loop)", startCoords, maxCoords
#             newrow = 1
#             if (leni > 0 and len1 <= startCoords < len2):
#                 startl = self.starts[irow] + (startCoords-len1)
#                 print "maxCoords, startl, leni-->", maxCoords, startl, leni
#                 #if maxCoords >= leni - (startCoords-len1):
#                 if (startl + leni) < maxCoords:
#                     # Values do fit on buffer
#                     stopl = startl + leni
#                     maxCoords -= leni - (startCoords-len1)
#                     startCoords += leni - (startCoords-len1)
#                     #startCoords = len1 + leni
#                     newrow=0  # Don't increment the row number in next iter
#                 elif (startl + leni) > len2:
#                     # Values fit on buffer, but we run out this section
#                     stopl = len2
#                     lenj = stopl - startl
#                     maxCoords -= lenj
#                     startCoords = len1 + lenj
#                     #startCoords += lenj - (startCoords-len1)
#                     newrow=0  # Don't increment the row number in next iter
#                 else:
#                     # Read maxCoords as maximum
#                     stopl = startl + maxCoords
#                     # Stop after this iteration
#                     stop = 1
# ####                # Correction if stopl exceeds the limits
#                 if stopl > self.nelemslice:
#                     stopl = self.nelemslice
#                     newrow=0  # Don't increment the row number in next iter
#                 print "irow, startl, stopl-->", irow, startl, stopl, stop
#                 self.indices._g_readIndex(irow, startl, stopl, relCoords)
#                 print "arrAbs-->", self.indices.arrAbs
#                 relCoords += stopl - startl
#                 if stop:
#                     break
#                 len1 += stopl-startl
#             else:
#                 len1 += leni
#             if newrow: self.irow += 1

#         # I don't know if sorting the coordinates is better or not actually
#         # Some careful tests must be carried out in order to do that
#         selections = numarray.sort(self.indices.arrAbs[:relCoords])
#         #selections = self.indices.arrAbs[:relCoords]
#         print "selections-->", selections
#         #print "time doing revIndexing:", time.time()-t1
#         return selections

    def _getLookupRange(self, column):
        #import time
        table = column.table
        # Get the coordenates for those values
        ilimit = table.opsValues
        ctype = column.type
        itemsize = table.colitemsizes[column.name]
        notequal = 0
        # Boolean types are a special case
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
            op1, op2 = table.ops
            item1, item2 = ilimit
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

    def _f_remove(self):
        """Remove this Index object"""

        # First, close the Index Group
        self._f_close()
        # Then, delete it (this is defined in hdf5Extension)
        self._g_deleteGroup()

    def _f_close(self):
        # close the indices
        self.sorted._close()
        self.indices._close()
#         self.sorted.flush()
#         self.indices.flush()
        # Delete some references to the object tree
#         del self.indices._v_parent
#         del self.sorted._v_parent
#         if hasattr(self.indices, "_v_file"):
#             del self.indices._v_file
#             del self.sorted._v_file
        del self.indices
        del self.sorted
        del self._v_parent
        # Close this group
        self._g_closeGroup()
        self.__dict__.clear()

    def __str__(self):
        """This provides more metainfo in addition to standard __repr__"""
        return "Index()"
        
    def __repr__(self):
        """This provides more metainfo in addition to standard __repr__"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  nelemslice = %s
  chunksize = %s
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.nelemslice, self.chunksize, self.byteorder)
