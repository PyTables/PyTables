########################################################################
#
#       License: BSD
#       Created: June 08, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Index.py,v $
#       $Id: Index.py,v 1.7 2004/07/07 17:11:14 falted Exp $
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

__version__ = "$Revision: 1.7 $"
# default version for INDEX objects
obversion = "1.0"    # initial version

import cPickle
import types, warnings, sys
from IndexArray import IndexArray
from VLArray import Atom
from AttributeSet import AttributeSet
import hdf5Extension
import numarray
import time

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
                 title = "", filters = None, expectedrows = 1000):
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
        object._v_attrs._g_setAttrStr('TITLE',  object._v_new_title)
        object._v_attrs._g_setAttrStr('CLASS', klassname)
        object._v_attrs._g_setAttrStr('VERSION', object._v_version)
        # Set the filters object
        if object._v_new_filters is None:
            # If not filters has been passed in the constructor,
            filters = object._v_parent._v_filters
        else:
            filters = object._v_new_filters
        filtersPickled = cPickle.dumps(filters, 0)
        object._v_attrs._g_setAttrStr('FILTERS', filtersPickled)
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
                            self.filters, self._v_expectedrows)
        object.name = object._v_name = object._v_hdf5name = "sortedArray"
        object._g_new(self, object.name)
        object.filters = self.filters
        object._create()
        object._v_parent = self
        self._addAttrs(object, "IndexArray")
        self.sorted = object
        # Create the IndexArray for index values
        object = IndexArray(Atom("Int32", shape=1), "Reverse indices",
                            self.filters, self._v_expectedrows)
        object.name = object._v_name = object._v_hdf5name = "revIndexArray"
        object._g_new(self, object.name)
        object.filters = self.filters
        object._create()
        object._v_parent = self
        self.nrows = object.nrows
        self.nelemslice = object.nelemslice
        self._addAttrs(object, "IndexArray")
        self.indices = object
            
    def _open(self):
        """Get the metadata info for an array in file."""
        self._g_new(self._v_parent, self.name)
        self._v_objectID = self._g_openIndex()
        # Get the filters info
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
            print "arr-->", arr
            self.indices.append(arr.argsort())
            arr.sort()
            self.sorted.append(arr)
        else:
            self.sorted.append(numarray.sort(arr))
            self.indices.append(numarray.argsort(arr))
        
# This has been passed to Pyrex. However, with pyrex it has the same speed,
# but anyway
    def search(self, item, notequal):
        """Do a binary search in this index for an item"""
        #t1=time.time()
        ntotaliter = 0; tlen = 0
        self.starts = []; self.lengths = []
        bufsize = self.sorted._v_chunksize[1] # number of elements/chunksize
        self.nelemslice = self.sorted.nelemslice   # number of columns/slice
        self.sorted._initSortedSlice(bufsize)
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
# but anyway
    def getCoords(self, startCoords, maxCoords):
        """Get the coordinates of indices satisfiying the cuts"""
        #t1=time.time()
        len1 = 0; len2 = 0;
        stop = 0; relCoords = 0
        #print "startCoords, maxCoords-->", startCoords, maxCoords
        #print "lengths-->", self.lengths
        for irow in xrange(self.sorted.nrows):
            leni = self.lengths[irow]; len2 += leni
            if (leni > 0 and len1 <= startCoords < len2):
                startl = self.starts[irow] + (startCoords-len1)
                #print "leni, maxCoords, startl, len1-->",leni, maxCoords, startl, len1
                #if maxCoords >= leni - (startCoords-len1):
                if (startl + leni) < maxCoords:
                    # Values fit on buffer
                    stopl = startl + leni
                else:
                    stopl = startl + maxCoords
                    # Correction if stopl exceeds the limits
                    # Perhaps some cases are not figured out here
                    # I must do exhaustive a test suite!
                    if stopl > self.nelemslice:
                        stopl = self.nelemslice
                    # Stop after this iteration
                    stop = 1
                #print "startl, stopl-->", startl, stopl, stop
                self.indices._g_readIndex(irow, startl, stopl, relCoords)
                relCoords += stopl - startl
                if stop:
                    break
                maxCoords -= leni - (startCoords-len1)
                startCoords += leni - (startCoords-len1)
            len1 += leni
                
        selections = numarray.sort(self.indices.arrAbs[:relCoords])
        #selections = self.indices.arrAbs[:relCoords]
        #print "time doing revIndexing:", time.time()-t1
        return selections

    def _f_close(self):
        # flush the info for the indices
        self.sorted.flush()
        self.indices.flush()
        # Delete some references to the object tree
        del self.indices._v_parent
        del self.sorted._v_parent
        if hasattr(self.indices, "_v_file"):
            del self.indices._v_file
            del self.sorted._v_file
        del self.indices
        del self.sorted
        del self._v_parent
        # Close this group
        self._g_closeGroup()

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  nelemslice = %s
  chunksize = %s
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.nelemslice, self.chunksize, self.byteorder)
