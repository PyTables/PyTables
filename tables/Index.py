########################################################################
#
#       License: BSD
#       Created: June 08, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Index.py,v $
#       $Id: Index.py,v 1.1 2004/06/18 12:31:08 falted Exp $
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

__version__ = "$Revision: 1.1 $"
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

class Index(hdf5Extension.Group, object):
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
        self._v_parent = where.table._v_parent  # Parent o table in object tree
        # Check whether we have to create a new object or read their contents
        # from disk
        if atom is not None:
            self._v_new = 1
            self.atom = atom
            self._create()
        else:
            self._v_new = 0
            self._open()
        # useful variables for getCoords
        self.startl = numarray.array(None,type="Int64",shape=(2,))
        self.stopl = numarray.array(None,type="Int64",shape=(2,))
        self.stepl = numarray.array((1,1),type="Int64",shape=(2,))

    def _newBuffer(self, maxCoords):
        """Create arrays for keeping the coordinates during selections """
        self.arrRel = numarray.zeros(type="Int32",shape=(maxCoords,))
        self.arrAbs = numarray.zeros(type="Int64",shape=(maxCoords,))

    def _delBuffer(self):
        """Destroy buffer arrays"""
        del self.arrRel
        del self.arrAbs

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
        self.nelem = 0
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
        self.indices = object

    def append(self, arr):
        """Append the object to this (enlargeable) object"""

        # Save the sorted array
        self.sorted.append(numarray.sort(arr))
        self.indices.append(numarray.argsort(arr))

    def search(self, item, notequal=0):
        """Do a binary search in this index for an item"""
        t1=time.time()
        ntotaliter = 0  # for counting the number of reads on each
        tlen = 0
        self.lengths = []
        self.starts = []; self.stops = [];
        bufsize = self.sorted._v_chunksize[1] # number of elements/chunksize
        self.nelemslice = self.sorted.nelemslice   # number of columns/slice
        self.sorted._initSortedSlice(bufsize)
        # Do the lookup for values fullfilling the conditions
        for i in xrange(self.sorted.nrows):
            (start, stop, niter) = self.sorted._searchBin(i, item)
            #print "selected values-->", self.sorted[i][start:stop]
            self.starts.append(start); self.stops.append(stop);
            tlen += stop - start
            self.lengths.append(stop - start)
            ntotaliter += niter
        self.sorted._destroySortedSlice()
        #print "time reading indices:", time.time()-t1
        self.exception = -1
        # There is a lot to be done for != operator yet 
        if tlen > 0 and notequal:
            cont = 1
            for i in xrange(self.sorted.nrows+1):
                print "lengths[%s]-->%s" % (i, self.lengths[i])
                if self.lengths[i] <> 0:
                    if cont:
                        self.starts.insert(i+1, self.stops[i])
                        self.stops[i] = self.starts[i]
                        self.starts[i] = 0
                        self.stops.insert(i+1, self.nelemslice)
                        self.lengths[i] = self.stops[i]-self.starts[i]
                        self.lengths.insert(i+1, self.stops[i+1]-self.starts[i+1])
                        cont = 0 # skip the index that was added
                        self.exception = i+1
                else:
                    self.starts[i] = 0
                    self.stops[i] = self.nelemslice
                    self.lengths[i] = self.nelemslice
                    cont = 1
                print "(2)lengths[%s]-->%s" % (i, self.lengths[i])
                print "(2)starts[%s]-->%s" % (i, self.starts[i])
                print "(2)stops[%s]-->%s" % (i, self.stops[i])
            tlen = self.nelemslice*self.sorted.nrows-tlen
            print "tlen-->", tlen
            print "starts, stops, lengths-->", self.starts, self.stops, self.lengths
        return tlen

    def getCoords(self, startCoords, maxCoords):
        """Get the coordinates of indices satisfiying the cuts"""
        t1=time.time()
        len1 = 0; len2 = self.lengths[0];
        stop = 0; relCoords = 0; lastvalidentry = -1;
        #print "(Entrant) startCoords, maxCoords -->", startCoords, maxCoords
        #for i in xrange(len(self.starts)):
        for i in xrange(self.sorted.nrows):
            #print "len1, len2-->", len1, len2
            if (self.lengths[i] > 0 and len1 <= startCoords < len2):
                self.startl[0] = i; self.stopl[0] = i+1;
                self.startl[1] = self.starts[i] + (startCoords-len1)
                #print "maxCoords, lengths, len-->", maxCoords, self.lengths[i], len1
                if maxCoords >= self.lengths[i] - (startCoords-len1):
                    # Values fit on buffer
                    self.stopl[1] = self.stops[i]
                else:
                    # Stop after this iteration
                    self.stopl[1] = self.startl[1]+maxCoords
                    stop = 1
                #print "startl, stopl -->", self.startl, self.stopl
                self.indices._g_readIndex(i, self.nelemslice,
                                          self.stopl[1]-self.startl[1],
                                          self.startl, self.stopl, self.stepl,
                                          self.arrRel[relCoords:],
                                          self.arrAbs[relCoords:])
                lastvalidentry = relCoords+(self.stopl[1]-self.startl[1])
                if stop:
                    break
                maxCoords -= self.lengths[i] - (startCoords-len1)
                startCoords += self.lengths[i] - (startCoords-len1)
                relCoords += self.stopl[1]-self.startl[1]
            len1 += self.lengths[i]
            if i < self.sorted.nrows-1:
                len2 += self.lengths[i+1]
                
        #print "time doing revIndexing:", time.time()-t1
        selections = numarray.sort(self.arrAbs[:lastvalidentry])
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
