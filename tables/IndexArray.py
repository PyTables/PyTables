########################################################################
#
#       License: BSD
#       Created: June 02, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /cvsroot/pytables/pytables/tables/IndexArray.py,v $
#       $Id$
#
########################################################################

"""Here is defined the IndexArray class.

See IndexArray class docstring for more info.

Classes:

    IndexArray

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.12 $"
# default version for IndexARRAY objects
obversion = "1.0"    # initial version

import warnings, sys
from EArray import EArray
from VLArray import Atom, StringAtom
import hdf5Extension
import numarray
import numarray.strings as strings
import numarray.records as records
from bisect import bisect_left, bisect_right
from time import time

def calcChunksize(expectedrows, testmode=0):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimize the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized doing more experiments.

    """

    expKrows = expectedrows / 1000000.  # Multiples of one million
    if testmode:
        if expKrows < 0.0001: # expected rows < 1 hundred
            nelemslice = 10  # > 1/100th
            chunksize = 5
        elif expKrows < 0.001: # expected rows < 1 thousand
            nelemslice = 100  # > 1/10th
            chunksize = 50
        elif expKrows <= 0.01: # expected rows < 10 thousand
            nelemslice = 1000  # > 1/100th
            chunksize = 500
        else:
            raise ValueError, \
                  "expected rows cannot be larger than 10000 in test mode"
        return (nelemslice, chunksize)

    #print "expKrows:", expKrows
    if expKrows < 0.1: # expected rows < 100 thousand
        chunksize = 1000            # best value
        nelemslice = 2*chunksize    #  "     "
        #chunksize = 500             # experimental
        #nelemslice = 2*chunksize    #      "       
    elif expKrows < 1: # expected rows < 1 milion
        chunksize = 1000
        #nelemslice = 5*chunksize 
        nelemslice = 2*chunksize 
        #chunksize = 2000           # experimental
        #nelemslice = 2*chunksize   #    "
    elif expKrows < 10:  # expected rows < 10 milion
        chunksize = 2500
        #nelemslice = 50*chunksize  
        #nelemslice = 25*chunksize  
        nelemslice = 15*chunksize     # best speed/compression ratio
        #chunksize = 5000             # experiment
        #nelemslice = 5*chunksize     #     "
    elif expKrows < 100: # expected rows < 100 milions
        chunksize = 5000              # optimum for NoNoise and Noise
        #nelemslice = 100*chunksize   # very aggressive
        #nelemslice = 50*chunksize    # moderately aggressive
        #nelemslice = 25*chunksize    # moderately conservative
        nelemslice = 15*chunksize     # moderately conservative (II)
        #nelemslice = 10*chunksize    # very conservative
        #chunksize = 7500             # experimental (I)
        #nelemslice = 15*chunksize    # experimental (I)
        #chunksize = 10000            # experimental (II)
        #nelemslice = 10*chunksize    # experimental (II)
    elif expKrows < 1000: # expected rows < 1000 millions
        chunksize = 5000
        #nelemslice = 200*chunksize # very aggressive
        #nelemslice = 100*chunksize # moderately aggressive
        #nelemslice =  50*chunksize # moderately conservative
        nelemslice =  35*chunksize  # moderately conservative (II)
        #nelemslice =  25*chunksize # moderately conservative (III)
    elif expKrows < 10*1000: # expected rows < 10 (american) billions
        #chunksize = 10000          # very aggressive
        #nelemslice = 50*chunksize  #   "       "
        #chunksize = 7500           # moderately aggressive
        #nelemslice = 50*chunksize  #    "           "
        #chunksize = 5000           # moderately aggressive (II)
        #nelemslice = 100*chunksize #    "           "
        chunksize = 7500            # moderately conservative
        nelemslice = 35*chunksize   #    "           "
        #chunksize = 5000           # moderately conservative (II)
        #nelemslice = 50*chunksize  #    "           "
    elif expKrows < 100*1000: # expected rows < 100 (american) billions
        #chunksize = 10000          # very aggressive
        #nelemslice = 50*chunksize  #   "       "
        #chunksize = 7500           # moderately aggressive
        #nelemslice = 100*chunksize #    "           "
        chunksize = 7500            # moderately conservative
        nelemslice = 50*chunksize   #    "           "
        #chunksize = 5000           # moderately conservative (II)
        #nelemslice = 100*chunksize #    "           "
    else:  # expected rows >= 1 (american) trillion (perhaps in year 2010
           # this will be useful, who knows...)
        #chunksize = 10000          # very aggressive
        #nelemslice = 100*chunksize #   "       "
        chunksize = 10000           # moderately aggressive
        nelemslice = 50*chunksize   #   "            "
        #chunksize = 7500           # moderately conservative
        #nelemslice = 100*chunksize #    "           "

    #print "nelemslice, chunksize:", (nelemslice, chunksize)
    return (nelemslice, chunksize)

def calcChunksize_orig(expectedrows, testmode=0):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimize the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized doing more experiments.

    """

    expKrows = expectedrows / 1000000.  # Multiples of one million

    if testmode:
        if expKrows < 0.0001: # expected rows < 1 hundred
            nelemslice = 10  # > 1/100th
            chunksize = 5
        elif expKrows < 0.001: # expected rows < 1 thousand
            nelemslice = 100  # > 1/10th
            chunksize = 50
        elif expKrows <= 0.01: # expected rows < 10 thousand
            nelemslice = 1000  # > 1/100th
            chunksize = 500
        else:
            raise ValueError, \
                  "expected rows cannot be larger than 10000 in test mode"
        return (nelemslice, chunksize)

    # expKrows < 0.01 is to few for indexing to represent a significant gain
    # (that has been checked experimentally)
#     if expKrows < 0.01: # expected rows < 10 thousand
#         nelemslice = 1000  # > 1/100th
#         chunksize = 1000
    if expKrows < 0.1: # expected rows < 100 thousand
        chunksize = 1000
        nelemslice = 5*chunksize  # (best experimental)
#         nelemslice = 5*1024 
#         chunksize = 1024
    elif expKrows < 1: # expected rows < 1 milion
        chunksize = 2000   # (best experimental)
        nelemslice = 10*chunksize  # (best experimental)
#         nelemslice = 5*1024 
#         chunksize = 1024
#         chunksize = 2048  # (best experimental)
#         nelemslice = 10*chunksize   # (best experimental)
       #chunksize = 2048
       #nelemslice = 10*chunksize 
    elif expKrows < 10:  # expected rows < 10 milion
        #nelemslice = 500000  # > 1/20th (best for best case)
        #chunksize = 5000  # Experimental (best for best case)
        #chunksize = 2000  # best for worst case (experimental)
        chunksize = 2000  # best for worst case (experimental)
        nelemslice = 50*chunksize # Best for worst case (experimental)
        #chunksize = 4096  # (best experimental)
        #nelemslice = 10*chunksize   # (best experimental)
#         nelemslice = 20*4096 
#         chunksize = 4096
    elif expKrows < 100: # expected rows < 100 milions
        chunksize = 5000
        nelemslice = 30*chunksize
        #nelemslice = 20*4096 
        #chunksize = 4096
    elif expKrows < 1000: # expected rows < 1000 millions
        chunksize = 5000
        nelemslice = 40*chunksize # Experimental (best)
        #chunksize = 10000   # Experimental (best)
    else:  # expected rows >= 1 billion
        #nelemslice = 1000000 # 1/1000  # Better for small machines
        #nelemslice = 2000000 # 2/1000  # Better for big machines
        #chunksize = 5000
        chunksize = 10000  # Experimental
        nelemslice = 50*chunksize

    #print "nelemslice, chunksize:", (nelemslice, chunksize)
    return (nelemslice, chunksize)

class IndexArray(EArray, hdf5Extension.IndexArray, object):
    """Represent the index (sorted or reverse index) dataset in HDF5 file.

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

      Common to all EArray's:
        read(start, stop, step)
        iterrows(start, stop, step)
        append(object)

        
    Instance variables:

      Common to all EArray's:

        type -- The type class for the array.
        itemsize -- The size of the atomic items. Specially useful for
            CharArrays.
        flavor -- The flavor of this object.
        nrow -- On iterators, this is the index of the row currently
            dealed with.

      Specific of IndexArray:
        extdim -- The enlargeable dimension (always the first, or 0).
        nrows -- The number of slices in index.
        nelemslice -- The number of elements per slice.
        chunksize -- The HDF5 chunksize for the slice dimension (the 1).
            

    """
    
    def __init__(self, parent = None, atom = None, title = "",
                 filters = None, expectedrows = 1000000,
                 testmode=0):
        """Create an IndexArray instance.

        Keyword arguments:

        parent -- The Index class from which this object will hang off

        atom -- An Atom object representing the shape, type and flavor
            of the atomic objects to be saved. Only scalar atoms are
            supported.
        
        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- Represents an user estimate about the number
            of elements to index. If not provided, the default
            value is 1000000 slices.

        """
        self._v_parent = parent
        self._v_new_title = title
        self._v_new_filters = filters
        self._v_expectedrows = expectedrows
        self.testmode = testmode
        self.flavor = "NumArray"  # Needed by Array methods
        # Check if we have to create a new object or read their contents
        # from disk
        if atom is not None:
            self._v_new = 1
            self.atom = atom
        else:
            self._v_new = 0
        # initialize some index buffers
        self._startl = numarray.array(None, shape=(2,), type=numarray.Int64)
        self._stopl = numarray.array(None, shape=(2,), type=numarray.Int64)
        self._stepl = numarray.array([1,1], shape=(2,), type=numarray.Int64)
            
    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        global obversion

        assert isinstance(self.atom, Atom), "The object passed to the IndexArray constructor must be a descendent of the Atom class."
        assert self.atom.shape == 1, "Only scalar columns can be indexed."
        # Version, type, shape, flavor, byteorder
        self._v_version = obversion
        self.type = self.atom.type
        if self.type == "CharType" or isinstance(self.type, records.Char):
            self.byteorder = "non-relevant"
        else:
            # Only support for creating objects in system byteorder
            self.byteorder  = sys.byteorder
        # Compute the optimal chunksize
        (self.nelemslice, self.chunksize) = \
                          calcChunksize(self._v_expectedrows,
                                        testmode=self.testmode)
        # The next is needed by hdf5Extension.Array._createEArray
        self._v_chunksize = (1, self.chunksize)
        self.nrows = 0   # No rows initially
        self.itemsize = self.atom.itemsize
        self.rowsize = self.atom.atomsize() * self.nelemslice
        self.shape = (0, self.nelemslice)
        
        # extdim computation
        self.extdim = 0
        # Compute the optimal maxTuples
        # Ten chunks for each buffer would be enough for IndexArray objects
        # This is really necessary??
        self._v_maxTuples = 10  
        # Create the IndexArray
        self._createEArray("INDEXARRAY", self._v_new_title)

        # Create a buffer for bounds array
        nbounds = (self.nelemslice // self.chunksize) - 1
        if str(self.type) == "CharType":        
            self._bounds = strings.array(None, itemsize=self.itemsize,
                                         shape=(nbounds,))
        else:
            self._bounds = numarray.array(None, shape=(nbounds,),
                                          type=self.type)            

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize,
         self.byteorder, chunksizes) = self._openArray()
        self.chunksize = chunksizes[1]  # Get the second dim
        # Post-condition
        assert self.extdim == 0, "extdim != 0: this should never happen!"
        self.nelemslice = self.shape[1] 
        # Create the atom instance. Not for strings yet!
        if str(self.type) == "CharType":
            self.atom = StringAtom(shape=1, length=self.itemsize)
        else:
            self.atom = Atom(dtype=self.type, shape=1)
        # Compute the rowsize for each element
        self.rowsize = self.atom.atomsize() * self.nelemslice
        # nrows in this instance
        self.nrows = self.shape[0]
        # Compute the optimal maxTuples
        # Ten chunks for each buffer would be enough for IndexArray objects
        # This is really necessary??
        self._v_maxTuples = 10  

        # Create a buffer for bounds array
        nbounds = (self.nelemslice // self.chunksize) - 1
        if str(self.type) == "CharType":        
            self._bounds = strings.array(None, itemsize=self.itemsize,
                                         shape=(nbounds,))
        else:
            self._bounds = numarray.array(None, shape=(nbounds,),
                                          type=self.type)            

    def append(self, arr):
        """Append the object to this (enlargeable) object"""
        arr.shape = (1, arr.shape[0])
        self._append(arr)

    # This version of searchBin uses both rangeValues (1st level) and
    # bounds (2nd level) caches. This is more than 40% faster than the
    # version that only uses the 1st cache.    
    def _searchBin(self, nrow, item):
        item1, item2 = item
        result1 = -1; result2 = -1
        hi = self.nelemslice
        rangeValues = self._v_parent.rvcache
        #t1=time()
        # First, look at the beginning of the slice
        #begin, end = rangeValues[nrow]  # this is slower 
        begin = rangeValues[nrow,0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
        if item2 < begin:
            result2 = 0
        if result1 >=0 and result2 >= 0:
            #print "done 1-->", time()-t1
            return (result1, result2)
        # Then, look for items at the end of the sorted slice
        end = rangeValues[nrow,1]
        if result1 < 0:
            if item1 > end:
                result1 = hi
        if result2 < 0:
            if item2 >= end:
                result2 = hi
        if result1 >= 0 and result2 >= 0:
            #print "done 2-->", time()-t1
            return (result1, result2)
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        chunksize = self.chunksize # Number of elements/chunksize
        nbounds = (self.nelemslice // self.chunksize) - 1
        bounds = self._bounds
        pbounds = self._v_parent.bounds
        # Read the bounds array
        #bounds = self._v_parent.bounds[nrow]
        # This optimization adds little speed-up (5%), but...
        self._startl[0] = nrow; self._startl[1] = 0
        self._stopl[0] = nrow+1; self._stopl[1] = nbounds
        pbounds._g_readSlice(self._startl, self._stopl, self._stepl, bounds)
        if result1 < 0:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect_left(bounds, item1)
            chunk = self._readSortedSlice(nrow, chunksize*nchunk,
                                          chunksize*(nchunk+1))
            result1 = self._bisect_left(chunk, item1, chunksize)
            result1 += chunksize*nchunk
        # Lookup in the middle of slice for item2
        if result2 < 0:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect_right(bounds, item2)
            chunk = self._readSortedSlice(nrow, chunksize*nchunk,
                                          chunksize*(nchunk+1))
            result2 = self._bisect_right(chunk, item2, chunksize)
            result2 += chunksize*nchunk
        return (result1, result2)

    def _searchBin_orig(self, nrow, item):
        item1, item2 = item
        item1done = 0; item2done = 0
        hi = self.nelemslice
        rangeValues = self._v_parent.rvcache
        t1=time()
        # First, look at the beginning of the slice
        #begin, end = rangeValues[nrow]  # this is slower 
        begin = rangeValues[nrow,0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
            item1done = 1
        if item2 < begin:
            result2 = 0
            item2done = 1
        if item1done and item2done:
            #print "done 1"
            #print "done 1-->", time()-t1
            return (result1, result2)
        # Then, look for items at the end of the sorted slice
        end = rangeValues[nrow,1]
        if not item1done:
            if item1 > end:
                result1 = hi
                item1done = 1
        if not item2done:
            if item2 >= end:
                result2 = hi
                item2done = 1
        if item1done and item2done:
            #print "done 2"
            #print "done 2-->", time()-t1
            return (result1, result2)
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        chunksize = self.chunksize # Number of elements/chunksize
        nbounds = (self.nelemslice // self.chunksize) - 1
        bounds = self._bounds
        pbounds = self._v_parent.bounds
        # Read the bounds array
        #bounds = self._v_parent.bounds[nrow]
        # This optimization adds little speed-up (5%), but...
        self._startl[0] = nrow; self._startl[1] = 0
        self._stopl[0] = nrow+1; self._stopl[1] = nbounds
        pbounds._g_readSlice(self._startl, self._stopl, self._stepl, bounds)
        if not item1done:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect_left(bounds, item1)
            chunk = self._readSortedSlice(nrow, chunksize*nchunk,
                                          chunksize*(nchunk+1))
            result1 = self._bisect_left(chunk, item1, chunksize)
            result1 += chunksize*nchunk
        # Lookup in the middle of slice for item2
        if not item2done:
            # Search the appropriate chunk in bounds cache
            nchunk = bisect_right(bounds, item2)
            chunk = self._readSortedSlice(nrow, chunksize*nchunk,
                                          chunksize*(nchunk+1))
            result2 = self._bisect_right(chunk, item2, chunksize)
            result2 += chunksize*nchunk
        return (result1, result2)

    # This version of searchBin only uses the rangeValues (1st cache)
    def _g_searchBin(self, nrow, item, rangeValues):
        #item1, item2 = self.item
        #rangeValues = self.rv
        item1, item2 = item
        nelemslice = self.shape[1]
        hi = nelemslice   
        item1done = 0; item2done = 0
        chunksize = self.chunksize # Number of elements/chunksize # change here
        #niter = 1
        niter = 0

        # First, look at the beginning of the slice
        # (that could save lots of time)
        #begin = self[nrow,0]
        #begin = self._v_parent.rangeValues[nrow,0]
        begin = rangeValues[nrow,0]
        # Look for items at the beginning of sorted slices
        if item1 <= begin:
            result1 = 0
            item1done = 1
        if item2 < begin:
            result2 = 0
            item2done = 1
        if item1done and item2done:
            #print "done 1"
            return (result1, result2, niter)

        #niter += 1
        # Then, look for items at the end of the sorted slice
        end = rangeValues[nrow,1]
        if not item1done:
            if item1 > end:
                result1 = hi
                item1done = 1
        if not item2done:
            if item2 >= end:
                result2 = hi
                item2done = 1
        if item1done and item2done:
            #print "done 2"
            return (result1, result2, niter)
    
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        if not item1done:
            lo = 0
            hi = nelemslice
            beginning = 1
            result1 = 1  # a number different from 0
            while beginning and result1 != 0:
                (result1, beginning, iter) = \
                          self._interSearch_left(nrow, chunksize,
                                                 item1, lo, hi)
                tmpresult1 = result1
                niter = niter + iter
                if result1 == hi:  # The item is completely at right
                    break
                else:
                    hi = result1        # one chunk to the left
                    lo = hi - chunksize  
            result1 = tmpresult1
        # Lookup in the middle of slice for item2
        if not item2done:
            lo = 0
            hi = nelemslice
            ending = 1
            result2 = 1  # a number different from 0
            while ending and result2 != nelemslice:
                (result2, ending, iter) = \
                          self._interSearch_right(nrow, chunksize,
                                                  item2, lo, hi)
                tmpresult2 = result2
                niter = niter + iter
                if result2 == lo:  # The item is completely at left
                    break
                else:
                    hi = result2 + chunksize      # one chunk to the right
                    lo = result2
            result2 = tmpresult2
            niter = niter + iter
        return (result1, result2, niter)

    # This version of searchBin does not use caches (1st or 2nd) at all
    # This is coded in pyrex as well, but the improvement in speed is very
    # little. So, it's better to let _searchBin live here.
    def _searchBin1_0(self, nrow, item):
        nelemslice = self.shape[1]
        hi = nelemslice   
        item1, item2 = item
        item1done = 0; item2done = 0
        chunksize = self.chunksize # Number of elements/chunksize # change here
        niter = 1

        # First, look at the beginning of the slice (that could save lots of time)
        buffer = self._readSortedSlice(nrow, 0, chunksize)
        #buffer = xrange(0, chunksize)  # test  # 0.02 over 0.5 seg
        # Look for items at the beginning of sorted slices
        result1 = self._bisect_left(buffer, item1, chunksize)
        if 0 <= result1 < chunksize:
            item1done = 1
        result2 = self._bisect_right(buffer, item2, chunksize)
        if 0 <= result2 < chunksize:
            item2done = 1
        if item1done and item2done:
            #print "done 1"
            return (result1, result2)

        # Then, look for items at the end of the sorted slice
        buffer = self._readSortedSlice(nrow, hi-chunksize, hi)
        #buffer = xrange(hi-chunksize, hi)  # test
        niter += 1
        if not item1done:
            result1 = self._bisect_left(buffer, item1, chunksize)
            if 0 < result1 <= chunksize:
                item1done = 1
                result1 = hi - chunksize + result1
        if not item2done:
            result2 = self._bisect_right(buffer, item2, chunksize)
            if 0 < result2 <= chunksize:
                item2done = 1
                result2 = hi - chunksize + result2
        if item1done and item2done:
            #print "done 2"
            return (result1, result2)
    
        # Finally, do a lookup for item1 and item2 if they were not found
        # Lookup in the middle of slice for item1
        if not item1done:
            lo = 0
            hi = nelemslice
            beginning = 1
            result1 = 1  # a number different from 0
            while beginning and result1 != 0:
                (result1, beginning, iter) = \
                          self._interSearch_left(nrow, chunksize,
                                                 item1, lo, hi)
                tmpresult1 = result1
                niter = niter + iter
                if result1 == hi:  # The item is completely at right
                    break
                else:
                    hi = result1        # one chunk to the left
                    lo = hi - chunksize  
            result1 = tmpresult1
        # Lookup in the middle of slice for item2
        if not item2done:
            lo = 0
            hi = nelemslice
            ending = 1
            result2 = 1  # a number different from 0
            while ending and result2 != nelemslice:
                (result2, ending, iter) = \
                          self._interSearch_right(nrow, chunksize,
                                                  item2, lo, hi)
                tmpresult2 = result2
                niter = niter + iter
                if result2 == lo:  # The item is completely at left
                    break
                else:
                    hi = result2 + chunksize      # one chunk to the right
                    lo = result2
            result2 = tmpresult2
            niter = niter + iter
        return (result1, result2)

    def _close(self):
        """Close this object and exit"""
        # First, flush the buffers:
        self.flush()
        # Delete back references
        del self._v_parent
        del self._v_file
        del self.type
        del self.atom
        del self.filters
        self.__dict__.clear()

    def __str__(self):
        "A compact representation of this class"
        return "IndexArray(path=%s)" % \
               (self._v_parent._g_join(self.name))

    def __repr__(self):
        """A verbose representation of this class"""

        return """%s
  type = %r
  shape = %s
  itemsize = %s
  nrows = %s
  nelemslice = %s
  chunksize = %s
  byteorder = %r""" % (self, self.type, self.shape, self.itemsize, self.nrows,
                       self.nelemslice, self.chunksize, self.byteorder)
