# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import sys
import unittest
import os
import tempfile

from numarray import *
import numarray.strings as strings
from tables import *

try:
    import Numeric
    numeric = 1
except:
    numeric = 0

from test_all import verbose

def allequal(a,b):
    """Checks if two numarrays are equal"""

    if a.shape <> b.shape:
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if str(equal(a,b)) == '1':
            return 1
        else:
            return 0

    # Multidimensional case
    result = (a == b)
    for i in range(len(a.shape)):
        result = logical_and.reduce(result)

    return result

class BasicTestCase(unittest.TestCase):
    # Default values
    type = Int32
    shape = (2,0)
    chunksize = 5
    step = 1
    nappends = 10
    compress = 0
    complib = "zlib"  # Default compression library
    shuffle = 0

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()
        # Close the file (eventually destroy the extended type)
        self.fileh.close()
        
    def populateFile(self):
        group = self.rootgroup
        object = zeros(type=self.type, shape=self.shape)
        title = self.__class__.__name__
        earray = self.fileh.createArray(group, 'earray1', object, title,
                                        compress = self.compress,
                                        complib = self.complib,
                                        shuffle = self.shuffle,
                                        expectedrows = 1)

        # Fill it with rows
        self.rowshape = list(earray.shape)
        self.objsize = 1
        for i in self.rowshape:
            if i <> 0:
                self.objsize *= i
        self.objsize *= self.chunksize
        self.rowshape[earray.extdim] = self.chunksize
        object = arange(self.objsize, shape=self.rowshape, type=earray.type)
        if verbose:
            # print "-->", object
            print "-->", object.info()
        for i in range(self.nappends):
            #print "i-->", i
            earray.append(object*i)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    #----------------------------------------

    def test01_readEArray(self):
        """Checking enlargeable array read"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readEArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray._v_maxTuples = 3
        if verbose:
            print "Array descr:", repr(earray)
        # Build the array to do comparisons
        object_ = arange(self.objsize, shape=self.rowshape, type=earray.type)
        object_.swapaxes(earray.extdim, 0)
        # Read all the array
        for row in earray(step=self.step):
            chunk = (earray.nrow % self.chunksize)
            if chunk == 0:
                 object__ = object_ * (earray.nrow / self.chunksize)
            object = object__[chunk]
#             if verbose:
#                 print "number of row ==>", earray.nrow
#                 print "shape of row ==>", earray.shape
#                 print "shape should look as:", object.shape
#                 print "row in earray ==>", repr(row)
#                 print "Should look like ==>", repr(object)

            assert self.nappends*self.chunksize == earray.nrows
            assert allequal(row, object)
            assert len(row.shape) == len(self.shape) - 1

    def notest02_emptyEarray(self):
        """Checking creation of empty VL arrays"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_emptyEarray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "w")
        earray = self.fileh.createEarray(self.fileh.root, 'earray2',
                                           Int32Atom(),
                                           "ragged array if ints",
                                           compress = self.compress,
                                           complib = self.complib)
        # Try to read info from there:
        row = earray.read()
        # The result should be the empty list
        assert row == []

class BasicWriteTestCase(BasicTestCase):
    type = Int32
    shape = (2, 0)
    chunksize = 5
    nappends = 10

# This is good for testing
class MD3WriteTestCase(BasicTestCase):
    type = Int32
    shape = (2, 0, 3)
    chunksize = 4
    step = 2

class MD5WriteTestCase(BasicTestCase):
    type = Int32
    shape = (2, 0, 3, 4, 5)  # ok
    #shape = (2, 0, 3, 4, 5)  # ok
    #shape = (1, 1, 0, 1)  # Minimum shape that shows problems with HDF5 1.6.1
    #shape = (2, 3, 0, 4, 5)  # Floating point exception (HDF5 1.6.1)
    chunksize = 1
    #shape = (2, 3, 3, 0, 5, 6) # Segmentation fault (HDF5 1.6.1)
    shape = (2, 3, 3, 4, 5, 0) # Segmentation fault (HDF5 1.6.1)
    #chunksize = 500
    #nappends = 100

class MD10WriteTestCase(BasicTestCase):
    type = Int32
    shape = (1, 2, 3, 4, 5, 5, 4, 3, 2, 0)
    #shape.append(0)
    chunksize = 5
    nappends = 10

class ZlibComprTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"

class ZlibShuffleComprTestCase(BasicTestCase):
    shuffle = 1   # That should be enough to activate the compression
    complib = "zlib"

class LZOComprTestCase(BasicTestCase):
    compress = 1
    chunksize = 1
    nappends = 1
    complib = "lzo"

class UCLComprTestCase(BasicTestCase):
    compress = 1
    complib = "ucl"

class FloatTypeTestCase(BasicTestCase):
    type = Float64


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1

    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(MD3WriteTestCase))
    #theSuite.addTest(unittest.makeSuite(MD5WriteTestCase))
    #theSuite.addTest(unittest.makeSuite(MD10WriteTestCase))
    #theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
    #theSuite.addTest(unittest.makeSuite(ZlibShuffleComprTestCase))
    #theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
    #theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
    #theSuite.addTest(unittest.makeSuite(FloatTypeTestCase))

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD3WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD5WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD10WriteTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibShuffleComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
        theSuite.addTest(unittest.makeSuite(FloatTypeTestCase))
    

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
