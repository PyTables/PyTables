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

    if not hasattr(b, "shape"):
        return a == b

    if a.shape <> b.shape:
        return 0

    if a.type() <> b.type():
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if str(equal(a,b)) == '1':
            return 1
        else:
            return 0

    # Null arrays
    if len(a._data) == 0:  # len(a) is not correct for generic shapes
        if len(b._data) == 0:
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
    start = 0
    stop = 10
    step = 1
    chunksize = 5
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
        self.extdim = earray.extdim
        self.objsize *= self.chunksize
        self.rowshape[earray.extdim] = self.chunksize
        object = arange(self.objsize, shape=self.rowshape, type=earray.type)
        if verbose:
            # print "-->", object
            print "Object to append -->", object.info()
        for i in range(self.nappends):
            #print "i-->", i
            earray.append(object*i)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    #----------------------------------------

    def test01_iterArray(self):
        """Checking enlargeable array iterator"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_iterArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray._v_maxTuples = 3
        if verbose:
            print "Array descr:", repr(earray)
            print "shape of read array ==>", earray.shape
        # Build the array to do comparisons
        object_ = arange(self.objsize, shape=self.rowshape, type=earray.type)
        object_.swapaxes(earray.extdim, 0)
        # Read all the array
        for row in earray:
            chunk = (earray.nrow % self.chunksize)
            if chunk == 0:
                 object__ = object_ * (earray.nrow / self.chunksize)
            object = object__[chunk]
            # The next adds much more verbosity
            if verbose and 0:
                print "number of row ==>", earray.nrow
                if hasattr(object, "shape"):
                    print "shape should look as:", object.shape
                print "row in earray ==>", repr(row)
                print "Should look like ==>", repr(object)

            assert self.nappends*self.chunksize == earray.nrows
            assert allequal(row, object)
            if hasattr(row, "shape"):
                assert len(row.shape) == len(self.shape) - 1
            else:
                # Scalar case
                assert len(self.shape) == 1

    def test02_sssArray(self):
        """Checking enlargeable array iterator with (start, stop, step)"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_sssArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray._v_maxTuples = 3
        if verbose:
            print "Array descr:", repr(earray)
            print "shape of read array ==>", earray.shape
        # Build the array to do comparisons
        object_ = arange(self.objsize, shape=self.rowshape, type=earray.type)
        object_.swapaxes(earray.extdim, 0)
        # Read all the array
        for row in earray(start=self.start, stop=self.stop, step=self.step):
            chunk = (earray.nrow % self.chunksize)
            if (chunk - self.start) == 0:
                 object__ = object_ * (earray.nrow / self.chunksize)
            object = object__[chunk]
            # The next adds much more verbosity
            if verbose and 0:
                print "number of row ==>", earray.nrow
                if hasattr(object, "shape"):
                    print "shape should look as:", object.shape
                print "row in earray ==>", repr(row)
                print "Should look like ==>", repr(object)

            assert self.nappends*self.chunksize == earray.nrows
            assert allequal(row, object)
            if hasattr(row, "shape"):
                assert len(row.shape) == len(self.shape) - 1
            else:
                # Scalar case
                assert len(self.shape) == 1

    def test03_readArray(self):
        """Checking read() of enlargeable arrays"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_readArray..." % self.__class__.__name__
            
        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray._v_maxTuples = 3
        if verbose:
            print "Array descr:", repr(earray)
            print "shape of read array ==>", earray.shape

        # Build the array to do comparisons
        object_ = arange(self.objsize, shape=self.rowshape, type=earray.type)
        object_.swapaxes(earray.extdim, 0)
        rowshape = self.rowshape
        rowshape[self.extdim] *= self.nappends
        object__ = array(None, shape = rowshape, type=self.type)
        object__.swapaxes(0, self.extdim)
        for i in range(self.nappends):
            j = i * self.chunksize
            object__[j:j+self.chunksize] = object_ * i
        object__.swapaxes(0, self.extdim)
        stop = self.stop
        if self.nappends:
            # Protection against number of elements less than existing
            if rowshape[self.extdim] < self.stop or self.stop == 0:
                # self.stop == 0 means last row
                stop = rowshape[self.extdim]
            object = take(object__, range(self.start,stop,self.step),
                          axis = self.extdim)
        else:
            object = array(None, shape = self.shape, type=self.type)
                
        # Read all the array
        try:
            row = earray.read(self.start,self.stop,self.step)
        except IndexError:
            row = array(None, shape = self.shape, type=self.type)

        # The next adds much more verbosity
        if verbose and 1:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(row)
            print "Should look like ==>", repr(object)

        assert self.nappends*self.chunksize == earray.nrows
        assert allequal(row, object)
        if hasattr(row, "shape"):
            assert len(row.shape) == len(self.shape)
        else:
            # Scalar case
            assert len(self.shape) == 1

    def test04_getitemArray(self):
        """Checking enlargeable array __getitem__ special method"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_getitemArray..." % self.__class__.__name__
            
        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray._v_maxTuples = 3
        if verbose:
            print "Array descr:", repr(earray)
            print "shape of read array ==>", earray.shape

        # Build the array to do comparisons
        object_ = arange(self.objsize, shape=self.rowshape, type=earray.type)
        object_.swapaxes(earray.extdim, 0)
        rowshape = self.rowshape
        rowshape[self.extdim] *= self.nappends
        object__ = array(None, shape = rowshape, type=self.type)
        object__.swapaxes(0, self.extdim)
        for i in range(self.nappends):
            j = i * self.chunksize
            object__[j:j+self.chunksize] = object_ * i
        object__.swapaxes(0, self.extdim)
        stop = self.stop
        if self.nappends:
            # Protection against number of elements less than existing
            if rowshape[self.extdim] < self.stop or self.stop == 0:
                # self.stop == 0 means last row
                stop = rowshape[self.extdim]
            object = take(object__, range(self.start,stop,self.step),
                          axis = self.extdim)
        else:
            object = array(None, shape = self.shape, type=self.type)
        if (len(range(self.start, stop, self.step)) == 1 and
            self.extdim > 0):
            object.swapaxes(self.extdim, 0)
            object = object[0]
            correction = 1
        else:
            correction = 0
                
        # Read all the array
        try:
            row = earray[self.start:self.stop:self.step]
        except IndexError:
            row = array(None, shape = self.shape, type=self.type)

        # The next adds much more verbosity
        if verbose and 1:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(row)
            print "Should look like ==>", repr(object)

        assert self.nappends*self.chunksize == earray.nrows
        assert allequal(row, object)
        if hasattr(row, "shape"):
            assert len(row.shape) == len(self.shape) - correction
        else:
            # Scalar case
            assert len(self.shape) == 1


class BasicWriteTestCase(BasicTestCase):
    type = Int32
    shape = (0,)
    chunksize = 5
    nappends = 10
    step = 1

class EmptyArrayTestCase(BasicTestCase):
    type = Int32
    shape = (2, 0)
    chunksize = 5
    nappends = 0
    start = 0
    step = 1

class MD3WriteTestCase(BasicTestCase):
    type = Int32
    shape = (2, 0, 3)
    chunksize = 4
    step = 2

class MD5WriteTestCase(BasicTestCase):
    type = Int32
    shape = (2, 0, 3, 4, 5)  # ok
    #shape = (1, 1, 0, 1)  # Minimum shape that shows problems with HDF5 1.6.1
    #shape = (2, 3, 0, 4, 5)  # Floating point exception (HDF5 1.6.1)
    #shape = (2, 3, 3, 0, 5, 6) # Segmentation fault (HDF5 1.6.1)
    chunksize = 1
    nappends = 1
    start = 1
    stop = 10
    step = 10

class MD10WriteTestCase(BasicTestCase):
    type = Int32
    shape = (1, 2, 3, 4, 5, 5, 4, 3, 2, 0)
    chunksize = 5
    nappends = 10
    start = -1
    stop = -1
    step = 10

class ZlibComprTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    start = 3
    stop = 0   # means last row
    step = 10

class ZlibShuffleTestCase(BasicTestCase):
    shuffle = 1   # That should be enough to activate the compression
    complib = "zlib"
    # case start < stop , i.e. no rows read
    start = 3
    stop = 1
    step = 10

class LZOComprTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    chunksize = 10
    nappends = 100
    start = 3
    stop = 10
    step = 3

class LZOShuffleTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    complib = "lzo"
    chunksize = 100
    nappends = 10
    start = 3
    stop = 10
    step = 7

class UCLComprTestCase(BasicTestCase):
    compress = 1
    complib = "ucl"
    chunksize = 100
    nappends = 10
    start = 3
    stop = 10
    step = 8

class UCLShuffleTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    complib = "ucl"
    chunksize = 100
    nappends = 10
    start = 3
    stop = 10
    step = 6

class FloatTypeTestCase(BasicTestCase):
    type = Float64
    shape = (2,0)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20

# Provar a afegir tests per a scalars i chararrays

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1

    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(EmptyArrayTestCase))
    #theSuite.addTest(unittest.makeSuite(MD3WriteTestCase))
    #theSuite.addTest(unittest.makeSuite(MD5WriteTestCase))
    #theSuite.addTest(unittest.makeSuite(MD10WriteTestCase))
    #theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
    #theSuite.addTest(unittest.makeSuite(ZlibShuffleTestCase))
    #theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
    #theSuite.addTest(unittest.makeSuite(LZOShuffleTestCase))
    #theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
    #theSuite.addTest(unittest.makeSuite(UCLShuffleTestCase))
    #theSuite.addTest(unittest.makeSuite(FloatTypeTestCase))

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(EmptyArrayTestCase))
        theSuite.addTest(unittest.makeSuite(MD3WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD5WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD10WriteTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
        theSuite.addTest(unittest.makeSuite(UCLShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(FloatTypeTestCase))
    

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
