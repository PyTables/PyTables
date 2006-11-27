# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import sys
import unittest
import os
import tempfile

import numpy

try:
    import Numeric
    numeric = 1
except:
    numeric = 0

try:
    import numarray
    from numarray import strings
    numarray_imported = 1
except:
    numarray_imported = 0

from tables import *
from tables.utils import convertNPToNumArray, convertNPToNumeric
from tables.tests.common import verbose, typecode, allequal, cleanup, heavy

# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup


class BasicTestCase(unittest.TestCase):
    # Default values
    flavor = "numpy"
    type = 'int32'
    shape = (2,2)
    start = 0
    stop = 10
    step = 1
    length = 1
    chunkshape = (5,5)
    compress = 0
    complib = "zlib"  # Default compression library
    shuffle = 0
    fletcher32 = 0
    reopen = 1  # Tells whether the file has to be reopened on each test or not

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()
        if self.reopen:
            # Close the file
            self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        if self.type == "StringType":
            atom = StringAtom(length=self.length, flavor=self.flavor)
        else:
            atom = Atom(dtype=self.type, flavor=self.flavor)
        title = self.__class__.__name__
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        carray = self.fileh.createCArray(group, 'carray1', atom, self.shape,
                                         title, filters = filters,
                                         chunkshape = self.chunkshape)

        # Fill it with data
        self.rowshape = list(carray.shape)
        self.objsize = self.length * numpy.product(self.shape)
        if self.flavor == "numarray":
            if self.type == "StringType":
                object = strings.array("a"*self.objsize, shape=self.shape,
                                       itemsize=carray.itemsize)
            else:
                object = numarray.arange(self.objsize, shape=self.shape,
                                         type=carray.ptype)
        elif self.flavor == "numpy":
            if self.type == "StringType":
                object = numpy.ndarray(buffer="a"*self.objsize,
                                       shape=self.shape,
                                       dtype="S%s" % carray.itemsize)
            else:
                object = numpy.arange(self.objsize, dtype=carray.dtype)
                object.shape = self.shape
        else:  # Numeric flavor
            object = Numeric.arange(self.objsize,
                                    typecode=typecode[carray.ptype])
            object = Numeric.reshape(object, self.shape)
        if verbose:
            print "Object to append -->", repr(object)

        carray[...] = object


    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test01_readCArray(self):
        """Checking read() of chunked layout arrays"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readCArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        carray = self.fileh.getNode("/carray1")

        # Choose a small value for buffer size
        carray._v_nrowsinbuf = 3
        if verbose:
            print "CArray descr:", repr(carray)
            print "shape of read array ==>", carray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.flavor == "numarray":
            if self.type == "StringType":
                object_ = strings.array("a"*self.objsize, shape=self.shape,
                                        itemsize=carray.itemsize)
            else:
                object_ = numarray.arange(self.objsize, shape=self.shape,
                                          type=carray.ptype)
        elif self.flavor == "numpy":
            if self.type == "StringType":
                object_ = numpy.ndarray(buffer="a"*self.objsize,
                                        shape=self.shape,
                                        dtype="S%s" % carray.itemsize)
            else:
                object_ = numpy.arange(self.objsize, dtype=carray.dtype)
                object_.shape = self.shape
        else:
            object_ = Numeric.arange(self.objsize,
                                     typecode=typecode[carray.ptype])
            object_ = Numeric.reshape(object_, self.shape)

        stop = self.stop
        # stop == None means read only the element designed by start
        # (in read() contexts)
        if self.stop == None:
            if self.start == -1:  # corner case
                stop = carray.nrows
            else:
                stop = self.start + 1
        # Protection against number of elements less than existing
        #if rowshape[self.extdim] < self.stop or self.stop == 0:
        if carray.nrows < stop:
            # self.stop == 0 means last row only in read()
            # and not in [::] slicing notation
            stop = carray.nrows
        # do a copy() in order to ensure that len(object._data)
        # actually do a measure of its length
        # Numeric 23.8 will issue an error with slices like -1:20:20
        # but this is an error with this Numeric version (and perhaps
        # lower ones).
        object = object_[self.start:stop:self.step].copy()

        # Read all the array
        try:
            data = carray.read(self.start,stop,self.step)
        except IndexError:
            if self.flavor == "numarray":
                data = numarray.array(None, shape=self.shape, type=self.ptype)
            elif self.flavor == "numpy":
                data = numpy.empty(shape=self.shape, dtype=self.type)
            else:
                data = Numeric.zeros(self.shape, typecode[self.type])

        if verbose:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(data)
            print "Should look like ==>", repr(object)

        if hasattr(data, "shape"):
            assert len(data.shape) == len(self.shape)
        else:
            # Scalar case
            assert len(self.shape) == 1
        assert carray._v_chunkshape == self.chunkshape
        assert allequal(data, object, self.flavor)

    def test02_getitemCArray(self):
        """Checking chunked layout array __getitem__ special method"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_getitemCArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        carray = self.fileh.getNode("/carray1")

        if verbose:
            print "CArray descr:", repr(carray)
            print "shape of read array ==>", carray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "StringType":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.shape,
                                    dtype="S%s" % carray.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=carray.dtype)
            object_.shape = self.shape

        stop = self.stop
        # do a copy() in order to ensure that len(object._data)
        # actually do a measure of its length
        object = object_.__getitem__(self.slices).copy()

        if self.flavor == "numarray":
            # Convert the object to Numarray
            object = convertNPToNumArray(object)
        elif self.flavor == "numeric":
            # Convert the object to Numeric
            object = convertNPToNumeric(object)

        # Read data from the array
        try:
            data = carray.__getitem__(self.slices)
        except IndexError:
            print "IndexError!"
            if self.flavor == "numarray":
                data = numarray.array(None, shape=self.shape, type=self.ptype)
            elif self.flavor == "numpy":
                data = numpy.empty(shape=self.shape, dtype=self.type)
            else:
                data = Numeric.zeros(self.shape, typecode[self.type])

        if verbose:
            print "Object read:\n", repr(data) #, data.info()
            print "Should look like:\n", repr(object) #, object.info()
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", data.shape
                print "shape should look as:", object.shape

        if not hasattr(data, "shape"):
            # Scalar case
            assert len(self.shape) == 1
        assert carray._v_chunkshape == self.chunkshape
        assert allequal(data, object, self.flavor)

    def test03_setitemCArray(self):
        """Checking chunked layout array __setitem__ special method"""

        rootgroup = self.rootgroup
        if self.__class__.__name__ == "Ellipsis6CArrayTestCase":
            # see test_earray.py BasicTestCase.test03_setitemEArray
            return
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_setitemCArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "a")
        carray = self.fileh.getNode("/carray1")

        if verbose:
            print "CArray descr:", repr(carray)
            print "shape of read array ==>", carray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "StringType":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.shape,
                                    dtype="S%s" % carray.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=carray.dtype)
            object_.shape = self.shape

        stop = self.stop
        # do a copy() in order to ensure that len(object._data)
        # actually do a measure of its length
        object = object_.__getitem__(self.slices).copy()

        if self.flavor == "numarray":
            # Convert the object to numarray
            object = convertNPToNumArray(object)
        elif self.flavor == "numeric":
            # Convert the object to Numeric
            object = convertNPToNumeric(object)

        if self.type == "StringType":
            if hasattr(self, "wslice"):
                object[self.wslize] = "xXx"
                carray[self.wslice] = "xXx"
            elif sum(object[self.slices].shape) <> 0 :
                object[:] = "xXx"
                if object.size > 0:
                    carray[self.slices] = object
        else:
            if hasattr(self, "wslice"):
                object[self.wslice] = object[self.wslice] * 2 + 3
                carray[self.wslice] = carray[self.wslice] * 2 + 3
            elif sum(object[self.slices].shape) <> 0:
                object = object * 2 + 3
                if reduce(lambda x,y:x*y, object.shape) > 0:
                    carray[self.slices] = carray[self.slices] * 2 + 3
            # Cast again object to its original type
            object = numpy.array(object, dtype=carray.dtype)
        # Read datafrom the array
        try:
            data = carray.__getitem__(self.slices)
        except IndexError:
            print "IndexError!"
            if self.flavor == "numarray":
                data = numarray.array(None, shape=self.shape, type=self.ptype)
            elif self.flavor == "numpy":
                data = numpy.empty(shape=self.shape, dtype=self.type)
            else:
                data = Numeric.zeros(self.shape, typecode[self.type])

        if verbose:
            print "Object read:\n", repr(data) #, data.info()
            print "Should look like:\n", repr(object) #, object.info()
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", data.shape
                print "shape should look as:", object.shape

        if not hasattr(data, "shape"):
            # Scalar case
            assert len(self.shape) == 1
        assert carray._v_chunkshape == self.chunkshape
        assert allequal(data, object, self.flavor)


class BasicWriteTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2,)
    chunkshape = (5,)
    step = 1
    wslice = 1  # single element case

class BasicWrite2TestCase(BasicTestCase):
    type = 'Int32'
    shape = (2,)
    chunkshape = (5,)
    step = 1
    wslice = slice(shape[0]-2,shape[0],2)  # range of elements
    reopen = 0  # This case does not reopen files

class EmptyCArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    start = 0
    stop = 10
    step = 1

class EmptyCArray2TestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    start = 0
    stop = 10
    step = 1
    reopen = 0  # This case does not reopen files

class SlicesCArrayTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    type = 'Int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    slices = (slice(1,2,1), slice(1,3,1))

class EllipsisCArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    #slices = (slice(1,2,1), Ellipsis)
    slices = (Ellipsis, slice(1,2,1))

class Slices2CArrayTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    type = 'Int32'
    shape = (2, 2, 4)
    chunkshape = (5, 5, 5)
    slices = (slice(1,2,1), slice(None, None, None), slice(1,4,2))

class Ellipsis2CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 2, 4)
    chunkshape = (5, 5, 5)
    slices = (slice(1,2,1), Ellipsis, slice(1,4,2))

class Slices3CArrayTestCase(BasicTestCase):
    compress = 1      # To show the chunks id DEBUG is on
    complib = "lzo"
    type = 'Int32'
    shape = (2, 3, 4, 2)
    chunkshape = (5, 5, 5, 5)
    slices = (slice(1, 2, 1), slice(0, None, None), slice(1,4,2))  # Don't work
    #slices = (slice(None, None, None), slice(0, None, None), slice(1,4,1)) # W
    #slices = (slice(None, None, None), slice(None, None, None), slice(1,4,2)) # N
    #slices = (slice(1,2,1), slice(None, None, None), slice(1,4,2)) # N
    # Disable the failing test temporarily with a working test case
    slices = (slice(1,2,1), slice(1, 4, None), slice(1,4,2)) # Y
    #slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,1)) # Y
    slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2)) # N
    #slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2), slice(0,100,1)) # N

class Slices4CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 4, 2, 5, 6)
    chunkshape = (5,5, 5, 5, 5, 5)
    slices = (slice(1, 2, 1), slice(0, None, None), slice(1,4,2),
              slice(0,4,2), slice(3,5,2), slice(2,7,1))

class Ellipsis3CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 4, 2)
    chunkshape = (5, 5, 5, 5)
    slices = (Ellipsis, slice(0, 4, None), slice(1,4,2))
    slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2), Ellipsis)

class Ellipsis4CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    slices = (Ellipsis, slice(0, 4, None), slice(1,4,2))
    slices = (slice(1,2,1), Ellipsis, slice(1,4,2))

class Ellipsis5CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    slices = (slice(1,2,1), slice(0, 4, None), Ellipsis)

class Ellipsis6CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    # The next slices gives problems with setting values (test03)
    # This is a problem on the test design, not the Array.__setitem__
    # code, though. See # see test_earray.py Ellipsis6EArrayTestCase
    slices = (slice(1,2,1), slice(0, 4, None), 2, Ellipsis)

class Ellipsis7CArrayTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    slices = (slice(1,2,1), slice(0, 4, None), slice(2,3), Ellipsis)

class MD3WriteTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 2, 3)
    chunkshape = (4, 4, 4)
    step = 2

class MD5WriteTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 2, 3, 4, 5)  # ok
    #shape = (1, 1, 2, 1)  # Minimum shape that shows problems with HDF5 1.6.1
    #shape = (2, 3, 2, 4, 5)  # Floating point exception (HDF5 1.6.1)
    #shape = (2, 3, 3, 2, 5, 6) # Segmentation fault (HDF5 1.6.1)
    chunkshape = (1, 1, 1, 1, 1)
    start = 1
    stop = 10
    step = 10

class MD6WriteTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 3, 2, 5, 6)
    chunkshape = (1, 1, 1, 1, 5, 6)
    start = 1
    stop = 10
    step = 3

class MD6WriteTestCase__(BasicTestCase):
    type = 'Int32'
    shape = (2, 2)
    chunkshape = (1, 1)
    start = 1
    stop = 3
    step = 1

class MD7WriteTestCase(BasicTestCase):
    type = 'Int32'
    shape = (2, 3, 3, 4, 5, 2, 3)
    chunkshape = (10, 10, 10, 10, 10, 10, 10)
    start = 1
    stop = 10
    step = 2

class MD10WriteTestCase(BasicTestCase):
    type = 'Int32'
    shape = (1, 2, 3, 4, 5, 5, 4, 3, 2, 2)
    chunkshape = (5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    start = -1
    stop = -1
    step = 10

class ZlibComprTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    start = 3
    #stop = 0   # means last row
    stop = None   # means last row from 0.8 on
    step = 10

class ZlibShuffleTestCase(BasicTestCase):
    shuffle = 1
    compress = 1
    complib = "zlib"
    # case start < stop , i.e. no rows read
    start = 3
    stop = 1
    step = 10

class LZOComprTestCase(BasicTestCase):
    compress = 1  # sss
    complib = "lzo"
    chunkshape = (10,10)
    start = 3
    stop = 10
    step = 3

class LZOShuffleTestCase(BasicTestCase):
    shape = (20,30)
    compress = 1
    shuffle = 1
    complib = "lzo"
    chunkshape = (100,100)
    start = 3
    stop = 10
    step = 7

class BZIP2ComprTestCase(BasicTestCase):
    shape = (20,30)
    compress = 1
    complib = "bzip2"
    chunkshape = (100,100)
    start = 3
    stop = 10
    step = 8

class BZIP2ShuffleTestCase(BasicTestCase):
    shape = (20,30)
    compress = 1
    shuffle = 1
    complib = "bzip2"
    chunkshape = (100,100)
    start = 3
    stop = 10
    step = 6

class Fletcher32TestCase(BasicTestCase):
    shape = (60,50)
    compress = 0
    fletcher32 = 1
    chunkshape = (50,50)
    start = 4
    stop = 20
    step = 7

class AllFiltersTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    fletcher32 = 1
    complib = "zlib"
    chunkshape = (20,20)  # sss
    start = 2
    stop = 99
    step = 6

class FloatTypeTestCase(BasicTestCase):
    type = 'Float64'
    shape = (2,2)
    chunkshape = (5,5)
    start = 3
    stop = 10
    step = 20

class ComplexTypeTestCase(BasicTestCase):
    type = 'Complex64'
    shape = (2,2)
    chunkshape = (5,5)
    start = 3
    stop = 10
    step = 20

class StringTypeTestCase(BasicTestCase):
    type = "StringType"
    length = 20
    shape = (2, 2)
    #shape = (2,2,20)
    chunkshape = (5,5)
    start = 3
    stop = 10
    step = 20
    slices = (slice(0,1),slice(1,2))

class StringType2TestCase(BasicTestCase):
    type = "StringType"
    length = 20
    shape = (2, 20)
    chunkshape = (5,5)
    start = 1
    stop = 10
    step = 2

class StringTypeComprTestCase(BasicTestCase):
    type = "StringType"
    length = 20
    shape = (20,2,10)
    #shape = (20,0,10,20)
    compr = 1
    #shuffle = 1  # this shouldn't do nothing on chars
    chunkshape = (50,50,2)
    start = -1
    stop = 100
    step = 20

class NumarrayInt8TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Int8"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumarrayInt16TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Int16"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = 1
    stop = 100
    step = 1

class NumarrayInt32TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Int32"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumarrayFloat32TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Float32"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumarrayFloat64TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Float64"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumarrayComplex32TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Complex32"
    shape = (4,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumarrayComplex64TestCase(BasicTestCase):
    flavor = "numarray"
    type = "Complex64"
    shape = (20,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumarrayComprTestCase(BasicTestCase):
    flavor = "numarray"
    type = "Float64"
    compress = 1
    shuffle = 1
    shape = (200,)
    compr = 1
    chunkshape = (21,)
    start = 51
    stop = 100
    step = 7

class NumericInt8TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Int8"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumericInt16TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Int16"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = 1
    stop = 100
    step = 1

class NumericInt32TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Int32"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumericFloat32TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Float32"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumericFloat64TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Float64"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumericComplex32TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Complex32"
    shape = (4,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumericComplex64TestCase(BasicTestCase):
    flavor = "numeric"
    type = "Complex64"
    shape = (20,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumericComprTestCase(BasicTestCase):
    flavor = "numeric"
    type = "Float64"
    compress = 1
    shuffle = 1
    shape = (200,)
    compr = 1
    chunkshape = (21,)
    start = 51
    stop = 100
    step = 7

# It remains a test of Numeric char types, but the code is getting too messy

class OffsetStrideTestCase(unittest.TestCase):
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test01a_String(self):
        """Checking carray with offseted NumPy strings appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_String..." % self.__class__.__name__

        shape = (3,2,2)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'strings',
                                         StringAtom(length=3), shape,
                                         "Array of strings",
                                         chunkshape=(1,2,2))
        a = numpy.array([[["a","b"],["123", "45"],["45", "123"]]], dtype="S3")
        carray[0] = a[:,1:]
        a = numpy.array([[["s", "a"],["ab", "f"],["s", "abc"],["abc", "f"]]])
        carray[1] = a[:,2:]

        # Read all the data:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Second row in carray ==>", data[1].tolist()

        assert carray.nrows == 3
        assert data[0].tolist() == [["123", "45"],["45", "123"]]
        assert data[1].tolist() == [["s", "abc"],["abc", "f"]]
        assert len(data[0]) == 2
        assert len(data[1]) == 2

    def test01b_String(self):
        """Checking carray with strided NumPy strings appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_String..." % self.__class__.__name__

        shape = (3,2,2)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'strings',
                                         StringAtom(length=3), shape,
                                         "Array of strings",
                                         chunkshape=(1,2,2))
        a = numpy.array([[["a","b"],["123", "45"],["45", "123"]]], dtype="S3")
        carray[0] = a[:,::2]
        a = numpy.array([[["s", "a"],["ab", "f"],["s", "abc"],["abc", "f"]]])
        carray[1] = a[:,::2]

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Second row in carray ==>", data[1].tolist()

        assert carray.nrows == 3
        assert data[0].tolist() == [["a","b"],["45", "123"]]
        assert data[1].tolist() == [["s", "a"],["s", "abc"]]
        assert len(data[0]) == 2
        assert len(data[1]) == 2

    def test02a_int(self):
        """Checking carray with offseted NumPy ints appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_int..." % self.__class__.__name__

        shape = (3,3)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'CAtom',
                                         Int32Atom(), shape,
                                         "array of ints",
                                         chunkshape=(1,3))
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], dtype='int32')
        carray[0:2] = a[2:]  # Introduce an offset
        a = numpy.array([(1,1,1), (-1,0,0)], dtype='int32')
        carray[2:3] = a[1:]  # Introduce an offset

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        assert carray.nrows == 3
        assert allequal(data[0], numpy.array([1,1,1], dtype='int32'))
        assert allequal(data[1], numpy.array([0,0,0], dtype='int32'))
        assert allequal(data[2], numpy.array([-1,0,0], dtype='int32'))

    def test02b_int(self):
        """Checking carray with strided NumPy ints appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_int..." % self.__class__.__name__

        shape = (3,3)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'CAtom',
                                         Int32Atom(), shape,
                                         "array of ints",
                                         chunkshape=(1,3))
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='int32')
        carray[0:2] = a[::3]  # Create an offset
        a = numpy.array([(1,1,1), (-1,0,0)], dtype='int32')
        carray[2:3] = a[::2]  # Create an offset

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        assert carray.nrows == 3
        assert allequal(data[0], numpy.array([0,0,0], dtype='Int32'))
        assert allequal(data[1], numpy.array([3,3,3], dtype='Int32'))
        assert allequal(data[2], numpy.array([1,1,1], dtype='Int32'))


class NumarrayOffsetStrideTestCase(unittest.TestCase):
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test02a_int(self):
        """Checking carray with offseted numarray ints appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_int..." % self.__class__.__name__

        shape = (3,3)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'CAtom',
                                         Int32Atom(), shape,
                                         "array of ints",
                                         chunkshape=(1,3))
        a = numarray.array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], type='Int32')
        carray[0:2] = a[2:]  # Introduce an offset
        a = numarray.array([(1,1,1), (-1,0,0)], type='Int32')
        carray[2:3] = a[1:]  # Introduce an offset

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        assert carray.nrows == 3
        assert allequal(data[0], numpy.array([1,1,1], dtype='i4'))
        assert allequal(data[1], numpy.array([0,0,0], dtype='i4'))
        assert allequal(data[2], numpy.array([-1,0,0], dtype='i4'))

    def test02b_int(self):
        """Checking carray with strided numarray ints appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_int..." % self.__class__.__name__

        shape = (3,3)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'CAtom',
                                         Int32Atom(), shape,
                                         "array of ints",
                                         chunkshape=(1,3))
        a = numarray.array([(0,0,0), (1,0,3), (1,2,1), (3,2,3)], type='Int32')
        carray[0:2] = a[::3]  # Create a strided object
        a = numarray.array([(1,0,1), (-1,0,0)], type='Int32')
        carray[2:3] = a[::2]  # Create a strided object

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        assert carray.nrows == 3
        assert allequal(data[0], numpy.array([0,0,0], dtype='i4'))
        assert allequal(data[1], numpy.array([3,2,3], dtype='i4'))
        assert allequal(data[2], numpy.array([1,0,1], dtype='i4'))


class NumericOffsetStrideTestCase(unittest.TestCase):
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test02a_int(self):
        """Checking carray with offseted Numeric ints appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_int..." % self.__class__.__name__

        shape = (3,3)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'CAtom',
                                         Int32Atom(), shape,
                                         "array of ints",
                                         chunkshape=(1,3))
        a = Numeric.array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], typecode='i')
        carray[0:2] = a[2:]  # Introduce an offset
        a = Numeric.array([(1,1,1), (-1,0,0)], typecode='i')
        carray[2:3] = a[1:]  # Introduce an offset

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        assert carray.nrows == 3
        assert allequal(data[0], numpy.array([1,1,1], dtype='i4'))
        assert allequal(data[1], numpy.array([0,0,0], dtype='i4'))
        assert allequal(data[2], numpy.array([-1,0,0], dtype='i4'))

    def test02b_int(self):
        """Checking carray with strided Numeric ints appends"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_int..." % self.__class__.__name__

        shape = (3,3)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'CAtom',
                                         Int32Atom(), shape,
                                         "array of ints",
                                         chunkshape=(1,3))
        a=Numeric.array([(0,0,0), (1,0,3), (1,2,1), (3,2,3)], typecode='i')
        carray[0:2] = a[::3]  # Create a strided object
        a=Numeric.array([(1,0,1), (-1,0,0)], typecode='i')
        carray[2:3] = a[::2]  # Create a strided object

        # Read all the rows:
        data = carray.read()
        if verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        assert carray.nrows == 3
        assert allequal(data[0], numpy.array([0,0,0], dtype='i'))
        assert allequal(data[1], numpy.array([3,2,3], dtype='i'))
        assert allequal(data[2], numpy.array([1,0,1], dtype='i'))


class CopyTestCase(unittest.TestCase):

    def test01a_copy(self):
        """Checking CArray.copy() method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (2,2)
        arr = Atom(dtype='Int16')
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        assert allequal(array1.read(), array2.read())

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test01b_copy(self):
        """Checking CArray.copy() method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (2,2)
        arr = Atom(dtype='Int16')
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(5, 5))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        assert allequal(array1.read(), array2.read())

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test01c_copy(self):
        """Checking CArray.copy() method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (5,5)
        arr = Atom(dtype='Int16')
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[:2,:2] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        assert allequal(array1.read(), array2.read())

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking CArray.copy() method (where specified)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (5,5)
        arr = Atom(dtype='Int16')
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[:2,:2] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        group1 = fileh.createGroup("/", "group1")
        array2 = array1.copy(group1, 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.group1.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        assert allequal(array1.read(), array2.read())

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test03_copy(self):
        """Checking CArray.copy() method (Numeric flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        if numeric:
            arr = Atom(dtype='Int16', flavor="numeric")
        else:
            arr = Atom(dtype='Int16')

        shape = (2,2)
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor   # Very important here!
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test03c_copy(self):
        """Checking CArray.copy() method (python flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03c_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        shape = (2,2)
        arr = Atom(dtype='Int16', flavor="python")
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = [[456, 2],[3, 457]]

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all elements are equal
        assert array1.read() == array2.read()
        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor   # Very important here!
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test03d_copy(self):
        """Checking CArray.copy() method (string python flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03d_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        shape = (2,2)
        arr = StringAtom(length=4, flavor="python")
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = [["456", "2"],["3", "457"]]

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "type value-->", type(array2[:][0][0])
            print "value-->", array2[:]
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all elements are equal
        assert array1.read() == array2.read()

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor   # Very important here!
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test03e_copy(self):
        """Checking CArray.copy() method (chararray flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03e_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        shape = (2,2)
        arr = StringAtom(length=4)
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = numpy.array([["456", "2"],["3", "457"]], dtype="S4")

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all elements are equal
        assert allequal(array1.read(), array2.read())
        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.extdim == array2.extdim
        assert array1.flavor == array2.flavor   # Very important here!
        assert array1.dtype == array2.dtype
        assert array1.ptype == array2.ptype
        assert array1.title == array2.title
        assert str(array1.atom) == str(array2.atom)
        assert array1._v_chunkshape == array2._v_chunkshape

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking CArray.copy() method (checking title copying)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (2,2)
        atom = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', atom, shape,
                                    "title array1", chunkshape=(2,2))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', title="title array2")

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        # Assert user attributes
        if verbose:
            print "title of destination array-->", array2.title
        array2.title == "title array2"

        # Close the file
        fileh.close()
        os.remove(file)

    def test05_copy(self):
        """Checking CArray.copy() method (user attributes copied)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (2,2)
        atom = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', atom, shape,
                                    "title array1", chunkshape=(2,2))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=1)

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Assert user attributes
        array2.attrs.attr1 == "attr1"
        array2.attrs.attr2 == 2

        # Close the file
        fileh.close()
        os.remove(file)

    def test05b_copy(self):
        """Checking CArray.copy() method (user attributes not copied)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        shape = (2,2)
        atom = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', atom, shape,
                                    "title array1", chunkshape=(2,2))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=0)

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Assert user attributes
        hasattr(array2.attrs, "attr1") == 0
        hasattr(array2.attrs, "attr2") == 0

        # Close the file
        fileh.close()
        os.remove(file)


class CloseCopyTestCase(CopyTestCase):
    close = 1

class OpenCopyTestCase(CopyTestCase):
    close = 0

class CopyIndexTestCase(unittest.TestCase):
    nrowsinbuf = 2

    def test01_index(self):
        """Checking CArray.copy() method with indexes"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (100,2)
        atom = Int32Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', atom, shape,
                                    "title array1", chunkshape=(2,2))
        r = numpy.arange(200, dtype='int32')
        r.shape = shape
        array1[...] = r

        # Select a different buffer size:
        array1._v_nrowsinbuf = self.nrowsinbuf

        # Copy to another array
        array2 = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        assert allequal(r2, array2.read())

        # Assert the number of rows in array
        if verbose:
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", r2.shape[0]

        assert array1._v_chunkshape == array2._v_chunkshape
        assert r2.shape[0] == array2.nrows

        # Close the file
        fileh.close()
        os.remove(file)

    def _test02_indexclosef(self):
        """Checking CArray.copy() method with indexes (close file version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_indexclosef..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (100,2)
        atom = Int32Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', atom, shape,
                                    "title array1", chunkshape=(2,2))
        r = numpy.arange(200, dtype='int32')
        r.shape = shape
        array1[...] = r

        # Select a different buffer size:
        array1._v_nrowsinbuf = self.nrowsinbuf

        # Copy to another array
        array2 = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        # Close and reopen the file
        fileh.close()
        fileh = openFile(file, mode = "r")
        array1 = fileh.root.array1
        array2 = fileh.root.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        assert array1._v_chunkshape == array2._v_chunkshape
        assert allequal(r2, array2.read())

        # Assert the number of rows in array
        if verbose:
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", r2.shape[0]
        assert r2.shape[0] == array2.nrows

        # Close the file
        fileh.close()
        os.remove(file)

class CopyIndex1TestCase(CopyIndexTestCase):
    nrowsinbuf = 1
    start = 0
    stop = 7
    step = 1

class CopyIndex2TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    start = 0
    stop = -1
    step = 1

class CopyIndex3TestCase(CopyIndexTestCase):
    nrowsinbuf = 3
    start = 1
    stop = 7
    step = 1

class CopyIndex4TestCase(CopyIndexTestCase):
    nrowsinbuf = 4
    start = 0
    stop = 6
    step = 1

class CopyIndex5TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    start = 3
    stop = 7
    step = 1

class CopyIndex6TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    start = 3
    stop = 6
    step = 2

class CopyIndex7TestCase(CopyIndexTestCase):
    start = 0
    stop = 7
    step = 10

class CopyIndex8TestCase(CopyIndexTestCase):
    start = 6
    stop = -1  # Negative values means starting from the end
    step = 1

class CopyIndex9TestCase(CopyIndexTestCase):
    start = 3
    stop = 4
    step = 1

class CopyIndex10TestCase(CopyIndexTestCase):
    nrowsinbuf = 1
    start = 3
    stop = 4
    step = 2

class CopyIndex11TestCase(CopyIndexTestCase):
    start = -3
    stop = -1
    step = 2

class CopyIndex12TestCase(CopyIndexTestCase):
    start = -1   # Should point to the last element
    stop = None  # None should mean the last element (including it)
    step = 1

# The next test should be run only in **heavy** mode
class Rows64bitsTestCase(unittest.TestCase):
    narows = 1000*1000L   # each array will have 1 million entries
    #narows = 1000L        # for testing only
    nanumber = 1000*3L    # That should account for more than 2**31-1

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = self.fileh = openFile(self.file, "a")
        # Create an CArray
        shape = (self.narows*self.nanumber,)
        array = fileh.createCArray(fileh.root, 'array',
                                   Int8Atom(), shape,
                                   filters=Filters(complib='lzo', complevel=1),
                                   chunkshape=(1024,))

        # Fill the array
        na = numpy.arange(self.narows, dtype='int8')
        #~ for i in xrange(self.nanumber):
            #~ s = slice(i*self.narows, (i+1)*self.narows)
            #~ array[s] = na
        s = slice(0, self.narows)
        array[s] = na
        s = slice((self.nanumber-1)*self.narows, self.nanumber*self.narows)
        array[s] = na


    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test01_basiccheck(self):
        "Some basic checks for carrays exceeding 2**31 rows"

        fileh = self.fileh
        array = fileh.root.array

        if self.close:
            if verbose:
                # Check how many entries there are in the array
                print "Before closing"
                print "Entries:", array.nrows, type(array.nrows)
                print "Entries:", array.nrows / (1000*1000), "Millions"
                print "Shape:", array.shape
            # Close the file
            fileh.close()
            # Re-open the file
            fileh = self.fileh = openFile(self.file)
            array = fileh.root.array
            if verbose:
                print "After re-open"

        # Check how many entries there are in the array
        if verbose:
            print "Entries:", array.nrows, type(array.nrows)
            print "Entries:", array.nrows / (1000*1000), "Millions"
            print "Shape:", array.shape
            print "Last 10 elements-->", array[-10:]
            stop = self.narows%256
            if stop > 127:
                stop -= 256
            start = stop - 10
            #print "start, stop-->", start, stop
            print "Should look like:", numpy.arange(start, stop, dtype='int8')

        nrows = self.narows*self.nanumber
        # check nrows
        assert array.nrows == nrows
        # Check shape
        assert array.shape == (nrows,)
        # check the 10 first elements
        assert allequal(array[:10], numpy.arange(10, dtype='int8'))
        # check the 10 last elements
        stop = self.narows%256
        if stop > 127:
            stop -= 256
        start = stop - 10
        assert allequal(array[-10:], numpy.arange(start, stop, dtype='int8'))


class Rows64bitsTestCase1(Rows64bitsTestCase):
    close = 0

class Rows64bitsTestCase2(Rows64bitsTestCase):
    close = 1

#----------------------------------------------------------------------


def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1
    #heavy = 1  # uncomment this only for testing purposes

    #theSuite.addTest(unittest.makeSuite(BasicTestCase))
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(BasicWrite2TestCase))
        theSuite.addTest(unittest.makeSuite(EmptyCArrayTestCase))
        theSuite.addTest(unittest.makeSuite(EmptyCArray2TestCase))
        theSuite.addTest(unittest.makeSuite(SlicesCArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Slices2CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(EllipsisCArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis2CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis3CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(BZIP2ComprTestCase))
        theSuite.addTest(unittest.makeSuite(BZIP2ShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(FloatTypeTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexTypeTestCase))
        theSuite.addTest(unittest.makeSuite(StringTypeTestCase))
        theSuite.addTest(unittest.makeSuite(StringType2TestCase))
        theSuite.addTest(unittest.makeSuite(StringTypeComprTestCase))
        if numarray_imported:
            theSuite.addTest(unittest.makeSuite(NumarrayInt8TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayInt16TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayInt32TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayFloat32TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayFloat64TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayComplex32TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayComplex64TestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayComprTestCase))
            theSuite.addTest(unittest.makeSuite(NumarrayOffsetStrideTestCase))
        if numeric:
            theSuite.addTest(unittest.makeSuite(NumericInt8TestCase))
            theSuite.addTest(unittest.makeSuite(NumericInt16TestCase))
            theSuite.addTest(unittest.makeSuite(NumericInt32TestCase))
            theSuite.addTest(unittest.makeSuite(NumericFloat32TestCase))
            theSuite.addTest(unittest.makeSuite(NumericFloat64TestCase))
            theSuite.addTest(unittest.makeSuite(NumericComplex32TestCase))
            theSuite.addTest(unittest.makeSuite(NumericComplex64TestCase))
            theSuite.addTest(unittest.makeSuite(NumericComprTestCase))
            theSuite.addTest(unittest.makeSuite(NumericOffsetStrideTestCase))

        theSuite.addTest(unittest.makeSuite(OffsetStrideTestCase))
        theSuite.addTest(unittest.makeSuite(Fletcher32TestCase))
        theSuite.addTest(unittest.makeSuite(AllFiltersTestCase))
        theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
        theSuite.addTest(unittest.makeSuite(OpenCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex2TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex3TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex4TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex5TestCase))
    if heavy:
        theSuite.addTest(unittest.makeSuite(Slices3CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Slices4CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis4CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis5CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis6CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis7CArrayTestCase))
        theSuite.addTest(unittest.makeSuite(MD3WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD5WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD6WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD7WriteTestCase))
        theSuite.addTest(unittest.makeSuite(MD10WriteTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex6TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex7TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex8TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex9TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex10TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex11TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex12TestCase))
        theSuite.addTest(unittest.makeSuite(Rows64bitsTestCase1))
        theSuite.addTest(unittest.makeSuite(Rows64bitsTestCase2))

    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
