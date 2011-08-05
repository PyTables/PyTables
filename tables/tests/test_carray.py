# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import unittest
import os
import tempfile

import numpy

from tables import *
from tables.flavor import flavor_to_flavor
from tables.tests import common
from tables.tests.common import (
    typecode, allequal, numeric_imported, numarray_imported)

if numarray_imported:
    import numarray
if numeric_imported:
    import Numeric

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


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
        if self.type == "string":
            atom = StringAtom(itemsize=self.length)
        else:
            atom = Atom.from_type(self.type)
        title = self.__class__.__name__
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        carray = self.fileh.createCArray(group, 'carray1', atom, self.shape,
                                         title, filters=filters,
                                         chunkshape = self.chunkshape)
        carray.flavor = self.flavor

        # Fill it with data
        self.rowshape = list(carray.shape)
        self.objsize = self.length * numpy.prod(self.shape)
        if self.flavor == "numarray":
            if self.type == "string":
                object = strings.array("a"*self.objsize, shape=self.shape,
                                       itemsize=carray.atom.itemsize)
            else:
                type_ = numpy.sctypeNA[numpy.sctypeDict[carray.atom.type]]
                object = numarray.arange(self.objsize, shape=self.shape,
                                         type=type_)
        elif self.flavor == "numpy":
            if self.type == "string":
                object = numpy.ndarray(buffer="a"*self.objsize,
                                       shape=self.shape,
                                       dtype="S%s" % carray.atom.itemsize)
            else:
                object = numpy.arange(self.objsize, dtype=carray.atom.dtype)
                object.shape = self.shape
        else:  # Numeric flavor
            object = Numeric.arange(self.objsize,
                                    typecode=typecode[carray.atom.type])
            object = Numeric.reshape(object, self.shape)
        if common.verbose:
            print "Object to append -->", repr(object)

        carray[...] = object


    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01_readCArray(self):
        """Checking read() of chunked layout arrays"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readCArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        carray = self.fileh.getNode("/carray1")

        # Choose a small value for buffer size
        carray.nrowsinbuf = 3
        if common.verbose:
            print "CArray descr:", repr(carray)
            print "shape of read array ==>", carray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.flavor == "numarray":
            if self.type == "string":
                object_ = strings.array("a"*self.objsize, shape=self.shape,
                                        itemsize=carray.atom.itemsize)
            else:
                type_ = numpy.sctypeNA[numpy.sctypeDict[carray.atom.type]]
                object_ = numarray.arange(self.objsize, shape=self.shape,
                                          type=type_)
        elif self.flavor == "numpy":
            if self.type == "string":
                object_ = numpy.ndarray(buffer="a"*self.objsize,
                                        shape=self.shape,
                                        dtype="S%s" % carray.atom.itemsize)
            else:
                object_ = numpy.arange(self.objsize, dtype=carray.atom.dtype)
                object_.shape = self.shape
        else:
            object_ = Numeric.arange(self.objsize,
                                     typecode=typecode[carray.atom.type])
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
            stop = int(carray.nrows)
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
                data = numarray.array(None, shape=self.shape, type=self.type)
            elif self.flavor == "numpy":
                data = numpy.empty(shape=self.shape, dtype=self.type)
            else:
                data = Numeric.zeros(self.shape, typecode[self.type])

        if common.verbose:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(data)
            print "Should look like ==>", repr(object)

        if hasattr(data, "shape"):
            self.assertEqual(len(data.shape), len(self.shape))
        else:
            # Scalar case
            self.assertEqual(len(self.shape), 1)
        self.assertEqual(carray.chunkshape, self.chunkshape)
        self.assertTrue(allequal(data, object, self.flavor))

    def test02_getitemCArray(self):
        """Checking chunked layout array __getitem__ special method"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_getitemCArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        carray = self.fileh.getNode("/carray1")

        if common.verbose:
            print "CArray descr:", repr(carray)
            print "shape of read array ==>", carray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.shape,
                                    dtype="S%s" % carray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=carray.atom.dtype)
            object_.shape = self.shape

        stop = self.stop
        # do a copy() in order to ensure that len(object._data)
        # actually do a measure of its length
        object = object_.__getitem__(self.slices).copy()

        if self.flavor == "numarray":
            # Convert the object to Numarray
            object = flavor_to_flavor(object, 'numpy', 'numarray')
        elif self.flavor == "numeric":
            # Convert the object to Numeric
            object = flavor_to_flavor(object, 'numpy', 'numeric')

        # Read data from the array
        try:
            data = carray.__getitem__(self.slices)
        except IndexError:
            print "IndexError!"
            if self.flavor == "numarray":
                data = numarray.array(None, shape=self.shape, type=self.type)
            elif self.flavor == "numpy":
                data = numpy.empty(shape=self.shape, dtype=self.type)
            else:
                data = Numeric.zeros(self.shape, typecode[self.type])

        if common.verbose:
            print "Object read:\n", repr(data) #, data.info()
            print "Should look like:\n", repr(object) #, object.info()
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", data.shape
                print "shape should look as:", object.shape

        if not hasattr(data, "shape"):
            # Scalar case
            self.assertEqual(len(self.shape), 1)
        self.assertEqual(carray.chunkshape, self.chunkshape)
        self.assertTrue(allequal(data, object, self.flavor))

    def test03_setitemCArray(self):
        """Checking chunked layout array __setitem__ special method"""

        rootgroup = self.rootgroup
        if self.__class__.__name__ == "Ellipsis6CArrayTestCase":
            # see test_earray.py BasicTestCase.test03_setitemEArray
            return
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_setitemCArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "a")
        carray = self.fileh.getNode("/carray1")

        if common.verbose:
            print "CArray descr:", repr(carray)
            print "shape of read array ==>", carray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.shape,
                                    dtype="S%s" % carray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=carray.atom.dtype)
            object_.shape = self.shape

        stop = self.stop
        # do a copy() in order to ensure that len(object._data)
        # actually do a measure of its length
        object = object_.__getitem__(self.slices).copy()

        if self.flavor == "numarray":
            # Convert the object to numarray
            object = flavor_to_flavor(object, 'numpy', 'numarray')
        elif self.flavor == "numeric":
            # Convert the object to Numeric
            object = flavor_to_flavor(object, 'numpy', 'numeric')

        if self.type == "string":
            if hasattr(self, "wslice"):
                object[self.wslize] = "xXx"
                carray[self.wslice] = "xXx"
            elif sum(object[self.slices].shape) != 0 :
                object[:] = "xXx"
                if object.size > 0:
                    carray[self.slices] = object
        else:
            if hasattr(self, "wslice"):
                object[self.wslice] = object[self.wslice] * 2 + 3
                carray[self.wslice] = carray[self.wslice] * 2 + 3
            elif sum(object[self.slices].shape) != 0:
                object = object * 2 + 3
                if reduce(lambda x,y:x*y, object.shape) > 0:
                    carray[self.slices] = carray[self.slices] * 2 + 3
            # Cast again object to its original type
            object = numpy.array(object, dtype=carray.atom.dtype)
        # Read datafrom the array
        try:
            data = carray.__getitem__(self.slices)
        except IndexError:
            print "IndexError!"
            if self.flavor == "numarray":
                data = numarray.array(None, shape=self.shape, type=self.type)
            elif self.flavor == "numpy":
                data = numpy.empty(shape=self.shape, dtype=self.type)
            else:
                data = Numeric.zeros(self.shape, typecode[self.type])

        if common.verbose:
            print "Object read:\n", repr(data) #, data.info()
            print "Should look like:\n", repr(object) #, object.info()
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", data.shape
                print "shape should look as:", object.shape

        if not hasattr(data, "shape"):
            # Scalar case
            self.assertEqual(len(self.shape), 1)
        self.assertEqual(carray.chunkshape, self.chunkshape)
        self.assertTrue(allequal(data, object, self.flavor))


class BasicWriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2,)
    chunkshape = (5,)
    step = 1
    wslice = 1  # single element case

class BasicWrite2TestCase(BasicTestCase):
    type = 'int32'
    shape = (2,)
    chunkshape = (5,)
    step = 1
    wslice = slice(shape[0]-2,shape[0],2)  # range of elements
    reopen = 0  # This case does not reopen files

class EmptyCArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    start = 0
    stop = 10
    step = 1

class EmptyCArray2TestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    start = 0
    stop = 10
    step = 1
    reopen = 0  # This case does not reopen files

class SlicesCArrayTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    type = 'int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    slices = (slice(1,2,1), slice(1,3,1))

class EllipsisCArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 2)
    chunkshape = (5, 5)
    #slices = (slice(1,2,1), Ellipsis)
    slices = (Ellipsis, slice(1,2,1))

class Slices2CArrayTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    type = 'int32'
    shape = (2, 2, 4)
    chunkshape = (5, 5, 5)
    slices = (slice(1,2,1), slice(None, None, None), slice(1,4,2))

class Ellipsis2CArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 2, 4)
    chunkshape = (5, 5, 5)
    slices = (slice(1,2,1), Ellipsis, slice(1,4,2))

class Slices3CArrayTestCase(BasicTestCase):
    compress = 1      # To show the chunks id DEBUG is on
    complib = "lzo"
    type = 'int32'
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
    type = 'int32'
    shape = (2, 3, 4, 2, 5, 6)
    chunkshape = (5,5, 5, 5, 5, 5)
    slices = (slice(1, 2, 1), slice(0, None, None), slice(1,4,2),
              slice(0,4,2), slice(3,5,2), slice(2,7,1))

class Ellipsis3CArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 2)
    chunkshape = (5, 5, 5, 5)
    slices = (Ellipsis, slice(0, 4, None), slice(1,4,2))
    slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2), Ellipsis)

class Ellipsis4CArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    slices = (Ellipsis, slice(0, 4, None), slice(1,4,2))
    slices = (slice(1,2,1), Ellipsis, slice(1,4,2))

class Ellipsis5CArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    slices = (slice(1,2,1), slice(0, 4, None), Ellipsis)

class Ellipsis6CArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    # The next slices gives problems with setting values (test03)
    # This is a problem on the test design, not the Array.__setitem__
    # code, though. See # see test_earray.py Ellipsis6EArrayTestCase
    slices = (slice(1,2,1), slice(0, 4, None), 2, Ellipsis)

class Ellipsis7CArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 5)
    chunkshape = (5, 5, 5, 5)
    slices = (slice(1,2,1), slice(0, 4, None), slice(2,3), Ellipsis)

class MD3WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 2, 3)
    chunkshape = (4, 4, 4)
    step = 2

class MD5WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 2, 3, 4, 5)  # ok
    #shape = (1, 1, 2, 1)  # Minimum shape that shows problems with HDF5 1.6.1
    #shape = (2, 3, 2, 4, 5)  # Floating point exception (HDF5 1.6.1)
    #shape = (2, 3, 3, 2, 5, 6) # Segmentation fault (HDF5 1.6.1)
    chunkshape = (1, 1, 1, 1, 1)
    start = 1
    stop = 10
    step = 10

class MD6WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 3, 2, 5, 6)
    chunkshape = (1, 1, 1, 1, 5, 6)
    start = 1
    stop = 10
    step = 3

class MD6WriteTestCase__(BasicTestCase):
    type = 'int32'
    shape = (2, 2)
    chunkshape = (1, 1)
    start = 1
    stop = 3
    step = 1

class MD7WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 3, 4, 5, 2, 3)
    chunkshape = (10, 10, 10, 10, 10, 10, 10)
    start = 1
    stop = 10
    step = 2

class MD10WriteTestCase(BasicTestCase):
    type = 'int32'
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

class BloscComprTestCase(BasicTestCase):
    compress = 1  # sss
    complib = "blosc"
    chunkshape = (10,10)
    start = 3
    stop = 10
    step = 3

class BloscShuffleTestCase(BasicTestCase):
    shape = (20,30)
    compress = 1
    shuffle = 1
    complib = "blosc"
    chunkshape = (100,100)
    start = 3
    stop = 10
    step = 7

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

class Bzip2ComprTestCase(BasicTestCase):
    shape = (20,30)
    compress = 1
    complib = "bzip2"
    chunkshape = (100,100)
    start = 3
    stop = 10
    step = 8

class Bzip2ShuffleTestCase(BasicTestCase):
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
    type = 'float64'
    shape = (2,2)
    chunkshape = (5,5)
    start = 3
    stop = 10
    step = 20

class ComplexTypeTestCase(BasicTestCase):
    type = 'complex128'
    shape = (2,2)
    chunkshape = (5,5)
    start = 3
    stop = 10
    step = 20

class StringTestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (2, 2)
    #shape = (2,2,20)
    chunkshape = (5,5)
    start = 3
    stop = 10
    step = 20
    slices = (slice(0,1),slice(1,2))

class String2TestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (2, 20)
    chunkshape = (5,5)
    start = 1
    stop = 10
    step = 2

class StringComprTestCase(BasicTestCase):
    type = "string"
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
    type = "int8"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumarrayInt16TestCase(BasicTestCase):
    flavor = "numarray"
    type = "int16"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = 1
    stop = 100
    step = 1

class NumarrayInt32TestCase(BasicTestCase):
    flavor = "numarray"
    type = "int32"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumarrayFloat32TestCase(BasicTestCase):
    flavor = "numarray"
    type = "float32"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumarrayFloat64TestCase(BasicTestCase):
    flavor = "numarray"
    type = "float64"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumarrayComplex64TestCase(BasicTestCase):
    flavor = "numarray"
    type = "complex64"
    shape = (4,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumarrayComplex128TestCase(BasicTestCase):
    flavor = "numarray"
    type = "complex128"
    shape = (20,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumarrayComprTestCase(BasicTestCase):
    flavor = "numarray"
    type = "float64"
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
    type = "int8"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumericInt16TestCase(BasicTestCase):
    flavor = "numeric"
    type = "int16"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = 1
    stop = 100
    step = 1

class NumericInt32TestCase(BasicTestCase):
    flavor = "numeric"
    type = "int32"
    shape = (2,2)
    compress = 1
    shuffle = 1
    chunkshape = (50,50)
    start = -1
    stop = 100
    step = 20

class NumericFloat32TestCase(BasicTestCase):
    flavor = "numeric"
    type = "float32"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumericFloat64TestCase(BasicTestCase):
    flavor = "numeric"
    type = "float64"
    shape = (200,)
    compress = 1
    shuffle = 1
    chunkshape = (20,)
    start = -1
    stop = 100
    step = 20

class NumericComplex64TestCase(BasicTestCase):
    flavor = "numeric"
    type = "complex64"
    shape = (4,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumericComplex128TestCase(BasicTestCase):
    flavor = "numeric"
    type = "complex128"
    shape = (20,)
    compress = 1
    shuffle = 1
    chunkshape = (2,)
    start = -1
    stop = 100
    step = 20

class NumericComprTestCase(BasicTestCase):
    flavor = "numeric"
    type = "float64"
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
        common.cleanup(self)

    #----------------------------------------

    def test01a_String(self):
        """Checking carray with offseted NumPy strings appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_String..." % self.__class__.__name__

        shape = (3,2,2)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'strings',
                                         StringAtom(itemsize=3), shape,
                                         "Array of strings",
                                         chunkshape=(1,2,2))
        a = numpy.array([[["a","b"],["123", "45"],["45", "123"]]], dtype="S3")
        carray[0] = a[0,1:]
        a = numpy.array([[["s", "a"],["ab", "f"],["s", "abc"],["abc", "f"]]])
        carray[1] = a[0,2:]

        # Read all the data:
        data = carray.read()
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Second row in carray ==>", data[1].tolist()

        self.assertEqual(carray.nrows, 3)
        self.assertEqual(data[0].tolist(), [["123", "45"],["45", "123"]])
        self.assertEqual(data[1].tolist(), [["s", "abc"],["abc", "f"]])
        self.assertEqual(len(data[0]), 2)
        self.assertEqual(len(data[1]), 2)

    def test01b_String(self):
        """Checking carray with strided NumPy strings appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_String..." % self.__class__.__name__

        shape = (3,2,2)
        # Create an string atom
        carray = self.fileh.createCArray(root, 'strings',
                                         StringAtom(itemsize=3), shape,
                                         "Array of strings",
                                         chunkshape=(1,2,2))
        a = numpy.array([[["a","b"],["123", "45"],["45", "123"]]], dtype="S3")
        carray[0] = a[0,::2]
        a = numpy.array([[["s", "a"],["ab", "f"],["s", "abc"],["abc", "f"]]])
        carray[1] = a[0,::2]

        # Read all the rows:
        data = carray.read()
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Second row in carray ==>", data[1].tolist()

        self.assertEqual(carray.nrows, 3)
        self.assertEqual(data[0].tolist(), [["a","b"],["45", "123"]])
        self.assertEqual(data[1].tolist(), [["s", "a"],["s", "abc"]])
        self.assertEqual(len(data[0]), 2)
        self.assertEqual(len(data[1]), 2)

    def test02a_int(self):
        """Checking carray with offseted NumPy ints appends"""

        root = self.rootgroup
        if common.verbose:
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
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        self.assertEqual(carray.nrows, 3)
        self.assertTrue(allequal(data[0], numpy.array([1,1,1], dtype='int32')))
        self.assertTrue(allequal(data[1], numpy.array([0,0,0], dtype='int32')))
        self.assertTrue(allequal(data[2], numpy.array([-1,0,0], dtype='int32')))

    def test02b_int(self):
        """Checking carray with strided NumPy ints appends"""

        root = self.rootgroup
        if common.verbose:
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
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        self.assertEqual(carray.nrows, 3)
        self.assertTrue(allequal(data[0], numpy.array([0,0,0], dtype='int32')))
        self.assertTrue(allequal(data[1], numpy.array([3,3,3], dtype='int32')))
        self.assertTrue(allequal(data[2], numpy.array([1,1,1], dtype='int32')))


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
        common.cleanup(self)

    #----------------------------------------

    def test02a_int(self):
        """Checking carray with offseted numarray ints appends"""

        root = self.rootgroup
        if common.verbose:
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
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        self.assertEqual(carray.nrows, 3)
        self.assertTrue(allequal(data[0], numpy.array([1,1,1], dtype='i4')))
        self.assertTrue(allequal(data[1], numpy.array([0,0,0], dtype='i4')))
        self.assertTrue(allequal(data[2], numpy.array([-1,0,0], dtype='i4')))

    def test02b_int(self):
        """Checking carray with strided numarray ints appends"""

        root = self.rootgroup
        if common.verbose:
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
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        self.assertEqual(carray.nrows, 3)
        self.assertTrue(allequal(data[0], numpy.array([0,0,0], dtype='i4')))
        self.assertTrue(allequal(data[1], numpy.array([3,2,3], dtype='i4')))
        self.assertTrue(allequal(data[2], numpy.array([1,0,1], dtype='i4')))


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
        common.cleanup(self)

    #----------------------------------------

    def test02a_int(self):
        """Checking carray with offseted Numeric ints appends"""

        root = self.rootgroup
        if common.verbose:
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
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        self.assertEqual(carray.nrows, 3)
        self.assertTrue(allequal(data[0], numpy.array([1,1,1], dtype='i4')))
        self.assertTrue(allequal(data[1], numpy.array([0,0,0], dtype='i4')))
        self.assertTrue(allequal(data[2], numpy.array([-1,0,0], dtype='i4')))

    def test02b_int(self):
        """Checking carray with strided Numeric ints appends"""

        root = self.rootgroup
        if common.verbose:
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
        if common.verbose:
            print "Object read:", data
            print "Nrows in", carray._v_pathname, ":", carray.nrows
            print "Third row in carray ==>", data[2]

        self.assertEqual(carray.nrows, 3)
        self.assertTrue(allequal(data[0], numpy.array([0,0,0], dtype='i')))
        self.assertTrue(allequal(data[1], numpy.array([3,2,3], dtype='i')))
        self.assertTrue(allequal(data[2], numpy.array([1,0,1], dtype='i')))


class CopyTestCase(unittest.TestCase):

    def test01a_copy(self):
        """Checking CArray.copy() method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (2,2)
        arr = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test01b_copy(self):
        """Checking CArray.copy() method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (2,2)
        arr = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(5, 5))
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # By default, the chunkshape should be the same
        self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test01c_copy(self):
        """Checking CArray.copy() method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (5,5)
        arr = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[:2,:2] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking CArray.copy() method (where specified)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an CArray
        shape = (5,5)
        arr = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[:2,:2] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        group1 = fileh.createGroup("/", "group1")
        array2 = array1.copy(group1, 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.group1.array2

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    # Numeric is now deprecated
    def _test03_copy(self):
        """Checking CArray.copy() method (Numeric flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        if numeric_imported:
            flavor="numeric"
        else:
            flavor="numpy"

        arr = Int16Atom()
        shape = (2,2)
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1.flavor = flavor
        array1[...] = numpy.array([[456, 2],[3, 457]], dtype='int16')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor) # Very important here!
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test03c_copy(self):
        """Checking CArray.copy() method (python flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03c_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        shape = (2,2)
        arr = Int16Atom()
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1.flavor = "python"
        array1[...] = [[456, 2],[3, 457]]

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all elements are equal
        self.assertEqual(array1.read(), array2.read())
        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor) # Very important here!
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test03d_copy(self):
        """Checking CArray.copy() method (string python flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03d_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        shape = (2,2)
        arr = StringAtom(itemsize=4)
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1.flavor = "python"
        array1[...] = [["456", "2"],["3", "457"]]

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "type value-->", type(array2[:][0][0])
            print "value-->", array2[:]
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all elements are equal
        self.assertEqual(array1.read(), array2.read())

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor) # Very important here!
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test03e_copy(self):
        """Checking CArray.copy() method (chararray flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03e_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        shape = (2,2)
        arr = StringAtom(itemsize=4)
        array1 = fileh.createCArray(fileh.root, 'array1', arr, shape,
                                    "title array1", chunkshape=(2, 2))
        array1[...] = numpy.array([["456", "2"],["3", "457"]], dtype="S4")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))
        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.extdim, array2.extdim)
        self.assertEqual(array1.flavor, array2.flavor) # Very important here!
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))
        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #self.assertEqual(array1.chunkshape, array2.chunkshape)

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking CArray.copy() method (checking title copying)"""

        if common.verbose:
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
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', title="title array2")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        # Assert user attributes
        if common.verbose:
            print "title of destination array-->", array2.title
        self.assertEqual(array2.title, "title array2")

        # Close the file
        fileh.close()
        os.remove(file)

    def test05_copy(self):
        """Checking CArray.copy() method (user attributes copied)"""

        if common.verbose:
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
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=1)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Assert user attributes
        self.assertEqual(array2.attrs.attr1, "attr1")
        self.assertEqual(array2.attrs.attr2, 2)

        # Close the file
        fileh.close()
        os.remove(file)

    def test05b_copy(self):
        """Checking CArray.copy() method (user attributes not copied)"""

        if common.verbose:
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
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=0)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Assert user attributes
        self.assertEqual(hasattr(array2.attrs, "attr1"), 0)
        self.assertEqual(hasattr(array2.attrs, "attr2"), 0)

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

        if common.verbose:
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
        array1.nrowsinbuf = self.nrowsinbuf

        # Copy to another array
        array2 = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        self.assertTrue(allequal(r2, array2.read()))

        # Assert the number of rows in array
        if common.verbose:
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", r2.shape[0]

        # The next line is commented out because a copy should not
        # keep the same chunkshape anymore.
        # F. Alted 2006-11-27
        #assert array1.chunkshape == array2.chunkshape
        self.assertEqual(r2.shape[0], array2.nrows)

        # Close the file
        fileh.close()
        os.remove(file)

    def _test02_indexclosef(self):
        """Checking CArray.copy() method with indexes (close file version)"""

        if common.verbose:
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
        array1.nrowsinbuf = self.nrowsinbuf

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

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        self.assertEqual(array1.chunkshape, array2.chunkshape)
        self.assertTrue(allequal(r2, array2.read()))

        # Assert the number of rows in array
        if common.verbose:
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", r2.shape[0]
        self.assertEqual(r2.shape[0], array2.nrows)

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
    narows = 1000*1000   # each array will have 1 million entries
    #narows = 1000        # for testing only
    nanumber = 1000*3    # That should account for more than 2**31-1

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = self.fileh = openFile(self.file, "a")
        # Create an CArray
        shape = (self.narows*self.nanumber,)
        array = fileh.createCArray(fileh.root, 'array',
                                   Int8Atom(), shape,
                                   filters=Filters(complib='lzo',
                                                   complevel=1))

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
        common.cleanup(self)

    #----------------------------------------

    def test01_basiccheck(self):
        "Some basic checks for carrays exceeding 2**31 rows"

        fileh = self.fileh
        array = fileh.root.array

        if self.close:
            if common.verbose:
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
            if common.verbose:
                print "After re-open"

        # Check how many entries there are in the array
        if common.verbose:
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
        self.assertEqual(array.nrows, nrows)
        # Check shape
        self.assertEqual(array.shape, (nrows,))
        # check the 10 first elements
        self.assertTrue(allequal(array[:10], numpy.arange(10, dtype='int8')))
        # check the 10 last elements
        stop = self.narows%256
        if stop > 127:
            stop -= 256
        start = stop - 10
        self.assertTrue(allequal(array[-10:],
                                 numpy.arange(start, stop, dtype='int8')))


class Rows64bitsTestCase1(Rows64bitsTestCase):
    close = 0

class Rows64bitsTestCase2(Rows64bitsTestCase):
    close = 1

class BigArrayTestCase(common.TempFileMixin, common.PyTablesTestCase):
    shape = (3000000000,)  # more than 2**31-1

    def setUp(self):
        super(BigArrayTestCase, self).setUp()
        # This should be fast since disk space isn't actually allocated,
        # so this case is OK for non-heavy test runs.
        self.h5file.createCArray('/', 'array', Int8Atom(), self.shape)

    def test00_shape(self):
        """Check that the shape doesn't overflow."""
        # See ticket #147.
        self.assertEqual(self.h5file.root.array.shape, self.shape)
        try:
            self.assertEqual(len(self.h5file.root.array), self.shape[0])
        except OverflowError:
            # This can't be avoided in 32-bit platforms.
            self.assertTrue(self.shape[0] > numpy.iinfo(int).max,
                          "Array length overflowed but ``int`` "
                          "is wide enough." )

    def test01_shape_reopen(self):
        """Check that the shape doesn't overflow after reopening."""
        self._reopen('r')
        self.test00_shape()


# Test for default values when creating arrays.
class DfltAtomTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test00_dflt(self):
        "Check that Atom.dflt is honored (string version)."

        # Create a CArray with default values
        self.h5file.createCArray(
            '/', 'bar', StringAtom(itemsize=5, dflt="abdef"), (10,10))

        if self.reopen:
            self._reopen()

        # Check the values
        values = self.h5file.root.bar[:]
        if common.verbose:
            print "Read values:", values
        self.assertTrue(allequal(values,
                                 numpy.array(["abdef"]*100, "S5").reshape(10,10)))

    def test01_dflt(self):
        "Check that Atom.dflt is honored (int version)."

        # Create a CArray with default values
        self.h5file.createCArray('/', 'bar', IntAtom(dflt=1), (10,10))

        if self.reopen:
            self._reopen()

        # Check the values
        values = self.h5file.root.bar[:]
        if common.verbose:
            print "Read values:", values
        self.assertTrue(allequal(values, numpy.ones((10,10), "i4")))

    def test02_dflt(self):
        "Check that Atom.dflt is honored (float version)."

        # Create a CArray with default values
        self.h5file.createCArray('/', 'bar', FloatAtom(dflt=1.134), (10,10))

        if self.reopen:
            self._reopen()

        # Check the values
        values = self.h5file.root.bar[:]
        if common.verbose:
            print "Read values:", values
        self.assertTrue(allequal(values, numpy.ones((10,10), "f8")*1.134))


class DfltAtomNoReopen(DfltAtomTestCase):
    reopen = False

class DfltAtomReopen(DfltAtomTestCase):
    reopen = True


# Test for representation of defaults in atoms. Ticket #212.
class AtomDefaultReprTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test00a_zeros(self):
        "Testing default values.  Zeros (scalar)."
        N = ()
        atom = StringAtom(itemsize=3, shape=N, dflt="")
        ca = self.h5file.createCArray('/', 'test', atom, (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", repr(ca[0])
            print "Defaults-->", repr(ca.atom.dflt)
        self.assertTrue(allequal(ca[0], numpy.zeros(N, 'S3')))
        self.assertTrue(allequal(ca.atom.dflt, numpy.zeros(N, 'S3')))

    def test00b_zeros(self):
        "Testing default values.  Zeros (array)."
        N = 2
        atom = StringAtom(itemsize=3, shape=N, dflt="")
        ca = self.h5file.createCArray('/', 'test', atom, (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", ca[0]
            print "Defaults-->", ca.atom.dflt
        self.assertTrue(allequal(ca[0], numpy.zeros(N, 'S3')))
        self.assertTrue(allequal(ca.atom.dflt, numpy.zeros(N, 'S3')))

    def test01a_values(self):
        "Testing default values.  Ones."
        N = 2
        atom = Int32Atom(shape=N, dflt=1)
        ca = self.h5file.createCArray('/', 'test', atom, (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", ca[0]
            print "Defaults-->", ca.atom.dflt
        self.assertTrue(allequal(ca[0], numpy.ones(N, 'i4')))
        self.assertTrue(allequal(ca.atom.dflt, numpy.ones(N, 'i4')))

    def test01b_values(self):
        "Testing default values.  Generic value."
        N = 2
        generic = 112.32
        atom = Float32Atom(shape=N, dflt=generic)
        ca = self.h5file.createCArray('/', 'test', atom, (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", ca[0]
            print "Defaults-->", ca.atom.dflt
        self.assertTrue(allequal(ca[0], numpy.ones(N, 'f4')*generic))
        self.assertTrue(allequal(ca.atom.dflt, numpy.ones(N, 'f4')*generic))

    def test02a_None(self):
        "Testing default values.  None (scalar)."
        N = ()
        atom = Int32Atom(shape=N, dflt=None)
        ca = self.h5file.createCArray('/', 'test', atom, (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", repr(ca[0])
            print "Defaults-->", repr(ca.atom.dflt)
        self.assertTrue(allequal(ca.atom.dflt, numpy.zeros(N, 'i4')))

    def test02b_None(self):
        "Testing default values.  None (array)."
        N = 2
        atom = Int32Atom(shape=N, dflt=None)
        ca = self.h5file.createCArray('/', 'test', atom, (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", ca[0]
            print "Defaults-->", ca.atom.dflt
        self.assertTrue(allequal(ca.atom.dflt, numpy.zeros(N, 'i4')))


class AtomDefaultReprNoReopen(AtomDefaultReprTestCase):
    reopen = False

class AtomDefaultReprReopen(AtomDefaultReprTestCase):
    reopen = True


class TruncateTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def test(self):
        """Test for unability to truncate Array objects."""
        array1 = self.h5file.createArray('/', 'array1', [0, 2])
        self.assertRaises(TypeError, array1.truncate, 0)


# Test for dealing with multidimensional atoms
class MDAtomTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test01a_assign(self):
        "Assign a row to a (unidimensional) CArray with a MD atom."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,2)), (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign one row
        ca[0] = [[1,3],[4,5]]
        self.assertEqual(ca.nrows, 1)
        if common.verbose:
            print "First row-->", ca[0]
        self.assertTrue(allequal(ca[0], numpy.array([[1,3],[4,5]], 'i4')))

    def test01b_assign(self):
        "Assign several rows to a (unidimensional) CArray with a MD atom."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,2)), (3,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign three rows
        ca[:] = [[[1]], [[2]], [[3]]]   # Simple broadcast
        self.assertEqual(ca.nrows, 3)
        if common.verbose:
            print "Third row-->", ca[2]
        self.assertTrue(allequal(ca[2], numpy.array([[3,3],[3,3]], 'i4')))

    def test02a_assign(self):
        "Assign a row to a (multidimensional) CArray with a MD atom."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,)), (1,3))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign one row
        ca[:] = [[[1,3],[4,5],[7,9]]]
        self.assertEqual(ca.nrows, 1)
        if common.verbose:
            print "First row-->", ca[0]
        self.assertTrue(allequal(ca[0], numpy.array([[1,3],[4,5],[7,9]], 'i4')))

    def test02b_assign(self):
        "Assign several rows to a (multidimensional) CArray with a MD atom."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,)), (3,3))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign three rows
        ca[:] = [[[1,-3],[4,-5],[-7,9]],
                 [[-1,3],[-4,5],[7,-8]],
                 [[-2,3],[-5,5],[7,-9]]]
        self.assertEqual(ca.nrows, 3)
        if common.verbose:
            print "Third row-->", ca[2]
        self.assertTrue(allequal(ca[2],
                                 numpy.array([[-2,3],[-5,5],[7,-9]], 'i4')))

    def test03a_MDMDMD(self):
        "Complex assign of a MD array in a MD CArray with a MD atom."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,4)), (3,2,3))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign values
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2*3*2*4, dtype='i4').reshape((2,3,2,4))
        ca[:] = [a*1, a*2, a*3]
        self.assertEqual(ca.nrows, 3)
        if common.verbose:
            print "Third row-->", ca[2]
        self.assertTrue(allequal(ca[2], a*3))

    def test03b_MDMDMD(self):
        "Complex assign of a MD array in a MD CArray with a MD atom (II)."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,4)), (2,3,3))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign values
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2*3*3*2*4, dtype='i4').reshape((2,3,3,2,4))
        ca[:] = a
        self.assertEqual(ca.nrows, 2)
        if common.verbose:
            print "Third row-->", ca[:,2,...]
        self.assertTrue(allequal(ca[:,2,...], a[:,2,...]))

    def test03c_MDMDMD(self):
        "Complex assign of a MD array in a MD CArray with a MD atom (III)."
        # Create an CArray
        ca = self.h5file.createCArray('/', 'test', Int32Atom((2,4)), (3,1,2))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Assign values
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(3*1*2*2*4, dtype='i4').reshape((3,1,2,2,4))
        ca[:] = a
        self.assertEqual(ca.nrows, 3)
        if common.verbose:
            print "Second row-->", ca[:,:,1,...]
        self.assertTrue(allequal(ca[:,:,1,...], a[:,:,1,...]))


class MDAtomNoReopen(MDAtomTestCase):
    reopen = False

class MDAtomReopen(MDAtomTestCase):
    reopen = True


# Test for building very large MD atoms without defaults.  Ticket #211.
class MDLargeAtomTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test01_create(self):
        "Create a CArray with a very large MD atom."
        N = 2**16      # 4x larger than maximum object header size (64 KB)
        ca = self.h5file.createCArray('/', 'test', Int32Atom(shape=N), (1,))
        if self.reopen:
            self._reopen('a')
            ca = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", ca[0]
        self.assertTrue(allequal(ca[0], numpy.zeros(N, 'i4')))


class MDLargeAtomNoReopen(MDLargeAtomTestCase):
    reopen = False

class MDLargeAtomReopen(MDLargeAtomTestCase):
    reopen = True



#----------------------------------------------------------------------


def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1
    #common.heavy = 1  # uncomment this only for testing purposes

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
        theSuite.addTest(unittest.makeSuite(BloscComprTestCase))
        theSuite.addTest(unittest.makeSuite(BloscShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(Bzip2ComprTestCase))
        theSuite.addTest(unittest.makeSuite(Bzip2ShuffleTestCase))
        theSuite.addTest(unittest.makeSuite(FloatTypeTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexTypeTestCase))
        theSuite.addTest(unittest.makeSuite(StringTestCase))
        theSuite.addTest(unittest.makeSuite(String2TestCase))
        theSuite.addTest(unittest.makeSuite(StringComprTestCase))

        # numarray is now deprecated
        #if numarray_imported:
        #    theSuite.addTest(unittest.makeSuite(NumarrayInt8TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayInt16TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayInt32TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayFloat32TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayFloat64TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayComplex64TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayComplex128TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayComprTestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayOffsetStrideTestCase))

        # Numeric is now deprecated
        #if numeric_imported:
        #    theSuite.addTest(unittest.makeSuite(NumericInt8TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericInt16TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericInt32TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericFloat32TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericFloat64TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericComplex64TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericComplex128TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericComprTestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericOffsetStrideTestCase))

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
        theSuite.addTest(unittest.makeSuite(BigArrayTestCase))
        theSuite.addTest(unittest.makeSuite(DfltAtomNoReopen))
        theSuite.addTest(unittest.makeSuite(DfltAtomReopen))
        theSuite.addTest(unittest.makeSuite(AtomDefaultReprNoReopen))
        theSuite.addTest(unittest.makeSuite(AtomDefaultReprReopen))
        theSuite.addTest(unittest.makeSuite(TruncateTestCase))
        theSuite.addTest(unittest.makeSuite(MDAtomNoReopen))
        theSuite.addTest(unittest.makeSuite(MDAtomReopen))
        theSuite.addTest(unittest.makeSuite(MDLargeAtomNoReopen))
        theSuite.addTest(unittest.makeSuite(MDLargeAtomReopen))
    if common.heavy:
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
