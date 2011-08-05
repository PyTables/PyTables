# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import sys
import unittest
import os
import tempfile

import numpy

from tables import *
from tables.flavor import flavor_to_flavor
from tables.tests import common
from tables.tests.common import (
    typecode, allequal, numeric_imported, numarray_imported)
from tables.utils import byteorders

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
    dtype = 'int32'
    shape = (2,0)
    start = 0
    stop = 10
    step = 1
    length = 1
    chunksize = 5
    nappends = 10
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
        earray = self.fileh.createEArray(group, 'earray1', atom, self.shape,
                                         title, filters = filters,
                                         expectedrows = 1)
        earray.flavor = self.flavor

        # Fill it with rows
        self.rowshape = list(earray.shape)
        self.objsize = self.length
        for i in self.rowshape:
            if i != 0:
                self.objsize *= i
        self.extdim = earray.extdim
        self.objsize *= self.chunksize
        self.rowshape[earray.extdim] = self.chunksize

        if self.type == "string":
            object = numpy.ndarray(buffer="a"*self.objsize,
                                   shape=self.rowshape,
                                   dtype="S%s" % earray.atom.itemsize)
        else:
            object = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object.shape = self.rowshape
        if self.flavor == "numarray":
            object = flavor_to_flavor(object, 'numpy', 'numarray')
        elif self.flavor == "numeric":
            object = Numeric.asarray(object)

        if common.verbose:
            if self.flavor == "numpy":
                print "Object to append -->", object
            else:
                print "Object to append -->", repr(object)
        for i in range(self.nappends):
            if self.type == "string":
                earray.append(object)
            elif self.flavor in ["numarray","numpy"]:
                earray.append(object*i)
            else:
                object = object * i
                # For Numeric arrays, we still have to undo the type upgrade
                earray.append(object.astype(typecode[earray.atom.type]))

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01_iterEArray(self):
        """Checking enlargeable array iterator"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_iterEArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)
        if self.flavor == "numarray":
            object_ = flavor_to_flavor(object_, 'numpy', 'numarray')
        elif self.flavor == "numeric":
            object_ = Numeric.asarray(object_)

        # Read all the array
        for row in earray:
            chunk = int(earray.nrow % self.chunksize)
            if chunk == 0:
                if self.type == "string":
                    object__ = object_
                else:
                    object__ = object_ * (int(earray.nrow) / self.chunksize)
                    if self.flavor == "numeric":
                        object__ = object__.astype(typecode[earray.atom.type])
            object = object__[chunk]
            # The next adds much more verbosity
            if common.verbose and 0:
                print "number of row ==>", earray.nrow
                if hasattr(object, "shape"):
                    print "shape should look as:", object.shape
                print "row in earray ==>", repr(row)
                print "Should look like ==>", repr(object)

            self.assertEqual(self.nappends*self.chunksize, earray.nrows)
            self.assertTrue(allequal(row, object, self.flavor))
            if hasattr(row, "shape"):
                self.assertEqual(len(row.shape), len(self.shape) - 1)
            else:
                # Scalar case
                self.assertEqual(len(self.shape), 1)

            # Check filters:
            if self.compress != earray.filters.complevel and common.verbose:
                print "Error in compress. Class:", self.__class__.__name__
                print "self, earray:", self.compress, earray.filters.complevel
            self.assertEqual(earray.filters.complevel, self.compress)
            if self.compress > 0 and whichLibVersion(self.complib):
                self.assertEqual(earray.filters.complib, self.complib)
            if self.shuffle != earray.filters.shuffle and common.verbose:
                print "Error in shuffle. Class:", self.__class__.__name__
                print "self, earray:", self.shuffle, earray.filters.shuffle
            self.assertEqual(self.shuffle, earray.filters.shuffle)
            if self.fletcher32 != earray.filters.fletcher32 and common.verbose:
                print "Error in fletcher32. Class:", self.__class__.__name__
                print "self, earray:", self.fletcher32, earray.filters.fletcher32
            self.assertEqual(self.fletcher32, earray.filters.fletcher32)

    def test02_sssEArray(self):
        """Checking enlargeable array iterator with (start, stop, step)"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_sssEArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)
        if self.flavor == "numarray":
            object_ = flavor_to_flavor(object_, 'numpy', 'numarray')
        elif self.flavor == "numeric":
            object_ = Numeric.asarray(object_)


        # Read all the array
        for row in earray.iterrows(start=self.start, stop=self.stop,
                                   step=self.step):
            if self.chunksize == 1:
                index = 0
            else:
                index = int(earray.nrow % self.chunksize)
            if self.type == "string":
                object__ = object_
            else:
                object__ = object_ * (int(earray.nrow) / self.chunksize)
                if self.flavor == "numeric":
                    object__ = object__.astype(typecode[earray.atom.type])
            object = object__[index]
            # The next adds much more verbosity
            if common.verbose and 0:
                print "number of row ==>", earray.nrow
                if hasattr(object, "shape"):
                    print "shape should look as:", object.shape
                print "row in earray ==>", repr(row)
                print "Should look like ==>", repr(object)

            self.assertEqual(self.nappends*self.chunksize, earray.nrows)
            self.assertTrue(allequal(row, object, self.flavor))
            if hasattr(row, "shape"):
                self.assertEqual(len(row.shape), len(self.shape) - 1)
            else:
                # Scalar case
                self.assertEqual(len(self.shape), 1)

    def test03_readEArray(self):
        """Checking read() of enlargeable arrays"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_readEArray..." % self.__class__.__name__

        # This conversion made just in case indices are numpy scalars
        if self.start is not None: self.start = long(self.start)
        if self.stop is not None: self.stop = long(self.stop)
        if self.step is not None: self.step = long(self.step)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)
        if self.flavor == "numarray":
            object_ = flavor_to_flavor(object_, 'numpy', 'numarray')
        elif self.flavor == "numeric":
            object_ = Numeric.asarray(object_)

        rowshape = self.rowshape
        rowshape[self.extdim] *= self.nappends
        if self.type == "string":
            object__ = numpy.empty(shape=rowshape, dtype="S%s"%earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)

        if self.flavor == "numarray":
            object__ = flavor_to_flavor(object__, 'numpy', 'numarray')
            # This creates memory crashes
            #object__ = numarray.swapaxes(object__, 0, self.extdim)
            object__.swapaxes(0, self.extdim)
        elif self.flavor == "numeric":
            object__ = Numeric.asarray(object__)
            object__ = Numeric.swapaxes(object__, 0, self.extdim)
        else:
            object__ = object__.swapaxes(0, self.extdim)

        for i in range(self.nappends):
            j = i * self.chunksize
            if self.type == "string":
                object__[j:j+self.chunksize] = object_
            else:
                if self.flavor in ["numarray", "numpy"]:
                    object__[j:j+self.chunksize] = object_ * i
                else:
                    object__[j:j+self.chunksize] = (object_ * i).astype(typecode[earray.atom.type])
        stop = self.stop

        if self.nappends:
            # stop == None means read only the element designed by start
            # (in read() contexts)
            if self.stop == None:
                if self.start == -1:  # corner case
                    stop = earray.nrows
                else:
                    stop = self.start + 1
            # Protection against number of elements less than existing
            #if rowshape[self.extdim] < self.stop or self.stop == 0:
            if rowshape[self.extdim] < stop:
                # self.stop == 0 means last row only in read()
                # and not in [::] slicing notation
                stop = rowshape[self.extdim]
            # do a copy() in order to ensure that len(object._data)
            # actually do a measure of its length
            object = object__[self.start:stop:self.step].copy()
            # Swap the axes again to have normal ordering
            if self.flavor == "numarray":
                object.swapaxes(0, self.extdim)
            elif self.flavor == "numpy":
                object = object.swapaxes(0, self.extdim)
            elif self.flavor == "numeric":
                object = Numeric.swapaxes(object, 0, self.extdim)
        else:
            object = numpy.empty(shape=self.shape, dtype=self.dtype)
            if self.flavor == "numarray":
                object = numarray.asarray(object)
            elif self.flavor == "numeric":
                object = Numeric.asarray(object)

        # Read all the array
        try:
            row = earray.read(self.start, self.stop, self.step)
        except IndexError:
            row = numpy.empty(shape=self.shape, dtype=self.dtype)
            if self.flavor == "numarray":
                row = numarray.asarray(row)
            elif self.flavor == "numeric":
                row = Numeric.asarray(row)

        if common.verbose:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(row)
            print "Should look like ==>", repr(object)

        self.assertEqual(self.nappends*self.chunksize, earray.nrows)
        self.assertTrue(allequal(row, object, self.flavor))
        if hasattr(row, "shape"):
            self.assertEqual(len(row.shape), len(self.shape))
            if self.flavor in ("numarray", "numeric"):
                self.assertEqual(row.itemsize(), earray.atom.itemsize)
            elif self.flavor in "numpy":
                self.assertEqual(row.itemsize, earray.atom.itemsize)
        else:
            # Scalar case
            self.assertEqual(len(self.shape), 1)

    def test04_getitemEArray(self):
        """Checking enlargeable array __getitem__ special method"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_getitemEArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            # This conversion made just in case indices are numpy scalars
            if self.start is not None: self.start = long(self.start)
            if self.stop is not None: self.stop = long(self.stop)
            if self.step is not None: self.step = long(self.step)
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "r")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        #earray.nrowsinbuf = 3   # this does not really changes the chunksize
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s"%earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape

        # Additional conversion for the numarray case
        if self.flavor == "numarray":
            object_ = flavor_to_flavor(object_, 'numpy', 'numarray')
            object_.swapaxes(earray.extdim, 0)
        else:
            object_ = object_.swapaxes(earray.extdim, 0)

        rowshape = self.rowshape
        rowshape[self.extdim] *= self.nappends
        if self.type == "string":
            object__ = numpy.empty(shape=rowshape, dtype="S%s"%earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)
            # Additional conversion for the numpy case
        if self.flavor == "numarray":
            object__ = flavor_to_flavor(object__, 'numpy', 'numarray')
            object__.swapaxes(0, earray.extdim)
        else:
            object__ = object__.swapaxes(0, earray.extdim)

        for i in range(self.nappends):
            j = i * self.chunksize
            if self.type == "string":
                object__[j:j+self.chunksize] = object_
            else:
                object__[j:j+self.chunksize] = object_ * i

        stop = self.stop
        if self.nappends:
            # Swap the axes again to have normal ordering
            if self.flavor == "numpy":
                object__ = object__.swapaxes(0, self.extdim)
            else:
                object__.swapaxes(0, self.extdim)
            # do a copy() in order to ensure that len(object._data)
            # actually do a measure of its length
            object = object__.__getitem__(self.slices).copy()
        else:
            object = numpy.empty(shape=self.shape, dtype=self.dtype)

        if self.flavor == "numarray":
            # Convert the object to numarray
            object = numarray.asarray(object)
        elif self.flavor == "numeric":
            # Convert the object to Numeric
            object = Numeric.asarray(object)

        # Read all the array
        try:
            row = earray.__getitem__(self.slices)
        except IndexError:
            if self.flavor == "numarray":
                row = numpy.asarray(row)
            elif self.flavor == "numeric":
                row = Numeric.asarray(row)

        if common.verbose:
            print "Object read:\n", repr(row)
            print "Should look like:\n", repr(object)
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", row.shape
                print "shape should look as:", object.shape

        self.assertEqual(self.nappends*self.chunksize, earray.nrows)
        self.assertTrue(allequal(row, object, self.flavor))
        if not hasattr(row, "shape"):
            # Scalar case
            self.assertEqual(len(self.shape), 1)


    def test05_setitemEArray(self):
        """Checking enlargeable array __setitem__ special method"""

        rootgroup = self.rootgroup
        if self.__class__.__name__ == "Ellipsis6EArrayTestCase":
            # We have a problem with test design here, but I think
            # it is not worth the effort to solve it
            # F.Alted 2004-10-27
            return
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_setitemEArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            # This conversion made just in case indices are numpy scalars
            if self.start is not None: self.start = long(self.start)
            if self.stop is not None: self.stop = long(self.stop)
            if self.step is not None: self.step = long(self.step)
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = openFile(self.file, "a")
        earray = self.fileh.getNode("/earray1")

        # Choose a small value for buffer size
        #earray.nrowsinbuf = 3   # this does not really changes the chunksize
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer="a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s"%earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape

        # Additional conversion for the numarray case
        if self.flavor == "numarray":
            object_ = flavor_to_flavor(object_, 'numpy', 'numarray')
            object_.swapaxes(earray.extdim, 0)
        else:
            object_ = object_.swapaxes(earray.extdim, 0)

        rowshape = self.rowshape
        rowshape[self.extdim] *= self.nappends
        if self.type == "string":
            object__ = numpy.empty(shape=rowshape, dtype="S%s"%earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)
            # Additional conversion for the numpy case
        if self.flavor == "numarray":
            object__ = flavor_to_flavor(object__, 'numpy', 'numarray')
            object__.swapaxes(0, earray.extdim)
        else:
            object__ = object__.swapaxes(0, earray.extdim)

        for i in range(self.nappends):
            j = i * self.chunksize
            if self.type == "string":
                object__[j:j+self.chunksize] = object_
            else:
                object__[j:j+self.chunksize] = object_ * i
                # Modify the earray
                #earray[j:j+self.chunksize] = object_ * i
                #earray[self.slices] = 1

        stop = self.stop
        if self.nappends:
            # Swap the axes again to have normal ordering
            if self.flavor == "numpy":
                object__ = object__.swapaxes(0, self.extdim)
            else:
                object__.swapaxes(0, self.extdim)
            # do a copy() in order to ensure that len(object._data)
            # actually do a measure of its length
            object = object__.__getitem__(self.slices).copy()
        else:
            object = numpy.empty(shape=self.shape, dtype=self.dtype)

        if self.flavor == "numpy":
            object = numpy.asarray(object)
        elif self.flavor == "numeric":
            object = Numeric.asarray(object)

        if self.type == "string":
            if hasattr(self, "wslice"):
                # These sentences should be equivalent
                #object[self.wslize] = object[self.wslice].pad("xXx")
                #earray[self.wslice] = earray[self.wslice].pad("xXx")
                object[self.wslize] = "xXx"
                earray[self.wslice] = "xXx"
            elif sum(object[self.slices].shape) != 0 :
                #object[:] = object.pad("xXx")
                object[:] = "xXx"
                if self.flavor == "numarray":
                    if object.size() > 0:
                        earray[self.slices] = object
                else:
                    if object.size > 0:
                        earray[self.slices] = object
        else:
            if hasattr(self, "wslice"):
                object[self.wslice] = object[self.wslice] * 2 + 3
                earray[self.wslice] = earray[self.wslice] * 2 + 3
            elif sum(object[self.slices].shape) != 0:
                object = object * 2 + 3
                if reduce(lambda x,y:x*y, object.shape) > 0:
                    earray[self.slices] = earray[self.slices] * 2 + 3
        # Read all the array
        row = earray.__getitem__(self.slices)
        try:
            row = earray.__getitem__(self.slices)
        except IndexError:
            print "IndexError!"
            row = numpy.empty(shape=self.shape, dtype=self.dtype)
            if self.flavor == "numarray":
                row = numarray.asarray(row)
            elif self.flavor == "numeric":
                row = Numeric.asarray(self.shape)

        if common.verbose:
            print "Object read:\n", repr(row)
            print "Should look like:\n", repr(object)
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", row.shape
                print "shape should look as:", object.shape

        self.assertEqual(self.nappends*self.chunksize, earray.nrows)
        self.assertTrue(allequal(row, object, self.flavor))
        if not hasattr(row, "shape"):
            # Scalar case
            self.assertEqual(len(self.shape), 1)


class BasicWriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (0,)
    chunksize = 5
    nappends = 10
    step = 1
    #wslice = slice(1,nappends,2)
    wslice = 1  # single element case

class Basic2WriteTestCase(BasicTestCase):
    type = 'int32'
    dtype = 'i4'
    shape = (0,)
    chunksize = 5
    nappends = 10
    step = 1
    wslice = slice(chunksize-2, nappends, 2)  # range of elements
    reopen = 0  # This case does not reopen files

class EmptyEArrayTestCase(BasicTestCase):
    type = 'int32'
    dtype = numpy.dtype('int32')
    shape = (2, 0)
    chunksize = 5
    nappends = 0
    start = 0
    stop = 10
    step = 1

class NP_EmptyEArrayTestCase(BasicTestCase):
    type = 'int32'
    dtype = numpy.dtype('()int32')
    shape = (2, 0)
    chunksize = 5
    nappends = 0
    # numarray is now deprecated
    #if numarray_imported:
    #    start = numpy.uint8(0)
    #    stop = numpy.uint32(10)
    #    step = numpy.int64(1)

class Empty2EArrayTestCase(BasicTestCase):
    type = 'int32'
    dtype = 'int32'
    shape = (2, 0)
    chunksize = 5
    nappends = 0
    start = 0
    stop = 10
    step = 1
    reopen = 0  # This case does not reopen files

class SlicesEArrayTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    type = 'int32'
    shape = (2, 0)
    chunksize = 5
    nappends = 2
    slices = (slice(1,2,1), slice(1,3,1))

class Slices2EArrayTestCase(BasicTestCase):
    compress = 1
    complib = "blosc"
    type = 'int32'
    shape = (2, 0, 4)
    chunksize = 5
    nappends = 20
    slices = (slice(1,2,1), slice(None, None, None), slice(1,4,2))

class EllipsisEArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0)
    chunksize = 5
    nappends = 2
    #slices = (slice(1,2,1), Ellipsis)
    slices = (Ellipsis, slice(1,2,1))

class Ellipsis2EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0, 4)
    chunksize = 5
    nappends = 20
    slices = (slice(1,2,1), Ellipsis, slice(1,4,2))

class Slices3EArrayTestCase(BasicTestCase):
    compress = 1      # To show the chunks id DEBUG is on
    complib = "blosc"
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), slice(0, None, None), slice(1,4,2))  # Don't work
    #slices = (slice(None, None, None), slice(0, None, None), slice(1,4,1)) # W
    #slices = (slice(None, None, None), slice(None, None, None), slice(1,4,2)) # N
    #slices = (slice(1,2,1), slice(None, None, None), slice(1,4,2)) # N
    # Disable the failing test temporarily with a working test case
    slices = (slice(1,2,1), slice(1, 4, None), slice(1,4,2)) # Y
    #slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,1)) # Y
    slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2)) # N
    #slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2), slice(0,100,1)) # N

class Slices4EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0, 5, 6)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), slice(0, None, None), slice(1,4,2),
              slice(0,4,2), slice(3,5,2), slice(2,7,1))

class Ellipsis3EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (Ellipsis, slice(0, 4, None), slice(1,4,2))
    slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2), Ellipsis)

class Ellipsis4EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (Ellipsis, slice(0, 4, None), slice(1,4,2))
    slices = (slice(1,2,1), Ellipsis, slice(1,4,2))

class Ellipsis5EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (slice(1,2,1), slice(0, 4, None), Ellipsis)

class Ellipsis6EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 2
    # The next slices gives problems with setting values (test05)
    # This is a problem on the test design, not the Array.__setitem__
    # code, though.
    slices = (slice(1,2,1), slice(0, 4, None), 2, Ellipsis)

class Ellipsis7EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 2
    slices = (slice(1,2,1), slice(0, 4, None), slice(2,3), Ellipsis)

class MD3WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0, 3)
    chunksize = 4
    step = 2

class MD5WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0, 3, 4, 5)  # ok
    #shape = (1, 1, 0, 1)  # Minimum shape that shows problems with HDF5 1.6.1
    #shape = (2, 3, 0, 4, 5)  # Floating point exception (HDF5 1.6.1)
    #shape = (2, 3, 3, 0, 5, 6) # Segmentation fault (HDF5 1.6.1)
    chunksize = 1
    nappends = 1
    start = 1
    stop = 10
    step = 10

class MD6WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 3, 0, 5, 6)
    chunksize = 1
    nappends = 10
    start = 1
    stop = 10
    step = 3

class NP_MD6WriteTestCase(BasicTestCase):
    "Testing NumPy scalars as indexes"
    type = 'int32'
    shape = (2, 3, 3, 0, 5, 6)
    chunksize = 1
    nappends = 10
    # numarray is now deprecated
    #if numarray_imported:
    #    start = numpy.int8(1)
    #    stop = numpy.int32(10)
    #    step = numpy.int64(3)

class MD6WriteTestCase__(BasicTestCase):
    type = 'int32'
    shape = (2, 0)
    chunksize = 1
    nappends = 3
    start = 1
    stop = 3
    step = 1

class MD7WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 3, 4, 5, 0, 3)
    chunksize = 10
    nappends = 1
    start = 1
    stop = 10
    step = 2

class MD10WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (1, 2, 3, 4, 5, 5, 4, 3, 2, 0)
    chunksize = 5
    nappends = 10
    start = -1
    stop = -1
    step = 10

class NP_MD10WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (1, 2, 3, 4, 5, 5, 4, 3, 2, 0)
    chunksize = 5
    nappends = 10
    # numarray is now deprecated
    #if numarray_imported:
    #    start = numpy.int8(-1)
    #    stop = numpy.int64(-1)
    #    step = numpy.uint8(10)

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
    chunksize = 10
    nappends = 100
    start = 3
    stop = 10
    step = 3

class BloscShuffleTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    complib = "blosc"
    chunksize = 100
    nappends = 10
    start = 3
    stop = 10
    step = 7

class LZOComprTestCase(BasicTestCase):
    compress = 1  # sss
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

class Bzip2ComprTestCase(BasicTestCase):
    compress = 1
    complib = "bzip2"
    chunksize = 100
    nappends = 10
    start = 3
    stop = 10
    step = 8

class Bzip2ShuffleTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    complib = "bzip2"
    chunksize = 100
    nappends = 10
    start = 3
    stop = 10
    step = 6

class Fletcher32TestCase(BasicTestCase):
    compress = 0
    fletcher32 = 1
    chunksize = 50
    nappends = 20
    start = 4
    stop = 20
    step = 7

class AllFiltersTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    fletcher32 = 1
    complib = "zlib"
    chunksize = 20  # sss
    nappends = 50
    start = 2
    stop = 99
    step = 6
#     chunksize = 3
#     nappends = 2
#     start = 1
#     stop = 10
#     step = 2

class FloatTypeTestCase(BasicTestCase):
    type = 'float64'
    dtype = 'float64'
    shape = (2,0)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20

class ComplexTypeTestCase(BasicTestCase):
    type = 'complex128'
    dtype = 'complex128'
    shape = (2,0)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20

class StringTestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (2, 0)
    #shape = (2,0,20)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20
    slices = (slice(0,1),slice(1,2))

class String2TestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (0,)
    #shape = (0, 20)
    chunksize = 5
    nappends = 10
    start = 1
    stop = 10
    step = 2

class StringComprTestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (20, 0, 10)
    #shape = (20,0,10,20)
    compr = 1
    #shuffle = 1  # this shouldn't do nothing on chars
    chunksize = 50
    nappends = 10
    start = -1
    stop = 100
    step = 20

class _Numarray1TestCase(BasicTestCase):
    # Setting flavor to Numeric here gives some problems due,
    # most probably to test implementation, not library code
    flavor = "numarray"
    type = "int32"
    shape = (2,0)
    compress = 1
    shuffle = 1
    chunksize = 50
    nappends = 20
    start = -1
    stop = 100
    step = 20

class Numarray2TestCase(BasicTestCase):
    flavor = "numarray"
    # type = Float32 gives some problems on tests. It is *not* a
    # problem with Array.__setitem__(), just with test design
    #type = 'float32'
    type = "float64"
    dtype = "float64"
    shape = (0,)
    compress = 1
    shuffle = 1
    chunksize = 2
    nappends = 1
    start = -1
    stop = 100
    step = 20

class NumarrayComprTestCase(BasicTestCase):
    flavor = "numarray"
    type = "float64"
    dtype = "float64"
    compress = 1
    shuffle = 1
    shape = (0,)
    compr = 1
    chunksize = 2
    nappends = 1
    start = 51
    stop = 100
    step = 7

class StringNumarrayTestCase(BasicTestCase):
    flavor = "numarray"
    type = "string"
    length = 20
    shape = (2, 0)
    #shape = (2,0,20)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20
    slices = (slice(0,1),slice(1,2))

class String2NumarrayTestCase(BasicTestCase):
    flavor = "numarray"
    type = "string"
    length = 20
    shape = (0,)
    #shape = (0, 20)
    chunksize = 5
    nappends = 10
    start = 1
    stop = 10
    step = 2

class StringComprNumarrayTestCase(BasicTestCase):
    flavor = "numarray"
    type = "string"
    length = 20
    shape = (20,0,10)
    #shape = (20,0,10,20)
    compr = 1
    #shuffle = 1  # this shouldn't do nothing on chars
    chunksize = 50
    nappends = 10
    start = -1
    stop = 100
    step = 20

class Numeric1TestCase(BasicTestCase):
    # Setting flavor to Numeric here gives some problems due,
    # most probably, to test implementation, not library code
    #flavor = "numeric"
    type = "int32"
    shape = (2,0)
    compress = 1
    shuffle = 1
    chunksize = 50
    nappends = 20
    start = -1
    stop = 100
    step = 20

class Numeric2TestCase(BasicTestCase):
    flavor = "numeric"
    # type = Float32 gives some problems on tests. It is *not* a
    # problem with Array.__setitem__(), just with test design
    #type = 'float32'
    type = 'float64'
    dtype = 'float64'
    shape = (0,)
    compress = 1
    shuffle = 1
    chunksize = 2
    nappends = 1
    start = -1
    stop = 100
    step = 20

class NumericComprTestCase(BasicTestCase):
    flavor = "numeric"
    type = 'float64'
    dtype = 'float64'
    compress = 1
    shuffle = 1
    shape = (0,)
    compr = 1
    chunksize = 2
    nappends = 1
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
        """Checking earray with offseted numpy strings appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_StringAtom..." % self.__class__.__name__

        earray = self.fileh.createEArray(root, 'strings',
                                         StringAtom(itemsize=3), (0,2,2),
                                         "Array of strings")
        a=numpy.array([[["a","b"],["123", "45"],["45", "123"]]], dtype="S3")
        earray.append(a[:,1:])
        a=numpy.array([[["s", "a"],["ab", "f"],["s", "abc"],["abc", "f"]]])
        earray.append(a[:,2:])

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Second row in earray ==>", row[1].tolist()

        self.assertEqual(earray.nrows, 2)
        self.assertEqual(row[0].tolist(), [["123", "45"],["45", "123"]])
        self.assertEqual(row[1].tolist(), [["s", "abc"],["abc", "f"]])
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 2)

    def test01b_String(self):
        """Checking earray with strided numpy strings appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_StringAtom..." % self.__class__.__name__

        earray = self.fileh.createEArray(root, 'strings',
                                         StringAtom(itemsize=3), (0,2,2),
                                         "Array of strings")
        a=numpy.array([[["a","b"],["123", "45"],["45", "123"]]], dtype="S3")
        earray.append(a[:,::2])
        a=numpy.array([[["s", "a"],["ab", "f"],["s", "abc"],["abc", "f"]]])
        earray.append(a[:,::2])

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Second row in earray ==>", row[1].tolist()

        self.assertEqual(earray.nrows, 2)
        self.assertEqual(row[0].tolist(), [["a","b"],["45", "123"]])
        self.assertEqual(row[1].tolist(), [["s", "a"],["s", "abc"]])
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 2)

    def test02a_int(self):
        """Checking earray with offseted NumPy ints appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_int..." % self.__class__.__name__

        # Create an string atom
        earray = self.fileh.createEArray(root, 'EAtom',
                                         Int32Atom(), (0,3),
                                         "array of ints")
        a=numpy.array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], dtype='int32')
        earray.append(a[2:])  # Create an offset
        a=numpy.array([(1,1,1), (-1,0,0)], dtype='int32')
        earray.append(a[1:])  # Create an offset

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Third row in vlarray ==>", row[2]

        self.assertEqual(earray.nrows, 3)
        self.assertTrue(allequal(row[0], numpy.array([1,1,1], dtype='int32')))
        self.assertTrue(allequal(row[1], numpy.array([0,0,0], dtype='int32')))
        self.assertTrue(allequal(row[2], numpy.array([-1,0,0], dtype='int32')))

    def test02b_int(self):
        """Checking earray with strided NumPy ints appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_int..." % self.__class__.__name__

        earray = self.fileh.createEArray(root, 'EAtom',
                                         Int32Atom(), (0,3),
                                         "array of ints")
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='int32')
        earray.append(a[::3])  # Create an offset
        a = numpy.array([(1,1,1), (-1,0,0)], dtype='int32')
        earray.append(a[::2])  # Create an offset

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Third row in vlarray ==>", row[2]

        self.assertEqual(earray.nrows, 3)
        self.assertTrue(allequal(row[0], numpy.array([0,0,0], dtype='int32')))
        self.assertTrue(allequal(row[1], numpy.array([3,3,3], dtype='int32')))
        self.assertTrue(allequal(row[2], numpy.array([1,1,1], dtype='int32')))


    def test03a_int(self):
        """Checking earray with byteswapped appends (ints)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_int..." % self.__class__.__name__

        earray = self.fileh.createEArray(root, 'EAtom',
                                         Int32Atom(), (0,3),
                                         "array of ints")
        # Add a native ordered array
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='Int32')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4,:]
        swapped = earray[4:,:]
        if common.verbose:
            print "Native rows:", native
            print "Byteorder native rows:", native.dtype.byteorder
            print "Swapped rows:", swapped
            print "Byteorder swapped rows:", swapped.dtype.byteorder

        self.assertTrue(allequal(native, swapped))


    def test03b_float(self):
        """Checking earray with byteswapped appends (floats)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_float..." % self.__class__.__name__

        earray = self.fileh.createEArray(root, 'EAtom',
                                         Float64Atom(), (0,3),
                                         "array of floats")
        # Add a native ordered array
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='Float64')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4,:]
        swapped = earray[4:,:]
        if common.verbose:
            print "Native rows:", native
            print "Byteorder native rows:", native.dtype.byteorder
            print "Swapped rows:", swapped
            print "Byteorder swapped rows:", swapped.dtype.byteorder

        self.assertTrue(allequal(native, swapped))


    def test04a_int(self):
        """Checking earray with byteswapped appends (2, ints)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04a_int..." % self.__class__.__name__

        byteorder = {'little':'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.createEArray(root, 'EAtom',
                                         Int32Atom(), (0,3),
                                         "array of ints",
                                         byteorder=byteorder)
        # Add a native ordered array
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='Int32')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4,:]
        swapped = earray[4:,:]
        if common.verbose:
            print "Byteorder native rows:", byteorders[native.dtype.byteorder]
            print "Byteorder earray on-disk:", earray.byteorder

        self.assertEqual(byteorders[native.dtype.byteorder], sys.byteorder)
        self.assertEqual(earray.byteorder, byteorder)
        self.assertTrue(allequal(native, swapped))


    def test04b_int(self):
        """Checking earray with byteswapped appends (2, ints, reopen)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_int..." % self.__class__.__name__

        byteorder = {'little':'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.createEArray(root, 'EAtom',
                                         Int32Atom(), (0,3),
                                         "array of ints",
                                         byteorder=byteorder)
        self.fileh.close()
        self.fileh = openFile(self.file, "a")
        earray = self.fileh.getNode("/EAtom")
        # Add a native ordered array
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='Int32')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4,:]
        swapped = earray[4:,:]
        if common.verbose:
            print "Byteorder native rows:", byteorders[native.dtype.byteorder]
            print "Byteorder earray on-disk:", earray.byteorder

        self.assertEqual(byteorders[native.dtype.byteorder], sys.byteorder)
        self.assertEqual(earray.byteorder, byteorder)
        self.assertTrue(allequal(native, swapped))


    def test04c_float(self):
        """Checking earray with byteswapped appends (2, floats)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04c_float..." % self.__class__.__name__

        byteorder = {'little':'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.createEArray(root, 'EAtom',
                                         Float64Atom(), (0,3),
                                         "array of floats",
                                         byteorder=byteorder)
        # Add a native ordered array
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='Float64')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4,:]
        swapped = earray[4:,:]
        if common.verbose:
            print "Byteorder native rows:", byteorders[native.dtype.byteorder]
            print "Byteorder earray on-disk:", earray.byteorder

        self.assertEqual(byteorders[native.dtype.byteorder], sys.byteorder)
        self.assertEqual(earray.byteorder, byteorder)
        self.assertTrue(allequal(native, swapped))


    def test04d_float(self):
        """Checking earray with byteswapped appends (2, floats, reopen)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04d_float..." % self.__class__.__name__

        byteorder = {'little':'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.createEArray(root, 'EAtom',
                                         Float64Atom(), (0,3),
                                         "array of floats",
                                         byteorder=byteorder)
        self.fileh.close()
        self.fileh = openFile(self.file, "a")
        earray = self.fileh.getNode("/EAtom")
        # Add a native ordered array
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (3,3,3)], dtype='Float64')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4,:]
        swapped = earray[4:,:]
        if common.verbose:
            print "Byteorder native rows:", byteorders[native.dtype.byteorder]
            print "Byteorder earray on-disk:", earray.byteorder

        self.assertEqual(byteorders[native.dtype.byteorder], sys.byteorder)
        self.assertEqual(earray.byteorder, byteorder)
        self.assertTrue(allequal(native, swapped))


class CopyTestCase(unittest.TestCase):

    def test01_copy(self):
        """Checking EArray.copy() method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        arr = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking EArray.copy() method (where specified)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        arr = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    # numarray is now deprecated
    def _test03_copy(self):
        """Checking EArray.copy() method ('numarray' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        if numarray_imported:
            flavor = "numarray"
        else:
            flavor = "numpy"

        arr = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.flavor = flavor
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test03a_copy(self):
        """Checking EArray.copy() method (Numeric flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        if numeric_imported:
            flavor = "numeric"
        else:
            flavor = "numpy"

        arr = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.flavor = flavor
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test03b_copy(self):
        """Checking EArray.copy() method (python flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        arr = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.flavor = "python"
        array1.append(((456, 2),(3, 457)))

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test03d_copy(self):
        """Checking EArray.copy() method (python string flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03d_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        arr = StringAtom(itemsize=3)
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.flavor = "python"
        array1.append([["456", "2"],["3", "457"]])

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test03e_copy(self):
        """Checking EArray.copy() method (String flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03e_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        arr = StringAtom(itemsize=4)
        array1 = fileh.createEArray(fileh.root, 'array1', arr, (0, 2),
                                    "title array1")
        array1.flavor = "numpy"
        array1.append(numpy.array([["456", "2"],["3", "457"]], dtype="S4"))

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
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking EArray.copy() method (checking title copying)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        atom = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', atom, (0, 2),
                                    "title array1")
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))
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
        """Checking EArray.copy() method (user attributes copied)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        atom = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', atom, (0, 2),
                                    "title array1")
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))
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
        """Checking EArray.copy() method (user attributes not copied)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        atom = Int16Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', atom, (0, 2),
                                    "title array1")
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))
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
        """Checking EArray.copy() method with indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        atom = Int32Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', atom, (0, 2),
                                    "title array1")
        r = numpy.arange(200, dtype='int32')
        r.shape=(100,2)
        array1.append(r)

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
        self.assertEqual(r2.shape[0], array2.nrows)

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_indexclosef(self):
        """Checking EArray.copy() method with indexes (close file version)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_indexclosef..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        atom = Int32Atom()
        array1 = fileh.createEArray(fileh.root, 'array1', atom, (0, 2),
                                    "title array1")
        r = numpy.arange(200, dtype='int32')
        r.shape=(100,2)
        array1.append(r)

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


class TruncateTestCase(unittest.TestCase):

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")

        # Create an EArray
        arr = Int16Atom(dflt=3)
        array1 = self.fileh.createEArray(
            self.fileh.root, 'array1', arr, (0, 2), "title array1")
        # Add a couple of rows
        array1.append(numpy.array([[456, 2],[3, 457]], dtype='Int16'))

    def tearDown(self):
        # Close the file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test00_truncate(self):
        """Checking EArray.truncate() method (truncating to 0 rows)"""

        # Only run this test for HDF5 >= 1.8.0
        if whichLibVersion("hdf5")[1] < "1.8.0":
            return

        array1 = self.fileh.root.array1
        # Truncate to 0 elements
        array1.truncate(0)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertTrue(allequal(
            array1[:], numpy.array([], dtype='Int16').reshape(0,2)))

    def test01_truncate(self):
        """Checking EArray.truncate() method (truncating to 1 rows)"""

        array1 = self.fileh.root.array1
        # Truncate to 1 element
        array1.truncate(1)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertTrue(allequal(
            array1.read(), numpy.array([[456, 2]], dtype='Int16')))

    def test02_truncate(self):
        """Checking EArray.truncate() method (truncating to == self.nrows)"""

        array1 = self.fileh.root.array1
        # Truncate to 2 elements
        array1.truncate(2)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertTrue(allequal(array1.read(),
                                 numpy.array([[456, 2],[3, 457]],
                                 dtype='Int16')))

    def test03_truncate(self):
        """Checking EArray.truncate() method (truncating to > self.nrows)"""

        array1 = self.fileh.root.array1
        # Truncate to 4 elements
        array1.truncate(4)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertEqual(array1.nrows, 4)
        # Check the original values
        self.assertTrue(allequal(array1[:2], numpy.array([[456, 2],[3, 457]],
                                                   dtype='Int16')))
        # Check that the added rows have the default values
        self.assertTrue(allequal(array1[2:], numpy.array([[3, 3],[3, 3]],
                                                dtype='Int16')))


class TruncateOpenTestCase(TruncateTestCase):
    close = 0

class TruncateCloseTestCase(TruncateTestCase):
    close = 1


# The next test should be run only in **common.heavy** mode
class Rows64bitsTestCase(unittest.TestCase):
    narows = 1000*1000   # each numpy object will have 1 million entries
    #narows = 1000   # for testing only
    nanumber = 1000*3    # That should account for more than 2**31-1

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = self.fileh = openFile(self.file, "a")
        # Create an EArray
        array = fileh.createEArray(fileh.root, 'array',
                                   Int8Atom(), (0,),
                                   filters=Filters(complib='lzo',
                                                   complevel=1),
                                   # Specifying expectedrows takes more
                                   # CPU, but less disk
                                   expectedrows=self.narows*self.nanumber)

        # Fill the array
        na = numpy.arange(self.narows, dtype='Int8')
        for i in range(self.nanumber):
            array.append(na)


    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01_basiccheck(self):
        "Some basic checks for earrays exceeding 2**31 rows"

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
            print "Should look like-->", numpy.arange(start, stop,
                                                      dtype='Int8')

        nrows = self.narows*self.nanumber
        # check nrows
        self.assertEqual(array.nrows, nrows)
        # Check shape
        self.assertEqual(array.shape, (nrows,))
        # check the 10 first elements
        self.assertTrue(allequal(array[:10], numpy.arange(10, dtype='Int8')))
        # check the 10 last elements
        stop = self.narows%256
        if stop > 127:
            stop -= 256
        start = stop - 10
        self.assertTrue(allequal(array[-10:],
                                 numpy.arange(start, stop, dtype='Int8')))


class Rows64bitsTestCase1(Rows64bitsTestCase):
    close = 0

class Rows64bitsTestCase2(Rows64bitsTestCase):
    close = 1


# Test for appending zero-sized arrays
class ZeroSizedTestCase(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "a")
        # Create an EArray
        ea = self.fileh.createEArray('/', 'test', Int32Atom(), (3,0))
        # Append a single row
        ea.append([[1], [2], [3]])


    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test01_canAppend(self):
        "Appending zero length array."

        fileh = self.fileh
        ea = fileh.root.test
        np = numpy.empty(shape=(3,0), dtype='int32')
        ea.append(np)
        self.assertEqual(ea.nrows, 1, "The number of rows should be 1.")

    def test02_appendWithWrongShape(self):
        "Appending zero length array with wrong dimension."

        fileh = self.fileh
        ea = fileh.root.test
        np = numpy.empty(shape=(3,0,3), dtype='int32')
        self.assertRaises(ValueError, ea.append, np)


# Test for dealing with multidimensional atoms
class MDAtomTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test01a_append(self):
        "Append a row to a (unidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,2)), (0,))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append one row
        ea.append([[[1,3],[4,5]]])
        self.assertEqual(ea.nrows, 1)
        if common.verbose:
            print "First row-->", ea[0]
        self.assertTrue(allequal(ea[0], numpy.array([[1,3],[4,5]], 'i4')))

    def test01b_append(self):
        "Append several rows to a (unidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,2)), (0,))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        ea.append([[[1]], [[2]], [[3]]])   # Simple broadcast
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[2]
        self.assertTrue(allequal(ea[2], numpy.array([[3,3],[3,3]], 'i4')))

    def test02a_append(self):
        "Append a row to a (multidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,)), (0,3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append one row
        ea.append([[[1,3],[4,5],[7,9]]])
        self.assertEqual(ea.nrows, 1)
        if common.verbose:
            print "First row-->", ea[0]
        self.assertTrue(allequal(ea[0], numpy.array([[1,3],[4,5],[7,9]], 'i4')))

    def test02b_append(self):
        "Append several rows to a (multidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,)), (0,3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        ea.append([[[1,-3],[4,-5],[-7,9]],
                   [[-1,3],[-4,5],[7,-8]],
                   [[-2,3],[-5,5],[7,-9]]])
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[2]
        self.assertTrue(allequal(ea[2],
                                 numpy.array([[-2,3],[-5,5],[7,-9]], 'i4')))

    def test03a_MDMDMD(self):
        "Complex append of a MD array in a MD EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,4)), (0,2,3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2*3*2*4, dtype='i4').reshape((2,3,2,4))
        ea.append([a*1, a*2, a*3])
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[2]
        self.assertTrue(allequal(ea[2], a*3))

    def test03b_MDMDMD(self):
        "Complex append of a MD array in a MD EArray with a MD atom (II)."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,4)), (2,0,3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2*3*2*4, dtype='i4').reshape((2,1,3,2,4))
        ea.append(a*1)
        ea.append(a*2)
        ea.append(a*3)
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[:,2,...]
        self.assertTrue(allequal(ea[:,2,...], a.reshape((2,3,2,4))*3))

    def test03c_MDMDMD(self):
        "Complex append of a MD array in a MD EArray with a MD atom (III)."
        # Create an EArray
        ea = self.h5file.createEArray('/', 'test', Int32Atom((2,4)), (2,3,0))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2*3*2*4, dtype='i4').reshape((2,3,1,2,4))
        ea.append(a*1)
        ea.append(a*2)
        ea.append(a*3)
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[:,:,2,...]
        self.assertTrue(allequal(ea[:,:,2,...], a.reshape((2,3,2,4))*3))


class MDAtomNoReopen(MDAtomTestCase):
    reopen = False

class MDAtomReopen(MDAtomTestCase):
    reopen = True



#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    global numarray_imported
    niter = 1
    #common.heavy = 1  # uncomment this only for testing purposes

    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(Rows64bitsTestCase1))
    #theSuite.addTest(unittest.makeSuite(Rows64bitsTestCase2))
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2WriteTestCase))
        theSuite.addTest(unittest.makeSuite(EmptyEArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Empty2EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(SlicesEArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Slices2EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(EllipsisEArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis2EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis3EArrayTestCase))
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
        # Numeric is now deprecated
        #if numeric_imported:
        #    theSuite.addTest(unittest.makeSuite(Numeric1TestCase))
        #    theSuite.addTest(unittest.makeSuite(Numeric2TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericComprTestCase))
        # numarray is now deprecated
        #if numarray_imported:
        #    theSuite.addTest(unittest.makeSuite(Numarray2TestCase))
        #    theSuite.addTest(unittest.makeSuite(NumarrayComprTestCase))
        #    theSuite.addTest(unittest.makeSuite(StringNumarrayTestCase))
        #    theSuite.addTest(unittest.makeSuite(String2NumarrayTestCase))
        #    theSuite.addTest(unittest.makeSuite(NP_EmptyEArrayTestCase))
        #    theSuite.addTest(unittest.makeSuite(NP_MD6WriteTestCase))
        #    theSuite.addTest(unittest.makeSuite(NP_MD10WriteTestCase))
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
        theSuite.addTest(unittest.makeSuite(TruncateOpenTestCase))
        theSuite.addTest(unittest.makeSuite(TruncateCloseTestCase))
        theSuite.addTest(unittest.makeSuite(ZeroSizedTestCase))
        theSuite.addTest(unittest.makeSuite(MDAtomNoReopen))
        theSuite.addTest(unittest.makeSuite(MDAtomReopen))
    if common.heavy:
        theSuite.addTest(unittest.makeSuite(Slices3EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Slices4EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis4EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis5EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis6EArrayTestCase))
        theSuite.addTest(unittest.makeSuite(Ellipsis7EArrayTestCase))
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

        # numarray is now deprecated
        # XYX
        # The StringComprNumpyTestCase takes muchs more time than
        # its equivalent in numarray StringComprTestCase.
        # This should be further analyzed.
        # F. Alted 2006-02-03
        #if numarray_imported:
        #    theSuite.addTest(unittest.makeSuite(StringComprNumarrayTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
