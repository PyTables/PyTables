# -*- coding: utf-8 -*-

import sys
import unittest
import os
import tempfile

import numpy

from tables import *
from tables.utils import byteorders
from tables.tests import common
from tables.tests.common import allequal

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


class BasicTestCase(unittest.TestCase):
    # Default values
    obj = None
    flavor = "numpy"
    type = 'int32'
    dtype = 'int32'
    shape = (2, 0)
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
        self.fileh = open_file(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()
        if self.reopen:
            # Close the file
            self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        obj = self.obj
        if obj is None:
            if self.type == "string":
                atom = StringAtom(itemsize=self.length)
            else:
                atom = Atom.from_type(self.type)
        else:
            atom = None
        title = self.__class__.__name__
        filters = Filters(complevel=self.compress,
                          complib=self.complib,
                          shuffle=self.shuffle,
                          fletcher32=self.fletcher32)
        earray = self.fileh.create_earray(group, 'earray1',
                                          atom=atom, shape=self.shape,
                                          title=title, filters=filters,
                                          expectedrows=1, obj=obj)
        earray.flavor = self.flavor

        # Fill it with rows
        self.rowshape = list(earray.shape)
        if obj is not None:
            self.rowshape[0] = 0
        self.objsize = self.length
        for i in self.rowshape:
            if i != 0:
                self.objsize *= i
        self.extdim = earray.extdim
        self.objsize *= self.chunksize
        self.rowshape[earray.extdim] = self.chunksize

        if self.type == "string":
            object = numpy.ndarray(buffer=b"a"*self.objsize,
                                   shape=self.rowshape,
                                   dtype="S%s" % earray.atom.itemsize)
        else:
            object = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object.shape = self.rowshape

        if common.verbose:
            if self.flavor == "numpy":
                print "Object to append -->", object
            else:
                print "Object to append -->", repr(object)
        for i in range(self.nappends):
            if self.type == "string":
                earray.append(object)
            else:
                earray.append(object * i)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def _get_shape(self):
        if self.shape is not None:
            shape = self.shape
        else:
            shape = numpy.asarray(self.obj).shape

        return shape

    def test00_attributes(self):
        if self.reopen:
            self.fileh = open_file(self.file, "r")
        obj = self.fileh.get_node("/earray1")

        shape = self._get_shape()
        shape = list(shape)
        shape[self.extdim] = self.chunksize * self.nappends
        if self.obj is not None:
            shape[self.extdim] += len(self.obj)
        shape = tuple(shape)

        self.assertEqual(obj.flavor, self.flavor)
        self.assertEqual(obj.shape, shape)
        self.assertEqual(obj.ndim, len(shape))
        self.assertEqual(obj.nrows, shape[self.extdim])
        self.assertEqual(obj.atom.type, self.type)

    def test01_iterEArray(self):
        """Checking enlargeable array iterator"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_iterEArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = open_file(self.file, "r")
        earray = self.fileh.get_node("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer=b"a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)

        if self.obj is not None:
            initialrows = len(self.obj)
        else:
            initialrows = 0

        shape = self._get_shape()

        # Read all the array
        for idx, row in enumerate(earray):
            if idx < initialrows:
                self.assertTrue(
                    allequal(row, numpy.asarray(self.obj[idx]), self.flavor))
                continue

            chunk = int((earray.nrow - initialrows) % self.chunksize)
            if chunk == 0:
                if self.type == "string":
                    object__ = object_
                else:
                    i = int(earray.nrow - initialrows)
                    object__ = object_ * (i // self.chunksize)

            object = object__[chunk]
            # The next adds much more verbosity
            if common.verbose and 0:
                print "number of row ==>", earray.nrow
                if hasattr(object, "shape"):
                    print "shape should look as:", object.shape
                print "row in earray ==>", repr(row)
                print "Should look like ==>", repr(object)

            self.assertEqual(initialrows + self.nappends * self.chunksize,
                             earray.nrows)
            self.assertTrue(allequal(row, object, self.flavor))
            if hasattr(row, "shape"):
                self.assertEqual(len(row.shape), len(shape) - 1)
            else:
                # Scalar case
                self.assertEqual(len(shape), 1)

            # Check filters:
            if self.compress != earray.filters.complevel and common.verbose:
                print "Error in compress. Class:", self.__class__.__name__
                print "self, earray:", self.compress, earray.filters.complevel
            self.assertEqual(earray.filters.complevel, self.compress)
            if self.compress > 0 and which_lib_version(self.complib):
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

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_sssEArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = open_file(self.file, "r")
        earray = self.fileh.get_node("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer=b"a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)

        if self.obj is not None:
            initialrows = len(self.obj)
        else:
            initialrows = 0

        shape = self._get_shape()

        # Read all the array
        for idx, row in enumerate(earray.iterrows(start=self.start,
                                                  stop=self.stop,
                                                  step=self.step)):
            if idx < initialrows:
                self.assertTrue(
                    allequal(row, numpy.asarray(self.obj[idx]), self.flavor))
                continue

            if self.chunksize == 1:
                index = 0
            else:
                index = int((earray.nrow - initialrows) % self.chunksize)

            if self.type == "string":
                object__ = object_
            else:
                i = int(earray.nrow - initialrows)
                object__ = object_ * (i // self.chunksize)
            object = object__[index]

            # The next adds much more verbosity
            if common.verbose and 0:
                print "number of row ==>", earray.nrow
                if hasattr(object, "shape"):
                    print "shape should look as:", object.shape
                print "row in earray ==>", repr(row)
                print "Should look like ==>", repr(object)

            self.assertEqual(initialrows + self.nappends * self.chunksize,
                             earray.nrows)
            self.assertTrue(allequal(row, object, self.flavor))
            if hasattr(row, "shape"):
                self.assertEqual(len(row.shape), len(shape) - 1)
            else:
                # Scalar case
                self.assertEqual(len(shape), 1)

    def test03_readEArray(self):
        """Checking read() of enlargeable arrays"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_readEArray..." % self.__class__.__name__

        # This conversion made just in case indices are numpy scalars
        if self.start is not None:
            self.start = long(self.start)
        if self.stop is not None:
            self.stop = long(self.stop)
        if self.step is not None:
            self.step = long(self.step)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = open_file(self.file, "r")
        earray = self.fileh.get_node("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer=b"a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)

        if self.obj is not None:
            initialrows = len(self.obj)
        else:
            initialrows = 0

        rowshape = self.rowshape
        rowshape[self.extdim] *= (self.nappends + initialrows)
        if self.type == "string":
            object__ = numpy.empty(
                shape=rowshape, dtype="S%s" % earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)

        object__ = object__.swapaxes(0, self.extdim)

        if initialrows:
            object__[0:initialrows] = self.obj

        for i in range(self.nappends):
            j = initialrows + i * self.chunksize
            if self.type == "string":
                object__[j:j + self.chunksize] = object_
            else:
                object__[j:j + self.chunksize] = object_ * i

        stop = self.stop

        if self.nappends:
            # stop == None means read only the element designed by start
            # (in read() contexts)
            if self.stop is None:
                if self.start == -1:  # corner case
                    stop = earray.nrows
                else:
                    stop = self.start + 1
            # Protection against number of elements less than existing
            # if rowshape[self.extdim] < self.stop or self.stop == 0:
            if rowshape[self.extdim] < stop:
                # self.stop == 0 means last row only in read()
                # and not in [::] slicing notation
                stop = rowshape[self.extdim]
            # do a copy() in order to ensure that len(object._data)
            # actually do a measure of its length
            #object = object__[self.start:stop:self.step].copy()
            object = object__[self.start:self.stop:self.step].copy()
            # Swap the axes again to have normal ordering
            if self.flavor == "numpy":
                object = object.swapaxes(0, self.extdim)
        else:
            object = numpy.empty(shape=self.shape, dtype=self.dtype)

        # Read all the array
        try:
            row = earray.read(self.start, self.stop, self.step)
        except IndexError:
            row = numpy.empty(shape=self.shape, dtype=self.dtype)

        if common.verbose:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(row)
            print "Should look like ==>", repr(object)

        self.assertEqual(initialrows + self.nappends * self.chunksize,
                         earray.nrows)
        self.assertTrue(allequal(row, object, self.flavor))

        shape = self._get_shape()
        if hasattr(row, "shape"):
            self.assertEqual(len(row.shape), len(shape))
            if self.flavor == "numpy":
                self.assertEqual(row.itemsize, earray.atom.itemsize)
        else:
            # Scalar case
            self.assertEqual(len(shape), 1)

    def test03_readEArray_out_argument(self):
        """Checking read() of enlargeable arrays"""

        # This conversion made just in case indices are numpy scalars
        if self.start is not None:
            self.start = long(self.start)
        if self.stop is not None:
            self.stop = long(self.stop)
        if self.step is not None:
            self.step = long(self.step)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = open_file(self.file, "r")
        earray = self.fileh.get_node("/earray1")

        # Choose a small value for buffer size
        earray.nrowsinbuf = 3
        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer=b"a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape
        object_ = object_.swapaxes(earray.extdim, 0)

        if self.obj is not None:
            initialrows = len(self.obj)
        else:
            initialrows = 0

        rowshape = self.rowshape
        rowshape[self.extdim] *= (self.nappends + initialrows)
        if self.type == "string":
            object__ = numpy.empty(
                shape=rowshape, dtype="S%s" % earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)

        object__ = object__.swapaxes(0, self.extdim)

        if initialrows:
            object__[0:initialrows] = self.obj

        for i in range(self.nappends):
            j = initialrows + i * self.chunksize
            if self.type == "string":
                object__[j:j + self.chunksize] = object_
            else:
                object__[j:j + self.chunksize] = object_ * i

        stop = self.stop

        if self.nappends:
            # stop == None means read only the element designed by start
            # (in read() contexts)
            if self.stop is None:
                if self.start == -1:  # corner case
                    stop = earray.nrows
                else:
                    stop = self.start + 1
            # Protection against number of elements less than existing
            # if rowshape[self.extdim] < self.stop or self.stop == 0:
            if rowshape[self.extdim] < stop:
                # self.stop == 0 means last row only in read()
                # and not in [::] slicing notation
                stop = rowshape[self.extdim]
            # do a copy() in order to ensure that len(object._data)
            # actually do a measure of its length
            #object = object__[self.start:stop:self.step].copy()
            object = object__[self.start:self.stop:self.step].copy()
            # Swap the axes again to have normal ordering
            if self.flavor == "numpy":
                object = object.swapaxes(0, self.extdim)
        else:
            object = numpy.empty(shape=self.shape, dtype=self.dtype)

        # Read all the array
        try:
            row = numpy.empty(earray.shape, dtype=earray.atom.dtype)
            slice_obj = [slice(None)] * len(earray.shape)
            #slice_obj[earray.maindim] = slice(self.start, stop, self.step)
            slice_obj[earray.maindim] = slice(self.start, self.stop, self.step)
            row = row[slice_obj].copy()
            earray.read(self.start, self.stop, self.step, out=row)
        except IndexError:
            row = numpy.empty(shape=self.shape, dtype=self.dtype)

        if common.verbose:
            if hasattr(object, "shape"):
                print "shape should look as:", object.shape
            print "Object read ==>", repr(row)
            print "Should look like ==>", repr(object)

        self.assertEqual(initialrows + self.nappends * self.chunksize,
                         earray.nrows)
        self.assertTrue(allequal(row, object, self.flavor))

        shape = self._get_shape()
        if hasattr(row, "shape"):
            self.assertEqual(len(row.shape), len(shape))
            if self.flavor == "numpy":
                self.assertEqual(row.itemsize, earray.atom.itemsize)
        else:
            # Scalar case
            self.assertEqual(len(shape), 1)

    def test04_getitemEArray(self):
        """Checking enlargeable array __getitem__ special method"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_getitemEArray..." % self.__class__.__name__

        if not hasattr(self, "slices"):
            # If there is not a slices attribute, create it
            # This conversion made just in case indices are numpy scalars
            if self.start is not None:
                self.start = long(self.start)
            if self.stop is not None:
                self.stop = long(self.stop)
            if self.step is not None:
                self.step = long(self.step)
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = open_file(self.file, "r")
        earray = self.fileh.get_node("/earray1")

        # Choose a small value for buffer size
        # earray.nrowsinbuf = 3   # this does not really changes the chunksize
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer=b"a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape

        object_ = object_.swapaxes(earray.extdim, 0)

        if self.obj is not None:
            initialrows = len(self.obj)
        else:
            initialrows = 0

        rowshape = self.rowshape
        rowshape[self.extdim] *= (self.nappends + initialrows)
        if self.type == "string":
            object__ = numpy.empty(
                shape=rowshape, dtype="S%s" % earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)
            # Additional conversion for the numpy case
        object__ = object__.swapaxes(0, earray.extdim)

        if initialrows:
            object__[0:initialrows] = self.obj

        for i in range(self.nappends):
            j = initialrows + i * self.chunksize
            if self.type == "string":
                object__[j:j + self.chunksize] = object_
            else:
                object__[j:j + self.chunksize] = object_ * i

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

        # Read all the array
        try:
            row = earray.__getitem__(self.slices)
        except IndexError:
            row = numpy.empty(shape=self.shape, dtype=self.dtype)

        if common.verbose:
            print "Object read:\n", repr(row)
            print "Should look like:\n", repr(object)
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", row.shape
                print "shape should look as:", object.shape

        self.assertEqual(initialrows + self.nappends * self.chunksize,
                         earray.nrows)
        self.assertTrue(allequal(row, object, self.flavor))
        if not hasattr(row, "shape"):
            # Scalar case
            self.assertEqual(len(self.shape), 1)

    def test05_setitemEArray(self):
        """Checking enlargeable array __setitem__ special method"""

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
            if self.start is not None:
                self.start = long(self.start)
            if self.stop is not None:
                self.stop = long(self.stop)
            if self.step is not None:
                self.step = long(self.step)
            self.slices = (slice(self.start, self.stop, self.step),)

        # Create an instance of an HDF5 Table
        if self.reopen:
            self.fileh = open_file(self.file, "a")
        earray = self.fileh.get_node("/earray1")

        # Choose a small value for buffer size
        # earray.nrowsinbuf = 3   # this does not really changes the chunksize
        if common.verbose:
            print "EArray descr:", repr(earray)
            print "shape of read array ==>", earray.shape
            print "reopening?:", self.reopen

        # Build the array to do comparisons
        if self.type == "string":
            object_ = numpy.ndarray(buffer=b"a"*self.objsize,
                                    shape=self.rowshape,
                                    dtype="S%s" % earray.atom.itemsize)
        else:
            object_ = numpy.arange(self.objsize, dtype=earray.atom.dtype.base)
            object_.shape = self.rowshape

        object_ = object_.swapaxes(earray.extdim, 0)

        if self.obj is not None:
            initialrows = len(self.obj)
        else:
            initialrows = 0

        rowshape = self.rowshape
        rowshape[self.extdim] *= (self.nappends + initialrows)
        if self.type == "string":
            object__ = numpy.empty(
                shape=rowshape, dtype="S%s" % earray.atom.itemsize)
        else:
            object__ = numpy.empty(shape=rowshape, dtype=self.dtype)
            # Additional conversion for the numpy case
        object__ = object__.swapaxes(0, earray.extdim)

        for i in range(self.nappends):
            j = initialrows + i * self.chunksize
            if self.type == "string":
                object__[j:j + self.chunksize] = object_
            else:
                object__[j:j + self.chunksize] = object_ * i
                # Modify the earray
                # earray[j:j + self.chunksize] = object_ * i
                # earray[self.slices] = 1

        if initialrows:
            object__[0:initialrows] = self.obj

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

        if self.type == "string":
            if hasattr(self, "wslice"):
                # These sentences should be equivalent
                # object[self.wslize] = object[self.wslice].pad("xXx")
                # earray[self.wslice] = earray[self.wslice].pad("xXx")
                object[self.wslize] = "xXx"
                earray[self.wslice] = "xXx"
            elif sum(object[self.slices].shape) != 0:
                # object[:] = object.pad("xXx")
                object[:] = "xXx"
                if object.size > 0:
                    earray[self.slices] = object
        else:
            if hasattr(self, "wslice"):
                object[self.wslice] = object[self.wslice] * 2 + 3
                earray[self.wslice] = earray[self.wslice] * 2 + 3
            elif sum(object[self.slices].shape) != 0:
                object = object * 2 + 3
                if numpy.prod(object.shape) > 0:
                    earray[self.slices] = earray[self.slices] * 2 + 3
        # Read all the array
        row = earray.__getitem__(self.slices)
        try:
            row = earray.__getitem__(self.slices)
        except IndexError:
            print "IndexError!"
            row = numpy.empty(shape=self.shape, dtype=self.dtype)

        if common.verbose:
            print "Object read:\n", repr(row)
            print "Should look like:\n", repr(object)
            if hasattr(object, "shape"):
                print "Original object shape:", self.shape
                print "Shape read:", row.shape
                print "shape should look as:", object.shape

        self.assertEqual(initialrows + self.nappends * self.chunksize,
                         earray.nrows)
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
    # wslice = slice(1,nappends,2)
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


class Basic3WriteTestCase(BasicTestCase):
    obj = [1, 2]
    type = numpy.asarray(obj).dtype.name
    dtype = numpy.asarray(obj).dtype.str
    shape = (0,)
    chunkshape = (5,)
    step = 1
    reopen = 0  # This case does not reopen files


class Basic4WriteTestCase(BasicTestCase):
    obj = numpy.array([1, 2])
    type = obj.dtype.name
    dtype = obj.dtype.str
    shape = None
    chunkshape = (5,)
    step = 1
    reopen = 0  # This case does not reopen files


class Basic5WriteTestCase(BasicTestCase):
    obj = [1, 2]
    type = numpy.asarray(obj).dtype.name
    dtype = numpy.asarray(obj).dtype.str
    shape = (0,)
    chunkshape = (5,)
    step = 1
    reopen = 1  # This case does reopen files


class Basic6WriteTestCase(BasicTestCase):
    obj = numpy.array([1, 2])
    type = obj.dtype.name
    dtype = obj.dtype.str
    shape = None
    chunkshape = (5,)
    step = 1
    reopen = 1  # This case does reopen files


class Basic7WriteTestCase(BasicTestCase):
    obj = [[1, 2], [3, 4]]
    type = numpy.asarray(obj).dtype.name
    dtype = numpy.asarray(obj).dtype.str
    shape = (0, 2)
    chunkshape = (5,)
    step = 1
    reopen = 0  # This case does not reopen files


class Basic8WriteTestCase(BasicTestCase):
    obj = [[1, 2], [3, 4]]
    type = numpy.asarray(obj).dtype.name
    dtype = numpy.asarray(obj).dtype.str
    shape = (0, 2)
    chunkshape = (5,)
    step = 1
    reopen = 1  # This case does reopen files


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
    slices = (slice(1, 2, 1), slice(1, 3, 1))


class Slices2EArrayTestCase(BasicTestCase):
    compress = 1
    complib = "blosc"
    type = 'int32'
    shape = (2, 0, 4)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), slice(None, None, None), slice(1, 4, 2))


class EllipsisEArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0)
    chunksize = 5
    nappends = 2
    # slices = (slice(1,2,1), Ellipsis)
    slices = (Ellipsis, slice(1, 2, 1))


class Ellipsis2EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0, 4)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), Ellipsis, slice(1, 4, 2))


class Slices3EArrayTestCase(BasicTestCase):
    compress = 1      # To show the chunks id DEBUG is on
    complib = "blosc"
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), slice(
        0, None, None), slice(1, 4, 2))  # Don't work
    # slices = (slice(None, None, None), slice(0, None, None), slice(1,4,1)) # W
    # slices = (slice(None, None, None), slice(None, None, None), slice(1,4,2)) # N
    # slices = (slice(1,2,1), slice(None, None, None), slice(1,4,2)) # N
    # Disable the failing test temporarily with a working test case
    slices = (slice(1, 2, 1), slice(1, 4, None), slice(1, 4, 2))  # Y
    # slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,1)) # Y
    slices = (slice(1, 2, 1), slice(0, 4, None), slice(1, 4, 2))  # N
    # slices = (slice(1,2,1), slice(0, 4, None), slice(1,4,2), slice(0,100,1))
    # # N


class Slices4EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0, 5, 6)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), slice(0, None, None), slice(1, 4, 2),
              slice(0, 4, 2), slice(3, 5, 2), slice(2, 7, 1))


class Ellipsis3EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (Ellipsis, slice(0, 4, None), slice(1, 4, 2))
    slices = (slice(1, 2, 1), slice(0, 4, None), slice(1, 4, 2), Ellipsis)


class Ellipsis4EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (Ellipsis, slice(0, 4, None), slice(1, 4, 2))
    slices = (slice(1, 2, 1), Ellipsis, slice(1, 4, 2))


class Ellipsis5EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 20
    slices = (slice(1, 2, 1), slice(0, 4, None), Ellipsis)


class Ellipsis6EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 2
    # The next slices gives problems with setting values (test05)
    # This is a problem on the test design, not the Array.__setitem__
    # code, though.
    slices = (slice(1, 2, 1), slice(0, 4, None), 2, Ellipsis)


class Ellipsis7EArrayTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 3, 4, 0)
    chunksize = 5
    nappends = 2
    slices = (slice(1, 2, 1), slice(0, 4, None), slice(2, 3), Ellipsis)


class MD3WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0, 3)
    chunksize = 4
    step = 2


class MD5WriteTestCase(BasicTestCase):
    type = 'int32'
    shape = (2, 0, 3, 4, 5)  # ok
    # shape = (1, 1, 0, 1)  # Minimum shape that shows problems with HDF5 1.6.1
    # shape = (2, 3, 0, 4, 5)  # Floating point exception (HDF5 1.6.1)
    # shape = (2, 3, 3, 0, 5, 6) # Segmentation fault (HDF5 1.6.1)
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


class ZlibComprTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    start = 3
    # stop = 0   # means last row
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
    shape = (2, 0)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20


class ComplexTypeTestCase(BasicTestCase):
    type = 'complex128'
    dtype = 'complex128'
    shape = (2, 0)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20


class StringTestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (2, 0)
    # shape = (2,0,20)
    chunksize = 5
    nappends = 10
    start = 3
    stop = 10
    step = 20
    slices = (slice(0, 1), slice(1, 2))


class String2TestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (0,)
    # shape = (0, 20)
    chunksize = 5
    nappends = 10
    start = 1
    stop = 10
    step = 2


class StringComprTestCase(BasicTestCase):
    type = "string"
    length = 20
    shape = (20, 0, 10)
    # shape = (20,0,10,20)
    compr = 1
    # shuffle = 1  # this shouldn't do nothing on chars
    chunksize = 50
    nappends = 10
    start = -1
    stop = 100
    step = 20


class SizeOnDiskInMemoryPropertyTestCase(unittest.TestCase):

    def setUp(self):
        self.array_size = (0, 10)
        # set chunkshape so it divides evenly into array_size, to avoid
        # partially filled chunks
        self.chunkshape = (1000, 10)
        # approximate size (in bytes) of non-data portion of hdf5 file
        self.hdf_overhead = 6000
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, mode="w")

    def tearDown(self):
        self.fileh.close()
        # Then, delete the file
        os.remove(self.file)
        common.cleanup(self)

    def create_array(self, complevel):
        filters = Filters(complevel=complevel, complib='blosc')
        self.array = self.fileh.create_earray('/', 'earray', atom=Int32Atom(),
                                              shape=self.array_size,
                                              filters=filters,
                                              chunkshape=self.chunkshape)

    def test_zero_length(self):
        complevel = 0
        self.create_array(complevel)
        self.assertEqual(self.array.size_on_disk, 0)
        self.assertEqual(self.array.size_in_memory, 0)

    # add 10 chunks of data in one append
    def test_no_compression_one_append(self):
        complevel = 0
        self.create_array(complevel)
        self.array.append([tuple(range(10))] * self.chunkshape[0] * 10)
        self.assertEqual(self.array.size_on_disk, 10 * 1000 * 10 * 4)
        self.assertEqual(self.array.size_in_memory, 10 * 1000 * 10 * 4)

    # add 10 chunks of data in two appends
    def test_no_compression_multiple_appends(self):
        complevel = 0
        self.create_array(complevel)
        self.array.append([tuple(range(10))] * self.chunkshape[0] * 5)
        self.array.append([tuple(range(10))] * self.chunkshape[0] * 5)
        self.assertEqual(self.array.size_on_disk, 10 * 1000 * 10 * 4)
        self.assertEqual(self.array.size_in_memory, 10 * 1000 * 10 * 4)

    def test_with_compression(self):
        complevel = 1
        self.create_array(complevel)
        self.array.append([tuple(range(10))] * self.chunkshape[0] * 10)
        file_size = os.stat(self.file).st_size
        self.assertTrue(
            abs(self.array.size_on_disk - file_size) <= self.hdf_overhead)
        self.assertEqual(self.array.size_in_memory, 10 * 1000 * 10 * 4)
        self.assertTrue(self.array.size_on_disk < self.array.size_in_memory)


class OffsetStrideTestCase(unittest.TestCase):
    mode = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, self.mode)
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

        earray = self.fileh.create_earray(root, 'strings',
                                          atom=StringAtom(itemsize=3),
                                          shape=(0, 2, 2),
                                          title="Array of strings")
        a = numpy.array([[["a", "b"], [
                        "123", "45"], ["45", "123"]]], dtype="S3")
        earray.append(a[:, 1:])
        a = numpy.array([[["s", "a"], [
                        "ab", "f"], ["s", "abc"], ["abc", "f"]]])
        earray.append(a[:, 2:])

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Second row in earray ==>", row[1].tolist()

        self.assertEqual(earray.nrows, 2)
        self.assertEqual(row[0].tolist(), [[b"123", b"45"], [b"45", b"123"]])
        self.assertEqual(row[1].tolist(), [[b"s", b"abc"], [b"abc", b"f"]])
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 2)

    def test01b_String(self):
        """Checking earray with strided numpy strings appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_StringAtom..." % self.__class__.__name__

        earray = self.fileh.create_earray(root, 'strings',
                                          atom=StringAtom(itemsize=3),
                                          shape=(0, 2, 2),
                                          title="Array of strings")
        a = numpy.array([[["a", "b"], [
                        "123", "45"], ["45", "123"]]], dtype="S3")
        earray.append(a[:, ::2])
        a = numpy.array([[["s", "a"], [
                        "ab", "f"], ["s", "abc"], ["abc", "f"]]])
        earray.append(a[:, ::2])

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Second row in earray ==>", row[1].tolist()

        self.assertEqual(earray.nrows, 2)
        self.assertEqual(row[0].tolist(), [[b"a", b"b"], [b"45", b"123"]])
        self.assertEqual(row[1].tolist(), [[b"s", b"a"], [b"s", b"abc"]])
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 2)

    def test02a_int(self):
        """Checking earray with offseted NumPy ints appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_int..." % self.__class__.__name__

        # Create an string atom
        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Int32Atom(), shape=(0, 3),
                                          title="array of ints")
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (0, 0, 0)], dtype='int32')
        earray.append(a[2:])  # Create an offset
        a = numpy.array([(1, 1, 1), (-1, 0, 0)], dtype='int32')
        earray.append(a[1:])  # Create an offset

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Third row in vlarray ==>", row[2]

        self.assertEqual(earray.nrows, 3)
        self.assertTrue(allequal(row[
                        0], numpy.array([1, 1, 1], dtype='int32')))
        self.assertTrue(allequal(row[
                        1], numpy.array([0, 0, 0], dtype='int32')))
        self.assertTrue(allequal(row[
                        2], numpy.array([-1, 0, 0], dtype='int32')))

    def test02b_int(self):
        """Checking earray with strided NumPy ints appends"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_int..." % self.__class__.__name__

        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Int32Atom(), shape=(0, 3),
                                          title="array of ints")
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='int32')
        earray.append(a[::3])  # Create an offset
        a = numpy.array([(1, 1, 1), (-1, 0, 0)], dtype='int32')
        earray.append(a[::2])  # Create an offset

        # Read all the rows:
        row = earray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", earray._v_pathname, ":", earray.nrows
            print "Third row in vlarray ==>", row[2]

        self.assertEqual(earray.nrows, 3)
        self.assertTrue(allequal(row[
                        0], numpy.array([0, 0, 0], dtype='int32')))
        self.assertTrue(allequal(row[
                        1], numpy.array([3, 3, 3], dtype='int32')))
        self.assertTrue(allequal(row[
                        2], numpy.array([1, 1, 1], dtype='int32')))

    def test03a_int(self):
        """Checking earray with byteswapped appends (ints)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_int..." % self.__class__.__name__

        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Int32Atom(), shape=(0, 3),
                                          title="array of ints")
        # Add a native ordered array
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='Int32')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4, :]
        swapped = earray[4:, :]
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

        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Float64Atom(), shape=(0, 3),
                                          title="array of floats")
        # Add a native ordered array
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='Float64')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4, :]
        swapped = earray[4:, :]
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

        byteorder = {'little': 'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Int32Atom(), shape=(0, 3),
                                          title="array of ints",
                                          byteorder=byteorder)
        # Add a native ordered array
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='Int32')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4, :]
        swapped = earray[4:, :]
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

        byteorder = {'little': 'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Int32Atom(), shape=(0, 3),
                                          title="array of ints",
                                          byteorder=byteorder)
        self.fileh.close()
        self.fileh = open_file(self.file, "a")
        earray = self.fileh.get_node("/EAtom")
        # Add a native ordered array
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='Int32')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4, :]
        swapped = earray[4:, :]
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

        byteorder = {'little': 'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Float64Atom(), shape=(0, 3),
                                          title="array of floats",
                                          byteorder=byteorder)
        # Add a native ordered array
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='Float64')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4, :]
        swapped = earray[4:, :]
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

        byteorder = {'little': 'big', 'big': 'little'}[sys.byteorder]
        earray = self.fileh.create_earray(root, 'EAtom',
                                          atom=Float64Atom(), shape=(0, 3),
                                          title="array of floats",
                                          byteorder=byteorder)
        self.fileh.close()
        self.fileh = open_file(self.file, "a")
        earray = self.fileh.get_node("/EAtom")
        # Add a native ordered array
        a = numpy.array([(0, 0, 0), (1, 0, 3), (
            1, 1, 1), (3, 3, 3)], dtype='Float64')
        earray.append(a)
        # Change the byteorder of the array
        a = a.byteswap()
        a = a.newbyteorder()
        # Add a byteswapped array
        earray.append(a)

        # Read all the rows:
        native = earray[:4, :]
        swapped = earray[4:, :]
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
        fileh = open_file(file, "w")

        # Create an EArray
        atom = Int16Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.append(numpy.array([[456, 2], [3, 457]], dtype='Int16'))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy it to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            # print "dirs-->", dir(array1), dir(array2)
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
        fileh = open_file(file, "w")

        # Create an EArray
        atom = Int16Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.append(numpy.array([[456, 2], [3, 457]], dtype='Int16'))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy to another location
        group1 = fileh.create_group("/", "group1")
        array2 = array1.copy(group1, 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            array1 = fileh.root.array1
            array2 = fileh.root.group1.array2

        if common.verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            # print "dirs-->", dir(array1), dir(array2)
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

    def test03a_copy(self):
        """Checking EArray.copy() method (python flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        atom = Int16Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.flavor = "python"
        array1.append(((456, 2), (3, 457)))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
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
        self.assertEqual(array1.flavor, array2.flavor)  # Very important here!
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test03b_copy(self):
        """Checking EArray.copy() method (python string flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03d_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        atom = StringAtom(itemsize=3)
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.flavor = "python"
        array1.append([["456", "2"], ["3", "457"]])

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
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
        self.assertEqual(array1.flavor, array2.flavor)  # Very important here!
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(array1.atom.itemsize, array2.atom.itemsize)
        self.assertEqual(array1.title, array2.title)
        self.assertEqual(str(array1.atom), str(array2.atom))

        # Close the file
        fileh.close()
        os.remove(file)

    def test03c_copy(self):
        """Checking EArray.copy() method (String flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03e_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        atom = StringAtom(itemsize=4)
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.flavor = "numpy"
        array1.append(numpy.array([["456", "2"], ["3", "457"]], dtype="S4"))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy to another location
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
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
        self.assertEqual(array1.flavor, array2.flavor)  # Very important here!
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
        fileh = open_file(file, "w")

        # Create an EArray
        atom = Int16Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.append(numpy.array([[456, 2], [3, 457]], dtype='Int16'))
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', title="title array2")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
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
        fileh = open_file(file, "w")

        # Create an EArray
        atom = Int16Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.append(numpy.array([[456, 2], [3, 457]], dtype='Int16'))
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=1)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
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
        fileh = open_file(file, "w")

        # Create an Array
        atom = Int16Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        array1.append(numpy.array([[456, 2], [3, 457]], dtype='Int16'))
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            array1 = fileh.root.array1

        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=0)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
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
        fileh = open_file(file, "w")

        # Create an EArray
        atom = Int32Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        r = numpy.arange(200, dtype='int32')
        r.shape = (100, 2)
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
        fileh = open_file(file, "w")

        # Create an EArray
        atom = Int32Atom()
        array1 = fileh.create_earray(fileh.root, 'array1',
                                     atom=atom, shape=(0, 2),
                                     title="title array1")
        r = numpy.arange(200, dtype='int32')
        r.shape = (100, 2)
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
        fileh = open_file(file, mode="r")
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
        self.fileh = open_file(self.file, "w")

        # Create an EArray
        atom = Int16Atom(dflt=3)
        array1 = self.fileh.create_earray(self.fileh.root, 'array1',
                                          atom=atom, shape=(0, 2),
                                          title="title array1")
        # Add a couple of rows
        array1.append(numpy.array([[456, 2], [3, 457]], dtype='Int16'))

    def tearDown(self):
        # Close the file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test00_truncate(self):
        """Checking EArray.truncate() method (truncating to 0 rows)"""

        array1 = self.fileh.root.array1
        # Truncate to 0 elements
        array1.truncate(0)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = open_file(self.file, mode="r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertTrue(allequal(
            array1[:], numpy.array([], dtype='Int16').reshape(0, 2)))

    def test01_truncate(self):
        """Checking EArray.truncate() method (truncating to 1 rows)"""

        array1 = self.fileh.root.array1
        # Truncate to 1 element
        array1.truncate(1)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = open_file(self.file, mode="r")
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
            self.fileh = open_file(self.file, mode="r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertTrue(allequal(array1.read(),
                                 numpy.array([[456, 2], [3, 457]],
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
            self.fileh = open_file(self.file, mode="r")
            array1 = self.fileh.root.array1

        if common.verbose:
            print "array1-->", array1.read()

        self.assertEqual(array1.nrows, 4)
        # Check the original values
        self.assertTrue(allequal(array1[:2], numpy.array([[456, 2], [3, 457]],
                                                         dtype='Int16')))
        # Check that the added rows have the default values
        self.assertTrue(allequal(array1[2:], numpy.array([[3, 3], [3, 3]],
                                                         dtype='Int16')))


class TruncateOpenTestCase(TruncateTestCase):
    close = 0


class TruncateCloseTestCase(TruncateTestCase):
    close = 1


# The next test should be run only in **common.heavy** mode
class Rows64bitsTestCase(unittest.TestCase):
    narows = 1000 * 1000   # each numpy object will have 1 million entries
    # narows = 1000   # for testing only
    nanumber = 1000 * 3    # That should account for more than 2**31-1

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = self.fileh = open_file(self.file, "a")
        # Create an EArray
        array = fileh.create_earray(fileh.root, 'array',
                                    atom=Int8Atom(), shape=(0,),
                                    filters=Filters(complib='lzo',
                                                    complevel=1),
                                    # Specifying expectedrows takes more
                                    # CPU, but less disk
                                    expectedrows=self.narows * self.nanumber)

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
                print "Entries:", array.nrows / (1000 * 1000), "Millions"
                print "Shape:", array.shape
            # Close the file
            fileh.close()
            # Re-open the file
            fileh = self.fileh = open_file(self.file)
            array = fileh.root.array
            if common.verbose:
                print "After re-open"

        # Check how many entries there are in the array
        if common.verbose:
            print "Entries:", array.nrows, type(array.nrows)
            print "Entries:", array.nrows / (1000 * 1000), "Millions"
            print "Shape:", array.shape
            print "Last 10 elements-->", array[-10:]
            stop = self.narows % 256
            if stop > 127:
                stop -= 256
            start = stop - 10
            print "Should look like-->", numpy.arange(start, stop,
                                                      dtype='Int8')

        nrows = self.narows * self.nanumber
        # check nrows
        self.assertEqual(array.nrows, nrows)
        # Check shape
        self.assertEqual(array.shape, (nrows,))
        # check the 10 first elements
        self.assertTrue(allequal(array[:10], numpy.arange(10, dtype='Int8')))
        # check the 10 last elements
        stop = self.narows % 256
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
        self.fileh = open_file(self.file, "a")
        # Create an EArray
        ea = self.fileh.create_earray('/', 'test',
                                      atom=Int32Atom(), shape=(3, 0))
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
        np = numpy.empty(shape=(3, 0), dtype='int32')
        ea.append(np)
        self.assertEqual(ea.nrows, 1, "The number of rows should be 1.")

    def test02_appendWithWrongShape(self):
        "Appending zero length array with wrong dimension."

        fileh = self.fileh
        ea = fileh.root.test
        np = numpy.empty(shape=(3, 0, 3), dtype='int32')
        self.assertRaises(ValueError, ea.append, np)


# Test for dealing with multidimensional atoms
class MDAtomTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test01a_append(self):
        "Append a row to a (unidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test',
                                       atom=Int32Atom((2, 2)), shape=(0,))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append one row
        ea.append([[[1, 3], [4, 5]]])
        self.assertEqual(ea.nrows, 1)
        if common.verbose:
            print "First row-->", ea[0]
        self.assertTrue(allequal(ea[0], numpy.array([[1, 3], [4, 5]], 'i4')))

    def test01b_append(self):
        "Append several rows to a (unidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test',
                                       atom=Int32Atom((2, 2)), shape=(0,))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        ea.append([[[1]], [[2]], [[3]]])   # Simple broadcast
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[2]
        self.assertTrue(allequal(ea[2], numpy.array([[3, 3], [3, 3]], 'i4')))

    def test02a_append(self):
        "Append a row to a (multidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test',
                                       atom=Int32Atom((2,)), shape=(0, 3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append one row
        ea.append([[[1, 3], [4, 5], [7, 9]]])
        self.assertEqual(ea.nrows, 1)
        if common.verbose:
            print "First row-->", ea[0]
        self.assertTrue(allequal(ea[0], numpy.array(
            [[1, 3], [4, 5], [7, 9]], 'i4')))

    def test02b_append(self):
        "Append several rows to a (multidimensional) EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test',
                                       atom=Int32Atom((2,)), shape=(0, 3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        ea.append([[[1, -3], [4, -5], [-7, 9]],
                   [[-1, 3], [-4, 5], [7, -8]],
                   [[-2, 3], [-5, 5], [7, -9]]])
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[2]
        self.assertTrue(allequal(
            ea[2], numpy.array([[-2, 3], [-5, 5], [7, -9]], 'i4')))

    def test03a_MDMDMD(self):
        "Complex append of a MD array in a MD EArray with a MD atom."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test', atom=Int32Atom((2, 4)),
                                       shape=(0, 2, 3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2 * 3*2*4, dtype='i4').reshape((2, 3, 2, 4))
        ea.append([a * 1, a*2, a*3])
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[2]
        self.assertTrue(allequal(ea[2], a * 3))

    def test03b_MDMDMD(self):
        "Complex append of a MD array in a MD EArray with a MD atom (II)."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test', atom=Int32Atom((2, 4)),
                                       shape=(2, 0, 3))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2 * 3*2*4, dtype='i4').reshape((2, 1, 3, 2, 4))
        ea.append(a * 1)
        ea.append(a * 2)
        ea.append(a * 3)
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[:, 2, ...]
        self.assertTrue(allequal(ea[:, 2, ...], a.reshape((2, 3, 2, 4))*3))

    def test03c_MDMDMD(self):
        "Complex append of a MD array in a MD EArray with a MD atom (III)."
        # Create an EArray
        ea = self.h5file.create_earray('/', 'test', atom=Int32Atom((2, 4)),
                                       shape=(2, 3, 0))
        if self.reopen:
            self._reopen('a')
            ea = self.h5file.root.test
        # Append three rows
        # The shape of the atom should be added at the end of the arrays
        a = numpy.arange(2 * 3*2*4, dtype='i4').reshape((2, 3, 1, 2, 4))
        ea.append(a * 1)
        ea.append(a * 2)
        ea.append(a * 3)
        self.assertEqual(ea.nrows, 3)
        if common.verbose:
            print "Third row-->", ea[:, :, 2, ...]
        self.assertTrue(allequal(ea[:, :, 2, ...], a.reshape((2, 3, 2, 4))*3))


class MDAtomNoReopen(MDAtomTestCase):
    reopen = False


class MDAtomReopen(MDAtomTestCase):
    reopen = True


class AccessClosedTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def setUp(self):
        super(AccessClosedTestCase, self).setUp()
        self.array = self.h5file.create_earray(self.h5file.root, 'array',
                                               atom=Int32Atom(), shape=(0, 10))
        self.array.append(numpy.zeros((10, 10)))

    def test_read(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.read)

    def test_getitem(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.__getitem__, 0)

    def test_setitem(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.__setitem__, 0, 0)

    def test_append(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.append,
                          numpy.zeros((10, 10)))


class TestCreateEArrayArgs(common.TempFileMixin, common.PyTablesTestCase):
    obj = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    where = '/'
    name = 'earray'
    atom = Atom.from_dtype(obj.dtype)
    shape = (0,) + obj.shape[1:]
    title = 'title'
    filters = None
    expectedrows = 1000
    chunkshape = (1, 2)
    byteorder = None
    createparents = False

    def test_positional_args_01(self):
        self.h5file.create_earray(self.where, self.name,
                                  self.atom, self.shape,
                                  self.title, self.filters,
                                  self.expectedrows, self.chunkshape)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.nrows, 0)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)

    def test_positional_args_02(self):
        ptarr = self.h5file.create_earray(self.where, self.name,
                                          self.atom, self.shape,
                                          self.title,
                                          self.filters,
                                          self.expectedrows,
                                          self.chunkshape)
        ptarr.append(self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_positional_args_obj(self):
        self.h5file.create_earray(self.where, self.name,
                                  None, None,
                                  self.title,
                                  self.filters,
                                  self.expectedrows,
                                  self.chunkshape,
                                  self.byteorder,
                                  self.createparents,
                                  self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj(self):
        self.h5file.create_earray(self.where, self.name, title=self.title,
                                  chunkshape=self.chunkshape,
                                  obj=self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_atom_shape_01(self):
        ptarr = self.h5file.create_earray(self.where, self.name,
                                          title=self.title,
                                          chunkshape=self.chunkshape,
                                          atom=self.atom, shape=self.shape)
        ptarr.append(self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_atom_shape_02(self):
        ptarr = self.h5file.create_earray(self.where, self.name,
                                          title=self.title,
                                          chunkshape=self.chunkshape,
                                          atom=self.atom, shape=self.shape)
        #ptarr.append(self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.nrows, 0)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)

    def test_kwargs_obj_atom(self):
        ptarr = self.h5file.create_earray(self.where, self.name,
                                          title=self.title,
                                          chunkshape=self.chunkshape,
                                          obj=self.obj,
                                          atom=self.atom)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_shape(self):
        ptarr = self.h5file.create_earray(self.where, self.name,
                                          title=self.title,
                                          chunkshape=self.chunkshape,
                                          obj=self.obj,
                                          shape=self.shape)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_atom_shape(self):
        ptarr = self.h5file.create_earray(self.where, self.name,
                                          title=self.title,
                                          chunkshape=self.chunkshape,
                                          obj=self.obj,
                                          atom=self.atom,
                                          shape=self.shape)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.obj.shape)
        self.assertEqual(ptarr.nrows, self.obj.shape[0])
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertEqual(ptarr.chunkshape, self.chunkshape)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_atom_error(self):
        atom = Atom.from_dtype(numpy.dtype('complex'))
        #shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_earray,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=atom)

    def test_kwargs_obj_shape_error(self):
        #atom = Atom.from_dtype(numpy.dtype('complex'))
        shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_earray,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          shape=shape)

    def test_kwargs_obj_atom_shape_error_01(self):
        atom = Atom.from_dtype(numpy.dtype('complex'))
        #shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_earray,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=atom,
                          shape=self.shape)

    def test_kwargs_obj_atom_shape_error_02(self):
        #atom = Atom.from_dtype(numpy.dtype('complex'))
        shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_earray,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=self.atom,
                          shape=shape)

    def test_kwargs_obj_atom_shape_error_03(self):
        atom = Atom.from_dtype(numpy.dtype('complex'))
        shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_earray,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=atom,
                          shape=shape)


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    # common.heavy = 1  # uncomment this only for testing purposes

    # theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    # theSuite.addTest(unittest.makeSuite(Rows64bitsTestCase1))
    # theSuite.addTest(unittest.makeSuite(Rows64bitsTestCase2))
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2WriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic3WriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic4WriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic5WriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic6WriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic7WriteTestCase))
        theSuite.addTest(unittest.makeSuite(Basic8WriteTestCase))
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
        theSuite.addTest(unittest.makeSuite(
            SizeOnDiskInMemoryPropertyTestCase))
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
        theSuite.addTest(unittest.makeSuite(AccessClosedTestCase))
        theSuite.addTest(unittest.makeSuite(TestCreateEArrayArgs))
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

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
