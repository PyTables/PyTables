# -*- coding: latin-1 -*-

import sys
import unittest
import os
import tempfile

import numpy

from tables import *
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

class C:
    c = (3,4.5)

class BasicTestCase(unittest.TestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    flavor = "numpy"

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        vlarray = self.fileh.createVLArray(group, 'vlarray1',
                                           Int32Atom(),
                                           "ragged array if ints",
                                           filters = filters,
                                           expectedsizeinMB = 1)
        vlarray.flavor = self.flavor

        # Fill it with 5 rows
        vlarray.append([1, 2])
        if self.flavor == "numarray":
            vlarray.append(numarray.array([3, 4, 5], type='Int32'))
            vlarray.append(numarray.array([], type='Int32'))    # Empty entry
        elif self.flavor == "numpy":
            vlarray.append(numpy.array([3, 4, 5], dtype='int32'))
            vlarray.append(numpy.array([], dtype='int32'))     # Empty entry
        elif self.flavor == "numeric":
            vlarray.append(Numeric.array([3, 4, 5], typecode='i'))
            vlarray.append(Numeric.array([], typecode='i'))     # Empty entry
        elif self.flavor == "python":
            vlarray.append((3, 4, 5))
            vlarray.append(())         # Empty entry
        vlarray.append([6, 7, 8, 9])
        vlarray.append([10, 11, 12, 13, 14])

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01_read(self):
        """Checking vlarray read"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_read..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.getNode("/vlarray1")

        # Choose a small value for buffer size
        vlarray.nrowsinbuf = 3
        # Read some rows
        row = vlarray.read(0)[0]
        row2 = vlarray.read(2)[0]
        if common.verbose:
            print "Flavor:", vlarray.flavor
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row

        nrows = 5
        self.assertEqual(nrows, vlarray.nrows)
        if self.flavor == "numarray":
            self.assertTrue(
                allequal(row, numarray.array([1, 2], type='Int32'), self.flavor))
            self.assertTrue(
                allequal(row2, numarray.array([], type='Int32'), self.flavor))
        elif self.flavor == "numpy":
            self.assertEqual(type(row), numpy.ndarray)
            self.assertTrue(
                allequal(row, numpy.array([1, 2], dtype='int32'), self.flavor))
            self.assertTrue(
                allequal(row2, numpy.array([], dtype='int32'), self.flavor))
        elif self.flavor == "numeric":
            self.assertEqual(type(row), type(Numeric.array([1, 2])))
            # The next two lines has been corrected by Ciro Catutto
            # (2004-04-20)
            self.assertTrue(allequal(row, (1, 2), self.flavor))
            self.assertTrue(
                allequal(row2, Numeric.array([], typecode='i'), self.flavor))
        elif self.flavor == "python":
            self.assertEqual(row, [1, 2])
            self.assertEqual(row2, [])
        self.assertEqual(len(row), 2)

        # Check filters:
        if self.compress != vlarray.filters.complevel and common.verbose:
            print "Error in compress. Class:", self.__class__.__name__
            print "self, vlarray:", self.compress, vlarray.filters.complevel
        self.assertEqual(vlarray.filters.complevel, self.compress)
        if self.compress > 0 and whichLibVersion(self.complib):
            self.assertEqual(vlarray.filters.complib, self.complib)
        if self.shuffle != vlarray.filters.shuffle and common.verbose:
            print "Error in shuffle. Class:", self.__class__.__name__
            print "self, vlarray:", self.shuffle, vlarray.filters.shuffle
        self.assertEqual(self.shuffle, vlarray.filters.shuffle)
        if self.fletcher32 != vlarray.filters.fletcher32 and common.verbose:
            print "Error in fletcher32. Class:", self.__class__.__name__
            print "self, vlarray:", self.fletcher32, vlarray.filters.fletcher32
        self.assertEqual(self.fletcher32, vlarray.filters.fletcher32)


    def test02a_getitem(self):
        """Checking vlarray __getitem__ (slices)"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_getitem..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.getNode("/vlarray1")

        rows = [[1, 2], [3,4,5], [], [6, 7, 8, 9], [10, 11, 12, 13, 14]]

        slices = [
            slice(None, None, None), slice(1,1,1), slice(30, None, None),
            slice(0, None, None), slice(3, None, 1), slice(3, None, 2),
            slice(None, 1, None), slice(None, 2, 1), slice(None, 30, 2),
            slice(None, None, 1), slice(None, None, 2), slice(None, None, 3),
                  ]
        for slc in slices:
            # Read the rows in slc
            rows2 = vlarray[slc]
            rows1 = rows[slc]
            rows1f = []
            if common.verbose:
                print "Flavor:", vlarray.flavor
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "Original rows ==>", rows1
                print "Rows read in vlarray ==>", rows2

            if self.flavor == "numarray":
                for val in rows1:
                    rows1f.append(numarray.array(val, type='Int32'))
                for i in range(len(rows1f)):
                    self.assertTrue(allequal(rows2[i], rows1f[i], self.flavor))
            elif self.flavor == "numpy":
                for val in rows1:
                    rows1f.append(numpy.array(val, dtype='int32'))
                for i in range(len(rows1f)):
                    self.assertTrue(allequal(rows2[i], rows1f[i], self.flavor))
            elif self.flavor == "numeric":
                for val in rows1:
                    rows1f.append(Numeric.array(val, typecode='i'))
                for i in range(len(rows1f)):
                    self.assertTrue(allequal(rows2[i], rows1f[i], self.flavor))
            elif self.flavor == "python":
                    self.assertEqual(rows2, rows1)


    def test02b_getitem(self):
        """Checking vlarray __getitem__ (scalars)"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_getitem..." % self.__class__.__name__

        if self.flavor != "numpy":
            # This test is only valid for NumPy
            return

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.getNode("/vlarray1")

        # Get a numpy array of objects
        rows = numpy.array(vlarray[:], dtype=numpy.object)

        for slc in [ 0, numpy.array(1), 2, numpy.array([3]), [4] ]:
            # Read the rows in slc
            rows2 = vlarray[slc]
            rows1 = rows[slc]
            if common.verbose:
                print "Flavor:", vlarray.flavor
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "Original rows ==>", rows1
                print "Rows read in vlarray ==>", rows2

            for i in range(len(rows1)):
                self.assertTrue(allequal(rows2[i], rows1[i], self.flavor))


    def test03_append(self):
        """Checking vlarray append"""

        rootgroup = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_append..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.getNode("/vlarray1")
        # Append a new row
        vlarray.append([7, 8, 9, 10])

        # Choose a small value for buffer size
        vlarray.nrowsinbuf = 3
        # Read some rows:
        row1 = vlarray[0]
        row2 = vlarray[2]
        row3 = vlarray[-1]
        if common.verbose:
            print "Flavor:", vlarray.flavor
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row1

        nrows = 6
        self.assertEqual(nrows, vlarray.nrows)
        if self.flavor == "numarray":
            self.assertEqual(
                allequal(row1,
                         numarray.array([1, 2], type='Int32'), self.flavor))
            self.assertEqual(
                allequal(row2, numarray.array([], type='Int32'), self.flavor))
            self.assertEqual(
                allequal(row3, numarray.array([7, 8, 9, 10], type='Int32'),
                         self.flavor))
        elif self.flavor == "numpy":
            self.assertEqual(type(row1), type(numpy.array([1, 2])))
            self.assertTrue(
                allequal(row1, numpy.array([1, 2], dtype='int32'), self.flavor))
            self.assertTrue(
                allequal(row2, numpy.array([], dtype='int32'), self.flavor))
            self.assertTrue(
                allequal(row3, numpy.array([7, 8, 9, 10], dtype='int32'),
                         self.flavor))
        elif self.flavor == "numeric":
            self.assertEqual(type(row1), type(Numeric.array([1, 2])))
            # The next two lines has been corrected by Ciro Catutto
            # (2004-04-20)
            self.assertTrue(allequal(row1, (1, 2), self.flavor))
            self.assertTrue(
                allequal(row2, Numeric.array([], typecode='i'), self.flavor))
            self.assertTrue(
                allequal(row3, Numeric.array([7, 8, 9, 10], typecode='i'),
                         self.flavor))
        elif self.flavor == "python":
            self.assertEqual(row1, [1, 2])
            self.assertEqual(row2, [])
            self.assertEqual(row3, [7, 8, 9, 10])
        self.assertEqual(len(row3), 4)


class BasicNumPyTestCase(BasicTestCase):
    flavor = "numpy"

class BasicNumArrayTestCase(BasicTestCase):
    flavor = "numarray"

class BasicNumericTestCase(BasicTestCase):
    flavor = "numeric"

class BasicPythonTestCase(BasicTestCase):
    flavor = "python"

class ZlibComprTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"

class BloscComprTestCase(BasicTestCase):
    compress = 9
    shuffle = 0
    complib = "blosc"

class BloscShuffleComprTestCase(BasicTestCase):
    compress = 6
    shuffle = 1
    complib = "blosc"

class LZOComprTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"

class Bzip2ComprTestCase(BasicTestCase):
    compress = 1
    complib = "bzip2"

class ShuffleComprTestCase(BasicTestCase):
    compress = 1
    shuffle = 1

class Fletcher32TestCase(BasicTestCase):
    fletcher32 = 1

class AllFiltersTestCase(BasicTestCase):
    compress = 1
    shuffle = 1
    fletcher32 = 1

class TypesTestCase(unittest.TestCase):
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01_StringAtom(self):
        """Checking vlarray with NumPy string atoms ('numpy' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "numpy"
        vlarray.append(numpy.array(["1", "12", "123", "1234", "12345"]))
        vlarray.append(numpy.array(["1", "12345"]))

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array(["1", "12", "123", "123", "123"])))
        self.assertTrue(allequal(row[1], numpy.array(["1", "123"])))
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 2)

    # This test doesn't compile without numarray installed
    def _test01_1_StringAtom(self):
        """Checking vlarray with NumPy string atoms ('numarray' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_1_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "numarray"
        vlarray.append(numpy.array(["1", "12", "123", "1234", "12345"],
                                   dtype="S"))
        vlarray.append(numpy.array(["1", "12345"], dtype="S"))

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray:", row[0]
            print "Should look like:", \
                  strings.array(['1','12','123','123','123'], itemsize=3)

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], strings.array(["1", "12", "123", "123", "123"]),
                                           flavor="numarray"))
        self.assertTrue(
            allequal(row[1], strings.array(["1", "123"]), flavor="numarray"))
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 2)

    def test01a_StringAtom(self):
        """Checking vlarray with NumPy string atoms ('numpy' flavor, strided)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "numpy"
        vlarray.append(numpy.array(["1", "12", "123", "1234", "12345"][::2]))
        vlarray.append(numpy.array(["1", "12345","2", "321"])[::3])

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(allequal(row[0], numpy.array(["1", "123", "123"])))
        self.assertTrue(allequal(row[1], numpy.array(["1", "321"])))
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 2)

    def test01a_2_StringAtom(self):
        """Checking vlarray with NumPy string atoms (NumPy flavor, no conv)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_2_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "numpy"
        vlarray.append(numpy.array(["1", "12", "123", "123"]))
        vlarray.append(numpy.array(["1", "2", "321"]))

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array(["1", "12", "123", "123"])))
        self.assertTrue(allequal(row[1], numpy.array(["1", "2", "321"])))
        self.assertEqual(len(row[0]), 4)
        self.assertEqual(len(row[1]), 3)

    def test01b_StringAtom(self):
        """Checking vlarray with NumPy string atoms (python flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom2',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "python"
        vlarray.append(["1", "12", "123", "1234", "12345"])
        vlarray.append(["1", "12345"])

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Testing String flavor"
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], ["1", "12", "123", "123", "123"])
        self.assertEqual(row[1], ["1", "123"])
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 2)

    def test01c_StringAtom(self):
        """Checking updating vlarray with NumPy string atoms ('numpy' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "numpy"
        vlarray.append(numpy.array(["1", "12", "123", "1234", "12345"]))
        vlarray.append(numpy.array(["1", "12345"]))

        # Modify the rows
        vlarray[0] = numpy.array(["1", "123", "12", "", "12345"])
        vlarray[1] = numpy.array(["44", "4"])  # This should work as well

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array(["1", "123", "12", "", "123"])))
        self.assertTrue(allequal(row[1], numpy.array(["44", "4"], dtype="S3")))
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 2)

    def test01d_StringAtom(self):
        """Checking updating vlarray with string atoms (String flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01d_StringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'stringAtom2',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        vlarray.flavor = "python"
        vlarray.append(["1", "12", "123", "1234", "12345"])
        vlarray.append(["1", "12345"])

        # Modify the rows
        vlarray[0] = ["1", "123", "12", "", "12345"]
        vlarray[1] = ["44", "4"]

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Testing String flavor"
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], ["1", "123", "12", "", "123"])
        self.assertEqual(row[1], ["44", "4"])
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 2)

    def test02_BoolAtom(self):
        """Checking vlarray with boolean atoms"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_BoolAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'BoolAtom',
                                           BoolAtom(),
                                           "Ragged array of Booleans")
        vlarray.append([1,0,3])
        vlarray.append([-1,0])

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(allequal(row[0], numpy.array([1,0,1], dtype='bool')))
        self.assertTrue(allequal(row[1], numpy.array([1,0], dtype='bool')))
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 2)

    def test02b_BoolAtom(self):
        """Checking setting vlarray with boolean atoms"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_BoolAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', 'BoolAtom',
                                           BoolAtom(),
                                           "Ragged array of Booleans")
        vlarray.append([1,0,3])
        vlarray.append([-1,0])

        # Modify the rows
        vlarray[0] = (0,1,3)
        vlarray[1] = (0,-1)

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(allequal(row[0], numpy.array([0,1,1], dtype='bool')))
        self.assertTrue(allequal(row[1], numpy.array([0,1], dtype='bool')))
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 2)

    def test03_IntAtom(self):
        """Checking vlarray with integer atoms"""

        ttypes = ["Int8",
                  "UInt8",
                  "Int16",
                  "UInt16",
                  "Int32",
                  "UInt32",
                  "Int64",
                  #"UInt64",  # Unavailable in some platforms
                  ]
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        for atype in ttypes:
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(atype))
            vlarray.append([1,2,3])
            vlarray.append([-1,0])

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([1,2,3], dtype=atype)))
            self.assertTrue(allequal(row[1], numpy.array([-1,0], dtype=atype)))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test03a_IntAtom(self):
        """Checking vlarray with integer atoms (byteorder swapped)"""

        ttypes = {"Int8": numpy.int8,
                  "UInt8": numpy.uint8,
                  "Int16": numpy.int16,
                  "UInt16": numpy.uint16,
                  "Int32": numpy.int32,
                  "UInt32": numpy.uint32,
                  "Int64": numpy.int64,
                  #"UInt64": numpy.int64,  # Unavailable in some platforms
                  }
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_IntAtom..." % self.__class__.__name__

        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(ttypes[atype]))
            a0 = numpy.array([1,2,3], dtype=atype)
            a0 = a0.byteswap(); a0 = a0.newbyteorder()
            vlarray.append(a0)
            a1 = numpy.array([-1,0], dtype=atype)
            a1 = a1.byteswap(); a1 = a1.newbyteorder()
            vlarray.append(a1)

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([1,2,3], dtype=ttypes[atype])))
            self.assertTrue(
                allequal(row[1], numpy.array([-1,0], dtype=ttypes[atype])))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test03b_IntAtom(self):
        """Checking updating vlarray with integer atoms"""

        ttypes = ["Int8",
                  "UInt8",
                  "Int16",
                  "UInt16",
                  "Int32",
                  "UInt32",
                  "Int64",
                  #"UInt64",  # Unavailable in some platforms
                  ]
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        for atype in ttypes:
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(atype))
            vlarray.append([1,2,3])
            vlarray.append([-1,0])

            # Modify rows
            vlarray[0] = (3,2,1)
            vlarray[1] = (0,-1)

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([3,2,1], dtype=atype)))
            self.assertTrue(allequal(row[1], numpy.array([0,-1], dtype=atype)))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test03c_IntAtom(self):
        """Checking updating vlarray with integer atoms (byteorder swapped)"""

        ttypes = {"Int8": numpy.int8,
                  "UInt8": numpy.uint8,
                  "Int16": numpy.int16,
                  "UInt16": numpy.uint16,
                  "Int32": numpy.int32,
                  "UInt32": numpy.uint32,
                  "Int64": numpy.int64,
                  #"UInt64": numpy.int64,  # Unavailable in some platforms
                  }
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03c_IntAtom..." % self.__class__.__name__

        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(ttypes[atype]))
            a0 = numpy.array([1,2,3], dtype=atype)
            vlarray.append(a0)
            a1 = numpy.array([-1,0], dtype=atype)
            vlarray.append(a1)


            # Modify rows
            a0 = numpy.array([3,2,1], dtype=atype)
            a0 = a0.byteswap(); a0 = a0.newbyteorder()
            vlarray[0] = a0
            a1 = numpy.array([0, -1], dtype=atype)
            a1 = a1.byteswap(); a1 = a1.newbyteorder()
            vlarray[1] = a1

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([3,2,1], dtype=ttypes[atype])))
            self.assertTrue(
                allequal(row[1], numpy.array([0,-1], dtype=ttypes[atype])))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test03d_IntAtom(self):
        """Checking updating vlarray with integer atoms (another byteorder)"""

        ttypes = {"Int8": numpy.int8,
                  "UInt8": numpy.uint8,
                  "Int16": numpy.int16,
                  "UInt16": numpy.uint16,
                  "Int32": numpy.int32,
                  "UInt32": numpy.uint32,
                  "Int64": numpy.int64,
                  #"UInt64": numpy.int64,  # Unavailable in some platforms
                  }
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03d_IntAtom..." % self.__class__.__name__

        byteorder = {'little':'big', 'big': 'little'}[sys.byteorder]
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(ttypes[atype]),
                                               byteorder=byteorder)
            a0 = numpy.array([1,2,3], dtype=atype)
            vlarray.append(a0)
            a1 = numpy.array([-1,0], dtype=atype)
            vlarray.append(a1)


            # Modify rows
            a0 = numpy.array([3,2,1], dtype=atype)
            a0 = a0.byteswap(); a0 = a0.newbyteorder()
            vlarray[0] = a0
            a1 = numpy.array([0, -1], dtype=atype)
            a1 = a1.byteswap(); a1 = a1.newbyteorder()
            vlarray[1] = a1

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            byteorder2 = byteorders[row[0].dtype.byteorder]
            if byteorder2 != "irrelevant":
                self.assertEqual(byteorders[row[0].dtype.byteorder],
                                 sys.byteorder)
                self.assertEqual(vlarray.byteorder, byteorder)
            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([3,2,1], dtype=ttypes[atype])))
            self.assertTrue(
                allequal(row[1], numpy.array([0,-1], dtype=ttypes[atype])))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04_FloatAtom(self):
        """Checking vlarray with floating point atoms"""

        ttypes = ["Float32",
                  "Float64",
                  ]
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_FloatAtom..." % self.__class__.__name__

        for atype in ttypes:
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(atype))
            vlarray.append([1.3,2.2,3.3])
            vlarray.append([-1.3e34,1.e-32])

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([1.3,2.2,3.3], atype)))
            self.assertTrue(allequal(row[1], numpy.array([-1.3e34,1.e-32], atype)))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04a_FloatAtom(self):
        """Checking vlarray with float atoms (byteorder swapped)"""

        ttypes = {"Float32": numpy.float32,
                  "Float64": numpy.float64,
                  }
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04a_FloatAtom..." % self.__class__.__name__

        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(ttypes[atype]))
            a0 = numpy.array([1.3,2.2,3.3], dtype=atype)
            a0 = a0.byteswap(); a0 = a0.newbyteorder()
            vlarray.append(a0)
            a1 = numpy.array([-1.3e34,1.e-32], dtype=atype)
            a1 = a1.byteswap(); a1 = a1.newbyteorder()
            vlarray.append(a1)

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([1.3,2.2,3.3],
                                                         dtype=ttypes[atype])))
            self.assertTrue(allequal(row[1], numpy.array([-1.3e34,1.e-32],
                                                         dtype=ttypes[atype])))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04b_FloatAtom(self):
        """Checking updating vlarray with floating point atoms"""

        ttypes = ["Float32",
                  "Float64",
                  ]
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_FloatAtom..." % self.__class__.__name__

        for atype in ttypes:
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(atype))
            vlarray.append([1.3,2.2,3.3])
            vlarray.append([-1.3e34,1.e-32])

            # Modifiy some rows
            vlarray[0] = (4.3,2.2,4.3)
            vlarray[1] = (-1.1e34,1.3e-32)

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([4.3,2.2,4.3], atype)))
            self.assertTrue(
                allequal(row[1], numpy.array([-1.1e34,1.3e-32], atype)))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04c_FloatAtom(self):
        """Checking updating vlarray with float atoms (byteorder swapped)"""

        ttypes = {"Float32": numpy.float32,
                  "Float64": numpy.float64,
                  }
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04c_FloatAtom..." % self.__class__.__name__

        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(ttypes[atype]))
            a0 = numpy.array([1.3,2.2,3.3], dtype=atype)
            vlarray.append(a0)
            a1 = numpy.array([-1,0], dtype=atype)
            vlarray.append(a1)


            # Modify rows
            a0 = numpy.array([4.3,2.2,4.3], dtype=atype)
            a0 = a0.byteswap(); a0 = a0.newbyteorder()
            vlarray[0] = a0
            a1 = numpy.array([-1.1e34,1.3e-32], dtype=atype)
            a1 = a1.byteswap(); a1 = a1.newbyteorder()
            vlarray[1] = a1

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([4.3,2.2,4.3],
                                                         dtype=ttypes[atype])))
            self.assertTrue(allequal(row[1], numpy.array([-1.1e34,1.3e-32],
                                                         dtype=ttypes[atype])))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04d_FloatAtom(self):
        """Checking updating vlarray with float atoms (another byteorder)"""

        ttypes = {"Float32": numpy.float32,
                  "Float64": numpy.float64,
                  }
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04d_FloatAtom..." % self.__class__.__name__

        byteorder = {'little':'big', 'big': 'little'}[sys.byteorder]
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(ttypes[atype]),
                                               byteorder = byteorder)
            a0 = numpy.array([1.3,2.2,3.3], dtype=atype)
            vlarray.append(a0)
            a1 = numpy.array([-1,0], dtype=atype)
            vlarray.append(a1)


            # Modify rows
            a0 = numpy.array([4.3,2.2,4.3], dtype=atype)
            a0 = a0.byteswap(); a0 = a0.newbyteorder()
            vlarray[0] = a0
            a1 = numpy.array([-1.1e34,1.3e-32], dtype=atype)
            a1 = a1.byteswap(); a1 = a1.newbyteorder()
            vlarray[1] = a1

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.byteorder, byteorder)
            byteorder2 = byteorders[row[0].dtype.byteorder]
            self.assertTrue(byteorders[row[0].dtype.byteorder], sys.byteorder)
            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(allequal(row[0], numpy.array([4.3,2.2,4.3],
                                                         dtype=ttypes[atype])))
            self.assertTrue(allequal(row[1], numpy.array([-1.1e34,1.3e-32],
                                                         dtype=ttypes[atype])))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04_ComplexAtom(self):
        """Checking vlarray with numerical complex atoms"""

        ttypes = ["Complex32",
                  "Complex64",
                  ]
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_ComplexAtom..." % self.__class__.__name__

        for atype in ttypes:
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(atype))
            vlarray.append([(1.3+0j),(0+2.2j),(3.3+3.3j)])
            vlarray.append([(0-1.3e34j),(1.e-32+0j)])

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([(1.3+0j),(0+2.2j),(3.3+3.3j)],
                                             atype)))
            self.assertTrue(
                allequal(row[1], numpy.array([(0-1.3e34j),(1.e-32+0j)], atype)))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test04b_ComplexAtom(self):
        """Checking modifying vlarray with numerical complex atoms"""

        ttypes = ["Complex32",
                  "Complex64",
                  ]
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_ComplexAtom..." % self.__class__.__name__

        for atype in ttypes:
            vlarray = self.fileh.createVLArray('/', atype,
                                               Atom.from_sctype(atype))
            vlarray.append([(1.3+0j),(0+2.2j),(3.3+3.3j)])
            vlarray.append([(0-1.3e34j),(1.e-32+0j)])

            # Modify the rows
            vlarray[0] = ((1.4+0j),(0+4.2j),(3.3+4.3j))
            vlarray[1] = ((4-1.3e34j),(1.e-32+4j))

            if self.reopen:
                name = vlarray._v_pathname
                self.fileh.close()
                self.fileh = openFile(self.file, "a")
                vlarray = self.fileh.getNode(name)

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([(1.4+0j),(0+4.2j),(3.3+4.3j)],
                                             atype)))
            self.assertTrue(
                allequal(row[1], numpy.array([(4-1.3e34j),(1.e-32+4j)], atype)))
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 2)

    def test05_VLStringAtom(self):
        """Checking vlarray with variable length strings"""

        # Skip the test if the default encoding has been mangled.
        if sys.getdefaultencoding() != 'ascii':
            return

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_VLStringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', "VLStringAtom", VLStringAtom())
        vlarray.append("asd")
        vlarray.append("asd\xe4")
        vlarray.append(u"aaana")
        vlarray.append("")
        # Check for ticket #62.
        self.assertRaises(TypeError, vlarray.append, ["foo", "bar"])
        # `VLStringAtom` makes no encoding assumptions.  See ticket #51.
        self.assertRaises(UnicodeEncodeError, vlarray.append, u"asd\xe4")

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 4)
        self.assertEqual(row[0], "asd")
        self.assertEqual(row[1], "asd\xe4")
        self.assertEqual(row[2], "aaana")
        self.assertEqual(row[3], "")
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        self.assertEqual(len(row[3]), 0)

    def test05b_VLStringAtom(self):
        """Checking updating vlarray with variable length strings"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_VLStringAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', "VLStringAtom", VLStringAtom())
        vlarray.append("asd")
        vlarray.append(u"aaana")

        # Modify values
        vlarray[0] = "as4"
        vlarray[1] = "aaanc"
        self.assertRaises(ValueError, vlarray.__setitem__, 1, "shrt")
        self.assertRaises(ValueError, vlarray.__setitem__, 1, "toolong")

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", `row[0]`
            print "Second row in vlarray ==>", `row[1]`

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], "as4")
        self.assertEqual(row[1], "aaanc")
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 5)

    def test06a_Object(self):
        """Checking vlarray with object atoms """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06a_Object..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', "Object", ObjectAtom())
        vlarray.append([[1,2,3], "aaa", u"aaaççç"])
        vlarray.append([3,4, C()])
        vlarray.append(42)

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 3)
        self.assertEqual(row[0], [[1,2,3], "aaa", u"aaaççç"])
        list1 = list(row[1])
        obj = list1.pop()
        self.assertEqual(list1, [3,4])
        self.assertEqual(obj.c, C().c)
        self.assertEqual(row[2], 42)
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 3)
        self.assertRaises(TypeError, len, row[2])

    def test06b_Object(self):
        """Checking updating vlarray with object atoms """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06b_Object..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', "Object", ObjectAtom())
        # When updating an object, this seems to change the number
        # of bytes that cPickle.dumps generates
        #vlarray.append(([1,2,3], "aaa", u"aaaççç"))
        vlarray.append(([1,2,3], "aaa", u"çç4"))
        #vlarray.append([3,4, C()])
        vlarray.append([3,4, [24]])

        # Modify the rows
        #vlarray[0] = ([1,2,4], "aa4", u"aaaçç4")
        vlarray[0] = ([1,2,4], "aa4", u"çç5")
        #vlarray[1] = (3,4, C())
        vlarray[1] = [4,4, [24]]

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], ([1,2,4], "aa4", u"çç5"))
        list1 = list(row[1])
        obj = list1.pop()
        self.assertEqual(list1, [4,4])
        #self.assertEqual(obj.c, C().c)
        self.assertEqual(obj, [24])
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 3)

    def test06c_Object(self):
        """Checking vlarray with object atoms (numpy arrays as values)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06c_Object..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', "Object", ObjectAtom())
        vlarray.append(numpy.array([[1,2], [0,4]], 'i4'))
        vlarray.append(numpy.array([0,1,2,3], 'i8'))
        vlarray.append(numpy.array(42, 'i1'))

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 3)
        self.assertTrue(allequal(row[0], numpy.array([[1,2], [0,4]], 'i4')))
        self.assertTrue(allequal(row[1], numpy.array([0,1,2,3], 'i8')))
        self.assertTrue(allequal(row[2], numpy.array(42, 'i1')))

    def test06d_Object(self):
        """Checking updating vlarray with object atoms (numpy arrays)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06d_Object..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray('/', "Object", ObjectAtom())
        vlarray.append(numpy.array([[1,2], [0,4]], 'i4'))
        vlarray.append(numpy.array([0,1,2,3], 'i8'))
        vlarray.append(numpy.array(42, 'i1'))

        # Modify the rows.  Since PyTables 2.2.1 we use a binary
        # pickle for arrays and ObjectAtoms, so the next should take
        # the same space than the above.
        vlarray[0] = numpy.array([[1,0], [0,4]], 'i4')
        vlarray[1] = numpy.array([0,1,0,3], 'i8')
        vlarray[2] = numpy.array(22, 'i1')

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 3)
        self.assertTrue(allequal(row[0], numpy.array([[1,0], [0,4]], 'i4')))
        self.assertTrue(allequal(row[1], numpy.array([0,1,0,3], 'i8')))
        self.assertTrue(allequal(row[2], numpy.array(22, 'i1')))

    def test07_VLUnicodeAtom(self):
        """Checking vlarray with variable length Unicode strings"""

        # Skip the test if the default encoding has been mangled.
        if sys.getdefaultencoding() != 'ascii':
            return

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_VLUnicodeAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray(
            '/', "VLUnicodeAtom", VLUnicodeAtom() )
        vlarray.append("asd")
        vlarray.append(u"asd\u0140")
        vlarray.append(u"aaana")
        vlarray.append(u"")
        # Check for ticket #62.
        self.assertRaises(TypeError, vlarray.append, ["foo", "bar"])
        # `VLUnicodeAtom` makes no encoding assumptions.
        self.assertRaises(UnicodeDecodeError, vlarray.append, "asd\xe4")

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 4)
        self.assertEqual(row[0], u"asd")
        self.assertEqual(row[1], u"asd\u0140")
        self.assertEqual(row[2], u"aaana")
        self.assertEqual(row[3], u"")
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        self.assertEqual(len(row[3]), 0)

    def test07b_VLUnicodeAtom(self):
        """Checking updating vlarray with variable length Unicode strings"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07b_VLUnicodeAtom..." % self.__class__.__name__

        vlarray = self.fileh.createVLArray(
            '/', "VLUnicodeAtom", VLUnicodeAtom() )
        vlarray.append("asd")
        vlarray.append(u"aaan\xe4")

        # Modify values
        vlarray[0] = u"as\xe4"
        vlarray[1] = u"aaan\u0140"
        self.assertRaises(ValueError, vlarray.__setitem__, 1, "shrt")
        self.assertRaises(ValueError, vlarray.__setitem__, 1, "toolong")

        if self.reopen:
            name = vlarray._v_pathname
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            vlarray = self.fileh.getNode(name)

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", `row[0]`
            print "Second row in vlarray ==>", `row[1]`

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], u"as\xe4")
        self.assertEqual(row[1], u"aaan\u0140")
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 5)


class TypesReopenTestCase(TypesTestCase):
    title = "Reopen"
    reopen = True

class TypesNoReopenTestCase(TypesTestCase):
    title = "No reopen"
    reopen = False

class MDTypesTestCase(unittest.TestCase):
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

    def test01_StringAtom(self):
        """Checking vlarray with MD NumPy string atoms"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(itemsize=3, shape=(2,)),
                                           "Ragged array of strings")
        vlarray.append([["123", "45"],["45", "123"]])
        vlarray.append([["s", "abc"],["abc", "f"],
                        ["s", "ab"],["ab", "f"]])

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array([["123", "45"],["45", "123"]])))
        self.assertTrue(
            allequal(row[1], numpy.array([["s", "abc"],["abc", "f"],
                                          ["s", "ab"],["ab", "f"]])))
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 4)

    def test01b_StringAtom(self):
        """Checking vlarray with MD NumPy string atoms ('python' flavor)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(itemsize=3, shape=(2,)),
                                           "Ragged array of strings")
        vlarray.flavor = "python"
        vlarray.append([["123", "45"],["45", "123"]])
        vlarray.append([["s", "abc"],["abc", "f"],
                        ["s", "ab"],["ab", "f"]])

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], [["123", "45"],["45", "123"]])
        self.assertEqual(row[1], [["s", "abc"],["abc", "f"],
                                  ["s", "ab"],["ab", "f"]])
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 4)


    def test01c_StringAtom(self):
        """Checking vlarray with MD NumPy string atoms (with offset)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(itemsize=3, shape=(2,)),
                                           "Ragged array of strings")
        vlarray.flavor = "python"
        a = numpy.array([["a","b"],["123", "45"],["45", "123"]], dtype="S3")
        vlarray.append(a[1:])
        a = numpy.array([["s", "a"],["ab", "f"],
                         ["s", "abc"],["abc", "f"],
                         ["s", "ab"],["ab", "f"]])
        vlarray.append(a[2:])

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], [["123", "45"],["45", "123"]])
        self.assertEqual(row[1], [["s", "abc"],["abc", "f"],
                                  ["s", "ab"],["ab", "f"]])
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 4)

    def test01d_StringAtom(self):
        """Checking vlarray with MD NumPy string atoms (with stride)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01d_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(itemsize=3, shape=(2,)),
                                           "Ragged array of strings")
        vlarray.flavor = "python"
        a = numpy.array([["a","b"],["123", "45"],["45", "123"]], dtype="S3")
        vlarray.append(a[1::2])
        a = numpy.array([["s", "a"],["ab", "f"],
                         ["s", "abc"],["abc", "f"],
                         ["s", "ab"],["ab", "f"]])
        vlarray.append(a[::3])

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertEqual(row[0], [["123", "45"]])
        self.assertEqual(row[1], [["s", "a"],["abc", "f"]])
        self.assertEqual(len(row[0]), 1)
        self.assertEqual(len(row[1]), 2)

    def test02_BoolAtom(self):
        """Checking vlarray with MD boolean atoms"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(shape = (3,)),
                                           "Ragged array of Booleans")
        vlarray.append([(1,0,3), (1,1,1), (0,0,0)])
        vlarray.append([(-1,0,0)])

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array([[1,0,1],[1,1,1],[0,0,0]],
                                         dtype='bool')))
        self.assertTrue(
            allequal(row[1], numpy.array([[1,0,0]], dtype='bool')))
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 1)

    def test02b_BoolAtom(self):
        """Checking vlarray with MD boolean atoms (with offset)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(shape = (3,)),
                                           "Ragged array of Booleans")
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], dtype='bool')
        vlarray.append(a[1:])  # Create an offset
        a = numpy.array([(1,1,1), (-1,0,0)], dtype='bool')
        vlarray.append(a[1:])  # Create an offset

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array([[1,0,1],[1,1,1],[0,0,0]],
                                         dtype='bool')))
        self.assertTrue(allequal(row[1], numpy.array([[1,0,0]], dtype='bool')))
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 1)

    def test02c_BoolAtom(self):
        """Checking vlarray with MD boolean atoms (with strides)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02c_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(shape = (3,)),
                                           "Ragged array of Booleans")
        a = numpy.array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], dtype='bool')
        vlarray.append(a[1::2])  # Create an strided array
        a = numpy.array([(1,1,1), (-1,0,0), (0,0,0)], dtype='bool')
        vlarray.append(a[::2])  # Create an strided array

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, 2)
        self.assertTrue(
            allequal(row[0], numpy.array([[1,0,1],[0,0,0]], dtype='bool')))
        self.assertTrue(
            allequal(row[1], numpy.array([[1,1,1],[0,0,0]], dtype='bool')))
        self.assertEqual(len(row[0]), 2)
        self.assertEqual(len(row[1]), 2)

    def test03_IntAtom(self):
        """Checking vlarray with MD integer atoms"""

        ttypes = ["Int8",
                  "UInt8",
                  "Int16",
                  "UInt16",
                  "Int32",
                  "UInt32",
                  "Int64",
                  #"UInt64",  # Unavailable in some platforms
                  ]
        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes:
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom.from_sctype(atype, (2,3)))
            vlarray.append([numpy.ones((2,3), atype),
                            numpy.zeros((2,3), atype)])
            vlarray.append([numpy.ones((2,3), atype)*100])

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "Second row in vlarray ==>", repr(row[1])

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([numpy.ones((2,3)),
                                              numpy.zeros((2,3))],
                                              atype)))
            self.assertTrue(
                allequal(row[1], numpy.array([numpy.ones((2,3))*100], atype)))
            self.assertEqual(len(row[0]), 2)
            self.assertEqual(len(row[1]), 1)

    def test04_FloatAtom(self):
        """Checking vlarray with MD floating point atoms"""

        ttypes = ["Float32",
                  "Float64",
                  "Complex32",
                  "Complex64",
                  ]
        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_FloatAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes:
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom.from_sctype(atype, (5,2,6)))
            vlarray.append([numpy.ones((5,2,6), atype)*1.3,
                            numpy.zeros((5,2,6), atype)])
            vlarray.append([numpy.ones((5,2,6), atype)*2.e4])

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing type:", atype
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "Second row in vlarray ==>", row[1]

            self.assertEqual(vlarray.nrows, 2)
            self.assertTrue(
                allequal(row[0], numpy.array([numpy.ones((5,2,6))*1.3,
                                              numpy.zeros((5,2,6))],
                                             atype)))
            self.assertTrue(
                allequal(row[1], numpy.array([numpy.ones((5,2,6))*2.e4],
                                             atype)))
            self.assertEqual(len(row[0]), 2)
            self.assertEqual(len(row[1]), 1)


class MDTypesNumPyTestCase(MDTypesTestCase):
    title = "MDTypes"

class AppendShapeTestCase(unittest.TestCase):
    mode  = "w"

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

    def test00_difinputs(self):
        """Checking vlarray.append() with different inputs"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_difinputs..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        vlarray.flavor = "python"

        # Check different ways to input
        # All of the next should lead to the same rows
        vlarray.append((1,2,3)) # a tuple
        vlarray.append([1,2,3]) # a unique list
        vlarray.append(numpy.array([1,2,3], dtype='int32')) # and array

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read all the vlarray
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 3)
        self.assertEqual(row[0], [1,2,3])
        self.assertEqual(row[1], [1,2,3])
        self.assertEqual(row[2], [1,2,3])

    def test01_toomanydims(self):
        """Checking vlarray.append() with too many dimensions"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_toomanydims..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           StringAtom(itemsize=3),
                                           "Ragged array of strings")
        # Adding an array with one dimensionality more than allowed
        try:
            vlarray.append([["123", "456", "3"]])
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected a ValueError")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read all the rows (there should be none)
        row = vlarray.read()
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        self.assertEqual(vlarray.nrows, 0)

    def test02_zerodims(self):
        """Checking vlarray.append() with a zero-dimensional array"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_zerodims..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        vlarray.append(numpy.zeros(dtype='int32', shape=(6,0)))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read the only row in vlarray
        row = vlarray.read(0)[0]
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", repr(row)

        self.assertEqual(vlarray.nrows, 1)
        self.assertTrue(allequal(row, numpy.zeros(dtype='int32', shape=(0,))))
        self.assertEqual(len(row), 0)

    def test03a_cast(self):
        """Checking vlarray.append() with a casted array (upgrading case)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_cast..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        # This type has to be upgraded
        vlarray.append(numpy.array([1,2], dtype='int16'))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read the only row in vlarray
        row = vlarray.read(0)[0]
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", repr(row)

        self.assertEqual(vlarray.nrows, 1)
        self.assertTrue(allequal(row, numpy.array([1,2], dtype='int32')))
        self.assertEqual(len(row), 2)

    def test03b_cast(self):
        """Checking vlarray.append() with a casted array (downgrading case)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_cast..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        # This type has to be downcasted
        vlarray.append(numpy.array([1,2], dtype='float64'))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read the only row in vlarray
        row = vlarray.read(0)[0]
        if common.verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", repr(row)

        self.assertEqual(vlarray.nrows, 1)
        self.assertTrue(allequal(row, numpy.array([1,2], dtype='int32')))
        self.assertEqual(len(row), 2)


class OpenAppendShapeTestCase(AppendShapeTestCase):
    close = 0

class CloseAppendShapeTestCase(AppendShapeTestCase):
    close = 1

class FlavorTestCase(unittest.TestCase):
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

    def test01a_EmptyVLArray(self):
        """Checking empty vlarrays with different flavors (closing the file)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_EmptyVLArray..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "vlarray",
                                           Atom.from_kind('int', itemsize=4))
        vlarray.flavor = self.flavor
        self.fileh.close()
        self.fileh = openFile(self.file, "r")
        # Read all the rows (it should be empty):
        vlarray = self.fileh.root.vlarray
        row = vlarray.read()
        if common.verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row, repr(row)
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
        # Check that the object read is effectively empty
        self.assertEqual(vlarray.nrows, 0)
        self.assertEqual(row, [])

    def test01b_EmptyVLArray(self):
        """Checking empty vlarrays with different flavors (no closing file)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_EmptyVLArray..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "vlarray",
                                           Atom.from_kind('int', itemsize=4))
        vlarray.flavor = self.flavor
        # Read all the rows (it should be empty):
        row = vlarray.read()
        if common.verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
        # Check that the object read is effectively empty
        self.assertEqual(vlarray.nrows, 0)
        self.assertEqual(row, [])

    def test02_BooleanAtom(self):
        """Checking vlarray with different flavors (boolean versions)"""

        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "Bool", BoolAtom())
        vlarray.flavor = self.flavor
        vlarray.append([1,2,3])
        vlarray.append(())   # Empty row
        vlarray.append([100,0])

        # Read all the rows:
        row = vlarray.read()
        if common.verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        self.assertEqual(vlarray.nrows, 3)
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 0)
        self.assertEqual(len(row[2]), 2)
        if self.flavor == "python":
            arr1 = [1,1,1]
            arr2 = []
            arr3 = [1,0]
        elif self.flavor == "numpy":
            arr1 = numpy.array([1,1,1], dtype="bool")
            arr2 = numpy.array([], dtype="bool")
            arr3 = numpy.array([1,0], dtype="bool")
        elif self.flavor == "numeric":
            arr1 = Numeric.array([1,1,1], typecode="1")
            arr2 = Numeric.array([], typecode="1")
            arr3 = Numeric.array([1,0], typecode="1")
        elif self.flavor == "numarray":
            arr1 = numarray.array([1,1,1], type='Bool')
            arr2 = numarray.array([], type='Bool')
            arr3 = numarray.array([1,0], type='Bool')

        if self.flavor in ['numpy', 'numarray', 'numeric']:
            allequal(row[0], arr1, self.flavor)
            allequal(row[1], arr2, self.flavor)
            allequal(row[1], arr2, self.flavor)
        else:
            # 'python' flavor
            self.assertEqual(row[0], arr1)
            self.assertEqual(row[1], arr2)
            self.assertEqual(row[2], arr3)

    def test03_IntAtom(self):
        """Checking vlarray with different flavors (integer versions)"""

        ttypes = ["Int8",
                  "UInt8",
                  "Int16",
                  "UInt16",
                  "Int32",
                  # Not checked because of Numeric <-> numarray
                  # conversion problems
                  #"UInt32",
                  #"Int64",
                  # Not checked because some platforms does not support it
                  #"UInt64",
                  ]
        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes:
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom.from_sctype(atype))
            vlarray.flavor = self.flavor
            vlarray.append([1,2,3])
            vlarray.append(())
            vlarray.append([100,0])

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 3)
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 0)
            self.assertEqual(len(row[2]), 2)
            if self.flavor == "python":
                arr1 = [1,2,3]
                arr2 = []
                arr3 = [100,0]
            elif self.flavor == "numpy":
                arr1 = numpy.array([1,2,3], dtype=atype)
                arr2 = numpy.array([], dtype=atype)
                arr3 = numpy.array([100,0], dtype=atype)
            elif self.flavor == "numeric":
                type_ = numpy.dtype(atype).base.name
                arr1 = Numeric.array([1,2,3], typecode=typecode[type_])
                arr2 = Numeric.array([], typecode=typecode[type_])
                arr3 = Numeric.array([100,0], typecode=typecode[type_])
            elif self.flavor == "numarray":
                arr1 = numarray.array([1,2,3], type=atype)
                arr2 = numarray.array([], type=atype)
                arr3 = numarray.array([100, 0], type=atype)

            if self.flavor in ["numpy", "numarray", "numeric"]:
                allequal(row[0], arr1, self.flavor)
                allequal(row[1], arr2, self.flavor)
                allequal(row[2], arr3, self.flavor)
            else:
                # "python" flavor
                self.assertEqual(row[0], arr1)
                self.assertEqual(row[1], arr2)
                self.assertEqual(row[2], arr3)

    def test03b_IntAtom(self):
        """Checking vlarray flavors (integer versions and closed file)"""

        ttypes = ["Int8",
                  "UInt8",
                  "Int16",
                  "UInt16",
                  "Int32",
                  # Not checked because of Numeric <-> NumPy
                  # conversion problems
                  #"UInt32",
                  #"Int64",
                  # Not checked because some platforms does not support it
                  #"UInt64",
                  ]
        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes:
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom.from_sctype(atype))
            vlarray.flavor = self.flavor
            vlarray.append([1,2,3])
            vlarray.append(())
            vlarray.append([100,0])
            self.fileh.close()
            self.fileh = openFile(self.file, "a")  # open in "a"ppend mode
            root = self.fileh.root  # Very important!
            vlarray = self.fileh.getNode(root, str(atype))
            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 3)
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 0)
            self.assertEqual(len(row[2]), 2)
            if self.flavor == "python":
                arr1 = [1,2,3]
                arr2 = []
                arr3 = [100,0]
            elif self.flavor == "numpy":
                arr1 = numpy.array([1,2,3], dtype=atype)
                arr2 = numpy.array([], dtype=atype)
                arr3 = numpy.array([100,0], dtype=atype)
            elif self.flavor == "numeric":
                type_ = numpy.dtype(atype).base.name
                arr1 = Numeric.array([1,2,3], typecode=typecode[type_])
                arr2 = Numeric.array([], typecode=typecode[type_])
                arr3 = Numeric.array([100,0], typecode=typecode[type_])
            elif self.flavor == "numarray":
                arr1 = numarray.array([1,2,3], type=atype)
                arr2 = numarray.array([], type=atype)
                arr3 = numarray.array([100, 0], type=atype)

            if self.flavor in ["numpy", "numarray", "numeric"]:
                allequal(row[0], arr1, self.flavor)
                allequal(row[1], arr2, self.flavor)
                allequal(row[2], arr3, self.flavor)
            else:
                # Tuple or List flavors
                self.assertEqual(row[0], arr1)
                self.assertEqual(row[1], arr2)
                self.assertEqual(row[2], arr3)

    def test04_FloatAtom(self):
        """Checking vlarray with different flavors (floating point versions)"""


        ttypes = ["Float32",
                  "Float64",
                  "Complex32",
                  "Complex64",
                  ]
        root = self.rootgroup
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_FloatAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes:
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom.from_sctype(atype))
            vlarray.flavor = self.flavor
            vlarray.append([1.3,2.2,3.3])
            vlarray.append(())
            vlarray.append([-1.3e34,1.e-32])

            # Read all the rows:
            row = vlarray.read()
            if common.verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            self.assertEqual(vlarray.nrows, 3)
            self.assertEqual(len(row[0]), 3)
            self.assertEqual(len(row[1]), 0)
            self.assertEqual(len(row[2]), 2)
            if self.flavor == "python":
                arr1 = list(numpy.array([1.3,2.2,3.3], atype))
                arr2 = list(numpy.array([], atype))
                arr3 = list(numpy.array([-1.3e34,1.e-32], atype))
            elif self.flavor == "numpy":
                arr1 = numpy.array([1.3,2.2,3.3], dtype=atype)
                arr2 = numpy.array([], dtype=atype)
                arr3 = numpy.array([-1.3e34,1.e-32], dtype=atype)
            elif self.flavor == "numeric":
                type_ = numpy.dtype(atype).base.name
                arr1 = Numeric.array([1.3,2.2,3.3], typecode[type_])
                arr2 = Numeric.array([], typecode[type_])
                arr3 = Numeric.array([-1.3e34,1.e-32], typecode[type_])
            elif self.flavor == "numarray":
                arr1 = numarray.array([1.3,2.2,3.3], type=atype)
                arr2 = numarray.array([], type=atype)
                arr3 = numarray.array([-1.3e34,1.e-32], type=atype)

            if self.flavor in ["numpy", "numarray", "numeric"]:
                allequal(row[0], arr1, self.flavor)
                allequal(row[1], arr2, self.flavor)
                allequal(row[2], arr3, self.flavor)
            else:
                # Tuple or List flavors
                self.assertEqual(row[0], arr1)
                self.assertEqual(row[1], arr2)
                self.assertEqual(row[2], arr3)

class NumPyFlavorTestCase(FlavorTestCase):
    flavor = "numpy"

class NumArrayFlavorTestCase(FlavorTestCase):
    flavor = "numarray"

class NumericFlavorTestCase(FlavorTestCase):
    flavor = "numeric"

class PythonFlavorTestCase(FlavorTestCase):
    flavor = "python"

class ReadRangeTestCase(unittest.TestCase):
    nrows = 100
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        filters = Filters(complevel = self.compress,
                          complib = self.complib)
        vlarray = self.fileh.createVLArray(group, 'vlarray', Int32Atom(),
                                           "ragged array if ints",
                                           filters = filters,
                                           expectedsizeinMB = 1)

        # Fill it with 100 rows with variable length
        for i in range(self.nrows):
            vlarray.append(range(i))

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #------------------------------------------------------------------

    def test01_start(self):
        "Checking reads with only a start value"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        # Read some rows:
        row = []
        row.append(vlarray.read(0)[0])
        row.append(vlarray.read(10)[0])
        row.append(vlarray.read(99)[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')))
        self.assertTrue(allequal(row[2], numpy.arange(99, dtype='int32')))

    def test01b_start(self):
        "Checking reads with only a start value in a slice"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        # Read some rows:
        row = []
        row.append(vlarray[0])
        row.append(vlarray[10])
        row.append(vlarray[99])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')))
        self.assertTrue(allequal(row[2], numpy.arange(99, dtype='int32')))

    def test01np_start(self):
        "Checking reads with only a start value in a slice (numpy indexes)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01np_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        # Read some rows:
        row = []
        row.append(vlarray[numpy.int8(0)])
        row.append(vlarray[numpy.int32(10)])
        row.append(vlarray[numpy.int64(99)])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')))
        self.assertTrue(allequal(row[2], numpy.arange(99, dtype='int32')))

    def test02_stop(self):
        "Checking reads with only a stop value"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_stop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray.read(stop=1))
        row.append(vlarray.read(stop=10))
        row.append(vlarray.read(stop=99))
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 1)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0][0], numpy.arange(0, dtype='int32')))
        for x in range(10):
            self.assertTrue(allequal(row[1][x], numpy.arange(x, dtype='int32')))
        for x in range(99):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))

    def test02b_stop(self):
        "Checking reads with only a stop value in a slice"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_stop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[:1])
        row.append(vlarray[:10])
        row.append(vlarray[:99])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 1)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        for x in range(1):
            self.assertTrue(allequal(row[0][x], numpy.arange(0, dtype='int32')))
        for x in range(10):
            self.assertTrue(allequal(row[1][x], numpy.arange(x, dtype='int32')))
        for x in range(99):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))


    def test03_startstop(self):
        "Checking reads with a start and stop values"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_startstop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray.read(0,10))
        row.append(vlarray.read(5,15))
        row.append(vlarray.read(0,100))  # read all the array
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 10)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 100)
        for x in range(0,10):
            self.assertTrue(allequal(row[0][x], numpy.arange(x, dtype='int32')))
        for x in range(5,15):
            self.assertTrue(
                allequal(row[1][x-5], numpy.arange(x, dtype='int32')))
        for x in range(0,100):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))

    def test03b_startstop(self):
        "Checking reads with a start and stop values in slices"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_startstop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[0:10])
        row.append(vlarray[5:15])
        row.append(vlarray[:])  # read all the array
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 10)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 100)
        for x in range(0,10):
            self.assertTrue(allequal(row[0][x], numpy.arange(x, dtype='int32')))
        for x in range(5,15):
            self.assertTrue(
                allequal(row[1][x-5], numpy.arange(x, dtype='int32')))
        for x in range(0,100):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))

    def test04_startstopstep(self):
        "Checking reads with a start, stop & step values"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_startstopstep..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray.read(0,10,2))
        row.append(vlarray.read(5,15,3))
        row.append(vlarray.read(0,100,20))
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 4)
        self.assertTrue(len(row[2]), 5)
        for x in range(0,10,2):
            self.assertTrue(
                allequal(row[0][x/2], numpy.arange(x, dtype='int32')))
        for x in range(5,15,3):
            self.assertTrue(
                allequal(row[1][(x-5)/3], numpy.arange(x, dtype='int32')))
        for x in range(0,100,20):
            self.assertTrue(
                allequal(row[2][x/20], numpy.arange(x, dtype='int32')))

    def test04np_startstopstep(self):
        "Checking reads with a start, stop & step values (numpy indices)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04np_startstopstep..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray.read(numpy.int8(0),numpy.int8(10),numpy.int8(2)))
        row.append(vlarray.read(numpy.int8(5),numpy.int8(15),numpy.int8(3)))
        row.append(vlarray.read(numpy.int8(0),numpy.int8(100),numpy.int8(20)))
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        for x in range(0,10,2):
            self.assertTrue(
                allequal(row[0][x/2], numpy.arange(x, dtype='int32')))
        for x in range(5,15,3):
            self.assertTrue(
                allequal(row[1][(x-5)/3], numpy.arange(x, dtype='int32')))
        for x in range(0,100,20):
            self.assertTrue(
                allequal(row[2][x/20], numpy.arange(x, dtype='int32')))

    def test04b_slices(self):
        "Checking reads with start, stop & step values in slices"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_slices..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[0:10:2])
        row.append(vlarray[5:15:3])
        row.append(vlarray[0:100:20])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        for x in range(0,10,2):
            self.assertTrue(
                allequal(row[0][x/2], numpy.arange(x, dtype='int32')))
        for x in range(5,15,3):
            self.assertTrue(
                allequal(row[1][(x-5)/3], numpy.arange(x, dtype='int32')))
        for x in range(0,100,20):
            self.assertTrue(
                allequal(row[2][x/20], numpy.arange(x, dtype='int32')))

    def test04bnp_slices(self):
        "Checking reads with start, stop & step values in slices (numpy indices)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04bnp_slices..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[numpy.int16(0):numpy.int16(10):numpy.int32(2)])
        row.append(vlarray[numpy.int16(5):numpy.int16(15):numpy.int64(3)])
        row.append(vlarray[numpy.uint16(0):numpy.int32(100):numpy.int8(20)])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        for x in range(0,10,2):
            self.assertTrue(
                allequal(row[0][x/2], numpy.arange(x, dtype='int32')))
        for x in range(5,15,3):
            self.assertTrue(
                allequal(row[1][(x-5)/3], numpy.arange(x, dtype='int32')))
        for x in range(0,100,20):
            self.assertTrue(
                allequal(row[2][x/20], numpy.arange(x, dtype='int32')))

    def test05_out_of_range(self):
        "Checking out of range reads"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_out_of_range..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            row = vlarray.read(1000)[0]
            print "row-->", row
        except IndexError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IndexError was catched!"
                print value
            self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a IndexError and got:\n%s" % value)


class GetItemRangeTestCase(unittest.TestCase):
    nrows = 100
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        filters = Filters(complevel = self.compress,
                          complib = self.complib)
        vlarray = self.fileh.createVLArray(group, 'vlarray', Int32Atom(),
                                           "ragged array if ints",
                                           filters = filters,
                                           expectedsizeinMB = 1)

        # Fill it with 100 rows with variable length
        for i in range(self.nrows):
            vlarray.append(range(i))

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #------------------------------------------------------------------

    def test01_start(self):
        "Checking reads with only a start value"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        # Read some rows:
        row = []
        row.append(vlarray[0])
        # rank-0 array should work as a regular index (see #303)
        row.append(vlarray[numpy.array(10)])
        row.append(vlarray[99])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(
            allequal(row[0], numpy.arange(0, dtype='int32')))
        self.assertTrue(
            allequal(row[numpy.array(1)], numpy.arange(10, dtype='int32')))
        self.assertTrue(
            allequal(row[numpy.array([2])], numpy.arange(99, dtype='int32')))

    def test01b_start(self):
        "Checking reads with only a start value in a slice"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        # Read some rows:
        row = []
        row.append(vlarray[0])
        row.append(vlarray[10])
        row.append(vlarray[99])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')))
        self.assertTrue(allequal(row[2], numpy.arange(99, dtype='int32')))

    def test02_stop(self):
        "Checking reads with only a stop value"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_stop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[:1])
        row.append(vlarray[:10])
        row.append(vlarray[:99])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 1)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0][0], numpy.arange(0, dtype='int32')))
        for x in range(10):
            self.assertTrue(allequal(row[1][x], numpy.arange(x, dtype='int32')))
        for x in range(99):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))

    def test02b_stop(self):
        "Checking reads with only a stop value in a slice"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_stop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[:1])
        row.append(vlarray[:10])
        row.append(vlarray[:99])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 1)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        for x in range(1):
            self.assertTrue(allequal(row[0][x], numpy.arange(0, dtype='int32')))
        for x in range(10):
            self.assertTrue(allequal(row[1][x], numpy.arange(x, dtype='int32')))
        for x in range(99):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))


    def test03_startstop(self):
        "Checking reads with a start and stop values"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_startstop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[0:10])
        row.append(vlarray[5:15])
        row.append(vlarray[0:100])  # read all the array
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 10)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 100)
        for x in range(0,10):
            self.assertTrue(allequal(row[0][x], numpy.arange(x, dtype='int32')))
        for x in range(5,15):
            self.assertTrue(
                allequal(row[1][x-5], numpy.arange(x, dtype='int32')))
        for x in range(0,100):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))

    def test03b_startstop(self):
        "Checking reads with a start and stop values in slices"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_startstop..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[0:10])
        row.append(vlarray[5:15])
        row.append(vlarray[:])  # read all the array
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 10)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 100)
        for x in range(0,10):
            self.assertTrue(allequal(row[0][x], numpy.arange(x, dtype='int32')))
        for x in range(5,15):
            self.assertTrue(
                allequal(row[1][x-5], numpy.arange(x, dtype='int32')))
        for x in range(0,100):
            self.assertTrue(allequal(row[2][x], numpy.arange(x, dtype='int32')))

    def test04_slices(self):
        "Checking reads with a start, stop & step values"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_slices..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[0:10:2])
        row.append(vlarray[5:15:3])
        row.append(vlarray[0:100:20])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 4)
        self.assertTrue(len(row[2]), 5)
        for x in range(0,10,2):
            self.assertTrue(
                allequal(row[0][x/2], numpy.arange(x, dtype='int32')))
        for x in range(5,15,3):
            self.assertTrue(
                allequal(row[1][(x-5)/3], numpy.arange(x, dtype='int32')))
        for x in range(0,100,20):
            self.assertTrue(
                allequal(row[2][x/20], numpy.arange(x, dtype='int32')))

    def test04bnp_slices(self):
        "Checking reads with start, stop & step values (numpy indices)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04np_slices..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3

        # Read some rows:
        row = []
        row.append(vlarray[numpy.int8(0):numpy.int8(10):numpy.int8(2)])
        row.append(vlarray[numpy.int8(5):numpy.int8(15):numpy.int8(3)])
        row.append(vlarray[numpy.int8(0):numpy.int8(100):numpy.int8(20)])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 5)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        for x in range(0,10,2):
            self.assertTrue(
                allequal(row[0][x/2], numpy.arange(x, dtype='int32')))
        for x in range(5,15,3):
            self.assertTrue(
                allequal(row[1][(x-5)/3], numpy.arange(x, dtype='int32')))
        for x in range(0,100,20):
            self.assertTrue(
                allequal(row[2][x/20], numpy.arange(x, dtype='int32')))

    def test05_out_of_range(self):
        "Checking out of range reads"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_out_of_range..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            row = vlarray[1000]
            print "row-->", row
        except IndexError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IndexError was catched!"
                print value
            self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a IndexError and got:\n%s" % value)

    def test05np_out_of_range(self):
        "Checking out of range reads (numpy indexes)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05np_out_of_range..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray

        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            row = vlarray[numpy.int32(1000)]
            print "row-->", row
        except IndexError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IndexError was catched!"
                print value
            self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a IndexError and got:\n%s" % value)


class SetRangeTestCase(unittest.TestCase):
    nrows = 100
    mode  = "w"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        filters = Filters(complevel = self.compress,
                          complib = self.complib)
        vlarray = self.fileh.createVLArray(group, 'vlarray', Int32Atom(),
                                           "ragged array if ints",
                                           filters = filters,
                                           expectedsizeinMB = 1)

        # Fill it with 100 rows with variable length
        for i in range(self.nrows):
            vlarray.append(range(i))

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #------------------------------------------------------------------

    def test01_start(self):
        "Checking updates that modifies a complete row"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        # Modify some rows:
        vlarray[0] = vlarray[0]*2+3
        vlarray[10] = vlarray[10]*2+3
        vlarray[99] = vlarray[99]*2+3

        # Read some rows:
        row = []
        row.append(vlarray.read(0)[0])
        row.append(vlarray.read(10)[0])
        row.append(vlarray.read(99)[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')*2+3))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')*2+3))
        self.assertTrue(allequal(row[2], numpy.arange(99, dtype='int32')*2+3))

    def test01np_start(self):
        "Checking updates that modifies a complete row"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01np_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        # Modify some rows:
        vlarray[numpy.int8(0)] = vlarray[numpy.int16(0)]*2+3
        vlarray[numpy.int8(10)] = vlarray[numpy.int8(10)]*2+3
        vlarray[numpy.int32(99)] = vlarray[numpy.int64(99)]*2+3

        # Read some rows:
        row = []
        row.append(vlarray.read(numpy.int8(0))[0])
        row.append(vlarray.read(numpy.int8(10))[0])
        row.append(vlarray.read(numpy.int8(99))[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 99)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')*2+3))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')*2+3))
        self.assertTrue(allequal(row[2], numpy.arange(99, dtype='int32')*2+3))

    def test02_partial(self):
        "Checking updates with only a part of a row"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_partial..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        # Modify some rows:
        vlarray[0] = vlarray[0]*2+3
        vlarray[10] = vlarray[10]*2+3
        vlarray[96] = vlarray[99][3:]*2+3

        # Read some rows:
        row = []
        row.append(vlarray.read(0)[0])
        row.append(vlarray.read(10)[0])
        row.append(vlarray.read(96)[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 96)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')*2+3))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')*2+3))
        a = numpy.arange(3,99, dtype='int32'); a = a*2+3
        self.assertTrue(allequal(row[2], a))

    def test03a_several_rows(self):
        "Checking updating several rows at once (slice style)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_several_rows..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        # Modify some rows:
        vlarray[3:6] = (vlarray[3]*2+3,
                        vlarray[4]*2+3,
                        vlarray[5]*2+3)

        # Read some rows:
        row = []
        row.append(vlarray.read(3)[0])
        row.append(vlarray.read(4)[0])
        row.append(vlarray.read(5)[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 3)
        self.assertEqual(len(row[1]), 4)
        self.assertEqual(len(row[2]), 5)
        self.assertTrue(allequal(row[0], numpy.arange(3, dtype='int32')*2+3))
        self.assertTrue(allequal(row[1], numpy.arange(4, dtype='int32')*2+3))
        self.assertTrue(allequal(row[2], numpy.arange(5, dtype='int32')*2+3))

    def test03b_several_rows(self):
        "Checking updating several rows at once (list style)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_several_rows..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        # Modify some rows:
        vlarray[[0,10,96]] = (vlarray[0]*2+3,
                              vlarray[10]*2+3,
                              vlarray[96]*2+3)

        # Read some rows:
        row = []
        row.append(vlarray.read(0)[0])
        row.append(vlarray.read(10)[0])
        row.append(vlarray.read(96)[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 96)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')*2+3))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')*2+3))
        self.assertTrue(allequal(row[2], numpy.arange(96, dtype='int32')*2+3))

    def test03c_several_rows(self):
        "Checking updating several rows at once (NumPy's where style)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03c_several_rows..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        # Modify some rows:
        vlarray[(numpy.array([0,10,96]),)] = (vlarray[0]*2+3,
                                              vlarray[10]*2+3,
                                              vlarray[96]*2+3)

        # Read some rows:
        row = []
        row.append(vlarray.read(0)[0])
        row.append(vlarray.read(10)[0])
        row.append(vlarray.read(96)[0])
        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        self.assertEqual(vlarray.nrows, self.nrows)
        self.assertEqual(len(row[0]), 0)
        self.assertEqual(len(row[1]), 10)
        self.assertEqual(len(row[2]), 96)
        self.assertTrue(allequal(row[0], numpy.arange(0, dtype='int32')*2+3))
        self.assertTrue(allequal(row[1], numpy.arange(10, dtype='int32')*2+3))
        self.assertTrue(allequal(row[2], numpy.arange(96, dtype='int32')*2+3))

    def test04_out_of_range(self):
        "Checking out of range updates (first index)"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_out_of_range..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            vlarray[1000] = [1]
        except IndexError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IndexError was catched!"
                print value
            self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a IndexError and got:\n%s" % value)

    def test05_value_error(self):
        "Checking out value errors"
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_value_error..." % self.__class__.__name__

        self.fileh = openFile(self.file, "a")
        vlarray = self.fileh.root.vlarray

        if common.verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            vlarray[10] = [1]*100
            print "row-->", row
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a ValueError and got:\n%s" % value)


class CopyTestCase(unittest.TestCase):
    close = True

    def test01a_copy(self):
        """Checking VLArray.copy() method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Vlarray
        arr = Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', arr, "title array1")
        array1.flavor = "python"
        array1.append([[2,3]])
        array1.append(())  # an empty row
        array1.append([[3, 457],[2,4]])

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
            print "array1-->", repr(array1)
            print "array2-->", repr(array2)
            print "array1[:]-->", repr(array1.read())
            print "array2[:]-->", repr(array2.read())
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertEqual(array1.read(), array2.read())

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(repr(array1.atom), repr(array2.atom))

        self.assertEqual(array1.title, array2.title)

        # Close the file
        fileh.close()
        os.remove(file)

    def test01b_copy(self):
        """Checking VLArray.copy() method. Pseudo-atom case."""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Vlarray
        arr = VLStringAtom()
        array1 = fileh.createVLArray(fileh.root, 'array1', arr, "title array1")
        array1.flavor = "python"
        array1.append("a string")
        array1.append("")  # an empty row
        array1.append("another string")

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
            print "array1-->", repr(array1)
            print "array2-->", repr(array2)
            print "array1[:]-->", repr(array1.read())
            print "array2[:]-->", repr(array2.read())
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertEqual(array1.read(), array2.read())

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.type, array2.atom.type)
        self.assertEqual(repr(array1.atom), repr(array2.atom))

        self.assertEqual(array1.title, array2.title)

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking VLArray.copy() method (where specified)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        arr = Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', arr, "title array1")
        array1.flavor = "python"
        array1.append([[2,3]])
        array1.append(())  # an empty row
        array1.append([[3, 457],[2,4]])

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
            print "array1-->", repr(array1)
            print "array2-->", repr(array2)
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)

        # Check that all the elements are equal
        self.assertEqual(array1.read(), array2.read())

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.shape, array2.shape)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(repr(array1.atom), repr(array1.atom))
        self.assertEqual(array1.title, array2.title)

        # Close the file
        fileh.close()
        os.remove(file)

    # numarray is now deprecated
    def _test03_copy(self):
        """Checking VLArray.copy() method ('numarray' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        if numarray_imported:
            flavor = "numarray"
        else:
            flavor = "numpy"
        arr = Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', arr,
                                     "title array1")
        array1.flavor = flavor
        array1.append([[2,3]])
        array1.append(())  # an empty row
        array1.append([[3, 457],[2,4]])

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
        self.assertEqual(array1.flavor, array2.flavor) # Very important here
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(repr(array1.atom), repr(array1.atom))
        self.assertEqual(array1.title, array2.title)

        # Close the file
        fileh.close()
        os.remove(file)

    # Numeric is now deprecated
    def _test03a_copy(self):
        """Checking VLArray.copy() method ('numeric' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        if numeric_imported:
            flavor = "numeric"
        else:
            flavor = "numpy"
        arr = Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', arr,
                                     "title array1")
        array1.flavor = flavor
        array1.append([[2,3]])
        array1.append(())  # an empty row
        array1.append([[3, 457],[2,4]])

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
        self.assertEqual(array1.flavor, array2.flavor) # Very important here
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(repr(array1.atom), repr(array1.atom))
        self.assertEqual(array1.title, array2.title)

        # Close the file
        fileh.close()
        os.remove(file)

    def test03b_copy(self):
        """Checking VLArray.copy() method ('python' flavor)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        arr = Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', arr,
                                     "title array1")
        array1.flavor = "python"
        array1.append(((2,3),))
        array1.append(())  # an empty row
        array1.append(((3, 457),(2,4)))

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
        self.assertEqual(array1.flavor, array2.flavor) # Very important here
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(repr(array1.atom), repr(array1.atom))
        self.assertEqual(array1.title, array2.title)

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking VLArray.copy() method (checking title copying)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        atom = Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', atom,
                                     "title array1")
        array1.append(((2,3),))
        array1.append(())  # an empty row
        array1.append(((3, 457),(2,4)))
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
        """Checking VLArray.copy() method (user attributes copied)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        atom=Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', atom,
                                     "title array1")
        array1.append(((2,3),))
        array1.append(())  # an empty row
        array1.append(((3, 457),(2,4)))
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

    def notest05b_copy(self):
        """Checking VLArray.copy() method (user attributes not copied)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        atom=Int16Atom(shape=2)
        array1 = fileh.createVLArray(fileh.root, 'array1', atom,
                                     "title array1")
        array1.append(((2,3),))
        array1.append(())  # an empty row
        array1.append(((3, 457),(2,4)))
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
        self.assertEqual(array2.attrs.attr1, None)
        self.assertEqual(array2.attrs.attr2, None)

        # Close the file
        fileh.close()
        os.remove(file)


class CloseCopyTestCase(CopyTestCase):
    close = 1

class OpenCopyTestCase(CopyTestCase):
    close = 0

class CopyIndexTestCase(unittest.TestCase):

    def test01_index(self):
        """Checking VLArray.copy() method with indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        atom = Int32Atom(shape=(2,))
        array1 = fileh.createVLArray(fileh.root, 'array1', atom, "t array1")
        array1.flavor = "python"
        # The next creates 20 rows of variable length
        r = []
        for row in range(20):
            r.append([[row, row+1]])
            array1.append([row, row+1])

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

        # Copy to another array
        array2 = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)

        r2 = r[self.start:self.stop:self.step]
        if common.verbose:
            print "r2-->", r2
            print "array2-->", array2[:]
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", len(r2)
        # Check that all the elements are equal
        self.assertEqual(r2, array2[:])
        # Assert the number of rows in array
        self.assertEqual(len(r2), array2.nrows)

        # Close the file
        fileh.close()
        os.remove(file)


class CopyIndex1TestCase(CopyIndexTestCase):
    close = 0
    start = 0
    stop = 7
    step = 1

class CopyIndex2TestCase(CopyIndexTestCase):
    close = 1
    start = 0
    stop = -1
    step = 1

class CopyIndex3TestCase(CopyIndexTestCase):
    close = 0
    start = 1
    stop = 7
    step = 1

class CopyIndex4TestCase(CopyIndexTestCase):
    close = 1
    start = 0
    stop = 6
    step = 1

class CopyIndex5TestCase(CopyIndexTestCase):
    close = 0
    start = 3
    stop = 7
    step = 1

class CopyIndex6TestCase(CopyIndexTestCase):
    close = 1
    start = 3
    stop = 6
    step = 2

class CopyIndex7TestCase(CopyIndexTestCase):
    close = 0
    start = 0
    stop = 7
    step = 10

class CopyIndex8TestCase(CopyIndexTestCase):
    close = 1
    start = 6
    stop = -1  # Negative values means starting from the end
    step = 1

class CopyIndex9TestCase(CopyIndexTestCase):
    close = 0
    start = 3
    stop = 4
    step = 1

class CopyIndex10TestCase(CopyIndexTestCase):
    close = 1
    start = 3
    stop = 4
    step = 2

class CopyIndex11TestCase(CopyIndexTestCase):
    close = 0
    start = -3
    stop = -1
    step = 2

class CopyIndex12TestCase(CopyIndexTestCase):
    close = 1
    start = -1   # Should point to the last element
    stop = None  # None should mean the last element (including it)
    step = 1


class ChunkshapeTestCase(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.mktemp('.h5')
        self.fileh = openFile(self.file, 'w', title='Chunkshape test')
        atom = Int32Atom(shape=(2,))
        self.fileh.createVLArray('/', 'vlarray', atom, "t array1",
                                 chunkshape=13)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def test00(self):
        """Test setting the chunkshape in a table (no reopen)."""

        vla = self.fileh.root.vlarray
        if common.verbose:
            print "chunkshape-->", vla.chunkshape
        self.assertEqual(vla.chunkshape, (13,))

    def test01(self):
        """Test setting the chunkshape in a table (reopen)."""

        self.fileh.close()
        self.fileh = openFile(self.file, 'r')
        vla = self.fileh.root.vlarray
        if common.verbose:
            print "chunkshape-->", vla.chunkshape
        self.assertEqual(vla.chunkshape, (13,))


class VLUEndianTestCase(common.PyTablesTestCase):
    def test(self):
        """Accessing ``vlunicode`` data of a different endianness."""
        h5fname = self._testFilename('vlunicode_endian.h5')
        h5f = openFile(h5fname)
        try:
            bedata = h5f.root.vlunicode_big[0]
            ledata = h5f.root.vlunicode_little[0]
            self.assertEqual(bedata, u'para\u0140lel')
            self.assertEqual(ledata, u'para\u0140lel')
        finally:
            h5f.close()


class TruncateTestCase(unittest.TestCase):

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")

        # Create an EArray
        arr = Int16Atom(dflt=3)
        array1 = self.fileh.createVLArray(
            self.fileh.root, 'array1', arr, "title array1")
        # Add a couple of rows
        array1.append(numpy.array([456, 2], dtype='Int16'))
        array1.append(numpy.array([3], dtype='Int16'))

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

        self.assertEqual(array1.nrows, 0)
        self.assertEqual(array1[:], [])

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

        self.assertEqual(array1.nrows, 1)
        self.assertTrue(
            allequal(array1[0], numpy.array([456, 2], dtype='Int16')))

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

        self.assertEqual(array1.nrows, 2)
        self.assertTrue(
            allequal(array1[0], numpy.array([456, 2], dtype='Int16')))
        self.assertTrue(allequal(array1[1], numpy.array([3], dtype='Int16')))

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
        self.assertTrue(
            allequal(array1[0], numpy.array([456, 2], dtype='Int16')))
        self.assertTrue(allequal(array1[1], numpy.array([3], dtype='Int16')))
        # Check that the added rows are empty
        self.assertTrue(allequal(array1[2], numpy.array([], dtype='Int16')))
        self.assertTrue(allequal(array1[3], numpy.array([], dtype='Int16')))


class TruncateOpenTestCase(TruncateTestCase):
    close = 0

class TruncateCloseTestCase(TruncateTestCase):
    close = 1


class PointSelectionTestCase(common.PyTablesTestCase):

    def setUp(self):

        # The next are valid selections for both NumPy and PyTables
        self.working_keyset = [
            [],                    # empty list
            [2],                   # single-entry list
            [0,2],                 # list
            [0,-2],                # negative values
            ([0,2],),              # tuple of list
            numpy.array([], dtype="i4"),       # empty array
            numpy.array([1], dtype="i4"),      # single-entry array
            numpy.array([True,False, True]),   # array of bools
            ]

        # The next are invalid selections for VLArrays
        self.not_working_keyset = [
            [1,2,100],               # coordinate 100 > len(vlarray)
            ([True,False, True],),   # tuple of bools
            ]

        # Create an instance of an HDF5 Array
        self.file = tempfile.mktemp(".h5")
        self.fileh = fileh = openFile(self.file, "w")
        # Create a sample array
        arr1 = numpy.array([5, 6], dtype="i4")
        arr2 = numpy.array([5, 6, 7], dtype="i4")
        arr3 = numpy.array([5, 6, 9, 8], dtype="i4")
        self.nparr = nparr = numpy.array([arr1, arr2, arr3], dtype="object")
        # Create the VLArray
        self.vlarr = vlarr = fileh.createVLArray(
            fileh.root, 'vlarray', Int32Atom())
        vlarr.append(arr1)
        vlarr.append(arr2)
        vlarr.append(arr3)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test01a_read(self):
        """Test for point-selections (read, boolean keys)."""
        nparr = self.nparr
        vlarr = self.vlarr
        for key in self.working_keyset:
            if common.verbose:
                print "Selection to test:", `key`
            a = nparr[key].tolist()
            b = vlarr[key]
            # if common.verbose:
            #     print "NumPy selection:", a, type(a)
            #     print "PyTables selection:", b, type(b)
            self.assertEqual(repr(a), repr(b),
                "NumPy array and PyTables selections does not match.")

    def test01b_read(self):
        """Test for point-selections (not working selections, read)."""
        nparr = self.nparr
        vlarr = self.vlarr
        for key in self.not_working_keyset:
            if common.verbose:
                print "Selection to test:", key
            self.assertRaises(IndexError, vlarr.__getitem__, key)



#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicNumPyTestCase))
        theSuite.addTest(unittest.makeSuite(BasicPythonTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
        theSuite.addTest(unittest.makeSuite(BloscComprTestCase))
        theSuite.addTest(unittest.makeSuite(BloscShuffleComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(Bzip2ComprTestCase))
        theSuite.addTest(unittest.makeSuite(TypesReopenTestCase))
        theSuite.addTest(unittest.makeSuite(TypesNoReopenTestCase))
        theSuite.addTest(unittest.makeSuite(MDTypesNumPyTestCase))
        theSuite.addTest(unittest.makeSuite(OpenAppendShapeTestCase))
        theSuite.addTest(unittest.makeSuite(CloseAppendShapeTestCase))
        theSuite.addTest(unittest.makeSuite(PythonFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(NumPyFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(ReadRangeTestCase))
        theSuite.addTest(unittest.makeSuite(GetItemRangeTestCase))
        theSuite.addTest(unittest.makeSuite(SetRangeTestCase))
        theSuite.addTest(unittest.makeSuite(ShuffleComprTestCase))
        theSuite.addTest(unittest.makeSuite(Fletcher32TestCase))
        theSuite.addTest(unittest.makeSuite(AllFiltersTestCase))
        theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
        theSuite.addTest(unittest.makeSuite(OpenCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex2TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex3TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex4TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex5TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex6TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex7TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex8TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex9TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex10TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex11TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex12TestCase))
        theSuite.addTest(unittest.makeSuite(ChunkshapeTestCase))
        theSuite.addTest(unittest.makeSuite(VLUEndianTestCase))
        theSuite.addTest(unittest.makeSuite(TruncateOpenTestCase))
        theSuite.addTest(unittest.makeSuite(TruncateCloseTestCase))
        theSuite.addTest(unittest.makeSuite(PointSelectionTestCase))

        # Numeric is now deprecated
        #if numeric_imported:
        #    theSuite.addTest(unittest.makeSuite(BasicNumericTestCase))
        #    theSuite.addTest(unittest.makeSuite(NumericFlavorTestCase))
        # numarray is now deprecated
        #if numarray_imported:
        #    theSuite.addTest(unittest.makeSuite(BasicNumArrayTestCase))
        #    theSuite.addTest(unittest.makeSuite(NumArrayFlavorTestCase))

    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
