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

from test_all import verbose, allequal

class C:
    c = (3,4.5) 

class BasicTestCase(unittest.TestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    flavor = "NumArray"

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
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        vlarray = self.fileh.createVLArray(group, 'vlarray1',
                                           Int32Atom(flavor=self.flavor),
                                           "ragged array if ints",
                                           filters = filters,
                                           expectedsizeinMB = 1)

        # Fill it with 5 rows
        vlarray.append(1, 2)
        if self.flavor == "NumArray":
            vlarray.append(array([3, 4, 5], type=Int32))
            vlarray.append(array([], type=Int32))    # Empty entry
        elif self.flavor == "Numeric":
            vlarray.append(Numeric.array([3, 4, 5], typecode='i'))
            vlarray.append(Numeric.array([], typecode='i'))     # Empty entry
        elif self.flavor == "Tuple":
            vlarray.append((3, 4, 5))
            vlarray.append(())         # Empty entry
        vlarray.append([6, 7, 8, 9])
        vlarray.append(10, 11, 12, 13, 14)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    #----------------------------------------

    def test01_readVLArray(self):
        """Checking vlarray read"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readVLArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.getNode("/vlarray1")

        # Choose a small value for buffer size
        vlarray._v_maxTuples = 3
        # Read the first row:
        row = vlarray.read(0)
        row2 = vlarray.read(2)
        if verbose:
            print "Flavor:", vlarray.flavor
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row
            
        nrows = 5
        assert nrows == vlarray.nrows
        if self.flavor == "NumArray":
            assert allequal(row, array([1, 2], type=Int32))
            assert allequal(row2, array([], type=Int32))
        elif self.flavor == "Numeric":
            assert type(row) == type(Numeric.array([1, 2]))
            assert allequal(row, Numeric.array([1, 2]), self.flavor)
            assert allequal(row2, Numeric.array([]), self.flavor)
        elif self.flavor == "Tuple":
            assert row == (1, 2)
            assert row2 == ()
        assert len(row) == 2

        # Check filters:
        if self.compress <> vlarray.filters.complevel and verbose:
            print "Error in compress. Class:", self.__class__.__name__
            print "self, vlarray:", self.compress, vlarray.filters.complevel
        tinfo = whichLibVersion(self.complib)
        if tinfo[0] == 0:
            self.complib = "zlib"
        assert vlarray.filters.complib == self.complib
        assert vlarray.filters.complevel == self.compress
        if self.shuffle <> vlarray.filters.shuffle and verbose:
            print "Error in shuffle. Class:", self.__class__.__name__
            print "self, vlarray:", self.shuffle, vlarray.filters.shuffle
        assert self.shuffle == vlarray.filters.shuffle
        if self.fletcher32 <> vlarray.filters.fletcher32 and verbose:
            print "Error in fletcher32. Class:", self.__class__.__name__
            print "self, vlarray:", self.fletcher32, vlarray.filters.fletcher32
        assert self.fletcher32 == vlarray.filters.fletcher32


class BasicNumArrayTestCase(BasicTestCase):
    flavor = "NumArray"

class BasicNumericTestCase(BasicTestCase):
    flavor = "Numeric"

class BasicTupleTestCase(BasicTestCase):
    flavor = "Tuple"

class ZlibComprTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"

class LZOComprTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"

class UCLComprTestCase(BasicTestCase):
    compress = 1
    complib = "ucl"

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
        self.rootgroup = self.fileh.root

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    #----------------------------------------

    def test01_StringAtom(self):
        """Checking vlarray with numarray string atoms"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(length=3),
                                           "Ragged array of strings")
        vlarray.append(["123", "456", "3"])
        vlarray.append(["456", "3"])

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            
        assert vlarray.nrows == 2
        assert allequal(row[0], strings.array(["123", "456", "3"]))
        assert allequal(row[1], strings.array(["456", "3"]))
        assert len(row[0]) == 3
        assert len(row[1]) == 2

    def test01b_StringAtom(self):
        """Checking vlarray with numarray string atoms (String flavor)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom2',
                                           StringAtom(length=3,
                                                      flavor="String"),
                                           "Ragged array of strings")
        vlarray.append(["123", "456", "3"])
        vlarray.append(["456", "3"])

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Testing String flavor"
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            
        assert vlarray.nrows == 2
        assert row[0] == ["123", "456", "3"]
        assert row[1] == ["456", "3"]
        assert len(row[0]) == 3
        assert len(row[1]) == 2

    # Strings Atoms with UString (unicode strings) flavor can't be safely
    # implemented because the strings can be cut in the middle of a utf-8
    # codification and that can lead to errors like:
    #     >>> print 'a\xc3'.decode('utf-8')
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in ?
    # UnicodeDecodeError: 'utf8' codec can't decode byte 0xc3 in position 1: unexpected end of data

#     def test01c_StringAtom(self):
#         """Checking vlarray with numarray string atoms (UString flavor)"""

#         root = self.rootgroup
#         if verbose:
#             print '\n', '-=' * 30
#             print "Running %s.test01c_StringAtom..." % self.__class__.__name__

#         # Create an string atom
#         vlarray = self.fileh.createVLArray(root, 'stringAtom2',
#                                            StringAtom(length=3,
#                                                       flavor="UString"),
#                                            "Ragged array of unicode strings")
#         vlarray.append(["áéç", "èàòÉ", "ñ"])
#         vlarray.append(["ççççç", "asaËÏÖÜ"])

#         # Read all the rows:
#         row = vlarray.read()
#         if verbose:
#             print "Testing String flavor"
#             print "Object read:", row
#             print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
#             print "First row in vlarray ==>", row[0]
            
#         assert vlarray.nrows == 2
#         assert row[0] == ("123", "456", "3")
#         assert row[1] == ("456", "3")
#         assert len(row[0]) == 3
#         assert len(row[1]) == 2

    def test02_BoolAtom(self):
        """Checking vlarray with boolean atoms"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(),
                                           "Ragged array of Booleans")
        vlarray.append(1,0,3)
        vlarray.append(-1,0)

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            
        assert vlarray.nrows == 2
        assert allequal(row[0], array([1,0,1], type=Bool))
        assert allequal(row[1], array([1,0], type=Bool))
        assert len(row[0]) == 3
        assert len(row[1]) == 2

    def test03_IntAtom(self):
        """Checking vlarray with integer atoms"""

        ttypes = {"Int8": Int8,
                  "UInt8": UInt8,
                  "Int16": Int16,
                  "UInt16": UInt16,
                  "Int32": Int32,
                  "UInt32": UInt32,
                  "Int64": Int64,
                  #"UInt64": UInt64,  # Unavailable in some platforms
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype]))
            vlarray.append(1,2,3)
            vlarray.append(-1,0)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 2
            assert allequal(row[0], array([1,2,3], type=ttypes[atype]))
            assert allequal(row[1], array([-1,0], type=ttypes[atype]))
            assert len(row[0]) == 3
            assert len(row[1]) == 2

    def test04_FloatAtom(self):
        """Checking vlarray with floating point atoms"""

        ttypes = {"Float32": Float32,
                  "Float64": Float64,
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_FloatAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype]))
            vlarray.append(1.3,2.2,3.3)
            vlarray.append(-1.3e34,1.e-32)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing type:", atype
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 2
            assert allequal(row[0], array([1.3,2.2,3.3], ttypes[atype]))
            assert allequal(row[1], array([-1.3e34,1.e-32], ttypes[atype]))
            assert len(row[0]) == 3
            assert len(row[1]) == 2

    def test05_VLStringAtom(self):
        """Checking vlarray with variable length strings"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_VLStringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "VLStringAtom", VLStringAtom())
        vlarray.append(u"asd")
        vlarray.append(u"aaañá")

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        assert vlarray.nrows == 2
        assert row[0] == u"asd"
        assert row[1] == u"aaañá"
        assert len(row[0]) == 3
        assert len(row[1]) == 5

    def test06_Object(self):
        """Checking vlarray with object atoms """

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_Object..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "Object", ObjectAtom())
        vlarray.append([1,2,3], "aaa", u"aaaççç")
        vlarray.append(3,4, C())

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        assert vlarray.nrows == 2
        assert row[0] == ([1,2,3], "aaa", u"aaaççç")
        list1 = list(row[1])
        obj = list1.pop()
        assert list1 == [3,4]
        assert obj.c == C().c
        assert len(row[0]) == 3
        assert len(row[1]) == 3


class TypesNumArrayTestCase(TypesTestCase):
    title = "Types"

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
        
    #----------------------------------------

    def test01_StringAtom(self):
        """Checking vlarray with MD numarray string atoms"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(length=3, shape=(2,)),
                                           "Ragged array of strings")
        vlarray.append([["123", "45"],["45", "123"]])
        vlarray.append(["s", "abc"],["abc", "f"],
                       ["s", "ab"],["ab", "f"])

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert allequal(row[0], strings.array([["123", "45"],["45", "123"]]))
        assert allequal(row[1], strings.array([["s", "abc"],["abc", "f"],
                                              ["s", "ab"],["ab", "f"]]))
        assert len(row[0]) == 2
        assert len(row[1]) == 4

    def test01b_StringAtom(self):
        """Checking vlarray with MD numarray string atoms (String flavor)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(length=3, shape=(2,),
                                                      flavor="String"),
                                           "Ragged array of strings")
        vlarray.append([["123", "45"],["45", "123"]])
        vlarray.append(["s", "abc"],["abc", "f"],
                       ["s", "ab"],["ab", "f"])

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert row[0] == [["123", "45"],["45", "123"]]
        assert row[1] == [["s", "abc"],["abc", "f"],
                          ["s", "ab"],["ab", "f"]]
        assert len(row[0]) == 2
        assert len(row[1]) == 4


    def test01c_StringAtom(self):
        """Checking vlarray with MD numarray string atoms (with offset)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(length=3, shape=(2,),
                                                      flavor="String"),
                                           "Ragged array of strings")
        a=strings.array([["a","b"],["123", "45"],["45", "123"]], itemsize=3)
        vlarray.append(a[1:])
        a=strings.array([["s", "a"],["ab", "f"],
                         ["s", "abc"],["abc", "f"],
                         ["s", "ab"],["ab", "f"]])
        vlarray.append(a[2:])

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert row[0] == [["123", "45"],["45", "123"]]
        assert row[1] == [["s", "abc"],["abc", "f"],
                          ["s", "ab"],["ab", "f"]]
        assert len(row[0]) == 2
        assert len(row[1]) == 4

    def test01d_StringAtom(self):
        """Checking vlarray with MD numarray string atoms (with stride)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01d_StringAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'stringAtom',
                                           StringAtom(length=3, shape=(2,),
                                                      flavor="String"),
                                           "Ragged array of strings")
        a=strings.array([["a","b"],["123", "45"],["45", "123"]], itemsize=3)
        vlarray.append(a[1::2])
        a=strings.array([["s", "a"],["ab", "f"],
                         ["s", "abc"],["abc", "f"],
                         ["s", "ab"],["ab", "f"]])
        vlarray.append(a[::3])

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert row[0] == [["123", "45"]]
        assert row[1] == [["s", "a"],["abc", "f"]]
        assert len(row[0]) == 1
        assert len(row[1]) == 2

    def test02_BoolAtom(self):
        """Checking vlarray with MD boolean atoms"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(shape = (3,)),
                                           "Ragged array of Booleans")
        vlarray.append((1,0,3), (1,1,1), (0,0,0))
        vlarray.append((-1,0,0))

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert allequal(row[0], array([[1,0,1],[1,1,1],[0,0,0]], type=Bool))
        assert allequal(row[1], array([[1,0,0]], type=Bool))
        assert len(row[0]) == 3
        assert len(row[1]) == 1

    def test02b_BoolAtom(self):
        """Checking vlarray with MD boolean atoms (with offset)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(shape = (3,)),
                                           "Ragged array of Booleans")
        a=array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], type=Bool)
        vlarray.append(a[1:])  # Create an offset
        a=array([(1,1,1), (-1,0,0)], type=Bool)
        vlarray.append(a[1:])  # Create an offset

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert allequal(row[0], array([[1,0,1],[1,1,1],[0,0,0]], type=Bool))
        assert allequal(row[1], array([[1,0,0]], type=Bool))
        assert len(row[0]) == 3
        assert len(row[1]) == 1

    def test02c_BoolAtom(self):
        """Checking vlarray with MD boolean atoms (with strides)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02c_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'BoolAtom',
                                           BoolAtom(shape = (3,)),
                                           "Ragged array of Booleans")
        a=array([(0,0,0), (1,0,3), (1,1,1), (0,0,0)], type=Bool)
        vlarray.append(a[1::2])  # Create an strided array
        a=array([(1,1,1), (-1,0,0), (0,0,0)], type=Bool)
        vlarray.append(a[::2])  # Create an strided array

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]
            
        assert vlarray.nrows == 2
        assert allequal(row[0], array([[1,0,1],[0,0,0]], type=Bool))
        assert allequal(row[1], array([[1,1,1],[0,0,0]], type=Bool))
        assert len(row[0]) == 2
        assert len(row[1]) == 2

    def test03_IntAtom(self):
        """Checking vlarray with MD integer atoms"""

        ttypes = {"Int8": Int8,
                  "UInt8": UInt8,
                  "Int16": Int16,
                  "UInt16": UInt16,
                  "Int32": Int32,
                  "UInt32": UInt32,
                  "Int64": Int64,
                  #"UInt64": UInt64,
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype],
                                                    shape = (2,3)))
            vlarray.append(ones((2,3), ttypes[atype]),
                           zeros((2,3), ttypes[atype]))
            vlarray.append(ones((2,3), ttypes[atype])*100)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing type:", atype
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "Second row in vlarray ==>", repr(row[1])

            assert vlarray.nrows == 2
            assert allequal(row[0], array([ones((2,3)),
                                          zeros((2,3))], ttypes[atype]))
            assert allequal(row[1], array([ones((2,3))*100], ttypes[atype]))
            assert len(row[0]) == 2
            assert len(row[1]) == 1

    def test04_FloatAtom(self):
        """Checking vlarray with MD floating point atoms"""

        ttypes = {"Float32": Float32,
                  "Float64": Float64,
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_FloatAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype],
                                                    shape=(5,2,6)))
            vlarray.append(ones((5,2,6), ttypes[atype])*1.3,
                           zeros((5,2,6), ttypes[atype]))
            vlarray.append(ones((5,2,6), ttypes[atype])*2.e4)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing type:", atype
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "Second row in vlarray ==>", row[1]

            assert vlarray.nrows == 2
            assert allequal(row[0], array([ones((5,2,6))*1.3,
                                          zeros((5,2,6))], ttypes[atype]))
            assert allequal(row[1], array([ones((5,2,6))*2.e4], ttypes[atype]))
            assert len(row[0]) == 2
            assert len(row[1]) == 1


class MDTypesNumArrayTestCase(MDTypesTestCase):
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
        
    #----------------------------------------

    def test00_difinputs(self):
        """Checking vlarray.append() with different inputs"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_difinputs..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(flavor="Tuple"),
                                           "Ragged array of ints")
        # Check different ways to input
        # All of the next should lead to the same rows
        vlarray.append(1,2,3) # a list of parameters
        vlarray.append((1,2,3)) # a tuple
        vlarray.append([1,2,3]) # a unique list
        vlarray.append(array([1,2,3], type=Int32)) # and array

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read all the vlarray
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            
        assert vlarray.nrows == 4
        assert row[0] == (1,2,3)
        assert row[1] == (1,2,3)
        assert row[2] == (1,2,3)
        assert row[3] == (1,2,3)

    def test01_toomanydims(self):
        """Checking vlarray.append() with too many dimensions"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_toomanydims..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           StringAtom(length=3),
                                           "Ragged array of strings")
        # Adding an array with one dimensionality more than allowed
        try:
            vlarray.append([["123", "456", "3"]])
        except ValueError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
		print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected a ValueError")

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray
            
        # Read all the rows (there should be none)
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            
        assert vlarray.nrows == 0

    def test02_zerodims(self):
        """Checking vlarray.append() with a zero-dimensional array"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_zerodims..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        vlarray.append(zeros(type=Int32, shape=(6,0)))

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read the only row in vlarray
        row = vlarray.read(0)
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row.info()
            
        assert vlarray.nrows == 1
        assert allequal(row, zeros(type=Int32, shape=(0,)))
        assert len(row) == 0

    def test03a_cast(self):
        """Checking vlarray.append() with a casted array (upgrading case)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a_cast..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        # This type has to be upgraded
        vlarray.append(array([1,2], type=Int16))

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read the only row in vlarray
        row = vlarray.read(0)
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row.info()
            
        assert vlarray.nrows == 1
        assert allequal(row, array([1,2], type=Int32))
        assert len(row) == 2

    def test03b_cast(self):
        """Checking vlarray.append() with a casted array (downgrading case)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b_cast..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, 'vlarray',
                                           Int32Atom(),
                                           "Ragged array of ints")
        # This type has to be dowgraded
        vlarray.append(array([1,2], type=Float64))

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r")
            vlarray = self.fileh.root.vlarray

        # Read the only row in vlarray
        row = vlarray.read(0)
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row.info()
            
        assert vlarray.nrows == 1
        assert allequal(row, array([1,2], type=Int32))
        assert len(row) == 2

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
        
    #----------------------------------------

    def test01a_EmptyVLArray(self):
        """Checking empty vlarrays with different flavors (closing the file)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_EmptyVLArray..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "vlarray",
                                           IntAtom(itemsize=4,
                                                   flavor=self.flavor))
        self.fileh.close()
        self.fileh = openFile(self.file, "r")
        # Read all the rows (it should be empty):
        vlarray = self.fileh.root.vlarray
        row = vlarray.read()
        if verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row, repr(row)
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
        # Check that the object read is effectively empty
        assert vlarray.nrows == 0
        assert row == []

    def test01b_EmptyVLArray(self):
        """Checking empty vlarrays with different flavors (no closing file)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_EmptyVLArray..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "vlarray",
                                           IntAtom(itemsize=4,
                                                   flavor=self.flavor))
        # Read all the rows (it should be empty):
        row = vlarray.read()
        if verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
        # Check that the object read is effectively empty
        assert vlarray.nrows == 0
        assert row == []

    def test02_BooleanAtom(self):
        """Checking vlarray with different flavors (boolean versions)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_BoolAtom..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "Bool",
                                           BoolAtom(flavor=self.flavor))
        vlarray.append(1,2,3)
        vlarray.append()   # Empty row
        vlarray.append(100,0)

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        assert vlarray.nrows == 3
        assert len(row[0]) == 3
        assert len(row[1]) == 0
        assert len(row[2]) == 2
        if self.flavor == "Tuple":
            arr1 = (1,1,1)
            arr2 = ()
            arr3 = (1,0)                    
        elif self.flavor == "List":
            arr1 = [1,1,1]
            arr2 = []
            arr3 = [1,0]
        elif self.flavor == "Numeric":
            arr1 = Numeric.array([1,1,1], typecode="1")
            arr2 = Numeric.array([], typecode="1")
            arr3 = Numeric.array([1,0], typecode="1")
        else:  # Default (NumArray)
            arr1 = array([1,1,1], type=Bool)
            arr2 = array([], type=Bool)
            arr3 = array([1,0], type=Bool)

        if self.flavor == "Numeric":
            allequal(row[0], arr1, "Numeric")
            allequal(row[1], arr2, "Numeric")
            allequal(row[2], arr3, "Numeric")
        elif self.flavor == "NumArray":
            allequal(row[0], arr1)
            allequal(row[1], arr2)
            allequal(row[1], arr2)
        else:
            # Tuple or List flavors
            assert row[0] == arr1
            assert row[1] == arr2
            assert row[2] == arr3

    def test03_IntAtom(self):
        """Checking vlarray with different flavors (integer versions)"""

        ttypes = {"Int8": Int8,
                  "UInt8": UInt8,
                  "Int16": Int16,
                  "UInt16": UInt16,
                  "Int32": Int32,
                  # Not checked because of Numeric <-> numarray
                  # conversion problems
                  #"UInt32": UInt32,
                  #"Int64": Int64,
                  # Not checked because some platforms does not support it
                  #"UInt64": UInt64,
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype],
                                                    flavor=self.flavor))
            vlarray.append(1,2,3)
            vlarray.append()
            vlarray.append(100,0)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 3
            assert len(row[0]) == 3
            assert len(row[1]) == 0
            assert len(row[2]) == 2
            if self.flavor == "Tuple":
                arr1 = (1,2,3)
                arr2 = ()
                arr3 = (100,0)                    
            elif self.flavor == "List":
                arr1 = [1,2,3]
                arr2 = []
                arr3 = [100,0]
            elif self.flavor == "Numeric":
                arr1 = Numeric.array([1,2,3], typecode=typecode[ttypes[atype]])
                arr2 = Numeric.array([], typecode=typecode[ttypes[atype]])
                arr3 = Numeric.array([100,0], typecode=typecode[ttypes[atype]])
            else:  # Default (NumArray)
                arr1 = array([1,2,3], type=ttypes[atype])
                arr2 = array([], type=ttypes[atype])
                arr3 = array([100, 0], type=ttypes[atype])

            if self.flavor == "Numeric":
                allequal(row[0], arr1, "Numeric")
                allequal(row[1], arr2, "Numeric")
                allequal(row[2], arr3, "Numeric")
            elif self.flavor == "NumArray":
                allequal(row[0], arr1)
                allequal(row[1], arr2)
                allequal(row[2], arr3)
            else:
                # Tuple or List flavors
                assert row[0] == arr1
                assert row[1] == arr2
                assert row[2] == arr3

    def test03b_IntAtom(self):
        """Checking vlarray flavors (integer versions and closed file)"""

        ttypes = {"Int8": Int8,
                  "UInt8": UInt8,
                  "Int16": Int16,
                  "UInt16": UInt16,
                  "Int32": Int32,
                  # Not checked because of Numeric <-> numarray
                  # conversion problems
                  #"UInt32": UInt32,
                  #"Int64": Int64,
                  # Not checked because some platforms does not support it
                  #"UInt64": UInt64,
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_IntAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype],
                                                    flavor=self.flavor))
            vlarray.append(1,2,3)
            vlarray.append()
            vlarray.append(100,0)
            self.fileh.close()
            self.fileh = openFile(self.file, "a")  # open in "a"ppend mode
            root = self.fileh.root  # Very important!
            vlarray = self.fileh.getNode(root, str(atype))
            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 3
            assert len(row[0]) == 3
            assert len(row[1]) == 0
            assert len(row[2]) == 2
            if self.flavor == "Tuple":
                arr1 = (1,2,3)
                arr2 = ()
                arr3 = (100,0)                    
            elif self.flavor == "List":
                arr1 = [1,2,3]
                arr2 = []
                arr3 = [100,0]
            elif self.flavor == "Numeric":
                arr1 = Numeric.array([1,2,3], typecode=typecode[ttypes[atype]])
                arr2 = Numeric.array([], typecode=typecode[ttypes[atype]])
                arr3 = Numeric.array([100,0], typecode=typecode[ttypes[atype]])
            else:  # Default (NumArray)
                arr1 = array([1,2,3], type=ttypes[atype])
                arr2 = array([], type=ttypes[atype])
                arr3 = array([100, 0], type=ttypes[atype])

            if self.flavor == "Numeric":
                allequal(row[0], arr1, "Numeric")
                allequal(row[1], arr2, "Numeric")
                allequal(row[2], arr3, "Numeric")
            elif self.flavor == "NumArray":
                allequal(row[0], arr1)
                allequal(row[1], arr2)
                allequal(row[2], arr3)
            else:
                # Tuple or List flavors
                assert row[0] == arr1
                assert row[1] == arr2
                assert row[2] == arr3

    def test04_FloatAtom(self):
        """Checking vlarray with different flavors (floating point versions)"""
        

        ttypes = {"Float32": Float32,
                  "Float64": Float64,
                  }
        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_FloatAtom..." % self.__class__.__name__

        # Create an string atom
        for atype in ttypes.iterkeys():
            vlarray = self.fileh.createVLArray(root, atype,
                                               Atom(ttypes[atype],
                                                    flavor=self.flavor))
            vlarray.append(1.3,2.2,3.3)
            vlarray.append()
            vlarray.append(-1.3e34,1.e-32)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 3
            assert len(row[0]) == 3
            assert len(row[1]) == 0
            assert len(row[2]) == 2
            if self.flavor == "Tuple":
                arr1 = tuple(array([1.3,2.2,3.3], typecode[ttypes[atype]]))
                arr2 = tuple(array([], typecode[ttypes[atype]]))
                arr3 = tuple(array([-1.3e34,1.e-32], typecode[ttypes[atype]]))
            elif self.flavor == "List":
                arr1 = list(array([1.3,2.2,3.3], typecode[ttypes[atype]]))
                arr2 = list(array([], typecode[ttypes[atype]]))
                arr3 = list(array([-1.3e34,1.e-32], typecode[ttypes[atype]]))
            elif self.flavor == "Numeric":
                arr1 = Numeric.array([1.3,2.2,3.3], typecode[ttypes[atype]])
                arr2 = Numeric.array([], typecode[ttypes[atype]])
                arr3 = Numeric.array([-1.3e34,1.e-32], typecode[ttypes[atype]])
            else:   # Default (NumArray)
                arr1 = array([1.3,2.2,3.3], type=ttypes[atype])
                arr2 = array([], type=ttypes[atype])
                arr3 = array([-1.3e34,1.e-32], type=ttypes[atype])
                
            if self.flavor == "Numeric":
                allequal(row[0], arr1, "Numeric")
                allequal(row[1], arr2, "Numeric")
                allequal(row[2], arr3, "Numeric")
            elif self.flavor == "NumArray":
                allequal(row[0], arr1)
                allequal(row[1], arr2)
                allequal(row[2], arr3)
            else:
                # Tuple or List flavors
                assert row[0] == arr1
                assert row[1] == arr2
                assert row[2] == arr3

class NumArrayFlavorTestCase(FlavorTestCase):
    flavor = "NumArray"

class TupleFlavorTestCase(FlavorTestCase):
    flavor = "Tuple"

class ListFlavorTestCase(FlavorTestCase):
    flavor = "List"

class NumericFlavorTestCase(FlavorTestCase):
    flavor = "Numeric"

class RangeTestCase(unittest.TestCase):
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
        
    #------------------------------------------------------------------

    def test01_start(self):
        "Checking reads with only a start value"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        
        # Read some rows:
        row = []
        row.append(vlarray.read(0))
        row.append(vlarray.read(10))
        row.append(vlarray.read(99))
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 0
        assert len(row[1]) == 10
        assert len(row[2]) == 99
        assert allequal(row[0], arange(0, type=Int32))
        assert allequal(row[1], arange(10, type=Int32))
        assert allequal(row[2], arange(99, type=Int32))

    def test01b_start(self):
        "Checking reads with only a start value in a slice"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_start..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        
        # Read some rows:
        row = []
        row.append(vlarray[0])
        row.append(vlarray[10])
        row.append(vlarray[99])
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 0
        assert len(row[1]) == 10
        assert len(row[2]) == 99
        assert allequal(row[0], arange(0, type=Int32))
        assert allequal(row[1], arange(10, type=Int32))
        assert allequal(row[2], arange(99, type=Int32))

    def test02_stop(self):
        "Checking reads with only a stop value"
        if verbose:
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
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 0
        assert len(row[1]) == 10
        assert len(row[2]) == 99
        assert allequal(row[0], arange(0, type=Int32))
        for x in range(10):
            assert allequal(row[1][x], arange(x, type=Int32))
        for x in range(99):
            assert allequal(row[2][x], arange(x, type=Int32))

    def test02b_stop(self):
        "Checking reads with only a stop value in a slice"
        if verbose:
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
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 1
        assert len(row[1]) == 10
        assert len(row[2]) == 99
        for x in range(1):
            assert allequal(row[0][x], arange(0, type=Int32))
        for x in range(10):
            assert allequal(row[1][x], arange(x, type=Int32))
        for x in range(99):
            assert allequal(row[2][x], arange(x, type=Int32))


    def test03_startstop(self):
        "Checking reads with a start and stop values"
        if verbose:
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
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 10
        assert len(row[1]) == 10
        assert len(row[2]) == 100
        for x in range(0,10):
            assert allequal(row[0][x], arange(x, type=Int32))
        for x in range(5,15):
            assert allequal(row[1][x-5], arange(x, type=Int32))
        for x in range(0,100):
            assert allequal(row[2][x], arange(x, type=Int32))

    def test03b_startstop(self):
        "Checking reads with a start and stop values in slices"
        if verbose:
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
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 10
        assert len(row[1]) == 10
        assert len(row[2]) == 100
        for x in range(0,10):
            assert allequal(row[0][x], arange(x, type=Int32))
        for x in range(5,15):
            assert allequal(row[1][x-5], arange(x, type=Int32))
        for x in range(0,100):
            assert allequal(row[2][x], arange(x, type=Int32))

    def test04_startstopstep(self):
        "Checking reads with a start, stop & step values"
        if verbose:
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
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 5
        assert len(row[1]) == 4
        assert len(row[2]) == 5
        for x in range(0,10,2):
            assert allequal(row[0][x/2], arange(x, type=Int32))
        for x in range(5,15,3):
            assert allequal(row[1][(x-5)/3], arange(x, type=Int32))
        for x in range(0,100,20):
            assert allequal(row[2][x/20], arange(x, type=Int32))

    def test04b_slices(self):
        "Checking reads with start, stop & step values in slices"
        if verbose:
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
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "Second row in vlarray ==>", row[1]

        assert vlarray.nrows == self.nrows
        assert len(row[0]) == 5
        assert len(row[1]) == 4
        assert len(row[2]) == 5
        for x in range(0,10,2):
            assert allequal(row[0][x/2], arange(x, type=Int32))
        for x in range(5,15,3):
            assert allequal(row[1][(x-5)/3], arange(x, type=Int32))
        for x in range(0,100,20):
            assert allequal(row[2][x/20], arange(x, type=Int32))

    def test05_out_of_range(self):
        "Checking out of range reads"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_out_of_range..." % self.__class__.__name__

        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.root.vlarray
        
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            row = vlarray.read(1000)
            print "row-->", row
        except IndexError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
		print "\nGreat!, the next IndexError was catched!"
                print value
	    self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a IndexError and got:\n%s" % value)

class CopyTestCase(unittest.TestCase):

    def test01_copy(self):
        """Checking VLArray.copy() method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Vlarray
        arr = Int16Atom(shape=2, flavor="List")
        array1 = fileh.createVLArray(fileh.root, 'array1', arr, "title array1")
        array1.append([[2,3]])
        array1.append()  # an empty row
        array1.append([[3, 457],[2,4]])

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
            print "array1-->", repr(array1.read())
            print "array2-->", repr(array2.read())
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            
        # Check that all the elements are equal
        assert array1.read() == array2.read()

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.atom.flavor == array2.atom.flavor
        assert array1.atom.type == array2.atom.type
        assert array1.atom.itemsize == array2.atom.itemsize
        assert repr(array1.atom) == repr(array1.atom)

        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking VLArray.copy() method (where specified)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        arr = Int16Atom(shape=2, flavor="List")
        array1 = fileh.createVLArray(fileh.root, 'array1', arr, "title array1")
        array1.append([[2,3]])
        array1.append()  # an empty row
        array1.append([[3, 457],[2,4]])

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
        assert array1.read() == array2.read()

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.shape == array2.shape
        assert array1.atom.flavor == array2.atom.flavor
        assert array1.atom.type == array2.atom.type
        assert array1.atom.itemsize == array2.atom.itemsize
        assert repr(array1.atom) == repr(array1.atom)
        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test03_copy(self):
        """Checking VLArray.copy() method (Numeric flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        if numeric:
            arr = Int16Atom(shape=2, flavor="Numeric")
        else:
            arr = Int16Atom(shape=2, flavor="NumArray")
        array1 = fileh.createVLArray(fileh.root, 'array1', arr,
                                     "title array1")
        array1.append([[2,3]])
        array1.append()  # an empty row
        array1.append([[3, 457],[2,4]])
            
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
        assert array1.atom.flavor == array2.atom.flavor  # Very important here
        assert array1.atom.type == array2.atom.type
        assert array1.atom.itemsize == array2.atom.itemsize
        assert repr(array1.atom) == repr(array1.atom)
        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test03b_copy(self):
        """Checking VLArray.copy() method (Tuple flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        arr = Int16Atom(shape=2, flavor="Tuple")
        array1 = fileh.createVLArray(fileh.root, 'array1', arr,
                                     "title array1")
        array1.append(((2,3),))
        array1.append()  # an empty row
        array1.append(((3, 457),(2,4)))
            
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
        assert array1.atom.flavor == array2.atom.flavor  # Very important here
        assert array1.atom.type == array2.atom.type
        assert array1.atom.itemsize == array2.atom.itemsize
        assert repr(array1.atom) == repr(array1.atom)
        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking VLArray.copy() method (checking title copying)"""

        if verbose:
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
        array1.append()  # an empty row
        array1.append(((3, 457),(2,4)))
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
        """Checking VLArray.copy() method (user attributes copied)"""

        if verbose:
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
        array1.append()  # an empty row
        array1.append(((3, 457),(2,4)))
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

    def notest05b_copy(self):
        """Checking VLArray.copy() method (user attributes not copied)"""

        if verbose:
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
        array1.append()  # an empty row
        array1.append(((3, 457),(2,4)))
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
        array2.attrs.attr1 == None
        array2.attrs.attr2 == None

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

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an VLArray
        atom = Int32Atom(shape=(2,), flavor="List")
        array1 = fileh.createVLArray(fileh.root, 'array1', atom, "t array1")
        # The next creates 20 rows of variable length
        r = []
        for row in range(20):
            r.append([[row, row+1]])
            array1.append([row, row+1])
        
        if self.close:
            if verbose:
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
        if verbose:
            print "r2-->", r2
            print "array2-->", array2[:]
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", len(r2)
        # Check that all the elements are equal
        assert r2 == array2[:]
        # Assert the number of rows in array
        assert len(r2) == array2.nrows

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_indexclosef(self):
        """Checking VLArray.copy() method with indexes (close file version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_indexclosef..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an EArray
        atom = Int32Atom(shape=(2,), flavor="Tuple")
        array1 = fileh.createVLArray(fileh.root, 'array1', atom, "t array1")
        # The next creates 20 rows of variable length
        r = []
        for row in range(20):
            r.append(([row, row+1],))
            array1.append([row, row+1])
        
        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "a")
            array1 = fileh.root.array1

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

        r2 = r[self.start:self.stop:self.step]
        if verbose:
            print "r2-->", r2
            print "array2-->", array2[:]
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", len(r2)
        # Check that all the elements are equal
        assert r2 == array2[:]
        # Assert the number of rows in array
        assert len(r2) == array2.nrows

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



#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1

    #theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
    #theSuite.addTest(unittest.makeSuite(OpenAppendShapeTestCase))
    #theSuite.addTest(unittest.makeSuite(CloseAppendShapeTestCase))
    #theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
    #theSuite.addTest(unittest.makeSuite(BasicNumArrayTestCase))
    #if numeric:
    #    theSuite.addTest(unittest.makeSuite(BasicNumericTestCase))
    #theSuite.addTest(unittest.makeSuite(BasicTupleTestCase))
    #theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
    #theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
    #theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
    #theSuite.addTest(unittest.makeSuite(TypesNumArrayTestCase))
    #theSuite.addTest(unittest.makeSuite(MDTypesNumArrayTestCase))
    #theSuite.addTest(unittest.makeSuite(TupleFlavorTestCase))
    #theSuite.addTest(unittest.makeSuite(ListFlavorTestCase))
    #theSuite.addTest(unittest.makeSuite(NumericFlavorTestCase))
    #theSuite.addTest(unittest.makeSuite(RangeTestCase))
    #theSuite.addTest(unittest.makeSuite(NumArrayFlavorTestCase))
    
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicNumArrayTestCase))
        if numeric:
            theSuite.addTest(unittest.makeSuite(BasicNumericTestCase))
        theSuite.addTest(unittest.makeSuite(BasicTupleTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
        theSuite.addTest(unittest.makeSuite(TypesNumArrayTestCase))
        theSuite.addTest(unittest.makeSuite(MDTypesNumArrayTestCase))
        theSuite.addTest(unittest.makeSuite(OpenAppendShapeTestCase))
        theSuite.addTest(unittest.makeSuite(CloseAppendShapeTestCase))
        theSuite.addTest(unittest.makeSuite(TupleFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(ListFlavorTestCase))
        if numeric:
            theSuite.addTest(unittest.makeSuite(NumericFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(NumArrayFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(RangeTestCase))
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
    
    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
