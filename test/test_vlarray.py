# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import sys
import unittest
import os
import tempfile

#import numarray
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

class C:
    c = (3,4.5) 

class BasicTestCase(unittest.TestCase):
    compress = 0
    complib = "zlib"  # Default compression library

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
        vlarray = self.fileh.createVLArray(group, 'vlarray1', Int32Atom(),
                                           "ragged array if ints",
                                           compress = self.compress,
                                           complib = self.complib,
                                           expectedsizeinMB = 1)

        # Fill it with 4 rows
        vlarray.append(1, 2)
        vlarray.append(array([3, 4, 5]))
        vlarray.append([6, 7, 8, 9])
        vlarray.append(10, 11, 12, 13, 14)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    #----------------------------------------

    def test01_readVLArray(self):
        """Checking vlarray read and cuts"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readVLArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        vlarray = self.fileh.getNode("/vlarray1")

        # Choose a small value for buffer size
        vlarray._nrowsinbuf = 3
        # Read the first row:
        row = vlarray.read(0)
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row
            
        nrows = 4
        assert nrows == vlarray.nrows
        assert allequal(row, array([1, 2]))
        assert len(row) == 2

    def test02_emptyVLArray(self):
        """Checking creation of empty VL arrays"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_emptyVLArray..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "w")
        vlarray = self.fileh.createVLArray(self.fileh.root, 'vlarray2',
                                           Int32Atom(),
                                           "ragged array if ints",
                                           compress = self.compress,
                                           complib = self.complib)
        # Try to read info from there:
        row = vlarray.read()
        # The result should be the empty list
        assert row == []

class BasicWriteTestCase(BasicTestCase):
    title = "BasicWrite"

class ZlibComprTestCase(BasicTestCase):
    title = "ZlibCompr"
    compress = 1
    complib = "zlib"  # Default compression library

class LZOComprTestCase(BasicTestCase):
    title = "LZOCompr"
    compress = 1
    complib = "lzo"  # Default compression library

class UCLComprTestCase(BasicTestCase):
    title = "UCLCompr"
    compress = 1
    complib = "ucl"  # Default compression library

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
    # supported because the strings can be cut in the middle of a utf-8
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
                  "UInt64": UInt64,
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

    def test03_IntAtom(self):
        """Checking vlarray with MD integer atoms"""

        ttypes = {"Int8": Int8,
                  "UInt8": UInt8,
                  "Int16": Int16,
                  "UInt16": UInt16,
                  "Int32": Int32,
                  "UInt32": UInt32,
                  "Int64": Int64,
                  "UInt64": UInt64,
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
            assert allequal(row[0], array([ones((2,3), ttypes[atype]),
                                          zeros((2,3), ttypes[atype])]))
            assert allequal(row[1], array([ones((2,3), ttypes[atype])*100]))
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
            assert allequal(row[0], array([ones((5,2,6), ttypes[atype])*1.3,
                                          zeros((5,2,6), ttypes[atype])]))
            assert allequal(row[1], array([ones((5,2,6), ttypes[atype])*2.e4]))
            assert len(row[0]) == 2
            assert len(row[1]) == 1


class MDTypesNumArrayTestCase(MDTypesTestCase):
    title = "MDTypes"

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
        vlarray.append(100,0)

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Testing flavor:", self.flavor
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]

        assert vlarray.nrows == 2
        assert len(row[0]) == 3
        assert len(row[1]) == 2
        if self.flavor == "Tuple":
            arr1 = (1,1,1)
            arr2 = (1,0)                    
        elif self.flavor == "List":
            arr1 = [1,1,1]
            arr2 = [1,0]
        elif self.flavor == "Numeric":
            arr1 = Numeric.array([1,1,1], typecode="1")
            arr2 = Numeric.array([1,0], typecode="1")
        assert row[0] == arr1
        assert row[1] == arr2

    def test03_IntAtom(self):
        """Checking vlarray with different flavors (integer versions)"""

        ttypes = {"Int8": Int8,
                  "UInt8": UInt8,
                  "Int16": Int16,
                  "UInt16": UInt16,
                  "Int32": Int32,
                  #"UInt32": UInt32,   # Not checked
                  #"Int64": Int64,     # Not checked
                  #"UInt64": UInt64,   # Not checked
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
            vlarray.append(100,0)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 2
            assert len(row[0]) == 3
            assert len(row[1]) == 2
            if self.flavor == "Tuple":
                arr1 = (1,2,3)
                arr2 = (100,0)                    
            elif self.flavor == "List":
                arr1 = [1,2,3]
                arr2 = [100,0]
            elif self.flavor == "Numeric":
                arr1 = Numeric.array([1,2,3], typecode=typecode[ttypes[atype]])
                arr2 = Numeric.array([100,0], typecode=typecode[ttypes[atype]])
            assert row[0] == arr1
            assert row[1] == arr2

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
            vlarray.append(-1.3e34,1.e-32)

            # Read all the rows:
            row = vlarray.read()
            if verbose:
                print "Testing flavor:", self.flavor
                print "Object read:", row
                print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
                print "First row in vlarray ==>", row[0]

            assert vlarray.nrows == 2
            assert len(row[0]) == 3
            assert len(row[1]) == 2
            if self.flavor == "Tuple":
                arr1 = tuple(array([1.3,2.2,3.3], typecode[ttypes[atype]]))
                arr2 = tuple(array([-1.3e34,1.e-32], typecode[ttypes[atype]]))
            elif self.flavor == "List":
                arr1 = list(array([1.3,2.2,3.3], typecode[ttypes[atype]]))
                arr2 = list(array([-1.3e34,1.e-32], typecode[ttypes[atype]]))
            elif self.flavor == "Numeric":
                arr1 = Numeric.array([1.3,2.2,3.3], typecode[ttypes[atype]])
                arr2 = Numeric.array([-1.3e34,1.e-32], typecode[ttypes[atype]])
            if self.flavor == "Numeric":
                Numeric.alltrue(Numeric.equal(row[0],arr1))
                Numeric.alltrue(Numeric.equal(row[1],arr2))
            else:
                assert row[0] == arr1
                assert row[1] == arr2

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
        vlarray = self.fileh.createVLArray(group, 'vlarray', Int32Atom(),
                                           "ragged array if ints",
                                           compress = self.compress,
                                           complib = self.complib,
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

        fileh = openFile(self.file, "r")
        vlarray = fileh.root.vlarray
        
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
        assert allequal(row[0], arange(0))
        assert allequal(row[1], arange(10))
        assert allequal(row[2], arange(99))

    def test02_stop(self):
        "Checking reads with only a stop value"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_stop..." % self.__class__.__name__

        fileh = openFile(self.file, "r")
        vlarray = fileh.root.vlarray
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
        assert allequal(row[0], arange(0))
        for x in range(10):
            assert allequal(row[1][x], arange(x))
        for x in range(99):
            assert allequal(row[2][x], arange(x))


    def test03_startstop(self):
        "Checking reads with a start and stop values"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_startstop..." % self.__class__.__name__

        fileh = openFile(self.file, "r")
        vlarray = fileh.root.vlarray
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
            assert allequal(row[0][x], arange(x))
        for x in range(5,15):
            assert allequal(row[1][x-5], arange(x))
        for x in range(0,100):
            assert allequal(row[2][x], arange(x))

    def test04_startstopstep(self):
        "Checking reads with a start, stop & step values"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_startstopstep..." % self.__class__.__name__

        fileh = openFile(self.file, "r")
        vlarray = fileh.root.vlarray
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
            assert allequal(row[0][x/2], arange(x))
        for x in range(5,15,3):
            assert allequal(row[1][(x-5)/3], arange(x))
        for x in range(0,100,20):
            assert allequal(row[2][x/20], arange(x))

    def test04b_slices(self):
        "Checking reads with slices"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_slices..." % self.__class__.__name__

        fileh = openFile(self.file, "r")
        vlarray = fileh.root.vlarray
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
            assert allequal(row[0][x/2], arange(x))
        for x in range(5,15,3):
            assert allequal(row[1][(x-5)/3], arange(x))
        for x in range(0,100,20):
            assert allequal(row[2][x/20], arange(x))

    def test05_out_of_range(self):
        "Checking out of range reads"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_out_of_range..." % self.__class__.__name__

        fileh = openFile(self.file, "r")
        vlarray = fileh.root.vlarray
        
        if verbose:
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows

        try:
            row = vlarray.read(1000)
        except IndexError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
		print "\nGreat!, the next IndexError was catched!"
                print value
	    self.fileh.close()
        else:
            (type, value, traceback) = sys.exc_info()
            self.fail("expected a IndexError and got:\n%s" % value)


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 1

#     theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
#     theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
#     theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
#     theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
#     theSuite.addTest(unittest.makeSuite(TypesNumArrayTestCase))
#     theSuite.addTest(unittest.makeSuite(MDTypesNumArrayTestCase))
#     theSuite.addTest(unittest.makeSuite(TupleFlavorTestCase))
#     theSuite.addTest(unittest.makeSuite(ListFlavorTestCase))
#     if numeric:
#         theSuite.addTest(unittest.makeSuite(NumericFlavorTestCase))
#    theSuite.addTest(unittest.makeSuite(RangeTestCase))
    
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibComprTestCase))
        theSuite.addTest(unittest.makeSuite(LZOComprTestCase))
        theSuite.addTest(unittest.makeSuite(UCLComprTestCase))
        theSuite.addTest(unittest.makeSuite(TypesNumArrayTestCase))
        theSuite.addTest(unittest.makeSuite(MDTypesNumArrayTestCase))
        theSuite.addTest(unittest.makeSuite(TupleFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(ListFlavorTestCase))
        if numeric:
            theSuite.addTest(unittest.makeSuite(NumericFlavorTestCase))
        theSuite.addTest(unittest.makeSuite(RangeTestCase))
            
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
