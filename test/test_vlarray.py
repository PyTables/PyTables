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
    mode  = "w" 
    title = "This is the table title"
    compress = 0
    complib = "zlib"  # Default compression library

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
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
            print "Total rows in vlarray ==> ", vlarray.nrows
        nrows = 4
        assert nrows == vlarray.nrows
        assert allequal(row, array([1, 2]))
        assert len(row) == 2


class BasicWriteTestCase(BasicTestCase):
    title = "BasicWrite"

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
            print "Total rows in vlarray ==> ", vlarray.nrows
            
        assert vlarray.nrows == 2
        assert allequal(row[0], strings.array(["123", "456", "3"]))
        assert allequal(row[1], strings.array(["456", "3"]))
        assert len(row[0]) == 3
        assert len(row[1]) == 2

    def test0b1_StringAtom(self):
        """Checking vlarray with numarray string atoms (String flavor)"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_StringAtom..." % self.__class__.__name__

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
            print "Total rows in vlarray ==> ", vlarray.nrows
            
        assert vlarray.nrows == 2
        assert row[0] == ("123", "456", "3")
        assert row[1] == ("456", "3")
        assert len(row[0]) == 3
        assert len(row[1]) == 2

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
            print "Total rows in vlarray ==> ", vlarray.nrows
            
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
                print "Total rows in vlarray ==> ", vlarray.nrows

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
                print "Total rows in vlarray ==> ", vlarray.nrows

            assert vlarray.nrows == 2
            assert allequal(row[0], array([1.3,2.2,3.3], ttypes[atype]))
            assert allequal(row[1], array([-1.3e34,1.e-32], ttypes[atype]))
            assert len(row[0]) == 3
            assert len(row[1]) == 2

    def test05_VLString(self):
        """Checking vlarray with variable length strings"""

        root = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_VLString..." % self.__class__.__name__

        # Create an string atom
        vlarray = self.fileh.createVLArray(root, "VLString", VLString())
        vlarray.append(u"asd")
        vlarray.append(u"aaañá")

        # Read all the rows:
        row = vlarray.read()
        if verbose:
            print "Object read:", row
            print "Nrows in", vlarray._v_pathname, ":", vlarray.nrows
            print "First row in vlarray ==>", row[0]
            print "Total rows in vlarray ==> ", vlarray.nrows

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
            print "Total rows in vlarray ==> ", vlarray.nrows

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
            print "Total rows in vlarray ==> ", vlarray.nrows

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
                print "Total rows in vlarray ==> ", vlarray.nrows

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
                print "Total rows in vlarray ==> ", vlarray.nrows

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


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    global numeric
    niter = 0

    theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    theSuite.addTest(unittest.makeSuite(TypesNumArrayTestCase))
    theSuite.addTest(unittest.makeSuite(TupleFlavorTestCase))
    theSuite.addTest(unittest.makeSuite(ListFlavorTestCase))
    if numeric:
        theSuite.addTest(unittest.makeSuite(NumericFlavorTestCase))

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
            
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
