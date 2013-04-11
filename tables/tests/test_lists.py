# -*- coding: utf-8 -*-

import sys
import unittest
import os
import tempfile

from tables import *
from tables.tests import common

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

def WriteRead(filename, testTuple):
    if common.verbose:
        print '\n', '-=' * 30
        print "Running test for object %s" % \
                  (type(testTuple))

    # Create an instance of HDF5 Table
    fileh = open_file(filename, mode = "w")
    root = fileh.root
    try:
        # Create the array under root and name 'somearray'
        a = testTuple
        fileh.create_array(root, 'somearray', a, "Some array")
    finally:
        # Close the file
        fileh.close()

    # Re-open the file in read-only mode
    fileh = open_file(filename, mode = "r")
    root = fileh.root

    # Read the saved array
    try:
        b = root.somearray.read()
        # Compare them. They should be equal.
        if not a == b and common.verbose:
            print "Write and read lists/tuples differ!"
            print "Object written:", a
            print "Object read:", b

        # Check strictly the array equality
        assert a == b
    finally:
        # Close the file
        fileh.close()


class BasicTestCase(unittest.TestCase):
    def test00_char(self):
        "Data integrity during recovery (character types)"

        a = self.charList
        fname = tempfile.mktemp(".h5")
        try:
            WriteRead(fname, a)
        finally:
            os.remove(fname)

    def test01_types(self):
        "Data integrity during recovery (numerical types)"

        a = self.numericalList
        fname = tempfile.mktemp(".h5")
        try:
            WriteRead(fname, a)
        finally:
            os.remove(fname)


class Basic0DOneTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 1"
    numericalList = 3
    charList = b"3"

class Basic0DTwoTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 2"
    numericalList = 33.34
    charList = b"33"*500

# This does not work anymore because I've splitted the chunked arrays to happen
# mainly in EArray objects
# class Basic1DZeroTestCase(BasicTestCase):
#     title = "Rank-1 case 0"
#     numericalList = []
#     charList = []

class Basic1DOneTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 1"
    numericalList = [3]
    charList = [b"a"]


class Basic1DTwoTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 2"
    numericalList = [3.2, 4.2]
    charList = [b"aaa"]


class Basic2DTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2 case 1"
    numericalList = [[1, 2]]*5
    charList = [[b"qq", b"zz"]]*5


class Basic10DTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 case 1"
    numericalList = [[[[[[[[[[1, 2], [3, 4]]]]]]]]]]*5
    # Dimensions greather than 6 in strings gives some warnings
    charList = [[[[[[[[[[b"a", b"b"], [b"qq", b"zz"]]]]]]]]]]*5


class ExceptionTestCase(unittest.TestCase):
    def test00_char(self):
        "Non suppported lists objects (character objects)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running test for %s" % \
                  (self.title)
        a = self.charList
        try:
            fname = tempfile.mktemp(".h5")
            try:
                WriteRead(fname, a)
            finally:
                os.remove(fname)
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next error was catched!"
                print type, ":", value
        else:
            self.fail("expected a ValueError")

    def test01_types(self):
        "Non supported lists object (numerical types)"

        a = self.numericalList
        try:
            fname = tempfile.mktemp(".h5")
            try:
                WriteRead(fname, a)
            finally:
                os.remove(fname)
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next was catched!"
                print value
        else:
            self.fail("expected an ValueError")


class Basic1DFourTestCase(ExceptionTestCase):
    title = "Rank-1 case 4 (non-regular list)"
    numericalList = [3, [4, 5.2]]
    charList = [b"aaa", [b"bbb", b"ccc"]]


class GetItemTestCase(unittest.TestCase):
    def test00_single(self):
        "Single element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charList
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original first element:", a[0]
            print "Read first element:", arr[0]
        self.assertEqual(a[0], arr[0])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test01_single(self):
        "Single element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original first element:", a[0]
            print "Read first element:", arr[0]
        self.assertEqual(a[0], arr[0])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test02_range(self):
        "Range element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4]
            print "Read elements:", arr[1:4]
        self.assertEqual(a[1:4], arr[1:4])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test03_range(self):
        "Range element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4]
            print "Read elements:", arr[1:4]
        self.assertEqual(a[1:4], arr[1:4])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test04_range(self):
        "Range element access, strided (character types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4:2]
            print "Read elements:", arr[1:4:2]
        self.assertEqual(a[1:4:2], arr[1:4:2])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test05_range(self):
        "Range element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4:2]
            print "Read elements:", arr[1:4:2]
        self.assertEqual(a[1:4:2], arr[1:4:2])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test06_negativeIndex(self):
        "Negative Index element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original last element:", a[-1]
            print "Read last element:", arr[-1]
        self.assertEqual(a[-1], arr[-1])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test07_negativeIndex(self):
        "Negative Index element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original before last element:", a[-2]
            print "Read before last element:", arr[-2]
        self.assertEqual(a[-2], arr[-2])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test08_negativeRange(self):
        "Negative range element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original last elements:", a[-4:-1]
            print "Read last elements:", arr[-4:-1]
        self.assertEqual(a[-4:-1], arr[-4:-1])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test09_negativeRange(self):
        "Negative range element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original last elements:", a[-4:-1]
            print "Read last elements:", arr[-4:-1]
        self.assertEqual(a[-4:-1], arr[-4:-1])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)


class GI1ListTestCase(GetItemTestCase):
    title = "Rank-1 case 1 (lists)"
    numericalList = [3]
    numericalListME = [3, 2, 1, 0, 4, 5, 6]
    charList = [b"3"]
    charListME = [b"321", b"221", b"121", b"021", b"421", b"521", b"621"]


class GI2ListTestCase(GetItemTestCase):
    # A more complex example
    title = "Rank-1,2 case 2 (lists)"
    numericalList = [3, 4]
    numericalListME = [[3, 2, 1, 0, 4, 5, 6],
                       [2, 1, 0, 4, 5, 6, 7],
                       [4, 3, 2, 1, 0, 4, 5],
                       [3, 2, 1, 0, 4, 5, 6],
                       [3, 2, 1, 0, 4, 5, 6]]

    charList = [b"a", b"b"]
    charListME = [[b"321", b"221", b"121", b"021", b"421", b"521", b"621"],
                  [b"21", b"21", b"11", b"02", b"42", b"21", b"61"],
                  [b"31", b"21", b"12", b"21", b"41", b"51", b"621"],
                  [b"321", b"221", b"121", b"021", b"421", b"521", b"621"],
                  [b"3241", b"2321", b"13216", b"0621", b"4421", b"5421", b"a621"],
                  [b"a321", b"s221", b"d121", b"g021", b"b421", b"5vvv21", b"6zxzxs21"]]


class GeneratorTestCase(unittest.TestCase):
    def test00a_single(self):
        "Testing generator access to Arrays, single elements (char)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charList
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        self.assertEqual(ga, garr)

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test00b_me(self):
        "Testing generator access to Arrays, multiple elements (char)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if isinstance(a[0], tuple):
            ga = [list(i) for i in a]
        else:
            ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        self.assertEqual(ga, garr)

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test01a_single(self):
        "Testing generator access to Arrays, single elements (numeric)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        self.assertEqual(ga, garr)

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)

    def test01b_me(self):
        "Testing generator access to Arrays, multiple elements (numeric)"

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.create_array(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if isinstance(a[0], tuple):
            ga = [list(i) for i in a]
        else:
            ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        self.assertEqual(ga, garr)

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)


class GE1ListTestCase(GeneratorTestCase):
    # Scalar case
    title = "Rank-1 case 1 (lists)"
    numericalList = [3]
    numericalListME = [3, 2, 1, 0, 4, 5, 6]
    charList = [b"3"]
    charListME = [b"321", b"221", b"121", b"021", b"421", b"521", b"621"]


class GE2ListTestCase(GeneratorTestCase):
    # Scalar case
    title = "Rank-1,2 case 2 (lists)"
    numericalList = [3, 4]
    numericalListME = [[3, 2, 1, 0, 4, 5, 6],
                       [2, 1, 0, 4, 5, 6, 7],
                       [4, 3, 2, 1, 0, 4, 5],
                       [3, 2, 1, 0, 4, 5, 6],
                       [3, 2, 1, 0, 4, 5, 6]]

    charList = [b"a", b"b"]
    charListME = [[b"321", b"221", b"121", b"021", b"421", b"521", b"621"],
                  [b"21", b"21", b"11", b"02", b"42", b"21", b"61"],
                  [b"31", b"21", b"12", b"21", b"41", b"51", b"621"],
                  [b"321", b"221", b"121", b"021", b"421", b"521", b"621"],
                  [b"3241", b"2321", b"13216", b"0621", b"4421", b"5421", b"a621"],
                  [b"a321", b"s221", b"d121", b"g021", b"b421", b"5vvv21", b"6zxzxs21"]]


def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
        #theSuite.addTest(unittest.makeSuite(Basic1DZeroTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DFourTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2DTestCase))
        theSuite.addTest(unittest.makeSuite(Basic10DTestCase))
        theSuite.addTest(unittest.makeSuite(GI1ListTestCase))
        theSuite.addTest(unittest.makeSuite(GI2ListTestCase))
        theSuite.addTest(unittest.makeSuite(GE1ListTestCase))
        theSuite.addTest(unittest.makeSuite(GE2ListTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )






