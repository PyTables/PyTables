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
    fileh = openFile(filename, mode = "w")
    root = fileh.root
    try:
        # Create the array under root and name 'somearray'
        a = testTuple
        fileh.createArray(root, 'somearray', a, "Some array")
    finally:
        # Close the file
        fileh.close()

    # Re-open the file in read-only mode
    fileh = openFile(filename, mode = "r")
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
        return

    def test01_types(self):
        "Data integrity during recovery (numerical types)"

        a = self.numericalList
        fname = tempfile.mktemp(".h5")
        try:
            WriteRead(fname, a)
        finally:
            os.remove(fname)
        return

class Basic0DOneTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 1"
    numericalList = 3
    charList = "3"

class Basic0DTwoTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 2"
    numericalList = 33.34
    charList = "33"*500

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
    charList = ["a"]

class Basic1DTwoTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 2"
    numericalList = [3.2, 4.2]
    charList = ["aaa"]

class Basic2DTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2 case 1"
    numericalList = [[1,2]]*5
    charList = [["qq","zz"]]*5

class Basic10DTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 case 1"
    numericalList = [[[[[[[[[[1,2],[3,4]]]]]]]]]]*5
    # Dimensions greather than 6 in strings gives some warnings
    charList = [[[[[[[[[["a","b"],["qq","zz"]]]]]]]]]]*5


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


        return

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

        return


class Basic1DFourTestCase(ExceptionTestCase):
    title = "Rank-1 case 4 (non-regular list)"
    numericalList = [3, [4, 5.2]]
    charList = ["aaa", ["bbb", "ccc"]]


class GetItemTestCase(unittest.TestCase):

    def test00_single(self):
        "Single element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charList
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original first element:", a[0]
            print "Read first element:", arr[0]
        self.assertEqual(a[0], arr[0])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test01_single(self):
        "Single element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original first element:", a[0]
            print "Read first element:", arr[0]
        self.assertEqual(a[0], arr[0])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test02_range(self):
        "Range element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4]
            print "Read elements:", arr[1:4]
        self.assertEqual(a[1:4], arr[1:4])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test03_range(self):
        "Range element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4]
            print "Read elements:", arr[1:4]
        self.assertEqual(a[1:4], arr[1:4])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test04_range(self):
        "Range element access, strided (character types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4:2]
            print "Read elements:", arr[1:4:2]
        self.assertEqual(a[1:4:2], arr[1:4:2])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test05_range(self):
        "Range element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original elements:", a[1:4:2]
            print "Read elements:", arr[1:4:2]
        self.assertEqual(a[1:4:2], arr[1:4:2])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test06_negativeIndex(self):
        "Negative Index element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original last element:", a[-1]
            print "Read last element:", arr[-1]
        self.assertEqual(a[-1], arr[-1])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test07_negativeIndex(self):
        "Negative Index element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original before last element:", a[-2]
            print "Read before last element:", arr[-2]
        self.assertEqual(a[-2], arr[-2])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test08_negativeRange(self):
        "Negative range element access (character types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original last elements:", a[-4:-1]
            print "Read last elements:", arr[-4:-1]
        self.assertEqual(a[-4:-1], arr[-4:-1])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

    def test09_negativeRange(self):
        "Negative range element access (numerical types)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if common.verbose:
            print "Original last elements:", a[-4:-1]
            print "Read last elements:", arr[-4:-1]
        self.assertEqual(a[-4:-1], arr[-4:-1])

        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

class GI1ListTestCase(GetItemTestCase):
    title = "Rank-1 case 1 (lists)"
    numericalList = [3]
    numericalListME = [3,2,1,0,4,5,6]
    charList = ["3"]
    charListME = ["321","221","121","021","421","521","621"]

class GI2ListTestCase(GetItemTestCase):
    # A more complex example
    title = "Rank-1,2 case 2 (lists)"
    numericalList = [3,4]
    numericalListME = [[3,2,1,0,4,5,6],
                       [2,1,0,4,5,6,7],
                       [4,3,2,1,0,4,5],
                       [3,2,1,0,4,5,6],
                       [3,2,1,0,4,5,6]]

    charList = ["a","b"]
    charListME = [["321","221","121","021","421","521","621"],
                  ["21","21","11","02","42","21","61"],
                  ["31","21","12","21","41","51","621"],
                  ["321","221","121","021","421","521","621"],
                  ["3241","2321","13216","0621","4421","5421","a621"],
                  ["a321","s221","d121","g021","b421","5vvv21","6zxzxs21"]]


class GeneratorTestCase(unittest.TestCase):

    def test00a_single(self):
        "Testing generator access to Arrays, single elements (char)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charList
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

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
        return

    def test00b_me(self):
        "Testing generator access to Arrays, multiple elements (char)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if type(a[0]) == tuple:
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
        return

    def test01a_single(self):
        "Testing generator access to Arrays, single elements (numeric)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

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
        return

    def test01b_me(self):
        "Testing generator access to Arrays, multiple elements (numeric)"

        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if type(a[0]) == tuple:
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
        return

class GE1ListTestCase(GeneratorTestCase):
    # Scalar case
    title = "Rank-1 case 1 (lists)"
    numericalList = [3]
    numericalListME = [3,2,1,0,4,5,6]
    charList = ["3"]
    charListME = ["321","221","121","021","421","521","621"]

class GE2ListTestCase(GeneratorTestCase):
    # Scalar case
    title = "Rank-1,2 case 2 (lists)"
    numericalList = [3,4]
    numericalListME = [[3,2,1,0,4,5,6],
                       [2,1,0,4,5,6,7],
                       [4,3,2,1,0,4,5],
                       [3,2,1,0,4,5,6],
                       [3,2,1,0,4,5,6]]

    charList = ["a","b"]
    charListME = [["321","221","121","021","421","521","621"],
                  ["21","21","11","02","42","21","61"],
                  ["31","21","12","21","41","51","621"],
                  ["321","221","121","021","421","521","621"],
                  ["3241","2321","13216","0621","4421","5421","a621"],
                  ["a321","s221","d121","g021","b421","5vvv21","6zxzxs21"]]


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
