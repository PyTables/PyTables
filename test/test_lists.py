import sys
import unittest
import os
import tempfile
#import warnings
from tables import *

from test_all import verbose

file = ""
fileh = None

def WriteRead(testTuple):
    if verbose:
        print '\n', '-=' * 30
        print "Running test for object %s" % \
                  (type(testTuple))

    # Create an instance of HDF5 Table
    global file
    global fileh
    file = tempfile.mktemp(".h5")
    fileh = openFile(file, mode = "w")
    root = fileh.root

    # Create the array under root and name 'somearray'
    a = testTuple

    fileh.createArray(root, 'somearray', a, "Some array")

    # Close the file
    fileh.close()

    # Re-open the file in read-only mode
    fileh = openFile(file, mode = "r")

    root = fileh.root

    # Read the saved array
    b = root.somearray.read()

    # Compare them. They should be equal.
    if not a == b and verbose:
        print "Write and read lists/tuples differ!"
        print "Object written:", a
        print "Object read:", b


    # Check strictly the array equality
    assert a == b

    # Close the file (eventually destroy the extended type)
    fileh.close()
    # Then, delete the file
    os.remove(file)
    return
    
class BasicTestCase(unittest.TestCase):

    def test00_char(self):
        "Data integrity during recovery (character objects)"

        a = self.charList
	WriteRead(a)
	return

    def test01_types(self):
        "Data integrity during recovery (numerical types)"
        
        a = self.numericalList
        WriteRead(a)
        return

class ExceptionTestCase(unittest.TestCase):

    def test00_char(self):
        "Non suppported lists objects (character objects)"

	global file
	global fileh
	
        a = self.charList
        try:
            WriteRead(a)
        except ValueError:
	    # Close the file (eventually destroy the extended type)
	    fileh.close()
	    # Then, delete the file
	    os.remove(file)
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
        else:
            self.fail("expected an ValueError")
            

	return

    def test01_types(self):
        "Non supported lists object (numerical types)"
        
	global file
	global fileh
	
        a = self.numericalList
        try:
            WriteRead(a)
        except ValueError:
	    # Close the file (eventually destroy the extended type)
	    fileh.close()
	    # Then, delete the file
	    os.remove(file)
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
        else:
            self.fail("expected an ValueError")
            
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
    
class Basic1DZeroTestCase(BasicTestCase):
    # This test works from pytables 0.8 on, because chunked arrays are being
    # supported
    title = "Rank-1 case 0"
    numericalList = []
    charList = []

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
    
class Basic1DThreeTestCase(BasicTestCase):
    # 1D case
    # Detection for tuples (only works for 0-dimensional tuples)
    # i.e. it won't work with ((23,2)) or [(23,2)]
    title = "Rank-1 case 3"
    numericalList = (3, 4, 5.2)
    charList = ("aaa", "bbb")
    
class Basic1DFourTestCase(ExceptionTestCase):
    # numeric is still not able to detect that
    # Activate when numarray support this kind of detection
    title = "Rank-1 case 4 (non-regular list)"
    numericalList = [3, [4, 5.2]]
    charList = ["aaa", ["bbb", "ccc"]]
    
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
    

def suite():
    theSuite = unittest.TestSuite()

    # The scalar case test should be refined in order to work
    theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
    theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DZeroTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DThreeTestCase))
    # Activate this line when numarray detects non-regular input as non-valid
    #theSuite.addTest(unittest.makeSuite(Basic1DFourTestCase))
    theSuite.addTest(unittest.makeSuite(Basic2DTestCase))
    theSuite.addTest(unittest.makeSuite(Basic10DTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
