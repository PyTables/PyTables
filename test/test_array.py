import sys
import unittest
import os
import tempfile
from Numeric import *
from tables import File, IsRecord, isHDF5

from test_all import verbose

class TypesTestCase(unittest.TestCase):
    
    def test00_integerArray(self):
        file = tempfile.mktemp(".h5")
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_integerArray..." % self.__class__.__name__
            #print "Filename ==>", file

        # Create an instance of HDF5 Table
        fileh = File(name = file, mode = "w")
        root = fileh.getRootGroup()
        # Create an integer array
	#a = array([1,2,3],'i')
	a = array([[[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
                   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
                   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
                   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
                   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
		   ], "d")
        hdfarray = fileh.newArray(root, 'array_i', a)
        # Close the file
        fileh.close()
	# Re-open the file in read-only mode
        fileh = File(name = file, mode = "r")
        root = fileh.getRootGroup()
	# Read the saved array
	b = fileh.readArray(root.array_i)
        if verbose:
	    print "Retrieved array:"
	    print b
	# Compare them. They should be equal.
	assert a == b
        # Then, delete the file
        os.remove(file)


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(TypesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
