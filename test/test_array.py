import sys
import unittest
import os
import tempfile
from Numeric import *
from tables import *

from test_all import verbose

class ArrayTestCase(unittest.TestCase):
    """Basic test for all the supported typecodes present in Numeric.
    All of them are included on pytables.
    """
    
    def setUp(self):
        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w")
        self.root = self.fileh.root

    def tearDown(self):
        # Close the file (eventually destroy the extended type)
        self.fileh.close()

        # Then, delete the file
        os.remove(self.file)

    def WriteRead(self, testArray):
        if verbose:
            print '\n', '-=' * 30
	    print "Running test for array with typecode '%s'" % \
	          testArray.typecode()

	# Create the array under root and name 'somearray'
	a = testArray
        #self.root.somearray = createArray(a, "Some array")
        self.fileh.createArray(self.root, 'somearray', a, "Some array")
	
        # Close the file
        self.fileh.close()
	
	# Re-open the file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        self.root = self.fileh.root
	
	# Read the saved array
	b = self.root.somearray.read()
	
	# Compare them. They should be equal.
	if a.tolist() <> b.tolist() and verbose:
	    print "Write and read arrays differ!"
	    print "Array written:", a
	    print "Array written shape:", a.shape
	    print "Array written itemsize:", a.itemsize()
	    print "Array written typecode:", a.typecode()
	    print "Array read:", b
	    print "Array read shape:", b.shape
	    print "Array read itemsize:", b.itemsize()
	    print "Array read typecode:", b.typecode()

        # Check the array equality in this way, not as in:
        # assert a == b
        # because the result is not what we want.
        assert a.shape == b.shape
        assert a.shape == self.root.somearray.shape
        if a.typecode() == "l":
            # Special expection. We have no way to distinguish between
            # "l" and "i" typecode, and we can consider them the same
            # to all practical effects
            assert b.typecode() == "i"
            assert self.root.somearray.typecode == "i"
        else:
            assert a.typecode() == b.typecode()
            assert a.typecode() ==  self.root.somearray.typecode
	assert a.tolist() == b.tolist()
	return
    
    def test00_char(self):
        "Checking a character array"
        
	a = array(['1', '2', '4'],
	          "c")
	self.WriteRead(a)
	return

    def test01_unsignedByte(self):
        "Checking a unsigned byte array"
        
	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "b")
	self.WriteRead(a)
	return
    
    def test02_signedByte(self):
        "Checking a signed byte array"
        
	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "1")
	self.WriteRead(a)
	return
    
    def test03_signedShort(self):
        "Checking a signed short integer array"
                
	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "s")
	self.WriteRead(a)
	return
        
    def test04_unsignedShort(self):
        "Checking an unsigned short integer array"
                
	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "w")
	self.WriteRead(a)
	return
        
    def test05_signedInt(self):
        "Checking a signed integer array"

	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "i")
	self.WriteRead(a)
	return
    
    def test06_unsignedInt(self):
        "Checking an unsigned integer array"

	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "u")
	self.WriteRead(a)
	return
    
    def test07_long(self):
        "Checking a signed long integer array"

	a = array([[[-1, -3, -8], [2, 3, 8],[3, 3, 8]],
		   [[4, 3, 8], [5, 3, 8],[6, 3, 8]],
		   [[7, 3, 8], [8, 3, 8],[9, 3, 8]],
		   [[10, 3, 8], [11, 3, 8],[12, 3, 8]],
		   ], "l")
	self.WriteRead(a)
	return
    
    def test08_float(self):
        "Checking a single precision floating point array"

	a = array([[[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
		   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
		   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
		   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
		   [[1.0, 3.5, 8.4], [1.0, 3.5, 8.4],[1.0, 3.5, 8.4]],
		   ], "f")
	self.WriteRead(a)
	return
    
    def test09_double(self):
        "Checking a double precision floating point array"

	a = array([[1.0, 3.5, 8.4],
	           [2.3, 6.6, 4.1],
		   [2.3, 6.6, 4.1],
		   [2.3, 6.6, 4.1],
		   [2.3, 6.6, 4.1],
		   ], "d")
	self.WriteRead(a)
	return

    def test10_complexSimple(self):
        "Checking a complex floating point array (not supported)"

	a = array([[1.0, 3.5],
	           [2.3, 6.6],
		   [2.3, 6.6],
		   [2.3, 6.6],
		   [2.3, 6.6],
		   ], "F")
        try:
            self.WriteRead(a)
        except TypeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next TypeError was catched!"
                print value
        else:
            self.fail("expected an TypeError")
            
    def test11_complexDouble(self):
        "Checking a complex floating point array (not supported)"

	a = array([[1.0, 3.5],
	           [2.3, 6.6],
		   [2.3, 6.6],
		   [2.3, 6.6],
		   [2.3, 6.6],
		   ], "D")
        try:
            self.WriteRead(a)
        except TypeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next TypeError was catched!"
                print value
        else:
            self.fail("expected an TypeError")
            

class GroupsArrayTestCase(unittest.TestCase):
    """This test class checks combinations of arrays with groups.
    It also uses arrays ranks which ranges until 10.
    """

    def test00_iterativeGroups(self):
	
	"""Checking combinations of arrays with groups
	It also uses arrays ranks which ranges until 10.
	"""
	
	if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_iterativeGroups..." % \
	          self.__class__.__name__
		  
	# Open a new empty HDF5 file
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
	
	# Get the root group
	group = fileh.root
	
	# Set the type codes to test
	typecodes = ["c", 'b', '1', 's', 'w', 'i', 'u', 'l', 'f', 'd']
	i = 1
	for typecode in typecodes:
	    # Create an array of typecode, with incrementally bigger ranges
	    a = ones((3,) * i, typecode)
	    # Save it on the HDF5 file
	    dsetname = 'array_' + typecode
	    if verbose:
		print "Creating dataset:", group._f_join(dsetname)
	    hdfarray = fileh.createArray(group, dsetname, a, "Large array")
	    # Create a new group
	    group = fileh.createGroup(group, 'group' + str(i))
	    # increment the range for next iteration
	    i += 1

	# Close the file
	fileh.close()

	# Open the previous HDF5 file in read-only mode
        fileh = openFile(file, mode = "r")
	# Get the root group
	group = fileh.root

	# Get the metadata on the previosly saved arrays
	for i in range(1,10):
	    # Create an array for later comparison
	    a = ones((3,) * i, typecodes[i - 1])
	    # Get the dset object hanging from group
	    dset = getattr(group, 'array_' + typecodes[i-1])
	    # Get the actual array
	    b = dset.read()
	    if verbose:
		print "Info from dataset:", dset._v_pathname
		print "  Shape: ==>", dset.shape, 
		print "  typecode ==> %c" % dset.typecode
		print "Array b read from file. Shape: ==>", b.shape,
		print ". Typecode ==> %c" % b.typecode()
	    assert a.shape == b.shape
            if a.typecode() == "l":
                # Special expection. We have no way to distinguish between
                # "l" and "i" typecode, and we can consider them the same
                # to all practical effects
                assert b.typecode() == "i"
            else:
                assert a.typecode() == b.typecode()            
            assert a.tolist() == b.tolist()
	    # Iterate over the next group
	    group = getattr(group, 'group' + str(i))

	# Close the file
	fileh.close()

	# Then, delete the file
        os.remove(file)

    def test01_largeRankArrays(self):
	"""Checking creation of large rank arrays (0 < rank <= 32)
	It also uses arrays ranks which ranges until maxrank.
	"""
	
	# maximum level of recursivity (deepest group level) achieved:
	# maxrank = 32 (for a effective maximum rank of 32)
        # This limit is due to HDF5 library limitations.
	# There seems to exist a bug in Numeric when dealing with
	# arrays with rank greater than 20. Also hdf5Extension has a
	# bug getting the shape of the object, that creates lots of
	# problems (segmentation faults, memory faults...)
	minrank = 1
	maxrank = 32
	
	if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_largeRankArrays..." % \
	          self.__class__.__name__
	    print "Maximum rank for tested arrays:", maxrank
	# Open a new empty HDF5 file
        #file = tempfile.mktemp(".h5")
        file = "test_array.h5"
        fileh = openFile(file, mode = "w")
	group = fileh.root
	if verbose:
	    print "Rank array writing progress: ",
	for rank in range(minrank, maxrank + 1):
	    # Create an array of integers, with incrementally bigger ranges
	    a = ones((1,) * rank, 'i')
	    if verbose:
		print "%3d," % (rank),
	    fileh.createArray(group, "array", a, "Rank: %s" % rank)
	    group = fileh.createGroup(group, 'group' + str(rank))
        # Flush the buffers
        fileh.flush()
	# Close the file
	fileh.close()
        
	# Open the previous HDF5 file in read-only mode
        fileh = openFile(file, mode = "r")
	group = fileh.root
	if verbose:
	    print
	    print "Rank array reading progress: "
	# Get the metadata on the previosly saved arrays
        for rank in range(minrank, maxrank + 1):
	    # Create an array for later comparison
	    a = ones((1,) * rank, 'i')
	    # Get the actual array
	    b = group.array.read()
	    if verbose:
		print "%3d," % (rank),
	    if verbose and 0:
		print "Info from dataset:", dset._v_pathname
		print "  Shape: ==>", dset.shape, 
		print "  typecode ==> %c" % dset.typecode
		print "Array b read from file. Shape: ==>", b.shape,
		print ". Typecode ==> %c" % b.typecode()
                
            # ************** WARNING!!! *****************
            # If we compare to arrays of dimensions bigger than 20
            # we get a segmentation fault! It is most probably a bug
            # located on Numeric package
            # I've discovered that comparing shapes and using the
            # tolist() conversion is the best to compare Numeric
            # arrays!. At least, tolist() do not crash!.
            # In addition, a == b is comparing the arrays element to
            # element and in ranks are different, the smaller is
            # promoted to the bigger rank. This is definitely not what
            # we want to do!!
            # ************** WARNING!!! *****************
            assert a.shape == b.shape
            if a.typecode() == "l":
                # Special expection. We have no way to distinguish between
                # "l" and "i" typecode, and we can consider them the same
                # to all practical effects
                assert b.typecode() == "i"
            else:
                assert a.typecode() == b.typecode()            
            assert a.tolist() == b.tolist()
            #print fileh
	    # Iterate over the next group
	    group = fileh.getNode(group, 'group' + str(rank))

        if verbose:
            print # This flush the stdout buffer
	# Close the file
	fileh.close()
	
	# Delete the file
        os.remove(file)
	
def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(ArrayTestCase))
    theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
