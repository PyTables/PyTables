import sys
import unittest
import os
import tempfile
import warnings
from numarray import *
import chararray
import recarray
from tables import *

from test_all import verbose

# If the next line is uncommented, the logical_and.reduce in loop issues an
# OverflowWarning in the case of type Int8 (and only in that case).
#warnings.resetwarnings()

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

class BasicTestCase(unittest.TestCase):
    """Basic test for all the supported typecodes present in numarray.
    All of them are included on pytables.
    """
    endiancheck = 0
    atomictype = 0
    
    def WriteRead(self, testArray):
        if verbose:
            print '\n', '-=' * 30
            if isinstance(testArray, chararray.CharArray):
                print "Running test for array with type '%s'" % \
                      testArray.__class__.__name__,
            else:
                print "Running test for array with type '%s'" % \
                      testArray.type(),
            print "for class check:", self.title
            
        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w")
        self.root = self.fileh.root

	# Create the array under root and name 'somearray'
	a = testArray
        if self.endiancheck and not (isinstance(a, chararray.CharArray)):
            a._byteswap()
            a.togglebyteorder()

        self.fileh.createArray(self.root, 'somearray', a,
                               "Some array", atomictype=self.atomictype)
	
        # Close the file
        self.fileh.close()
	
	# Re-open the file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        self.root = self.fileh.root
	
	# Read the saved array
	b = self.root.somearray.read()
	
	# Compare them. They should be equal.
	if not allequal(a,b) and verbose:
	    print "Write and read arrays differ!"
	    print "Array written:", a
	    print "Array written shape:", a.shape
	    print "Array written itemsize:", a.itemsize()
	    print "Array written type:", a.type()
	    print "Array read:", b
	    print "Array read shape:", b.shape
	    print "Array read itemsize:", b.itemsize()
	    print "Array read type:", b.type()
            if not (isinstance(a, chararray.CharArray)):
                print "Array written type:", a._byteorder
                print "Array read type:", b._byteorder

        # Check strictly the array equality
        assert a.shape == b.shape
        assert a.shape == self.root.somearray.shape
        if (isinstance(a, chararray.CharArray)):
            assert str(self.root.somearray.typeclass) == "CharType"
        else:
            assert a.type() == b.type()
            assert a.type() == self.root.somearray.typeclass
            assert a._byteorder == b._byteorder
            assert a._byteorder == self.root.somearray.byteorder
            if self.endiancheck:
                assert b._byteorder <> sys.byteorder

        assert allequal(a,b)

        # Close the file (eventually destroy the extended type)
        self.fileh.close()

        # Then, delete the file
        os.remove(self.file)

	return
    
    def test00_char(self):
        "Data integrity during recovery (character objects)"

        a = chararray.array(self.tupleChar)
	self.WriteRead(a)
	return

    def test01_char_nc(self):
        "Data integrity during recovery (non-contiguous character objects)"
                
	a = chararray.array(self.tupleChar)
        b = a[::2]
        # Ensure that this chararray is non-contiguous
        assert b.iscontiguous() == 0
	self.WriteRead(b)
	return

    def test02_types(self):
        "Data integrity during recovery (numerical types)"
        
	typecodes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
                     Float32, Float64]

	for typecode in typecodes:
            a = array(self.tupleInt, typecode)
            self.WriteRead(a)
            
        return

    def test03_types_nc(self):
        "Data integrity during recovery (non-contiguous numerical types)"
        
	typecodes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
                     Float32, Float64]

	for typecode in typecodes:
            a = array(self.tupleInt, typecode)
            # This should not be tested for the rank-0 case
            if len(a.shape) == 0:
                return
            b = a[::2]
            # Ensure that this array is non-contiguous
            assert b.iscontiguous() == 0
            self.WriteRead(b)

        return

class Basic0DOneTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 1"
    tupleInt = 3
    tupleChar = "3"
    endiancheck = 1
    atomictype = 1 # Rank-0 is only supported by atomictype=1
    
class Basic0DTwoTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 2"
    tupleInt = 33
    tupleChar = "33"
    endiancheck = 1
    atomictype = 1 # Rank-0 is only supported by atomictype=1
    
class Basic1DZeroTestCase(BasicTestCase):
    # This test doesn't work at all, and that's normal
    # 1D case
    title = "Rank-1 case 0"
    tupleInt = ()
    tupleChar = ()
    endiancheck = 0
    atomictype = 0

class Basic1DOneTestCase(BasicTestCase):
    "Method doc"
    # 1D case
    title = "Rank-1 case 1"
    tupleInt = (3,)
    tupleChar = ("a",)
    endiancheck = 1
    atomictype = 0
    
class Basic1DTwoTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 2"
    tupleInt = (3, 4)
    tupleChar = ("aaa",)
    endiancheck = 1
    
class Basic1DThreeTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 3"
    tupleInt = (3, 4, 5)
    tupleChar = ("aaa", "bbb",)
    endiancheck = 1
    atomictype = 0
    
class Basic2DTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2 case 1"
    tupleInt = array(arange((4)**2), shape=(4,)*2) 
    tupleChar = chararray.array("abc"*3**2, shape=(3,)*2, itemsize=3)
    endiancheck = 1
    atomictype = 0
    
class Basic10DOneTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 case 1"
    tupleInt = array(arange((2)**10), shape=(2,)*10)
    # Dimensions greather than 6 in chararray gives some warnings
    #tupleChar = chararray.array("abc"*2**8, shape=(2,)*8, itemsize=3)
    tupleChar = chararray.array("abc"*2**6, shape=(2,)*6, itemsize=3)
    atomictype=0
    endiancheck = 1
    
class Basic10DTwoTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 case 2"
    tupleInt = array(arange((2)**10), shape=(2,)*10)
    # Dimensions greather than 6 in chararray gives some warnings
    #tupleChar = chararray.array("abc"*2**8, shape=(2,)*8, itemsize=3)
    tupleChar = chararray.array("abc"*2**6, shape=(2,)*6, itemsize=3)
    atomictype = 1
    endiancheck = 0
    
class Basic32DTestCase(BasicTestCase):
    # 32D case (maximum)
    tupleInt = array((32,), shape=(1,)*32) 
    tupleChar = chararray.array("121", shape=(1,)*32, itemsize=3)

class UnalignedAndComplexTestCase(unittest.TestCase):
    """Basic test for all the supported typecodes present in numarray.
    All of them are included on pytables.
    """
    endiancheck = 0

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
            if isinstance(testArray, chararray.CharArray):
                print "Running test for array with type '%s'" % \
                      testArray.__class__.__name__,
            else:
                print "Running test for array with type '%s'" % \
                      testArray.type(),

	# Create the array under root and name 'somearray'
	a = testArray
        if self.endiancheck and not (isinstance(a, chararray.CharArray)):
            a._byteswap()
            a.togglebyteorder()

        self.fileh.createArray(self.root, 'somearray',
                               a, "Some array")
	
        # Close the file
        self.fileh.close()
	
	# Re-open the file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        self.root = self.fileh.root
	
	# Read the saved array
	b = self.root.somearray.read()
	
	# Compare them. They should be equal.
	if not allequal(a,b) and verbose:
	    print "Write and read arrays differ!"
	    print "Array written:", a
	    print "Array written shape:", a.shape
	    print "Array written itemsize:", a.itemsize()
	    print "Array written type:", a.type()
	    print "Array read:", b
	    print "Array read shape:", b.shape
	    print "Array read itemsize:", b.itemsize()
	    print "Array read type:", b.type()

        # Check strictly the array equality
        assert a.shape == b.shape
        assert a.shape == self.root.somearray.shape
        if (isinstance(a, chararray.CharArray)):
            assert str(self.root.somearray.typeclass) == "CharType"
        else:
            assert a.type() == b.type()
            assert a.type() == self.root.somearray.typeclass
            assert a._byteorder == b._byteorder
            assert a._byteorder == self.root.somearray.byteorder
            
        assert allequal(a,b)

	return

    def test01_signedShort_unaligned(self):
        "Checking an unaligned signed short integer array"

        r=recarray.array('a'*200,'b,f,s',10)        
	a = r.field("c3")
        # Ensure that this array is non-contiguous
        assert a.iscontiguous() == 0
        assert a._type == Int16
	self.WriteRead(a)
	return

    def test02_float_unaligned(self):
        "Checking an unaligned single precision array"

        r=recarray.array('a'*200,'b,f,s',10)        
	a = r.field("c2")
        # Ensure that this array is non-contiguous
        assert a.iscontiguous() == 0
        assert a._type == Float32
	self.WriteRead(a)
	return
    
    def test10_complexSimple(self):
        "Checking a complex floating point array (not supported)"
	a = array( [1,2], Complex32)
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

	a = array( [1,2], Complex64)
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
	#typecodes = ["c", 'b', '1', 's', 'w', 'i', 'u', 'l', 'f', 'd']
	typecodes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
                     Float32, Float64]
	i = 1
	for typecode in typecodes:
	    # Create an array of typecode, with incrementally bigger ranges
	    a = ones((3,) * i, typecode)
	    # Save it on the HDF5 file
	    dsetname = 'array_' + `typecode`
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
	for i in range(1,len(typecodes)):
	    # Create an array for later comparison
	    a = ones((3,) * i, typecodes[i - 1])
	    # Get the dset object hanging from group
	    dset = getattr(group, 'array_' + `typecodes[i-1]`)
	    # Get the actual array
	    b = dset.read()
	    if verbose:
		print "Info from dataset:", dset._v_pathname
		print "  Shape: ==>", dset.shape, 
		print "  typeclass ==> %s" % dset.typeclass
		print "Array b read from file. Shape: ==>", b.shape,
		print ". Type ==>" % b.type()
	    assert a.shape == b.shape
            assert a.type() == b.type()
            assert allequal(a,b)

	    # Iterate over the next group
	    group = getattr(group, 'group' + str(i))

	# Close the file
	fileh.close()

	# Then, delete the file
        os.remove(file)
        del a, b, fileh

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
	    if verbose and not allequal(a,b):
		print "Info from dataset:", dset._v_pathname
		print "  Shape: ==>", dset.shape, 
		print "  typecode ==> %c" % dset.typecode
		print "Array b read from file. Shape: ==>", b.shape,
		print ". Type ==> %c" % b.type()
                
            # ************** WARNING!!! *****************
            # If we compare to arrays of dimensions bigger than 20
            # we get a segmentation fault! It is most probably a bug
            # located on Numeric package
            # ************** WARNING!!! *****************
            assert a.shape == b.shape
            assert a.type() == b.type()
            assert allequal(a,b)

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

    # The scalar case test should be refined in order to work
    theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
    theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
    #theSuite.addTest(unittest.makeSuite(Basic1DZeroTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
    theSuite.addTest(unittest.makeSuite(Basic1DThreeTestCase))
    theSuite.addTest(unittest.makeSuite(Basic2DTestCase))
    theSuite.addTest(unittest.makeSuite(Basic10DOneTestCase))
    theSuite.addTest(unittest.makeSuite(Basic10DTwoTestCase))
    # The 32 dimensions case is tested on GroupsArray
    #theSuite.addTest(unittest.makeSuite(Basic32DTestCase))
    theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))
    theSuite.addTest(unittest.makeSuite(UnalignedAndComplexTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
