import sys
import unittest
import os
import tempfile
import warnings
import types
import numarray
from numarray import *
import numarray.strings as strings
import numarray.records as records
try:
    import Numeric
    numeric = 1
except:
    numeric = 0
from tables import *

from test_all import verbose, allequal

warnings.resetwarnings()

class BasicTestCase(unittest.TestCase):
    """Basic test for all the supported typecodes present in numarray.
    All of them are included on pytables.
    """
    endiancheck = 0
    
    def WriteRead(self, testArray):
        if verbose:
            print '\n', '-=' * 30
            if isinstance(testArray, strings.CharArray):
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
        if self.endiancheck and not (isinstance(a, strings.CharArray)):
            a._byteswap()
            a.togglebyteorder()

        self.fileh.createArray(self.root, 'somearray', a, "Some array")
	
        # Close the file
        self.fileh.close()
	
	# Re-open the file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        self.root = self.fileh.root
	
	# Read the saved array
	b = self.root.somearray.read()
	
	# Compare them. They should be equal.
	if verbose and not allequal(a,b):
	    print "Write and read arrays differ!"
	    #print "Array written:", a
	    print "Array written shape:", a.shape
	    print "Array written itemsize:", a.itemsize()
	    print "Array written type:", a.type()
	    #print "Array read:", b
	    print "Array read shape:", b.shape
	    print "Array read itemsize:", b.itemsize()
	    print "Array read type:", b.type()
            if not (isinstance(a, strings.CharArray)):
                print "Array written type:", a._byteorder
                print "Array read type:", b._byteorder

        # Check strictly the array equality
        assert a.shape == b.shape
        assert a.shape == self.root.somearray.shape
        if (isinstance(a, strings.CharArray)):
            assert str(self.root.somearray.type) == "CharType"
        else:
            assert a.type() == b.type()
            assert a.type() == self.root.somearray.type
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

        a = strings.array(self.tupleChar)
	self.WriteRead(a)
	return

    def test01_char_nc(self):
        "Data integrity during recovery (non-contiguous character objects)"
                
	a = strings.array(self.tupleChar)
        b = a[::2]
        # Ensure that this numarray string is non-contiguous
        assert b.iscontiguous() == 0
	self.WriteRead(b)
	return

    def test02_types(self):
        "Data integrity during recovery (numerical types)"

        # UInt64 seems to be unsupported on 64-bit machines!
	typecodes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64,
                     Float32, Float64, Complex32, Complex64]

	for typecode in typecodes:
            a = array(self.tupleInt, typecode)
            self.WriteRead(a)
            
        return

    def test03_types_nc(self):
        "Data integrity during recovery (non-contiguous numerical types)"

        # UInt64 seems to be unsupported on 64-bit machines!
	typecodes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64,
                     Float32, Float64, Complex32, Complex64]

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
    
class Basic0DTwoTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 2"
    tupleInt = 33
    tupleChar = "33"
    endiancheck = 1
    
class Basic1DZeroTestCase(BasicTestCase):
    # This test doesn't work at all, and that's normal
    # 1D case
    title = "Rank-1 case 0"
    tupleInt = ()
    tupleChar = ()   # This is not supported yet by numarray
    # This test needs at least numarray 0.8 to run 
    #tupleChar = strings.array(None, shape=(0,), itemsize=1)
    endiancheck = 0

class Basic1DOneTestCase(BasicTestCase):
    "Method doc"
    # 1D case
    title = "Rank-1 case 1"
    tupleInt = (3,)
    tupleChar = ("a",)
    endiancheck = 1
    
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
    
class Basic2DTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2"
    tupleInt = numarray.array(numarray.arange((4)**2), shape=(4,)*2) 
    tupleChar = strings.array("abc"*3**2, itemsize=3, shape=(3,)*2)
    endiancheck = 1
    
class Basic10DTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 test"
    tupleInt = numarray.array(numarray.arange((2)**10), shape=(2,)*10)
    # Dimensions greather than 6 in numarray strings gives some warnings
    tupleChar = strings.array("abc"*2**6, shape=(2,)*6, itemsize=3)
    endiancheck = 1
    
class Basic32DTestCase(BasicTestCase):
    # 32D case (maximum)
    title = "Rank-32 test"
    tupleInt = numarray.array((32,), shape=(1,)*32) 
    tupleChar = strings.array("121", shape=(1,)*32, itemsize=3)

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
            if isinstance(testArray, strings.CharArray):
                print "\nRunning test for array with type '%s'" % \
                      testArray.__class__.__name__,
            else:
                print "\nRunning test for array with type '%s'" % \
                      testArray.type(),

	# Create the array under root and name 'somearray'
	a = testArray
        if self.endiancheck and not (isinstance(a, strings.CharArray)):
            a._byteswap()
            a.togglebyteorder()

        self.fileh.createArray(self.root, 'somearray',
                               a, "Some array")

	# Do not close and re-open the file to cath-up
        # possible errors during the creation and later reading
        # of an array without an close/open in the middle
        # Close the file
        #self.fileh.close()
	# Re-open the file in read-only mode
        #self.fileh = openFile(self.file, mode = "r")
        #self.root = self.fileh.root
	
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
        if (isinstance(a, strings.CharArray)):
            assert str(self.root.somearray.type) == "CharType"
        else:
            assert a.type() == b.type()
            assert a.type() == self.root.somearray.type
            assert a._byteorder == b._byteorder
            assert a._byteorder == self.root.somearray.byteorder
            
        assert allequal(a,b)

	return

    def test01_signedShort_unaligned(self):
        "Checking an unaligned signed short integer array"

        r=records.array('a'*200,'i1,f4,i2',10)        
	a = r.field("c3")
        # Ensure that this array is non-aligned
        assert a.isaligned() == 0
        assert a._type == Int16
	self.WriteRead(a)
	return

    def test02_float_unaligned(self):
        "Checking an unaligned single precision array"

        r=records.array('a'*200,'i1,f4,i2',10)        
	a = r.field("c2")
        # Ensure that this array is non-aligned
        assert a.isaligned() == 0
        assert a._type == Float32
	self.WriteRead(a)
	return
    
    def test03_byte_offset(self):
        "Checking an offsetted byte array"

        r=numarray.arange(100, type=numarray.Int8, shape=(10,10))
	a = r[2]
        assert a._byteoffset > 0
	self.WriteRead(a)
	return
    
    def test04_short_offset(self):
        "Checking an offsetted unsigned short int precision array"

        r=numarray.arange(100, type=numarray.UInt32, shape=(10,10))
	a = r[2]
        assert a._byteoffset > 0
	self.WriteRead(a)
	return
    
    def test05_int_offset(self):
        "Checking an offsetted integer array"

        r=numarray.arange(100, type=numarray.Int32, shape=(10,10))
	a = r[2]
        assert a._byteoffset > 0
	self.WriteRead(a)
	return
    
    def test06_longlongint_offset(self):
        "Checking an offsetted long long integer array"

        r=numarray.arange(100, type=numarray.Int64, shape=(10,10))
	a = r[2]
        assert a._byteoffset > 0
	self.WriteRead(a)
	return
    
    def test07_float_offset(self):
        "Checking an offsetted single precision array"

        r=numarray.arange(100, type=numarray.Float32, shape=(10,10))
	a = r[2]
        assert a._byteoffset > 0
	self.WriteRead(a)
	return    

    def test08_double_offset(self):
        "Checking an offsetted double precision array"

        r=numarray.arange(100, type=numarray.Float64, shape=(10,10))
	a = r[2]
        assert a._byteoffset > 0
	self.WriteRead(a)
	return    
    
    def test09_float_offset_unaligned(self):
        "Checking an unaligned and offsetted single precision array"

        r=records.array('a'*200,'i1,3f4,i2',10)        
	a = r.field("c2")[3]
        # Ensure that this array is non-aligned
        assert a.isaligned() == 0
        assert a._byteoffset > 0
        assert a.type() == numarray.Float32
	self.WriteRead(a)
	return
    
    def test10_double_offset_unaligned(self):
        "Checking an unaligned and offsetted double precision array"

        r=records.array('a'*400,'i1,3f8,i2',10)        
	a = r.field("c2")[3]
        # Ensure that this array is non-aligned
        assert a.isaligned() == 0
        assert a._byteoffset > 0
        assert a.type() == numarray.Float64
	self.WriteRead(a)
	return

    
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
        # UInt64 seems to be unsupported on 64-bit machines!
	typecodes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64,
                     Float32, Float64, Complex32, Complex64]
	i = 1
	for typecode in typecodes:
	    # Create an array of typecode, with incrementally bigger ranges
	    a = ones((3,) * i, typecode)
	    # Save it on the HDF5 file
	    dsetname = 'array_' + `typecode`
	    if verbose:
		print "Creating dataset:", group._g_join(dsetname)
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
		print "  shape ==>", dset.shape, 
		print "  type ==> %s" % dset.type
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
	# arrays with rank greater than 20.
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
            # located on the Numeric package
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

class CopyTestCase(unittest.TestCase):

    def test01_copy(self):
        """Checking Array.copy() method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        arr=array([[456, 2],[3, 457]], type=Int16)
        array1 = fileh.createArray(fileh.root, 'array1', arr, "title array1")

        # Copy to another Array
        array2, size = array1.copy('/', 'array2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            array1 = fileh.root.array1
            array2 = fileh.root.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            #print "dirs-->", dir(array1), dir(array2)
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            
        # Check that all the elements are equal
        allequal(array1.read(), array2.read())

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.flavor == array2.flavor
        assert array1.type == array2.type
        assert array1.itemsize == array2.itemsize
        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking Array.copy() method (where specified)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        arr=array([[456, 2],[3, 457]], type=Int16)
        array1 = fileh.createArray(fileh.root, 'array1', arr, "title array1")

        # Copy to another Array
        group1 = fileh.createGroup("/", "group1")
        array2, size = array1.copy(group1, 'array2')

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
        allequal(array1.read(), array2.read())

        # Assert other properties in array
        assert array1.nrows == array2.nrows
        assert array1.flavor == array2.flavor
        assert array1.type == array2.type
        assert array1.itemsize == array2.itemsize
        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test03_copy(self):
        """Checking Array.copy() method (Numeric flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array (Numeric flavor)
        if numeric:
            arr=Numeric.array([[456, 2],[3, 457]], typecode='s')
        else:
            # If Numeric not installed, use a numarray object
            arr=array([[456, 2],[3, 457]], type=Int16)
            
        array1 = fileh.createArray(fileh.root, 'array1', arr, "title array1")

        # Copy to another Array
        array2, size = array1.copy('/', 'array2')

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
        assert array1.flavor == array2.flavor   # Very important here!
        assert array1.type == array2.type
        assert array1.itemsize == array2.itemsize
        assert array1.title == array2.title

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking Array.copy() method (checking title copying)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        arr=array([[456, 2],[3, 457]], type=Int16)
        array1 = fileh.createArray(fileh.root, 'array1', arr, "title array1")
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2
        # Copy it to another Array
        array2, size = array1.copy('/', 'array2', title="title array2")

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
        """Checking Array.copy() method (user attributes copied)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        arr=array([[456, 2],[3, 457]], type=Int16)
        array1 = fileh.createArray(fileh.root, 'array1', arr, "title array1")
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2
        # Copy it to another Array
        array2, size = array1.copy('/', 'array2', copyuserattrs=1)

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

    def test05b_copy(self):
        """Checking Array.copy() method (user attributes not copied)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create an Array
        arr=array([[456, 2],[3, 457]], type=Int16)
        array1 = fileh.createArray(fileh.root, 'array1', arr, "title array1")
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2
        # Copy it to another Array
        array2, size = array1.copy('/', 'array2', copyuserattrs=0)

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
        """Checking Array.copy() method with indexes"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a numarray
        r=arange(200, type=Int32, shape=(100,2))
        # Save it in a array:
        array1 = fileh.createArray(fileh.root, 'array1', r, "title array1")
        
        # Copy to another array
        array2, size = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            
        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        allequal(r2, array2.read())

        # Assert the number of rows in array
        if verbose:
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", r2.shape[0]
        assert r2.shape[0] == array2.nrows

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_indexclosef(self):
        """Checking Array.copy() method with indexes (close file version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_indexclosef..." % self.__class__.__name__

        # Create an instance of an HDF5 Array
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a numarray
        r=arange(200, type=Int32, shape=(100,2))
        # Save it in a array:
        array1 = fileh.createArray(fileh.root, 'array1', r, "title array1")
        
        # Copy to another array
        array2, size = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        # Close and reopen the file
        fileh.close()
        fileh = openFile(file, mode = "r")
        array1 = fileh.root.array1
        array2 = fileh.root.array2

        if verbose:
            print "array1-->", array1.read()
            print "array2-->", array2.read()
            print "attrs array1-->", repr(array1.attrs)
            print "attrs array2-->", repr(array2.attrs)
            
        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        allequal(r2, array2.read())

        # Assert the number of rows in array
        if verbose:
            print "nrows in array2-->", array2.nrows
            print "and it should be-->", r2.shape[0]
        assert r2.shape[0] == array2.nrows

        # Close the file
        fileh.close()
        os.remove(file)


class CopyIndex1TestCase(CopyIndexTestCase):
    start = 0
    stop = 7
    step = 1

class CopyIndex2TestCase(CopyIndexTestCase):
    start = 0
    stop = -1
    step = 1
	
class CopyIndex3TestCase(CopyIndexTestCase):
    start = 1
    stop = 7
    step = 1

class CopyIndex4TestCase(CopyIndexTestCase):
    start = 0
    stop = 6
    step = 1

class CopyIndex5TestCase(CopyIndexTestCase):
    start = 3
    stop = 7
    step = 1

class CopyIndex6TestCase(CopyIndexTestCase):
    start = 3
    stop = 6
    step = 2

class CopyIndex7TestCase(CopyIndexTestCase):
    start = 0
    stop = 7
    step = 10

class CopyIndex8TestCase(CopyIndexTestCase):
    start = 6
    stop = -1  # Negative values means starting from the end
    step = 1

class CopyIndex9TestCase(CopyIndexTestCase):
    start = 3
    stop = 4
    step = 1

class CopyIndex10TestCase(CopyIndexTestCase):
    start = 3
    stop = 4
    step = 2

class CopyIndex11TestCase(CopyIndexTestCase):
    start = -3
    stop = -1
    step = 2

class CopyIndex12TestCase(CopyIndexTestCase):
    start = -1   # Should point to the last element
    stop = None  # None should mean the last element (including it)
    step = 1

class GetItemTestCase(unittest.TestCase):

    def test00_single(self):
        "Single element access (character types)"
        
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Create the array under root and name 'somearray'
        a = self.charList
        arr=fileh.createArray(fileh.root, 'somearray', a, "Some array")

        # Get and compare an element
        if verbose:
            print "Original first element:", a[0]
            print "Read first element:", arr[0]
        assert allequal(a[0], arr[0])
        
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
        if verbose:
            print "Original first element:", a[0]
            print "Read first element:", arr[0]
        assert a[0] == arr[0]
        
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
        if verbose:
            print "Original elements:", a[1:4]
            print "Read elements:", arr[1:4]
        assert allequal(a[1:4], arr[1:4])
        
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
        if verbose:
            print "Original elements:", a[1:4]
            print "Read elements:", arr[1:4]
        assert allequal(a[1:4], arr[1:4])
        
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
        if verbose:
            print "Original elements:", a[1:4:2]
            print "Read elements:", arr[1:4:2]
        assert allequal(a[1:4:2], arr[1:4:2])
        
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
        if verbose:
            print "Original elements:", a[1:4:2]
            print "Read elements:", arr[1:4:2]
        assert allequal(a[1:4:2], arr[1:4:2])
        
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
        if verbose:
            print "Original last element:", a[-1]
            print "Read last element:", arr[-1]
        assert allequal(a[-1], arr[-1])
        
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
        if verbose:
            print "Original before last element:", a[-2]
            print "Read before last element:", arr[-2]
        if isinstance(a[-2], numarray.NumArray):
            assert allequal(a[-2], arr[-2])
        else:
            assert a[-2] == arr[-2]
        
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
        if verbose:
            print "Original last elements:", a[-4:-1]
            print "Read last elements:", arr[-4:-1]
        assert allequal(a[-4:-1], arr[-4:-1])
        
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
        if verbose:
            print "Original last elements:", a[-4:-1]
            print "Read last elements:", arr[-4:-1]
        assert allequal(a[-4:-1], arr[-4:-1])
        
        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return


class GI1NATestCase(GetItemTestCase):
    title = "Rank-1 case 1"
    numericalList = numarray.array([3])
    numericalListME = numarray.array([3,2,1,0,4,5,6])
    charList = strings.array(["3"])
    charListME = strings.array(["321","221","121","021","421","521","621"])
    
class GI2NATestCase(GetItemTestCase):
    # A more complex example
    title = "Rank-1,2 case 2"
    numericalList = numarray.array([3,4])
    numericalListME = numarray.array([[3,2,1,0,4,5,6],
                                      [2,1,0,4,5,6,7],
                                      [4,3,2,1,0,4,5],
                                      [3,2,1,0,4,5,6],
                                      [3,2,1,0,4,5,6]])
    
    charList = strings.array(["a","b"])
    charListME = strings.array([["321","221","121","021","421","521","621"],
                                ["21","21","11","02","42","21","61"],
                                ["31","21","12","21","41","51","621"],
                                ["321","221","121","021","421","521","621"],
                                ["3241","2321","13216","0621","4421","5421","a621"],
                                ["a321","s221","d121","g021","b421","5vvv21","6zxzxs21"]])
    

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
        if verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        assert ga == garr
        
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
        ga = [i for i in a]
        garr = [i for i in arr]
        if verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        for i in range(len(ga)):
            assert allequal(ga[i], garr[i])
        
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
        if verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        assert ga == garr
        
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
        ga = [i for i in a]
        garr = [i for i in arr]
        if verbose:
            print "Result of original iterator:", ga
            print "Result of read generator:", garr
        for i in range(len(ga)):
            assert allequal(ga[i], garr[i])
        
        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)
        return

class GE1NATestCase(GeneratorTestCase):
    title = "Rank-1 case 1"
    numericalList = numarray.array([3])
    numericalListME = numarray.array([3,2,1,0,4,5,6])
    charList = strings.array(["3"])
    charListME = strings.array(["321","221","121","021","421","521","621"])
    
class GE2NATestCase(GeneratorTestCase):
    # A more complex example
    title = "Rank-1,2 case 2"
    numericalList = numarray.array([3,4])
    numericalListME = numarray.array([[3,2,1,0,4,5,6],
                                      [2,1,0,4,5,6,7],
                                      [4,3,2,1,0,4,5],
                                      [3,2,1,0,4,5,6],
                                      [3,2,1,0,4,5,6]])
    
    charList = strings.array(["a","b"])
    charListME = strings.array([["321","221","121","021","421","521","621"],
                                ["21","21","11","02","42","21","61"],
                                ["31","21","12","21","41","51","621"],
                                ["321","221","121","021","421","521","621"],
                                ["3241","2321","13216","0621","4421","5421","a621"],
                                ["a321","s221","d121","g021","b421","5vvv21","6zxzxs21"]])
    

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        # The scalar case test should be refined in order to work
        theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
        #theSuite.addTest(unittest.makeSuite(Basic1DZeroTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DThreeTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2DTestCase))
        theSuite.addTest(unittest.makeSuite(Basic10DTestCase))
        # The 32 dimensions case is tested on GroupsArray
        #theSuite.addTest(unittest.makeSuite(Basic32DTestCase))
        theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))
        theSuite.addTest(unittest.makeSuite(UnalignedAndComplexTestCase))
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
        theSuite.addTest(unittest.makeSuite(GI1NATestCase))
        theSuite.addTest(unittest.makeSuite(GI2NATestCase))
        theSuite.addTest(unittest.makeSuite(GE1NATestCase))
        theSuite.addTest(unittest.makeSuite(GE2NATestCase))


    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
