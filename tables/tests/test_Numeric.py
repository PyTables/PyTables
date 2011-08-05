import unittest
import os
import tempfile
from Numeric import *

from tables import *
from tables.tests import common
from tables.tests.common import typecode, allequal

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

class BasicTestCase(unittest.TestCase):
    """Basic test for all the supported typecodes present in Numeric.
    All of them are included on pytables.
    """
    endiancheck = 0

    def WriteRead(self, testArray):
        if common.verbose:
            print '\n', '-=' * 30
            print "Running test for array with typecode '%s'" % \
                  testArray.typecode(),
            print "for class check:", self.title

        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w")
        self.root = self.fileh.root
        # Create the array under root and name 'somearray'
        a = testArray
        self.fileh.createArray(self.root, 'somearray', a, "Some array")

        # Close the file
        self.fileh.close()

        # Re-open the file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        self.root = self.fileh.root

        # Read the saved array
        b = self.root.somearray.read()

        # For cases that read returns a python type instead of a Numeric type
        if not hasattr(b, "shape"):
            b = array(b, typecode=a.typecode())

        # Compare them. They should be equal.
        if not allequal(a,b, "numeric") and common.verbose:
            print "Write and read arrays differ!"
            print "Array written:", a
            print "Array written shape:", a.shape
            print "Array written itemsize:", a.itemsize()
            print "Array written type:", a.typecode()
            print "Array read:", b
            print "Array read shape:", b.shape
            print "Array read itemsize:", b.itemsize()
            print "Array read type:", b.typecode()

        # Check strictly the array equality
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape, self.root.somearray.shape)
        if a.typecode() == "i" or a.typecode() == "l":
            # Special expection. We have no way to distinguish between
            # "l" and "i" typecode, and we can consider them the same
            # to all practical effects
            self.assertTrue(b.typecode() == "l" or b.typecode() == "i")
            # We have to add "N" that represent Int64 in 64-bit platforms
            self.assertTrue(typecode[self.root.somearray.atom.type] in
                            ["i", "l", "N"])
        elif a.typecode() == "c":
            self.assertEqual(a.typecode(), b.typecode())
            self.assertEqual(self.root.somearray.atom.type, "string")
        else:
            self.assertEqual(a.typecode(), b.typecode())
            self.assertEqual(a.typecode(),
                             typecode[self.root.somearray.atom.type])

        self.assertTrue(allequal(a,b, "numeric"))
        self.fileh.close()
        # Then, delete the file
        os.remove(self.file)
        return

    def test00_char(self):
        "Data integrity during recovery (character objects)"

        a = array(self.tupleChar,'c')
        self.WriteRead(a)
        return

    def test01_char_nc(self):
        "Data integrity during recovery (non-contiguous character objects)"

        a = array(self.tupleChar, 'c')
        if a.shape == ():
            b = a[()]
        else:
            b = a[::2]
            # Ensure that this Numeric string is non-contiguous
            self.assertEqual(b.iscontiguous(), False)
        self.WriteRead(b)
        return

    def test02_types(self):
        "Data integrity during recovery (numerical types)"

        typecodes = ['b', '1', 's', 'i', 'l', 'f', 'd', 'F', 'D']

        for typecode in typecodes:
            a = self.tupleInt.astype(typecode)
            self.WriteRead(a)

        return

    def test03_types_nc(self):
        "Data integrity during recovery (non-contiguous numerical types)"

        typecodes = ['b', '1', 's', 'i', 'l', 'f', 'd', 'F', 'D']

        for typecode in typecodes:
            a = self.tupleInt.astype(typecode)
            # This should not be tested for the rank-0 case
            if len(a.shape) == 0:
                return
            b = a[::2]
            # Ensure that this array is non-contiguous
            self.assertEqual(b.iscontiguous(), 0)
            self.WriteRead(b)

        return


class Basic0DOneTestCase(BasicTestCase):
    # Rank-0 case
    title = "Rank-0 case 1"
    tupleInt = array(3)
    tupleChar = "4"

class Basic0DTwoTestCase(BasicTestCase):
    # Rank-0 case
    title = "Rank-0 case 2"
    tupleInt = array(33)
    tupleChar = "44"

class Basic1DOneTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 1"
    tupleInt = array((3,))
    tupleChar = ("a",)

class Basic1DTwoTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 2"
    tupleInt = array((3, 4))
    tupleChar = ("aaa",)

class Basic1DThreeTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 3"
    tupleInt = array((3, 4, 5))
    tupleChar = ("aaa", "bbb",)

class Basic2DTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2 case 1"
    #tupleInt = reshape(array(arange((4)**2)), (4,)*2)
    tupleInt = ones((4,)*2)
    tupleChar = [["aa","dd"],["dd","ss"],["ss","tt"]]

class Basic10DTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 case 1"
    #tupleInt = reshape(array(arange((2)**10)), (2,)*10)
    tupleInt = ones((2,)*10)
    tupleChar = ones((2,)*8,'c')

class Basic32DTestCase(BasicTestCase):
    # 32D case (maximum)
    tupleInt = reshape(array((32,)), (1,)*32)
    # Strings seems to cause some problems with somewhat large dimensions
    # Reverting to 2D case
    tupleChar = [["aa","dd"],["dd","ss"]]


class GroupsArrayTestCase(unittest.TestCase):
    """This test class checks combinations of arrays with groups.
    It also uses arrays ranks which ranges until 10.
    """

    def test00_iterativeGroups(self):
        """Checking combinations of arrays with groups
        It also uses arrays ranks which ranges until 10.
        """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_iterativeGroups..." % \
                  self.__class__.__name__

        # Open a new empty HDF5 file
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        # Get the root group
        group = fileh.root

        # Set the type codes to test
        typecodes = ['c', 'b', '1', 's', 'w', 'i', 'u', 'l', 'f', 'd']
        i = 1
        for typecode in typecodes:
            # Create an array of typecode, with incrementally bigger ranges
            a = ones((2,) * i, typecode)
            # Save it on the HDF5 file
            dsetname = 'array_' + typecode
            if common.verbose:
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
            a = ones((2,) * i, typecodes[i - 1])
            # Get the dset object hanging from group
            dset = getattr(group, 'array_' + typecodes[i-1])
            # Get the actual array
            b = dset.read()
            if not allequal(a,b, "numeric") and common.verbose:
                print "Array a original. Shape: ==>", a.shape
                print "Array a original. Data: ==>", a
                print "Info from dataset:", dset._v_pathname
                print "  shape ==>", dset.shape,
                print "  dtype ==>", dset.dtype
                print "Array b read from file. Shape: ==>", b.shape,
                print ". Type ==> %s" % b.typecode()

            self.assertEqual(a.shape, b.shape)
            if (a.typecode() == "i" or a.typecode() == "l"):
                # Special expection. We have no way to distinguish between
                # "l" and "i" typecode, and we can consider them the same
                # to all practical effects
                self.assertTrue(b.typecode() == "l" or b.typecode() == "i")
            else:
                self.assertEqual(a.typecode(), b.typecode())
            self.assertTrue(allequal(a,b, "numeric"))

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
        #maxrank = 32 # old limit (Numeric <= 22.0)
        maxrank = 30  # This limit is set in Numeric 23.x and 24.x

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_largeRankArrays..." % \
                  self.__class__.__name__
            print "Maximum rank for tested arrays:", maxrank
        # Open a new empty HDF5 file
        #file = tempfile.mktemp(".h5")
        file = "test_array.h5"
        fileh = openFile(file, mode = "w")
        group = fileh.root
        if common.verbose:
            print "Rank array writing progress: ",
        for rank in range(minrank, maxrank + 1):
            # Create an array of integers, with incrementally bigger ranges
            a = ones((1,) * rank, 'i')
            if common.verbose:
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
        if common.verbose:
            print
            print "Rank array reading progress: "
        # Get the metadata on the previosly saved arrays
        for rank in range(minrank, maxrank + 1):
            # Create an array for later comparison
            a = ones((1,) * rank, 'i')
            # Get the actual array
            b = group.array.read()
            if common.verbose:
                print "%3d," % (rank),
            if not a.tolist() == b.tolist() and common.verbose:
                print "Info from dataset:", dset._v_pathname
                print "  Shape: ==>", dset.shape,
                print "  typecode ==> %c" % dset.typecode
                print "Array b read from file. Shape: ==>", b.shape,
                print ". Type ==> %c" % b.typecode()
            self.assertEqual(a.shape, b.shape)
            if a.typecode() == "i":
                # Special expection. We have no way to distinguish between
                # "l" and "i" typecode, and we can consider them the same
                # to all practical effects
                self.assertTrue(b.typecode() == "l" or b.typecode() == "i")
            else:
                self.assertEqual(a.typecode(), b.typecode())

            # ************** WARNING!!! *****************
            # If we compare to arrays of dimensions bigger than 20
            # we get a segmentation fault! It is most probably a bug
            # located on Numeric package.
            # I've discovered that comparing shapes and using the
            # tolist() conversion is the best to compare Numeric
            # arrays!. At least, tolist() do not crash!.
            # ************** WARNING!!! *****************
            #self.assertEqual(a.tolist(), b.tolist())
            self.assertEqual(a, b)

            # Iterate over the next group
            group = fileh.getNode(group, 'group' + str(rank))

        if common.verbose:
            print # This flush the stdout buffer
        # Close the file
        fileh.close()
        # Delete the file
        os.remove(file)


# Test Record class
class Record(IsDescription):
    var1  = StringCol(itemsize=4, dflt="abcd")
    var2  = StringCol(itemsize=1, dflt="a")
    var3  = BoolCol(dflt=1)
    var4  = Int8Col(dflt=1)
    var5  = UInt8Col(dflt=1)
    var6  = Int16Col(dflt=1)
    var7  = UInt16Col(dflt=1)
    var8  = Int32Col(dflt=1)
    var9  = UInt32Col(dflt=1)
    # Apparently, there is no way to convert a NumPy of 64-bits into
    # Numeric of 64-bits in 32-bit platforms
    # See
    # http://aspn.activestate.com/ASPN/Mail/Message/numpy-discussion/2569120
    # Uncomment this makes test breaks on 64-bit platforms 2005-09-23
    #var10 = Int64Col(1)
    var11 = Float32Col(dflt=1.0)
    var12 = Float64Col(dflt=1.0)
    var13 = ComplexCol(itemsize=8, dflt=(1.+0.j))
    var14 = ComplexCol(itemsize=16, dflt=(1.+0.j))


class TableReadTestCase(common.PyTablesTestCase):
    nrows = 100

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, "w")
        table = fileh.createTable(fileh.root, 'table', Record)
        for i in range(self.nrows):
            table.row.append()  # Fill 100 rows with default values
        fileh.close()
        self.fileh = openFile(self.file, "a")  # allow flavor changes

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test01_readTableChar(self):
        """Checking column conversion into Numeric in read(). Char flavor"""

        table = self.fileh.root.table
        table.flavor = "numeric"
        for colname in table.colnames:
            numcol = table.read(field=colname)
            typecol = table.coltypes[colname]
            itemsizecol = table.description._v_dtypes[colname].base.itemsize
            nctypecode = numcol.typecode()
            if typecol == "string":
                if itemsizecol > 1:
                    orignumcol = array(['abcd']*self.nrows, typecode='c')
                else:
                    orignumcol = array(['a']*self.nrows, typecode='c')
                    orignumcol.shape=(self.nrows,)
                if common.verbose:
                    print "Typecode of Numeric column read:", nctypecode
                    print "Should look like:", 'c'
                    print "Itemsize of column:", itemsizecol
                    print "Shape of Numeric column read:", numcol.shape
                    print "Should look like:", orignumcol.shape
                    print "First 3 elements of read col:", numcol[:3]
                # Check that both Numeric objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numeric"))

    def test01_readTableNum(self):
        """Checking column conversion into Numeric in read(). Numeric flavor"""

        table = self.fileh.root.table
        table.flavor="numeric"
        for colname in table.colnames:
            numcol = table.read(field=colname)
            typecol = table.coltypes[colname]
            nctypecode = numcol.typecode()
            if typecol != "string":
                if typecol == "int64":
                    return
                if typecol == "bool":
                    nctypecode = "B"
                if common.verbose:
                    print "Typecode of Numeric column read:", nctypecode
                    print "Should look like:", typecode[typecol]
                orignumcol = ones(shape=self.nrows, typecode=numcol.typecode())
                # Check that both Numeric objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numeric"))


    def test02_readCoordsChar(self):
        """Column conversion into Numeric in readCoords(). Chars"""

        table = self.fileh.root.table
        table.flavor = "numeric"
        coords = (1,2,3)
        self.nrows = len(coords)
        for colname in table.colnames:
            numcol = table.readCoordinates(coords, field=colname)
            typecol = table.coltypes[colname]
            itemsizecol = table.description._v_dtypes[colname].base.itemsize
            nctypecode = numcol.typecode()
            if typecol == "string":
                if itemsizecol > 1:
                    orignumcol = array(['abcd']*self.nrows, typecode='c')
                else:
                    orignumcol = array(['a']*self.nrows, typecode='c')
                    orignumcol.shape=(self.nrows,)
                if common.verbose:
                    print "Typecode of Numeric column read:", nctypecode
                    print "Should look like:", 'c'
                    print "Itemsize of column:", itemsizecol
                    print "Shape of Numeric column read:", numcol.shape
                    print "Should look like:", orignumcol.shape
                    print "First 3 elements of read col:", numcol[:3]
                # Check that both Numeric objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numeric"))

    def test02_readCoordsNum(self):
        """Column conversion into Numeric in readCoordinates(). Numerical"""

        table = self.fileh.root.table
        table.flavor = "numeric"
        coords = (1,2,3)
        self.nrows = len(coords)
        for colname in table.colnames:
            numcol = table.readCoordinates(coords, field=colname)
            typecol = table.coltypes[colname]
            nctypecode = numcol.typecode()
            if typecol != "string":
                if typecol == "int64":
                    return
                if typecol == "bool":
                    nctypecode = "B"
                if common.verbose:
                    print "Typecode of Numeric column read:", nctypecode
                    print "Should look like:", typecode[typecol]
                orignumcol = ones(shape=self.nrows, typecode=numcol.typecode())
                # Check that both Numeric objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numeric"))


#--------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    #theSuite.addTest(unittest.makeSuite(TableReadTestCase))
    for i in range(niter):
        # TODO: The scalar case test should be refined in order to work
        # specially after the solution for bug #968132
        theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DThreeTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2DTestCase))
        theSuite.addTest(unittest.makeSuite(Basic10DTestCase))
        # The 32 dimensions case is tested on GroupsArray
        #theSuite.addTest(unittest.makeSuite(Basic32DTestCase))
        theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))
        theSuite.addTest(unittest.makeSuite(TableReadTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
