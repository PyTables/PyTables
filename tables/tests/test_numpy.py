import sys
import unittest
import os
import tempfile
import numpy

from numpy import *

import tables
from tables import *
from tables.tests import common
from tables.tests.common import allequal

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

typecodes = ['b', 'h', 'i', 'l', 'q', 'f', 'd']
# UInt64 checking disabled on win platforms
# because this type is not supported
if sys.platform != 'win32':
    typecodes += ['B', 'H', 'I', 'L', 'Q', 'F', 'D']
else:
    typecodes += ['B', 'H', 'I', 'L', 'F', 'D']
typecodes += ['b1']   # boolean

byteorder = {'little': '<', 'big': '>'}[sys.byteorder]

class BasicTestCase(unittest.TestCase):
    """Basic test for all the supported typecodes present in NumPy.
    All of them are included on PyTables.
    """
    endiancheck = 0

    def WriteRead(self, testArray):
        if common.verbose:
            print '\n', '-=' * 30
            print "Running test for array with typecode '%s'" % \
                  testArray.dtype.char,
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
        # For cases that read returns a python type instead of a numpy type
        if not hasattr(b, "shape"):
            b = array(b, dtype=a.dtype.str)

        # Compare them. They should be equal.
        #if not allequal(a,b, "numpy") and common.verbose:
        if common.verbose:
            print "Array written:", a
            print "Array written shape:", a.shape
            print "Array written itemsize:", a.itemsize
            print "Array written type:", a.dtype.char
            print "Array read:", b
            print "Array read shape:", b.shape
            print "Array read itemsize:", b.itemsize
            print "Array read type:", b.dtype.char

        type_ = self.root.somearray.atom.type
        # Check strictly the array equality
        self.assertEqual(type(a), type(b))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape, self.root.somearray.shape)
        self.assertEqual(a.dtype, b.dtype)
        if a.dtype.char[0] == "S":
            self.assertEqual(type_, "string")
        else:
            self.assertEqual(a.dtype.base.name, type_)

        self.assertTrue(allequal(a,b, "numpy"))
        self.fileh.close()
        # Then, delete the file
        os.remove(self.file)
        return

    def test00_char(self):
        "Data integrity during recovery (character objects)"

        a = array(self.tupleChar,'S'+str(len(self.tupleChar)))
        self.WriteRead(a)
        return

    def test01_char_nc(self):
        "Data integrity during recovery (non-contiguous character objects)"

        a = array(self.tupleChar,'S'+str(len(self.tupleChar)))
        if a.shape == ():
            b = a               # We cannot use the indexing notation
        else:
            b = a[::2]
            # Ensure that this numarray string is non-contiguous
            if a.shape[0] > 2:
                self.assertEqual(b.flags['CONTIGUOUS'], False)
        self.WriteRead(b)
        return

    def test02_types(self):
        "Data integrity during recovery (numerical types)"

        for typecode in typecodes:
            if self.tupleInt.shape:
                a = self.tupleInt.astype(typecode)
            else:
                # shape is the empty tuple ()
                a = array(self.tupleInt, dtype=typecode)
            self.WriteRead(a)

        return

    def test03_types_nc(self):
        "Data integrity during recovery (non-contiguous numerical types)"

        for typecode in typecodes:
            if self.tupleInt.shape:
                a = self.tupleInt.astype(typecode)
            else:
                # shape is the empty tuple ()
                a = array(self.tupleInt, dtype=typecode)
            # This should not be tested for the rank-0 case
            if len(a.shape) == 0:
                return
            b = a[::2]
            # Ensure that this array is non-contiguous (for non-trivial case)
            if a.shape[0] > 2:
                self.assertEqual(b.flags['CONTIGUOUS'], False)
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
    tupleInt = array((0, 4))
    tupleChar = ("aaa",)

class Basic1DThreeTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 3"
    tupleInt = array((3, 4, 5))
    tupleChar = ("aaaa", "bbb",)

class Basic2DTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2 case 1"
    #tupleInt = reshape(array(arange((4)**2)), (4,)*2)
    tupleInt = ones((4,)*2)
    tupleChar = [["aaa","ddddd"],["d","ss"],["s","tt"]]

class Basic10DTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 case 1"
    #tupleInt = reshape(array(arange((2)**10)), (2,)*10)
    tupleInt = ones((2,)*10)
    #tupleChar = reshape(array([1],dtype="S1"),(1,)*10)
    # The next tuple consumes far more time, so this
    # test should be run in common.heavy mode.
    tupleChar = array(tupleInt, dtype="S1")


# class Basic32DTestCase(BasicTestCase):
#     # 32D case (maximum)
#     tupleInt = reshape(array((22,)), (1,)*32)
#     # Strings seems to be very slow with somewhat large dimensions
#     # This should not be run unless the numarray people address this problem
#     # F. Alted 2006-01-04
#     tupleChar = array(tupleInt, dtype="S1")


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
            if not allequal(a,b, "numpy") and common.verbose:
                print "Array a original. Shape: ==>", a.shape
                print "Array a original. Data: ==>", a
                print "Info from dataset:", dset._v_pathname
                print "  shape ==>", dset.shape,
                print "  dtype ==> %s" % dset.dtype
                print "Array b read from file. Shape: ==>", b.shape,
                print ". Type ==> %s" % b.dtype.char

            self.assertEqual(a.shape, b.shape)
            if dtype('l').itemsize == 4:
                if (a.dtype.char == "i" or a.dtype.char == "l"):
                    # Special expection. We have no way to distinguish between
                    # "l" and "i" typecode, and we can consider them the same
                    # to all practical effects
                    self.assertTrue(b.dtype.char == "l" or b.dtype.char == "i")
                elif (a.dtype.char == "I" or a.dtype.char == "L"):
                    # Special expection. We have no way to distinguish between
                    # "L" and "I" typecode, and we can consider them the same
                    # to all practical effects
                    self.assertTrue(b.dtype.char == "L" or b.dtype.char == "I")
                else:
                    self.assertTrue(allequal(a,b, "numpy"))
            elif dtype('l').itemsize == 8:
                if (a.dtype.char == "q" or a.dtype.char == "l"):
                    # Special expection. We have no way to distinguish between
                    # "q" and "l" typecode in 64-bit platforms, and we can
                    # consider them the same to all practical effects
                    self.assertTrue(b.dtype.char == "l" or b.dtype.char == "q")
                elif (a.dtype.char == "Q" or a.dtype.char == "L"):
                    # Special expection. We have no way to distinguish between
                    # "Q" and "L" typecode in 64-bit platforms, and we can
                    # consider them the same to all practical effects
                    self.assertTrue(b.dtype.char == "L" or b.dtype.char == "Q")
                else:
                    self.assertTrue(allequal(a,b, "numpy"))

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
        # This limit is due to a limit in the HDF5 library.
        minrank = 1
        maxrank = 32

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_largeRankArrays..." % \
                  self.__class__.__name__
            print "Maximum rank for tested arrays:", maxrank
        # Open a new empty HDF5 file
        file = tempfile.mktemp(".h5")
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
                print ". Type ==> %c" % b.dtype.char
            self.assertEqual(a.shape, b.shape)
            if a.dtype.char == "i":
                # Special expection. We have no way to distinguish between
                # "l" and "i" typecode, and we can consider them the same
                # to all practical effects
                self.assertTrue(b.dtype.char == "l" or b.dtype.char == "i")
            else:
                self.assertEqual(a.dtype.char, b.dtype.char)

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
    var1  = StringCol(itemsize=4, dflt="abcd", pos=0)
    var2  = StringCol(itemsize=1, dflt="a", pos=1)
    var3  = BoolCol(dflt=1)  # Typecode == '1' in Numeric. 'B' in numarray
    var4  = Int8Col(dflt=1)
    var5  = UInt8Col(dflt=1)
    var6  = Int16Col(dflt=1)
    var7  = UInt16Col(dflt=1)
    var8  = Int32Col(dflt=1)
    var9  = UInt32Col(dflt=1)
    var10 = Int64Col(dflt=1)
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
        """Checking column conversion into NumPy in read(). Char flavor"""

        table = self.fileh.root.table
        table.flavor = "numpy"
        for colname in table.colnames:
            numcol = table.read(field=colname)
            typecol = table.coltypes[colname]
            itemsizecol = table.description._v_dtypes[colname].base.itemsize
            nctypecode = numcol.dtype.char
            if typecol == "string":
                if itemsizecol > 1:
                    orignumcol = array(['abcd']*self.nrows, dtype='S4')
                else:
                    orignumcol = array(['a']*self.nrows, dtype='S1')
                if common.verbose:
                    print "Typecode of NumPy column read:", nctypecode
                    print "Should look like:", 'c'
                    print "Itemsize of column:", itemsizecol
                    print "Shape of NumPy column read:", numcol.shape
                    print "Should look like:", orignumcol.shape
                    print "First 3 elements of read col:", numcol[:3]
                # Check that both NumPy objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numpy"))

    def test01_readTableNum(self):
        """Checking column conversion into NumPy in read(). NumPy flavor"""

        table = self.fileh.root.table
        table.flavor = "numpy"
        for colname in table.colnames:
            numcol = table.read(field=colname)
            typecol = table.coltypes[colname]
            nctypecode = typeNA[numcol.dtype.char[0]]
            if typecol != "string":
                if common.verbose:
                    print "Typecode of NumPy column read:", nctypecode
                    print "Should look like:", typecol
                orignumcol = ones(shape=self.nrows, dtype=numcol.dtype.char)
                # Check that both NumPy objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numpy"))


    def test02_readCoordsChar(self):
        """Column conversion into NumPy in readCoords(). Chars"""

        table = self.fileh.root.table
        table.flavor = "numpy"
        coords = (1,2,3)
        self.nrows = len(coords)
        for colname in table.colnames:
            numcol = table.readCoordinates(coords, field=colname)
            typecol = table.coltypes[colname]
            itemsizecol = table.description._v_dtypes[colname].base.itemsize
            nctypecode = numcol.dtype.char
            if typecol == "string":
                if itemsizecol > 1:
                    orignumcol = array(['abcd']*self.nrows, dtype='S4')
                else:
                    orignumcol = array(['a']*self.nrows, dtype='S1')
                if common.verbose:
                    print "Typecode of NumPy column read:", nctypecode
                    print "Should look like:", 'c'
                    print "Itemsize of column:", itemsizecol
                    print "Shape of NumPy column read:", numcol.shape
                    print "Should look like:", orignumcol.shape
                    print "First 3 elements of read col:", numcol[:3]
                # Check that both NumPy objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numpy"))

    def test02_readCoordsNum(self):
        """Column conversion into NumPy in readCoordinates(). NumPy."""

        table = self.fileh.root.table
        table.flavor = "numpy"
        coords = (1,2,3)
        self.nrows = len(coords)
        for colname in table.colnames:
            numcol = table.readCoordinates(coords, field=colname)
            typecol = table.coltypes[colname]
            type_ = numcol.dtype.type
            if typecol != "string":
                if typecol == "int64":
                    return
                if common.verbose:
                    print "Type of read NumPy column:", type_
                    print "Should look like:", typecol
                orignumcol = ones(shape=self.nrows, dtype=numcol.dtype.char)
                # Check that both NumPy objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numpy"))

    def test03_getIndexNumPy(self):
        """Getting table rows specifyied as NumPy scalar integers."""

        table = self.fileh.root.table
        coords = numpy.array([1,2,3], dtype='int8')
        for colname in table.colnames:
            numcol = [ table[coord][colname] for coord in coords ]
            typecol = table.coltypes[colname]
            if typecol != "string":
                if typecol == "int64":
                    return
                numcol = numpy.array(numcol, typecol)
                if common.verbose:
                    type_ = numcol.dtype.type
                    print "Type of read NumPy column:", type_
                    print "Should look like:", typecol
                orignumcol = ones(shape=len(numcol), dtype=numcol.dtype.char)
                # Check that both NumPy objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numpy"))

    def test04_setIndexNumPy(self):
        """Setting table rows specifyied as NumPy integers."""

        self.fileh.close()
        self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        table.flavor = "numpy"
        coords = numpy.array([1,2,3], dtype='int8')
        # Modify row 1
        # From PyTables 2.0 on, assignments to records can be done
        # only as tuples (see http://projects.scipy.org/scipy/numpy/ticket/315)
        #table[coords[0]] = ["aasa","x"]+[232]*12
        table[coords[0]] = tuple(["aasa","x"]+[232]*12)
        #record = list(table[coords[0]])
        record = table.read(coords[0])
        if common.verbose:
            print """Original row:
['aasa', 'x', 232, -24, 232, 232, 1, 232L, 232, (232+0j), 232.0, 232L, (232+0j), 232.0]
"""
            print "Read row:\n", record
        self.assertEqual(record['var1'], 'aasa')
        self.assertEqual(record['var2'], 'x')
        self.assertEqual(record['var3'], True)
        self.assertEqual(record['var4'], -24)
        self.assertEqual(record['var7'], 232)


# The declaration of the nested table:
class Info(IsDescription):
    _v_pos = 3
    Name = StringCol(itemsize=2)
    Value = ComplexCol(itemsize=16)

class TestTDescr(IsDescription):

    """A description that has several nested columns."""

    x = Int32Col(dflt=0, shape=2, pos=0) #0
    y = FloatCol(dflt=1, shape=(2,2))
    z = UInt8Col(dflt=1)
    z3 = EnumCol({'r':4, 'g':2, 'b':1}, 'r', 'int32', shape=2)
    color = StringCol(itemsize=4, dflt="ab", pos=2)
    info = Info()
    class Info(IsDescription): #1
        _v_pos = 1
        name = StringCol(itemsize=2)
        value = ComplexCol(itemsize=16, pos=0) #0
        y2 = FloatCol(pos=1) #1
        z2 = UInt8Col()
        class Info2(IsDescription):
            y3 = Time64Col(shape=2)
            name = StringCol(itemsize=2)
            value = ComplexCol(itemsize=16, shape=2)


class TableNativeFlavorTestCase(common.PyTablesTestCase):
    nrows = 100

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, "w")
        table = fileh.createTable(fileh.root, 'table', TestTDescr,
                                  expectedrows=self.nrows)
        table.flavor = "numpy"
        for i in range(self.nrows):
            table.row.append()  # Fill 100 rows with default values
        table.flush()
        self.fileh = fileh

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test01a_basicTableRead(self):
        """Checking the return of a NumPy in read()."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        data = table[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the value of some columns
        # A flat column
        col = table.cols.x[:3]
        self.assertTrue(isinstance(col, ndarray))
        npcol = zeros((3,2), dtype="int32")
        self.assertTrue(allequal(col, npcol, "numpy"))
        # A nested column
        col = table.cols.Info[:3]
        self.assertTrue(isinstance(col, ndarray))
        dtype = [('value', 'c16'),
                 ('y2', 'f8'),
                 ('Info2',
                  [('name', 'S2'),
                   ('value', 'c16', (2,)),
                   ('y3', 'f8', (2,))]),
                 ('name', 'S2'),
                 ('z2', 'u1')]
        npcol = zeros((3,), dtype=dtype)
        self.assertEqual(col.dtype.descr, npcol.dtype.descr)
        if common.verbose:
            print "col-->", col
            print "npcol-->", npcol
        # A copy() is needed in case the buffer can be in different segments
        self.assertEqual(col.copy().data, npcol.data)

    def test01b_basicTableRead(self):
        """Checking the return of a NumPy in read() (strided version)."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        data = table[::3]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the value of some columns
        # A flat column
        col = table.cols.x[:9:3]
        self.assertTrue(isinstance(col, ndarray))
        npcol = zeros((3,2), dtype="int32")
        self.assertTrue(allequal(col, npcol, "numpy"))
        # A nested column
        col = table.cols.Info[:9:3]
        self.assertTrue(isinstance(col, ndarray))
        dtype = [('value', '%sc16' % byteorder),
                 ('y2', '%sf8' % byteorder),
                 ('Info2',
                  [('name', '|S2'),
                   ('value', '%sc16' % byteorder, (2,)),
                   ('y3', '%sf8' % byteorder, (2,))]),
                 ('name', '|S2'),
                 ('z2', '|u1')]
        npcol = zeros((3,), dtype=dtype)
        self.assertEqual(col.dtype.descr, npcol.dtype.descr)
        if common.verbose:
            print "col-->", col
            print "npcol-->", npcol
        # A copy() is needed in case the buffer can be in different segments
        self.assertEqual(col.copy().data, npcol.data)

    def test02_getWhereList(self):
        """Checking the return of NumPy in getWhereList method."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        data = table.getWhereList('z == 1')
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        self.assertTrue(allequal(data, arange(100, dtype="i8"), "numpy"))

    def test03a_readWhere(self):
        """Checking the return of NumPy in readWhere method (strings)."""

        table = self.fileh.root.table
        table.cols.color.createIndex()
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.readWhere('color == "ab"')
        if common.verbose:
            print "Type of read:", type(data)
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), self.nrows)

    def test03b_readWhere(self):
        """Checking the return of NumPy in readWhere method (numeric)."""

        table = self.fileh.root.table
        table.cols.z.createIndex()
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.readWhere('z == 0')
        if common.verbose:
            print "Type of read:", type(data)
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 0)

    def test04a_createTable(self):
        """Checking the Table creation from a numpy recarray."""

        dtype = [('value', '%sc16' % byteorder),
                 ('y2', '%sf8' % byteorder),
                 ('Info2',
                  [('name', '|S2'),
                   ('value', '%sc16' % byteorder, (2,)),
                   ('y3', '%sf8' % byteorder, (2,))]),
                 ('name', '|S2'),
                 ('z2', '|u1')]
        npdata = zeros((3,), dtype=dtype)
        table = self.fileh.createTable(self.fileh.root, 'table2', npdata)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table2
        data = table[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, npdata.dtype.descr)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # A copy() is needed in case the buffer would be in different segments
        self.assertEqual(data.copy().data, npdata.data)

    def test04b_appendTable(self):
        """Checking appending a numpy recarray."""

        table = self.fileh.root.table
        npdata = table[3:6]
        table.append(npdata)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table[-3:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "Last 3 elements of read:", data[-3:]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, npdata.dtype.descr)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # A copy() is needed in case the buffer would be in different segments
        self.assertEqual(data.copy().data, npdata.data)

    def test05a_assignColumn(self):
        """Checking assigning to a column."""

        table = self.fileh.root.table
        table.cols.z[:] = zeros((100,), dtype='u1')
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.z[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        self.assertTrue(allequal(data, zeros((100,), dtype="u1"), "numpy"))

    def test05b_modifyingColumns(self):
        """Checking modifying several columns at once."""

        table = self.fileh.root.table
        xcol = ones((3,2), 'int32')
        ycol = zeros((3,2,2), 'float64')
        zcol = zeros((3,), 'uint8')
        table.modifyColumns(3, 6, 1, [xcol, ycol, zcol], ['x', 'y', 'z'])
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, ycol.dtype.descr)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # A copy() is needed in case the buffer would be in different segments
        self.assertEqual(data.copy().data, ycol.data)

    def test05c_modifyingColumns(self):
        """Checking modifying several columns using a single numpy buffer."""

        table = self.fileh.root.table
        dtype=[('x', 'i4', (2,)), ('y', 'f8', (2, 2)), ('z', 'u1')]
        nparray = zeros((3,), dtype=dtype)
        table.modifyColumns(3, 6, 1, nparray, ['x', 'y', 'z'])
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        ycol = zeros((3, 2, 2), 'float64')
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, ycol.dtype.descr)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # A copy() is needed in case the buffer would be in different segments
        self.assertEqual(data.copy().data, ycol.data)

    def test06a_assignNestedColumn(self):
        """Checking assigning a nested column (using modifyColumn)."""

        table = self.fileh.root.table
        dtype = [('value', '%sc16' % byteorder),
                 ('y2', '%sf8' % byteorder),
                 ('Info2',
                  [('name', '|S2'),
                   ('value', '%sc16' % byteorder, (2,)),
                   ('y3', '%sf8' % byteorder, (2,))]),
                 ('name', '|S2'),
                 ('z2', '|u1')]
        npdata = zeros((3,), dtype=dtype)
        data = table.cols.Info[3:6]
        table.modifyColumn(3, 6, 1, column=npdata, colname='Info')
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.Info[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, npdata.dtype.descr)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # A copy() is needed in case the buffer would be in different segments
        self.assertEqual(data.copy().data, npdata.data)

    def test06b_assignNestedColumn(self):
        """Checking assigning a nested column (using the .cols accessor)."""

        table = self.fileh.root.table
        dtype = [('value', '%sc16' % byteorder),
                 ('y2', '%sf8' % byteorder),
                 ('Info2',
                  [('name', '|S2'),
                   ('value', '%sc16' % byteorder, (2,)),
                   ('y3', '%sf8' % byteorder, (2,))]),
                 ('name', '|S2'),
                 ('z2', '|u1')]
        npdata = zeros((3,), dtype=dtype)
#         self.assertRaises(NotImplementedError,
#                           table.cols.Info.__setitem__, slice(3,6,1),  npdata)
        table.cols.Info[3:6] = npdata
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.Info[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, npdata.dtype.descr)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # A copy() is needed in case the buffer would be in different segments
        self.assertEqual(data.copy().data, npdata.data)

    def test07a_modifyingRows(self):
        """Checking modifying several rows at once (using modifyRows)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[0:3]
        # Modify it somewhat
        chunk['y'][:] = -1
        table.modifyRows(3, 6, 1, rows=chunk)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        ycol = zeros((3,2,2), 'float64')-1
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, ycol.dtype.descr)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        self.assertTrue(allequal(ycol, data, "numpy"))

    def test07b_modifyingRows(self):
        """Checking modifying several rows at once (using cols accessor)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[0:3]
        # Modify it somewhat
        chunk['y'][:] = -1
        table.cols[3:6] = chunk
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check that some column has been actually modified
        ycol = zeros((3,2,2), 'float64')-1
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, ycol.dtype.descr)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        self.assertTrue(allequal(ycol, data, "numpy"))

    def test08a_modifyingRows(self):
        """Checking modifying just one row at once (using modifyRows)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[3:4]
        # Modify it somewhat
        chunk['y'][:] = -1
        table.modifyRows(6, 7, 1, chunk)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check that some column has been actually modified
        ycol = zeros((2,2), 'float64')-1
        data = table.cols.y[6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, ycol.dtype.descr)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        self.assertTrue(allequal(ycol, data, "numpy"))

    def test08b_modifyingRows(self):
        """Checking modifying just one row at once (using cols accessor)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[3:4]
        # Modify it somewhat
        chunk['y'][:] = -1
        table.cols[6] = chunk
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check that some column has been actually modified
        ycol = zeros((2,2), 'float64')-1
        data = table.cols.y[6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, ycol.dtype.descr)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        self.assertTrue(allequal(ycol, data, "numpy"))

    def test09a_getStrings(self):
        """Checking the return of string columns with spaces."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        rdata = table.getWhereList('color == "ab"')
        data = table.readCoordinates(rdata)
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        for idata in data['color']:
            self.assertEqual(idata, array("ab", dtype="|S4"))

    def test09b_getStrings(self):
        """Checking the return of string columns with spaces. (modify)"""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        for i in range(50):
            table.cols.color[i] = "a  "
        table.flush()
        data = table[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        for i in range(100):
            idata = data['color'][i]
            if i >= 50:
                self.assertEqual(idata, array("ab", dtype="|S4"))
            else:
                self.assertEqual(idata, array("a  ", dtype="|S4"))

    def test09c_getStrings(self):
        """Checking the return of string columns with spaces. (append)"""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        row = table.row
        for i in range(50):
            row["color"] = "a  "   # note the trailing spaces
            row.append()
        table.flush()
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        data = self.fileh.root.table[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.dtype.descr
            print "First 3 elements of read:", data[:3]
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 150)
        # Finally, check that the contents are ok
        # Finally, check that the contents are ok
        for i in range(150):
            idata = data['color'][i]
            if i < 100:
                self.assertEqual(idata, array("ab", dtype="|S4"))
            else:
                self.assertEqual(idata, array("a  ", dtype="|S4"))

class TableNativeFlavorOpenTestCase(TableNativeFlavorTestCase):
    close = 0

class TableNativeFlavorCloseTestCase(TableNativeFlavorTestCase):
    close = 1

class AttributesTestCase(common.PyTablesTestCase):

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        groups = self.fileh.createGroup(self.fileh.root, 'group')

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test01_writeAttribute(self):
        """Checking the creation of a numpy attribute."""
        group = self.fileh.root.group
        g_attrs = group._v_attrs
        g_attrs.numpy1 = zeros((1,1), dtype='int16')
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            group = self.fileh.root.group
            g_attrs = group._v_attrs
        # Check that we can retrieve a numpy object
        data = g_attrs.numpy1
        npcomp = zeros((1,1), dtype='int16')
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, npcomp.dtype.descr)
        if common.verbose:
            print "npcomp-->", npcomp
            print "data-->", data
        self.assertTrue(allequal(npcomp, data, "numpy"))

    def test02_updateAttribute(self):
        """Checking the modification of a numpy attribute."""

        group = self.fileh.root.group
        g_attrs = group._v_attrs
        g_attrs.numpy1 = zeros((1,2), dtype='int16')
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            group = self.fileh.root.group
            g_attrs = group._v_attrs
        # Update this attribute
        g_attrs.numpy1 = ones((1,2), dtype='int16')
        # Check that we can retrieve a numpy object
        data = g_attrs.numpy1
        npcomp = ones((1,2), dtype='int16')
        # Check that both NumPy objects are equal
        self.assertTrue(isinstance(data, ndarray))
        # Check the type
        self.assertEqual(data.dtype.descr, npcomp.dtype.descr)
        if common.verbose:
            print "npcomp-->", npcomp
            print "data-->", data
        self.assertTrue(allequal(npcomp, data, "numpy"))

class AttributesOpenTestCase(AttributesTestCase):
    close = 0

class AttributesCloseTestCase(AttributesTestCase):
    close = 1

class StrlenTestCase(common.PyTablesTestCase):

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.createGroup(self.fileh.root, 'group')
        tablelayout = {'Text': StringCol(itemsize=1000),}
        self.table = self.fileh.createTable(group, 'table', tablelayout)
        self.table.flavor = 'numpy'
        row = self.table.row
        row['Text'] = 'Hello Francesc!'
        row.append()
        row['Text'] = 'Hola Francesc!'
        row.append()
        self.table.flush()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test01(self):
        """Checking the lengths of strings (read field)."""
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            self.table = self.fileh.root.group.table
        # Get both strings
        str1 = self.table.col('Text')[0]
        str2 = self.table.col('Text')[1]
        if common.verbose:
            print "string1-->", str1
            print "string2-->", str2
        # Check that both NumPy objects are equal
        self.assertEqual(len(str1), len('Hello Francesc!'))
        self.assertEqual(len(str2), len('Hola Francesc!'))
        self.assertEqual(str1, 'Hello Francesc!')
        self.assertEqual(str2, 'Hola Francesc!')

    def test02(self):
        """Checking the lengths of strings (read recarray)."""
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            self.table = self.fileh.root.group.table
        # Get both strings
        str1 = self.table[:]['Text'][0]
        str2 = self.table[:]['Text'][1]
        # Check that both NumPy objects are equal
        self.assertEqual(len(str1), len('Hello Francesc!'))
        self.assertEqual(len(str2), len('Hola Francesc!'))
        self.assertEqual(str1, 'Hello Francesc!')
        self.assertEqual(str2, 'Hola Francesc!')


    def test03(self):
        """Checking the lengths of strings (read recarray, row by row)."""
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            self.table = self.fileh.root.group.table
        # Get both strings
        str1 = self.table[0]['Text']
        str2 = self.table[1]['Text']
        # Check that both NumPy objects are equal
        self.assertEqual(len(str1), len('Hello Francesc!'))
        self.assertEqual(len(str2), len('Hola Francesc!'))
        self.assertEqual(str1, 'Hello Francesc!')
        self.assertEqual(str2, 'Hola Francesc!')


class StrlenOpenTestCase(StrlenTestCase):
    close = 0

class StrlenCloseTestCase(StrlenTestCase):
    close = 1


#--------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    #theSuite.addTest(unittest.makeSuite(StrlenOpenTestCase))
    #theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
    #theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))
    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DThreeTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2DTestCase))
        theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))
        theSuite.addTest(unittest.makeSuite(TableReadTestCase))
        theSuite.addTest(unittest.makeSuite(TableNativeFlavorOpenTestCase))
        theSuite.addTest(unittest.makeSuite(TableNativeFlavorCloseTestCase))
        theSuite.addTest(unittest.makeSuite(AttributesOpenTestCase))
        theSuite.addTest(unittest.makeSuite(AttributesCloseTestCase))
        theSuite.addTest(unittest.makeSuite(StrlenOpenTestCase))
        theSuite.addTest(unittest.makeSuite(StrlenCloseTestCase))
        if common.heavy:
            theSuite.addTest(unittest.makeSuite(Basic10DTestCase))
            # The 32 dimensions case takes forever to run!!
            # theSuite.addTest(unittest.makeSuite(Basic32DTestCase))
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
