import sys
import unittest
import os
import tempfile

from numarray import strings
from numarray import records
from numarray import *

import tables
from tables import *
from tables import nra
from tables.tests import common
from tables.tests.common import allequal


# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

types = ['Int8', 'Int16', 'Int32', 'Int64', 'Float32', 'Float64']
types += ['UInt8', 'UInt16', 'UInt32', 'Complex32', 'Complex64']
# UInt64 checking disabled on win platforms
# because this type is not supported
if sys.platform != 'win32':
    types += ['UInt64']
types += ['Bool']

class BasicTestCase(unittest.TestCase):
    """Basic test for all the supported types present in numarray.
    All of them are included on PyTables.
    """
    endiancheck = 0

    def WriteRead(self, testArray):
        if common.verbose:
            print '\n', '-=' * 30
            if type(testArray) == NumArray:
                type_ = testArray.type()
            else:
                type_ = "String"
            print "Running test for array with type '%s'" % \
                  type_,
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
        # For cases that read returns a python type instead of a numarray type
        if not hasattr(b, "shape"):
            b = array(b, type=a.type())

        # Compare them. They should be equal.
        #if not allequal(a,b, "numarray") and common.verbose:
        if common.verbose and type(a) == NumArray:
            print "Array written:", a
            print "Array written shape:", a.shape
            print "Array written itemsize:", a.itemsize
            print "Array written type:", a.type()
            print "Array read:", b
            print "Array read shape:", b.shape
            print "Array read itemsize:", b.itemsize
            print "Array read type:", b.type()

        type_ = self.root.somearray.atom.type
        # Check strictly the array equality
        self.assertEqual(type(a), type(b))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape, self.root.somearray.shape)
        if type(a) == strings.CharArray:
            self.assertEqual(type_, "string")
        else:
            self.assertEqual(a.type(), b.type())
            if not type_.startswith('complex'):
                self.assertEqual(str(a.type()).lower(), type_)
            else:
                if type_ == 'complex64':
                    self.assertEqual(str(a.type()), "Complex32")
                else:
                    self.assertEqual(str(a.type()), "Complex64")

        self.assertTrue(allequal(a,b, "numarray"))
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
        if a.shape == ():
            b = a               # We cannot use the indexing notation
        else:
            b = a[::2]
            # Ensure that this numarray string is non-contiguous
            if a.shape[0] > 2:
                self.assertEqual(b.iscontiguous(), 0)
        self.WriteRead(b)
        return

    def test02_types(self):
        "Data integrity during recovery (numerical types)"

        for type_ in types:
            if self.tupleInt.shape:
                a = self.tupleInt.astype(type_)
            else:
                # shape is the empty tuple ()
                a = array(self.tupleInt, type=type_)
            self.WriteRead(a)

        return

    def test03_types_nc(self):
        "Data integrity during recovery (non-contiguous numerical types)"

        for type_ in types:
            if self.tupleInt.shape:
                a = self.tupleInt.astype(type_)
            else:
                # shape is the empty tuple ()
                a = array(self.tupleInt, dtype=type_)
            # This should not be tested for the rank-0 case
            if len(a.shape) == 0:
                return
            b = a[::2]
            # Ensure that this array is non-contiguous (for non-trivial case)
            if a.shape[0] > 2:
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
    tupleInt = ones((2,)*10, 'Int32')
    # The next tuple consumes far more time, so this
    # test should be run in common.heavy mode.
    # Dimensions greather than 6 in numarray strings gives some warnings
    tupleChar = strings.array("abc"*2**6, shape=(2,)*6, itemsize=3)



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
        for type_ in types:
            # Create an array of type_, with incrementally bigger ranges
            a = ones((2,) * i, type_)
            # Save it on the HDF5 file
            dsetname = 'array_' + type_
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
        for i in range(1,len(types)):
            # Create an array for later comparison
            a = ones((2,) * i, types[i - 1])
            # Get the dset object hanging from group
            dset = getattr(group, 'array_' + types[i-1])
            # Get the actual array
            b = dset.read()
            if not allequal(a,b, "numarray") and common.verbose:
                print "Array a original. Shape: ==>", a.shape
                print "Array a original. Data: ==>", a
                print "Info from dataset:", dset._v_pathname
                print "  shape ==>", dset.shape,
                print "  type ==> %s" % dset.atom.type
                print "Array b read from file. Shape: ==>", b.shape,
                print ". Type ==> %s" % b.type()

            self.assertEqual(a.shape, b.shape)
            self.assertTrue(allequal(a,b, "numarray"))

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
            a = ones((1,) * rank, 'Int32')
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
            a = ones((1,) * rank, 'Int32')
            # Get the actual array
            b = group.array.read()
            if common.verbose:
                print "%3d," % (rank),
            if not a.tolist() == b.tolist() and common.verbose:
                print "Array b read from file. Shape: ==>", b.shape,
                print ". Type ==> %c" % b.type()
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.type(), b.type())

            self.assertTrue(allequal(a, b, "numarray"))

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
    var3  = BoolCol(dflt=1)
    var4  = Int8Col(dflt=1)
    var5  = UInt8Col(dflt=1)
    var6  = Int16Col(dflt=1)
    var7  = UInt16Col(dflt=1)
    var8  = Int32Col(dflt=1)
    var9  = UInt32Col(dflt=1)
    var10 = Int64Col(dflt=1)
    var11 = Float32Col(dflt=1.0)
    var12 = Float64Col(dflt=1.0)
    var13 = ComplexCol(dflt=(1.+0.j), itemsize=8)
    var14 = ComplexCol(dflt=(1.+0.j), itemsize=16)


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
        """Checking column conversion into numarray in read(). Chars."""

        table = self.fileh.root.table
        table.flavor = "numarray"
        for colname in table.colnames:
            numcol = table.read(field=colname)
            typecol = table.coltypes[colname]
            itemsizecol = table.description._v_dtypes[colname].base.itemsize
            if typecol == "string":
                if itemsizecol > 1:
                    orignumcol = strings.array(['abcd']*self.nrows, itemsize=4)
                else:
                    orignumcol = strings.array(['a']*self.nrows, itemsize=1)
                if common.verbose:
                    print "Itemsize of column:", itemsizecol
                    print "Shape of numarray column read:", numcol.shape
                    print "Should look like:", orignumcol.shape
                    print "First 3 elements of read col:", numcol[:3]
                # Check that both numarray objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numarray"))

    def test01_readTableNum(self):
        """Checking column conversion into numarray in read(). Numerical."""

        table = self.fileh.root.table
        table.flavor="numarray"
        for colname in table.colnames:
            numcol = table.read(field=colname)
            typecol = table.coltypes[colname]
            if typecol != "string":
                type_ = numcol.type()
                if common.verbose:
                    print "Type of numarray column read:", type_
                    print "Should look like:", typecol
                orignumcol = ones(shape=self.nrows, dtype=numcol.type())
                # Check that both numarray objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numarray"))


    def test02_readCoordsChar(self):
        """Column conversion into numarray in readCoordinates(). Chars."""

        table = self.fileh.root.table
        table.flavor = "numarray"
        coords = (1,2,3)
        self.nrows = len(coords)
        for colname in table.colnames:
            numcol = table.readCoordinates(coords, field=colname)
            typecol = table.coltypes[colname]
            itemsizecol = table.description._v_dtypes[colname].base.itemsize
            if typecol == "string":
                if itemsizecol > 1:
                    orignumcol = strings.array(['abcd']*self.nrows, itemsize=4)
                else:
                    orignumcol = strings.array(['a']*self.nrows, itemsize=1)
                if common.verbose:
                    print "Itemsize of column:", itemsizecol
                    print "Shape of numarray column read:", numcol.shape
                    print "Should look like:", orignumcol.shape
                    print "First 3 elements of read col:", numcol[:3]
                # Check that both numarray objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numarray"))

    def test02_readCoordsNum(self):
        """Column conversion into numarray in readCoordinates(). Numerical."""

        table = self.fileh.root.table
        table.flavor="numarray"
        coords = (1,2,3)
        self.nrows = len(coords)
        for colname in table.colnames:
            numcol = table.readCoordinates(coords, field=colname)
            typecol = table.coltypes[colname]
            if typecol != "string":
                type_ = numcol.type()
                if typecol == "int64":
                    return
                if common.verbose:
                    print "Type of read numarray column:", type_
                    print "Should look like:", typecol
                orignumcol = ones(shape=self.nrows, type=numcol.type())
                # Check that both numarray objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numarray"))

    def test03_getIndexnumarray(self):
        """Getting table rows specifyied as numarray scalar integers."""

        table = self.fileh.root.table
        coords = array([1,2,3], type='Int8')
        for colname in table.colnames:
            numcol = [ table[coord][colname].item() for coord in coords ]
            typecol = table.coltypes[colname]
            if typecol != "string":
                if typecol == "bool":  # Special case for boolean translation
                    typecol = "Bool"
                numcol = array(numcol, dtype=typecol)
                if common.verbose:
                    type_ = numcol.type()
                    print "Type of read numarray column:", type_
                    print "Should look like:", typecol
                orignumcol = ones(shape=len(numcol), type=numcol.type())
                # Check that both numarray objects are equal
                self.assertTrue(allequal(numcol, orignumcol, "numarray"))

    def test04_setIndexnumarray(self):
        """Setting table rows specifyied as numarray integers."""

        self.fileh.close()
        self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        table.flavor = "numarray"
        coords = array([1,2,3], dtype='int8')
        # Modify row 1
        # From PyTables 2.0 on, assignments to records can be done
        # only as tuples (see http://projects.scipy.org/scipy/numarray/ticket/315)
        #table[coords[0]] = ["aasa","x"]+[232]*12
        table[coords[0]] = tuple(["aasa","x"]+[232]*12)
        #record = list(table[coords[0]])
        record = table.read(coords[0])[0]
        if common.verbose:
            print """Original row:
['aasa', 'x', 232, -24, 232, 232, 1, 232L, 232, (232+0j), 232.0, 232L, (232+0j), 232.0]
"""
            print "Read row:\n", record
        self.assertEqual(record.field('var1'), 'aasa')
        self.assertEqual(record.field('var2'), 'x')
        self.assertEqual(record.field('var3'), True)
        self.assertEqual(record.field('var4'), -24)
        self.assertEqual(record.field('var7'), 232)


# The declaration of the nested table:
class Info(IsDescription):
    _v_pos = 3
    Name = StringCol(itemsize=2)
    Value = ComplexCol(itemsize=16)

class TestTDescr(IsDescription):

    """A description that has several nested columns."""

    x = Int32Col(dflt=0, shape=2, pos=0) #0
    y = Float64Col(dflt=1, shape=(2,2))
    z = UInt8Col(dflt=1)
    z3 = EnumCol({'r':4, 'g':2, 'b':1}, dflt='r', base='int32', shape=2)
    color = StringCol(itemsize=4, dflt="ab", pos=2)
    info = Info()
    class Info(IsDescription): #1
        _v_pos = 1
        name = StringCol(itemsize=2)
        value = ComplexCol(itemsize=16, pos=0) #0
        y2 = Float64Col(pos=1) #1
        z2 = UInt8Col()
        class Info2(IsDescription):
            y3 = Time64Col(shape=2)
            name = StringCol(itemsize=2)
            value = ComplexCol(itemsize=16, shape=2)


class TableNativeFlavorTestCase(common.PyTablesTestCase):
    nrows = 100

    dtype = [('value', 'c16'),
             ('y2', 'f8'),
             ('Info2',
              [('name', 'a2'),
               ('value', '(2,)c16'),
               ('y3', '(2,)f8')]),
             ('name', 'a2'),
             ('z2', 'u1')]
    _infozeros = nra.array(descr=dtype, shape=3)
    # Set the contents to zero (or empty strings)
    _infozeros.field('value')[:] = 0
    _infozeros.field('y2')[:] = 0
    _infozeros.field('Info2/name')[:] = "\0"
    _infozeros.field('Info2/value')[:] = 0
    _infozeros.field('Info2/y3')[:] = 0
    _infozeros.field('name')[:] = "\0"
    _infozeros.field('z2')[:] = 0

    _infoones = nra.array(descr=dtype, shape=3)
    # Set the contents to one (or blank strings)
    _infoones.field('value')[:] = 1
    _infoones.field('y2')[:] = 1
    _infoones.field('Info2/name')[:] = " "
    _infoones.field('Info2/value')[:] = 1
    _infoones.field('Info2/y3')[:] = 1
    _infoones.field('name')[:] = " "
    _infoones.field('z2')[:] = 1

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, "w")
        table = fileh.createTable(fileh.root, 'table', TestTDescr,
                                  expectedrows=self.nrows)
        table.flavor = 'numarray'
        for i in range(self.nrows):
            table.row.append()  # Fill 100 rows with default values
        table.flush()
        self.fileh = fileh

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test01a_basicTableRead(self):
        """Checking the return of a numarray in read()."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        data = table[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Formats of the record:", data._formats
            print "First 3 elements of read:", data[:3]
        # Check the type of the recarray
        self.assertTrue(isinstance(data, records.RecArray))
        # Check the value of some columns
        # A flat column
        col = table.cols.x[:3]
        self.assertTrue(isinstance(col, NumArray))
        npcol = zeros((3,2), type="Int32")
        if common.verbose:
            print "Plain column:"
            print "read column-->", col
            print "should look like-->", npcol
        self.assertTrue(allequal(col, npcol, "numarray"))
        # A nested column
        col = table.cols.Info[:3]
        self.assertTrue(isinstance(col, records.RecArray))
        npcol = self._infozeros
        if common.verbose:
            print "Nested column:"
            print "read column-->", col
            print "should look like-->", npcol
        self.assertEqual(col.descr, npcol.descr)
        self.assertEqual(str(col), str(npcol))

    def test01b_basicTableRead(self):
        """Checking the return of a numarray in read() (strided version)."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        data = table[::3]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check the value of some columns
        # A flat column
        col = table.cols.x[:9:3]
        self.assertTrue(isinstance(col, NumArray))
        npcol = zeros((3,2), dtype="Int32")
        if common.verbose:
            print "Plain column:"
            print "read column-->", col
            print "should look like-->", npcol
        self.assertTrue(allequal(col, npcol, "numarray"))
        # A nested column
        col = table.cols.Info[:9:3]
        self.assertTrue(isinstance(col, records.RecArray))
        npcol = self._infozeros
        if common.verbose:
            print "Nested column:"
            print "read column-->", col
            print "should look like-->", npcol
        self.assertEqual(col.descr, npcol.descr)
        self.assertEqual(str(col), str(npcol))

    def test02_getWhereList(self):
        """Checking the return of numarray in getWhereList method."""

        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
        table = self.fileh.root.table
        data = table.getWhereList('z == 1')
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        self.assertTrue(allequal(data, arange(100, type="Int64"), "numarray"))

    def test03a_readWhere(self):
        """Checking the return of numarray in readWhere method (strings)."""

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
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), self.nrows)

    def test03b_readWhere(self):
        """Checking the return of numarray in readWhere method (numeric)."""

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
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 0)

    def test04a_createTable(self):
        """Checking the Table creation from a numarray recarray."""

        npdata = self._infozeros
        table = self.fileh.createTable(self.fileh.root, 'table2', npdata)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table2
        data = table[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check the type
        self.assertEqual(data.descr, npdata.descr)
        self.assertEqual(str(data), str(npdata))

    def test04b_appendTable(self):
        """Checking appending a numarray recarray."""

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
            print "Description of the record:", data.descr
            print "Last 3 elements of read:", data[-3:]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check the type
        self.assertEqual(data.descr, npdata.descr)
        self.assertEqual(str(data), str(npdata))

    def test05a_assignColumn(self):
        """Checking assigning to a column."""

        table = self.fileh.root.table
        table.cols.z[:] = ones((100,), dtype='UInt8')
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.z[:]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        self.assertTrue(allequal(data, ones((100,), dtype="UInt8"), "numarray"))

    def test05b_modifyingColumns(self):
        """Checking modifying several columns at once."""

        table = self.fileh.root.table
        xcol = ones((3,2), 'Int32')
        ycol = ones((3,2,2), 'Float64')
        zcol = zeros((3,), 'UInt8')
        table.modifyColumns(3, 6, 1, [xcol, ycol, zcol], ['x', 'y', 'z'])
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check the type
        self.assertEqual(data.type(), ycol.type())
        self.assertTrue(allequal(data, ycol, "numarray"))

    def test05c_modifyingColumns(self):
        """Checking modifying several columns using a numarray buffer."""

        table = self.fileh.root.table
        dtype=[('x', '(2,)i4'), ('y', '(2,2)f8'), ('z', 'u1')]
        nparray = nra.array(shape=(3,), descr=dtype)
        nparray.field('x')[:] = 1
        nparray.field('y')[:] = 1
        nparray.field('z')[:] = 2
        table.modifyColumns(3, 6, 1, nparray, ['x', 'y', 'z'])
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        ycol = ones((3, 2, 2), 'Float64')
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check the type
        self.assertEqual(data.type(), ycol.type())
        self.assertEqual(str(data), str(ycol))

    def test06a_assignNestedColumn(self):
        """Checking assigning a nested column (using modifyColumn)."""

        npdata = self._infoones
        table = self.fileh.root.table
        data = table.cols.Info[3:6]
        table.modifyColumn(3, 6, 1, column=npdata, colname='Info')
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.Info[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check the type
        self.assertEqual(data.descr, npdata.descr)
        self.assertEqual(str(data), str(npdata))

    def test06b_assignNestedColumn(self):
        """Checking assigning a nested column (using the .cols accessor)."""

        table = self.fileh.root.table
        npdata = self._infoones
        table.cols.Info[3:6] = npdata
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        data = table.cols.Info[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "npdata-->", npdata
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check the type
        self.assertEqual(data.descr, npdata.descr)
        self.assertEqual(str(data), str(npdata))

    def test07a_modifyingRows(self):
        """Checking modifying several rows at once (using modifyRows)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[0:3]
        # Modify it somewhat
        chunk.field('y')[:] = -1
        table.modifyRows(3, 6, 1, rows=chunk)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        ycol = zeros((3,2,2), 'Float64')-1
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check the type
        self.assertEqual(data.type(), ycol.type())
        self.assertTrue(allequal(ycol, data, "numarray"))

    def test07b_modifyingRows(self):
        """Checking modifying several rows at once (using cols accessor)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[0:3]
        # Modify it somewhat
        chunk.field('y')[:] = -1
        table.cols[3:6] = chunk
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check that some column has been actually modified
        ycol = zeros((3,2,2), 'Float64')-1
        data = table.cols.y[3:6]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check the type
        self.assertEqual(data.type(), ycol.type())
        self.assertTrue(allequal(ycol, data, "numarray"))

    def test08a_modifyingRows(self):
        """Checking modifying just one row at once (using modifyRows)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[3]
        # Modify it somewhat
        chunk.field('y')[:] = -1
        table.modifyRows(6, 7, 1, chunk)
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check that some column has been actually modified
        ycol = zeros((2,2), 'Float64')-1
        data = table.cols.y[6]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check the type
        self.assertEqual(data.type(), ycol.type())
        self.assertTrue(allequal(ycol, data, "numarray"))

    def test08b_modifyingRows(self):
        """Checking modifying just one row at once (using cols accessor)."""

        table = self.fileh.root.table
        # Read a chunk of the table
        chunk = table[3]
        # Modify it somewhat
        chunk['y'][:] = -1
        table.cols[6] = chunk
        if self.close:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check that some column has been actually modified
        ycol = zeros((2,2), 'Float64')-1
        data = table.cols.y[6]
        if common.verbose:
            print "Type of read:", type(data)
            print "First 3 elements of read:", data[:3]
            print "Length of the data read:", len(data)
        if common.verbose:
            print "ycol-->", ycol
            print "data-->", data
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, NumArray))
        # Check the type
        self.assertEqual(data.type(), ycol.type())
        self.assertTrue(allequal(ycol, data, "numarray"))

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
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        for idata in data.field('color'):
            self.assertEqual(idata, "ab")

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
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 100)
        # Finally, check that the contents are ok
        for i in range(100):
            idata = data.field('color')[i]
            if i >= 50:
                self.assertEqual(idata, "ab")
            else:
                self.assertEqual(idata, "a")

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
            print "Description of the record:", data.descr
            print "First 3 elements of read:", data[:3]
        # Check that both numarray objects are equal
        self.assertTrue(isinstance(data, records.RecArray))
        # Check that all columns have been selected
        self.assertEqual(len(data), 150)
        # Finally, check that the contents are ok
        # Finally, check that the contents are ok
        for i in range(150):
            idata = data.field('color')[i]
            if i < 100:
                self.assertEqual(idata, "ab")
            else:
                self.assertEqual(idata, "a")

class TableNativeFlavorOpenTestCase(TableNativeFlavorTestCase):
    close = 0

class TableNativeFlavorCloseTestCase(TableNativeFlavorTestCase):
    close = 1

class StrlenTestCase(common.PyTablesTestCase):

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.createGroup(self.fileh.root, 'group')
        tablelayout = {'Text': StringCol(itemsize=1000),}
        self.table = self.fileh.createTable(group, 'table', tablelayout)
        self.table.flavor = 'numarray'
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
        # Check that both numarray objects are equal
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
        str1 = self.table[:].field('Text')[0]
        str2 = self.table[:].field('Text')[1]
        # Check that both numarray objects are equal
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
        str1 = self.table[0].field('Text')
        str2 = self.table[1].field('Text')
        # Check that both numarray objects are equal
        self.assertEqual(len(str1), len('Hello Francesc!'))
        self.assertEqual(len(str2), len('Hola Francesc!'))
        self.assertEqual(str1, 'Hello Francesc!')
        self.assertEqual(str2, 'Hola Francesc!')


class StrlenOpenTestCase(StrlenTestCase):
    close = 0

class StrlenCloseTestCase(StrlenTestCase):
    close = 1


class ScalarTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def test(self):
        """Reading scalar arrays (see #98)."""

        arr = self.h5file.createArray('/', 'scalar_na', 1234)
        arr.flavor = 'numarray'

        self._reopen()

        arr = self.h5file.root.scalar_na

        common.verbosePrint("* %r == %r ?" % (arr.read(), array(1234)))
        self.assertTrue(all(arr.read() == array(1234)))
        common.verbosePrint("* %r == %r ?" % (arr[()], array(1234)))
        self.assertTrue(all(arr[()] == 1234))


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
        theSuite.addTest(unittest.makeSuite(StrlenOpenTestCase))
        theSuite.addTest(unittest.makeSuite(StrlenCloseTestCase))
        theSuite.addTest(unittest.makeSuite(ScalarTestCase))
        if common.heavy:
            theSuite.addTest(unittest.makeSuite(Basic10DTestCase))
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
