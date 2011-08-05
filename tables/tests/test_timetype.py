########################################################################
#
#       License: BSD
#       Created: December 15, 2004
#       Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/test/test_timetype.py,v $
#       $Id$
#
########################################################################

"Unit test for the Time datatypes."

import unittest
import tempfile
import os

import numpy

import tables
from tables.tests import common
from tables.tests.common import allequal

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


__revision__ = '$Id$'



class LeafCreationTestCase(common.PyTablesTestCase):
    "Tests creating Tables, VLArrays an EArrays with Time data."

    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the writable, empty, temporary HDF5 file
        """

        self.h5fname = tempfile.mktemp(suffix = '.h5')
        self.h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for creating a time leaves")


    def tearDown(self):
        """tearDown() -> None

        Closes 'h5file'; removes 'h5fname'.
        """

        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)


    def test00_UnidimLeaves(self):
        "Creating new nodes with unidimensional time elements."

        # Table creation.
        class MyTimeRow(tables.IsDescription):
            intcol = tables.IntCol()
            t32col = tables.Time32Col()
            t64col = tables.Time64Col()
        self.h5file.createTable('/', 'table', MyTimeRow)

        # VLArray creation.
        self.h5file.createVLArray('/', 'vlarray4', tables.Time32Atom())
        self.h5file.createVLArray('/', 'vlarray8', tables.Time64Atom())

        # EArray creation.
        self.h5file.createEArray('/', 'earray4',
                                 tables.Time32Atom(), shape=(0,))
        self.h5file.createEArray('/', 'earray8',
                                 tables.Time64Atom(), shape=(0,))


    def test01_MultidimLeaves(self):
        "Creating new nodes with multidimensional time elements."

        # Table creation.
        class MyTimeRow(tables.IsDescription):
            intcol = tables.IntCol(shape = (2, 1))
            t32col = tables.Time32Col(shape = (2, 1))
            t64col = tables.Time64Col(shape = (2, 1))
        self.h5file.createTable('/', 'table', MyTimeRow)

        # VLArray creation.
        self.h5file.createVLArray(
                '/', 'vlarray4', tables.Time32Atom(shape = (2, 1)))
        self.h5file.createVLArray(
                '/', 'vlarray8', tables.Time64Atom(shape = (2, 1)))

        # EArray creation.
        self.h5file.createEArray(
                '/', 'earray4', tables.Time32Atom(), shape=(0, 2, 1))
        self.h5file.createEArray(
                '/', 'earray8', tables.Time64Atom(), shape=(0, 2, 1))



class OpenTestCase(common.PyTablesTestCase):
    "Tests opening a file with Time nodes."

    # The description used in the test Table.
    class MyTimeRow(tables.IsDescription):
        t32col = tables.Time32Col(shape = (2, 1))
        t64col = tables.Time64Col(shape = (2, 1))

    # The atoms used in the test VLArrays.
    myTime32Atom = tables.Time32Atom(shape = (2, 1))
    myTime64Atom = tables.Time64Atom(shape = (2, 1))


    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file with '/table',
            '/vlarray4' and '/vlarray8' nodes.
        """

        self.h5fname = tempfile.mktemp(suffix = '.h5')

        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for creating time leaves")

        # Create test Table.
        h5file.createTable('/', 'table', self.MyTimeRow)

        # Create test VLArrays.
        h5file.createVLArray('/', 'vlarray4', self.myTime32Atom)
        h5file.createVLArray('/', 'vlarray8', self.myTime64Atom)

        h5file.close()


    def tearDown(self):
        """tearDown() -> None

        Removes 'h5fname'.
        """

        os.remove(self.h5fname)


    def test00_OpenFile(self):
        "Opening a file with Time nodes."

        h5file = tables.openFile(self.h5fname)

        # Test the Table node.
        tbl = h5file.root.table
        self.assertEqual(
                tbl.coldtypes['t32col'],
                self.MyTimeRow.columns['t32col'].dtype,
                "Column dtypes do not match.")
        self.assertEqual(
                tbl.coldtypes['t64col'],
                self.MyTimeRow.columns['t64col'].dtype,
                "Column dtypes do not match.")

        # Test the VLArray nodes.
        vla4 = h5file.root.vlarray4
        self.assertEqual(
                vla4.atom.dtype, self.myTime32Atom.dtype,
                "Atom types do not match.")
        self.assertEqual(
                vla4.atom.shape, self.myTime32Atom.shape,
                "Atom shapes do not match.")

        vla8 = h5file.root.vlarray8
        self.assertEqual(
                vla8.atom.dtype, self.myTime64Atom.dtype,
                "Atom types do not match.")
        self.assertEqual(
                vla8.atom.shape, self.myTime64Atom.shape,
                "Atom shapes do not match.")

        h5file.close()


    def test01_OpenFileStype(self):
        "Opening a file with Time nodes, comparing Atom.stype."

        h5file = tables.openFile(self.h5fname)

        # Test the Table node.
        tbl = h5file.root.table
        self.assertEqual(
                tbl.coltypes['t32col'],
                self.MyTimeRow.columns['t32col'].type,
                "Column types do not match.")
        self.assertEqual(
                tbl.coltypes['t64col'],
                self.MyTimeRow.columns['t64col'].type,
                "Column types do not match.")

        # Test the VLArray nodes.
        vla4 = h5file.root.vlarray4
        self.assertEqual(
                vla4.atom.type, self.myTime32Atom.type,
                "Atom types do not match.")

        vla8 = h5file.root.vlarray8
        self.assertEqual(
                vla8.atom.type, self.myTime64Atom.type,
                "Atom types do not match.")

        h5file.close()



class CompareTestCase(common.PyTablesTestCase):
    "Tests whether stored and retrieved time data is kept the same."

    # The description used in the test Table.
    class MyTimeRow(tables.IsDescription):
        t32col = tables.Time32Col(pos = 0)
        t64col = tables.Time64Col(shape = (2,), pos = 1)

    # The atoms used in the test VLArrays.
    myTime32Atom = tables.Time32Atom(shape = (2,))
    myTime64Atom = tables.Time64Atom(shape = (2,))


    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
        """

        self.h5fname = tempfile.mktemp(suffix = '.h5')


    def tearDown(self):
        """tearDown() -> None

        Removes 'h5fname'.
        """

        os.remove(self.h5fname)


    def test00_Compare32VLArray(self):
        "Comparing written 32-bit time data with read data in a VLArray."

        wtime = numpy.array((1234567890,) * 2, numpy.int32)

        # Create test VLArray with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time32 VL arrays")
        vla = h5file.createVLArray('/', 'test', self.myTime32Atom)
        vla.append(wtime)
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        rtime = h5file.root.test.read()[0][0]
        h5file.close()
        self.assertTrue(allequal(rtime, wtime),
                        "Stored and retrieved values do not match.")


    def test01_Compare64VLArray(self):
        "Comparing written 64-bit time data with read data in a VLArray."

        wtime = numpy.array((1234567890.123456,) * 2, numpy.float64)

        # Create test VLArray with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time64 VL arrays")
        vla = h5file.createVLArray('/', 'test', self.myTime64Atom)
        vla.append(wtime)
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        rtime = h5file.root.test.read()[0][0]
        h5file.close()
        self.assertTrue(allequal(rtime, wtime),
                        "Stored and retrieved values do not match.")


    def test01b_Compare64VLArray(self):
        "Comparing several written and read 64-bit time values in a VLArray."

        # Create test VLArray with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time64 VL arrays")
        vla = h5file.createVLArray('/', 'test', self.myTime64Atom)

        # Size of the test.
        nrows = vla.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        #nrows = 10

        for i in xrange(nrows):
            j = i*2
            vla.append((j + 0.012, j + 1 + 0.012))
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        arr = h5file.root.test.read()
        h5file.close()

        arr = numpy.array(arr)
        orig_val = numpy.arange(0, nrows*2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 1, 2)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", arr
        self.assertTrue(allequal(arr, orig_val),
                        "Stored and retrieved values do not match.")


    def test02_CompareTable(self):
        "Comparing written time data with read data in a Table."

        wtime = 1234567890.123456

        # Create test Table with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time tables")
        tbl = h5file.createTable('/', 'test', self.MyTimeRow)
        row = tbl.row
        row['t32col'] = int(wtime)
        row['t64col'] = (wtime, wtime)
        row.append()
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        recarr = h5file.root.test.read(0)
        h5file.close()

        self.assertEqual(recarr['t32col'][0], int(wtime),
                         "Stored and retrieved values do not match.")

        comp = (recarr['t64col'][0] == numpy.array((wtime, wtime)))
        self.assertTrue(numpy.alltrue(comp),
                        "Stored and retrieved values do not match.")


    def test02b_CompareTable(self):
        "Comparing several written and read time values in a Table."

        # Create test Table with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time tables")
        tbl = h5file.createTable('/', 'test', self.MyTimeRow)

        # Size of the test.
        nrows = tbl.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        ##nrows = 10

        row = tbl.row
        for i in xrange(nrows):
            row['t32col'] = i
            j = i*2
            row['t64col'] = (j+0.012, j+1+0.012)
            row.append()
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        recarr = h5file.root.test.read()
        h5file.close()

        # Time32 column.
        orig_val = numpy.arange(nrows, dtype=numpy.int32)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", recarr['t32col'][:]
        self.assertTrue(numpy.alltrue(recarr['t32col'][:] == orig_val),
                        "Stored and retrieved values do not match.")

        # Time64 column.
        orig_val = numpy.arange(0, nrows*2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 2)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", recarr['t64col'][:]
        self.assertTrue(allequal(recarr['t64col'][:], orig_val, numpy.float64),
                        "Stored and retrieved values do not match.")


    def test03_Compare64EArray(self):
        "Comparing written 64-bit time data with read data in an EArray."

        wtime = 1234567890.123456

        # Create test EArray with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time64 EArrays")
        ea = h5file.createEArray(
                '/', 'test', tables.Time64Atom(), shape=(0,))
        ea.append((wtime,))
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        rtime = h5file.root.test[0]
        h5file.close()
        self.assertTrue(allequal(rtime, wtime),
                        "Stored and retrieved values do not match.")


    def test03b_Compare64EArray(self):
        "Comparing several written and read 64-bit time values in an EArray."

        # Create test EArray with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time64 E arrays")
        ea = h5file.createEArray(
                '/', 'test', tables.Time64Atom(), shape=(0, 2))

        # Size of the test.
        nrows = ea.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        ##nrows = 10

        for i in xrange(nrows):
            j = i * 2
            ea.append(((j + 0.012, j + 1 + 0.012),))
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        arr = h5file.root.test.read()
        h5file.close()

        orig_val = numpy.arange(0, nrows*2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 2)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", arr
        self.assertTrue(allequal(arr, orig_val),
                        "Stored and retrieved values do not match.")



class UnalignedTestCase(common.PyTablesTestCase):
    "Tests writing and reading unaligned time values in a table."

    # The description used in the test Table.
    # Time fields are unaligned because of 'i8col'.
    class MyTimeRow(tables.IsDescription):
        i8col  = tables.Int8Col(pos = 0)
        t32col = tables.Time32Col(pos = 1)
        t64col = tables.Time64Col(shape = (2,), pos = 2)


    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
        """

        self.h5fname = tempfile.mktemp(suffix = '.h5')


    def tearDown(self):
        """tearDown() -> None

        Removes 'h5fname'.
        """

        os.remove(self.h5fname)


    def test00_CompareTable(self):
        "Comparing written unaligned time data with read data in a Table."

        # Create test Table with data.
        h5file = tables.openFile(
                self.h5fname, 'w', title = "Test for comparing Time tables")
        tbl = h5file.createTable('/', 'test', self.MyTimeRow)

        # Size of the test.
        nrows = tbl.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        ##nrows = 10

        row = tbl.row
        for i in xrange(nrows):
            row['i8col']  = i
            row['t32col'] = i
            j = i * 2
            row['t64col'] = (j+0.012, j+1+0.012)
            row.append()
        h5file.close()

        # Check the written data.
        h5file = tables.openFile(self.h5fname)
        recarr = h5file.root.test.read()
        h5file.close()

        # Int8 column.
        orig_val = numpy.arange(nrows, dtype=numpy.int8)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", recarr['i8col'][:]
        self.assertTrue(numpy.alltrue(recarr['i8col'][:] == orig_val),
                        "Stored and retrieved values do not match.")

        # Time32 column.
        orig_val = numpy.arange(nrows, dtype=numpy.int32)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", recarr['t32col'][:]
        self.assertTrue(numpy.alltrue(recarr['t32col'][:] == orig_val),
                        "Stored and retrieved values do not match.")

        # Time64 column.
        orig_val = numpy.arange(0, nrows*2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 2)
        if common.verbose:
            print "Original values:", orig_val
            print "Retrieved values:", recarr['t64col'][:]
        self.assertTrue(allequal(recarr['t64col'][:], orig_val, numpy.float64),
                        "Stored and retrieved values do not match.")



class BigEndianTestCase(common.PyTablesTestCase):
    "Tests for reading big-endian time values in arrays and nested tables."

    def setUp(self):
        filename = self._testFilename('times-nested-be.h5')
        self.h5f = tables.openFile(filename, 'r')


    def tearDown(self):
        self.h5f.close()


    def test00a_Read32Array(self):
        "Checking Time32 type in arrays."

        # Check the written data.
        earr = self.h5f.root.earr32[:]

        # Generate the expected Time32 array.
        start = 1178896298
        nrows = 10
        orig_val = numpy.arange(start, start+nrows, dtype=numpy.int32)

        if common.verbose:
            print "Retrieved values:", earr
            print "Should look like:", orig_val
        self.assertTrue(numpy.alltrue(earr == orig_val),
                        "Retrieved values do not match the expected values.")


    def test00b_Read64Array(self):
        "Checking Time64 type in arrays."

        # Check the written data.
        earr = self.h5f.root.earr64[:]

        # Generate the expected Time64 array.
        start = 1178896298.832258
        nrows = 10
        orig_val = numpy.arange(start, start+nrows, dtype=numpy.float64)

        if common.verbose:
            print "Retrieved values:", earr
            print "Should look like:", orig_val
        self.assertTrue(numpy.allclose(earr, orig_val, rtol=1.e-15),
                        "Retrieved values do not match the expected values.")


    def test01a_ReadPlainColumn(self):
        "Checking Time32 type in plain columns."

        # Check the written data.
        tbl = self.h5f.root.tbl
        t32 = tbl.cols.t32[:]

        # Generate the expected Time32 array.
        start = 1178896298
        nrows = 10
        orig_val = numpy.arange(start, start+nrows, dtype=numpy.int32)

        if common.verbose:
            print "Retrieved values:", t32
            print "Should look like:", orig_val
        self.assertTrue(numpy.alltrue(t32 == orig_val),
                        "Retrieved values do not match the expected values.")


    def test01b_ReadNestedColumn(self):
        "Checking Time64 type in nested columns."

        # Check the written data.
        tbl = self.h5f.root.tbl
        t64 = tbl.cols.nested.t64[:]

        # Generate the expected Time64 array.
        start = 1178896298.832258
        nrows = 10
        orig_val = numpy.arange(start, start+nrows, dtype=numpy.float64)

        if common.verbose:
            print "Retrieved values:", t64
            print "Should look like:", orig_val
        self.assertTrue(numpy.allclose(t64, orig_val, rtol=1.e-15),
                        "Retrieved values do not match the expected values.")


    def test02_ReadNestedColumnTwice(self):
        "Checking Time64 type in nested columns (read twice)."

        # Check the written data.
        tbl = self.h5f.root.tbl
        _ = tbl.cols.nested.t64[:]
        t64 = tbl.cols.nested.t64[:]

        # Generate the expected Time64 array.
        start = 1178896298.832258
        nrows = 10
        orig_val = numpy.arange(start, start+nrows, dtype=numpy.float64)

        if common.verbose:
            print "Retrieved values:", t64
            print "Should look like:", orig_val
        self.assertTrue(numpy.allclose(t64, orig_val, rtol=1.e-15),
                        "Retrieved values do not match the expected values.")


#----------------------------------------------------------------------

def suite():
    """suite() -> test suite

    Returns a test suite consisting of all the test cases in the module.
    """

    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(LeafCreationTestCase))
    theSuite.addTest(unittest.makeSuite(OpenTestCase))
    theSuite.addTest(unittest.makeSuite(CompareTestCase))
    theSuite.addTest(unittest.makeSuite(UnalignedTestCase))
    theSuite.addTest(unittest.makeSuite(BigEndianTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest = 'suite')



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
