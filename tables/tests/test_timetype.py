# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: December 15, 2004
# Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
#
# $Id$
#
########################################################################

"""Unit test for the Time datatypes."""


import numpy

import tables
from tables.tests import common
from tables.tests.common import allequal
from tables.tests.common import unittest, test_filename
from tables.tests.common import PyTablesTestCase as TestCase


class LeafCreationTestCase(common.TempFileMixin, TestCase):
    """Tests creating Tables, VLArrays an EArrays with Time data."""

    def test00_UnidimLeaves(self):
        """Creating new nodes with unidimensional time elements."""

        # Table creation.
        class MyTimeRow(tables.IsDescription):
            intcol = tables.IntCol()
            t32col = tables.Time32Col()
            t64col = tables.Time64Col()

        self.h5file.create_table('/', 'table', MyTimeRow)

        # VLArray creation.
        self.h5file.create_vlarray('/', 'vlarray4', tables.Time32Atom())
        self.h5file.create_vlarray('/', 'vlarray8', tables.Time64Atom())

        # EArray creation.
        self.h5file.create_earray('/', 'earray4',
                                  tables.Time32Atom(), shape=(0,))
        self.h5file.create_earray('/', 'earray8',
                                  tables.Time64Atom(), shape=(0,))

    def test01_MultidimLeaves(self):
        """Creating new nodes with multidimensional time elements."""

        # Table creation.
        class MyTimeRow(tables.IsDescription):
            intcol = tables.IntCol(shape=(2, 1))
            t32col = tables.Time32Col(shape=(2, 1))
            t64col = tables.Time64Col(shape=(2, 1))
        self.h5file.create_table('/', 'table', MyTimeRow)

        # VLArray creation.
        self.h5file.create_vlarray(
            '/', 'vlarray4', tables.Time32Atom(shape=(2, 1)))
        self.h5file.create_vlarray(
            '/', 'vlarray8', tables.Time64Atom(shape=(2, 1)))

        # EArray creation.
        self.h5file.create_earray(
            '/', 'earray4', tables.Time32Atom(), shape=(0, 2, 1))
        self.h5file.create_earray(
            '/', 'earray8', tables.Time64Atom(), shape=(0, 2, 1))


class OpenTestCase(common.TempFileMixin, TestCase):
    """Tests opening a file with Time nodes."""

    # The description used in the test Table.
    class MyTimeRow(tables.IsDescription):
        t32col = tables.Time32Col(shape=(2, 1))
        t64col = tables.Time64Col(shape=(2, 1))

    # The atoms used in the test VLArrays.
    myTime32Atom = tables.Time32Atom(shape=(2, 1))
    myTime64Atom = tables.Time64Atom(shape=(2, 1))

    def setUp(self):
        super(OpenTestCase, self).setUp()

        # Create test Table.
        self.h5file.create_table('/', 'table', self.MyTimeRow)

        # Create test VLArrays.
        self.h5file.create_vlarray('/', 'vlarray4', self.myTime32Atom)
        self.h5file.create_vlarray('/', 'vlarray8', self.myTime64Atom)

        self._reopen()

    def test00_OpenFile(self):
        """Opening a file with Time nodes."""

        # Test the Table node.
        tbl = self.h5file.root.table
        self.assertEqual(
            tbl.coldtypes['t32col'],
            self.MyTimeRow.columns['t32col'].dtype,
            "Column dtypes do not match.")
        self.assertEqual(
            tbl.coldtypes['t64col'],
            self.MyTimeRow.columns['t64col'].dtype,
            "Column dtypes do not match.")

        # Test the VLArray nodes.
        vla4 = self.h5file.root.vlarray4
        self.assertEqual(
            vla4.atom.dtype, self.myTime32Atom.dtype,
            "Atom types do not match.")
        self.assertEqual(
            vla4.atom.shape, self.myTime32Atom.shape,
            "Atom shapes do not match.")

        vla8 = self.h5file.root.vlarray8
        self.assertEqual(
            vla8.atom.dtype, self.myTime64Atom.dtype,
            "Atom types do not match.")
        self.assertEqual(
            vla8.atom.shape, self.myTime64Atom.shape,
            "Atom shapes do not match.")

    def test01_OpenFileStype(self):
        """Opening a file with Time nodes, comparing Atom.stype."""

        # Test the Table node.
        tbl = self.h5file.root.table
        self.assertEqual(
            tbl.coltypes['t32col'],
            self.MyTimeRow.columns['t32col'].type,
            "Column types do not match.")
        self.assertEqual(
            tbl.coltypes['t64col'],
            self.MyTimeRow.columns['t64col'].type,
            "Column types do not match.")

        # Test the VLArray nodes.
        vla4 = self.h5file.root.vlarray4
        self.assertEqual(
            vla4.atom.type, self.myTime32Atom.type,
            "Atom types do not match.")

        vla8 = self.h5file.root.vlarray8
        self.assertEqual(
            vla8.atom.type, self.myTime64Atom.type,
            "Atom types do not match.")


class CompareTestCase(common.TempFileMixin, TestCase):
    """Tests whether stored and retrieved time data is kept the same."""

    # The description used in the test Table.
    class MyTimeRow(tables.IsDescription):
        t32col = tables.Time32Col(pos=0)
        t64col = tables.Time64Col(shape=(2,), pos = 1)

    # The atoms used in the test VLArrays.
    myTime32Atom = tables.Time32Atom(shape=(2,))
    myTime64Atom = tables.Time64Atom(shape=(2,))

    def test00_Compare32VLArray(self):
        """Comparing written 32-bit time data with read data in a VLArray."""

        wtime = numpy.array((1234567890,) * 2, numpy.int32)

        # Create test VLArray with data.
        vla = self.h5file.create_vlarray('/', 'test', self.myTime32Atom)
        vla.append(wtime)
        self._reopen()

        # Check the written data.
        rtime = self.h5file.root.test.read()[0][0]
        self.h5file.close()
        self.assertTrue(allequal(rtime, wtime),
                        "Stored and retrieved values do not match.")

    def test01_Compare64VLArray(self):
        """Comparing written 64-bit time data with read data in a VLArray."""

        wtime = numpy.array((1234567890.123456,) * 2, numpy.float64)

        # Create test VLArray with data.
        vla = self.h5file.create_vlarray('/', 'test', self.myTime64Atom)
        vla.append(wtime)
        self._reopen()

        # Check the written data.
        rtime = self.h5file.root.test.read()[0][0]
        self.h5file.close()
        self.assertTrue(allequal(rtime, wtime),
                        "Stored and retrieved values do not match.")

    def test01b_Compare64VLArray(self):
        """Comparing several written and read 64-bit time values in a
        VLArray."""

        # Create test VLArray with data.
        vla = self.h5file.create_vlarray('/', 'test', self.myTime64Atom)

        # Size of the test.
        nrows = vla.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        # nrows = 10

        for i in range(nrows):
            j = i * 2
            vla.append((j + 0.012, j + 1 + 0.012))
        self._reopen()

        # Check the written data.
        arr = self.h5file.root.test.read()
        self.h5file.close()

        arr = numpy.array(arr)
        orig_val = numpy.arange(0, nrows * 2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 1, 2)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", arr)
        self.assertTrue(allequal(arr, orig_val),
                        "Stored and retrieved values do not match.")

    def test02_CompareTable(self):
        """Comparing written time data with read data in a Table."""

        wtime = 1234567890.123456

        # Create test Table with data.
        tbl = self.h5file.create_table('/', 'test', self.MyTimeRow)
        row = tbl.row
        row['t32col'] = int(wtime)
        row['t64col'] = (wtime, wtime)
        row.append()
        self._reopen()

        # Check the written data.
        recarr = self.h5file.root.test.read(0)
        self.h5file.close()

        self.assertEqual(recarr['t32col'][0], int(wtime),
                         "Stored and retrieved values do not match.")

        comp = (recarr['t64col'][0] == numpy.array((wtime, wtime)))
        self.assertTrue(numpy.alltrue(comp),
                        "Stored and retrieved values do not match.")

    def test02b_CompareTable(self):
        """Comparing several written and read time values in a Table."""

        # Create test Table with data.
        tbl = self.h5file.create_table('/', 'test', self.MyTimeRow)

        # Size of the test.
        nrows = tbl.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        # nrows = 10

        row = tbl.row
        for i in range(nrows):
            row['t32col'] = i
            j = i * 2
            row['t64col'] = (j + 0.012, j+1+0.012)
            row.append()

        self._reopen()

        # Check the written data.
        recarr = self.h5file.root.test.read()
        self.h5file.close()

        # Time32 column.
        orig_val = numpy.arange(nrows, dtype=numpy.int32)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", recarr['t32col'][:])
        self.assertTrue(numpy.alltrue(recarr['t32col'][:] == orig_val),
                        "Stored and retrieved values do not match.")

        # Time64 column.
        orig_val = numpy.arange(0, nrows * 2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 2)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", recarr['t64col'][:])
        self.assertTrue(allequal(recarr['t64col'][:], orig_val, numpy.float64),
                        "Stored and retrieved values do not match.")

    def test03_Compare64EArray(self):
        """Comparing written 64-bit time data with read data in an EArray."""

        wtime = 1234567890.123456

        # Create test EArray with data.
        ea = self.h5file.create_earray(
            '/', 'test', tables.Time64Atom(), shape=(0,))
        ea.append((wtime,))
        self._reopen()

        # Check the written data.
        rtime = self.h5file.root.test[0]
        self.h5file.close()
        self.assertTrue(allequal(rtime, wtime),
                        "Stored and retrieved values do not match.")

    def test03b_Compare64EArray(self):
        """Comparing several written and read 64-bit time values in an
        EArray."""

        # Create test EArray with data.
        ea = self.h5file.create_earray('/', 'test', tables.Time64Atom(),
                                       shape=(0, 2))

        # Size of the test.
        nrows = ea.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        # nrows = 10

        for i in range(nrows):
            j = i * 2
            ea.append(((j + 0.012, j + 1 + 0.012),))
        self._reopen()

        # Check the written data.
        arr = self.h5file.root.test.read()
        self.h5file.close()

        orig_val = numpy.arange(0, nrows * 2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 2)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", arr)
        self.assertTrue(allequal(arr, orig_val),
                        "Stored and retrieved values do not match.")


class UnalignedTestCase(common.TempFileMixin, TestCase):
    """Tests writing and reading unaligned time values in a table."""

    # The description used in the test Table.
    # Time fields are unaligned because of 'i8col'.
    class MyTimeRow(tables.IsDescription):
        i8col = tables.Int8Col(pos=0)
        t32col = tables.Time32Col(pos=1)
        t64col = tables.Time64Col(shape=(2,), pos = 2)

    def test00_CompareTable(self):
        """Comparing written unaligned time data with read data in a Table."""

        # Create test Table with data.
        tbl = self.h5file.create_table('/', 'test', self.MyTimeRow)

        # Size of the test.
        nrows = tbl.nrowsinbuf + 34  # Add some more rows than buffer.
        # Only for home checks; the value above should check better
        # the I/O with multiple buffers.
        # nrows = 10

        row = tbl.row
        for i in range(nrows):
            row['i8col'] = i
            row['t32col'] = i
            j = i * 2
            row['t64col'] = (j + 0.012, j+1+0.012)
            row.append()

        self._reopen()

        # Check the written data.
        recarr = self.h5file.root.test.read()
        self.h5file.close()

        # Int8 column.
        orig_val = numpy.arange(nrows, dtype=numpy.int8)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", recarr['i8col'][:])
        self.assertTrue(numpy.alltrue(recarr['i8col'][:] == orig_val),
                        "Stored and retrieved values do not match.")

        # Time32 column.
        orig_val = numpy.arange(nrows, dtype=numpy.int32)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", recarr['t32col'][:])
        self.assertTrue(numpy.alltrue(recarr['t32col'][:] == orig_val),
                        "Stored and retrieved values do not match.")

        # Time64 column.
        orig_val = numpy.arange(0, nrows * 2, dtype=numpy.int32) + 0.012
        orig_val.shape = (nrows, 2)
        if common.verbose:
            print("Original values:", orig_val)
            print("Retrieved values:", recarr['t64col'][:])
        self.assertTrue(allequal(recarr['t64col'][:], orig_val, numpy.float64),
                        "Stored and retrieved values do not match.")


class BigEndianTestCase(TestCase):
    """Tests for reading big-endian time values in arrays and nested tables."""

    def setUp(self):
        super(BigEndianTestCase, self).setUp()
        filename = test_filename('times-nested-be.h5')
        self.h5file = tables.open_file(filename, 'r')

    def tearDown(self):
        self.h5file.close()
        super(BigEndianTestCase, self).tearDown()

    def test00a_Read32Array(self):
        """Checking Time32 type in arrays."""

        # Check the written data.
        earr = self.h5file.root.earr32[:]

        # Generate the expected Time32 array.
        start = 1178896298
        nrows = 10
        orig_val = numpy.arange(start, start + nrows, dtype=numpy.int32)

        if common.verbose:
            print("Retrieved values:", earr)
            print("Should look like:", orig_val)
        self.assertTrue(numpy.alltrue(earr == orig_val),
                        "Retrieved values do not match the expected values.")

    def test00b_Read64Array(self):
        """Checking Time64 type in arrays."""

        # Check the written data.
        earr = self.h5file.root.earr64[:]

        # Generate the expected Time64 array.
        start = 1178896298.832258
        nrows = 10
        orig_val = numpy.arange(start, start + nrows, dtype=numpy.float64)

        if common.verbose:
            print("Retrieved values:", earr)
            print("Should look like:", orig_val)
        self.assertTrue(numpy.allclose(earr, orig_val, rtol=1.e-15),
                        "Retrieved values do not match the expected values.")

    def test01a_ReadPlainColumn(self):
        """Checking Time32 type in plain columns."""

        # Check the written data.
        tbl = self.h5file.root.tbl
        t32 = tbl.cols.t32[:]

        # Generate the expected Time32 array.
        start = 1178896298
        nrows = 10
        orig_val = numpy.arange(start, start + nrows, dtype=numpy.int32)

        if common.verbose:
            print("Retrieved values:", t32)
            print("Should look like:", orig_val)
        self.assertTrue(numpy.alltrue(t32 == orig_val),
                        "Retrieved values do not match the expected values.")

    def test01b_ReadNestedColumn(self):
        """Checking Time64 type in nested columns."""

        # Check the written data.
        tbl = self.h5file.root.tbl
        t64 = tbl.cols.nested.t64[:]

        # Generate the expected Time64 array.
        start = 1178896298.832258
        nrows = 10
        orig_val = numpy.arange(start, start + nrows, dtype=numpy.float64)

        if common.verbose:
            print("Retrieved values:", t64)
            print("Should look like:", orig_val)
        self.assertTrue(numpy.allclose(t64, orig_val, rtol=1.e-15),
                        "Retrieved values do not match the expected values.")

    def test02_ReadNestedColumnTwice(self):
        """Checking Time64 type in nested columns (read twice)."""

        # Check the written data.
        tbl = self.h5file.root.tbl
        dummy = tbl.cols.nested.t64[:]
        self.assertIsNotNone(dummy)
        t64 = tbl.cols.nested.t64[:]

        # Generate the expected Time64 array.
        start = 1178896298.832258
        nrows = 10
        orig_val = numpy.arange(start, start + nrows, dtype=numpy.float64)

        if common.verbose:
            print("Retrieved values:", t64)
            print("Should look like:", orig_val)
        self.assertTrue(numpy.allclose(t64, orig_val, rtol=1.e-15),
                        "Retrieved values do not match the expected values.")


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
    import sys
    common.parse_argv(sys.argv)
    common.print_versions()
    unittest.main(defaultTest='suite')


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
