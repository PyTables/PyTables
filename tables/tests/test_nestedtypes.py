"""
Test module for nested types under PyTables
===========================================

:Author:   Ivan Vilata
:Author:   Francesc Altet
:Contact:  ivilata@carabos.com
:Created:  2005-05-18
:License:  BSD
:Revision: $Id$
"""

import unittest

import numarray
import numarray.records

import common
from common import verbose
import tables as t
import tables.nestedrecords as nr
from tables.IsDescription import Description
from tables.indexes import minRowIndex



# This is the structure of the table used for testing (DON'T PANIC!):
#
# +-+---------------------------------+-----+----------+-+-+
# |x|Info                             |color|info      |y|z|
# | +-----+--+----------------+----+--+     +----+-----+ | |
# | |value|y2|Info2           |name|z2|     |Name|Value| | |
# | |     |  +----+-----+--+--+    |  |     |    |     | | |
# | |     |  |name|value|y3|z3|    |  |     |    |     | | |
# +-+-----+--+----+-----+--+--+----+--+-----+----+-----+-+-+
#
# Please note that some fields are explicitly ordered while others are
# ordered alphabetically by name.

# The declaration of the nested table:
class Info(t.IsDescription):
    _v_pos = 3
    Name = t.StringCol(length=2)
    Value = t.Complex64Col()

class TestTDescr(t.IsDescription):

    """A description that has several nested columns."""

    x = t.Int32Col(0, shape=2, pos=0) #0
    y = t.FloatCol(1, shape=(2,2))
    z = t.UInt8Col(1)
    color = t.StringCol(2, " ", pos=2)
    info = Info()
    class Info(t.IsDescription): #1
        _v_pos = 1
        name = t.StringCol(length=2)
        value = t.Complex64Col(pos=0) #0
        y2 = t.FloatCol(1, pos=1) #1
        z2 = t.UInt8Col(1)
        class Info2(t.IsDescription):
            y3 = t.Time64Col(1, shape=2)
            z3 = t.EnumCol({'r':4, 'g':2, 'b':1}, 'r', shape=2)
            name = t.StringCol(length=2)
            value = t.Complex64Col(shape=2)

# The corresponding nested array description:
testADescr = [
    ('x', '(2,)Int32'),
    ('Info', [
        ('value', 'Complex64'),
        ('y2', 'Float64'),
        ('Info2', [
            ('name', 'a2'),
            ('value', '(2,)Complex64'),
            ('y3', '(2,)Float64'),
            ('z3', '(2,)UInt32')]),
        ('name', 'a2'),
        ('z2', 'UInt8')]),
    ('color', 'a2'),
    ('info', [
        ('Name', 'a2'),
        ('Value', 'Complex64')]),
    ('y', '(2,2)Float64'),
    ('z', 'UInt8')]

# The corresponding nested array description (brief version):
testADescr2 = [
    ('x', '(2,)i4'),
    ('Info', [
        ('value', '1c16'),
        ('y2', '1f8'),
        ('Info2', [
            ('name', '1a2'),
            ('value', '(2,)c16'),
            ('y3', '(2,)f8'),
            ('z3', '(2,)u4')]),
        ('name', '1a2'),
        ('z2', '1u1')]),
    ('color', '1a2'),
    ('info', [
        ('Name', '1a2'),
        ('Value', '1c16')]),
    ('y', '(2, 2)f8'),
    ('z', '1u1')]

# A nested array for testing:
testABuffer = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    [(3,2), [6j, 6., ['nn', (6j,4j), (6.,4.), (1,2)], 'NN', 8], 'cc', ['NN', 6j], ((6.,4.),(6.,4.)), 8],
    [(4,3), [7j, 7., ['oo', (7j,5j), (7.,5.), (2,1)], 'OO', 9], 'dd', ['OO', 7j], ((7.,5.),(7.,5.)), 9],
    ]
testAData = nr.array(testABuffer, descr=testADescr)
# The name of the column to be searched:
testCondCol = 'Info/z2'
# The name of a nested column (it can not be searched):
testNestedCol = 'Info'
# The condition to be applied on the column (all but the last row match it):
testCondition = lambda col: 2 < col < 9



def areDescriptionsEqual(desc1, desc2):
    """
    Are both `desc1` and `desc2` equivalent descriptions?

    The arguments may be description objects (``IsDescription``,
    ``Description``) or dictionaries.
    """

    if isinstance(desc1, t.Col):
        # This is a rough comparison but it suffices here.
        return (desc1.stype == desc2.stype and desc2.type == desc2.type
                and desc1.shape == desc2.shape
                and desc1.itemsize == desc2.itemsize
                and desc1._v_pos == desc2._v_pos
                and desc1.indexed == desc2.indexed
                #and desc1.dflt == desc2.dflt)
                and common.areArraysEqual(desc1.dflt, desc2.dflt))

    if hasattr(desc1, '_v_colObjects'):  # quacks like a Description
        cols1 = desc1._v_colObjects
    elif hasattr(desc1, 'columns'):  # quacks like an IsDescription
        cols1 = desc1.columns
    else:  # hope it quacks like a dictionary
        cols1 = desc1

    if hasattr(desc2, '_v_colObjects'):  # quacks like a Description
        cols2 = desc2._v_colObjects
    elif hasattr(desc2, 'columns'):  # quacks like an IsDescription
        cols2 = desc2.columns
    else:  # hope it quacks like a dictionary
        cols2 = desc2

    if len(cols1) != len(cols2):
        return False

    for (colName, colobj1) in cols1.iteritems():
        colobj2 = cols2[colName]
        if colName in ('_v_indexprops', '_v_pos'):
            # The comparison may not be quite exhaustive!
            return colobj1 == colobj2
        if not areDescriptionsEqual(colobj1, colobj2):
            return False

    return True



class DescriptionTestCase(common.PyTablesTestCase):

    """Test creating nested column descriptions."""

    _TestTDescr = TestTDescr
    _testADescr = testADescr
    _testADescr2 = testADescr2
    _testAData = testAData

    def test00_instance(self):
        """Creating an instance of a nested description."""

        self._verboseHeader()
        self.assert_(
            areDescriptionsEqual(self._TestTDescr, self._TestTDescr()),
            "Table description does not match the given one.")

    def test01_instance(self):
        """Checking attrs of an instance of a nested description."""

        self._verboseHeader()
        descr = Description(self._TestTDescr().columns)
        if verbose:
            print "Generated description:", descr._v_nestedDescr
            print "Should look like:", self._testADescr2
        self.assert_(self._testADescr2 == descr._v_nestedDescr,
                     "Description._v_nestedDescr does not match.")



class CreateTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test creating a nested table and opening it."""

    _TestTDescr = TestTDescr
    _testABuffer = testABuffer
    _testAData = testAData


    def _checkColumns(self, cols, desc):
        """
        Check that `cols` has all the accessors for `self._TestTDescr`.
        """

        # ``_desc`` is a leaf column and ``cols`` a ``Column``.
        if isinstance(desc, t.Col):
            return isinstance(cols, t.Column)

        # ``_desc`` is a description object and ``cols`` a ``Cols``.
        descColumns = desc._v_colObjects
        for colName in descColumns:
            if colName not in cols._v_colnames:
                return False
            if not self._checkColumns(cols._f_col(colName),
                                      descColumns[colName]):
                return False

        return True


    def _checkDescription(self, table):
        """
        Check that description of `table` matches `self._TestTDescr`.
        """

        # Compare descriptions.
        self.assert_(
            areDescriptionsEqual(self._TestTDescr, table.description),
            "Table description does not match the given one.")
        # Check access to columns.
        self._checkColumns(table.cols, table.description)


    def test00_create(self):
        """Creating a nested table."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        self._checkDescription(tbl)


    def test01_open(self):
        """Opening a nested table."""

        self._verboseHeader()
        self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        self._reopen()
        self._checkDescription(self.h5file.root.test)


    def test02_NestedRecArrayCompat(self):
        """Creating a compatible ``NestedRecArray``."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        self.assert_(common.areArraysEqual(nrarr, self._testAData),
                     "Can not create a compatible record array.")


    def test03_NRA(self):
        """Creating a table from a NestedRecArray object."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._testAData, title=self._getMethodName())
        tbl.flush()

        readAData = tbl.read()
        self.assert_(common.areArraysEqual(self._testAData, readAData),
                     "Written and read values differ.")

    def test04_NRA2(self):
        """Creating a table from a generated NestedRecArray object."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        readAData = tbl.read()

        tbl2 = self.h5file.createTable(
            '/', 'test2', readAData, title=self._getMethodName())
        readAData2 = tbl2.read()

        self.assert_(common.areArraysEqual(self._testAData, readAData2),
                     "Written and read values differ.")


class WriteTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test writing data in a nested table."""

    _TestTDescr = TestTDescr
    _testAData = testAData
    _testCondCol = testCondCol
    _testNestedCol = testNestedCol


    def _testCondition(self, col):
        """Get a condition on `col` that matches all but the last row."""
        return testCondition(col)


    def _appendRow(self, row, index):
        """
        Append the `index`-th row in `self._testAData` to `row`.

        Values are set field-by-field (be it nested or not).
        """

        record = self._testAData[index]
        for fieldName in self._testAData._names:
            row[fieldName] = record.field(fieldName)
        row.append()


    def _appendRowRecursive(self, row, index):
        """
        Append the `index`-th row in `self._testAData` to `row`.

        Values are set atom-by-atom.
        """

        record = self._testAData[index]
        # Flatten the record if it is nested,
        # to iterate easily over atomic fields.
        # If the record is not nested, do nothing.
        if isinstance(record, nr.NestedRecord):
            record = record.asRecord()

        # Add as an ordinary flat record.
        for fieldName in self._testAData._names:
            row[fieldName] = record.field(fieldName)
        row.append()


    def test00_append(self):
        """Appending a set of rows."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        readAData = tbl.read()
        self.assert_(common.areArraysEqual(self._testAData, readAData),
                     "Written and read values differ.")


    def test01_row(self):
        """Appending individual rows."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        row = tbl.row
        # Add the first row atom by atom.
        self._appendRowRecursive(row, 0)
        # Add the rest of the rows field by field.
        for i in range(1, len(self._testAData)):
            self._appendRow(row, i)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        readAData = tbl.read()
        self.assert_(common.areArraysEqual(self._testAData, readAData),
                     "Written and read values differ.")


    def test02_where(self):
        """Searching nested data."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        searchedCoords = tbl.getWhereList(
            self._testCondition(tbl.cols._f_col(self._testCondCol)))

        # All but the last row match the condition.
        searchedCoords.sort()
        self.assertEqual(searchedCoords.tolist(),
                         range(len(self._testAData) - 1),
                         "Search returned incorrect results.")


    def test02b_whereAppend(self):
        """Searching nested data and appending it to another table."""

        self._verboseHeader()
        tbl1 = self.h5file.createTable(
            '/', 'test1', self._TestTDescr, title=self._getMethodName())
        tbl1.append(self._testAData)
        tbl1.flush()

        tbl2 = self.h5file.createTable(
            '/', 'test2', self._TestTDescr, title=self._getMethodName())
        tbl1.whereAppend(
            tbl2, self._testCondition(tbl1.cols._f_col(self._testCondCol)))

        if self._reopen():
            tbl1 = self.h5file.root.test1
            tbl2 = self.h5file.root.test2

        searchedCoords = tbl2.getWhereList(
            self._testCondition(tbl2.cols._f_col(self._testCondCol)))

        # All but the last row match the condition.
        searchedCoords.sort()
        self.assertEqual(searchedCoords.tolist(),
                         range(len(self._testAData) - 1),
                         "Search returned incorrect results.")


    def test03_colscond(self):
        """Searching on a column with nested columns."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        self.assertRaises(
            TypeError, self._testCondition,
            tbl.cols._f_col(self._testNestedCol))


    def test04_modifyColumn(self):
        """Modifying one single nested column (modifyColumn)."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        nColumn = self._testNestedCol
        # Get the nested column data and swap the first and last rows.
        raTable = self._testAData.copy()
        raColumn = raTable.field(nColumn)
        # The next will not work until NestedRecords supports copies
        (raColumn[0], raColumn[-1]) = (raColumn[-1], raColumn[0])

        # Write the resulting column and re-read the whole table.
        tbl.modifyColumn(colname=nColumn, column=raColumn)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        raReadTable = tbl.read()
        if verbose:
            print "Table read:", raReadTable
            print "Should look like:", raTable

        # Compare it to the written one.
        self.assert_(common.areArraysEqual(raTable, raReadTable),
                     "Written and read values differ.")

    def test05a_modifyColumns(self):
        """Modifying one nested column (modifyColumns)."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        nColumn = self._testNestedCol
        # Get the nested column data and swap the first and last rows.
        raTable = self._testAData.copy()
        raColumn = raTable.field(nColumn)
        (raColumn[0], raColumn[-1]) = (raColumn[-1].copy(), raColumn[0].copy())

        # Write the resulting column and re-read the whole table.
        tbl.modifyColumns(names=[nColumn], columns=raColumn)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        raReadTable = tbl.read()
        if verbose:
            print "Table read:", raReadTable
            print "Should look like:", raTable

        # Compare it to the written one.
        self.assert_(common.areArraysEqual(raTable, raReadTable),
                     "Written and read values differ.")

    def test05b_modifyColumns(self):
        """Modifying two nested columns (modifyColumns)."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        # Get the nested column data and swap the first and last rows.
        colnames = ['x', 'color']  # Get the first two columns
        raCols = nr.fromarrays([self._testAData.copy().field('x'),
                                self._testAData.copy().field('color')],
                               descr=[('x','(2,)i4'), ('color', '1a2')])
                               #descr=tbl.description._v_nestedDescr[0:2])
                               # or...
                               # names=tbl.description._v_nestedNames[0:2],
                               # formats=tbl.description._v_nestedFormats[0:2])
        (raCols[0], raCols[-1]) = (raCols[-1].copy(), raCols[0].copy())

        # Write the resulting columns
        tbl.modifyColumns(names=colnames, columns=raCols)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        # Re-read the appropriate columns

        # ivb (2005-09-13): The ``[:]`` here would not be necessary
        # if ``numarray.strings.array()`` was more lax when checking
        # for sequences in its ``buffer`` argument,
        # just as ``numarray.array()`` does.  See SF bug #1286168.

        raCols2 = nr.fromarrays([tbl.cols._f_col('x'),
                                 tbl.cols._f_col('color')[:]],
                                descr=raCols.descr)
        if verbose:
            print "Table read:", raCols2
            print "Should look like:", raCols

        # Compare it to the written one.
        self.assert_(common.areArraysEqual(raCols, raCols2),
                     "Written and read values differ.")

    def test06_modifyRows(self):
        "Checking modifying several rows at once (using nestedrecarray)"

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)
        tbl.flush()

        # Get the nested record and swap the first and last rows.
        raTable = self._testAData.copy()
        (raTable[0], raTable[-1]) = (raTable[-1].copy(), raTable[0].copy())

        # Write the resulting nested record and re-read the whole table.
        tbl.modifyRows(start=0, stop=2, rows=raTable)
        tbl.flush()

        if self._reopen():
            tbl = self.h5file.root.test

        raReadTable = tbl.read()
        if verbose:
            print "Table read:", raReadTable
            print "Should look like:", raTable

        # Compare it to the written one.
        self.assert_(common.areArraysEqual(raTable, raReadTable),
                     "Written and read values differ.")

    def test07_index(self):
        """Checking indexes of nested columns"""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName(),
            expectedrows = minRowIndex*2)
        for i in range(minRowIndex):
            tbl.append(self._testAData)
        tbl.flush()
        coltoindex = tbl.cols._f_col(self._testCondCol)
        indexrows = coltoindex.createIndex(testmode=1)

        if self._reopen():
            tbl = self.h5file.root.test
            coltoindex = tbl.cols._f_col(self._testCondCol)

        if verbose:
            print "Number of written rows:", tbl.nrows
            print "Number of indexed rows:", coltoindex.index.nelements

        # Check indexing flags:
        self.assert_(tbl.indexed == True, "Table not indexed")
        self.assert_(coltoindex.index <> None, "Column not indexed")
        self.assert_(tbl.colindexed[self._testCondCol], "Column not indexed")
        # Do a look-up for values
        searchedCoords = tbl.getWhereList(
            self._testCondition(tbl.cols._f_col(self._testCondCol)))

        if verbose:
            print "Searched coords:", searchedCoords

        # All even rows match the condition.
        searchedCoords.sort()
        self.assertEqual(searchedCoords.tolist(), range(0, minRowIndex*2, 2),
                         "Search returned incorrect results.")


class WriteNoReopen(WriteTestCase):
    def _reopen(self, mode='r'):
        return False

class WriteReopen(WriteTestCase):
    pass  # default ``_reopen()`` does reopen the file


class ReadTestCase(common.TempFileMixin, common.PyTablesTestCase):
    """Checking the Table.Cols accessor."""

    _TestTDescr = TestTDescr
    _testABuffer = testABuffer
    _testAData = testAData
    _testNestedCol = testNestedCol


    def test01_read(self):
        """Checking Table.read with subgroups with a range index with step."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.read(start=0, step=2, field='Info')
        nrarrcols = nrarr.field('Info')[0::2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test02_read(self):
        """Checking Table.read with a nested Column."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        tblcols = tbl.read(start=0, step=2, field='Info/value')
        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        nrarrcols = nrarr.field('Info/value')[0::2]
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

class ReadNoReopen(ReadTestCase):
    def _reopen(self, mode='r'):
        return False

class ReadReopen(ReadTestCase):
    pass  # default ``_reopen()`` does reopen the file


class ColsTestCase(common.TempFileMixin, common.PyTablesTestCase):
    """Checking the Table.Cols accessor."""

    _TestTDescr = TestTDescr
    _testABuffer = testABuffer
    _testAData = testAData
    _testNestedCol = testNestedCol


    def test01a_f_col(self):
        """Checking cols._f_col() with a subgroup."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        if self._reopen():
            tbl = self.h5file.root.test

        tblcol = tbl.cols._f_col(self._testNestedCol)
        if verbose:
            print "Column group name:", tblcol._v_desc._v_pathname
        self.assert_(tblcol._v_desc._v_pathname == self._testNestedCol,
                     "Column group name doesn't match.")

    def test01b_f_col(self):
        """Checking cols._f_col() with a column."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        if self._reopen():
            tbl = self.h5file.root.test

        tblcol = tbl.cols._f_col(self._testNestedCol+"/name")
        if verbose:
            print "Column name:", tblcol.name
        self.assert_(tblcol.name == "name",
                     "Column name doesn't match.")

    def test01c_f_col(self):
        """Checking cols._f_col() with a nested subgroup."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        tblcol = tbl.cols._f_col(self._testNestedCol+"/Info2")
        if verbose:
            print "Column group name:", tblcol._v_desc._v_pathname
        self.assert_(tblcol._v_desc._v_pathname == self._testNestedCol+"/Info2",
                     "Column group name doesn't match.")

    def test02a__len__(self):
        """Checking cols.__len__() in root level."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        if self._reopen():
            tbl = self.h5file.root.test

        length = len(tbl.cols)
        if verbose:
            print "Column group length:", length
        self.assert_(length == 6,
                     "Column group length doesn't match.")

    def test02b__len__(self):
        """Checking cols.__len__() in subgroup level."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())

        if self._reopen():
            tbl = self.h5file.root.test

        length = len(tbl.cols.Info)
        if verbose:
            print "Column group length:", length
        self.assert_(length == 5,
                     "Column group length doesn't match.")

    def test03a__getitem__(self):
        """Checking cols.__getitem__() with a single index."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols[1]
        nrarrcols = nrarr[1]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test03b__getitem__(self):
        """Checking cols.__getitem__() with a range index."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols[0:2]
        nrarrcols = nrarr[0:2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test03c__getitem__(self):
        """Checking cols.__getitem__() with a range index with step."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols[0::2]
        nrarrcols = nrarr[0::2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")


    def test04a__getitem__(self):
        """Checking cols.__getitem__() with subgroups with a single index."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols._f_col('Info')[1]
        nrarrcols = nrarr.field('Info')[1]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test04b__getitem__(self):
        """Checking cols.__getitem__() with subgroups with a range index."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols._f_col('Info')[0:2]
        nrarrcols = nrarr.field('Info')[0:2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test04c__getitem__(self):
        """Checking cols.__getitem__() with subgroups with a range index with step."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols._f_col('Info')[0::2]
        nrarrcols = nrarr.field('Info')[0::2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test05a__getitem__(self):
        """Checking cols.__getitem__() with a column with a single index."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols._f_col('Info/value')[1]
        nrarrcols = nrarr.field('Info/value')[1]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(nrarrcols == tblcols,
                     "Original array are retrieved doesn't match.")

    def test05b__getitem__(self):
        """Checking cols.__getitem__() with a column with a range index."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols._f_col('Info/value')[0:2]
        nrarrcols = nrarr.field('Info/value')[0:2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")

    def test05c__getitem__(self):
        """Checking cols.__getitem__() with a column with a range index with step."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._TestTDescr, title=self._getMethodName())
        tbl.append(self._testAData)

        if self._reopen():
            tbl = self.h5file.root.test

        nrarr = nr.array(testABuffer,
                         names=tbl.description._v_nestedNames,
                         formats=tbl.description._v_nestedFormats)
        tblcols = tbl.cols._f_col('Info/value')[0::2]
        nrarrcols = nrarr.field('Info/value')[0::2]
        if verbose:
            print "Read cols:", tblcols
            print "Should look like:", nrarrcols
        self.assert_(common.areArraysEqual(nrarrcols, tblcols),
                     "Original array are retrieved doesn't match.")


class ColsNoReopen(ColsTestCase):
    def _reopen(self, mode='r'):
        return False

class ColsReopen(ColsTestCase):
    pass  # default ``_reopen()`` does reopen the file



#----------------------------------------------------------------------

def suite():
    """Return a test suite consisting of all the test cases in the module."""

    theSuite = unittest.TestSuite()
    niter = 1
    #heavy = 1  # uncomment this only for testing purposes

    #theSuite.addTest(unittest.makeSuite(DescriptionTestCase))
    #theSuite.addTest(unittest.makeSuite(WriteReopen))
    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(DescriptionTestCase))
        theSuite.addTest(unittest.makeSuite(CreateTestCase))
        theSuite.addTest(unittest.makeSuite(WriteNoReopen))
        theSuite.addTest(unittest.makeSuite(WriteReopen))
        theSuite.addTest(unittest.makeSuite(ColsNoReopen))
        theSuite.addTest(unittest.makeSuite(ColsReopen))
        theSuite.addTest(unittest.makeSuite(ReadNoReopen))
        theSuite.addTest(unittest.makeSuite(ReadReopen))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
