# -*- coding: utf-8 -*-

import sys
import unittest
import os
import tempfile
import warnings

import numpy as np
from numpy import rec as records
from numpy import testing as npt

from tables import *
from tables.utils import SizeType, byteorders
from tables.tests import common
from tables.tests.common import allequal, areArraysEqual
from tables.description import descr_from_dtype

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

# Test Record class


class Record(IsDescription):
    var1 = StringCol(itemsize=4, dflt=b"abcd", pos=0)  # 4-character String
    var2 = IntCol(dflt=1, pos=1)                   # integer
    var3 = Int16Col(dflt=2, pos=2)                 # short integer
    var4 = Float64Col(dflt=3.1, pos=3)             # double (double-precision)
    var5 = Float32Col(dflt=4.2, pos=4)             # float  (single-precision)
    var6 = UInt16Col(dflt=5, pos=5)                # unsigned short integer
    var7 = StringCol(itemsize=1, dflt=b"e", pos=6)  # 1-character String
    var8 = BoolCol(dflt=True, pos=7)               # boolean
    var9 = ComplexCol(itemsize=8, dflt=(
        0.+1.j), pos=8)  # Complex single precision
    var10 = ComplexCol(itemsize=16, dflt=(
        1.-0.j), pos=9)  # Complex double precision
    if 'Float16Col' in globals():
        var11 = Float16Col(dflt=6.4)               # float  (half-precision)
    if 'Float96Col' in globals():
        var12 = Float96Col(
            dflt=6.4)               # float  (extended precision)
    if 'Float128Col' in globals():
        var13 = Float128Col(
            dflt=6.4)              # float  (extended precision)
    if 'Complex192Col' in globals():
        var14 = ComplexCol(itemsize=24, dflt=(
            1.-0.j))  # Complex double (extended precision)
    if 'Complex256Col' in globals():
        var15 = ComplexCol(itemsize=32, dflt=(
            1.-0.j))  # Complex double (extended precision)

#  Dictionary definition
RecordDescriptionDict = {
    'var1': StringCol(itemsize=4, dflt=b"abcd", pos=0),  # 4-character String
    'var2': IntCol(dflt=1, pos=1),              # integer
    'var3': Int16Col(dflt=2, pos=2),            # short integer
    'var4': Float64Col(dflt=3.1, pos=3),        # double (double-precision)
    'var5': Float32Col(dflt=4.2, pos=4),        # float  (single-precision)
    'var6': UInt16Col(dflt=5, pos=5),           # unsigned short integer
    'var7': StringCol(itemsize=1, dflt=b"e", pos=6),  # 1-character String
    'var8': BoolCol(dflt=True, pos=7),          # boolean
    'var9': ComplexCol(itemsize=8, dflt=(0.+1.j), pos=8),   # Complex single precision
    'var10': ComplexCol(itemsize=16, dflt=(1.-0.j), pos=9),  # Complex double precision
}

if 'Float16Col' in globals():
    RecordDescriptionDict['var11'] = Float16Col(
        dflt=6.4)    # float  (half-precision)
if 'Float96Col' in globals():
    RecordDescriptionDict['var12'] = Float96Col(
        dflt=6.4)    # float  (extended precision)
if 'Float128Col' in globals():
    RecordDescriptionDict['var13'] = Float128Col(
        dflt=6.4)   # float  (extended precision)
if 'Complex192Col' in globals():
    RecordDescriptionDict['var14'] = ComplexCol(itemsize=24, dflt=(
        1.-0.j))  # Complex double (extended precision)
if 'Complex256Col' in globals():
    RecordDescriptionDict['var15'] = ComplexCol(itemsize=32, dflt=(
        1.-0.j))  # Complex double (extended precision)


# Old fashion of defining tables (for testing backward compatibility)
class OldRecord(IsDescription):
    var1 = StringCol(itemsize=4, dflt=b"abcd", pos=0)
    var2 = Col.from_type("int32", (), 1, pos=1)
    var3 = Col.from_type("int16", (), 2, pos=2)
    var4 = Col.from_type("float64", (), 3.1, pos=3)
    var5 = Col.from_type("float32", (), 4.2, pos=4)
    var6 = Col.from_type("uint16", (), 5, pos=5)
    var7 = StringCol(itemsize=1, dflt=b"e", pos=6)
    var8 = Col.from_type("bool", shape=(), dflt=1, pos=7)
    var9 = ComplexCol(itemsize=8, shape=(), dflt=(0.+1.j), pos=8)
    var10 = ComplexCol(itemsize=16, shape=(), dflt=(1.-0.j), pos = 9)
    if 'Float16Col' in globals():
        var11 = Col.from_type("float16", (), 6.4)
    if 'Float96Col' in globals():
        var12 = Col.from_type("float96", (), 6.4)
    if 'Float128Col' in globals():
        var13 = Col.from_type("float128", (), 6.4)
    if 'Complex192Col' in globals():
        var14 = ComplexCol(itemsize=24, shape=(), dflt=(1.-0.j))
    if 'Complex256Col' in globals():
        var15 = ComplexCol(itemsize=32, shape=(), dflt=(1.-0.j))


class BasicTestCase(common.PyTablesTestCase):
    # file  = "test.h5"
    mode = "w"
    title = "This is the table title"
    expectedrows = 100
    appendrows = 20
    compress = 0
    shuffle = 0
    fletcher32 = 0
    complib = "zlib"  # Default compression library
    record = Record
    recarrayinit = 0
    maxshort = 1 << 15

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def initRecArray(self):
        record = self.recordtemplate
        row = record[0]
        buflist = []
        # Fill the recarray
        for i in xrange(self.expectedrows):
            tmplist = []
            var1 = '%04d' % (self.expectedrows - i)
            tmplist.append(var1)
            var2 = i
            tmplist.append(var2)
            var3 = i % self.maxshort
            tmplist.append(var3)
            if isinstance(row['var4'], np.ndarray):
                tmplist.append([float(i), float(i * i)])
            else:
                tmplist.append(float(i))
            if isinstance(row['var5'], np.ndarray):
                tmplist.append(np.array((float(i),)*4))
            else:
                tmplist.append(float(i))
            # var6 will be like var3 but byteswaped
            tmplist.append(((var3 >> 8) & 0xff) + ((var3 << 8) & 0xff00))
            var7 = var1[-1]
            tmplist.append(var7)
            if isinstance(row['var8'], np.ndarray):
                tmplist.append([0, 10])  # should be equivalent to [0,1]
            else:
                tmplist.append(10)  # should be equivalent to 1
            if isinstance(row['var9'], np.ndarray):
                tmplist.append([0.+float(i)*1j, float(i)+0.j])
            else:
                tmplist.append(float(i)+0j)
            if isinstance(row['var10'], np.ndarray):
                tmplist.append([float(i)+0j, 1 + float(i)*1j])
            else:
                tmplist.append(1 + float(i)*1j)
            if 'Float16Col' in globals():
                if isinstance(row['var11'], np.ndarray):
                    tmplist.append(np.array((float(i),)*4))
                else:
                    tmplist.append(float(i))
            if 'Float96Col' in globals():
                if isinstance(row['var12'], np.ndarray):
                    tmplist.append(np.array((float(i),)*4))
                else:
                    tmplist.append(float(i))
            if 'Float128Col' in globals():
                if isinstance(row['var13'], np.ndarray):
                    tmplist.append(np.array((float(i),)*4))
                else:
                    tmplist.append(float(i))
            if 'Complex192Col' in globals():
                if isinstance(row['var14'], np.ndarray):
                    tmplist.append([float(i)+0j, 1 + float(i)*1j])
                else:
                    tmplist.append(1 + float(i)*1j)
            if 'Complex256Col' in globals():
                if isinstance(row['var15'], np.ndarray):
                    tmplist.append([float(i)+0j, 1 + float(i)*1j])
                else:
                    tmplist.append(1 + float(i)*1j)

            buflist.append(tmplist)

        self.record = records.array(buflist, dtype=record.dtype,
                                    shape=self.expectedrows)

    def populateFile(self):
        group = self.rootgroup
        if self.recarrayinit:
            # Initialize an starting buffer, if any
            self.initRecArray()
        for j in range(3):
            # Create a table
            filterprops = Filters(complevel=self.compress,
                                  shuffle=self.shuffle,
                                  fletcher32=self.fletcher32,
                                  complib=self.complib)
            if j < 2:
                byteorder = sys.byteorder
            else:
                # table2 will be byteswapped
                byteorder = {"little": "big", "big": "little"}[sys.byteorder]
            table = self.fileh.create_table(group, 'table'+str(j), self.record,
                                            title=self.title,
                                            filters=filterprops,
                                            expectedrows=self.expectedrows,
                                            byteorder=byteorder)
            if not self.recarrayinit:
                # Get the row object associated with the new table
                row = table.row
                # Fill the table
                for i in xrange(self.expectedrows):
                    s = '%04d' % (self.expectedrows - i)
                    row['var1'] = s.encode('ascii')
                    row['var7'] = s[-1].encode('ascii')
                    # row['var7'] = ('%04d' % (self.expectedrows - i))[-1]
                    row['var2'] = i
                    row['var3'] = i % self.maxshort
                    if isinstance(row['var4'], np.ndarray):
                        row['var4'] = [float(i), float(i * i)]
                    else:
                        row['var4'] = float(i)
                    if isinstance(row['var8'], np.ndarray):
                        row['var8'] = [0, 1]
                    else:
                        row['var8'] = 1
                    if isinstance(row['var9'], np.ndarray):
                        row['var9'] = [0.+float(i)*1j, float(i)+0.j]
                    else:
                        row['var9'] = float(i)+0.j
                    if isinstance(row['var10'], np.ndarray):
                        row['var10'] = [float(i)+0.j, 1.+float(i)*1j]
                    else:
                        row['var10'] = 1.+float(i)*1j
                    if isinstance(row['var5'], np.ndarray):
                        row['var5'] = np.array((float(i),)*4)
                    else:
                        row['var5'] = float(i)
                    if 'Float16Col' in globals():
                        if isinstance(row['var11'], np.ndarray):
                            row['var11'] = np.array((float(i),)*4)
                        else:
                            row['var11'] = float(i)
                    if 'Float96Col' in globals():
                        if isinstance(row['var12'], np.ndarray):
                            row['var12'] = np.array((float(i),)*4)
                        else:
                            row['var12'] = float(i)
                    if 'Float128Col' in globals():
                        if isinstance(row['var13'], np.ndarray):
                            row['var13'] = np.array((float(i),)*4)
                        else:
                            row['var13'] = float(i)
                    if 'Complex192Col' in globals():
                        if isinstance(row['var14'], np.ndarray):
                            row['var14'] = [float(i)+0j, 1 + float(i)*1j]
                        else:
                            row['var14'] = 1 + float(i)*1j
                    if 'Complex256Col' in globals():
                        if isinstance(row['var15'], np.ndarray):
                            row['var15'] = [float(i)+0j, 1 + float(i)*1j]
                        else:
                            row['var15'] = 1 + float(i)*1j

                    # var6 will be like var3 but byteswaped
                    row['var6'] = (((row['var3'] >> 8) & 0xff) +
                                   ((row['var3'] << 8) & 0xff00))
                    # print("Saving -->", row)
                    row.append()

            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.create_group(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test00_description(self):
        """Checking table description and descriptive fields"""

        self.fileh = open_file(self.file)

        tbl = self.fileh.get_node('/table0')
        desc = tbl.description

        if isinstance(self.record, dict):
            columns = self.record
        elif isinstance(self.record, np.ndarray):
            descr, _ = descr_from_dtype(self.record.dtype)
            columns = descr._v_colobjects
        elif isinstance(self.record, np.dtype):
            descr, _ = descr_from_dtype(self.record)
            columns = descr._v_colobjects
        else:
            # This is an ordinary description.
            columns = self.record.columns

        # Check table and description attributes at the same time.
        # These checks are only valid for non-nested tables.

        # Column names.
        fix_n_column = 10
        expectedNames = ['var%d' % n for n in range(1, fix_n_column + 1)]
        types = ("float16", "float96", "float128", "complex192", "complex256")
        for n, typename in enumerate(types, fix_n_column + 1):
            name = typename.capitalize() + 'Col'
            if name in globals():
                expectedNames.append('var%d' % n)

        self.assertEqual(expectedNames, list(tbl.colnames))
        self.assertEqual(expectedNames, list(desc._v_names))

        # Column instances.
        for colname in expectedNames:
            self.assertTrue(tbl.colinstances[colname]
                            is tbl.cols._f_col(colname))

        # Column types.
        expectedTypes = [columns[colname].dtype
                         for colname in expectedNames]
        self.assertEqual(expectedTypes,
                         [tbl.coldtypes[v] for v in expectedNames])
        self.assertEqual(expectedTypes,
                         [desc._v_dtypes[v] for v in expectedNames])

        # Column string types.
        expectedTypes = [columns[colname].type
                         for colname in expectedNames]
        self.assertEqual(expectedTypes,
                         [tbl.coltypes[v] for v in expectedNames])
        self.assertEqual(expectedTypes,
                         [desc._v_types[v] for v in expectedNames])

        # Column defaults.
        for v in expectedNames:
            if common.verbose:
                print "dflt-->", columns[v].dflt, type(columns[v].dflt)
                print "coldflts-->", tbl.coldflts[v], type(tbl.coldflts[v])
                print "desc.dflts-->", desc._v_dflts[v], type(desc._v_dflts[v])
            self.assertTrue(areArraysEqual(tbl.coldflts[v], columns[v].dflt))
            self.assertTrue(areArraysEqual(desc._v_dflts[v], columns[v].dflt))

        # Column path names.
        self.assertEqual(expectedNames, list(desc._v_pathnames))

        # Column objects.
        for colName in expectedNames:
            expectedCol = columns[colName]
            col = desc._v_colobjects[colName]

            self.assertEqual(expectedCol.dtype, col.dtype)
            self.assertEqual(expectedCol.type, col.type)

    def test01_readTable(self):
        """Checking table read"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Choose a small value for buffer size
        table.nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [rec['var2'] for rec in table.iterrows() if rec['var2'] < 20]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        rec = list(table.iterrows())[-1]
        self.assertEqual((rec['var1'], rec['var2'], rec['var7']),
                         (b"0001", nrows, b"1"))
        if isinstance(rec['var5'], np.ndarray):
            self.assertTrue(allequal(rec['var5'],
                                     np.array((float(nrows),)*4, np.float32)))
        else:
            self.assertEqual(rec['var5'], float(nrows))
        if isinstance(rec['var9'], np.ndarray):
            self.assertTrue(
                allequal(rec['var9'],
                         np.array([0.+float(nrows)*1.j, float(nrows)+0.j],
                                  np.complex64)))
        else:
            self.assertEqual((rec['var9']), float(nrows)+0.j)
        self.assertEqual(len(result), 20)

    def test01a_fetch_all_fields(self):
        """Checking table read (using Row.fetch_all_fields)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_fetch_all_fields..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Choose a small value for buffer size
        table.nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [rec.fetch_all_fields() for rec in table.iterrows()
                  if rec['var2'] < 20]
        rec = result[-1]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = 20 - 1
        strnrows = "%04d" % (self.expectedrows - nrows)
        strnrows = strnrows.encode('ascii')
        self.assertEqual((rec['var1'], rec['var2'], rec['var7']),
                         (strnrows, nrows, b"1"))
        if isinstance(rec['var5'], np.ndarray):
            self.assertTrue(allequal(rec['var5'],
                                     np.array((float(nrows),)*4, np.float32)))
        else:
            self.assertEqual(rec['var5'], float(nrows))
        if isinstance(rec['var9'], np.ndarray):
            self.assertTrue(
                allequal(rec['var9'],
                         np.array([0.+float(nrows)*1.j, float(nrows)+0.j],
                                  np.complex64)))
        else:
            self.assertEqual(rec['var9'], float(nrows)+0.j)
        self.assertEqual(len(result), 20)

    def test01a_integer(self):
        """Checking table read (using Row[integer])"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_integer..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Choose a small value for buffer size
        table.nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [rec[1] for rec in table.iterrows()
                  if rec['var2'] < 20]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Total selected records in table ==> ", len(result)
            print "All results ==>", result
        self.assertEqual(len(result), 20)
        self.assertEqual(result, range(20))

    def test01a_extslice(self):
        """Checking table read (using Row[::2])"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_extslice..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Choose a small value for buffer size
        table.nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [rec[::2] for rec in table.iterrows()
                  if rec['var2'] < 20]
        rec = result[-1]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = 20 - 1
        strnrows = "%04d" % (self.expectedrows - nrows)
        strnrows = strnrows.encode('ascii')
        self.assertEqual(rec[:2], (strnrows, 19))
        self.assertEqual(rec[3], b'1')
        if isinstance(rec[2], np.ndarray):
            self.assertTrue(allequal(rec[2],
                                     np.array((float(nrows),)*4, np.float32)))
        else:
            self.assertEqual(rec[2], nrows)
        if isinstance(rec[4], np.ndarray):
            self.assertTrue(
                allequal(rec[4],
                         np.array([0.+float(nrows)*1.j, float(nrows)+0.j],
                                  np.complex64)))
        else:
            self.assertEqual(rec[4], float(nrows)+0.j)
        self.assertEqual(len(result), 20)

    def test01a_nofield(self):
        """Checking table read (using Row['no-field'])"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_nofield..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Check that a KeyError is raised
        # self.assertRaises only work with functions
        # self.assertRaises(KeyError, [rec['no-field'] for rec in table])
        try:
            result = [rec['no-field'] for rec in table]
        except KeyError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next KeyError was catched!"
                print value
        else:
            print result
            self.fail("expected a KeyError")

    def test01a_badtypefield(self):
        """Checking table read (using Row[{}])"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_badtypefield..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Check that a TypeError is raised
        # self.assertRaises only work with functions
        # self.assertRaises(TypeError, [rec[{}] for rec in table])
        try:
            result = [rec[{}] for rec in table]
        except TypeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next TypeError was catched!"
                print value
        else:
            print result
            self.fail("expected a TypeError")

    def test01b_readTable(self):
        """Checking table read and cuts (multidimensional columns case)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Choose a small value for buffer size
        table.nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [rec['var5'] for rec in table.iterrows()
                  if rec['var2'] < 20]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "rec['var5'] ==>", rec['var5'],
            print "nrows ==>", table.nrows
            print "Total selected records in table ==> ", len(result)
        nrows = table.nrows
        rec = list(table.iterrows())[-1]
        if isinstance(rec['var5'], np.ndarray):
            npt.assert_array_equal(result[0],
                                   np.array((float(0),)*4, np.float32))
            npt.assert_array_equal(result[1],
                                   np.array((float(1),)*4, np.float32))
            npt.assert_array_equal(result[2],
                                   np.array((float(2),)*4, np.float32))
            npt.assert_array_equal(result[3],
                                   np.array((float(3),)*4, np.float32))
            npt.assert_array_equal(result[10],
                                   np.array((float(10),)*4, np.float32))
            npt.assert_array_equal(rec['var5'],
                                   np.array((float(nrows-1),)*4, np.float32))
        else:
            self.assertEqual(rec['var5'], float(nrows - 1))
        # Read the records and select those with "var2" file less than 20
        result = [rec['var10'] for rec in table.iterrows()
                  if rec['var2'] < 20]
        if isinstance(rec['var10'], np.ndarray):
            npt.assert_array_equal(
                result[0],
                np.array([float(0)+0.j, 1.+float(0)*1j], np.complex128))
            npt.assert_array_equal(
                result[1],
                np.array([float(1)+0.j, 1.+float(1)*1j], np.complex128))
            npt.assert_array_equal(
                result[2],
                np.array([float(2)+0.j, 1.+float(2)*1j], np.complex128))
            npt.assert_array_equal(
                result[3],
                np.array([float(3)+0.j, 1.+float(3)*1j], np.complex128))
            npt.assert_array_equal(
                result[10],
                np.array([float(10)+0.j, 1.+float(10)*1j], np.complex128))
            npt.assert_array_equal(
                rec['var10'],
                np.array([float(nrows-1)+0.j, 1.+float(nrows-1)*1j],
                         np.complex128))
        else:
            self.assertEqual(rec['var10'], 1.+float(nrows-1)*1j)
        self.assertEqual(len(result), 20)

    def test01c_readTable(self):
        """Checking nested iterators (reading)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Read the records and select those with "var2" file less than 20
        result = []
        for rec in table.iterrows(stop=2):
            for rec2 in table.iterrows(stop=2):
                if rec2['var2'] < 20:
                    result.append([rec['var2'], rec2['var2']])
        if common.verbose:
            print "result ==>", result

        self.assertEqual(result, [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test01d_readTable(self):
        """Checking nested iterators (reading, mixed conditions)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01d_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Read the records and select those with "var2" file less than 20
        result = []
        for rec in table.iterrows(stop=2):
            for rec2 in table.where('var2 < 20', stop=2):
                result.append([rec['var2'], rec2['var2']])
        if common.verbose:
            print "result ==>", result

        self.assertEqual(result, [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test01e_readTable(self):
        """Checking nested iterators (reading, both conditions)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01e_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Read the records and select those with "var2" file less than 20
        result = []
        for rec in table.where('var3 < 2'):
            for rec2 in table.where('var2 < 3'):
                result.append([rec['var2'], rec2['var3']])
        if common.verbose:
            print "result ==>", result

        self.assertEqual(result,
                         [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

    def test01f_readTable(self):
        """Checking nested iterators (reading, break in the loop)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01f_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Read the records and select those with "var2" file less than 20
        result = []
        for rec in table.where('var3 < 2'):
            for rec2 in table.where('var2 < 4'):
                if rec2['var2'] >= 3:
                    break
                result.append([rec['var2'], rec2['var3']])
        if common.verbose:
            print "result ==>", result

        self.assertEqual(result,
                         [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

    def test01g_readTable(self):
        """Checking iterator with an evanescent table."""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01g_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")

        # Read from an evanescent table
        result = [rec['var2'] for rec in self.fileh.get_node("/table0")
                  if rec['var2'] < 20]

        self.assertEqual(len(result), 20)

    def test02_AppendRows(self):
        """Checking whether appending record rows works or not"""

        # Now, open it, but in "append" mode
        self.fileh = open_file(self.file, mode="a")
        self.rootgroup = self.fileh.root
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_AppendRows..." % self.__class__.__name__

        # Get a table
        table = self.fileh.get_node("/group0/table1")
        # Get their row object
        row = table.row
        if common.verbose:
            print "Nrows in old", table._v_pathname, ":", table.nrows
            print "Record Format ==>", table.description._v_nested_formats
            print "Record Size ==>", table.rowsize
        # Append some rows
        for i in xrange(self.appendrows):
            s = '%04d' % (self.appendrows - i)
            row['var1'] = s.encode('ascii')
            row['var7'] = s[-1].encode('ascii')
            row['var2'] = i
            row['var3'] = i % self.maxshort
            if isinstance(row['var4'], np.ndarray):
                row['var4'] = [float(i), float(i * i)]
            else:
                row['var4'] = float(i)
            if isinstance(row['var8'], np.ndarray):
                row['var8'] = [0, 1]
            else:
                row['var8'] = 1
            if isinstance(row['var9'], np.ndarray):
                row['var9'] = [0.+float(i)*1j, float(i)+0.j]
            else:
                row['var9'] = float(i)+0.j
            if isinstance(row['var10'], np.ndarray):
                row['var10'] = [float(i)+0.j, 1.+float(i)*1j]
            else:
                row['var10'] = 1.+float(i)*1j
            if isinstance(row['var5'], np.ndarray):
                row['var5'] = np.array((float(i),)*4)
            else:
                row['var5'] = float(i)
            if 'Float16Col' in globals():
                if isinstance(row['var11'], np.ndarray):
                    row['var11'] = np.array((float(i),)*4)
                else:
                    row['var11'] = float(i)
            if 'Float96Col' in globals():
                if isinstance(row['var12'], np.ndarray):
                    row['var12'] = np.array((float(i),)*4)
                else:
                    row['var12'] = float(i)
            if 'Float128Col' in globals():
                if isinstance(row['var13'], np.ndarray):
                    row['var13'] = np.array((float(i),)*4)
                else:
                    row['var13'] = float(i)
            if 'Complex192Col' in globals():
                if isinstance(row['var14'], np.ndarray):
                    row['var14'] = [float(i)+0j, 1 + float(i)*1j]
                else:
                    row['var14'] = 1 + float(i)*1j
            if 'Complex256Col' in globals():
                if isinstance(row['var15'], np.ndarray):
                    row['var15'] = [float(i)+0j, 1 + float(i)*1j]
                else:
                    row['var15'] = 1 + float(i)*1j

            row.append()

        # Flush the buffer for this table and read it
        table.flush()
        result = [row['var2'] for row in table.iterrows() if row['var2'] < 20]

        nrows = self.appendrows - 1
        row = list(table.iterrows())[-1]
        self.assertEqual((row['var1'], row['var2'], row['var7']),
                         (b"0001", nrows, b"1"))
        if isinstance(row['var5'], np.ndarray):
            self.assertTrue(allequal(row['var5'],
                                     np.array((float(nrows),)*4, np.float32)))
        else:
            self.assertEqual(row['var5'], float(nrows))
        if self.appendrows <= 20:
            add = self.appendrows
        else:
            add = 20
        self.assertEqual(len(result), 20 + add)  # because we appended new rows

    # This test has been commented out because appending records without
    # flushing them explicitely is being warned from now on.
    # F. Alted 2006-08-03
    def _test02a_AppendRows(self):
        """Checking appending records without flushing explicitely"""

        # Now, open it, but in "append" mode
        self.fileh = open_file(self.file, mode="a")
        self.rootgroup = self.fileh.root
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a_AppendRows..." % self.__class__.__name__

        group = self.rootgroup
        for i in range(3):
            # Get a table
            table = self.fileh.get_node(group, 'table'+str(i))
            # Get the next group
            group = self.fileh.get_node(group, 'group'+str(i))
            # Get their row object
            row = table.row
            if common.verbose:
                print "Nrows in old", table._v_pathname, ":", table.nrows
                print "Record Format ==>", table.description._v_nested_formats
                print "Record Size ==>", table.rowsize
            # Append some rows
            for i in xrange(self.appendrows):
                row['var1'] = '%04d' % (self.appendrows - i)
                row['var7'] = row['var1'][-1]
                row['var2'] = i
                row['var3'] = i % self.maxshort
                if isinstance(row['var4'], np.ndarray):
                    row['var4'] = [float(i), float(i * i)]
                else:
                    row['var4'] = float(i)
                if isinstance(row['var8'], np.ndarray):
                    row['var8'] = [0, 1]
                else:
                    row['var8'] = 1
                if isinstance(row['var9'], np.ndarray):
                    row['var9'] = [0.+float(i)*1j, float(i)+0.j]
                else:
                    row['var9'] = float(i)+0.j
                if isinstance(row['var10'], np.ndarray):
                    row['var10'] = [float(i)+0.j, 1.+float(i)*1j]
                else:
                    row['var10'] = 1.+float(i)*1j
                if isinstance(row['var5'], np.ndarray):
                    row['var5'] = np.array((float(i),)*4)
                else:
                    row['var5'] = float(i)
                if 'Float16Col' in globals():
                    if isinstance(row['var11'], np.ndarray):
                        row['var11'] = np.array((float(i),)*4)
                    else:
                        row['var11'] = float(i)
                if 'Float96Col' in globals():
                    if isinstance(row['var12'], np.ndarray):
                        row['var12'] = np.array((float(i),)*4)
                    else:
                        row['var12'] = float(i)
                if 'Float128Col' in globals():
                    if isinstance(row['var13'], np.ndarray):
                        row['var13'] = np.array((float(i),)*4)
                    else:
                        row['var13'] = float(i)
                if 'Complex192Col' in globals():
                    if isinstance(row['var14'], np.ndarray):
                        row['var14'] = [float(i)+0j, 1 + float(i)*1j]
                    else:
                        row['var14'] = 1 + float(i)*1j
                if 'Complex256Col' in globals():
                    if isinstance(row['var15'], np.ndarray):
                        row['var15'] = [float(i)+0j, 1 + float(i)*1j]
                    else:
                        row['var15'] = 1 + float(i)*1j

                row.append()
            table.flush()

        # Close the file and re-open it.
        self.fileh.close()

        self.fileh = open_file(self.file, mode="a")
        table = self.fileh.root.table0
        # Flush the buffer for this table and read it
        result = [row['var2'] for row in table.iterrows()
                  if row['var2'] < 20]

        nrows = self.appendrows - 1
        self.assertEqual((row['var1'], row['var2'], row['var7']),
                         ("0001", nrows, "1"))
        if isinstance(row['var5'], np.ndarray):
            self.assertTrue(allequal(row['var5'],
                                     np.array((float(nrows),)*4, np.float32)))
        else:
            self.assertEqual(row['var5'], float(nrows))
        if self.appendrows <= 20:
            add = self.appendrows
        else:
            add = 20
        self.assertEqual(len(result), 20 + add)  # because we appended new rows

    def test02b_AppendRows(self):
        """Checking whether appending *and* reading rows works or not"""

        # Now, open it, but in "append" mode
        self.fileh = open_file(self.file, mode="a")
        self.rootgroup = self.fileh.root
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_AppendRows..." % self.__class__.__name__

        # Get a table
        table = self.fileh.get_node("/group0/table1")
        if common.verbose:
            print "Nrows in old", table._v_pathname, ":", table.nrows
            print "Record Format ==>", table.description._v_nested_formats
            print "Record Size ==>", table.rowsize
        # Set a small number of buffer to make this test faster
        table.nrowsinbuf = 3
        # Get their row object
        row = table.row
        # Append some rows (3 * table.nrowsinbuf is enough for
        # checking purposes)
        for i in xrange(3 * table.nrowsinbuf):
            s = '%04d' % (self.appendrows - i)
            row['var1'] = s.encode('ascii')
            row['var7'] = s[-1].encode('ascii')
            # row['var7'] = table.cols['var1'][i][-1]
            row['var2'] = i
            row['var3'] = i % self.maxshort
            if isinstance(row['var4'], np.ndarray):
                row['var4'] = [float(i), float(i * i)]
            else:
                row['var4'] = float(i)
            if isinstance(row['var8'], np.ndarray):
                row['var8'] = [0, 1]
            else:
                row['var8'] = 1
            if isinstance(row['var9'], np.ndarray):
                row['var9'] = [0.+float(i)*1j, float(i)+0.j]
            else:
                row['var9'] = float(i)+0.j
            if isinstance(row['var10'], np.ndarray):
                row['var10'] = [float(i)+0.j, 1.+float(i)*1j]
            else:
                row['var10'] = 1.+float(i)*1j
            if isinstance(row['var5'], np.ndarray):
                row['var5'] = np.array((float(i),)*4)
            else:
                row['var5'] = float(i)
            if 'Float16Col' in globals():
                if isinstance(row['var11'], np.ndarray):
                    row['var11'] = np.array((float(i),)*4)
                else:
                    row['var11'] = float(i)
            if 'Float96Col' in globals():
                if isinstance(row['var12'], np.ndarray):
                    row['var12'] = np.array((float(i),)*4)
                else:
                    row['var12'] = float(i)
            if 'Float128Col' in globals():
                if isinstance(row['var13'], np.ndarray):
                    row['var13'] = np.array((float(i),)*4)
                else:
                    row['var13'] = float(i)
            if 'Complex192Col' in globals():
                if isinstance(row['var14'], np.ndarray):
                    row['var14'] = [float(i)+0j, 1 + float(i)*1j]
                else:
                    row['var14'] = 1 + float(i)*1j
            if 'Complex256Col' in globals():
                if isinstance(row['var15'], np.ndarray):
                    row['var15'] = [float(i)+0j, 1 + float(i)*1j]
                else:
                    row['var15'] = 1 + float(i)*1j

            row.append()
            # the next call can mislead the counters
            result = [row2['var2'] for row2 in table]
            # warning! the next will result into wrong results
            # result = [ row['var2'] for row in table ]
            # This is because the iterator for writing and for reading
            # cannot be shared!

        # Do not flush the buffer for this table and try to read it
        # We are forced now to flush tables after append operations
        # because of unsolved issues with the LRU cache that are too
        # difficult to track.
        # F. Alted 2006-08-03
        table.flush()
        result = [row['var2'] for row in table.iterrows() if row['var2'] < 20]
        if common.verbose:
            print "Result length ==>", len(result)
            print "Result contents ==>", result
        self.assertEqual(len(result), 20 + 3 * table.nrowsinbuf)
        self.assertEqual(result, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                  0, 1, 2, 3, 4, 5, 6, 7, 8])
        # Check consistency of I/O buffers when doing mixed I/O operations
        # That is, the next should work in these operations
        # row['var1'] = '%04d' % (self.appendrows - i)
        # row['var7'] = row['var1'][-1]
        result7 = [row['var7'] for row in table.iterrows() if row['var2'] < 20]
        if common.verbose:
            print "Result7 length ==>", len(result7)
            print "Result7 contents ==>", result7
        self.assertEqual(
            result7,
            [b'0', b'9', b'8', b'7', b'6', b'5', b'4', b'3', b'2', b'1',
             b'0', b'9', b'8', b'7', b'6', b'5', b'4', b'3', b'2', b'1',
             b'0', b'9', b'8', b'7', b'6', b'5', b'4', b'3', b'2'])

    # This test is commented out as it should not work anymore due to
    # the new policy of not doing a flush in the middle of a __del__
    # operation. F. Alted 2006-08-24
    def _test02c_AppendRows(self):
        """Checking appending with evanescent table objects"""

        # This test is kind of magic, but it is a good sanity check anyway.

        # Now, open it, but in "append" mode
        self.fileh = open_file(self.file, mode="a")
        self.rootgroup = self.fileh.root
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02c_AppendRows..." % self.__class__.__name__

        # Get a table
        table = self.fileh.get_node("/group0/table1")
        if common.verbose:
            print "Nrows in old", table._v_pathname, ":", table.nrows
            print "Record Format ==>", table.description._v_nested_formats
            print "Record Size ==>", table.rowsize
        # Set a small number of buffer to make this test faster
        table.nrowsinbuf = 3
        # Get their row object
        self.row = table.row
        # delete the table reference
        del table
        # Append some rows
        for i in xrange(22):
            self.row['var2'] = 100 + i
            self.row.append()
        # del self.row # force the table object to be destroyed (and the
                       # user warned!)
        # convert a warning in an error
        warnings.filterwarnings('error', category=PerformanceWarning)
        self.assertRaises(PerformanceWarning, self.__dict__.pop, 'row')
#         try:
#             # force the table object to be destroyed
#             self.__dict__.pop('row')
#         except PerformanceWarning:
#             if common.verbose:
#                 (type, value, traceback) = sys.exc_info()
#                 print "\nGreat!, the next PerformanceWarning was catched:"
#                 print value
#             # Ignore the warning and actually flush the table
#             warnings.filterwarnings("ignore", category=PerformanceWarning)
#             table = self.fileh.get_node("/group0/table1")
#             table.flush()
#         else:
#             self.fail("expected a PeformanceWarning")
        # reset the warning
        warnings.filterwarnings('default', category=PerformanceWarning)
        result = [row['var2'] for row in table.iterrows()
                  if 100 <= row['var2'] < 122]
        if common.verbose:
            print "Result length ==>", len(result)
            print "Result contents ==>", result
        self.assertEqual(len(result), 22)
        self.assertEqual(
            result,
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
             111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121])

    def test02d_AppendRows(self):
        """Checking appending using the same Row object after flushing."""

        # This test is kind of magic, but it is a good sanity check anyway.

        # Now, open it, but in "append" mode
        self.fileh = open_file(self.file, mode="a")
        self.rootgroup = self.fileh.root
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02d_AppendRows..." % self.__class__.__name__

        # Get a table
        table = self.fileh.get_node("/group0/table1")
        if common.verbose:
            print "Nrows in old", table._v_pathname, ":", table.nrows
            print "Record Format ==>", table.description._v_nested_formats
            print "Record Size ==>", table.rowsize
        # Set a small number of buffer to make this test faster
        table.nrowsinbuf = 3
        # Get their row object
        row = table.row
        # Append some rows
        for i in xrange(10):
            row['var2'] = 100 + i
            row.append()
        # Force a flush
        table.flush()
        # Add new rows
        for i in xrange(9):
            row['var2'] = 110 + i
            row.append()
        table.flush()  # XXX al eliminar...
        result = [row['var2'] for row in table.iterrows()
                  if 100 <= row['var2'] < 120]
        if common.verbose:
            print "Result length ==>", len(result)
            print "Result contents ==>", result
        if table.nrows > 119:
            # Case for big tables
            self.assertEqual(len(result), 39)
            self.assertEqual(result,
                             [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                              110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                              100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                              110, 111, 112, 113, 114, 115, 116, 117, 118])
        else:
            self.assertEqual(len(result), 19)
            self.assertEqual(result,
                             [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                              110, 111, 112, 113, 114, 115, 116, 117, 118])

    def test02e_AppendRows(self):
        """Checking appending using the Row of an unreferenced table."""
        # See ticket #94 (http://www.pytables.org/trac/ticket/94).

        # Reopen the file in append mode.
        self.fileh = open_file(self.file, mode='a')

        # Get the row handler which will outlive the reference to the table.
        table = self.fileh.get_node('/group0/table1')
        oldnrows = table.nrows
        row = table.row

        # Few appends are made to avoid flushing the buffers in ``row``.

        # First case: append to an alive (referenced) table.
        row.append()
        table.flush()
        newnrows = table.nrows
        self.assertEqual(newnrows, oldnrows + 1,
                         "Append to alive table failed.")

        if self.fileh._aliveNodes.nodeCacheSlots == 0:
            # Skip this test from here on because the second case
            # won't work when thereis not a node cache.
            return

        # Second case: append to a dead (unreferenced) table.
        del table
        row.append()
        table = self.fileh.get_node('/group0/table1')
        table.flush()
        newnrows = table.nrows
        self.assertEqual(newnrows, oldnrows + 2,
                         "Append to dead table failed.")

    # CAVEAT: The next test only works for tables with rows < 2**15
    def test03_endianess(self):
        """Checking if table is endianess aware"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_endianess..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/group0/group1/table2")

        # Read the records and select the ones with "var3" column less than 20
        result = [rec['var2'] for rec in table.iterrows() if rec['var3'] < 20]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "On-disk byteorder ==>", table.byteorder
            print "Last record in table ==>", rec
            print "Selected records ==>", result
            print "Total selected records in table ==>", len(result)
        nrows = self.expectedrows - 1
        self.assertEqual(table.byteorder,
                         {"little": "big", "big": "little"}[sys.byteorder])
        rec = list(table.iterrows())[-1]
        self.assertEqual((rec['var1'], rec['var3']), (b"0001", nrows))
        self.assertEqual(len(result), 20)

    def test04_delete(self):
        """Checking whether a single row can be deleted"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "a")
        table = self.fileh.get_node("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result[-1]
            print "Total selected records in table ==>", len(result)

        nrows = table.nrows
        table.nrowsinbuf = 3  # small value of the buffer
        # Delete the twenty-th row
        table.remove_rows(19, 20)

        # Re-read the records
        result2 = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        self.assertEqual(table.nrows, nrows - 1)
        self.assertEqual(table.shape, (nrows - 1,))
        # Check that the new list is smaller than the original one
        self.assertEqual(len(result), len(result2) + 1)
        self.assertEqual(result[:-1], result2)

    def test04a_delete(self):
        """Checking whether a single row can be deleted"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "a")
        table = self.fileh.get_node("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result[-1]
            print "Total selected records in table ==>", len(result)

        nrows = table.nrows
        table.nrowsinbuf = 3  # small value of the buffer
        # Delete the twenty-th row
        table.remove_row(19)

        # Re-read the records
        result2 = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        self.assertEqual(table.nrows, nrows - 1)
        self.assertEqual(table.shape, (nrows - 1,))
        # Check that the new list is smaller than the original one
        self.assertEqual(len(result), len(result2) + 1)
        self.assertEqual(result[:-1], result2)

    def test04b_delete(self):
        """Checking whether a range of rows can be deleted"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "a")
        table = self.fileh.get_node("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result[-1]
            print "Total selected records in table ==>", len(result)

        nrows = table.nrows
        table.nrowsinbuf = 4  # small value of the buffer
        # Delete the last ten rows
        table.remove_rows(10, 20)

        # Re-read the records
        result2 = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        self.assertEqual(table.nrows, nrows - 10)
        self.assertEqual(table.shape, (nrows - 10,))
        # Check that the new list is smaller than the original one
        self.assertEqual(len(result), len(result2) + 10)
        self.assertEqual(result[:10], result2)

    def test04c_delete(self):
        """Checking whether removing a bad range of rows is detected"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04c_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "a")
        table = self.fileh.get_node("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        nrows = table.nrows
        table.nrowsinbuf = 5  # small value of the buffer
        # Delete a too large range of rows
        table.remove_rows(10, nrows + 100)

        # Re-read the records
        result2 = [r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        self.assertEqual(table.nrows, 10)
        self.assertEqual(table.shape, (10,))
        # Check that the new list is smaller than the original one
        self.assertEqual(len(result), len(result2) + 10)
        self.assertEqual(result[:10], result2)

    def test04d_delete(self):
        """Checking whether removing rows several times at once is working"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04d_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "a")
        table = self.fileh.get_node("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [r['var2'] for r in table if r['var2'] < 20]

        nrows = table.nrows
        nrowsinbuf = table.nrowsinbuf
        table.nrowsinbuf = 6  # small value of the buffer
        # Delete some rows
        table.remove_rows(10, 15)
        # It's necessary to restore the value of buffer to use the row object
        # afterwards...
        table.nrowsinbuf = nrowsinbuf

        # Append some rows
        row = table.row
        for i in xrange(10, 15):
            row['var1'] = '%04d' % (self.appendrows - i)
            # This line gives problems on Windows. Why?
            # row['var7'] = row['var1'][-1]
            row['var2'] = i
            row['var3'] = i % self.maxshort
            if isinstance(row['var4'], np.ndarray):
                row['var4'] = [float(i), float(i * i)]
            else:
                row['var4'] = float(i)
            if isinstance(row['var8'], np.ndarray):
                row['var8'] = [0, 1]
            else:
                row['var8'] = 1
            if isinstance(row['var9'], np.ndarray):
                row['var9'] = [0.+float(i)*1j, float(i)+0.j]
            else:
                row['var9'] = float(i)+0.j
            if isinstance(row['var10'], np.ndarray):
                row['var10'] = [float(i)+0.j, 1.+float(i)*1j]
            else:
                row['var10'] = 1.+float(i)*1j
            if isinstance(row['var5'], np.ndarray):
                row['var5'] = np.array((float(i),)*4)
            else:
                row['var5'] = float(i)
            if 'Float16Col' in globals():
                if isinstance(row['var11'], np.ndarray):
                    row['var11'] = np.array((float(i),)*4)
                else:
                    row['var11'] = float(i)
            if 'Float96Col' in globals():
                if isinstance(row['var12'], np.ndarray):
                    row['var12'] = np.array((float(i),)*4)
                else:
                    row['var12'] = float(i)
            if 'Float128Col' in globals():
                if isinstance(row['var13'], np.ndarray):
                    row['var13'] = np.array((float(i),)*4)
                else:
                    row['var13'] = float(i)
            if 'Complex192Col' in globals():
                if isinstance(row['var14'], np.ndarray):
                    row['var14'] = [float(i)+0j, 1 + float(i)*1j]
                else:
                    row['var14'] = 1 + float(i)*1j
            if 'Complex256Col' in globals():
                if isinstance(row['var15'], np.ndarray):
                    row['var15'] = [float(i)+0j, 1 + float(i)*1j]
                else:
                    row['var15'] = 1 + float(i)*1j

            row.append()
        # Flush the buffer for this table
        table.flush()

        # Delete 5 rows more
        table.remove_rows(5, 10)

        # Re-read the records
        result2 = [r['var2'] for r in table if r['var2'] < 20]

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        self.assertEqual(table.nrows, nrows - 5)
        self.assertEqual(table.shape, (nrows - 5,))
        # Check that the new list is smaller than the original one
        self.assertEqual(len(result), len(result2) + 5)
        # The last values has to be equal
        self.assertEqual(result[10:15], result2[10:15])

    def test05_filtersTable(self):
        """Checking tablefilters"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_filtersTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        # Check filters:
        if self.compress != table.filters.complevel and common.verbose:
            print "Error in compress. Class:", self.__class__.__name__
            print "self, table:", self.compress, table.filters.complevel
        self.assertEqual(table.filters.complevel, self.compress)
        if self.compress > 0 and which_lib_version(self.complib):
            self.assertEqual(table.filters.complib, self.complib)
        if self.shuffle != table.filters.shuffle and common.verbose:
            print "Error in shuffle. Class:", self.__class__.__name__
            print "self, table:", self.shuffle, table.filters.shuffle
        self.assertEqual(self.shuffle, table.filters.shuffle)
        if self.fletcher32 != table.filters.fletcher32 and common.verbose:
            print "Error in fletcher32. Class:", self.__class__.__name__
            print "self, table:", self.fletcher32, table.filters.fletcher32
        self.assertEqual(self.fletcher32, table.filters.fletcher32)

    def test06_attributes(self):
        self.fileh = open_file(self.file)
        obj = self.fileh.get_node('/table0')

        self.assertEqual(obj.flavor, 'numpy')
        self.assertEqual(obj.shape, (self.expectedrows,))
        self.assertEqual(obj.ndim, 1)
        self.assertEqual(obj.nrows, self.expectedrows)


class BasicWriteTestCase(BasicTestCase):
    title = "BasicWrite"


class OldRecordBasicWriteTestCase(BasicTestCase):
    title = "OldRecordBasicWrite"
    record = OldRecord


class DictWriteTestCase(BasicTestCase):
    # This checks also unidimensional arrays as columns
    title = "DictWrite"
    record = RecordDescriptionDict
    nrows = 21
    nrowsinbuf = 3  # Choose a small value for the buffer size
    start = 0
    stop = 10
    step = 3


if sys.version_info < (3,):
    class DictWriteTestCase2(DictWriteTestCase):
        record = RecordDescriptionDict.copy()
        record[unicode('var1')] = record.pop('var1')


# Pure NumPy dtype
class NumPyDTWriteTestCase(BasicTestCase):
    title = "NumPyDTWriteTestCase"
    formats = "a4,i4,i2,2f8,f4,i2,a1,b1,c8,c16".split(',')
    names = 'var1,var2,var3,var4,var5,var6,var7,var8,var9,var10'.split(',')

    if 'Float16Col' in globals():
        formats.append('f2')
        names.append('var11')
    if 'Float96Col' in globals():
        formats.append('f12')
        names.append('var12')
    if 'Float128Col' in globals():
        formats.append('f16')
        names.append('var13')
    if 'Complex192Col' in globals():
        formats.append('c24')
        names.append('var14')
    if 'Complex256Col' in globals():
        formats.append('c32')
        names.append('var15')

    record = np.dtype(','.join(formats))
    record.names = names


class RecArrayOneWriteTestCase(BasicTestCase):
    title = "RecArrayOneWrite"
    formats = "a4,i4,i2,2f8,f4,i2,a1,b1,c8,c16".split(',')
    names = 'var1,var2,var3,var4,var5,var6,var7,var8,var9,var10'.split(',')

    if 'Float16Col' in globals():
        formats.append('f2')
        names.append('var11')
    if 'Float96Col' in globals():
        formats.append('f12')
        names.append('var12')
    if 'Float128Col' in globals():
        formats.append('f16')
        names.append('var13')
    if 'Complex192Col' in globals():
        formats.append('c24')
        names.append('var14')
    if 'Complex256Col' in globals():
        formats.append('c32')
        names.append('var15')

    record = records.array(None, shape=0, formats=','.join(formats),
                           names=names)


class RecArrayTwoWriteTestCase(BasicTestCase):
    title = "RecArrayTwoWrite"
    expectedrows = 100
    recarrayinit = 1
    formats = "a4,i4,i2,2f8,f4,i2,a1,b1,c8,c16".split(',')
    names = 'var1,var2,var3,var4,var5,var6,var7,var8,var9,var10'.split(',')

    if 'Float16Col' in globals():
        formats.append('f2')
        names.append('var11')
    if 'Float96Col' in globals():
        formats.append('f12')
        names.append('var12')
    if 'Float128Col' in globals():
        formats.append('f16')
        names.append('var13')
    if 'Complex192Col' in globals():
        formats.append('c24')
        names.append('var14')
    if 'Complex256Col' in globals():
        formats.append('c32')
        names.append('var15')

    recordtemplate = records.array(None, shape=1, formats=','.join(formats),
                                   names=names)


class RecArrayThreeWriteTestCase(BasicTestCase):
    title = "RecArrayThreeWrite"
    expectedrows = 100
    recarrayinit = 1
    formats = "a4,i4,i2,2f8,f4,i2,a1,b1,c8,c16".split(',')
    names = 'var1,var2,var3,var4,var5,var6,var7,var8,var9,var10'.split(',')

    if 'Float16Col' in globals():
        formats.append('f2')
        names.append('var11')
    if 'Float96Col' in globals():
        formats.append('f12')
        names.append('var12')
    if 'Float128Col' in globals():
        formats.append('f16')
        names.append('var13')
    if 'Complex192Col' in globals():
        formats.append('c24')
        names.append('var14')
    if 'Complex256Col' in globals():
        formats.append('c32')
        names.append('var15')

    recordtemplate = records.array(None, shape=1, formats=','.join(formats),
                                   names=names)


class CompressBloscTablesTestCase(BasicTestCase):
    title = "CompressBloscTables"
    compress = 6
    complib = "blosc"


class CompressBloscShuffleTablesTestCase(BasicTestCase):
    title = "CompressBloscTables"
    compress = 1
    shuffle = 1
    complib = "blosc"


class CompressLZOTablesTestCase(BasicTestCase):
    title = "CompressLZOTables"
    compress = 1
    complib = "lzo"


class CompressLZOShuffleTablesTestCase(BasicTestCase):
    title = "CompressLZOTables"
    compress = 1
    shuffle = 1
    complib = "lzo"


class CompressBzip2TablesTestCase(BasicTestCase):
    title = "CompressBzip2Tables"
    compress = 1
    complib = "bzip2"


class CompressBzip2ShuffleTablesTestCase(BasicTestCase):
    title = "CompressBzip2Tables"
    compress = 1
    shuffle = 1
    complib = "bzip2"


class CompressZLIBTablesTestCase(BasicTestCase):
    title = "CompressOneTables"
    compress = 1
    complib = "zlib"


class CompressZLIBShuffleTablesTestCase(BasicTestCase):
    title = "CompressOneTables"
    compress = 1
    shuffle = 1
    complib = "zlib"


class Fletcher32TablesTestCase(BasicTestCase):
    title = "Fletcher32Tables"
    fletcher32 = 1
    shuffle = 0
    complib = "zlib"


class AllFiltersTablesTestCase(BasicTestCase):
    title = "AllFiltersTables"
    compress = 1
    fletcher32 = 1
    shuffle = 1
    complib = "zlib"


class CompressTwoTablesTestCase(BasicTestCase):
    title = "CompressTwoTables"
    compress = 1
    # This checks also unidimensional arrays as columns
    record = RecordDescriptionDict


class BigTablesTestCase(BasicTestCase):
    title = "BigTables"
    # 10000 rows takes much more time than we can afford for tests
    # reducing to 1000 would be more than enough
    # F. Alted 2004-01-19
    # Will be executed only in common.heavy mode
    expectedrows = 10000
    appendrows = 100


class SizeOnDiskInMemoryPropertyTestCase(unittest.TestCase):
    def setUp(self):
        # set chunkshape so it divides evenly into array_size, to avoid
        # partially filled chunks
        self.chunkshape = (1000, )
        self.dtype = np.format_parser(['i4'] * 10, [], []).dtype
        # approximate size (in bytes) of non-data portion of hdf5 file
        self.hdf_overhead = 6000
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, mode="w")

    def tearDown(self):
        self.fileh.close()
        # Then, delete the file
        os.remove(self.file)
        common.cleanup(self)

    def create_table(self, complevel):
        filters = Filters(complevel=complevel, complib='blosc')
        self.table = self.fileh.create_table('/', 'sometable', self.dtype,
                                             filters=filters,
                                             chunkshape=self.chunkshape)

    def test_zero_length(self):
        complevel = 0
        self.create_table(complevel)
        self.assertEqual(self.table.size_on_disk, 0)
        self.assertEqual(self.table.size_in_memory, 0)

    # add 10 chunks of data in one append
    def test_no_compression_one_append(self):
        complevel = 0
        self.create_table(complevel)
        self.table.append([tuple(range(10))] * self.chunkshape[0] * 10)
        self.assertEqual(self.table.size_on_disk, 10 * 1000 * 10 * 4)
        self.assertEqual(self.table.size_in_memory, 10 * 1000 * 10 * 4)

    # add 10 chunks of data in two appends
    def test_no_compression_multiple_appends(self):
        complevel = 0
        self.create_table(complevel)
        self.table.append([tuple(range(10))] * self.chunkshape[0] * 5)
        self.table.append([tuple(range(10))] * self.chunkshape[0] * 5)
        self.assertEqual(self.table.size_on_disk, 10 * 1000 * 10 * 4)
        self.assertEqual(self.table.size_in_memory, 10 * 1000 * 10 * 4)

    def test_with_compression(self):
        complevel = 1
        self.create_table(complevel)
        self.table.append([tuple(range(10))] * self.chunkshape[0] * 10)
        file_size = os.stat(self.file).st_size
        self.assertTrue(
            abs(self.table.size_on_disk - file_size) <= self.hdf_overhead)
        self.assertEqual(self.table.size_in_memory, 10 * 1000 * 10 * 4)
        self.assertTrue(self.table.size_on_disk < self.table.size_in_memory)


class NonNestedTableReadTestCase(unittest.TestCase):
    def setUp(self):
        self.dtype = np.format_parser(['i4'] * 10, [], []).dtype
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, mode="w")
        self.table = self.fileh.create_table('/', 'table', self.dtype)
        self.shape = (100, )
        self.populate_file()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def populate_file(self):
        self.array = np.zeros(self.shape, self.dtype)
        for row_num, row in enumerate(self.array):
            start = row_num * len(self.array.dtype.names)
            for value, col in enumerate(self.array.dtype.names, start):
                row[col] = value
        self.table.append(self.array)
        self.assertEqual(len(self.table), len(self.array))

    def test_read_all(self):
        output = self.table.read()
        npt.assert_array_equal(output, self.array)

    def test_read_slice1(self):
        output = self.table.read(0, 51)
        npt.assert_array_equal(output, self.array[0:51])

    def test_read_all_rows_specified_field(self):
        output = self.table.read(field='f1')
        npt.assert_array_equal(output, self.array['f1'])

    def test_read_slice1_specified_field(self):
        output = self.table.read(1, 64, field='f1')
        npt.assert_array_equal(output, self.array['f1'][1:64])

    def test_out_arg_with_non_numpy_flavor(self):
        output = np.empty(self.shape, self.dtype)
        self.table.flavor = 'python'
        self.assertRaises(TypeError, lambda: self.table.read(out=output))
        try:
            self.table.read(out=output)
        except TypeError as exc:
            self.assertTrue("Optional 'out' argument may only be" in str(exc))

    def test_read_all_out_arg(self):
        output = np.empty(self.shape, self.dtype)
        self.table.read(out=output)
        npt.assert_array_equal(output, self.array)

    def test_read_slice1_out_arg(self):
        output = np.empty((51, ), self.dtype)
        self.table.read(0, 51, out=output)
        npt.assert_array_equal(output, self.array[0:51])

    def test_read_all_rows_specified_field_out_arg(self):
        output = np.empty(self.shape, 'i4')
        self.table.read(field='f1', out=output)
        npt.assert_array_equal(output, self.array['f1'])

    def test_read_slice1_specified_field_out_arg(self):
        output = np.empty((63, ), 'i4')
        self.table.read(1, 64, field='f1', out=output)
        npt.assert_array_equal(output, self.array['f1'][1:64])

    def test_read_all_out_arg_sliced(self):
        output = np.empty((200, ), self.dtype)
        output['f0'] = np.random.randint(0, 10000, (200, ))
        output_orig = output.copy()
        self.table.read(out=output[0:100])
        npt.assert_array_equal(output[0:100], self.array)
        npt.assert_array_equal(output[100:], output_orig[100:])

    def test_all_fields_non_contiguous_slice_contiguous_buffer(self):
        output = np.empty((50, ), self.dtype)
        self.table.read(0, 100, 2, out=output)
        npt.assert_array_equal(output, self.array[0:100:2])

    def test_specified_field_non_contiguous_slice_contiguous_buffer(self):
        output = np.empty((50, ), 'i4')
        self.table.read(0, 100, 2, field='f3', out=output)
        npt.assert_array_equal(output, self.array['f3'][0:100:2])

    def test_all_fields_non_contiguous_buffer(self):
        output = np.empty((100, ), self.dtype)
        output_slice = output[0:100:2]
        self.assertRaises(ValueError, self.table.read, 0, 100, 2, None,
                          output_slice)
        # once Python 2.6 support is dropped, this could change
        # to assertRaisesRegexp to check exception type and message at once
        try:
            self.table.read(0, 100, 2, field=None, out=output_slice)
        except ValueError as exc:
            self.assertEqual('output array not C contiguous', str(exc))

    def test_specified_field_non_contiguous_buffer(self):
        output = np.empty((100, ), 'i4')
        output_slice = output[0:100:2]
        self.assertRaises(ValueError, self.table.read, 0, 100, 2, 'f3',
                          output_slice)
        try:
            self.table.read(0, 100, 2, field='f3', out=output_slice)
        except ValueError as exc:
            self.assertEqual('output array not C contiguous', str(exc))

    def test_all_fields_buffer_too_small(self):
        output = np.empty((99, ), self.dtype)
        self.assertRaises(ValueError, lambda: self.table.read(out=output))
        try:
            self.table.read(out=output)
        except ValueError as exc:
            self.assertTrue('output array size invalid, got' in str(exc))

    def test_specified_field_buffer_too_small(self):
        output = np.empty((99, ), 'i4')
        func = lambda: self.table.read(field='f5', out=output)
        self.assertRaises(ValueError, func)
        try:
            self.table.read(field='f5', out=output)
        except ValueError as exc:
            self.assertTrue('output array size invalid, got' in str(exc))

    def test_all_fields_buffer_too_large(self):
        output = np.empty((101, ), self.dtype)
        self.assertRaises(ValueError, lambda: self.table.read(out=output))
        try:
            self.table.read(out=output)
        except ValueError as exc:
            self.assertTrue('output array size invalid, got' in str(exc))


class TableReadByteorderTestCase(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, mode="w")
        self.system_byteorder = sys.byteorder
        self.other_byteorder = {
            'little': 'big', 'big': 'little'}[sys.byteorder]
        self.reverse_byteorders = {'little': '<', 'big': '>'}

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def create_table(self, byteorder):
        table_dtype_code = self.reverse_byteorders[byteorder] + 'i4'
        table_dtype = np.format_parser([table_dtype_code, 'a1'], [], []).dtype
        self.table = self.fileh.create_table('/', 'table', table_dtype,
                                             byteorder=byteorder)
        input_dtype = np.format_parser(['i4', 'a1'], [], []).dtype
        self.input_array = np.zeros((10, ), input_dtype)
        self.input_array['f0'] = np.arange(10)
        self.input_array['f1'] = b'a'
        self.table.append(self.input_array)

    def test_table_system_byteorder_no_out_argument(self):
        self.create_table(self.system_byteorder)
        output = self.table.read()
        self.assertEqual(byteorders[output['f0'].dtype.byteorder],
                         self.system_byteorder)
        npt.assert_array_equal(output['f0'], np.arange(10))

    def test_table_other_byteorder_no_out_argument(self):
        self.create_table(self.other_byteorder)
        output = self.table.read()
        self.assertEqual(byteorders[output['f0'].dtype.byteorder],
                         self.system_byteorder)
        npt.assert_array_equal(output['f0'], np.arange(10))

    def test_table_system_byteorder_out_argument_system_byteorder(self):
        self.create_table(self.system_byteorder)
        out_dtype_code = self.reverse_byteorders[self.system_byteorder] + 'i4'
        out_dtype = np.format_parser([out_dtype_code, 'a1'], [], []).dtype
        output = np.empty((10, ), out_dtype)
        self.table.read(out=output)
        self.assertEqual(byteorders[output['f0'].dtype.byteorder],
                         self.system_byteorder)
        npt.assert_array_equal(output['f0'], np.arange(10))

    def test_table_other_byteorder_out_argument_system_byteorder(self):
        self.create_table(self.other_byteorder)
        out_dtype_code = self.reverse_byteorders[self.system_byteorder] + 'i4'
        out_dtype = np.format_parser([out_dtype_code, 'a1'], [], []).dtype
        output = np.empty((10, ), out_dtype)
        self.table.read(out=output)
        self.assertEqual(byteorders[output['f0'].dtype.byteorder],
                         self.system_byteorder)
        npt.assert_array_equal(output['f0'], np.arange(10))

    def test_table_system_byteorder_out_argument_other_byteorder(self):
        self.create_table(self.system_byteorder)
        out_dtype_code = self.reverse_byteorders[self.other_byteorder] + 'i4'
        out_dtype = np.format_parser([out_dtype_code, 'a1'], [], []).dtype
        output = np.empty((10, ), out_dtype)
        self.assertRaises(ValueError, lambda: self.table.read(out=output))
        try:
            self.table.read(out=output)
        except ValueError as exc:
            self.assertTrue("array must be in system's byteorder" in str(exc))

    def test_table_other_byteorder_out_argument_other_byteorder(self):
        self.create_table(self.other_byteorder)
        out_dtype_code = self.reverse_byteorders[self.other_byteorder] + 'i4'
        out_dtype = np.format_parser([out_dtype_code, 'a1'], [], []).dtype
        output = np.empty((10, ), out_dtype)
        self.assertRaises(ValueError, lambda: self.table.read(out=output))
        try:
            self.table.read(out=output)
        except ValueError as exc:
            self.assertTrue("array must be in system's byteorder" in str(exc))


class BasicRangeTestCase(unittest.TestCase):
    # file  = "test.h5"
    mode = "w"
    title = "This is the table title"
    record = Record
    maxshort = 1 << 15
    expectedrows = 100
    compress = 0
    shuffle = 1
    # Default values
    nrows = 20
    nrowsinbuf = 3  # Choose a small value for the buffer size
    start = 1
    stop = nrows
    checkrecarray = 0
    checkgetCol = 0

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        for j in range(3):
            # Create a table
            filterprops = Filters(complevel=self.compress,
                                  shuffle=self.shuffle)
            table = self.fileh.create_table(group, 'table'+str(j), self.record,
                                            title=self.title,
                                            filters=filterprops,
                                            expectedrows=self.expectedrows)
            # Get the row object associated with the new table
            row = table.row

            # Fill the table
            for i in xrange(self.expectedrows):
                row['var1'] = '%04d' % (self.expectedrows - i)
                row['var7'] = row['var1'][-1]
                row['var2'] = i
                row['var3'] = i % self.maxshort
                if isinstance(row['var4'], np.ndarray):
                    row['var4'] = [float(i), float(i * i)]
                else:
                    row['var4'] = float(i)
                if isinstance(row['var5'], np.ndarray):
                    row['var5'] = np.array((float(i),)*4)
                else:
                    row['var5'] = float(i)
                # var6 will be like var3 but byteswaped
                row['var6'] = ((row['var3'] >> 8) & 0xff) + \
                              ((row['var3'] << 8) & 0xff00)
                row.append()

            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.create_group(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2

    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def check_range(self):
        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        table = self.fileh.get_node("/table0")

        table.nrowsinbuf = self.nrowsinbuf
        r = slice(self.start, self.stop, self.step)
        resrange = r.indices(table.nrows)
        reslength = len(range(*resrange))
        #print "self.checkrecarray = ", self.checkrecarray
        #print "self.checkgetCol = ", self.checkgetCol
        if self.checkrecarray:
            recarray = table.read(self.start, self.stop, self.step)
            result = []
            for nrec in xrange(len(recarray)):
                if recarray['var2'][nrec] < self.nrows and 0 < self.step:
                    result.append(recarray['var2'][nrec])
                elif recarray['var2'][nrec] > self.nrows and 0 > self.step:
                    result.append(recarray['var2'][nrec])
        elif self.checkgetCol:
            column = table.read(self.start, self.stop, self.step, 'var2')
            result = []
            for nrec in xrange(len(column)):
                if column[nrec] < self.nrows and 0 < self.step:
                    result.append(column[nrec])
                elif column[nrec] > self.nrows and 0 > self.step:
                    result.append(column[nrec])
        else:
            if 0 < self.step:
                result = [rec['var2'] for rec in table.iterrows(self.start,
                            self.stop, self.step) if rec['var2'] < self.nrows]
            elif 0 > self.step:
                result = [rec['var2'] for rec in table.iterrows(self.start,
                            self.stop, self.step) if rec['var2'] > self.nrows]

        if self.start < 0:
            startr = self.expectedrows + self.start
        else:
            startr = self.start

        if self.stop is None:
            if self.checkrecarray or self.checkgetCol:
                # data read using the read method
                stopr = startr + 1
            else:
                # data read using the iterrows method
                stopr = self.nrows
        elif self.stop < 0:
            stopr = self.expectedrows + self.stop
        else:
            stopr = self.stop

        if self.nrows < stopr:
            stopr = self.nrows

        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            if reslength:
                if self.checkrecarray:
                    print "Last record *read* in recarray ==>", recarray[-1]
                elif self.checkgetCol:
                    print "Last value *read* in getCol ==>", column[-1]
                else:
                    print "Last record *read* in table range ==>", rec
            print "Total number of selected records ==>", len(result)
            print "Selected records:\n", result
            print "Selected records should look like:\n", \
                  range(startr, stopr, self.step)
            print "start, stop, step ==>", self.start, self.stop, self.step
            print "startr, stopr, step ==>", startr, stopr, self.step

        self.assertEqual(result, range(startr, stopr, self.step))
        if not (self.checkrecarray or self.checkgetCol):
            if startr < stopr and 0 < self.step:
                rec = [r for r in table.iterrows(self.start, self.stop, self.step)
                       if r['var2'] < self.nrows][-1]
                if self.nrows < self.expectedrows:
                    self.assertEqual(rec['var2'],
                                     range(self.start, self.stop, self.step)[-1])
                else:
                    self.assertEqual(rec['var2'],
                                     range(startr, stopr, self.step)[-1])
            elif startr > stopr and 0 > self.step:
                rec = [r['var2'] for r in table.iterrows(self.start, self.stop, self.step)
                       if r['var2'] > self.nrows][0]
                if self.nrows < self.expectedrows:
                    self.assertEqual(rec,
                                     range(self.start, self.stop or -1, self.step)[0])
                else:
                    self.assertEqual(rec,
                                     range(startr, stopr or -1, self.step)[0])

        # Close the file
        self.fileh.close()

    def test01_range(self):
        """Checking ranges in table iterators (case1)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 2 * step
        self.nrows = 21
        self.nrowsinbuf = 3
        self.start = 0
        self.stop = self.expectedrows
        self.step = 2

        self.check_range()

    def test01a_range(self):
        """Checking ranges in table iterators (case1)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 2 * step
        self.nrows = 21
        self.nrowsinbuf = 3
        self.start = self.expectedrows - 1
        self.stop = None
        self.step = -2

        self.check_range()

    def test02_range(self):
        """Checking ranges in table iterators (case2)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 10 * step
        self.nrows = 21
        self.nrowsinbuf = 31
        self.start = 11
        self.stop = self.expectedrows
        self.step = 3

        self.check_range()

    def test03_range(self):
        """Checking ranges in table iterators (case3)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 1.1 * step
        self.nrows = self.expectedrows
        self.nrowsinbuf = 11  # Choose a small value for the buffer size
        self.start = 0
        self.stop = self.expectedrows
        self.step = 10

        self.check_range()

    def test04_range(self):
        """Checking ranges in table iterators (case4)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_range..." % self.__class__.__name__

        # Case where step == nrowsinbuf
        self.nrows = self.expectedrows
        self.nrowsinbuf = 11  # Choose a small value for the buffer size
        self.start = 1
        self.stop = self.expectedrows
        self.step = 11

        self.check_range()

    def test05_range(self):
        """Checking ranges in table iterators (case5)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_range..." % self.__class__.__name__

        # Case where step > 1.1 * nrowsinbuf
        self.nrows = 21
        self.nrowsinbuf = 10  # Choose a small value for the buffer size
        self.start = 1
        self.stop = self.expectedrows
        self.step = 11

        self.check_range()

    def test06_range(self):
        """Checking ranges in table iterators (case6)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_range..." % self.__class__.__name__

        # Case where step > 3 * nrowsinbuf
        self.nrows = 3
        self.nrowsinbuf = 3  # Choose a small value for the buffer size
        self.start = 2
        self.stop = self.expectedrows
        self.step = 10

        self.check_range()

    def test07_range(self):
        """Checking ranges in table iterators (case7)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_range..." % self.__class__.__name__

        # Case where start == stop
        self.nrows = 2
        self.nrowsinbuf = 3  # Choose a small value for the buffer size
        self.start = self.nrows
        self.stop = self.nrows
        self.step = 10

        self.check_range()

    def test08_range(self):
        """Checking ranges in table iterators (case8)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_range..." % self.__class__.__name__

        # Case where start > stop
        self.nrows = 2
        self.nrowsinbuf = 3  # Choose a small value for the buffer size
        self.start = self.nrows + 1
        self.stop = self.nrows
        self.step = 1

        self.check_range()

    def test09_range(self):
        """Checking ranges in table iterators (case9)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_range..." % self.__class__.__name__

        # Case where stop = None (last row)
        self.nrows = 100
        self.nrowsinbuf = 3  # Choose a small value for the buffer size
        self.start = 1
        self.stop = 2
        self.step = 1

        self.check_range()

    def test10_range(self):
        """Checking ranges in table iterators (case10)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_range..." % self.__class__.__name__

        # Case where start < 0 and stop = None (last row)
        self.nrows = self.expectedrows
        self.nrowsinbuf = 5  # Choose a small value for the buffer size
        self.start = -6
        self.startr = self.expectedrows + self.start
        self.stop = -5
        self.stopr = self.expectedrows
        self.step = 2

        self.check_range()

    def test10a_range(self):
        """Checking ranges in table iterators (case10a)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test10a_range..." % self.__class__.__name__

        # Case where start < 0 and stop = 0
        self.nrows = self.expectedrows
        self.nrowsinbuf = 5  # Choose a small value for the buffer size
        self.start = -6
        self.startr = self.expectedrows + self.start
        self.stop = 0
        self.stopr = self.expectedrows
        self.step = 2

        self.check_range()

    def test11_range(self):
        """Checking ranges in table iterators (case11)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_range..." % self.__class__.__name__

        # Case where start < 0 and stop < 0
        self.nrows = self.expectedrows
        self.nrowsinbuf = 5  # Choose a small value for the buffer size
        self.start = -6
        self.startr = self.expectedrows + self.start
        self.stop = -2
        self.stopr = self.expectedrows + self.stop
        self.step = 1

        self.check_range()

    def test12_range(self):
        """Checking ranges in table iterators (case12)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test12_range..." % self.__class__.__name__

        # Case where start < 0 and stop < 0 and start > stop
        self.nrows = self.expectedrows
        self.nrowsinbuf = 5  # Choose a small value for the buffer size
        self.start = -1
        self.startr = self.expectedrows + self.start
        self.stop = -2
        self.stopr = self.expectedrows + self.stop
        self.step = 1

        self.check_range()

    def test13_range(self):
        """Checking ranges in table iterators (case13)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test13_range..." % self.__class__.__name__

        # Case where step < 0
        self.step = -11
        try:
            self.check_range()
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            self.fileh.close()
        #else:
        #    print rec
        #    self.fail("expected a ValueError")

        # Case where step == 0
        self.step = 0
        try:
            self.check_range()
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            self.fileh.close()
        #else:
        #    print rec
        #    self.fail("expected a ValueError")


class IterRangeTestCase(BasicRangeTestCase):
    pass


class RecArrayRangeTestCase(BasicRangeTestCase):
    checkrecarray = 1


class getColRangeTestCase(BasicRangeTestCase):
    checkgetCol = 1

    def test01_nonexistentField(self):
        """Checking non-existing Field in getCol method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_nonexistentField..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = open_file(self.file, "r")
        self.root = self.fileh.root
        table = self.fileh.get_node("/table0")

        try:
            # column = table.read(field='non-existent-column')
            table.col('non-existent-column')
        except KeyError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next KeyError was catched!"
                print value
            self.fileh.close()
        else:
            print rec
            self.fail("expected a KeyError")


class getItemTestCase(unittest.TestCase):
    mode = "w"
    title = "This is the table title"
    record = Record
    maxshort = 1 << 15
    expectedrows = 100
    compress = 0
    shuffle = 1
    # Default values
    nrows = 20
    nrowsinbuf = 3  # Choose a small value for the buffer size
    start = 1
    stop = nrows
    checkrecarray = 0
    checkgetCol = 0

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        for j in range(3):
            # Create a table
            filterprops = Filters(complevel=self.compress,
                                  shuffle=self.shuffle)
            table = self.fileh.create_table(group, 'table'+str(j), self.record,
                                            title=self.title,
                                            filters=filterprops,
                                            expectedrows=self.expectedrows)
            # Get the row object associated with the new table
            row = table.row

            # Fill the table
            for i in xrange(self.expectedrows):
                row['var1'] = '%04d' % (self.expectedrows - i)
                row['var7'] = row['var1'][-1]
                row['var2'] = i
                row['var3'] = i % self.maxshort
                if isinstance(row['var4'], np.ndarray):
                    row['var4'] = [float(i), float(i * i)]
                else:
                    row['var4'] = float(i)
                if isinstance(row['var5'], np.ndarray):
                    row['var5'] = np.array((float(i),)*4)
                else:
                    row['var5'] = float(i)
                # var6 will be like var3 but byteswaped
                row['var6'] = ((row['var3'] >> 8) & 0xff) + \
                              ((row['var3'] << 8) & 0xff00)
                row.append()

            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.create_group(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2

    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01a_singleItem(self):
        """Checking __getitem__ method with single parameter (int) """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a_singleItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[2]
        self.assertEqual(result["var2"], 2)
        result = table[25]
        self.assertEqual(result["var2"], 25)
        result = table[self.expectedrows-1]
        self.assertEqual(result["var2"], self.expectedrows - 1)

    def test01b_singleItem(self):
        """Checking __getitem__ method with single parameter (neg. int)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_singleItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[-5]
        self.assertEqual(result["var2"], self.expectedrows - 5)
        result = table[-1]
        self.assertEqual(result["var2"], self.expectedrows - 1)
        result = table[-self.expectedrows]
        self.assertEqual(result["var2"], 0)

    def test01c_singleItem(self):
        """Checking __getitem__ method with single parameter (long)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01c_singleItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[2]
        self.assertEqual(result["var2"], 2)
        result = table[25]
        self.assertEqual(result["var2"], 25)
        result = table[self.expectedrows-1]
        self.assertEqual(result["var2"], self.expectedrows - 1)

    def test01d_singleItem(self):
        """Checking __getitem__ method with single parameter (neg. long)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01d_singleItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[-5]
        self.assertEqual(result["var2"], self.expectedrows - 5)
        result = table[-1]
        self.assertEqual(result["var2"], self.expectedrows - 1)
        result = table[-self.expectedrows]
        self.assertEqual(result["var2"], 0)

    def test01e_singleItem(self):
        """Checking __getitem__ method with single parameter (rank-0 ints)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01e_singleItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[np.array(2)]
        self.assertEqual(result["var2"], 2)
        result = table[np.array(25)]
        self.assertEqual(result["var2"], 25)
        result = table[np.array(self.expectedrows-1)]
        self.assertEqual(result["var2"], self.expectedrows - 1)

    def test02_twoItems(self):
        """Checking __getitem__ method with start, stop parameters """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_twoItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[2:6]
        self.assertEqual(result["var2"].tolist(), range(2, 6))
        result = table[2:-6]
        self.assertEqual(result["var2"].tolist(), range(
            2, self.expectedrows-6))
        result = table[2:]
        self.assertEqual(result["var2"].tolist(), range(2, self.expectedrows))
        result = table[-2:]
        self.assertEqual(result["var2"].tolist(),
                         range(self.expectedrows-2, self.expectedrows))

    def test03_threeItems(self):
        """Checking __getitem__ method with start, stop, step parameters """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_threeItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        result = table[2:6:3]
        self.assertEqual(result["var2"].tolist(), range(2, 6, 3))
        result = table[2::3]
        self.assertEqual(result["var2"].tolist(), range(
            2, self.expectedrows, 3))
        result = table[:6:2]
        self.assertEqual(result["var2"].tolist(), range(0, 6, 2))
        result = table[::]
        self.assertEqual(result["var2"].tolist(), range(
            0, self.expectedrows, 1))

    def test04_negativeStep(self):
        """Checking __getitem__ method with negative step parameter"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_negativeStep..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        try:
            table[2:3:-3]
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
        else:
            self.fail("expected a ValueError")

    def test06a_singleItemCol(self):
        """Checking __getitem__ method in Col with single parameter """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06a_singleItemCol..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        colvar2 = table.cols.var2
        self.assertEqual(colvar2[2], 2)
        self.assertEqual(colvar2[25], 25)
        self.assertEqual(colvar2[self.expectedrows-1], self.expectedrows - 1)

    def test06b_singleItemCol(self):
        """Checking __getitem__ method in Col with single parameter
        (negative)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06b_singleItem..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        colvar2 = table.cols.var2
        self.assertEqual(colvar2[-5], self.expectedrows - 5)
        self.assertEqual(colvar2[-1], self.expectedrows - 1)
        self.assertEqual(colvar2[-self.expectedrows], 0)

    def test07_twoItemsCol(self):
        """Checking __getitem__ method in Col with start, stop parameters """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_twoItemCol..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        colvar2 = table.cols.var2
        self.assertEqual(colvar2[2:6].tolist(), range(2, 6))
        self.assertEqual(colvar2[2:-6].tolist(), range(2, self.expectedrows-6))
        self.assertEqual(colvar2[2:].tolist(), range(2, self.expectedrows))
        self.assertEqual(colvar2[-2:].tolist(),
                         range(self.expectedrows-2, self.expectedrows))

    def test08_threeItemsCol(self):
        """Checking __getitem__ method in Col with start, stop, step
        parameters"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_threeItemCol..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        colvar2 = table.cols.var2
        self.assertEqual(colvar2[2:6:3].tolist(), range(2, 6, 3))
        self.assertEqual(colvar2[2::3].tolist(), range(
            2, self.expectedrows, 3))
        self.assertEqual(colvar2[:6:2].tolist(), range(0, 6, 2))
        self.assertEqual(colvar2[::].tolist(), range(0, self.expectedrows, 1))

    def test09_negativeStep(self):
        """Checking __getitem__ method in Col with negative step parameter"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_negativeStep..." % self.__class__.__name__

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        colvar2 = table.cols.var2
        try:
            colvar2[2:3:-3]
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
        else:
            self.fail("expected a ValueError")

    def test10_list_integers(self):
        """Checking accessing Table with a list of integers"""

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        idx = list(range(10, 70, 11))

        result = table[idx]
        self.assertEqual(result["var2"].tolist(), idx)

        result = table.read_coordinates(idx)
        self.assertEqual(result["var2"].tolist(), idx)

    def test11_list_booleans(self):
        """Checking accessing Table with a list of boolean values"""

        self.fileh = open_file(self.file, "r")
        table = self.fileh.root.table0
        idx = list(range(10, 70, 11))

        selection = [n in idx for n in range(self.expectedrows)]

        result = table[selection]
        self.assertEqual(result["var2"].tolist(), idx)

        result = table.read_coordinates(selection)
        self.assertEqual(result["var2"].tolist(), idx)


class Rec(IsDescription):
    col1 = IntCol(pos=1)
    col2 = StringCol(itemsize=3, pos=2)
    col3 = FloatCol(pos=3)


class setItem(common.PyTablesTestCase):
    def tearDown(self):
        self.fileh.close()
        # del self.fileh, self.rootgroup
        os.remove(self.file)
        common.cleanup(self)

    def test01(self):
        "Checking modifying one table row with __setitem__"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing row
        table[2] = (456, 'db2', 1.2)
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [456, b'db2', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test01b(self):
        "Checking modifying one table row with __setitem__ (long index)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing row
        table[2] = (456, 'db2', 1.2)
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [456, b'db2', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test02(self):
        "Modifying one row, with a step (__setitem__)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify two existing rows
        rows = records.array([[457, b'db1', 1.2]], formats="i4,a3,f8")
        table[1:3:2] = rows
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test03(self):
        "Checking modifying several rows at once (__setitem__)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify two existing rows
        rows = records.array([[457, b'db1', 1.2], [5, b'de1', 1.3]],
                             formats="i4,a3,f8")
        # table.modify_rows(start=1, rows=rows)
        table[1:3] = rows
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [5, b'de1', 1.3], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test04(self):
        "Modifying several rows at once, with a step (__setitem__)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify two existing rows
        rows = records.array([[457, b'db1', 1.2], [6, b'de2', 1.3]],
                             formats="i4,a3,f8")
        # table[1:4:2] = rows
        table[1::2] = rows
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [457, b'db1', 1.2], [6, b'de2', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test05(self):
        "Checking modifying one column (single element, __setitem__)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3]],
                          formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        table.cols.col1[1] = -1
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [-1, b'ded', 1.3],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test06a(self):
        "Checking modifying one column (several elements, __setitem__)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        table.cols.col1[1:4] = [2, 3, 4]
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [3, b'db1', 1.2], [4, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test06b(self):
        "Checking modifying one column (iterator, __setitem__)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        try:
            for row in table.iterrows():
                row['col1'] = row.nrow + 1
                row.append()
            table.flush()
        except NotImplementedError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NotImplementedError was catched!"
                print value
        else:
            self.fail("expected a NotImplementedError")

#         # Create the modified recarray
#         r1=records.array([[1,b'dbe',1.2],[2,b'ded',1.3],
#                           [3,b'db1',1.2],[4,b'de1',1.3]],
#                          formats="i4,a3,f8",
#                          names = "col1,col2,col3")
#         # Read the modified table
#         if self.reopen:
#             self.fileh.close()
#             self.fileh = open_file(self.file, "r")
#             table = self.fileh.root.recarray
#             table.nrowsinbuf = self.buffersize  # set buffer value
#         r2 = table.read()
#         if common.verbose:
#             print "Original table-->", repr(r2)
#             print "Should look like-->", repr(r1)
#         self.assertEqual(r1.tostring(), r2.tostring())
#         self.assertEqual(table.nrows, 4)

    def test07(self):
        "Modifying one column (several elements, __setitem__, step)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          1, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify just one existing column
        table.cols.col1[1:4:2] = [2, 3]
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [457, b'db1', 1.2], [3, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test08(self):
        "Modifying one column (one element, __setitem__, step)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        table.cols.col1[1:4:3] = [2]
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test09(self):
        "Modifying beyond the table extend (__setitem__, step)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Try to modify beyond the extend
        # This will silently exclude the non-fitting rows
        rows = records.array([[457, b'db1', 1.2], [6, b'de2', 1.3]],
                             formats="i4,a3,f8")
        table[1::2] = rows
        # How it should look like
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [457, b'db1', 1.2], [6, b'de2', 1.3]],
                           formats="i4,a3,f8")

        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)


class setItem1(setItem):
    reopen = 0
    buffersize = 1


class setItem2(setItem):
    reopen = 1
    buffersize = 2


class setItem3(setItem):
    reopen = 0
    buffersize = 1000


class setItem4(setItem):
    reopen = 1
    buffersize = 1000


class updateRow(common.PyTablesTestCase):
    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test01(self):
        "Checking modifying one table row with Row.update"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing row
        for row in table.iterrows(2, 3):
            (row['col1'], row['col2'], row['col3']) = [456, 'db2', 1.2]
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [456, b'db2', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test02(self):
        "Modifying one row, with a step (Row.update)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify two existing rows
        for row in table.iterrows(1, 3, 2):
            if row.nrow == 1:
                (row['col1'], row['col2'], row['col3']) = [457, 'db1', 1.2]
            elif row.nrow == 3:
                (row['col1'], row['col2'], row['col3']) = [6, 'de2', 1.3]
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test03(self):
        "Checking modifying several rows at once (Row.update)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify two existing rows
        for row in table.iterrows(1, 3):
            if row.nrow == 1:
                (row['col1'], row['col2'], row['col3']) = [457, 'db1', 1.2]
            elif row.nrow == 2:
                (row['col1'], row['col2'], row['col3']) = [5, 'de1', 1.3]
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [5, b'de1', 1.3], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test04(self):
        "Modifying several rows at once, with a step (Row.update)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify two existing rows
        for row in table.iterrows(1, stop=4, step=2):
            if row.nrow == 1:
                (row['col1'], row['col2'], row['col3']) = [457, 'db1', 1.2]
            elif row.nrow == 3:
                (row['col1'], row['col2'], row['col3']) = [6, 'de2', 1.3]
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [457, b'db1', 1.2], [6, b'de2', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test05(self):
        "Checking modifying one column (single element, Row.update)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3]],
                          formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        for row in table.iterrows(1, 2):
            row['col1'] = -1
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [-1, b'ded', 1.3],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test06(self):
        "Checking modifying one column (several elements, Row.update)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        for row in table.iterrows(1, 4):
            row['col1'] = row.nrow + 1
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [3, b'db1', 1.2], [4, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test07(self):
        "Modifying values from a selection"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          1, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify just rows with col1 < 456
        for row in table.where('col1 < 456'):
            row['col1'] = 2
            row['col2'] = 'ada'
            row.update()
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ada', 1.3],
                            [457, b'db1', 1.2], [2, b'ada', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

    def test08(self):
        "Modifying a large table (Row.update)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        nrows = 100
        # append new rows
        row = table.row
        for i in xrange(nrows):
            row['col1'] = i-1
            row['col2'] = 'a'+str(i-1)
            row['col3'] = -1.0
            row.append()
        table.flush()

        # Modify all the rows
        for row in table:
            row['col1'] = row.nrow
            row['col2'] = 'b'+str(row.nrow)
            row['col3'] = 0.0
            row.update()

        # Create the modified recarray
        r1 = records.array(None, shape=nrows,
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        for i in xrange(nrows):
            r1['col1'][i] = i
            r1['col2'][i] = 'b'+str(i)
            r1['col3'][i] = 0.0
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, nrows)

    def test08b(self):
        "Setting values on a large table without calling Row.update"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        nrows = 100
        # append new rows
        row = table.row
        for i in xrange(nrows):
            row['col1'] = i-1
            row['col2'] = 'a'+str(i-1)
            row['col3'] = -1.0
            row.append()
        table.flush()

        # Modify all the rows (actually don't)
        for row in table:
            row['col1'] = row.nrow
            row['col2'] = 'b'+str(row.nrow)
            row['col3'] = 0.0
            # row.update()

        # Create the modified recarray
        r1 = records.array(None, shape=nrows,
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        for i in xrange(nrows):
            r1['col1'][i] = i-1
            r1['col2'][i] = 'a'+str(i-1)
            r1['col3'][i] = -1.0
        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, nrows)

    def test09(self):
        "Modifying selected values on a large table"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        nrows = 100
        # append new rows
        row = table.row
        for i in xrange(nrows):
            row['col1'] = i-1
            row['col2'] = 'a'+str(i-1)
            row['col3'] = -1.0
            row.append()
        table.flush()

        # Modify selected rows
        for row in table.where('col1 > nrows-3'):
            row['col1'] = row.nrow
            row['col2'] = 'b'+str(row.nrow)
            row['col3'] = 0.0
            row.update()

        # Create the modified recarray
        r1 = records.array(None, shape=nrows,
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        for i in xrange(nrows):
            r1['col1'][i] = i-1
            r1['col2'][i] = 'a'+str(i-1)
            r1['col3'][i] = -1.0
        # modify just the last line
        r1['col1'][i] = i
        r1['col2'][i] = 'b'+str(i)
        r1['col3'][i] = 0.0

        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, nrows)

    def test09b(self):
        "Modifying selected values on a large table (alternate values)"

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")

        # Create a new table:
        table = self.fileh.create_table(self.fileh.root, 'recarray', Rec)
        table.nrowsinbuf = self.buffersize  # set buffer value

        nrows = 100
        # append new rows
        row = table.row
        for i in xrange(nrows):
            row['col1'] = i-1
            row['col2'] = 'a'+str(i-1)
            row['col3'] = -1.0
            row.append()
        table.flush()

        # Modify selected rows
        for row in table.iterrows(step=10):
            row['col1'] = row.nrow
            row['col2'] = 'b'+str(row.nrow)
            row['col3'] = 0.0
            row.update()

        # Create the modified recarray
        r1 = records.array(None, shape=nrows,
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        for i in xrange(nrows):
            if i % 10 > 0:
                r1['col1'][i] = i-1
                r1['col2'][i] = 'a'+str(i-1)
                r1['col3'][i] = -1.0
            else:
                r1['col1'][i] = i
                r1['col2'][i] = 'b'+str(i)
                r1['col3'][i] = 0.0

        # Read the modified table
        if self.reopen:
            self.fileh.close()
            self.fileh = open_file(self.file, "r")
            table = self.fileh.root.recarray
            table.nrowsinbuf = self.buffersize  # set buffer value
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, nrows)


class updateRow1(updateRow):
    reopen = 0
    buffersize = 1


class updateRow2(updateRow):
    reopen = 1
    buffersize = 2


class updateRow3(updateRow):
    reopen = 0
    buffersize = 1000


class updateRow4(updateRow):
    reopen = 1
    buffersize = 1000


class RecArrayIO(unittest.TestCase):
    def test00(self):
        "Checking saving a regular recarray"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')

        # Save it in a table:
        fileh.create_table(fileh.root, 'recarray', r)

        # Read it again
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
        r2 = fileh.root.recarray.read()
        self.assertEqual(r.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)

    def test01(self):
        "Checking saving a recarray with an offset in its buffer"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')

        # Get an offsetted bytearray
        r1 = r[1:]

        # Save it in a table:
        fileh.create_table(fileh.root, 'recarray', r1)

        # Read it again
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
        r2 = fileh.root.recarray.read()

        self.assertEqual(r1.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)

    def test02(self):
        "Checking saving a large recarray with an offset in its buffer"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array(b'a'*200000, 'f4,3i4,a5,i2', 3000)

        # Get an offsetted bytearray
        r1 = r[2000:]

        # Save it in a table:
        fileh.create_table(fileh.root, 'recarray', r1)

        # Read it again
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
        r2 = fileh.root.recarray.read()

        self.assertEqual(r1.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)

    def test03(self):
        "Checking saving a strided recarray with an offset in its buffer"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array(b'a'*200000, 'f4,3i4,a5,i2', 3000)

        # Get an strided recarray
        r2 = r[::2]

        # Get an offsetted bytearray
        r1 = r2[1200:]

        # Save it in a table:
        fileh.create_table(fileh.root, 'recarray', r1)

        # Read it again
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
        r2 = fileh.root.recarray.read()

        self.assertEqual(r1.tostring(), r2.tostring())
        fileh.close()
        os.remove(file)

    def test04(self):
        "Checking appending several rows at once"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        class Rec(IsDescription):
            col1 = IntCol(pos=1)
            col2 = StringCol(itemsize=3, pos=2)
            col3 = FloatCol(pos=3)

        # Save it in a table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Create the complete table
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the original table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = fileh.root.recarray.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test05(self):
        "Checking appending several rows at once (close file version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Save it in a table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        fileh.close()
        fileh = open_file(file, "r")
        table = fileh.root.recarray
        # Create the complete table
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the original table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = fileh.root.recarray.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test06a(self):
        "Checking modifying one table row (list version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06a..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([(456, b'dbe', 1.2), (
            2, b'ded', 1.3)], formats="i4,a3,f8")
        table.append(r)
        table.append([(457, b'db1', 1.2), (5, b'de1', 1.3)])
        # Modify just one existing rows
        table.modify_rows(start=1, rows=[(456, 'db1', 1.2)])
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [456, b'db1', 1.2],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test06b(self):
        "Checking modifying one table row (recarray version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06b..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify just one existing rows
        table.modify_rows(start=2, rows=records.array([[456, 'db2', 1.2]],
                                                      formats="i4,a3,f8"))
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [456, b'db2', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test07a(self):
        "Checking modifying several rows at once (list version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07a..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify two existing rows
        table.modify_rows(start=1, rows=[(457, 'db1', 1.2), (5, 'de1', 1.3)])
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [5, b'de1', 1.3], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test07b(self):
        "Checking modifying several rows at once (recarray version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07b..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify two existing rows
        rows = records.array([[457, b'db1', 1.2], [5, b'de1', 1.3]],
                             formats="i4,a3,f8")
        table.modify_rows(start=1, rows=rows)
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [457, b'db1', 1.2],
                            [5, b'de1', 1.3], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test07c(self):
        "Checking modifying several rows with a mismatching value"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07c..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify two existing rows
        rows = records.array([[457, b'db1', 1.2], [5, b'de1', 1.3]],
                             formats="i4,a3,f8")
        self.assertRaises(ValueError, table.modify_rows,
                          start=1, stop=2, rows=rows)
        fileh.close()
        os.remove(file)

    def test08a(self):
        "Checking modifying one column (single column version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08a..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        table.modify_columns(start=1, columns=[[2, 3, 4]], names=["col1"])
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [3, b'db1', 1.2], [4, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test08a2(self):
        "Checking modifying one column (single column version, modify_column)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08a2..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        table.modify_column(start=1, column=[2, 3, 4], colname="col1")
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [3, b'db1', 1.2], [4, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test08b(self):
        "Checking modifying one column (single column version, recarray)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08b..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        columns = records.fromarrays(np.array([[2, 3, 4]]), formats="i4")
        table.modify_columns(start=1, columns=columns, names=["col1"])
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [3, b'db1', 1.2], [4, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test08b2(self):
        """Checking modifying one column (single column version, recarray,
        modify_column)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08b2..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        columns = records.fromarrays(np.array([[2, 3, 4]]), formats="i4")
        table.modify_column(start=1, column=columns, colname="col1")
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'ded', 1.3],
                            [3, b'db1', 1.2], [4, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test08c(self):
        "Checking modifying one column (single column version, single element)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08c..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify just one existing column
        # columns = records.fromarrays(np.array([[4]]), formats="i4")
        # table.modify_columns(start=1, columns=columns, names=["col1"])
        table.modify_columns(start=1, columns=[[4]], names=["col1"])
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [4, b'ded', 1.3],
                            [457, b'db1', 1.2], [5, b'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test09a(self):
        "Checking modifying table columns (multiple column version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09a..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify a couple of columns
        columns = [["aaa", "bbb", "ccc"], [1.2, .1, .3]]
        table.modify_columns(start=1, columns=columns, names=["col2", "col3"])
        # Create the modified recarray
        r1 = records.array([[456, b'dbe', 1.2], [2, b'aaa', 1.2],
                            [457, b'bbb', .1], [5, b'ccc', .3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")

        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test09b(self):
        "Checking modifying table columns (multiple columns, recarray)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09b..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify a couple of columns
        columns = records.array([["aaa", 1.2], ["bbb", .1], ["ccc", .3]],
                                formats="a3,f8")
        table.modify_columns(start=1, columns=columns, names=["col2", "col3"])
        # Create the modified recarray
        r1 = records.array([[456, 'dbe', 1.2], [2, 'aaa', 1.2],
                            [457, 'bbb', .1], [5, 'ccc', .3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test09c(self):
        "Checking modifying table columns (single column, step)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09c..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])
        # Modify a couple of columns
        columns = records.array([["aaa", 1.2], ["bbb", .1]],
                                formats="a3,f8")
        table.modify_columns(start=1, step=2, columns=columns,
                             names=["col2", "col3"])
        # Create the modified recarray
        r1 = records.array([[456, 'dbe', 1.2], [2, 'aaa', 1.2],
                            [457, 'db1', 1.2], [5, 'bbb', .1]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test09d(self):
        "Checking modifying table columns (multiple columns, step)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09d..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        # Modify a couple of columns
        columns = records.array([["aaa", 1.3], ["bbb", .1]],
                                formats="a3,f8")
        table.modify_columns(start=0, step=2, columns=columns,
                             names=["col2", "col3"])
        # Create the modified recarray
        r1 = records.array([[456, 'aaa', 1.3], [2, 'ded', 1.3],
                            [457, 'bbb', .1], [5, 'de1', 1.3]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test10a(self):
        "Checking modifying rows using coordinates (readCoords/modifyCoords)."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test10a..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        columns = table.read_coordinates([0, 3])

        # Modify both rows
        columns['col1'][:] = [55, 56]
        columns['col3'][:] = [1.9, 1.8]

        # Modify the table in the same coordinates
        table.modify_coordinates([0, 3], columns)

        # Create the modified recarray
        r1 = records.array([[55, b'dbe', 1.9], [2, b'ded', 1.3],
                            [457, b'db1', 1.2], [56, b'de1', 1.8]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)

    def test10b(self):
        "Checking modifying rows using coordinates (getitem/setitem)."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test10b..." % self.__class__.__name__

        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a new table:
        table = fileh.create_table(fileh.root, 'recarray', Rec)

        # append new rows
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'ded', 1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457, b'db1', 1.2], [5, b'de1', 1.3]])

        columns = table[[0, 3]]

        # Modify both rows
        columns['col1'][:] = [55, 56]
        columns['col3'][:] = [1.9, 1.8]

        # Modify the table in the same coordinates
        table[[0, 3]] = columns

        # Create the modified recarray
        r1 = records.array([[55, b'dbe', 1.9], [2, b'ded', 1.3],
                            [457, b'db1', 1.2], [56, b'de1', 1.8]],
                           formats="i4,a3,f8",
                           names="col1,col2,col3")
        # Read the modified table
        if self.reopen:
            fileh.close()
            fileh = open_file(file, "r")
            table = fileh.root.recarray
        r2 = table.read()
        if common.verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        self.assertEqual(r1.tostring(), r2.tostring())
        self.assertEqual(table.nrows, 4)

        fileh.close()
        os.remove(file)


class RecArrayIO1(RecArrayIO):
    reopen = 0


class RecArrayIO2(RecArrayIO):
    reopen = 1


class CopyTestCase(unittest.TestCase):
    def assertEqualColinstances(self, table1, table2):
        """Assert that column instance maps of both tables are equal."""
        cinst1, cinst2 = table1.colinstances, table2.colinstances
        self.assertEqual(len(cinst1), len(cinst2))
        for (cpathname, col1) in cinst1.iteritems():
            self.assertTrue(cpathname in cinst2)
            col2 = cinst2[cpathname]
            self.assertTrue(isinstance(col1, type(col2)))
            if isinstance(col1, Column):
                self.assertEqual(col1.name, col2.name)
                self.assertEqual(col1.pathname, col2.pathname)
                self.assertEqual(col1.dtype, col2.dtype)
                self.assertEqual(col1.type, col2.type)
            elif isinstance(col1, Cols):
                self.assertEqual(col1._v_colnames, col2._v_colnames)
                self.assertEqual(col1._v_colpathnames, col2._v_colpathnames)

    def test01_copy(self):
        """Checking Table.copy() method """

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table
        table2 = table1.copy('/', 'table2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1
            table2 = fileh.root.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            # print "dirs-->", dir(table1), dir(table2)
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow   # current row
            # row1 is a Row instance, while table2[] is a
            # RecArray.Record instance
            # print "reprs-->", repr(row1), repr(table2.read(nrow))
            for colname in table1.colnames:
                # Both ways to compare works well
                # self.assertEqual(row1[colname], table2[nrow][colname))
                self.assertEqual(row1[colname],
                                 table2.read(nrow, field=colname)[0])

        # Assert other properties in table
        self.assertEqual(table1.nrows, table2.nrows)
        self.assertEqual(table1.shape, table2.shape)
        self.assertEqual(table1.colnames, table2.colnames)
        self.assertEqual(table1.coldtypes, table2.coldtypes)
        self.assertEqualColinstances(table1, table2)
        self.assertEqual(repr(table1.description), repr(table2.description))

        # This could be not the same when re-opening the file
        # self.assertEqual(table1.description._v_ColObjects,
        #                  table2.description._v_ColObjects)
        # Leaf attributes
        self.assertEqual(table1.title, table2.title)
        self.assertEqual(table1.filters.complevel, table2.filters.complevel)
        self.assertEqual(table1.filters.complib, table2.filters.complib)
        self.assertEqual(table1.filters.shuffle, table2.filters.shuffle)
        self.assertEqual(table1.filters.fletcher32, table2.filters.fletcher32)

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking Table.copy() method (where specified)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table in another group
        group1 = fileh.create_group("/", "group1")
        table2 = table1.copy(group1, 'table2')

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow   # current row
            for colname in table1.colnames:
                # Both ways to compare works well
                # self.assertEqual(row1[colname], table2[nrow][colname))
                self.assertEqual(row1[colname],
                                 table2.read(nrow, field=colname)[0])

        # Assert other properties in table
        self.assertEqual(table1.nrows, table2.nrows)
        self.assertEqual(table1.shape, table2.shape)
        self.assertEqual(table1.colnames, table2.colnames)
        self.assertEqual(table1.coldtypes, table2.coldtypes)
        self.assertEqualColinstances(table1, table2)
        self.assertEqual(repr(table1.description), repr(table2.description))

        # Leaf attributes
        self.assertEqual(table1.title, table2.title)
        self.assertEqual(table1.filters.complevel, table2.filters.complevel)
        self.assertEqual(table1.filters.complib, table2.filters.complib)
        self.assertEqual(table1.filters.shuffle, table2.filters.shuffle)
        self.assertEqual(table1.filters.fletcher32, table2.filters.fletcher32)

        # Close the file
        fileh.close()
        os.remove(file)

    def test03_copy(self):
        """Checking Table.copy() method (table larger than buffer)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray exceeding buffers capability
        # This works, but takes too much CPU for a test
        # It is better to reduce the buffer size (table1.nrowsinbuf)
#         r=records.array(b'aaaabbbbccccddddeeeeffffgggg'*20000,
#                         formats='2i2,i4, (2,3)u2, (1,)f4, f8',shape=700)
        r = records.array(b'aaaabbbbccccddddeeeeffffgggg' * 200,
                          formats='2i2,i4, (2,3)u2, (1,)f4, f8', shape=7)
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table in another group and other title
        group1 = fileh.create_group("/", "group1")
        table1.nrowsinbuf = 2  # small value of buffer
        table2 = table1.copy(group1, 'table2', title="title table2")
        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow   # current row
            for colname in table1.colnames:
                # self.assertTrue(allequal(row1[colname],
                # table2[nrow][colname]))
                self.assertTrue(allequal(row1[colname],
                                         table2.read(nrow, field=colname)[0]))

        # Assert other properties in table
        self.assertEqual(table1.nrows, table2.nrows)
        self.assertEqual(table1.shape, table2.shape)
        self.assertEqual(table1.colnames, table2.colnames)
        self.assertEqual(table1.coldtypes, table2.coldtypes)
        self.assertEqualColinstances(table1, table2)
        self.assertEqual(repr(table1.description), repr(table2.description))

        # Leaf attributes
        self.assertEqual("title table2", table2.title)
        self.assertEqual(table1.filters.complevel, table2.filters.complevel)
        self.assertEqual(table1.filters.complib, table2.filters.complib)
        self.assertEqual(table1.filters.shuffle, table2.filters.shuffle)
        self.assertEqual(table1.filters.fletcher32, table2.filters.fletcher32)

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking Table.copy() method (different compress level)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table in another group
        group1 = fileh.create_group("/", "group1")
        table2 = table1.copy(group1, 'table2',
                             filters=Filters(complevel=6))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow   # current row
            for colname in table1.colnames:
                # Both ways to compare works well
                # self.assertEqual(row1[colname], table2[nrow][colname))
                self.assertEqual(row1[colname],
                                 table2.read(nrow, field=colname)[0])

        # Assert other properties in table
        self.assertEqual(table1.nrows, table2.nrows)
        self.assertEqual(table1.shape, table2.shape)
        self.assertEqual(table1.colnames, table2.colnames)
        self.assertEqual(table1.coldtypes, table2.coldtypes)
        self.assertEqualColinstances(table1, table2)
        self.assertEqual(repr(table1.description), repr(table2.description))

        # Leaf attributes
        self.assertEqual(table1.title, table2.title)
        self.assertEqual(6, table2.filters.complevel)
        self.assertEqual(1, table2.filters.shuffle)
        self.assertEqual(table1.filters.fletcher32, table2.filters.fletcher32)

        # Close the file
        fileh.close()
        os.remove(file)

    def test05_copy(self):
        """Checking Table.copy() method (user attributes copied)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")
        # Add some user attributes
        table1.attrs.attr1 = "attr1"
        table1.attrs.attr2 = 2

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table in another group
        group1 = fileh.create_group("/", "group1")
        table2 = table1.copy(group1, 'table2',
                             copyuserattrs=1,
                             filters=Filters(complevel=6))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow   # current row
            for colname in table1.colnames:
                # self.assertEqual(row1[colname], table2[nrow][colname))
                self.assertEqual(row1[colname],
                                 table2.read(nrow, field=colname)[0])

        # Assert other properties in table
        self.assertEqual(table1.nrows, table2.nrows)
        self.assertEqual(table1.shape, table2.shape)
        self.assertEqual(table1.colnames, table2.colnames)
        self.assertEqual(table1.coldtypes, table2.coldtypes)
        self.assertEqualColinstances(table1, table2)
        self.assertEqual(repr(table1.description), repr(table2.description))

        # Leaf attributes
        self.assertEqual(table1.title, table2.title)
        self.assertEqual(6, table2.filters.complevel)
        self.assertEqual(1, table2.filters.shuffle)
        self.assertEqual(table1.filters.fletcher32, table2.filters.fletcher32)
        # User attributes
        self.assertEqual(table2.attrs.attr1, "attr1")
        self.assertEqual(table2.attrs.attr2, 2)

        # Close the file
        fileh.close()
        os.remove(file)

    def test05b_copy(self):
        """Checking Table.copy() method (user attributes not copied)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[456, b'dbe', 1.2], [
                          2, b'de', 1.3]], names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")
        # Add some user attributes
        table1.attrs.attr1 = "attr1"
        table1.attrs.attr2 = 2

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table in another group
        group1 = fileh.create_group("/", "group1")
        table2 = table1.copy(group1, 'table2',
                             copyuserattrs=0,
                             filters=Filters(complevel=6))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow   # current row
            for colname in table1.colnames:
                # self.assertEqual(row1[colname], table2[nrow][colname))
                self.assertEqual(row1[colname],
                                 table2.read(nrow, field=colname)[0])

        # Assert other properties in table
        self.assertEqual(table1.nrows, table2.nrows)
        self.assertEqual(table1.shape, table2.shape)
        self.assertEqual(table1.colnames, table2.colnames)
        self.assertEqual(table1.coldtypes, table2.coldtypes)
        self.assertEqualColinstances(table1, table2)
        self.assertEqual(repr(table1.description), repr(table2.description))

        # Leaf attributes
        self.assertEqual(table1.title, table2.title)
        self.assertEqual(6, table2.filters.complevel)
        self.assertEqual(1, table2.filters.shuffle)
        self.assertEqual(table1.filters.fletcher32, table2.filters.fletcher32)
        # User attributes
#       self.assertEqual(table2.attrs.attr1, None)
#       self.assertEqual(table2.attrs.attr2, None)
        self.assertEqual(hasattr(table2.attrs, "attr1"), 0)
        self.assertEqual(hasattr(table2.attrs, "attr2"), 0)

        # Close the file
        fileh.close()
        os.remove(file)


class CloseCopyTestCase(CopyTestCase):
    close = 1


class OpenCopyTestCase(CopyTestCase):
    close = 0


class CopyIndexTestCase(unittest.TestCase):
    def test01_index(self):
        """Checking Table.copy() method with indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray exceeding buffers capability
        r = records.array(b'aaaabbbbccccddddeeeeffffgggg' * 200,
                          formats='2i2, (1,)i4, (2,3)u2, (1,)f4, (1,)f8',
                          shape=10)
                          # The line below exposes a bug in numpy
                          # formats='2i2, i4, (2,3)u2, f4, f8',shape=10)
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table
        table1.nrowsinbuf = self.nrowsinbuf
        table2 = table1.copy("/", 'table2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        for nrow in range(r2.shape[0]):
            for colname in table1.colnames:
                self.assertTrue(allequal(r2[nrow][colname],
                                         table2[nrow][colname]))

        # Assert the number of rows in table
        if common.verbose:
            print "nrows in table2-->", table2.nrows
            print "and it should be-->", r2.shape[0]
        self.assertEqual(r2.shape[0], table2.nrows)

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_indexclosef(self):
        """Checking Table.copy() method with indexes (close file version)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_indexclosef..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray exceeding buffers capability
        r = records.array(b'aaaabbbbccccddddeeeeffffgggg' * 200,
                          formats='2i2, i4, (2,3)u2, f4, f8', shape=10)
        # Save it in a table:
        table1 = fileh.create_table(fileh.root, 'table1', r, "title table1")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            fileh.close()
            fileh = open_file(file, mode="a")
            table1 = fileh.root.table1

        # Copy to another table
        table1.nrowsinbuf = self.nrowsinbuf
        table2 = table1.copy("/", 'table2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)

        fileh.close()
        fileh = open_file(file, mode="r")
        table1 = fileh.root.table1
        table2 = fileh.root.table2

        if common.verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        for nrow in range(r2.shape[0]):
            for colname in table1.colnames:
                self.assertTrue(allequal(r2[nrow][colname],
                                         table2[nrow][colname]))

        # Assert the number of rows in table
        if common.verbose:
            print "nrows in table2-->", table2.nrows
            print "and it should be-->", r2.shape[0]
        self.assertEqual(r2.shape[0], table2.nrows)

        # Close the file
        fileh.close()
        os.remove(file)


class CopyIndex1TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 1
    start = 0
    stop = 7
    step = 1


class CopyIndex2TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 0
    start = 0
    stop = -1
    step = 1


class CopyIndex3TestCase(CopyIndexTestCase):
    nrowsinbuf = 3
    close = 1
    start = 1
    stop = 7
    step = 1


class CopyIndex4TestCase(CopyIndexTestCase):
    nrowsinbuf = 4
    close = 0
    start = 0
    stop = 6
    step = 1


class CopyIndex5TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 1
    start = 3
    stop = 7
    step = 1


class CopyIndex6TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 0
    start = 3
    stop = 6
    step = 2


class CopyIndex7TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 1
    start = 0
    stop = 7
    step = 10


class CopyIndex8TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 0
    start = 6
    stop = 3
    step = 1


class CopyIndex9TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 1
    start = 3
    stop = 4
    step = 1


class CopyIndex10TestCase(CopyIndexTestCase):
    nrowsinbuf = 1
    close = 0
    start = 3
    stop = 4
    step = 2


class CopyIndex11TestCase(CopyIndexTestCase):
    nrowsinbuf = 2
    close = 1
    start = -3
    stop = -1
    step = 2


class CopyIndex12TestCase(CopyIndexTestCase):
    nrowsinbuf = 3
    close = 0
    start = -1   # Should point to the last element
    stop = None  # None should mean the last element (including it)
    step = 1


class LargeRowSize(unittest.TestCase):
    def test00(self):
        "Checking saving a Table with a moderately large rowsize"
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[np.arange(100)]*2])

        # Save it in a table:
        fileh.create_table(fileh.root, 'largerow', r)

        # Read it again
        r2 = fileh.root.largerow.read()

        self.assertEqual(r.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)

    def test01(self):
        "Checking saving a Table with an extremely large rowsize"
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, "w")

        # Create a recarray
        r = records.array([[np.arange(40000)]*4])   # 640 KB

        # Save it in a table:
#         try:
#             fileh.create_table(fileh.root, 'largerow', r)
#         except ValueError:
#             if common.verbose:
#                 (type, value, traceback) = sys.exc_info()
#               print "\nGreat!, the next ValueError was catched!"
#                 print value
#         else:
#             self.fail("expected a ValueError")
        # From PyTables 1.3 on, we allow row sizes equal or larger than 640 KB
        fileh.create_table(fileh.root, 'largerow', r)

        # Read it again
        r2 = fileh.root.largerow.read()
        self.assertEqual(r.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)


class DefaultValues(unittest.TestCase):
    record = Record

    def test00(self):
        "Checking saving a Table with default values (using the same Row)"
        file = tempfile.mktemp(".h5")
        # file = "/tmp/test.h5"
        fileh = open_file(file, "w")

        # Create a table
        table = fileh.create_table(fileh.root, 'table', self.record)

        table.nrowsinbuf = 46   # minimum amount that reproduces a problem
        # Take a number of records a bit greater
        nrows = int(table.nrowsinbuf * 1.1)
        row = table.row
        # Fill the table with nrows records
        for i in xrange(nrows):
            if i == 3:
                row['var2'] = 2
            if i == 4:
                row['var3'] = 3
            # This injects the row values.
            row.append()

        # We need to flush the buffers in table in order to get an
        # accurate number of records on it.
        table.flush()

        # Create a recarray with the same default values
        values = [b"abcd", 1, 2, 3.1, 4.2, 5, "e", 1, 1j, 1 + 0j]
        formats = 'a4,i4,i2,f8,f4,u2,a1,b1,c8,c16'.split(',')

        if 'Float16Col' in globals():
            values.append(6.4)
            formats.append('f2')
        if 'Float96Col' in globals():
            values.append(6.4)
            formats.append('f12')
        if 'Float128Col' in globals():
            values.append(6.4)
            formats.append('f16')
        if 'Complex192Col' in globals():
            values.append(1.-0.j)
            formats.append('c24')
        if 'Complex256Col' in globals():
            values.append(1.-0.j)
            formats.append('c32')

        r = records.array([values]*nrows, formats=','.join(formats))

        # Assign the value exceptions
        r["f1"][3] = 2
        r["f2"][4] = 3

        # Read the table in another recarray
        # r2 = table.read()
        r2 = table[::]  # Equivalent to table.read()

        # This generates too much output. Activate only when
        # self.nrowsinbuf is very small (<10)
        if common.verbose:
            print "First 10 table values:"
            for row in table.iterrows(0, 10):
                print row
            print "The first 5 read recarray values:"
            print r2[:5]
            print "Records should look like:"
            print r[:5]

        for name1, name2 in zip(r.dtype.names, r2.dtype.names):
            self.assertTrue(allequal(r[name1], r2[name2]))

        # The following can give false errors when columns with extended
        # precision data type are present in the record.
        # It is probably due to some difference in the value of bits used
        # for patting (longdoubles use just 80 bits but are stored in 96 or
        # 128 bits in numpy arrays)
        # self.assertEqual(r.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)

    def test01(self):
        "Checking saving a Table with default values (using different Row)"
        file = tempfile.mktemp(".h5")
        # file = "/tmp/test.h5"
        fileh = open_file(file, "w")

        # Create a table
        table = fileh.create_table(fileh.root, 'table', self.record)

        table.nrowsinbuf = 46   # minimum amount that reproduces a problem
        # Take a number of records a bit greater
        nrows = int(table.nrowsinbuf * 1.1)
        # Fill the table with nrows records
        for i in xrange(nrows):
            if i == 3:
                table.row['var2'] = 2
            if i == 4:
                table.row['var3'] = 3
            # This injects the row values.
            table.row.append()

        # We need to flush the buffers in table in order to get an
        # accurate number of records on it.
        table.flush()

        # Create a recarray with the same default values
        values = [b"abcd", 1, 2, 3.1, 4.2, 5, "e", 1, 1j, 1 + 0j]
        formats = 'a4,i4,i2,f8,f4,u2,a1,b1,c8,c16'.split(',')

        if 'Float16Col' in globals():
            values.append(6.4)
            formats.append('f2')
        if 'Float96Col' in globals():
            values.append(6.4)
            formats.append('f12')
        if 'Float128Col' in globals():
            values.append(6.4)
            formats.append('f16')
        if 'Complex192Col' in globals():
            values.append(1.-0.j)
            formats.append('c24')
        if 'Complex256Col' in globals():
            values.append(1.-0.j)
            formats.append('c32')

        r = records.array([values]*nrows, formats=','.join(formats))

        # Assign the value exceptions
        r["f1"][3] = 2
        r["f2"][4] = 3

        # Read the table in another recarray
        # r2 = table.read()
        r2 = table[::]  # Equivalent to table.read()

        # This generates too much output. Activate only when
        # self.nrowsinbuf is very small (<10)
        if common.verbose:
            print "First 10 table values:"
            for row in table.iterrows(0, 10):
                print row
            print "The first 5 read recarray values:"
            print r2[:5]
            print "Records should look like:"
            print r[:5]

        for name1, name2 in zip(r.dtype.names, r2.dtype.names):
            self.assertTrue(allequal(r[name1], r2[name2]))

        # The following can give false errors when columns with extended
        # precision data type are present in the record.
        # It is probably due to some difference in the value of bits used
        # for patting (longdoubles use just 80 bits but are stored in 96 or
        # 128 bits in numpy arrays)
        # self.assertEqual(r.tostring(), r2.tostring())

        fileh.close()
        os.remove(file)


class OldRecordDefaultValues(DefaultValues):
    title = "OldRecordDefaultValues"
    record = OldRecord


class Record2(IsDescription):
    var1 = StringCol(itemsize=4, dflt=b"abcd")  # 4-character String
    var2 = IntCol(dflt=1)                       # integer
    var3 = Int16Col(dflt=2)                     # short integer
    var4 = Float64Col(dflt=3.1)                 # double (double-precision)


class LengthTestCase(unittest.TestCase):
    record = Record
    nrows = 20

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()

    def populateFile(self):
        # Create a table
        table = self.fileh.create_table(self.fileh.root, 'table',
                                        self.record, title="__length__ test")
        # Get the row object associated with the new table
        row = table.row

        # Fill the table
        for i in xrange(self.nrows):
            row.append()

        # Flush the buffer for this table
        table.flush()
        self.table = table

    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test01_lengthrows(self):
        """Checking __length__ in Table"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_lengthrows..." % self.__class__.__name__

        # Number of rows
        len(self.table) == self.nrows

    def test02_lengthcols(self):
        """Checking __length__ in Cols"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_lengthcols..." % self.__class__.__name__

        # Number of columns
        if self.record is Record:
            len(self.table.cols) == 8
        elif self.record is Record2:
            len(self.table.cols) == 4

    def test03_lengthcol(self):
        """Checking __length__ in Column"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_lengthcol..." % self.__class__.__name__

        # Number of rows for all columns column
        for colname in self.table.colnames:
            len(getattr(self.table.cols, colname)) == self.nrows


class Length1TestCase(LengthTestCase):
    record = Record
    nrows = 20


class Length2TestCase(LengthTestCase):
    record = Record2
    nrows = 100


class WhereAppendTestCase(common.TempFileMixin, common.PyTablesTestCase):
    """Tests `Table.append_where()` method."""

    class SrcTblDesc(IsDescription):
        id = IntCol()
        v1 = FloatCol()
        v2 = StringCol(itemsize=8)

    def setUp(self):
        super(WhereAppendTestCase, self).setUp()

        tbl = self.h5file.create_table('/', 'test', self.SrcTblDesc)
        row = tbl.row

        row['id'] = 1
        row['v1'] = 1.5
        row['v2'] = 'a' * 8
        row.append()

        row['id'] = 2
        row['v1'] = 2.5
        row['v2'] = 'b' * 6
        row.append()

        tbl.flush()

    def test00_same(self):
        """Query with same storage."""

        DstTblDesc = self.SrcTblDesc

        tbl1 = self.h5file.root.test
        tbl2 = self.h5file.create_table('/', 'test2', DstTblDesc)

        tbl1.append_where(tbl2, 'id > 1')

        # Rows resulting from the query are those in the new table.
        it2 = iter(tbl2)
        for r1 in tbl1.where('id > 1'):
            r2 = it2.next()
            self.assertTrue(r1['id'] == r2['id'] and r1['v1'] == r2['v1']
                            and r1['v2'] == r2['v2'])

        # There are no more rows.
        self.assertRaises(StopIteration, it2.next)

    def test01_compatible(self):
        """Query with compatible storage."""

        class DstTblDesc(IsDescription):
            id = FloatCol()  # float, not int
            v1 = FloatCol()
            v2 = StringCol(itemsize=16)  # a longer column
            v3 = FloatCol()  # extra column

        tbl1 = self.h5file.root.test
        tbl2 = self.h5file.create_table('/', 'test2', DstTblDesc)

        tbl1.append_where(tbl2, 'id > 1')

        # Rows resulting from the query are those in the new table.
        it2 = iter(tbl2)
        for r1 in tbl1.where('id > 1'):
            r2 = it2.next()
            self.assertTrue(r1['id'] == r2['id'] and r1['v1'] == r2['v1']
                            and r1['v2'] == r2['v2'])

        # There are no more rows.
        self.assertRaises(StopIteration, it2.next)

    def test02_lessPrecise(self):
        """Query with less precise storage."""

        class DstTblDesc(IsDescription):
            id = IntCol()
            v1 = IntCol()  # int, not float
            v2 = StringCol(itemsize=8)

        tbl1 = self.h5file.root.test
        tbl2 = self.h5file.create_table('/', 'test2', DstTblDesc)

        tbl1.append_where(tbl2, 'id > 1')

        # Rows resulting from the query are those in the new table.
        it2 = iter(tbl2)
        for r1 in tbl1.where('id > 1'):
            r2 = it2.next()
            self.assertTrue(r1['id'] == r2['id'] and int(r1['v1']) == r2['v1']
                            and r1['v2'] == r2['v2'])

        # There are no more rows.
        self.assertRaises(StopIteration, it2.next)

    def test03_incompatible(self):
        """Query with incompatible storage."""

        class DstTblDesc(IsDescription):
            id = StringCol(itemsize=4)  # string, not int
            v1 = FloatCol()
            v2 = StringCol(itemsize=8)

        tbl1 = self.h5file.root.test
        tbl2 = self.h5file.create_table('/', 'test2', DstTblDesc)

        self.assertRaises(NotImplementedError,
                          tbl1.append_where, tbl2, 'v1 == b"1"')

    def test04_noColumn(self):
        """Query with storage lacking columns."""

        class DstTblDesc(IsDescription):
            # no ``id`` field
            v1 = FloatCol()
            v2 = StringCol(itemsize=8)

        tbl1 = self.h5file.root.test
        tbl2 = self.h5file.create_table('/', 'test2', DstTblDesc)

        self.assertRaises(KeyError, tbl1.append_where, tbl2, 'id > 1')

    def test05_otherFile(self):
        """Appending to a table in another file."""

        h5fname2 = tempfile.mktemp(suffix='.h5')
        h5file2 = open_file(h5fname2, 'w')

        try:
            tbl1 = self.h5file.root.test
            tbl2 = h5file2.create_table('/', 'test', self.SrcTblDesc)

            # RW to RW.
            tbl1.append_where(tbl2, 'id > 1')

            # RW to RO.
            h5file2.close()
            h5file2 = open_file(h5fname2, 'r')
            tbl2 = h5file2.root.test
            self.assertRaises(FileModeError,
                              tbl1.append_where, tbl2, 'id > 1')

            # RO to RO.
            self._reopen('r')
            tbl1 = self.h5file.root.test
            self.assertRaises(FileModeError,
                              tbl1.append_where, tbl2, 'id > 1')

            # RO to RW.
            h5file2.close()
            h5file2 = open_file(h5fname2, 'a')
            tbl2 = h5file2.root.test
            tbl1.append_where(tbl2, 'id > 1')
        finally:
            h5file2.close()
            os.remove(h5fname2)


class DerivedTableTestCase(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.mktemp('.h5')
        self.fileh = open_file(self.file, 'w', title='DeriveFromTable')

        self.fileh.create_table('/', 'original', Record)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def test00(self):
        """Deriving a table from the description of another."""

        tbl1 = self.fileh.root.original
        tbl2 = self.fileh.create_table('/', 'derived', tbl1.description)

        self.assertEqual(tbl1.description, tbl2.description)


class ChunkshapeTestCase(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.mktemp('.h5')
        self.fileh = open_file(self.file, 'w', title='Chunkshape test')
        self.fileh.create_table('/', 'table', Record, chunkshape=13)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def test00(self):
        """Test setting the chunkshape in a table (no reopen)."""

        tbl = self.fileh.root.table
        if common.verbose:
            print "chunkshape-->", tbl.chunkshape
        self.assertEqual(tbl.chunkshape, (13,))

    def test01(self):
        """Test setting the chunkshape in a table (reopen)."""

        self.fileh.close()
        self.fileh = open_file(self.file, 'r')
        tbl = self.fileh.root.table
        if common.verbose:
            print "chunkshape-->", tbl.chunkshape
        self.assertEqual(tbl.chunkshape, (13,))


# Test for appending zero-sized recarrays
class ZeroSizedTestCase(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, "a")
        # Create a Table
        t = self.fileh.create_table('/', 'table',
                                    {'c1': Int32Col(), 'c2': Float64Col()})
        # Append a single row
        t.append([(1, 2.2)])

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test01_canAppend(self):
        "Appending zero length recarray."

        t = self.fileh.root.table
        a = np.empty(shape=(0,), dtype='i4,f8')
        t.append(a)
        self.assertEqual(t.nrows, 1, "The number of rows should be 1.")


# Case for testing ticket #103, i.e. selections in columns which are
# aligned but that its data length is not an exact multiple of the
# length of the record.  This exposes the problem only in 32-bit
# machines, because in 64-bit machine, 'c2' is unaligned.  However,
# this should check most platforms where, while not unaligned,
# len(datatype) > boundary_alignment is fullfilled.
class IrregularStrideTestCase(unittest.TestCase):
    def setUp(self):
        class IRecord(IsDescription):
            c1 = Int32Col(pos=1)
            c2 = Float64Col(pos=2)

        self.file = tempfile.mktemp('.h5')
        self.fileh = open_file(self.file, 'w', title='Chunkshape test')
        table = self.fileh.create_table('/', 'table', IRecord)
        for i in range(10):
            table.row['c1'] = i
            table.row['c2'] = i
            table.row.append()
        table.flush()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def test00(self):
        """Selecting rows in a table with irregular stride (but aligned)."""

        table = self.fileh.root.table
        coords1 = table.get_where_list('c1<5')
        coords2 = table.get_where_list('c2<5')
        if common.verbose:
            print "\nSelected coords1-->", coords1
            print "Selected coords2-->", coords2
        self.assertTrue(allequal(coords1, np.arange(5, dtype=SizeType)))
        self.assertTrue(allequal(coords2, np.arange(5, dtype=SizeType)))


class Issue262TestCase(unittest.TestCase):
    def setUp(self):
        class IRecord(IsDescription):
            c1 = Int32Col(pos=1)
            c2 = Float64Col(pos=2)

        self.file = tempfile.mktemp('.h5')
        self.fileh = open_file(self.file, 'w', title='Chunkshape test')
        table = self.fileh.create_table('/', 'table', IRecord)
        table.nrowsinbuf = 3

        for i in range(20):
            table.row['c1'] = i
            table.row['c2'] = i
            table.row.append()

            table.row['c1'] = i % 29
            table.row['c2'] = 300 - i
            table.row.append()

            table.row['c1'] = 300 - i
            table.row['c2'] = 100 + i % 30
            table.row.append()

        table.flush()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)

    def test_gh260(self):
        """Regression test for gh-260"""

        table = self.fileh.root.table
        coords1 = table.get_where_list('(c1>5)&(c2<30)', start=0, step=2)
        coords2 = table.get_where_list('(c1>5)&(c2<30)', start=1, step=2)
        data = table.read()
        data = data[np.where((data['c1'] > 5) & (data['c2'] < 30))]

        if common.verbose:
            print
            print "Selected coords1-->", coords1
            print "Selected coords2-->", coords2
            print "Selected data-->", data
        self.assertEqual(len(coords1) + len(coords2), len(data))

    def test_gh262_01(self):
        """Regression test for gh-262 (start=0, step=1)"""

        table = self.fileh.root.table
        data = table.get_where_list('(c1>5)&(~(c1>5))', start=0, step=1)

        if common.verbose:
            print
            print "data -->", data
        self.assertEqual(len(data), 0)

    def test_gh262_02(self):
        """Regression test for gh-262 (start=1, step=1)"""

        table = self.fileh.root.table
        data = table.get_where_list('(c1>5)&(~(c1>5))', start=1, step=1)

        if common.verbose:
            print
            print "data -->", data
        self.assertEqual(len(data), 0)

    def test_gh262_03(self):
        """Regression test for gh-262 (start=0, step=2)"""

        table = self.fileh.root.table
        data = table.get_where_list('(c1>5)&(~(c1>5))', start=0, step=2)

        if common.verbose:
            print
            print "data -->", data
        self.assertEqual(len(data), 0)

    def test_gh262_04(self):
        """Regression test for gh-262 (start=1, step=2)"""

        table = self.fileh.root.table
        data = table.get_where_list('(c1>5)&(~(c1>5))', start=1, step=2)

        if common.verbose:
            print
            print "data -->", data
        self.assertEqual(len(data), 0)


class TruncateTestCase(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.mktemp('.h5')
        self.fileh = open_file(self.file, 'w', title='Chunkshape test')
        table = self.fileh.create_table('/', 'table', self.IRecord)
        # Fill just a couple of rows
        for i in range(2):
            table.row['c1'] = i
            table.row['c2'] = i
            table.row.append()
        table.flush()
        # The defaults
        self.dflts = table.coldflts

    def tearDown(self):
        # Close the file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test00_truncate(self):
        """Checking Table.truncate() method (truncating to 0 rows)"""

        table = self.fileh.root.table
        # Truncate to 0 elements
        table.truncate(0)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = open_file(self.file, mode="r")
            table = self.fileh.root.table

        if common.verbose:
            print "table-->", table.read()

        self.assertEqual(table.nrows, 0)
        for row in table:
            self.assertEqual(row['c1'], row.nrow)

    def test01_truncate(self):
        """Checking Table.truncate() method (truncating to 1 rows)"""

        table = self.fileh.root.table
        # Truncate to 1 element
        table.truncate(1)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = open_file(self.file, mode="r")
            table = self.fileh.root.table

        if common.verbose:
            print "table-->", table.read()

        self.assertEqual(table.nrows, 1)
        for row in table:
            self.assertEqual(row['c1'], row.nrow)

    def test02_truncate(self):
        """Checking Table.truncate() method (truncating to == self.nrows)"""

        table = self.fileh.root.table
        # Truncate to 2 elements
        table.truncate(2)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = open_file(self.file, mode="r")
            table = self.fileh.root.table

        if common.verbose:
            print "table-->", table.read()

        self.assertEqual(table.nrows, 2)
        for row in table:
            self.assertEqual(row['c1'], row.nrow)

    def test03_truncate(self):
        """Checking Table.truncate() method (truncating to > self.nrows)"""

        table = self.fileh.root.table
        # Truncate to 4 elements
        table.truncate(4)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = open_file(self.file, mode="r")
            table = self.fileh.root.table

        if common.verbose:
            print "table-->", table.read()

        self.assertEqual(table.nrows, 4)
        # Check the original values
        for row in table.iterrows(start=0, stop=2):
            self.assertEqual(row['c1'], row.nrow)
        # Check that the added rows have the default values
        for row in table.iterrows(start=2, stop=4):
            self.assertEqual(row['c1'], self.dflts['c1'])
            self.assertEqual(row['c2'], self.dflts['c2'])


class TruncateOpen1(TruncateTestCase):
    class IRecord(IsDescription):
        c1 = Int32Col(pos=1)
        c2 = FloatCol(pos=2)
    close = 0


class TruncateOpen2(TruncateTestCase):
    class IRecord(IsDescription):
        c1 = Int32Col(pos=1, dflt=3)
        c2 = FloatCol(pos=2, dflt=-3.1)
    close = 0


class TruncateClose1(TruncateTestCase):
    class IRecord(IsDescription):
        c1 = Int32Col(pos=1)
        c2 = FloatCol(pos=2)
    close = 1


class TruncateClose2(TruncateTestCase):
    class IRecord(IsDescription):
        c1 = Int32Col(pos=1, dflt=4)
        c2 = FloatCol(pos=2, dflt=3.1)
    close = 1


class PointSelectionTestCase(common.PyTablesTestCase):
    def setUp(self):
        N = 100

        # Limits for selections
        self.limits = [
            (0, 1),  # just one element
            (20, -10),  # no elements
            (-10, 4),  # several elements
            (0, 10),   # several elements (again)
        ]

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = fileh = open_file(self.file, "w")
        # Create a sample tables
        self.data = data = np.arange(N)
        self.recarr = recarr = np.empty(N, dtype="i4,f4")
        recarr["f0"][:] = data
        recarr["f1"][:] = data
        self.table = fileh.create_table(fileh.root, 'table', recarr)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test01a_read(self):
        """Test for point-selections (read, boolean keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = (data >= value1) & (data < value2)
            if common.verbose:
                print "Selection to test:", key
            a = recarr[key]
            b = table[key]
            if common.verbose:
                print "NumPy selection:", a
                print "PyTables selection:", b
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables selections does not match.")

    def test01b_read(self):
        """Test for point-selections (read, tuples of integers keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = np.where((data >= value1) & (data < value2))
            if common.verbose:
                print "Selection to test:", key, type(key)
            a = recarr[key]
            b = table[key]
#             if common.verbose:
#                 print "NumPy selection:", a
#                 print "PyTables selection:", b
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables selections does not match.")

    def test01c_read(self):
        """Test for point-selections (read, tuples of floats keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = np.where((data >= value1) & (data < value2))
            if common.verbose:
                print "Selection to test:", key
            recarr[key]
            fkey = np.array(key, "f4")
            self.assertRaises(TypeError, table.__getitem__, fkey)

    def test01d_read(self):
        """Test for point-selections (read, numpy keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = np.where((data >= value1) & (data < value2))[0]
            if common.verbose:
                print "Selection to test:", key, type(key)
            a = recarr[key]
            b = table[key]
#             if common.verbose:
#                 print "NumPy selection:", a
#                 print "PyTables selection:", b
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables selections does not match.")

    def test01e_read(self):
        """Test for point-selections (read, list keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = np.where((data >= value1) & (data < value2))[0].tolist()
            if common.verbose:
                print "Selection to test:", key, type(key)
            a = recarr[key]
            b = table[key]
#             if common.verbose:
#                 print "NumPy selection:", a
#                 print "PyTables selection:", b
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables selections does not match.")

    def test02a_write(self):
        """Test for point-selections (write, boolean keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = np.where((data >= value1) & (data < value2))
            if common.verbose:
                print "Selection to test:", key
            s = recarr[key]
            # Modify the s recarray
            s["f0"][:] = data[:len(s)]*2
            s["f1"][:] = data[:len(s)]*3
            # Modify recarr and table
            recarr[key] = s
            table[key] = s
            a = recarr[:]
            b = table[:]
#             if common.verbose:
#                 print "NumPy modified array:", a
#                 print "PyTables modifyied array:", b
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables modifications does not match.")

    def test02b_write(self):
        """Test for point-selections (write, integer keys)."""
        data = self.data
        recarr = self.recarr
        table = self.table
        for value1, value2 in self.limits:
            key = np.where((data >= value1) & (data < value2))
            if common.verbose:
                print "Selection to test:", key
            s = recarr[key]
            # Modify the s recarray
            s["f0"][:] = data[:len(s)]*2
            s["f1"][:] = data[:len(s)]*3
            # Modify recarr and table
            recarr[key] = s
            table[key] = s
            a = recarr[:]
            b = table[:]
#             if common.verbose:
#                 print "NumPy modified array:", a
#                 print "PyTables modifyied array:", b
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables modifications does not match.")


# Test for building very large MD columns without defaults
class MDLargeColTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def test01_create(self):
        "Create a Table with a very large MD column.  Ticket #211."
        N = 2**18      # 4x larger than maximum object header size (64 KB)
        cols = {'col1': Int8Col(shape=N, dflt=0)}
        tbl = self.h5file.create_table('/', 'test', cols)
        tbl.row.append()   # add a single row
        tbl.flush()
        if self.reopen:
            self._reopen('a')
            tbl = self.h5file.root.test
        # Check the value
        if common.verbose:
            print "First row-->", tbl[0]['col1']
        npt.assert_array_equal(tbl[0]['col1'], np.zeros(N, 'i1'))


class MDLargeColNoReopen(MDLargeColTestCase):
    reopen = False


class MDLargeColReopen(MDLargeColTestCase):
    reopen = True


# Test with itertools.groupby that iterates on exhausted Row iterator
# See ticket #264.
class ExhaustedIter(common.PyTablesTestCase):
    def setUp(self):
        """Create small database"""
        class Observations(IsDescription):
            market_id = IntCol(pos=0)
            scenario_id = IntCol(pos=1)
            value = Float32Col(pos=3)

        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, 'w')
        table = self.fileh.create_table('/', 'observations', Observations,
                                        chunkshape=32)

        # fill the database
        observations = np.arange(225)
        row = table.row
        for market_id in xrange(5):
            for scenario_id in xrange(3):
                for obs in observations:
                    row['market_id'] = market_id
                    row['scenario_id'] = scenario_id
                    row['value'] = obs
                    row.append()
        table.flush()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def average(self, values):
        return sum(values, 0.0) / len(values)

    def f_scenario(self, row):
        return row['scenario_id']

    def test00_groupby(self):
        """Checking iterating an exhausted iterator (ticket #264)"""
        from itertools import groupby
        rows = self.fileh.root.observations.where('(market_id == 3)')
        scenario_means = []
        for scenario_id, rows_grouped in groupby(rows, self.f_scenario):
            vals = [row['value'] for row in rows_grouped]
            scenario_means.append(self.average(vals))
        if common.verbose:
            print 'Means -->', scenario_means
        self.assertEqual(scenario_means, [112.0, 112.0, 112.0])

    def test01_groupby(self):
        """Checking iterating an exhausted iterator (ticket #264). Reopen."""
        from itertools import groupby
        self.fileh.close()
        self.fileh = open_file(self.file, 'r')
        rows = self.fileh.root.observations.where('(market_id == 3)')
        scenario_means = []
        for scenario_id, rows_grouped in groupby(rows, self.f_scenario):
            vals = [row['value'] for row in rows_grouped]
            scenario_means.append(self.average(vals))
        if common.verbose:
            print 'Means -->', scenario_means
        self.assertEqual(scenario_means, [112.0, 112.0, 112.0])


class SpecialColnamesTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def test00_check_names(self):
        f = self.h5file
        a = np.array([(1, 2, 3)], dtype=[(
            "a", int), ("_b", int), ("__c", int)])
        t = f.create_table(f.root, "test", a)
        self.assertEqual(len(t.colnames), 3, "Number of columns incorrect")
        if common.verbose:
            print "colnames -->", t.colnames
        for name, name2 in zip(t.colnames, ("a", "_b", "__c")):
            self.assertEqual(name, name2)


class RowContainsTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def test00_row_contains(self):
        f = self.h5file
        a = np.array([(1, 2, 3)], dtype="i1,i2,i4")
        t = f.create_table(f.root, "test", a)
        row = [r for r in t.iterrows()][0]
        if common.verbose:
            print "row -->", row[:]
        for item in (1, 2, 3):
            self.assertTrue(item in row)
        self.assertTrue(4 not in row)


class AccessClosedTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def setUp(self):
        super(AccessClosedTestCase, self).setUp()
        self.table = self.h5file.create_table(
            self.h5file.root, 'table', Record)

        row = self.table.row
        for i in range(10):
            row['var1'] = '%04d' % i
            row['var2'] = i
            row['var3'] = i % 3
            row.append()
        self.table.flush()

    def test_read(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.read)

    def test_getitem(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.__getitem__, 0)

    def test_setitem(self):
        data = self.table[0]
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.__setitem__, 0, data)

    def test_append(self):
        data = self.table[0]
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.append, data)

    def test_readWhere(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.read_where, 'var2 > 3')

    def test_whereAppend(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.append_where, self.table,
                          'var2 > 3')

    def test_getWhereList(self):
        self.h5file.close()
        self.assertRaises(
            ClosedNodeError, self.table.get_where_list, 'var2 > 3')

    def test_readSorted(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.read_sorted, 'var2')

    def test_readCoordinates(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.table.read_coordinates, [2, 5])


class ColumnIterationTestCase(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.mktemp(".h5")
        self.fileh = open_file(self.file, mode="w")
        self.buffer_size = self.fileh.params['IO_BUFFER_SIZE']

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def create_non_nested_table(self, nrows, dtype):
        array = np.empty((nrows, ), dtype)
        for name in dtype.names:
            array[name] = np.random.randint(0, 10000, nrows)
        table = self.fileh.create_table('/', 'table', dtype)
        table.append(array)
        return array, table

    def iterate(self, array, table):
        row_num = 0
        for item in table.cols.f0:
            self.assertEqual(item, array['f0'][row_num])
            row_num += 1
        self.assertEqual(row_num, len(array))

    def test_less_than_io_buffer(self):
        dtype = np.format_parser(['i8'] * 3, [], []).dtype
        rows_in_buffer = self.buffer_size // dtype[0].itemsize
        array, table = self.create_non_nested_table(rows_in_buffer // 2, dtype)
        self.iterate(array, table)

    def test_more_than_io_buffer(self):
        dtype = np.format_parser(['i8'] * 3, [], []).dtype
        rows_in_buffer = self.buffer_size // dtype[0].itemsize
        array, table = self.create_non_nested_table(rows_in_buffer * 3, dtype)
        self.iterate(array, table)

    def test_partially_filled_buffer(self):
        dtype = np.format_parser(['i8'] * 3, [], []).dtype
        rows_in_buffer = self.buffer_size // dtype[0].itemsize
        array, table = self.create_non_nested_table(rows_in_buffer * 2 + 2,
                                                    dtype)
        self.iterate(array, table)

    def test_zero_length_table(self):
        dtype = np.format_parser(['i8'] * 3, [], []).dtype
        array, table = self.create_non_nested_table(0, dtype)
        self.assertEqual(len(table), 0)
        self.iterate(array, table)


class TestCreateTableArgs(common.TempFileMixin, common.PyTablesTestCase):
    obj = np.array(
        [('aaaa', 1, 2.1), ('bbbb', 2, 3.2)],
        dtype=[('name', 'S4'), ('icol', np.int32), ('fcol', np.float32)])
    where = '/'
    name = 'table'
    description, _ = descr_from_dtype(obj.dtype)
    title = 'title'
    filters = None
    expectedrows = 10000
    chunkshape = None
    byteorder = None
    createparents = False

    def test_positional_args_01(self):
        self.h5file.create_table(self.where, self.name,
                                 self.description,
                                 self.title, self.filters,
                                 self.expectedrows)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (0,))
        self.assertEqual(ptarr.nrows, 0)
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)

    def test_positional_args_02(self):
        ptarr = self.h5file.create_table(self.where, self.name,
                                         self.description,
                                         self.title,
                                         self.filters,
                                         self.expectedrows)
        ptarr.append(self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (len(self.obj),))
        self.assertEqual(ptarr.nrows, len(self.obj))
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)
        self.assertEqual(nparr.dtype, self.obj.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_positional_args_obj(self):
        self.h5file.create_table(self.where, self.name,
                                 None,
                                 self.title,
                                 self.filters,
                                 self.expectedrows,
                                 self.chunkshape,
                                 self.byteorder,
                                 self.createparents,
                                 self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (len(self.obj),))
        self.assertEqual(ptarr.nrows, len(self.obj))
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj(self):
        self.h5file.create_table(self.where, self.name, title=self.title,
                                 obj=self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (len(self.obj),))
        self.assertEqual(ptarr.nrows, len(self.obj))
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_description_01(self):
        ptarr = self.h5file.create_table(self.where, self.name,
                                         title=self.title,
                                         description=self.description)
        ptarr.append(self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (len(self.obj),))
        self.assertEqual(ptarr.nrows, len(self.obj))
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_description_02(self):
        ptarr = self.h5file.create_table(self.where, self.name,
                                         title=self.title,
                                         description=self.description)
        #ptarr.append(self.obj)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (0,))
        self.assertEqual(ptarr.nrows, 0)
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)

    def test_kwargs_obj_description(self):
        ptarr = self.h5file.create_table(self.where, self.name,
                                         title=self.title,
                                         obj=self.obj,
                                         description=self.description)
        self.h5file.close()

        self.h5file = open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, (len(self.obj),))
        self.assertEqual(ptarr.nrows, len(self.obj))
        self.assertEqual(tuple(ptarr.colnames), self.obj.dtype.names)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_description_error_01(self):
        self.assertRaises(TypeError,
                          self.h5file.create_table,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          description=Record)

    def test_kwargs_obj_description_error_02(self):
        self.assertRaises(TypeError,
                          self.h5file.create_table,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          description=Record())

    def test_kwargs_obj_description_error_03(self):
        self.assertRaises(TypeError,
                          self.h5file.create_table,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          description=RecordDescriptionDict)


#----------------------------------------------------------------------


def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    # common.heavy = 1  # uncomment this only for testing purposes

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(OldRecordBasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(DictWriteTestCase))
        if sys.version_info[0] < 3:
            theSuite.addTest(unittest.makeSuite(DictWriteTestCase2))
        theSuite.addTest(unittest.makeSuite(NumPyDTWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
        theSuite.addTest(unittest.makeSuite(CompressBloscTablesTestCase))
        theSuite.addTest(unittest.makeSuite(
            CompressBloscShuffleTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressLZOTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressLZOShuffleTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressZLIBTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressZLIBShuffleTablesTestCase))
        theSuite.addTest(unittest.makeSuite(Fletcher32TablesTestCase))
        theSuite.addTest(unittest.makeSuite(AllFiltersTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressTwoTablesTestCase))
        theSuite.addTest(unittest.makeSuite(
            SizeOnDiskInMemoryPropertyTestCase))
        theSuite.addTest(unittest.makeSuite(NonNestedTableReadTestCase))
        theSuite.addTest(unittest.makeSuite(TableReadByteorderTestCase))
        theSuite.addTest(unittest.makeSuite(IterRangeTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayRangeTestCase))
        theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
        theSuite.addTest(unittest.makeSuite(getItemTestCase))
        theSuite.addTest(unittest.makeSuite(setItem1))
        theSuite.addTest(unittest.makeSuite(setItem2))
        theSuite.addTest(unittest.makeSuite(setItem3))
        theSuite.addTest(unittest.makeSuite(setItem4))
        theSuite.addTest(unittest.makeSuite(updateRow1))
        theSuite.addTest(unittest.makeSuite(updateRow2))
        theSuite.addTest(unittest.makeSuite(updateRow3))
        theSuite.addTest(unittest.makeSuite(updateRow4))
        theSuite.addTest(unittest.makeSuite(RecArrayIO1))
        theSuite.addTest(unittest.makeSuite(RecArrayIO2))
        theSuite.addTest(unittest.makeSuite(OpenCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex2TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex3TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex4TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex5TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex6TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex7TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex8TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex9TestCase))
        theSuite.addTest(unittest.makeSuite(DefaultValues))
        theSuite.addTest(unittest.makeSuite(OldRecordDefaultValues))
        theSuite.addTest(unittest.makeSuite(Length1TestCase))
        theSuite.addTest(unittest.makeSuite(Length2TestCase))
        theSuite.addTest(unittest.makeSuite(WhereAppendTestCase))
        theSuite.addTest(unittest.makeSuite(DerivedTableTestCase))
        theSuite.addTest(unittest.makeSuite(ChunkshapeTestCase))
        theSuite.addTest(unittest.makeSuite(ZeroSizedTestCase))
        theSuite.addTest(unittest.makeSuite(IrregularStrideTestCase))
        theSuite.addTest(unittest.makeSuite(Issue262TestCase))
        theSuite.addTest(unittest.makeSuite(TruncateOpen1))
        theSuite.addTest(unittest.makeSuite(TruncateOpen2))
        theSuite.addTest(unittest.makeSuite(TruncateClose1))
        theSuite.addTest(unittest.makeSuite(TruncateClose2))
        theSuite.addTest(unittest.makeSuite(PointSelectionTestCase))
        theSuite.addTest(unittest.makeSuite(MDLargeColNoReopen))
        theSuite.addTest(unittest.makeSuite(MDLargeColReopen))
        theSuite.addTest(unittest.makeSuite(ExhaustedIter))
        theSuite.addTest(unittest.makeSuite(SpecialColnamesTestCase))
        theSuite.addTest(unittest.makeSuite(RowContainsTestCase))
        theSuite.addTest(unittest.makeSuite(AccessClosedTestCase))
        theSuite.addTest(unittest.makeSuite(ColumnIterationTestCase))
        theSuite.addTest(unittest.makeSuite(TestCreateTableArgs))

    if common.heavy:
        theSuite.addTest(unittest.makeSuite(CompressBzip2TablesTestCase))
        theSuite.addTest(unittest.makeSuite(
            CompressBzip2ShuffleTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex10TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex11TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex12TestCase))
        theSuite.addTest(unittest.makeSuite(LargeRowSize))
        theSuite.addTest(unittest.makeSuite(BigTablesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
