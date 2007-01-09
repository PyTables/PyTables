import sys
import unittest
import os
import tempfile

import numpy
from numpy import *

from tables import *
import tables.tests.common as common
from tables.tests.common import verbose, heavy, allequal, cleanup

# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

# It is important that columns are ordered according to their names
# to ease the comparison with record arrays.

# Test Record class
class Record(IsDescription):
    var0 = StringCol(itemsize=4, dflt="", shape=2)  # 4-character string array
    var1 = StringCol(itemsize=4, dflt=["abcd","efgh"], shape=(2,2))
    var1_= Col.from_kind('int', dflt=((1,1),), shape=2) # integer array
    var2 = Col.from_kind('int', dflt=((1,1),(1,1)), shape=(2,2)) # integer array
    var3 = Int16Col(dflt=2)                         # short integer
    var4 = Col.from_kind('float', dflt=3.1)         # double (double-precision)
    var5 = Float32Col(dflt=4.2)                     # float  (single-precision)
    var6 = UInt16Col(dflt=5)                        # unsigned short integer
    var7 = StringCol(itemsize=1, dflt="e")          # 1-character String

# A byte-reversed class definition
class RecordRevOrder(IsDescription):
    # Change the byteorder property for this table
    _v_byteorder = {"little":"big","big":"little"}[sys.byteorder]
    var0 = StringCol(itemsize=4, dflt="", shape=2)  # 4-character string array
    var1 = StringCol(itemsize=4, dflt=["abcd","efgh"], shape=(2,2))
    var1_= Col.from_kind('int', dflt=((1,1),), shape=2) # integer array
    var2 = Col.from_kind('int', dflt=((1,1),(1,1)), shape=(2,2)) # integer array
    var3 = Int16Col(dflt=2)                         # short integer
    var4 = Col.from_kind('float', dflt=3.1)         # double (double-precision)
    var5 = Float32Col(dflt=4.2)                     # float  (single-precision)
    var6 = UInt16Col(dflt=5)                        # unsigned short integer
    var7 = StringCol(itemsize=1, dflt="e")          # 1-character String

#  Dictionary definition
RecordDescriptionDict = {
    'var0': StringCol(itemsize=4, dflt="", shape=2), # 4-character string array
    'var1': StringCol(itemsize=4, dflt=["abcd","efgh"], shape=(2,2)),
#     'var0': StringCol(itemsize=4, shape=2),       # 4-character String
#     'var1': StringCol(itemsize=4, shape=(2,2)),   # 4-character String
    'var1_':Col.from_kind('int', shape=2),        # integer array
    'var2': Col.from_kind('int', shape=(2,2)),    # integer array
    'var3': Int16Col(),                           # short integer
    'var4': Col.from_kind('float'),               # double (double-precision)
    'var5': Float32Col(),                         # float  (single-precision)
    'var6': Int16Col(),                           # unsigned short integer
    'var7': StringCol(itemsize=1),                # 1-character String
    }

# A byte-reversed dictionary definition
RecordDescriptionDictRevOrder = {
    # Change the byteorder property for this table
    '_v_byteorder': {"little":"big","big":"little"}[sys.byteorder],
    'var0': StringCol(itemsize=4, dflt="", shape=2), # 4-character string array
    'var1': StringCol(itemsize=4, dflt=["abcd","efgh"], shape=(2,2)),
#     'var0': StringCol(itemsize=4, shape=2),       # 4-character String
#     'var1': StringCol(itemsize=4, shape=(2,2)),   # 4-character String
    'var1_':Col.from_kind('int', shape=2),        # integer array
    'var2': Col.from_kind('int', shape=(2,2)),    # integer array
    'var3': Int16Col(),                           # short integer
    'var4': Col.from_kind('float'),               # double (double-precision)
    'var5': Float32Col(),                         # float  (single-precision)
    'var6': Int16Col(),                           # unsigned short integer
    'var7': StringCol(itemsize=1),                # 1-character String
    }


# Record class with numpy dtypes (mixed shapes is checkd here)
class RecordDT(IsDescription):
    var0 = Col.from_dtype(numpy.dtype("2S4"), dflt="")  # shape in dtype
    var1 = Col.from_sctype("S4", shape=(2, 2), dflt=["abcd","efgh"]) # shape is a mix
    var1_= Col.from_dtype(numpy.dtype("2i4"), dflt=((1,1),))  # shape in dtype
    var2 = Col.from_sctype("i4", shape=(2, 2), dflt=((1,1),(1,1)))  # shape is a mix
    var3 = Col.from_dtype(numpy.dtype("i2"), dflt=2)
    var4 = Col.from_dtype(numpy.dtype("2f8"), dflt=3.1)
    var5 = Col.from_dtype(numpy.dtype("f4"), dflt=4.2)
    var6 = Col.from_dtype(numpy.dtype("()u2"), dflt=5)
    var7 = Col.from_dtype(numpy.dtype("1S1"), dflt="e")   # no shape 



class BasicTestCase(common.PyTablesTestCase):
    #file  = "test.h5"
    mode  = "w"
    title = "This is the table title"
    expectedrows = 100
    appendrows = 20
    compress = 0
    complib = "zlib"  # Default compression library
    record = Record
    recordro = RecordRevOrder
    recarrayinit = 0
    maxshort = 1 << 15

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def initRecArray(self):
        record = self.recordtemplate
        row = record[0]
        buflist = []
        # Fill the recarray
        #for i in xrange(self.expectedrows+1):
        for i in xrange(self.expectedrows+1):
            tmplist = []
            # Both forms (list or chararray) works
            var0 = ['%04d' % (self.expectedrows - i)] * 2
            tmplist.append(var0)
            var1 = [['%04d' % (self.expectedrows - i)] * 2] * 2
            tmplist.append(var1)
            var1_ = (i, 1)
            tmplist.append(var1_)
            var2 = ((i, 1), (1,1))           # *-*
            tmplist.append(var2)
            var3 = i % self.maxshort
            tmplist.append(var3)
            if isinstance(row['var4'], numpy.ndarray):
                tmplist.append([float(i), float(i*i)])
            else:
                tmplist.append(float(i))
            if isinstance(row['var5'], numpy.ndarray):
                tmplist.append(array((float(i),)*4))
            else:
                tmplist.append(float(i))
            # var6 will be like var3 but byteswaped
            tmplist.append(((var3>>8) & 0xff) + ((var3<<8) & 0xff00))
            var7 = var1[0][0][-1]
            tmplist.append(var7)
            buflist.append(tmplist)

        self.record=numpy.rec.array(buflist, dtype=record.dtype,
                                    shape = self.expectedrows)
        # The swapped version
        self.recordro = self.record.newbyteorder()
        return

    def populateFile(self):
        group = self.rootgroup
        if self.recarrayinit:
            # Initialize an starting buffer, if any
            self.initRecArray()
        for j in range(3):
            # Create a table
            filters = Filters(complevel = self.compress,
                              complib = self.complib)
            if j < 2:
                record = self.record
            else:
                # table2 will be byteswapped
                record = self.recordro
            table = self.fileh.createTable(group, 'table'+str(j), record,
                                           title = self.title,
                                           filters = filters,
                                           expectedrows = self.expectedrows)
            if not self.recarrayinit:
                # Get the row object associated with the new table
                row = table.row

                # Fill the table
                for i in xrange(self.expectedrows):
                    row['var0'] = '%04d' % (self.expectedrows - i)
                    row['var1'] = '%04d' % (self.expectedrows - i)
                    row['var7'] = row['var1'][0][0][-1]
                    row['var1_'] = (i, 1)
                    row['var2'] = ((i, 1), (1,1))  # *-*
                    row['var3'] = i % self.maxshort
                    if isinstance(row['var4'], numpy.ndarray):
                        row['var4'] = [float(i), float(i*i)]
                    else:
                        row['var4'] = float(i)
                    if isinstance(row['var5'], numpy.ndarray):
                        row['var5'] = array((float(i),)*4)
                    else:
                        row['var5'] = float(i)
                    # var6 will be like var3 but byteswaped
                    row['var6'] = ((row['var3']>>8) & 0xff) + \
                                  ((row['var3']<<8) & 0xff00)
                    #print("Saving -->", row)
                    row.append()

            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.createGroup(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2


    def tearDown(self):
        self.fileh.close()
        #del self.fileh, self.rootgroup
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test00_description(self):
        """Checking table description and descriptive fields"""

        self._verboseHeader()
        self.fileh = openFile(self.file)

        tbl = self.fileh.getNode('/table0')
        desc = tbl.description

        if isinstance(self.record, dict):
            columns = self.record
        elif isinstance(self.record, numpy.ndarray):
            # This way of getting a (dictionary) description
            # can be used as long as the method does not alter the table.
            # Maybe there is a better way of doing this.
            columns = tbl._descrFromRA(self.record)
        else:
            # This is an ordinary description.
            columns = self.record.columns

        # Check table and description attributes at the same time.
        # These checks are only valid for non-nested tables.

        # Column names.
        expectedNames = ['var0', 'var1', 'var1_', 'var2', 'var3', 'var4',
                         'var5', 'var6', 'var7']
        self.assertEqual(expectedNames, list(tbl.colnames))
        self.assertEqual(expectedNames, list(desc._v_names))

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
            if verbose:
                print "dflt-->", columns[v].dflt
                print "coldflts-->", tbl.coldflts[v]
                print "desc.dflts-->", desc._v_dflts[v]
            assert common.areArraysEqual(tbl.coldflts[v], columns[v].dflt)
            assert common.areArraysEqual(desc._v_dflts[v], columns[v].dflt)

        # Column path names.
        self.assertEqual(expectedNames, list(desc._v_pathnames))

        # Column objects.
        for colName in expectedNames:
            expectedCol = columns[colName]
            col = desc._v_colObjects[colName]
            self.assertEqual(expectedCol.dtype, col.dtype)
            self.assertEqual(expectedCol.type, col.type)

    def test01_readTable(self):
        """Checking table read and cuts"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/table0")

        # Choose a small value for buffer size
        table._v_nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [ rec['var2'][0][0] for rec in table.iterrows()
                   if rec['var2'][0][0] < 20 ]

        if verbose:
            print "Table:", repr(table)
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        assert ((
            rec['var0'][0],
            rec['var1'][0][0],
            rec['var1_'][0],
            rec['var2'][0][0],
            rec['var7']
            ) == (
            "0001",
            "0001",
            nrows,
            nrows,
            "1"))
        if isinstance(rec['var5'], numpy.ndarray):
            assert allequal(rec['var5'], array((nrows,)*4, float32))
        else:
            assert rec['var5'] == float(nrows)
        assert len(result) == 20

    def test01b_readTable(self):
        """Checking table read and cuts (multidimensional columns case)"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/table0")

        # Choose a small value for buffer size
        table._v_nrowsinbuf = 3
        # Read the records and select those with "var2" file less than 20
        result = [ rec['var5'] for rec in table.iterrows()
                   if rec['var2'][0][0] < 20 ]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = table.nrows
        if isinstance(rec['var5'], numpy.ndarray):
            assert allequal(result[0], array((float(0),)*4, float32))
            assert allequal(result[1], array((float(1),)*4, float32))
            assert allequal(result[2], array((float(2),)*4, float32))
            assert allequal(result[3], array((float(3),)*4, float32))
            assert allequal(result[10], array((float(10),)*4, float32))
            assert allequal(rec['var5'], array((float(nrows-1),)*4, float32))
        else:
            assert rec['var5'] == float(nrows-1)
        assert len(result) == 20

        # Read the records and select those with "var2" file less than 20
        result = [ rec['var1'] for rec in table.iterrows()
                   if rec['var2'][0][0] < 20 ]

        if rec['var1'].dtype.char == "S":
            a = numpy.array([['%04d' % (self.expectedrows - 0)]*2]*2)
            assert allequal(result[0], a)
            a = numpy.array([['%04d' % (self.expectedrows - 1)]*2]*2)
            assert allequal(result[1], a)
            a = numpy.array([['%04d' % (self.expectedrows - 2)]*2]*2)
            assert allequal(result[2], a)
            a = numpy.array([['%04d' % (self.expectedrows - 3)]*2]*2)
            assert allequal(result[3], a)
            a = numpy.array([['%04d' % (self.expectedrows - 10)]*2]*2)
            assert allequal(result[10], a)
            a = numpy.array([['%04d' % (1)]*2]*2)
            assert allequal(rec['var1'], a)
        else:
            assert rec['var1'] == "0001"
        assert len(result) == 20

    def test02_AppendRows(self):
        """Checking whether appending record rows works or not"""

        # Now, open it, but in "append" mode
        self.fileh = openFile(self.file, mode = "a")
        self.rootgroup = self.fileh.root
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_AppendRows..." % self.__class__.__name__

        # Get a table
        table = self.fileh.getNode("/group0/table1")
        # Get their row object
        row = table.row
        if verbose:
            print "Nrows in old", table._v_pathname, ":", table.nrows
            print "Record Format ==>", table.description._v_nestedFormats
            print "Record Size ==>", table.rowsize
        # Append some rows
        for i in xrange(self.appendrows):
            row['var0'] = '%04d' % (self.appendrows - i)
            row['var1'] = '%04d' % (self.appendrows - i)
            row['var7'] = row['var1'][0][0][-1]
            row['var1_'] = (i, 1)
            row['var2'] = ((i, 1), (1,1))   # *-*
            row['var3'] = i % self.maxshort
            if isinstance(row['var4'], numpy.ndarray):
                row['var4'] = [float(i), float(i*i)]
            else:
                row['var4'] = float(i)
            if isinstance(row['var5'], numpy.ndarray):
                row['var5'] = array((float(i),)*4)
            else:
                row['var5'] = float(i)
            row.append()

        # Flush the buffer for this table and read it
        table.flush()
        result = [ row['var2'][0][0] for row in table.iterrows()
                   if row['var2'][0][0] < 20 ]

        nrows = self.appendrows - 1
        assert ((
            row['var0'][0],
            row['var1'][0][0],
            row['var1_'][0],
            row['var2'][0][0],
            row['var7']
            ) == (
            "0001",
            "0001",
            nrows,
            nrows,
            "1"))
        if isinstance(row['var5'], numpy.ndarray):
            assert allequal(row['var5'], array((float(nrows),)*4, float32))
        else:
            assert row['var5'] == float(nrows)
        if self.appendrows <= 20:
            add = self.appendrows
        else:
            add = 20
        assert len(result) == 20 + add  # because we appended new rows
        #del table

    # CAVEAT: The next test only works for tables with rows < 2**15
    def test03_endianess(self):
        """Checking if table is endianess aware"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_endianess..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/group0/group1/table2")

        # Read the records and select the ones with "var6" column less than 20
        result = [ rec['var2'] for rec in table.iterrows() if rec['var6'] < 20]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==>", len(result)
        nrows = self.expectedrows - 1
        assert (rec['var1'][0][0], rec['var6']) == ("0001", nrows)
        assert len(result) == 20

class BasicWriteTestCase(BasicTestCase):
    title = "BasicWrite"
    pass

class DictWriteTestCase(BasicTestCase):
    # This checks also unidimensional arrays as columns
    title = "DictWrite"
    record = RecordDescriptionDict
    recordro = RecordDescriptionDictRevOrder
    nrows = 21
    nrowsinbuf = 3  # Choose a small value for the buffer size
    start = 0
    stop = 10
    step = 3

class DTypeWriteTestCase(BasicTestCase):
    title = "DTypeWriteTestCase"
    record=RecordDT

class RecArrayOneWriteTestCase(BasicTestCase):
    title = "RecArrayOneWrite"
    record=numpy.rec.array(
        None,
        formats="(2,)a4,(2,2)a4,(2,)i4,(2,2)i4,i2,2f8,f4,i2,a1",
        names='var0,var1,var1_,var2,var3,var4,var5,var6,var7',
        shape=0)

class RecArrayTwoWriteTestCase(BasicTestCase):
    title = "RecArrayTwoWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=numpy.rec.array(
        None,
        formats="(2,)a4,(2,2)a4,(2,)i4,(2,2)i4,i2,f8,f4,i2,a1",
        names='var0,var1,var1_,var2,var3,var4,var5,var6,var7',
        shape=1)

class RecArrayThreeWriteTestCase(BasicTestCase):
    title = "RecArrayThreeWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=numpy.rec.array(
        None,
        formats="(2,)a4,(2,2)a4,(2,)i4,(2,2)i4,i2,2f8,4f4,i2,a1",
        names='var0,var1,var1_,var2,var3,var4,var5,var6,var7',
        shape=1)

class CompressLZOTablesTestCase(BasicTestCase):
    title = "CompressLZOTables"
    compress = 1
    complib = "lzo"

class CompressBZIP2TablesTestCase(BasicTestCase):
    title = "CompressBZIP2Tables"
    compress = 1
    complib = "bzip2"

class CompressZLIBTablesTestCase(BasicTestCase):
    title = "CompressOneTables"
    compress = 1
    complib = "zlib"

class CompressTwoTablesTestCase(BasicTestCase):
    title = "CompressTwoTables"
    compress = 1
    # This checks also unidimensional arrays as columns
    record = RecordDescriptionDict
    recordro = RecordDescriptionDictRevOrder

class BigTablesTestCase(BasicTestCase):
    title = "BigTables"
    # 10000 rows takes much more time than we can afford for tests
    # reducing to 1000 would be more than enough
    # F. Altet 2004-01-19
#     expectedrows = 10000
#     appendrows = 1000
    expectedrows = 1000
    appendrows = 100

class BasicRangeTestCase(unittest.TestCase):
    #file  = "test.h5"
    mode  = "w"
    title = "This is the table title"
    record = Record
    maxshort = 1 << 15
    expectedrows = 100
    compress = 0
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
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        for j in range(3):
            # Create a table
            table = self.fileh.createTable(group, 'table'+str(j), self.record,
                                           title = self.title,
                                           filters = Filters(self.compress),
                                           expectedrows = self.expectedrows)
            # Get the row object associated with the new table
            row = table.row

            # Fill the table
            for i in xrange(self.expectedrows):
                row['var1'] = '%04d' % (self.expectedrows - i)
                row['var7'] = row['var1'][0][0][-1]
                row['var2'] = i
                row['var3'] = i % self.maxshort
                if isinstance(row['var4'], numpy.ndarray):
                    row['var4'] = [float(i), float(i*i)]
                else:
                    row['var4'] = float(i)
                if isinstance(row['var5'], numpy.ndarray):
                    row['var5'] = array((float(i),)*4)
                else:
                    row['var5'] = float(i)
                # var6 will be like var3 but byteswaped
                row['var6'] = ((row['var3'] >> 8) & 0xff) + \
                              ((row['var3'] << 8) & 0xff00)
                row.append()

            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.createGroup(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2


    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        #del self.fileh, self.rootgroup
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def check_range(self):

        rootgroup = self.rootgroup
        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/table0")

        table._v_nrowsinbuf = self.nrowsinbuf
        r = slice(self.start, self.stop, self.step)
        resrange = r.indices(table.nrows)
        reslength = len(range(*resrange))
        if self.checkrecarray:
            recarray = table.read(self.start, self.stop, self.step)
            result = []
            for nrec in range(len(recarray)):
                if recarray['var2'][nrec][0][0] < self.nrows:
                    result.append(recarray['var2'][nrec][0][0])
        elif self.checkgetCol:
            column = table.read(self.start, self.stop, self.step, 'var2')
            result = []
            for nrec in range(len(column)):
                if column[nrec][0][0] < self.nrows:    #*-*
                    result.append(column[nrec][0][0])  #*-*
        else:
            result = [ rec['var2'][0][0] for rec in
                       table.iterrows(self.start, self.stop, self.step)
                       if rec['var2'][0][0] < self.nrows ]

        if self.start < 0:
            startr = self.expectedrows + self.start
        else:
            startr = self.start

        if self.stop == None:
            stopr = startr + 1
        elif self.stop < 0:
            stopr = self.expectedrows + self.stop
        else:
            stopr = self.stop

        if self.nrows < stopr:
            stopr = self.nrows

        if verbose:
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
            print "start, stop, step ==>", startr, stopr, self.step

        assert result == range(startr, stopr, self.step)
        if startr < stopr and not (self.checkrecarray or self.checkgetCol):
            if self.nrows < self.expectedrows:
                assert rec['var2'][0][0] == \
                       range(self.start, self.stop, self.step)[-1]
            else:
                assert rec['var2'][0][0] == range(startr, stopr, self.step)[-1]

        # Close the file
        self.fileh.close()

    def test01_range(self):
        """Checking ranges in table iterators (case1)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 2*step
        self.nrows = 21
        self.nrowsinbuf = 3
        self.start = 0
        self.stop = self.expectedrows
        self.step = 2

        self.check_range()

    def test02_range(self):
        """Checking ranges in table iterators (case2)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 10*step
        self.nrows = 21
        self.nrowsinbuf = 31
        self.start = 11
        self.stop = self.expectedrows
        self.step = 3

        self.check_range()

    def test03_range(self):
        """Checking ranges in table iterators (case3)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_range..." % self.__class__.__name__

        # Case where step < nrowsinbuf < 1.1*step
        self.nrows = self.expectedrows
        self.nrowsinbuf = 11  # Choose a small value for the buffer size
        self.start = 0
        self.stop = self.expectedrows
        self.step = 10

        self.check_range()

    def test04_range(self):
        """Checking ranges in table iterators (case4)"""

        if verbose:
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

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_range..." % self.__class__.__name__

        # Case where step > 1.1*nrowsinbuf
        self.nrows = 21
        self.nrowsinbuf = 10  # Choose a small value for the buffer size
        self.start = 1
        self.stop = self.expectedrows
        self.step = 11

        self.check_range()

    def test06_range(self):
        """Checking ranges in table iterators (case6)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_range..." % self.__class__.__name__

        # Case where step > 3*nrowsinbuf
        self.nrows = 3
        self.nrowsinbuf = 3  # Choose a small value for the buffer size
        self.start = 2
        self.stop = self.expectedrows
        self.step = 10

        self.check_range()

    def test07_range(self):
        """Checking ranges in table iterators (case7)"""

        if verbose:
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

        if verbose:
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

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_range..." % self.__class__.__name__

        # Case where stop = None
        self.nrows = 100
        self.nrowsinbuf = 3  # Choose a small value for the buffer size
        self.start = 1
        self.stop = None
        self.step = 1

        self.check_range()

    def test10_range(self):
        """Checking ranges in table iterators (case10)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_range..." % self.__class__.__name__

        # Case where start < 0 and stop = 0
        self.nrows = self.expectedrows
        self.nrowsinbuf = 5  # Choose a small value for the buffer size
        self.start = -6
        self.startr = self.expectedrows + self.start
        self.stop = 0
        self.stopr = self.expectedrows + self.stop
        self.step = 2

        self.check_range()

    def test11_range(self):
        """Checking ranges in table iterators (case11)"""

        if verbose:
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

        if verbose:
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

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13_range..." % self.__class__.__name__

        # Case where step < 0
        self.step = -11
        try:
            self.check_range()
        except ValueError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            self.fileh.close()
        else:
            print rec
            self.fail("expected a ValueError")

        # Case where step == 0
        self.step = 0
        try:
            self.check_range()
        except ValueError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            self.fileh.close()
        else:
            print rec
            self.fail("expected a ValueError")


class IterRangeTestCase(BasicRangeTestCase):
    pass

class RecArrayRangeTestCase(BasicRangeTestCase):
    checkrecarray = 1

class getColRangeTestCase(BasicRangeTestCase):
    checkgetCol = 1

    def test01_nonexistentField(self):
        """Checking non-existing Field in getCol method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_nonexistentField..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        self.root = self.fileh.root
        table = self.fileh.getNode("/table0")

        try:
            column = table.read(field='non-existent-column')
        except KeyError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next KeyError was catched!"
                print value
            pass
        else:
            print rec
            self.fail("expected a KeyError")


class RecArrayIO(unittest.TestCase):

    def test00(self):
        "Checking saving a normal recarray"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[456,23]*3]*2
        intlist2 = array([[2,2]*3]*2, dtype=int)
        arrlist1 = [['dbe']*2]*3
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b = [[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=numpy.rec.array(b, formats='(2,6)i4,(3,2)a3,(4,6)f8',
                          names='col1,col2,col3')

        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r)

        # Read it again
        r2 = fileh.root.recarray.read()

        assert r.tostring() == r2.tostring()

        fileh.close()
        os.remove(file)

    def test01(self):
        "Checking saving a recarray with an offset in its buffer"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[456,23]*3]*2
        intlist2 = array([[2,2]*3]*2, dtype=int)
        arrlist1 = [['dbe']*2]*3
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b = [[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=numpy.rec.array(b, formats='(2,6)i4,(3,2)a3,(4,6)f8',
                          names='col1,col2,col3')

        # Get a view of the recarray
        r1 = r[1:]
        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r1)
        # Read it again
        r2 = fileh.root.recarray.read()

        assert r1.tostring() == r2.tostring()

        fileh.close()
        os.remove(file)

    def test02(self):
        "Checking saving a slice of a large recarray"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[[23,24,35]*6]*6]
        intlist2 = array([[[2,3,4]*6]*6], dtype=int)
        arrlist1 = [['dbe']*2]*3
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b=[[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=numpy.rec.array(b*300,  formats='(1,6,18)i4,(3,2)a3,(4,6)f8',
                          names='col1,col2,col3')

        # Get an slice of recarray
        r1 = r[290:292]
        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r1)
        # Read it again
        r2 = fileh.root.recarray.read()

        assert r1.tostring() == r2.tostring()

        fileh.close()
        os.remove(file)

    def test03(self):
        "Checking saving a slice of an strided recarray"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[[23,24,35]*6]*6]
        intlist2 = array([[[2,3,4]*6]*6], dtype=int)
        arrlist1 = [['dbe']*2]*3
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b = [[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=numpy.rec.array(b*300, formats='(1,6,18)i4,(3,2)a3,(4,6)f8',
                          names='col1,col2,col3', shape=300)

        # Get an strided recarray
        r2 = r[::2]
        # Get a slice
        r1 = r2[148:]
        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r1)
        # Read it again
        r2 = fileh.root.recarray.read()

        assert r1.tostring() == r2.tostring()

        fileh.close()
        os.remove(file)


class DefaultValues(unittest.TestCase):

    def test00(self):
        "Checking saving a Table MD with default values"
        file = tempfile.mktemp(".h5")
        #file = "/tmp/test.h5"
        fileh = openFile(file, "w")

        # Create a table
        table = fileh.createTable(fileh.root, 'table', Record)

        # Take a number of records a bit large
        #nrows = int(table._v_nrowsinbuf * 1.1)
        nrows = 5  # for test
        # Fill the table with nrows records
        for i in xrange(nrows):
            if i == 3 or i == 4:
                table.row['var2'] = ((2,2),(2,2))  #*-*
            # This injects the row values.
            table.row.append()

        # We need to flush the buffers in table in order to get an
        # accurate number of records on it.
        table.flush()

        # Create a recarray with the same default values
        buffer = [[
            ["\x00"]*2,  # just "" does not initialize the buffer properly
            [["abcd","efgh"]]*2,
            (1,1),
            ((1,1),(1,1)),
            2, 3.1, 4.2, 5, "e"]]
        r = numpy.rec.array(
            buffer*nrows,
            formats='(2,)a4,(2,2)a4,(2,)i4,(2,2)i4,i2,f8,f4,u2,a1',
            names = ['var0', 'var1', 'var1_', 'var2', 'var3', 'var4', 'var5',
                     'var6', 'var7'])  #*-*

        # Assign the value exceptions
        r["var2"][3] = ((2,2), (2,2))  #*-*
        r["var2"][4] = ((2,2), (2,2))  #*-*

        # Read the table in another recarray
        r2 = table.read()

        # This generates too much output. Activate only when
        # self._v_nrowsinbuf is very small (<10)
        if verbose and 1:
            print "Table values:"
            print r2
            print "Record values:"
            print r

        # Both checks do work, however, tostring() seems more stringent.
        assert r.tostring() == r2.tostring()
        #assert common.areArraysEqual(r,r2)

        fileh.close()
        os.remove(file)

class RecordT(IsDescription):
    var0 = Col.from_kind('int', dflt=1, shape=1) # native int
    var1 = Col.from_kind('int', dflt=[1], shape=(1,)) # 1-D int (one element)
    var2_s = Col.from_kind('int', dflt=[1,1], shape=2) # 1-D int (two elements)
    var2 = Col.from_kind('int', dflt=[1,1], shape=(2,)) # 1-D int (two elements)
    var3 = Col.from_kind('int', dflt=[[0,0],[1,1]], shape=(2,2)) # 2-D int

class ShapeTestCase(unittest.TestCase):

    def setUp(self):

        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.populateFile()

    def populateFile(self):
        table = self.fileh.createTable(self.fileh.root, 'table', RecordT)
        row = table.row
        # Fill the table with some rows with default values
        for i in xrange(1):
            row.append()

        # Flush the buffer for this table
        table.flush()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test00(self):
        "Checking scalar shapes"

        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file)
        table = self.fileh.root.table

        if verbose:
            print "The values look like:", table.cols.var0[:]
            print "They should look like:", [1]

        # The real check
        assert table.cols.var0[:].tolist() == [1]

    def test01(self):
        "Checking undimensional (one element) shapes"

        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file)
        table = self.fileh.root.table

        if verbose:
            print "The values look like:", table.cols.var1[:]
            print "They should look like:", [[1]]

        # The real check
        assert table.cols.var1[:].tolist() == [[1]]

    def test02(self):
        "Checking undimensional (two elements) shapes"

        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file)
        table = self.fileh.root.table

        if verbose:
            print "The values look like:", table.cols.var2[:]
            print "They should look like:", [[1,1]]

        # The real check
        assert table.cols.var2[:].tolist() == [[1,1]]
        assert table.cols.var2_s[:].tolist() == [[1,1]]

    def test03(self):
        "Checking bidimensional shapes"

        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file)
        table = self.fileh.root.table

        if verbose:
            print "The values look like:", table.cols.var3[:]
            print "They should look like:", [[[0,0],[1,1]]]

        # The real check
        assert table.cols.var3[:].tolist() == [[[0,0],[1,1]]]


class ShapeTestCase1(ShapeTestCase):
    reopen = 0

class ShapeTestCase2(ShapeTestCase):
    reopen = 1

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    #heavy = 1  # Uncomment this only for testing purposes

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(DictWriteTestCase))
        theSuite.addTest(unittest.makeSuite(DTypeWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
        theSuite.addTest(unittest.makeSuite(CompressZLIBTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressTwoTablesTestCase))
        theSuite.addTest(unittest.makeSuite(IterRangeTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayRangeTestCase))
        theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
        theSuite.addTest(unittest.makeSuite(DefaultValues))
        theSuite.addTest(unittest.makeSuite(RecArrayIO))
        theSuite.addTest(unittest.makeSuite(ShapeTestCase1))
        theSuite.addTest(unittest.makeSuite(ShapeTestCase2))
    if heavy:
        theSuite.addTest(unittest.makeSuite(CompressLZOTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressBZIP2TablesTestCase))
        theSuite.addTest(unittest.makeSuite(BigTablesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
