import sys
import unittest
import os
import tempfile

import numarray
from numarray import *
import numarray.records as records
from numarray import strings
from tables import *
from tables.hdf5Extension import getIndices

from test_all import verbose, heavy, allequal
# If we use the test_all.allequal function, a segmentation violation appears
# but only when the test runs *alone* and *without* verbose parameters!
# However, if we use the allequal in this module, everything seems to work well
# this should be further investigated!. F. Alted 2004/01/01

# Update: That seems to work well now. Perhaps a bug in numarray that
# has been solved? F. Alted 2004/08/06

# def allequal(a,b):
#     """Checks if two numarrays are equal"""

#     if a.shape <> b.shape:
#         return 0

#     # Rank-0 case
#     if len(a.shape) == 0:
#         if str(equal(a,b)) == '1':
#             return 1
#         else:
#             return 0
#     # Multidimensional case
#     result = (a == b)
#     for i in range(len(a.shape)):
#         result = logical_and.reduce(result)

#     return result

# Test Record class
class Record(IsDescription):
    var1 = StringCol(4, "abcd", shape=(2,2))    # 4-character string array
    var2 = IntCol(((1,1),(1,1)), shape=(2,2))   # integer array
    var3 = Int16Col(2)                          # short integer 
    var4 = FloatCol(3.1)                        # double (double-precision)
    var5 = Float32Col(4.2)                      # float  (single-precision)
    var6 = UInt16Col(5)                         # unsigned short integer 
    var7 = StringCol(length=1, dflt="e")        # 1-character String

# From 0.3 on, you can dynamically define the tables with a dictionary
RecordDescriptionDict = {
    'var1': StringCol(length=4, shape=(2,2)),     # 4-character String
    'var2': IntCol(shape=(2,2)),                  # integer array
    'var3': Int16Col(),                           # short integer 
    'var4': FloatCol(),                           # double (double-precision)
    'var5': Float32Col(),                         # float  (single-precision)
    'var6': Int16Col(),                           # unsigned short integer 
    'var7': StringCol(length=1),                  # 1-character String
    }

class BasicTestCase(unittest.TestCase):
    #file  = "test.h5"
    mode  = "w" 
    title = "This is the table title"
    expectedrows = 100
    appendrows = 20
    compress = 0
    complib = "zlib"  # Default compression library
    record = Record
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
            var1 = [['%04d' % (self.expectedrows - i)] * 2] * 2
#             var1 = strings.array([['%04d' % (self.expectedrows - i)] * 2] * 2,
#                                  itemsize = 4, shape=(2,2))
            tmplist.append(var1)
            var2 = ((i, 1), (1,1))           # *-*
            tmplist.append(var2)
            var3 = i % self.maxshort
            tmplist.append(var3)
            if isinstance(row.field('var4'), NumArray):
                tmplist.append([float(i), float(i*i)])
            else:
                tmplist.append(float(i))
            if isinstance(row.field('var5'), NumArray):
                tmplist.append(array((float(i),)*4))
            else:
                tmplist.append(float(i))
            # var6 will be like var3 but byteswaped
            tmplist.append(((var3>>8) & 0xff) + ((var3<<8) & 0xff00))
            var7 = var1[0][0][-1]
            tmplist.append(var7)
            buflist.append(tmplist)

        self.record=records.array(buflist, formats=record._formats,
                                   names=record._names,
                                   shape = self.expectedrows)

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
            table = self.fileh.createTable(group, 'table'+str(j), self.record,
                                           title = self.title,
                                           filters = filters,
                                           expectedrows = self.expectedrows)
            if not self.recarrayinit:
                # Get the row object associated with the new table
                row = table.row
	    
                # Fill the table
                for i in xrange(self.expectedrows):
                    row['var1'] = '%04d' % (self.expectedrows - i)
                    row['var7'] = row['var1'][0][0][-1]
                    row['var2'] = ((i, 1), (1,1))  # *-* 
                    row['var3'] = i % self.maxshort
                    if isinstance(row['var4'], NumArray):
                        row['var4'] = [float(i), float(i*i)]
                    else:
                        row['var4'] = float(i)
                    if isinstance(row['var5'], NumArray):
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
        
    #----------------------------------------

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
        table._v_maxTuples = 3
        # Read the records and select those with "var2" file less than 20
        result = [ rec['var2'][0][0] for rec in table.iterrows()
                   if rec['var2'][0][0] < 20 ]
        if verbose:
            print "Table:", repr(table)
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        assert (rec['var1'][0][0], rec['var2'][0][0], rec['var7']) == \
               ("0001",            nrows,             "1")
        if isinstance(rec['var5'], NumArray):
            assert allequal(rec['var5'], array((float(nrows),)*4, Float32))
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
        table._v_maxTuples = 3
        # Read the records and select those with "var2" file less than 20
        result = [ rec['var5'] for rec in table.iterrows()
                   if rec['var2'][0][0] < 20 ]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = table.row.nrow()
        if isinstance(rec['var5'], NumArray):
            assert allequal(result[0], array((float(0),)*4, Float32))
            assert allequal(result[1], array((float(1),)*4, Float32))
            assert allequal(result[2], array((float(2),)*4, Float32))
            assert allequal(result[3], array((float(3),)*4, Float32))
            assert allequal(result[10], array((float(10),)*4, Float32))
            assert allequal(rec['var5'], array((float(nrows),)*4, Float32))
        else:
            assert rec['var5'] == float(nrows)
        assert len(result) == 20

        # Read the records and select those with "var2" file less than 20
        result = [ rec['var1'] for rec in table.iterrows()
                   if rec['var2'][0][0] < 20 ]

        if isinstance(rec['var1'], strings.CharArray):
            a = strings.array([['%04d' % (self.expectedrows - 0)]*2]*2)
            assert allequal(result[0], a)
            a = strings.array([['%04d' % (self.expectedrows - 1)]*2]*2)
            assert allequal(result[1], a)
            a = strings.array([['%04d' % (self.expectedrows - 2)]*2]*2)
            assert allequal(result[2], a)
            a = strings.array([['%04d' % (self.expectedrows - 3)]*2]*2)
            assert allequal(result[3], a)
            a = strings.array([['%04d' % (self.expectedrows - 10)]*2]*2)
            assert allequal(result[10], a)
            a = strings.array([['%04d' % (1)]*2]*2)
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
            print "Record Format ==>", table._v_fmt
            print "Record Size ==>", table.rowsize
        # Append some rows
        for i in xrange(self.appendrows):
            row['var1'] = '%04d' % (self.appendrows - i)
            row['var7'] = row['var1'][0][0][-1]
            row['var2'] = ((i, 1), (1,1))   # *-*
            row['var3'] = i % self.maxshort
            if isinstance(row['var4'], NumArray):
                row['var4'] = [float(i), float(i*i)]
            else:
                row['var4'] = float(i)
            if isinstance(row['var5'], NumArray):
                row['var5'] = array((float(i),)*4)
            else:
                row['var5'] = float(i)
            row.append()
	    
	# Flush the buffer for this table and read it
        table.flush()
        result = [ row['var2'][0][0] for row in table.iterrows()
                   if row['var2'][0][0] < 20 ]
	
        nrows = self.appendrows - 1
        assert (row['var1'][0][0], row['var2'][0][0], row['var7']) == \
               ("0001", nrows, "1")
        if isinstance(row['var5'], NumArray):
            assert allequal(row['var5'], array((float(nrows),)*4, Float32))
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
        table = self.fileh.getNode("/table0")

        # Manually change the byteorder property for this table
        table.byteorder = {"little":"big","big":"little"}[table.byteorder]
        #table.togglebyteorder()
	
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
    nrows = 21
    maxTuples = 3  # Choose a small value for the buffer size
    start = 0
    stop = 10
    step = 3

class RecArrayOneWriteTestCase(BasicTestCase):
    title = "RecArrayOneWrite"
    record=records.array(formats="(2,2)a4,(2,2)i4,i2,2f8,f4,i2,a1",
                         names='var1,var2,var3,var4,var5,var6,var7')

class RecArrayTwoWriteTestCase(BasicTestCase):
    title = "RecArrayTwoWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=records.array(formats="(2,2)a4,(2,2)i4,i2,f8,f4,i2,a1",
                                 names='var1,var2,var3,var4,var5,var6,var7',
                                 shape=1)

class RecArrayThreeWriteTestCase(BasicTestCase):
    title = "RecArrayThreeWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=records.array(formats="(2,2)a4,(2,2)i4,i2,2f8,4f4,i2,a1",
                                 names='var1,var2,var3,var4,var5,var6,var7',
                                 shape=1)

class CompressLZOTablesTestCase(BasicTestCase):
    title = "CompressLZOTables"
    compress = 1
    complib = "lzo"
    
class CompressUCLTablesTestCase(BasicTestCase):
    title = "CompressUCLTables"
    compress = 1
    complib = "ucl"
    
class CompressZLIBTablesTestCase(BasicTestCase):
    title = "CompressOneTables"
    compress = 1
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
    maxTuples = 3  # Choose a small value for the buffer size
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
                if isinstance(row['var4'], NumArray):
                    row['var4'] = [float(i), float(i*i)]
                else:
                    row['var4'] = float(i)
                if isinstance(row['var5'], NumArray):
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
        
    #----------------------------------------

    def check_range(self):

        rootgroup = self.rootgroup
        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/table0")

        table._v_maxTuples = self.maxTuples
        r = slice(self.start, self.stop, self.step)
        #resrange = r.indices(table.nrows)
        resrange = getIndices(r,table.nrows)
        reslength = len(range(*resrange))
        if self.checkrecarray:
            recarray = table.read(self.start, self.stop, self.step)
            result = []
            for nrec in range(len(recarray)):
                if recarray.field('var2')[nrec][0][0] < self.nrows:
                    result.append(recarray.field('var2')[nrec][0][0])
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

        # Case where step < maxTuples < 2*step
        self.nrows = 21
        self.maxTuples = 3
        self.start = 0
        self.stop = self.expectedrows
        self.step = 2

        self.check_range()

    def test02_range(self):
        """Checking ranges in table iterators (case2)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_range..." % self.__class__.__name__

        # Case where step < maxTuples < 10*step
        self.nrows = 21
        self.maxTuples = 31
        self.start = 11
        self.stop = self.expectedrows
        self.step = 3

        self.check_range()

    def test03_range(self):
        """Checking ranges in table iterators (case3)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_range..." % self.__class__.__name__

        # Case where step < maxTuples < 1.1*step
        self.nrows = self.expectedrows
        self.maxTuples = 11  # Choose a small value for the buffer size
        self.start = 0
        self.stop = self.expectedrows
        self.step = 10

        self.check_range()

    def test04_range(self):
        """Checking ranges in table iterators (case4)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_range..." % self.__class__.__name__

        # Case where step == maxTuples 
        self.nrows = self.expectedrows
        self.maxTuples = 11  # Choose a small value for the buffer size
        self.start = 1
        self.stop = self.expectedrows
        self.step = 11

        self.check_range()

    def test05_range(self):
        """Checking ranges in table iterators (case5)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_range..." % self.__class__.__name__

        # Case where step > 1.1*maxTuples 
        self.nrows = 21
        self.maxTuples = 10  # Choose a small value for the buffer size
        self.start = 1
        self.stop = self.expectedrows
        self.step = 11

        self.check_range()

    def test06_range(self):
        """Checking ranges in table iterators (case6)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_range..." % self.__class__.__name__

        # Case where step > 3*maxTuples 
        self.nrows = 3
        self.maxTuples = 3  # Choose a small value for the buffer size
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
        self.maxTuples = 3  # Choose a small value for the buffer size
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
        self.maxTuples = 3  # Choose a small value for the buffer size
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
        self.maxTuples = 3  # Choose a small value for the buffer size
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
        self.maxTuples = 5  # Choose a small value for the buffer size
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
        self.maxTuples = 5  # Choose a small value for the buffer size
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
        self.maxTuples = 5  # Choose a small value for the buffer size
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
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
            pass
        else:
            print rec
            self.fail("expected a LookupError")


class RecArrayIO(unittest.TestCase):

    def test00(self):
        "Checking saving a normal recarray"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[456,23]*3]*2
        intlist2 = array([[2,2]*3]*2)
        arrlist1 = [['dbe']*2]*3
        #arrlist2 = strings.array([['de']*2]*3)
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b = [[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=records.array(b, names='col1,col2,col3')

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
        intlist2 = array([[2,2]*3]*2)
        arrlist1 = [['dbe']*2]*3
        #arrlist2 = strings.array([['de']*2]*3)
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b = [[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=records.array(b, names='col1,col2,col3')

        # Get an offsetted bytearray
        r1 = r[1:]
        assert r1._byteoffset > 0
        
        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r1)

        # Read it again
        r2 = fileh.root.recarray.read()

        assert r1.tostring() == r2.tostring()
        
        fileh.close()
        os.remove(file)

    def test02(self):
        "Checking saving a large recarray with an offset in its buffer"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[[23,24,35]*6]*6]
        intlist2 = array([[[2,3,4]*6]*6])
        arrlist1 = [['dbe']*2]*3
        #arrlist2 = strings.array([['de']*2]*3)
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b=[[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=records.array(b*300, names='col1,col2,col3')

        # Get an offsetted recarray
        r1 = r[290:292]
        if verbose:
            print "\noffseted recarray --> ", r1
        assert r1._byteoffset > 0
        
        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r1)

        # Read it again
        r2 = fileh.root.recarray.read()

        assert r1.tostring() == r2.tostring()
        
        fileh.close()
        os.remove(file)

    def test03(self):
        "Checking saving a strided recarray with an offset in its buffer"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        intlist1 = [[[23,24,35]*6]*6]
        intlist2 = array([[[2,3,4]*6]*6])
        arrlist1 = [['dbe']*2]*3
        #arrlist2 = strings.array([['de']*2]*3)
        arrlist2 = [['de']*2]*3
        floatlist1 = [[1.2,2.3]*3]*4
        floatlist2 = array([[4.5,2.4]*3]*4)
        b = [[intlist1, arrlist1, floatlist1],[intlist2, arrlist2, floatlist2]]
        r=records.array(b*300, names='col1,col2,col3', shape=300)

        # Get an strided recarray
        r2 = r[::2]

        # Get an offsetted bytearray
        r1 = r2[148:]
        if verbose:
            print "\noffseted and strided recarray --> ", r1
        assert r1._byteoffset > 0
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

        # Take a number of records a bit greater
        nrows = int(table._v_maxTuples * 1.1)
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
        buffer = [[[["abcd"]*2]*2, ((1,1),(1,1)), 2, 3.1, 4.2, 5, "e"]]
        r=records.array(buffer*nrows,
                        formats='(2,2)a4,(2,2)i4,i2,f8,f4,i2,a1')  #*-*
        
        # Assign the value exceptions
        r.field("c2")[3] = ((2,2), (2,2))  #*-*
        r.field("c2")[4] = ((2,2), (2,2))  #*-*
        
        # Read the table in another recarray
        r2 = table.read()

        # This generates too much output. Activate only when
        # self._v_maxTuples is very small (<10)
        if verbose and 0:
            print "Table values:"
            for row in table.iterrows():
                print row
            print r2
            print "Record values:"
            print r

        assert r.tostring() == r2.tostring()
        
        fileh.close()
        os.remove(file)


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    #heavy = 1  # Uncomment this only for testing purposes

    #theSuite.addTest(unittest.makeSuite(CompressUCLTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(DefaultValues))
    #theSuite.addTest(unittest.makeSuite(RecArrayIO))
    #theSuite.addTest(unittest.makeSuite(BigTablesTestCase))

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(DictWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
        theSuite.addTest(unittest.makeSuite(CompressLZOTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressUCLTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressZLIBTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressTwoTablesTestCase))
        theSuite.addTest(unittest.makeSuite(IterRangeTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayRangeTestCase))
        theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
        theSuite.addTest(unittest.makeSuite(DefaultValues))
        theSuite.addTest(unittest.makeSuite(RecArrayIO))
    if heavy:
        theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
            
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
