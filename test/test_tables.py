import sys
import unittest
import os
import tempfile

import numarray
from numarray import *
#import recarray
import numarray.records as records
from tables import *

from test_all import verbose

# Test Record class
class Record(IsDescription):
    var1 = StringCol(length=4, dflt="abcd")     # 4-character String
    var2 = IntCol(1)                            # integer
    var3 = Int16Col(2)                          # short integer 
    var4 = Float64Col(3.1)                      # double (double-precision)
    var5 = Float32Col(4.2)                      # float  (single-precision)
    var6 = UInt16Col(5)                         # unsigned short integer 
    var7 = StringCol(length=1, dflt="e")        # 1-character String

# From 0.3 on, you can dynamically define the tables with a dictionary
RecordDescriptionDict = {
    'var1': StringCol(4, "abcd"),               # 4-character String
    'var2': IntCol(1),                          # integer
    'var3': Int16Col(2),                        # short integer 
    'var4': FloatCol(3.1),                      # double (double-precision)
    'var5': Float32Col(4.2),                    # float  (single-precision)
    'var6': UInt16Col(5),                       # unsigned short integer 
    'var7': StringCol(1, "e"),                  # 1-character String
    }

# Old fashion of defining tables (for testing backward compatibility)
class OldRecord(IsDescription):
    var1 = Col("CharType", shape=4, dflt="abcd")   # 4-character String
    var2 = Col(Int32, 1, 1)                # integer
    var3 = Col(Int16, 1, 2)                # short integer
    var4 = Col("Float64", 1, 3.1)            # double (double-precision)
    var5 = Col("Float32", 1, 4.2)            # float  (single-precision)
    var6 = Col("UInt16", 1, 5)                # unisgned short integer 
    var7 = Col("CharType", shape=1, dflt="e")      # 1-character String

def allequal(a,b):
    """Checks if two numarrays are equal"""

    if a.shape <> b.shape:
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if str(equal(a,b)) == '1':
            return 1
        else:
            return 0

    # Multidimensional case
    result = (a == b)
    for i in range(len(a.shape)):
        result = logical_and.reduce(result)

    return result


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
        # Close the file (eventually destroy the extended type)
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
            var7 = var1[-1]
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
            table = self.fileh.createTable(group, 'table'+str(j), self.record,
                                           title = self.title,
                                           compress = self.compress,
                                           expectedrows = self.expectedrows,
                                           complib=self.complib)
            if not self.recarrayinit:
                # Get the row object associated with the new table
                row = table.row
	    
                # Fill the table
                for i in xrange(self.expectedrows):
                    row['var1'] = '%04d' % (self.expectedrows - i)
                    row['var7'] = row['var1'][-1]
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
        result = [ rec['var2'] for rec in table.iterrows()
                   if rec['var2'] < 20 ]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        assert (rec['var1'], rec['var2'], rec['var7']) == ("0001", nrows,"1")
        if isinstance(rec['var5'], NumArray):
            assert allequal(rec['var5'], array((float(nrows),)*4))
        else:
            assert rec['var5'] == float(nrows)
        assert len(result) == 20
        
    def test01b_readTable(self):
        """Checking table read and cuts (unidimensional columns case)"""

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
                   if rec['var2'] < 20 ]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "rec['var5'] ==>", rec['var5'],
            print "nrows ==>", table.row.nrow()
            print "Total selected records in table ==> ", len(result)
        nrows = table.row.nrow()
        if isinstance(rec['var5'], NumArray):
            assert allequal(result[0], array((float(0),)*4))
            assert allequal(result[1], array((float(1),)*4))
            assert allequal(result[2], array((float(2),)*4))
            assert allequal(result[3], array((float(3),)*4))
            assert allequal(result[10], array((float(10),)*4))
            assert allequal(rec['var5'], array((float(nrows),)*4))
        else:
            assert rec['var5'] == float(nrows)
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
            row['var7'] = row['var1'][-1]
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
            row.append()
	    
	# Flush the buffer for this table and read it
        table.flush()
        result = [ row['var2'] for row in table.iterrows()
                   if row['var2'] < 20 ]
	
        nrows = self.appendrows - 1
        assert (row['var1'], row['var2'], row['var7']) == ("0001", nrows, "1")
        if isinstance(row['var5'], NumArray):
            assert allequal(row['var5'], array((float(nrows),)*4))
        else:
            assert row['var5'] == float(nrows)
        if self.appendrows <= 20:
            add = self.appendrows
        else:
            add = 20
        assert len(result) == 20 + add  # because we appended new rows

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
	
        # Read the records and select the ones with "var6" column less than 20
        result = [ rec['var2'] for rec in table.iterrows() if rec['var6'] < 20]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Selected records ==>", result
            print "Total selected records in table ==>", len(result)
        nrows = self.expectedrows - 1
        assert (rec['var1'], rec['var6']) == ("0001", nrows)
        assert len(result) == 20

    def test04_delete(self):
        """Checking if a single row can be deleted"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "a")
        table = self.fileh.getNode("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [ r['var2'] for r in table.iterrows() if r['var2'] < 20]
        
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result[-1]
            print "Total selected records in table ==>", len(result)

        nrows = table.nrows
        # Delete the twenty-th row
        table.removeRows(19)

        # Re-read the records
        result2 = [ r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        assert table.nrows == nrows - 1
        # Check that the new list is smaller than the original one
        assert len(result) == len(result2) + 1
        assert result[:-1] == result2

    def test04b_delete(self):
        """Checking if a range of rows can be deleted"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "a")
        table = self.fileh.getNode("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [ r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result[-1]
            print "Total selected records in table ==>", len(result)

        nrows = table.nrows
        # Delete the last ten rows 
        table.removeRows(10, 20)

        # Re-read the records
        result2 = [ r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        assert table.nrows == nrows - 10
        # Check that the new list is smaller than the original one
        assert len(result) == len(result2) + 10
        assert result[:10] == result2

    def test04c_delete(self):
        """Checking if removing a bad range of rows is detected"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04c_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "a")
        table = self.fileh.getNode("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [ r['var2'] for r in table.iterrows() if r['var2'] < 20]
        
        nrows = table.nrows

        # Delete a too large range of rows 
        table.removeRows(10, nrows + 100)

        # Re-read the records
        result2 = [ r['var2'] for r in table.iterrows() if r['var2'] < 20]

        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        assert table.nrows == 10
        # Check that the new list is smaller than the original one
        assert len(result) == len(result2) + 10
        assert result[:10] == result2

    def test04d_delete(self):
        """Checking if removing several times at once is working"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04d_delete..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "a")
        table = self.fileh.getNode("/table0")

        # Read the records and select the ones with "var2" column less than 20
        result = [ r['var2'] for r in table if r['var2'] < 20]
        
        nrows = table.nrows

        # Delete some rows
        table.removeRows(10, 15)

        # Append some rows
        row = table.row
        for i in xrange(10, 15):
            row['var1'] = '%04d' % (self.appendrows - i)
            # This line gives problems on Windows. Why?
            #row['var7'] = row['var1'][-1]
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
            row.append()
	# Flush the buffer for this table
        table.flush()
        
        # Delete 5 rows more
        table.removeRows(5, 10)        

        # Re-read the records
        result2 = [ r['var2'] for r in table if r['var2'] < 20 ]

        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last selected value ==>", result2[-1]
            print "Total selected records in table ==>", len(result2)

        assert table.nrows == nrows - 5
        # Check that the new list is smaller than the original one
        assert len(result) == len(result2) + 5
        # The last values has to be equal
        assert result[10:15] == result2[10:15]

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
    maxTuples = 3  # Choose a small value for the buffer size
    start = 0
    stop = 10
    step = 3

class RecArrayOneWriteTestCase(BasicTestCase):
    title = "RecArrayOneWrite"
    record=records.array(formats="a4,i4,i2,2f8,f4,i2,a1",
                         names='var1,var2,var3,var4,var5,var6,var7')

class RecArrayTwoWriteTestCase(BasicTestCase):
    title = "RecArrayTwoWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=records.array(formats="a4,i4,i2,f8,f4,i2,a1",
                                 names='var1,var2,var3,var4,var5,var6,var7',
                                 shape=1)

class RecArrayThreeWriteTestCase(BasicTestCase):
    title = "RecArrayThreeWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=records.array(formats="a4,i4,i2,2f8,4f4,i2,a1",
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
    expectedrows = 10000
    appendrows = 1000
    #expectedrows = 100
    #appendrows = 10


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
        # Close the file (eventually destroy the extended type)
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        for j in range(3):
            # Create a table
            table = self.fileh.createTable(group, 'table'+str(j), self.record,
                                           title = self.title,
                                           compress = self.compress,
                                           expectedrows = self.expectedrows)
            # Get the row object associated with the new table
            row = table.row

            # Fill the table
            for i in xrange(self.expectedrows):
                row['var1'] = '%04d' % (self.expectedrows - i)
                row['var7'] = row['var1'][-1]
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
        if self.checkrecarray:
            #recarray = table.read(self.start, self.stop, self.step)
            recarray = table[self.start:self.stop:self.step]
            result = []
            for nrec in range(len(recarray)):
                if recarray.field('var2')[nrec] < self.nrows:
                    result.append(recarray.field('var2')[nrec])
        elif self.checkgetCol:
            column = table.read(self.start, self.stop, self.step, 'var2')
            result = []
            for nrec in range(len(column)):
                if column[nrec] < self.nrows:
                    result.append(column[nrec])
        else:
            result = [ rec['var2'] for rec in
                       table.iterrows(self.start, self.stop, self.step)
                       if rec['var2'] < self.nrows ]
        
        if self.start < 0:
            startr = self.expectedrows + self.start
        else:
            startr = self.start
            
        if self.stop <= 0:
            stopr = self.expectedrows + self.stop
        else:
            stopr = self.stop

        if self.nrows < stopr:
            stopr = self.nrows
            
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            if self.start < self.stop:
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

        assert result == range(startr, stopr, self.step)
        if startr < stopr and not (self.checkrecarray or self.checkgetCol):
            if self.nrows < self.expectedrows:
                assert rec['var2'] == range(self.start, self.stop, self.step)[-1]
            else:
                assert rec['var2'] == range(startr, stopr, self.step)[-1]

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

        # Case where stop = 0
        self.nrows = 100
        self.maxTuples = 3  # Choose a small value for the buffer size
        self.start = 1
        self.stop = 0
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
            #column = table.read(field='non-existent-column')
            column = table['non-existent-column']
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
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')

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
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')

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
        r=records.array('a'*200000,'f4,3i4,a5,i2',3000)

        # Get an offsetted bytearray
        r1 = r[2000:]
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
        r=records.array('a'*200000,'f4,3i4,a5,i2',3000)

        # Get an strided recarray
        r2 = r[::2]

        # Get an offsetted bytearray
        r1 = r2[1200:]
        assert r1._byteoffset > 0
        
        # Save it in a table:
        fileh.createTable(fileh.root, 'recarray', r1)

        # Read it again
        r2 = fileh.root.recarray.read()

        assert r1.tostring() == r2.tostring()
        
        fileh.close()
        os.remove(file)


class LargeRowSize(unittest.TestCase):

    def test00(self):
        "Checking saving a Table with a moderately large rowsize"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[arange(100)]*2])

        # Save it in a table:
        fileh.createTable(fileh.root, 'largerow', r)

        # Read it again
        r2 = fileh.root.largerow.read()

        assert r.tostring() == r2.tostring()
        
        fileh.close()
        os.remove(file)

    def test01(self):
        "Checking saving a Table with an extremely large rowsize"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[arange(1000)]*4])

        # Save it in a table:
        try:
            fileh.createTable(fileh.root, 'largerow', r)
        except RuntimeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
		print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected a RuntimeError")
            
        fileh.close()
        os.remove(file)


class DefaultValues(unittest.TestCase):
    record = Record

    def test00(self):
        "Checking saving a Table with default values"
        file = tempfile.mktemp(".h5")
        #file = "/tmp/test.h5"
        fileh = openFile(file, "w")

        # Create a table
        table = fileh.createTable(fileh.root, 'table', self.record)

        # Take a number of records a bit greater
        nrows = int(table._v_maxTuples * 1.1)
        # Fill the table with nrows records
        for i in xrange(nrows):
            if i == 3 or i == 4:
                table.row['var2'] = 2 
            # This injects the row values.
            table.row.append()

        # We need to flush the buffers in table in order to get an
        # accurate number of records on it.
        table.flush()

        # Create a recarray with the same default values
        r=records.array([["abcd", 1, 2, 3.1, 4.2, 5, "e"]]*nrows,
                          formats='a4,i4,i2,f8,f4,i2,a1')
        
        # Assign the value exceptions
        r.field("c2")[3] = 2 
        r.field("c2")[4] = 2
        
        # Read the table in another recarray
        #r2 = table.read()
        r2 = table[::]  # Equivalent to table.read()

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

class OldRecordDefaultValues(DefaultValues):
    title = "OldRecordDefaultValues"
    record = OldRecord


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    #theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
    #theSuite.addTest(unittest.makeSuite(CompressLZOTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(CompressUCLTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(IterRangeTestCase))

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(OldRecordBasicWriteTestCase))
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
        theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayIO))
        theSuite.addTest(unittest.makeSuite(LargeRowSize))
        theSuite.addTest(unittest.makeSuite(DefaultValues))
        theSuite.addTest(unittest.makeSuite(OldRecordDefaultValues))
            
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
