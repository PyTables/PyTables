import sys
import unittest
import os
import tempfile

import numarray
from numarray import *
import numarray.records as records
from tables import *
from tables.hdf5Extension import getIndices

from test_all import verbose, allequal

# Test Record class
class Record(IsDescription):
    var1 = StringCol(length=4, dflt="abcd")     # 4-character String
    var2 = IntCol(1)                            # integer
    var3 = Int16Col(2)                          # short integer 
    var4 = Float64Col(3.1)                      # double (double-precision)
    var5 = Float32Col(4.2)                      # float  (single-precision)
    var6 = UInt16Col(5)                         # unsigned short integer 
    var7 = StringCol(length=1, dflt="e")        # 1-character String
    var8 = BoolCol(1)                           # boolean

# From 0.3 on, you can dynamically define the tables with a dictionary
RecordDescriptionDict = {
    'var1': StringCol(4, "abcd"),               # 4-character String
    'var2': IntCol(1),                          # integer
    'var3': Int16Col(2),                        # short integer 
    'var4': FloatCol(3.1),                      # double (double-precision)
    'var5': Float32Col(4.2),                    # float  (single-precision)
    'var6': UInt16Col(5),                       # unsigned short integer 
    'var7': StringCol(1, "e"),                  # 1-character String
    'var8': BoolCol(1),                         # boolean
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
    var8 = Col("Bool", shape=1, dflt=1)      # boolean

class BasicTestCase(unittest.TestCase):
    #file  = "test.h5"
    mode  = "w" 
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
            if isinstance(row.field('var8'), NumArray):
                tmplist.append([0, 10])  # should be equivalent to [0,1]
            else:
                tmplist.append(10) # should be equivalent to 1
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
            filterprops = Filters(complevel = self.compress,
                                  shuffle = self.shuffle,
                                  fletcher32 = self.fletcher32,
                                  complib = self.complib)
            table = self.fileh.createTable(group, 'table'+str(j), self.record,
                                           title = self.title,
                                           filters = filterprops,
                                           expectedrows = self.expectedrows)
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
                    if isinstance(row['var8'], NumArray):
                        row['var8'] = [0, 1]
                    else:
                        row['var8'] = 1
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
            assert allequal(rec['var5'], array((float(nrows),)*4, Float32))
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
            assert allequal(result[0], array((float(0),)*4, Float32))
            assert allequal(result[1], array((float(1),)*4, Float32))
            assert allequal(result[2], array((float(2),)*4, Float32))
            assert allequal(result[3], array((float(3),)*4, Float32))
            assert allequal(result[10], array((float(10),)*4, Float32))
            assert allequal(rec['var5'], array((float(nrows),)*4, Float32))
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
            if isinstance(row['var8'], NumArray):
                row['var8'] = [0, 1]
            else:
                row['var8'] = 1
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
            assert allequal(row['var5'], array((float(nrows),)*4, Float32))
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
        """Checking if removing rows several times at once is working"""

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
            if isinstance(row['var8'], NumArray):
                row['var8'] = [0, 1]
            else:
                row['var8'] = 1
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

    def test05_filtersTable(self):
        """Checking tablefilters"""

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_filtersTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/table0")

        # Check filters:
        if self.compress <> table.filters.complevel and verbose:
            print "Error in compress. Class:", self.__class__.__name__
            print "self, table:", self.compress, table.filters.complevel
        tinfo = whichLibVersion(self.complib)
        if tinfo[0] == 0:
            self.complib = "zlib"
        assert table.filters.complib == self.complib
        assert table.filters.complevel == self.compress
        if self.shuffle <> table.filters.shuffle and verbose:
            print "Error in shuffle. Class:", self.__class__.__name__
            print "self, table:", self.shuffle, table.filters.shuffle
        assert self.shuffle == table.filters.shuffle
        if self.fletcher32 <> table.filters.fletcher32 and verbose:
            print "Error in fletcher32. Class:", self.__class__.__name__
            print "self, table:", self.fletcher32, table.filters.fletcher32
        assert self.fletcher32 == table.filters.fletcher32

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
    record=records.array(formats="a4,i4,i2,2f8,f4,i2,a1,b1",
                         names='var1,var2,var3,var4,var5,var6,var7,var8')

class RecArrayTwoWriteTestCase(BasicTestCase):
    title = "RecArrayTwoWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=records.array(formats="a4,i4,i2,f8,f4,i2,a1,b1",
                                 names='var1,var2,var3,var4,var5,var6,var7,var8',
                                 shape=1)

class RecArrayThreeWriteTestCase(BasicTestCase):
    title = "RecArrayThreeWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=records.array(formats="a4,i4,i2,2f8,4f4,i2,a1,b1",
                                  names='var1,var2,var3,var4,var5,var6,var7,var8',
                                  shape=1)

class CompressLZOTablesTestCase(BasicTestCase):
    title = "CompressLZOTables"
    compress = 1
    complib = "lzo"
    
class CompressLZOShuffleTablesTestCase(BasicTestCase):
    title = "CompressLZOTables"
    compress = 1
    shuffle = 1
    complib = "lzo"
    
class CompressUCLTablesTestCase(BasicTestCase):
    title = "CompressUCLTables"
    compress = 1
    complib = "ucl"
    
class CompressUCLShuffleTablesTestCase(BasicTestCase):
    title = "CompressUCLTables"
    compress = 1
    shuffle = 1
    complib = "ucl"
    
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
    shuffle = 1
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
            filterprops = Filters(complevel = self.compress,
                                  shuffle = self.shuffle)
            table = self.fileh.createTable(group, 'table'+str(j), self.record,
                                           title = self.title,
                                           filters = filterprops,
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
        r = slice(self.start, self.stop, self.step)
        #resrange = r.indices(table.nrows)
        resrange = getIndices(r,table.nrows)
        reslength = len(range(*resrange))
        if self.checkrecarray:
            recarray = table.read(self.start, self.stop, self.step)
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

        if self.stop == None:
            stopr = startr+1                
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

        # Case where stop = None (last row)
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

        # Case where start < 0 and stop = None (last row)
        self.nrows = self.expectedrows
        self.maxTuples = 5  # Choose a small value for the buffer size
        self.start = -6
        self.startr = self.expectedrows + self.start
        self.stop = 0
        self.stop = None
        self.stopr = self.expectedrows
        self.step = 2

        self.check_range()

    def test10a_range(self):
        """Checking ranges in table iterators (case10a)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10a_range..." % self.__class__.__name__

        # Case where start < 0 and stop = 0
        self.nrows = self.expectedrows
        self.maxTuples = 5  # Choose a small value for the buffer size
        self.start = -6
        self.startr = self.expectedrows + self.start
        self.stop = 0
        self.stopr = self.expectedrows
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
        "Checking saving a regular recarray"
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

    def test04(self):
        "Checking appending several rows at once"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        class Rec(IsDescription):
            col1 = IntCol(pos=1)
            col2 = StringCol(3, pos=2)
            col3 = FloatCol(pos=3)

        # Save it in a table:
        table = fileh.createTable(fileh.root, 'recarray', Rec)

        # append new rows
        r=records.array([[456,'dbe',1.2],[2,'ded',1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457,'db1',1.2],[5,'de1',1.3]])
        # Create the complete table
        r1=records.array([[456,'dbe',1.2],[2,'ded',1.3],
                          [457,'db1',1.2],[5,'de1',1.3]],
                         formats="i4,a3,f8",
                         names = "col1,col2,col3")
        # Read the original table
        r2 = fileh.root.recarray.read()
        if verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        assert r1.tostring() == r2.tostring()
        assert table.nrows == 4

        fileh.close()
        os.remove(file)

    def test05(self):
        "Checking appending several rows at once (close file version)"
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        class Rec(IsDescription):
            col1 = IntCol(pos=1)
            col2 = StringCol(3, pos=2)
            col3 = FloatCol(pos=3)

        # Save it in a table:
        table = fileh.createTable(fileh.root, 'recarray', Rec)

        # append new rows
        r=records.array([[456,'dbe',1.2],[2,'ded',1.3]], formats="i4,a3,f8")
        table.append(r)
        table.append([[457,'db1',1.2],[5,'de1',1.3]])

        fileh.close()
        fileh = openFile(file, "r")
        table = fileh.root.recarray
        
        # Create the complete table
        r1=records.array([[456,'dbe',1.2],[2,'ded',1.3],
                          [457,'db1',1.2],[5,'de1',1.3]],
                         formats="i4,a3,f8",
                         names = "col1,col2,col3")
        # Read the original table
        r2 = fileh.root.recarray.read()
        if verbose:
            print "Original table-->", repr(r2)
            print "Should look like-->", repr(r1)
        assert r1.tostring() == r2.tostring()
        assert table.nrows == 4

        fileh.close()
        os.remove(file)


class CopyTestCase(unittest.TestCase):

    def test01_copy(self):
        """Checking Table.copy() method """

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")

        # Copy to another table
        table2 = table1.copy('/', 'table2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            table1 = fileh.root.table1
            table2 = fileh.root.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            #print "dirs-->", dir(table1), dir(table2)
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow()   # current row
            # row1 is a Row instance, while table2[] is a
            # RecArray.Record instance
            #print "reprs-->", repr(row1), repr(table2.read(nrow))
            for colname in table1.colnames:
                # Both ways to compare works well
                assert row1[colname] == table2[nrow].field(colname)
                #assert row1[colname] == table2.read(nrow, field=colname)[0]

        # Assert other properties in table
        assert table1.nrows == table2.nrows
        assert table1.colnames == table2.colnames
        assert table1.coltypes == table2.coltypes
        assert table1.colshapes == table2.colshapes
        # This could be not the same when re-opening the file
        #assert table1.description._v_ColObjects == table2.description._v_ColObjects
        # Leaf attributes
        assert table1.title == table2.title
        assert table1.filters.complevel == table2.filters.complevel
        assert table1.filters.complib == table2.filters.complib
        assert table1.filters.shuffle == table2.filters.shuffle
        assert table1.filters.fletcher32 == table2.filters.fletcher32

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_copy(self):
        """Checking Table.copy() method (where specified)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")

        # Copy to another table in another group
        group1 = fileh.createGroup("/", "group1")
        table2 = table1.copy(group1, 'table2')

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow()   # current row
            for colname in table1.colnames:
                # Both ways to compare works well
                assert row1[colname] == table2[nrow].field(colname)
                #assert row1[colname] == table2.read(nrow, field=colname)[0]

        # Assert other properties in table
        assert table1.nrows == table2.nrows
        assert table1.colnames == table2.colnames
        assert table1.coltypes == table2.coltypes
        assert table1.colshapes == table2.colshapes
        # Leaf attributes
        assert table1.title == table2.title
        assert table1.filters.complevel == table2.filters.complevel
        assert table1.filters.complib == table2.filters.complib
        assert table1.filters.shuffle == table2.filters.shuffle
        assert table1.filters.fletcher32 == table2.filters.fletcher32

        # Close the file
        fileh.close()
        os.remove(file)

    def test03_copy(self):
        """Checking Table.copy() method (table larger than buffer)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray exceeding buffers capability
        # This works, but takes too much CPU for a test
        # It is better to reduce the buffer size (table1._v_maxTuples)
#         r=records.array('aaaabbbbccccddddeeeeffffgggg'*20000,
#                         formats='2i2,i4, (2,3)u2, (1,)f4, f8',shape=700)
        r=records.array('aaaabbbbccccddddeeeeffffgggg'*200,
                        formats='2i2,i4, (2,3)u2, (1,)f4, f8',shape=7)
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")
        
        # Copy to another table in another group and other title
        group1 = fileh.createGroup("/", "group1")
        table1._v_maxTuples = 2  # small value of buffer
        table2 = table1.copy(group1, 'table2', title="title table2")
        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow()   # current row
            for colname in table1.colnames:
                assert allequal(row1[colname], table2[nrow].field(colname))

        # Assert other properties in table
        assert table1.nrows == table2.nrows
        assert table1.colnames == table2.colnames
        assert table1.coltypes == table2.coltypes
        assert table1.colshapes == table2.colshapes
        # Leaf attributes
        assert "title table2" == table2.title
        assert table1.filters.complevel == table2.filters.complevel
        assert table1.filters.complib == table2.filters.complib
        assert table1.filters.shuffle == table2.filters.shuffle
        assert table1.filters.fletcher32 == table2.filters.fletcher32

        # Close the file
        fileh.close()
        os.remove(file)

    def test04_copy(self):
        """Checking Table.copy() method (different compress level)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")

        # Copy to another table in another group
        group1 = fileh.createGroup("/", "group1")
        table2 = table1.copy(group1, 'table2',
                             filters=Filters(complevel=6))

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow()   # current row
            for colname in table1.colnames:
                # Both ways to compare works well
                assert row1[colname] == table2[nrow].field(colname)
                #assert row1[colname] == table2.read(nrow, field=colname)[0]

        # Assert other properties in table
        assert table1.nrows == table2.nrows
        assert table1.colnames == table2.colnames
        assert table1.coltypes == table2.coltypes
        assert table1.colshapes == table2.colshapes
        # Leaf attributes
        assert table1.title == table2.title
        assert 6 == table2.filters.complevel
        assert table1.filters.complib == table2.filters.complib
        assert 1 == table2.filters.shuffle
        assert table1.filters.fletcher32 == table2.filters.fletcher32

        # Close the file
        fileh.close()
        os.remove(file)

    def test05_copy(self):
        """Checking Table.copy() method (user attributes copied)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")
        # Add some user attributes
        table1.attrs.attr1 = "attr1"
        table1.attrs.attr2 = 2
        # Copy to another table in another group
        group1 = fileh.createGroup("/", "group1")
        table2 = table1.copy(group1, 'table2',
                             copyuserattrs=1,
                             filters=Filters(complevel=6))

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow()   # current row
            for colname in table1.colnames:
                assert row1[colname] == table2[nrow].field(colname)

        # Assert other properties in table
        assert table1.nrows == table2.nrows
        assert table1.colnames == table2.colnames
        assert table1.coltypes == table2.coltypes
        assert table1.colshapes == table2.colshapes
        # Leaf attributes
        assert table1.title == table2.title
        assert 6 == table2.filters.complevel
        assert table1.filters.complib == table2.filters.complib
        assert 1 == table2.filters.shuffle
        assert table1.filters.fletcher32 == table2.filters.fletcher32
        # User attributes
        table2.attrs.attr1 == "attr1"
        table2.attrs.attr2 == 2

        # Close the file
        fileh.close()
        os.remove(file)

    def test05b_copy(self):
        """Checking Table.copy() method (user attributes not copied)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_copy..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray
        r=records.array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")
        # Add some user attributes
        table1.attrs.attr1 = "attr1"
        table1.attrs.attr2 = 2
        # Copy to another table in another group
        group1 = fileh.createGroup("/", "group1")
        table2 = table1.copy(group1, 'table2',
                             copyuserattrs=0,
                             filters=Filters(complevel=6))

        if self.close:
            if verbose:
                print "(closing file version)"
            fileh.close()
            fileh = openFile(file, mode = "r")
            table1 = fileh.root.table1
            table2 = fileh.root.group1.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        for row1 in table1:
            nrow = row1.nrow()   # current row
            for colname in table1.colnames:
                assert row1[colname] == table2[nrow].field(colname)

        # Assert other properties in table
        assert table1.nrows == table2.nrows
        assert table1.colnames == table2.colnames
        assert table1.coltypes == table2.coltypes
        assert table1.colshapes == table2.colshapes
        # Leaf attributes
        assert table1.title == table2.title
        assert 6 == table2.filters.complevel
        assert table1.filters.complib == table2.filters.complib
        assert 1 == table2.filters.shuffle
        assert table1.filters.fletcher32 == table2.filters.fletcher32
        # User attributes
        table2.attrs.attr1 == None
        table2.attrs.attr2 == None

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

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_index..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray exceeding buffers capability
        r=records.array('aaaabbbbccccddddeeeeffffgggg'*200,
                        #formats='2i2,i4, (2,3)u2, (1,)f4, f8',shape=10)
                        # to avoid an ugly warning
                        formats='2i2,i4, (2,3)u2, f4, f8',shape=10)
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")
        
        # Copy to another table
        table1._v_maxTuples = self.maxTuples
        table2 = table1.copy("/", 'table2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        for nrow in range(r2.shape[0]):
            for colname in table1.colnames:
                # The next gives a warning because a Table cannot distinguish
                # between a '(1,)f4' format and a 'f4' format.
                # This should be adressed? 2004-01-24
                assert allequal(r2[nrow].field(colname),
                                table2[nrow].field(colname))

        # Assert the number of rows in table
        if verbose:
            print "nrows in table2-->", table2.nrows
            print "and it should be-->", r2.shape[0]
        assert r2.shape[0] == table2.nrows

        # Close the file
        fileh.close()
        os.remove(file)

    def test02_indexclosef(self):
        """Checking Table.copy() method with indexes (close file version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_indexclosef..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")

        # Create a recarray exceeding buffers capability
        r=records.array('aaaabbbbccccddddeeeeffffgggg'*200,
                        formats='2i2,i4, (2,3)u2, f4, f8',shape=10)
        # Save it in a table:
        table1 = fileh.createTable(fileh.root, 'table1', r, "title table1")
        
        # Copy to another table
        table1._v_maxTuples = self.maxTuples
        table2 = table1.copy("/", 'table2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)

        fileh.close()
        fileh = openFile(file, mode = "r")
        table1 = fileh.root.table1
        table2 = fileh.root.table2

        if verbose:
            print "table1-->", table1.read()
            print "table2-->", table2.read()
            print "attrs table1-->", repr(table1.attrs)
            print "attrs table2-->", repr(table2.attrs)
            
        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        for nrow in range(r2.shape[0]):
            for colname in table1.colnames:
                assert allequal(r2[nrow].field(colname),
                                table2[nrow].field(colname))

        # Assert the number of rows in table
        if verbose:
            print "nrows in table2-->", table2.nrows
            print "and it should be-->", r2.shape[0]
        assert r2.shape[0] == table2.nrows

        # Close the file
        fileh.close()
        os.remove(file)

class CopyIndex1TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 0
    stop = 7
    step = 1

class CopyIndex2TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 0
    stop = -1
    step = 1

class CopyIndex3TestCase(CopyIndexTestCase):
    maxTuples = 3
    start = 1
    stop = 7
    step = 1

class CopyIndex4TestCase(CopyIndexTestCase):
    maxTuples = 4
    start = 0
    stop = 6
    step = 1

class CopyIndex5TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 3
    stop = 7
    step = 1

class CopyIndex6TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 3
    stop = 6
    step = 2

class CopyIndex7TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 0
    stop = 7
    step = 10

class CopyIndex8TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 6
    stop = 3
    step = 1

class CopyIndex9TestCase(CopyIndexTestCase):
    maxTuples = 2
    start = 3
    stop = 4
    step = 1

class CopyIndex10TestCase(CopyIndexTestCase):
    maxTuples = 1
    start = 3
    stop = 4
    step = 2

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
        r=records.array([["abcd", 1, 2, 3.1, 4.2, 5, "e", 1]]*nrows,
                          formats='a4,i4,i2,f8,f4,i2,a1,b1')
        
        # Assign the value exceptions
        r.field("c2")[3] = 2 
        r.field("c2")[4] = 2
        
        # Read the table in another recarray
        #r2 = table.read()
        r2 = table[::]  # Equivalent to table.read()

        # This generates too much output. Activate only when
        # self._v_maxTuples is very small (<10)
        if verbose and 1:
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

    #theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
    #theSuite.addTest(unittest.makeSuite(CompressLZOTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(CompressUCLTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
    #theSuite.addTest(unittest.makeSuite(DefaultValues))
    #theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(IterRangeTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayRangeTestCase))
    #theSuite.addTest(unittest.makeSuite(LargeRowSize)) 
    #theSuite.addTest(unittest.makeSuite(DefaultValues))
    #theSuite.addTest(unittest.makeSuite(OldRecordDefaultValues))
    #theSuite.addTest(unittest.makeSuite(OpenCopyTestCase))
    #theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
    #theSuite.addTest(unittest.makeSuite(Fletcher32TablesTestCase))
    #theSuite.addTest(unittest.makeSuite(AllFiltersTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(RecArrayIO))

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(OldRecordBasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(DictWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
        theSuite.addTest(unittest.makeSuite(CompressLZOTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressLZOShuffleTablesTestCase))
	theSuite.addTest(unittest.makeSuite(CompressUCLTablesTestCase))
	theSuite.addTest(unittest.makeSuite(CompressUCLShuffleTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressZLIBTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressZLIBShuffleTablesTestCase))
        theSuite.addTest(unittest.makeSuite(Fletcher32TablesTestCase))
        theSuite.addTest(unittest.makeSuite(AllFiltersTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressTwoTablesTestCase))
        theSuite.addTest(unittest.makeSuite(IterRangeTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayRangeTestCase))
        theSuite.addTest(unittest.makeSuite(getColRangeTestCase))
        theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayIO))
        theSuite.addTest(unittest.makeSuite(OpenCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex2TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex3TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex4TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex5TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex6TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex7TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex8TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex9TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex10TestCase))
        theSuite.addTest(unittest.makeSuite(LargeRowSize))
        theSuite.addTest(unittest.makeSuite(DefaultValues))
        theSuite.addTest(unittest.makeSuite(OldRecordDefaultValues))
            
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
