import sys
import unittest
import os
import tempfile

from numarray import *
import recarray
from tables import *

from test_all import verbose

# Test Record class
class Record(IsRecord):
    """ A record has several columns. Represent the here as class
    variables, whose values are their types. The IsRecord
    class will take care the user won't add any new variables and
    that their type is correct.  """
    
    var1 = Col("CharType", 4)   # 4-character String
    var2 = Col("Int32", 1)      # integer
    var3 = Col("Int16", 1)      # short integer 
    var4 = Col("Float64", 1)    # double (double-precision)
    var5 = Col("Float32", 1)    # float  (single-precision)
    var6 = Col("Int16", 1)      # short integer 

# In 0.3, you can dynamically define the tables with a dictionary
RecordDescriptionDict = {
    'var1': Col("CharType", 4),   # 4-character String
    'var2': Col("Int32", 1),      # integer
    'var3': Col("Int16", 1),      # short integer 
    'var4': Col("Float64", 2),    # double (double-precision)
    'var5': Col("Float32", 4),    # float  (single-precision)
    'var6': Col("Int16", 1),      # short integer 
    }

# And even as an idle (or non-idle) recarray object!
RecArrayDescription=recarray.array(formats="4a,i4,i2,2f8,f4,i2",
                                   names='var1,var2,var3,var4,var5,var6')

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
    file  = "test.h5"
    mode  = "w" 
    title = "This is the table title"
    expectedrows = 100
    appendrows = 20
    compress = 0
    record = Record()
    recarrayinit = 0
    maxshort = 1 << 15

    def setUp(self):
        # Create an instance of an HDF5 Table
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
            # bar6 will be like var2 but byteswaped
            tmplist.append(((var3>>8) & 0xff) + ((var3<<8) & 0xff00))
            buflist.append(tmplist)

        self.record=recarray.array(buflist, formats=record._formats,
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
                                           expectedrows = self.expectedrows)
            if not self.recarrayinit:
                # Get the row object associated with the new table
                row = table.row
	    
                # Fill the table
                for i in xrange(self.expectedrows):
                    row.var1 = '%04d' % (self.expectedrows - i)
                    row.var2 = i 
                    row.var3 = i % self.maxshort
                    if isinstance(row.var4, NumArray):
                        row.var4 = [float(i), float(i*i)]
                    else:
                        row.var4 = float(i)
                    if isinstance(row.var5, NumArray):
                        row.var5 = array((float(i),)*4)
                    else:
                        row.var5 = float(i)
                    # bar6 will be like var2 but byteswaped
                    row.var6 = ((row.var3>>8) & 0xff) + ((row.var3<<8) & 0xff00)
                    table.append(row)
		
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
	
        # Read the records and select those with "var2" file less than 20
        result = [ rec['var2'] for rec in table.fetchall()
                   if rec['var2'] < 20 ]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        assert (rec.var1, rec.var2) == ("0001", nrows)
        if isinstance(rec.var5, NumArray):
            assert allequal(rec.var5, array((float(nrows),)*4))
        else:
            assert rec.var5 == float(nrows)
        assert len(result) == 20
        #del table
        
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
            print "Record Size ==>", table._v_rowsize
        # Append some rows
        for i in xrange(self.appendrows):
            row.var1 = '%04d' % (self.appendrows - i)
            row.var2 = i 
            row.var3 = i % self.maxshort
            if isinstance(row.var4, NumArray):
                row.var4 = [float(i), float(i*i)]
            else:
                row.var4 = float(i)
            if isinstance(row.var5, NumArray):
                row.var5 = array((float(i),)*4)
            else:
                row.var5 = float(i)
            table.append(row)
	    
	# Flush the buffer for this table and read it
        table.flush()
        result = [ row.var2 for row in table.fetchall() if row.var2 < 20 ]
	
        nrows = self.appendrows - 1
        assert (row.var1, row.var2) == ("0001", nrows)
        if isinstance(row.var5, NumArray):
            assert allequal(row.var5, array((float(nrows),)*4))
        else:
            assert row.var5 == float(nrows)
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
        table._v_byteorder = {"little":"big","big":"little"}[table._v_byteorder]
	
        # Read the records and select the ones with "var6" column less than 20
        result = [ rec.var2 for rec in table.fetchall() if rec.var6 < 20]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==>", len(result)
        nrows = self.expectedrows - 1
        assert (rec.var1, rec.var6) == ("0001", nrows)
        assert len(result) == 20
        #del table
        
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
    #record = RecArrayDescription
    title = "RecArrayOneWrite"
    record=recarray.array(formats="4a,i4,i2,2f8,f4,i2",
                          names='var1,var2,var3,var4,var5,var6')

class RecArrayTwoWriteTestCase(BasicTestCase):
    title = "RecArrayTwoWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=recarray.array(formats="4a,i4,i2,f8,f4,i2",
                                  names='var1,var2,var3,var4,var5,var6',
                                  shape=1)

class RecArrayThreeWriteTestCase(BasicTestCase):
    title = "RecArrayThreeWrite"
    expectedrows = 100
    recarrayinit = 1
    recordtemplate=recarray.array(formats="4a,i4,i2,2f8,4f4,i2",
                                  names='var1,var2,var3,var4,var5,var6',
                                  shape=1)

class CompressOneTablesTestCase(BasicTestCase):
    title = "CompressOneTables"
    compress = 1

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
    file  = "test.h5"
    mode  = "w" 
    title = "This is the table title"
    record = Record()
    maxshort = 1 << 15
    expectedrows = 100
    compress = 0
    # Default values
    nrows = 20
    maxTuples = 3  # Choose a small value for the buffer size
    start = 1
    stop = nrows

    def setUp(self):
        # Create an instance of an HDF5 Table
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
                row.var1 = '%04d' % (self.expectedrows - i)
                row.var2 = i 
                row.var3 = i % self.maxshort
                if isinstance(row.var4, NumArray):
                    row.var4 = [float(i), float(i*i)]
                else:
                    row.var4 = float(i)
                if isinstance(row.var5, NumArray):
                    row.var5 = array((float(i),)*4)
                else:
                    row.var5 = float(i)
                # bar6 will be like var2 but byteswaped
                row.var6 = ((row.var3>>8) & 0xff) + ((row.var3<<8) & 0xff00)
                table.append(row)
		
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

    def check_range(self):

        rootgroup = self.rootgroup
        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, "r")
        table = self.fileh.getNode("/table0")

        table._v_maxTuples = self.maxTuples
        if self.checkrecarray:
            recarray = table.getRecArray(self.start, self.stop, self.step)
            result = []
            for nrec in range(len(recarray)):
                if recarray.field('var2')[nrec] < self.nrows:
                    result.append(recarray.field('var2')[nrec])
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
                else:
                    print "Last record *read* in table range ==>", rec
            print "Total number of selected records ==>", len(result)
            print "Selected records:\n", result
            print "Selected records should look like:\n", \
                  range(startr, stopr, self.step)
            print "start, stop, step ==>", startr, stopr, self.step

        assert result == range(startr, stopr, self.step)
        if startr < stopr and not self.checkrecarray:
            if self.nrows < self.expectedrows:
                assert rec.var2 == range(self.start, self.stop, self.step)[-1]
            else:
                assert rec.var2 == range(startr, stopr, self.step)[-1]

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
        """Checking ranges in table iterators (case11)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_range..." % self.__class__.__name__

        # Case where step < 0 
        self.step = -11
        try:
            self.check_range()
        except ValueError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            pass
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
            pass
        else:
            print rec
            self.fail("expected a ValueError")


class IterRangeTestCase(BasicRangeTestCase):
    checkrecarray = 0

class RecArrayRangeTestCase(BasicRangeTestCase):
    checkrecarray = 1

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
        theSuite.addTest(unittest.makeSuite(DictWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayOneWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayTwoWriteTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayThreeWriteTestCase))
        theSuite.addTest(unittest.makeSuite(CompressOneTablesTestCase))
        theSuite.addTest(unittest.makeSuite(CompressTwoTablesTestCase))
        theSuite.addTest(unittest.makeSuite(IterRangeTestCase))
        theSuite.addTest(unittest.makeSuite(RecArrayRangeTestCase))
        theSuite.addTest(unittest.makeSuite(BigTablesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
