import sys
import unittest
import os
import tempfile

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
    var3 = Col("Int16", 1)      # short integer. 
    var4 = Col("Float64", 1)    # double (double-precision)
    var5 = Col("Float32", 1)    # float  (single-precision)

class BasicTestCase(unittest.TestCase):
    file  = "test.h5"
    mode  = "w" 
    title = "This is the table title"
    expectedrows = 100
    appendrows = 20
    fast = 0
    compress = 0

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.fileh = openFile(self.file, self.mode)
        self.rootgroup = self.fileh.root
        self.populateFile()

    def populateFile(self):
        group = self.rootgroup
        maxshort = 1 << 15
        maxint   = 2147483647   # (2 ** 31 - 1)
        for j in range(3):
            # Create a table
            table = self.fileh.createTable(group, 'table'+str(j), Record(),
                                        title = self.title,
                                        compress = self.compress,
                                        expectedrows = self.expectedrows)
            # Get the row object associated with the new table
            d = table.row
	    
            # Fill the table
            for i in xrange(self.expectedrows):
                if self.fast:
                    table.appendAsValues('%04d' % (self.expectedrows - i),
                                         i,
                                         i % maxshort,
                                         float(i),
                                         float(i),)
                else:
                    d.var1 = '%04d' % (self.expectedrows - i)
                    d.var2 = i 
                    d.var3 = i % maxshort
                    d.var4 = float(i)
                    d.var5 = float(i)
                    table.appendAsRecord(d)
		
            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.createGroup(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2

    def tearDown(self):
        # Close the file (eventually destroy the extended type)
        self.fileh.close()

        os.remove(self.file)

    #----------------------------------------

    # """Checking table read and cuts"""
    def test01_readTable(self):

        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readTable..." % self.__class__.__name__

        table = self.fileh.getNode("/table0")
	
        # Read the records and select the ones with "var2" file less than 20
        result = [ rec.var2 for rec in table.readAsRecords() if rec.var2 < 20 ]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        assert (rec.var1, rec.var2, rec.var5) == ("0001", nrows, float(nrows))
        assert len(result) == 20
        
    # """Checking whether appending record rows works or not"""
    def test02_AppendRows(self):

        # First close the open file
        self.fileh.close()
        # Now, open it, but in "append" mode
        self.fileh = openFile(self.file, mode = "a")
        self.rootgroup = self.fileh.root
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_AppendRows..." % self.__class__.__name__

        maxshort = 1 << 15
        # Get a table
        table = self.fileh.getNode("/group0/table1")
        # Get their row object
        rec = table.row
        if verbose:
            print "Nrows in old", table._v_pathname, ":", table.nrows
            print "Record Format ==>", table._v_fmt
            print "Record Size ==>", table._v_rowsize
        # Append some records
        for i in xrange(self.appendrows):
            if self.fast:
                table.appendAsValues('%04d' % (self.appendrows - i),
                                     i,
                                     i % maxshort,
                                     float(i),
                                     float(i),)
            else:
                rec.var1 = '%04d' % (self.appendrows - i)
                rec.var2 = i 
                rec.var3 = i % maxshort
                rec.var4 = float(i)
                rec.var5 = float(i)
                table.appendAsRecord(rec)
	    
	# Flush the buffer for this table and read it
        table.flush()
        result = [ rec.var2 for rec in table.readAsRecords() if rec.var2 < 20 ]
	
        nrows = self.appendrows - 1
        assert (rec.var1, rec.var2, rec.var5) == ("0001", nrows, float(nrows))
        assert len(result) == 40 # because we appended new records
        
class BasicWriteTestCase(BasicTestCase):
    pass

class CompressTablesTestCase(BasicTestCase):
    compress = 1

class BigTablesTestCase(BasicTestCase):
    #expectedrows = 10000
    #appendrows = 1000
    expectedrows = 10000
    appendrows = 1000

class BigFastTablesTestCase(BasicTestCase):
    #expectedrows = 10000
    #appendrows = 1000
    expectedrows = 1000
    appendrows = 100
    fast = 1


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    theSuite.addTest(unittest.makeSuite(CompressTablesTestCase))
    theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(BigFastTablesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
