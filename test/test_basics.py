import sys
import unittest
import os
import tempfile
from tables import File, Table, Group, IsRecord, isHDF5

from test_all import verbose

# Test Record class
class Record(IsRecord):
    """ A record has several columns. Represent the here as class
    variables, whose values are their types. The IsRecord
    class will take care the user won't add any new variables and
    that their type is correct.  """
    
    var1 = '4s'   # 4-character String
    var2 = 'i'    # integer
    var3 = 'h'    # short integer. This is chosen in this place for 
                  # discovery of alignment issues!
    var4 = 'd'    # double (double-precision)
    var5 = 'f'    # float  (single-precision)

class CheckFileTestCase(unittest.TestCase):
    
    def test00_IsHDF5File(self):
        file = tempfile.mktemp(".h5")
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_IsHDF5File..." % self.__class__.__name__
            #print "Filename ==>", file

        # Create an instance of HDF5 Table
        fileh = File(name = file, mode = "w")
        group = fileh.getRootGroup()
        # Create a table
        table = fileh.newTable(group, 'table', Record(),
                                    tableTitle = "Title example")
        # For this method to run, it needs a closed file
        fileh.close()
        assert isHDF5(file) == 1
        # Then, delete the file
        os.remove(file)

class BasicTestCase(unittest.TestCase):
    file  = "test.h5"
    mode  = "w" 
    title = "This is the table title"
    expectedrows = 1000
    appendrows = 100
    fast = 0
    compress = 0

    def setUp(self):
        # Create an instance of HDF5 Table
        self.fileh = File(name = self.file, mode = self.mode)
        self.rootgroup = self.fileh.getRootGroup()
        self.populateFile()

    def populateFile(self):
        group = self.rootgroup
        maxshort = 1 << 15
        maxint   = 2147483647   # (2 ** 31 - 1)
        for j in range(3):
            # Create a table
            table = self.fileh.newTable(group, 'table'+str(j), Record(),
                                        tableTitle = self.title,
                                        compress = self.compress,
                                        expectedrows = self.expectedrows)
            # Get the record object associated with the new table
            d = table.record 
            # Fill the table
            for i in xrange(self.expectedrows):
                d.var1 = '%04d' % (self.expectedrows - i)
                d.var2 = i 
                d.var3 = i % maxshort
                d.var4 = float(i)
                d.var5 = float(i)
                table.appendRecord(d)      # This injects the Record values
                # table.appendRecord(d())     # The same, but slower
            # Flush the buffer for this table
            table.flush()
            # Create a new group (descendant of group)
            group2 = self.fileh.newGroup(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2
    
    def tearDown(self):
        # Close the file (eventually destroy the extended type)
        self.fileh.close()

        os.remove(self.file)

    #----------------------------------------

    def test00_getGroups(self):
        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_getGroups..." % self.__class__.__name__

        groups = []
        for (groupname, groupobj) in self.fileh.walkGroups(rootgroup):
            groups.append(groupname)

        if verbose:
            print "Present groups in file ==>", groups
            
        assert groups == ["/", "group0", "group1", "group2"]

    def test01_getTable(self):
        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_getTable..." % self.__class__.__name__
            print "Testing if /group0/table1 is a Table instance..."

        table = self.fileh.getNode("/group0/table1")
        assert isinstance(table, Table.Table)
        
    def test02_readTable(self):
        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_readTable..." % self.__class__.__name__

        table = self.fileh.getNode("/table0")
        # Read the records and select the ones with "var2" file less than 20
        result = [ rec.var2 for rec in table.readAsRecords() if rec.var2 < 20 ]
        if verbose:
            print "Nrecords in", table._v_pathname, ":", table.nrecords
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)
        nrows = self.expectedrows - 1
        assert (rec.var1, rec.var2, rec.var5) == ("0001", nrows, float(nrows))
        assert len(result) == 20
        
    def test03_TraverseTree(self):
        rootgroup = self.rootgroup
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_TraverseTree..." % self.__class__.__name__

        groups = []
        leaves = []
        for (groupname, groupobj) in self.fileh.walkGroups(rootgroup):
            groups.append(groupobj._v_pathname)
            if verbose:
                print "Group found in ==>", groupobj._v_pathname
            for (name, leave) in self.fileh.listLeaves(groupobj):
                leaves.append(leave._v_pathname)
                if verbose:
                    print "Leave found in ==>", leave._v_pathname

        assert groups == ["/", "/group0", "/group0/group1",
                          "/group0/group1/group2"]
        assert leaves == ["/table0", "/group0/table1", "/group0/group1/table2"]
        
    def test04_AppendRows(self):
        # First close the open file
        self.fileh.close()
        # Now, open it, but in "append" mode
        self.fileh = File(name = self.file, mode = "a")
        self.rootgroup = self.fileh.getRootGroup()
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_AppendRows..." % self.__class__.__name__

        maxshort = 1 << 15
        # Get a table
        table = self.fileh.getNode("/group0/table1")
        # Get their record object
        rec = table.record
        if verbose:
            print "Nrecords in old", table._v_pathname, ":", table.nrecords
            print "Record Format ==>", rec._v_fmt
            print "Record Size ==>", table._v_rowsize
        # Append some records
        for i in xrange(self.appendrows):
            rec.var1 = '%04d' % (self.appendrows - i)
            rec.var2 = i 
            rec.var3 = i % maxshort
            rec.var4 = float(i)
            rec.var5 = float(i)
            table.appendRecord(rec)      # This injects the Record values
            # table.appendRecord(rec())     # The same, but slower
        # Flush the buffer for this table
        table.flush()
        # Read the records and select the ones with "var2" file less than 20
        result = [ rec.var2 for rec in table.readAsRecords() if rec.var2 < 20 ]
        nrows = self.appendrows - 1
        assert (rec.var1, rec.var2, rec.var5) == ("0001", nrows, float(nrows))
        assert len(result) == 40 # because we appended new records
        
class BasicWriteTestCase(BasicTestCase):
    pass

class CompressTablesTestCase(BasicTestCase):
    compress = 1

class BigTablesTestCase(BasicTestCase):
    expectedrows = 10000
    appendrows = 1000

class BigFastTablesTestCase(BasicTestCase):
    expectedrows = 10000
    appendrows = 1000
    fast = 1


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(CheckFileTestCase))
    theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    theSuite.addTest(unittest.makeSuite(CompressTablesTestCase))
    theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
    theSuite.addTest(unittest.makeSuite(BigFastTablesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
