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

class TypesTestCase(unittest.TestCase):
    
    def test00_IsHDF5File(self):
        file = tempfile.mktemp(".h5")
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_IsHDF5File..." % self.__class__.__name__
            #print "Filename ==>", file

        # Create an instance of HDF5 Table
        fileh = File(filename = file, mode = "w")
        group = fileh.getRootGroup()
        # Create a table
        table = fileh.newTable(group, 'table', Record(),
                                    tableTitle = "Title example")
        # For this method to run, it needs a closed file
        fileh.close()
        assert isHDF5(file) == 1
        # Then, delete the file
        os.remove(file)

class RangeTestCase(unittest.TestCase):
    file  = "test.h5"
    title = "This is the table title"
    expectedrows = 100
    maxshort = 2 ** 15
    maxint   = 2147483648   # (2 ** 31)
    compress = 0

    def setUp(self):
        # Create an instance of HDF5 Table
        self.fileh = File(filename = self.file, mode = "w")
        self.rootgroup = self.fileh.getRootGroup()
        group = self.rootgroup
        # Create a table
        self.table = self.fileh.newTable(group, 'table', Record(), self.title)

    def tearDown(self):
        # Close the file (eventually destroy the extended type)
        self.fileh.close()

        os.remove(self.file)

    #----------------------------------------

    def test00_range(self):
        """Testing the range check"""
        rec = Record()
        # Save a record
        i = self.maxshort
        rec.var1 = '%04d' % (i)
        rec.var2 = i 
        #rec.var3 = i % self.maxshort
        rec.var3 = i
        rec.var4 = float(i)
        rec.var5 = float(i)
        try:
            self.table.appendRecord(rec)      # This injects the Record values
        except ValueError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            pass
        else:
            print rec
            self.fail("expected a ValueError")

    def test01_type(self):
        """Testing the type check"""
        rec = Record()
        # Save a record
        i = self.maxshort
        rec.var1 = '%04d' % (i)
        rec.var2 = i 
        rec.var3 = i % self.maxshort
        #rec.var3 = i
        rec.var4 = "124"
        rec.var5 = float(i)
        try:
            self.table.appendRecord(rec)      # This injects the Record values
        except ValueError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            pass
        else:
            print rec
            self.fail("expected a ValueError")

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(RangeTestCase))
    #theSuite.addTest(unittest.makeSuite(BasicWriteTestCase))
    #theSuite.addTest(unittest.makeSuite(CompressTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(BigTablesTestCase))
    #theSuite.addTest(unittest.makeSuite(BigFastTablesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
