import sys
import unittest
import os

from tables import *
from tables.tests import common

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

# Test Record class
class Record(IsDescription):
    var1 = StringCol(itemsize=4)  # 4-character String
    var2 = Col.from_kind('int')   # integer
    var3 = Col.from_kind('int', itemsize=2) # short integer
    var4 = Col.from_kind('float') # double (double-precision)
    var5 = Col.from_kind('float', itemsize=4) # float  (single-precision)


class RangeTestCase(unittest.TestCase):
    file  = "test.h5"
    title = "This is the table title"
    expectedrows = 100
    maxshort = 2 ** 15
    maxint   = 2147483648   # (2 ** 31)
    compress = 0

    def setUp(self):
        # Create an instance of HDF5 Table
        self.fileh = openFile(self.file, mode = "w")
        self.rootgroup = self.fileh.root

        # Create a table
        self.table = self.fileh.createTable(self.rootgroup, 'table',
                                            Record, self.title)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test00_range(self):
        """Testing the range check"""
        rec = self.table.row
        # Save a record
        i = self.maxshort
        rec['var1'] = '%04d' % (i)
        rec['var2'] = i
        rec['var3'] = i
        rec['var4'] = float(i)
        rec['var5'] = float(i)
        try:
            rec.append()
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            pass
        else:
            if common.verbose:
                print "\nNow, the range overflow no longer issues a ValueError"

    def test01_type(self):
        """Testing the type check"""
        rec = self.table.row
        # Save a record
        i = self.maxshort
        rec['var1'] = '%04d' % (i)
        rec['var2'] = i
        rec['var3'] = i % self.maxshort
        rec['var5'] = float(i)
        try:
            rec['var4'] = "124c"
        except TypeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next TypeError was catched!"
                print value
            pass
        else:
            print rec
            self.fail("expected a TypeError")


# Check the dtype read-only attribute
class DtypeTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test00a_table(self):
        """Check dtype accessor for Table objects"""
        a = self.h5file.createTable('/', 'table', Record)
        self.assertEqual(a.dtype, a.description._v_dtype)

    def test00b_column(self):
        """Check dtype accessor for Column objects"""
        a = self.h5file.createTable('/', 'table', Record)
        c = a.cols.var3
        self.assertEqual(c.dtype, a.description._v_dtype['var3'])

    def test01_array(self):
        """Check dtype accessor for Array objects"""
        a = self.h5file.createArray('/', 'array', [1,2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test02_carray(self):
        """Check dtype accessor for CArray objects"""
        a = self.h5file.createCArray('/', 'array', FloatAtom(), [1,2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test03_carray(self):
        """Check dtype accessor for EArray objects"""
        a = self.h5file.createEArray('/', 'array', FloatAtom(), [0,2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test04_vlarray(self):
        """Check dtype accessor for VLArray objects"""
        a = self.h5file.createVLArray('/', 'array', FloatAtom())
        self.assertEqual(a.dtype, a.atom.dtype)


#----------------------------------------------------------------------

def suite():
    import doctest
    import tables.atom

    theSuite = unittest.TestSuite()

    for i in range(1):
        theSuite.addTest(doctest.DocTestSuite(tables.atom))
        theSuite.addTest(unittest.makeSuite(RangeTestCase))
        theSuite.addTest(unittest.makeSuite(DtypeTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
