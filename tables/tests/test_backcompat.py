import warnings
import unittest

import numpy

from tables import *
from tables.tests import common
from tables.tests.common import allequal, numeric_imported

if numeric_imported:
    import Numeric

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

# Check read Tables from pytables version 0.8
class BackCompatTablesTestCase(common.PyTablesTestCase):

    #----------------------------------------

    def test01_readTable(self):
        """Checking backward compatibility of old formats of tables"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        warnings.filterwarnings("ignore", category=UserWarning)
        self.fileh = openFile(self._testFilename(self.file), "r")
        warnings.filterwarnings("default", category=UserWarning)

        table = self.fileh.getNode("/tuple0")

        # Read the 100 records
        result = [ rec['var2'] for rec in table]
        if common.verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)

        self.assertEqual(len(result), 100)
        self.fileh.close()


class Table2_1LZO(BackCompatTablesTestCase):
    file = "Table2_1_lzo_nrv2e_shuffle.h5"  # pytables 0.8.x versions and after

class Tables_LZO1(BackCompatTablesTestCase):
    file = "Tables_lzo1.h5"  # files compressed with LZO1

class Tables_LZO1_shuffle(BackCompatTablesTestCase):
    file = "Tables_lzo1_shuffle.h5"  # files compressed with LZO1 and shuffle

class Tables_LZO2(BackCompatTablesTestCase):
    file = "Tables_lzo2.h5"  # files compressed with LZO2

class Tables_LZO2_shuffle(BackCompatTablesTestCase):
    file = "Tables_lzo2_shuffle.h5"  # files compressed with LZO2 and shuffle

# Check read attributes from PyTables >= 1.0 properly
class BackCompatAttrsTestCase(common.PyTablesTestCase):
    file = "zerodim-attrs-%s.h5"

    def test01_readAttr(self):
        """Checking backward compatibility of old formats for attributes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readAttr..." % self.__class__.__name__

        # Read old formats
        filename = self._testFilename(self.file)
        self.fileh = openFile(filename % self.format, "r")
        a = self.fileh.getNode("/a")
        scalar = numpy.array(1, dtype="int32")
        vector = numpy.array([1], dtype="int32")
        if self.format == "1.3":
            self.assertTrue(allequal(a.attrs.arrdim1, vector))
            self.assertTrue(allequal(a.attrs.arrscalar, scalar))
            self.assertEqual(a.attrs.pythonscalar, 1)
        elif self.format == "1.4":
            self.assertTrue(allequal(a.attrs.arrdim1, vector))
            self.assertTrue(allequal(a.attrs.arrscalar, scalar))
            self.assertTrue(allequal(a.attrs.pythonscalar, scalar))

        self.fileh.close()

class Attrs_1_3(BackCompatAttrsTestCase):
    format = "1.3"    # pytables 1.0.x versions and earlier

class Attrs_1_4(BackCompatAttrsTestCase):
    format = "1.4"    # pytables 1.1.x versions and later

class VLArrayTestCase(common.PyTablesTestCase):

    def test01_backCompat(self):
        """Checking backward compatibility with old flavors of VLArray"""

        # Open a PYTABLES_FORMAT_VERSION=1.6 file
        filename = self._testFilename("flavored_vlarrays-format1.6.h5")
        fileh = openFile(filename, "r")
        # Check that we can read the contents without problems (nor warnings!)
        vlarray1 = fileh.root.vlarray1
        self.assertEqual(vlarray1.flavor, "numeric")
        if numeric_imported:
            self.assertTrue(
                allequal(vlarray1[1], Numeric.array([5, 6, 7],  typecode='i'),
                         "numeric"))
        vlarray2 = fileh.root.vlarray2
        self.assertEqual(vlarray2.flavor, "python")
        self.assertEqual(vlarray2[1], ['5', '6', '77'])

        fileh.close()


# Make sure that 1.x files with TimeXX types continue to be readable
# and that its byteorder is correctly retrieved.
class TimeTestCase(common.PyTablesTestCase):

    def setUp(self):
        # Open a PYTABLES_FORMAT_VERSION=1.x file
        filename = self._testFilename("time-table-vlarray-1_x.h5")
        self.fileh = openFile(filename, "r")

    def tearDown(self):
        self.fileh.close()


    def test00_table(self):
        """Checking backward compatibility with old TimeXX types (tables)."""

        # Check that we can read the contents without problems (nor warnings!)
        table = self.fileh.root.table
        self.assertEqual(table.byteorder, "little")

    def test01_vlarray(self):
        """Checking backward compatibility with old TimeXX types (vlarrays)."""

        # Check that we can read the contents without problems (nor warnings!)
        vlarray4 = self.fileh.root.vlarray4
        self.assertEqual(vlarray4.byteorder, "little")
        vlarray8 = self.fileh.root.vlarray4
        self.assertEqual(vlarray8.byteorder, "little")


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    lzo_avail = whichLibVersion("lzo") is not None
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(VLArrayTestCase))
        theSuite.addTest(unittest.makeSuite(TimeTestCase))
        if lzo_avail:
            theSuite.addTest(unittest.makeSuite(Table2_1LZO))
            theSuite.addTest(unittest.makeSuite(Tables_LZO1))
            theSuite.addTest(unittest.makeSuite(Tables_LZO1_shuffle))
            theSuite.addTest(unittest.makeSuite(Tables_LZO2))
            theSuite.addTest(unittest.makeSuite(Tables_LZO2_shuffle))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
