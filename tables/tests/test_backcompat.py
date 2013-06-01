# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import warnings
import unittest

import numpy

from tables import *
from tables.exceptions import FlavorWarning
from tables.tests import common
from tables.tests.common import allequal

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
        self.fileh = open_file(self._testFilename(self.file), "r")
        warnings.filterwarnings("default", category=UserWarning)

        table = self.fileh.get_node("/tuple0")

        # Read the 100 records
        result = [rec['var2'] for rec in table]
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
        self.fileh = open_file(filename % self.format, "r")
        a = self.fileh.get_node("/a")
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
        fileh = open_file(filename, "r")
        # Check that we can read the contents without problems (nor warnings!)
        vlarray1 = fileh.root.vlarray1
        self.assertEqual(vlarray1.flavor, "numeric")
        vlarray2 = fileh.root.vlarray2
        self.assertEqual(vlarray2.flavor, "python")
        self.assertEqual(vlarray2[1], [b'5', b'6', b'77'])

        fileh.close()


# Make sure that 1.x files with TimeXX types continue to be readable
# and that its byteorder is correctly retrieved.
class TimeTestCase(common.PyTablesTestCase):

    def setUp(self):
        # Open a PYTABLES_FORMAT_VERSION=1.x file
        filename = self._testFilename("time-table-vlarray-1_x.h5")
        self.fileh = open_file(filename, "r")

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


class OldFlavorsTestCase01(common.PyTablesTestCase):
    close = False

    # numeric
    def test01_open(self):
        """Checking opening of (X)Array (old 'numeric' flavor)"""

        # Open the HDF5 with old numeric flavor
        filename = self._testFilename("oldflavor_numeric.h5")
        fileh = open_file(filename)

        # Assert other properties in array
        self.assertEqual(fileh.root.array1.flavor, 'numeric')
        self.assertEqual(fileh.root.array2.flavor, 'python')
        self.assertEqual(fileh.root.carray1.flavor, 'numeric')
        self.assertEqual(fileh.root.carray2.flavor, 'python')
        self.assertEqual(fileh.root.vlarray1.flavor, 'numeric')
        self.assertEqual(fileh.root.vlarray2.flavor, 'python')

        # Close the file
        fileh.close()

    def test02_copy(self):
        """Checking (X)Array.copy() method ('numetic' flavor)"""

        srcfile = self._testFilename("oldflavor_numeric.h5")
        tmpfile = tempfile.mktemp(".h5")
        shutil.copy(srcfile, tmpfile)

        # Open the HDF5 with old numeric flavor
        fileh = open_file(tmpfile, "r+")

        # Copy to another location
        self.failUnlessWarns(FlavorWarning,
                             fileh.root.array1.copy, '/', 'array1copy')
        fileh.root.array2.copy('/', 'array2copy')
        fileh.root.carray1.copy('/', 'carray1copy')
        fileh.root.carray2.copy('/', 'carray2copy')
        fileh.root.vlarray1.copy('/', 'vlarray1copy')
        fileh.root.vlarray2.copy('/', 'vlarray2copy')

        if self.close:
            fileh.close()
            fileh = open_file(tmpfile)

        else:
            fileh.flush()

        # Assert other properties in array
        self.assertEqual(fileh.root.array1copy.flavor, 'numeric')
        self.assertEqual(fileh.root.array2copy.flavor, 'python')
        self.assertEqual(fileh.root.carray1copy.flavor, 'numeric')
        self.assertEqual(fileh.root.carray2copy.flavor, 'python')
        self.assertEqual(fileh.root.vlarray1copy.flavor, 'numeric')
        self.assertEqual(fileh.root.vlarray2copy.flavor, 'python')

        # Close the file
        fileh.close()
        os.remove(tmpfile)


class OldFlavorsTestCase02(common.PyTablesTestCase):
    close = True

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    lzo_avail = which_lib_version("lzo") is not None
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(VLArrayTestCase))
        theSuite.addTest(unittest.makeSuite(TimeTestCase))
        theSuite.addTest(unittest.makeSuite(OldFlavorsTestCase01))
        theSuite.addTest(unittest.makeSuite(OldFlavorsTestCase02))
        if lzo_avail:
            theSuite.addTest(unittest.makeSuite(Table2_1LZO))
            theSuite.addTest(unittest.makeSuite(Tables_LZO1))
            theSuite.addTest(unittest.makeSuite(Tables_LZO1_shuffle))
            theSuite.addTest(unittest.makeSuite(Tables_LZO2))
            theSuite.addTest(unittest.makeSuite(Tables_LZO2_shuffle))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
