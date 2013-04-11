# -*- coding: utf-8 -*-

import unittest

from tables import *
from tables.tests import common
from tables.tests.common import verbose, cleanup


# Check indexes from PyTables version 2.0
class IndexesTestCase(common.PyTablesTestCase):

    def setUp(self):
        self.fileh = open_file(self._testFilename(self.file_), "r")
        self.table1 = self.fileh.root.table1
        self.table2 = self.fileh.root.table2
        self.il = 0
        self.sl = self.table1.cols.var1.index.slicesize

    def tearDown(self):
        self.fileh.close()
        cleanup(self)


    #----------------------------------------

    def test00_version(self):
        """Checking index version."""

        t1var1 = self.table1.cols.var1
        if "2_0" in self.file_:
            self.assertEqual(t1var1.index._v_version, "2.0")
        elif "2_1" in self.file_:
            self.assertEqual(t1var1.index._v_version, "2.1")


    def test01_string(self):
        """Checking string indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_string..." % self.__class__.__name__

        table1 = self.table1
        table2 = self.table2

        # Convert the limits to the appropriate type
        il = str(self.il).encode('ascii')
        sl = str(self.sl).encode('ascii')

        # Do some selections and check the results
        # First selection
        t1var1 = table1.cols.var1
        self.assertTrue(t1var1 is not None)
        results1 = [p["var1"] for p in
                    table1.where('(il<=t1var1)&(t1var1<=sl)')]
        results2 = [p["var1"] for p in table2 if il <= p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Superior & inferior limits:", il, sl
#             print "Selection results (index):", results1
            print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)


    def test02_bool(self):
        """Checking bool indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_bool..." % self.__class__.__name__

        table1 = self.table1
        table2 = self.table2

        # Do some selections and check the results
        t1var2 = table1.cols.var2
        self.assertTrue(t1var2 is not None)
        results1 = [p["var2"] for p in table1.where('t1var2 == True')]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == True]
        if verbose:
            print "Selection results (index):", results1
            print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)


    def test03_int(self):
        """Checking int indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_int..." % self.__class__.__name__

        table1 = self.table1
        table2 = self.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3
        self.assertTrue(t1col is not None)

        # First selection
        results1 = [p["var3"] for p in table1.where('(il<=t1col)&(t1col<=sl)')]
        results2 = [p["var3"] for p in table2
                    if il <= p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)


    def test04_float(self):
        """Checking float indexes"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_float..." % self.__class__.__name__

        table1 = self.table1
        table2 = self.table2

        # Convert the limits to the appropriate type
        il = float(self.il)
        sl = float(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var4
        self.assertTrue(t1col is not None)

        # First selection
        results1 = [p["var4"] for p in table1.where('(il<=t1col)&(t1col<=sl)')]
        results2 = [p["var4"] for p in table2
                    if il <= p["var4"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1.sort(), results2.sort())


# Check indexes from PyTables version 2.0
class Indexes2_0TestCase(IndexesTestCase):
    file_ = "indexes_2_0.h5"

# Check indexes from PyTables version 2.1
class Indexes2_1TestCase(IndexesTestCase):
    file_ = "indexes_2_1.h5"


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(Indexes2_0TestCase))
        theSuite.addTest(unittest.makeSuite(Indexes2_1TestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )






