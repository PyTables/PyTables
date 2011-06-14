import unittest
import os
import tempfile
import random
import new

import numpy

from tables import *
from tables.idxutils import calcChunksize
from tables.tests import common
from tables.tests.common import verbose, heavy, cleanup

# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

# An alias for frozenset
fzset = frozenset

# To make the tests values reproductibles
random.seed(19)

# Sensible parameters for indexing with small blocksizes
small_blocksizes = (16, 8, 4, 2)  # The smaller set of parameters...
# The size for medium indexes
minRowIndex = 1000

class Small(IsDescription):
    var1 = StringCol(itemsize=4, dflt="")
    var2 = BoolCol(dflt=0)
    var3 = IntCol(dflt=0)
    var4 = FloatCol(dflt=0)

class SelectValuesTestCase(unittest.TestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 0
    chunkshape = 10
    buffersize = 0
    random = 0
    values = None
    reopen = False

    def setUp(self):
        # Create an instance of an HDF5 Table
        if verbose:
            print "Checking index kind-->", self.kind
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()

    def populateFile(self):
        # Set a seed for the random generator if needed.
        # This is useful when one need reproductible results.
        if self.random and hasattr(self, "seed"):
            random.seed(self.seed)
        group = self.rootgroup
        # Create an table
        title = "This is the IndexArray title"
        rowswritten = 0
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        table1 = self.fileh.createTable(group, 'table1', Small, title,
                                        filters, self.nrows,
                                        chunkshape=(self.chunkshape,))
        table2 = self.fileh.createTable(group, 'table2', Small, title,
                                        filters, self.nrows,
                                        chunkshape=(self.chunkshape,))
        count = 0
        for i in xrange(0, self.nrows, self.nrep):
            for j in range(self.nrep):
                if self.random:
                    k = random.randrange(self.nrows)
                elif self.values is not None:
                    lenvalues = len(self.values)
                    if i >= lenvalues:
                        i %= lenvalues
                    k = self.values[i]
                else:
                    k = i
                table1.row['var1'] = str(k)
                table2.row['var1'] = str(k)
                table1.row['var2'] = k % 2
                table2.row['var2'] = k % 2
                table1.row['var3'] = k
                table2.row['var3'] = k
                table1.row['var4'] = float(self.nrows - k - 1)
                table2.row['var4'] = float(self.nrows - k - 1)
                table1.row.append()
                table2.row.append()
                count += 1
        table1.flush()
        table2.flush()
        if self.buffersize:
            # Change the buffersize by default
            table1.nrowsinbuf = self.buffersize
        # Index all entries:
        for col in table1.colinstances.itervalues():
            indexrows = col.createIndex(
                kind=self.kind, _blocksizes=self.blocksizes)
        if verbose:
            print "Number of written rows:", table1.nrows
            print "Number of indexed rows:", indexrows

        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")  # for flavor changes
            self.table1 = self.fileh.root.table1
            self.table2 = self.fileh.root.table1

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)

    #----------------------------------------

    def test01a(self):
        """Checking selecting values from an Index (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        # First selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in
                    table1.where('(il<=t1var1)&(t1var1<=sl)')]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Superior & inferior limits:", il, sl
#             print "Selection results (index):", results1
            print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in
                    table1.where('(il<=t1var1)&(t1var1<sl)')]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] < sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in
                    table1.where('(il<t1var1)&(t1var1<=sl)')]
        results2 = [p["var1"] for p in table2
                    if il < p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Forth selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in
                    table1.where('(il<t1var1)&(t1var1<sl)')]
        results2 = [p["var1"] for p in table2
                    if il < p["var1"] < sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test01b(self):
        """Checking selecting values from an Index (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        # First selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where('t1var1 < sl')]
        results2 = [p["var1"] for p in table2
                    if p["var1"] < sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where('t1var1 <= sl')]
        results2 = [p["var1"] for p in table2
                    if p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where('t1var1 > sl')]
        results2 = [p["var1"] for p in table2
                    if p["var1"] > sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where('t1var1 >= sl')]
        results2 = [p["var1"] for p in table2
                    if p["var1"] >= sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test02a(self):
        """Checking selecting values from an Index (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Do some selections and check the results
        t1var2 = table1.cols.var2
        results1 = [p["var2"] for p in table1.where('t1var2 == True')]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == True]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test02b(self):
        """Checking selecting values from an Index (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Do some selections and check the results
        t1var2 = table1.cols.var2
        results1 = [p["var2"] for p in table1.where('t1var2 == False')]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == False]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test03a(self):
        """Checking selecting values from an Index (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3
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

        # Second selection
        results1 = [p["var3"] for p in table1.where('(il<=t1col)&(t1col<sl)')]
        results2 = [p["var3"] for p in table2
                    if il <= p["var3"] < sl]
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

        # Third selection
        results1 = [p["var3"] for p in table1.where('(il<t1col)&(t1col<=sl)')]
        results2 = [p["var3"] for p in table2
                    if il < p["var3"] <= sl]
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

        # Fourth selection
        results1 = [p["var3"] for p in table1.where('(il<t1col)&(t1col<sl)')]
        results2 = [p["var3"] for p in table2
                    if il < p["var3"] < sl]
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

    def test03b(self):
        """Checking selecting values from an Index (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3

        # First selection
        results1 = [p["var3"] for p in table1.where('t1col < sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        results1 = [p["var3"] for p in table1.where('t1col <= sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        results1 = [p["var3"] for p in table1.where('t1col > sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        results1 = [p["var3"] for p in table1.where('t1col >= sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test03c(self):
        """Checking selecting values from an Index (long flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03c..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = long(self.il)
        sl = long(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3

        # First selection
        results1 = [p["var3"] for p in table1.where('t1col < sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        results1 = [p["var3"] for p in table1.where('t1col <= sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        results1 = [p["var3"] for p in table1.where('t1col > sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        results1 = [p["var3"] for p in table1.where('t1col >= sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test03d(self):
        """Checking selecting values from an Index (long and int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03d..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = long(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3

        # First selection
        results1 = [p["var3"] for p in table1.where('t1col < sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        results1 = [p["var3"] for p in table1.where('t1col <= sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        results1 = [p["var3"] for p in table1.where('t1col > sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        results1 = [p["var3"] for p in table1.where('t1col >= sl')]
        results2 = [p["var3"] for p in table2
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test04a(self):
        """Checking selecting values from an Index (float flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = float(self.il)
        sl = float(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var4
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

        # Second selection
        results1 = [p["var4"] for p in table1.where('(il<=t1col)&(t1col<sl)')]
        results2 = [p["var4"] for p in table2
                    if il <= p["var4"] < sl]
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

        # Third selection
        results1 = [p["var4"] for p in table1.where('(il<t1col)&(t1col<=sl)')]
        results2 = [p["var4"] for p in table2
                    if il < p["var4"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        self.assertEqual(results1, results2)

        # Fourth selection
        results1 = [p["var4"] for p in table1.where('(il<t1col)&(t1col<sl)')]
        results2 = [p["var4"] for p in table2
                    if il < p["var4"] < sl]
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

    def test04b(self):
        """Checking selecting values from an Index (float flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = float(self.il)
        sl = float(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var4

        # First selection
        results1 = [p["var4"] for p in table1.where('t1col < sl')]
        results2 = [p["var4"] for p in table2
                    if p["var4"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        results1 = [p["var4"] for p in table1.where('t1col <= sl')]
        results2 = [p["var4"] for p in table2
                    if p["var4"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        results1 = [p["var4"] for p in table1.where('t1col > sl')]
        results2 = [p["var4"] for p in table2
                    if p["var4"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        results1 = [p["var4"] for p in table1.where('t1col >= sl')]
        results2 = [p["var4"] for p in table2
                    if p["var4"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test05a(self):
        """Checking getWhereList & itersequence (string, python flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        # First selection
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] <= sl]
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

        # Second selection
        condition = '(il<=t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] < sl]
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

        # Third selection
        condition = '(il<t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if il < p["var1"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = '(il<t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if il < p["var1"] < sl]
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

    def test05b(self):
        """Checking getWhereList & itersequence (numpy string lims & python flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = numpy.string_(self.il)
        sl = numpy.string_(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1

        # First selection
        condition = 't1col<sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = 't1col<=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't1col>sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var1'] for p in table1.itersequence(rowList1)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test06a(self):
        """Checking getWhereList & itersequence (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Do some selections and check the results
        t1var2 = table1.cols.var2
        condition = 't1var2==True'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1var2.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var2'] for p in table1.itersequence(rowList1)]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == True]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test06b(self):
        """Checking getWhereList & itersequence (numpy bool limits & flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Do some selections and check the results
        t1var2 = table1.cols.var2
        false = numpy.bool_(False)
        condition = 't1var2==false'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1var2.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var2'] for p in table1.itersequence(rowList1)]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == False]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test07a(self):
        """Checking getWhereList & itersequence (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3
        # First selection
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
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
        self.assertEqual(results1.sort(), results2.sort())

        # Second selection
        condition = '(il<=t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if il <= p["var3"] < sl]
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

        # Third selection
        condition = '(il<t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if il < p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = '(il<t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor="python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if il < p["var3"] < sl]
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

    def test07b(self):
        """Checking getWhereList & itersequence (numpy int limits & flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = numpy.int32(self.il)
        sl = numpy.uint16(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var3

        # First selection
        condition = 't1col<sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor="python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = 't1col<=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't1col>sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var3'] for p in table1.itersequence(rowList1)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test08a(self):
        """Checking getWhereList & itersequence (float flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = float(self.il)
        sl = float(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var4
        # First selection
        condition = '(il<=t1col)&(t1col<=sl)'
        #results1 = [p["var4"] for p in table1.where(condition)]
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
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

        # Second selection
        condition = '(il<=t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if il <= p["var4"] < sl]
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

        # Third selection
        condition = '(il<t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if il < p["var4"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = '(il<t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        table1.flavor = "python"
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if il < p["var4"] < sl]
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

    def test08b(self):
        """Checking getWhereList & itersequence (numpy float limits & flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = numpy.float32(self.il)
        sl = numpy.float64(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var4

        # First selection
        condition = 't1col<sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if p["var4"] < sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = 't1col<=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if p["var4"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't1col>sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if p["var4"] > sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        rowList1 = table1.getWhereList(condition)
        results1 = [p['var4'] for p in table1.itersequence(rowList1)]
        results2 = [p["var4"] for p in table2
                    if p["var4"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)


    def test09a(self):
        """Checking non-indexed where() (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        table1._disableIndexingInQueries()

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        # First selection
        condition = 't1col<=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in table1.where(condition,start=2,stop=10)]
        results2 = [p["var1"] for p in table2.iterrows(2, 10)
                    if p["var1"] <= sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = '(il<t1col)&(t1col<sl)'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=2)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,2)
                    if il<p["var1"]<sl]
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = '(il>t1col)&(t1col>sl)'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-5)]
        results2 = [p["var1"] for p in table2.iterrows(2, -5)  # Negative indices
                    if (il > p["var1"] > sl)]
        if verbose:
            print "Limits:", il, sl
            print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # This selection to be commented out
#         condition = 't1col>=sl'
#         self.assert_(not table1.willQueryUseIndexing(condition))
#         results1 = [p['var1'] for p in table1.where(condition,start=2,stop=-1,step=1)]
#         results2 = [p["var1"] for p in table2.iterrows(2, -1, 1)
#                     if p["var1"] >= sl]
#         if verbose:
#             print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
#             print "Length results:", len(results1)
#             print "Should be:", len(results2)
#         self.assertEqual(len(results1), len(results2))
#         self.assertEqual(results1, results2)

        # Fourth selection
        #results1 = [p['var1'] for p in table1.where(condition,start=2,stop=-1,step=3)]
        condition = 't1col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-1, step=3)]
        results2 = [p["var1"] for p in table2.iterrows(2, -1, 3)
                    if p["var1"] >= sl]
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Re-enable the indexing in queries basically to unnail the
        # condition cache and not raising the performance warning
        # about some indexes being dirty
        table1._enableIndexingInQueries()

    def test09b(self):
        """Checking non-indexed where() (float flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        table1._disableIndexingInQueries()

        # Convert the limits to the appropriate type
        il = float(self.il)
        sl = float(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var4

        # First selection
        condition = 't1col<sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var4'] for p in
                    table1.where(condition, start=2, stop=5)]
        results2 = [p["var4"] for p in table2.iterrows(2, 5)
                    if p["var4"] < sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = '(il<t1col)&(t1col<=sl)'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var4'] for p in
                    table1.where(condition, start=2, stop=-1, step=2)]
        results2 = [p["var4"] for p in table2.iterrows(2,-1,2)
                    if il < p["var4"] <= sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var4'] for p in
                    table1.where(condition, start=2, stop=-5)]
        results2 = [p["var4"] for p in table2.iterrows(2, -5)  # Negative indices
                    if il <= p["var4"] <= sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var4'] for p in
                    table1.where(condition, start=0, stop=-1, step=3)]
        results2 = [p["var4"] for p in table2.iterrows(0, -1, 3)
                    if p["var4"] >= sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (in-kernel):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Re-enable the indexing in queries basically to unnail the
        # condition cache and not raising the performance warning
        # about some indexes being dirty
        table1._enableIndexingInQueries()

    def test09c(self):
        "Check non-indexed where() w/ ranges, changing step (string flavor)"

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09c..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        table1._disableIndexingInQueries()

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1

        # First selection
        condition = 't1col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-1, step=3)]
        results2 = [p["var1"] for p in table2.iterrows(2, -1, 3)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition= 't1col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=5, stop=-1, step=10)]
        results2 = [p["var1"] for p in table2.iterrows(5, -1, 10)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't1col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=5, stop=-3, step=11)]
        results2 = [p["var1"] for p in table2.iterrows(5, -3, 11)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-1, step=300)]
        results2 = [p["var1"] for p in table2.iterrows(2, -1, 300)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Re-enable the indexing in queries basically to unnail the
        # condition cache and not raising the performance warning
        # about some indexes being dirty
        table1._enableIndexingInQueries()

    def test09d(self):
        "Checking non-indexed where() w/ ranges, changing step (int flavor)"

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09d..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        table1._disableIndexingInQueries()

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t3col = table1.cols.var3

        # First selection
        condition = 't3col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=-1, step=3)]
        results2 = [p["var3"] for p in table2.iterrows(2, -1, 3)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = 't3col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=5, stop=-1, step=10)]
        results2 = [p["var3"] for p in table2.iterrows(5, -1, 10)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't3col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=5, stop=-3, step=11)]
        results2 = [p["var3"] for p in table2.iterrows(5, -3, 11)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't3col>=sl'
        self.assert_(not table1.willQueryUseIndexing(condition))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=-1, step=300)]
        results2 = [p["var3"] for p in table2.iterrows(2, -1, 300)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Re-enable the indexing in queries basically to unnail the
        # condition cache and not raising the performance warning
        # about some indexes being dirty
        table1._enableIndexingInQueries()

    def test10a(self):
        """Checking indexed where() with ranges (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        # First selection
        condition = 't1col<=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=10)]
        results2 = [p["var1"] for p in table2.iterrows(2, 10)
                    if p["var1"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat second selection (testing caches)
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=2)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,2)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
            print "Selection results (indexed):", results1
            print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = '(il<t1col)&(t1col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-5)]
        results2 = [p["var1"] for p in table2.iterrows(2, -5)  # Negative indices
                    if (il < p["var1"] < sl)]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=1, stop=-1, step=3)]
        results2 = [p["var1"] for p in table2.iterrows(1, -1, 3)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test10b(self):
        """Checking indexed where() with ranges (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t3col = table1.cols.var3
        # First selection
        condition = 't3col<=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=10)]
        results2 = [p["var3"] for p in table2.iterrows(2, 10)
                    if p["var3"] <= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = '(il<=t3col)&(t3col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=30, step=2)]
        results2 = [p["var3"] for p in table2.iterrows(2,30,2)
                    if il<=p["var3"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = '(il<t3col)&(t3col<sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=-5)]
        results2 = [p["var3"] for p in table2.iterrows(2, -5)  # Negative indices
                    if (il < p["var3"] < sl)]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't3col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=1, stop=-1, step=3)]
        results2 = [p["var3"] for p in table2.iterrows(1, -1, 3)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test10c(self):
        """Checking indexed where() with ranges, changing step (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10c..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1

        # First selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-1, step=3)]
        results2 = [p["var1"] for p in table2.iterrows(2, -1, 3)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=5, stop=-1, step=10)]
        results2 = [p["var1"] for p in table2.iterrows(5, -1, 10)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=5, stop=-3, step=11)]
        results2 = [p["var1"] for p in table2.iterrows(5, -3, 11)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't1col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=-1, step=300)]
        results2 = [p["var1"] for p in table2.iterrows(2, -1, 300)
                    if p["var1"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test10d(self):
        """Checking indexed where() with ranges, changing step (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10d..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        # Do some selections and check the results
        t3col = table1.cols.var3

        # First selection
        condition = 't3col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=-1, step=3)]
        results2 = [p["var3"] for p in table2.iterrows(2, -1, 3)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection
        condition = 't3col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=5, stop=-1, step=10)]
        results2 = [p["var3"] for p in table2.iterrows(5, -1, 10)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection
        condition = 't3col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=5, stop=-3, step=11)]
        results2 = [p["var3"] for p in table2.iterrows(5, -3, 11)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Fourth selection
        condition = 't3col>=sl'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t3col.pathname]))
        results1 = [p['var3'] for p in
                    table1.where(condition, start=2, stop=-1, step=300)]
        results2 = [p["var3"] for p in table2.iterrows(2, -1, 300)
                    if p["var3"] >= sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test11a(self):
        """Checking selecting values from an Index via readCoordinates()"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do a selection and check the result
        t1var1 = table1.cols.var1
        condition = '(il<=t1var1)&(t1var1<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1var1.pathname]))
        coords1 = table1.getWhereList(condition)
        table1.flavor = "python"
        results1 = table1.readCoordinates(coords1, field="var1")
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Superior & inferior limits:", il, sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)


    def test12a(self):
        """Checking selecting values after a Table.append() operation."""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test12a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Append more rows in already created indexes
        count = 0
        for i in xrange(0, self.nrows/2, self.nrep):
            for j in range(self.nrep):
                if self.random:
                    k = random.randrange(self.nrows)
                elif self.values is not None:
                    lenvalues = len(self.values)
                    if i >= lenvalues:
                        i %= lenvalues
                    k = self.values[i]
                else:
                    k = i
                table1.row['var1'] = str(k)
                table2.row['var1'] = str(k)
                table1.row['var2'] = k % 2
                table2.row['var2'] = k % 2
                table1.row['var3'] = k
                table2.row['var3'] = k
                table1.row['var4'] = float(self.nrows - k - 1)
                table2.row['var4'] = float(self.nrows - k - 1)
                table1.row.append()
                table2.row.append()
                count += 1
        table1.flush()
        table2.flush()

        t1var1 = table1.cols.var1
        t1var2 = table1.cols.var2
        t1var3 = table1.cols.var3
        t1var4 = table1.cols.var4
        self.assert_(t1var1.index.dirty == False)
        self.assert_(t1var2.index.dirty == False)
        self.assert_(t1var3.index.dirty == False)
        self.assert_(t1var4.index.dirty == False)

        # Do some selections and check the results
        # First selection: string
        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        results1 = [p["var1"] for p in
                    table1.where('(il<=t1var1)&(t1var1<=sl)')]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Superior & inferior limits:", il, sl
#             print "Selection results (index):", results1
            print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Second selection: bool
        results1 = [p["var2"] for p in table1.where('t1var2 == True')]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == True]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Third selection: int
        # Convert the limits to the appropriate type
        il = int(self.il)
        sl = int(self.sl)

        t1var3 = table1.cols.var3
        results1 = [p["var3"] for p in table1.where('(il<=t1var3)&(t1var3<=sl)')]
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

        # Fourth selection: float
        # Convert the limits to the appropriate type
        il = float(self.il)
        sl = float(self.sl)

        # Do some selections and check the results
        results1 = [p["var4"] for p in table1.where('(il<=t1var4)&(t1var4<=sl)')]
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

    def test13a(self):
        """Checking repeated queries (checking caches)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection (testing caches)
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=2)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,2)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test13b(self):
        """Checking repeated queries, varying step (checking caches)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection (testing caches)
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=2, stop=30, step=2)]
        results2 = [p["var1"] for p in table2.iterrows(2,30,2)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test13c(self):
        """Checking repeated queries, varying start, stop, step"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13c..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=1, step=2)]
        results2 = [p["var1"] for p in table2.iterrows(0,1,2)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection (testing caches)
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=5, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,5,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test13d(self):
        """Checking repeated queries, varying start, stop, step (another twist)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13d..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=1, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,1,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection (testing caches)
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=1, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,1,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#            print "Selection results (indexed):", results1
#            print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test13e(self):
        """Checking repeated queries, with varying condition."""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13e..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection with a more complex condition
        t2col = table1.cols.var2
        condition = '(il<=t1col)&(t1col<=sl)&(t2col==True)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname,
                                                             t2col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl and p["var2"]==True]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test13f(self):
        """Checking repeated queries, with varying condition."""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13f..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Remove indexes in var2 column
        table1.cols.var2.removeIndex()
        table2.cols.var2.removeIndex()

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        t2col = table1.cols.var2
        condition = '(il<=t1col)&(t1col<=sl)&(t2col==True)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl and p["var2"]==True]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection with a simpler condition
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat again with the original condition, but with a constant
        constant = True
        condition = '(il<=t1col)&(t1col<=sl)&(t2col==constant)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl and p["var2"]==constant]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    def test13g(self):
        """Checking repeated queries, with different limits."""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13g..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Convert the limits to the appropriate type
        il = str(self.il)
        sl = str(self.sl)

        # Do some selections and check the results
        t1col = table1.cols.var1
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

        # Repeat the selection with different limits
        il, sl = (str(self.il+1), str(self.sl-2))
        t2col = table1.cols.var2
        condition = '(il<=t1col)&(t1col<=sl)'
        self.assert_(
            table1.willQueryUseIndexing(condition) == fzset([t1col.pathname]))
        results1 = [p['var1'] for p in
                    table1.where(condition, start=0, stop=10, step=1)]
        results2 = [p["var1"] for p in table2.iterrows(0,10,1)
                    if il<=p["var1"]<=sl]
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        results1.sort(); results2.sort()
        if verbose:
            print "Limits:", il, sl
#             print "Selection results (indexed):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)


class SV1aTestCase(SelectValuesTestCase):
    blocksizes = small_blocksizes
    chunkshape = 1
    buffersize = 2
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    nrep = ss
    il = 0
    sl = ss

class SV1bTestCase(SV1aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = blocksizes[2]/2**9
    buffersize = chunkshape*5

class SV2aTestCase(SelectValuesTestCase):
    blocksizes = small_blocksizes
    chunkshape = 2
    buffersize = 2
    ss = blocksizes[2]; nrows = ss*2-1
    reopen = 1
    nrep = 1
    il = 0
    sl = 2

class SV2bTestCase(SV2aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = blocksizes[2]/2**7
    buffersize = chunkshape*20

class SV3aTestCase(SelectValuesTestCase):
    blocksizes = small_blocksizes
    chunkshape = 2
    buffersize = 3
    ss = blocksizes[2]; nrows = ss*5-1
    reopen = 1
    nrep = 3
    il = 0
    sl = 3

class SV3bTestCase(SV3aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
#    chunkshape = 4
#    buffersize = 16
    chunkshape = 3
    buffersize = 9

class SV4aTestCase(SelectValuesTestCase):
    blocksizes = small_blocksizes
    buffersize = 10
    ss = blocksizes[2]; nrows = ss*3
    reopen = 0
    nrep = 1
    #il = nrows-cs
    il = 0
    sl = nrows

class SV4bTestCase(SV4aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = 500
    buffersize = 1000

class SV5aTestCase(SelectValuesTestCase):
    blocksizes = small_blocksizes
    ss = blocksizes[2]; nrows = ss*5
    reopen = 0
    nrep = 1
    il = 0
    sl = nrows

class SV5bTestCase(SV5aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)

class SV6aTestCase(SelectValuesTestCase):
    blocksizes = small_blocksizes
    ss = blocksizes[2]; nrows = ss*5+1
    reopen = 0
    cs = blocksizes[3]
    nrep = cs+1
    il = -1
    sl = nrows

class SV6bTestCase(SV6aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)

class SV7aTestCase(SelectValuesTestCase):
    random = 1
    blocksizes = small_blocksizes
    ss = blocksizes[2]; nrows = ss*5+3
    reopen = 0
    cs = blocksizes[3]
    nrep = cs-1
    il = -10
    sl = nrows

class SV7bTestCase(SV7aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)

class SV8aTestCase(SelectValuesTestCase):
    random = 0
    chunkshape = 1
    blocksizes = small_blocksizes
    ss = blocksizes[2]; nrows = ss*5-3
    reopen = 0
    cs = blocksizes[3]
    nrep = cs-1
    il = 10
    sl = nrows-10

class SV8bTestCase(SV8aTestCase):
    random = 0
    blocksizes = calcChunksize(minRowIndex, memlevel=1)

class SV9aTestCase(SelectValuesTestCase):
    random = 1
    blocksizes = small_blocksizes
    ss = blocksizes[2]; nrows = ss*5+11
    reopen = 0
    cs = blocksizes[3]
    nrep = cs-1
    il = 10
    sl = nrows-10

class SV9bTestCase(SV9aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)

class SV10aTestCase(SelectValuesTestCase):
    random = 1
    blocksizes = small_blocksizes
    chunkshape = 1
    buffersize = 1
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    nrep = ss
    il = 0
    sl = ss

class SV10bTestCase(SV10aTestCase):
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = 5
    buffersize = 6

class SV11aTestCase(SelectValuesTestCase):
    # This checks a special case that failed. It was discovered in a
    # random test above (SV10a). It is explicitely put here as a way
    # to always check that specific case.
    values = [1, 7, 6, 7, 0, 7, 4, 4, 9, 5]
    blocksizes = small_blocksizes
    chunkshape = 1
    buffersize = 1
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    nrep = ss
    il = 0
    sl = ss

class SV11bTestCase(SelectValuesTestCase):
    # This checks a special case that failed. It was discovered in a
    # random test above (SV10a). It is explicitely put here as a way
    # to always check that specific case.
    values = [1, 7, 6, 7, 0, 7, 4, 4, 9, 5]
    chunkshape = 2
    buffersize = 2
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    nrep = ss
    il = 0
    sl = ss

class SV12aTestCase(SelectValuesTestCase):
    # This checks a special case that failed. It was discovered in a
    # random test above (SV10b). It is explicitely put here as a way
    # to always check that specific case.
    #values = [0, 7, 0, 6, 5, 1, 6, 7, 0, 0]
    values = [4, 4, 1, 5, 2, 0, 1, 4, 3, 9]
    blocksizes = small_blocksizes
    chunkshape = 1
    buffersize = 1
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    nrep = ss
    il = 0
    sl = ss

class SV12bTestCase(SelectValuesTestCase):
    # This checks a special case that failed. It was discovered in a
    # random test above (SV10b). It is explicitely put here as a way
    # to always check that specific case.
    #values = [0, 7, 0, 6, 5, 1, 6, 7, 0, 0]
    values = [4, 4, 1, 5, 2, 0, 1, 4, 3, 9]
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = 2
    buffersize = 2
    ss = blocksizes[2]; nrows = ss
    reopen = 1
    nrep = ss
    il = 0
    sl = ss

class SV13aTestCase(SelectValuesTestCase):
    values = [0, 7, 0, 6, 5, 1, 6, 7, 0, 0]
    blocksizes = small_blocksizes
    chunkshape = 3
    buffersize = 5
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    nrep = ss
    il = 0
    sl = ss

class SV13bTestCase(SelectValuesTestCase):
    values = [0, 7, 0, 6, 5, 1, 6, 7, 0, 0]
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = 5
    buffersize = 10
    ss = blocksizes[2]; nrows = ss
    reopen = 1
    nrep = ss
    il = 0
    sl = ss

class SV14aTestCase(SelectValuesTestCase):
    values = [1, 7, 6, 7, 0, 7, 4, 4, 9, 5]
    blocksizes = small_blocksizes
    chunkshape = 2
    buffersize = 5
    ss = blocksizes[2]; nrows = ss
    reopen = 0
    cs = blocksizes[3]
    nrep = cs
    il = -5
    sl = 500

class SV14bTestCase(SelectValuesTestCase):
    values = [1, 7, 6, 7, 0, 7, 4, 4, 9, 5]
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    chunkshape = 9
    buffersize = 10
    ss = blocksizes[2]; nrows = ss
    reopen = 1
    nrep = 9
    il = 0
    cs = blocksizes[3]
    sl = ss-cs+1

class SV15aTestCase(SelectValuesTestCase):
    # Test that checks for case where there are not valid values in
    # the indexed part, but they exist in the non-indexed region.
    # At least, test01b takes account of that
    random = 1
    # Both values of seed below triggers a fail in indexing code
    #seed = 1885
    seed = 183
    blocksizes = small_blocksizes
    ss = blocksizes[2]; nrows = ss*5+1
    reopen = 0
    cs = blocksizes[3]
    nrep = cs-1
    il = -10
    sl = nrows

class SV15bTestCase(SelectValuesTestCase):
    # Test that checks for case where there are not valid values in
    # the indexed part, but they exist in the non-indexed region.
    # At least, test01b takes account of that
    random = 1
    # Both values of seed below triggers a fail in indexing code
    seed = 1885
    #seed = 183
    blocksizes = calcChunksize(minRowIndex, memlevel=1)
    ss = blocksizes[2]; nrows = ss*5+1
    reopen = 1
    cs = blocksizes[3]
    nrep = cs-1
    il = -10
    sl = nrows


class LastRowReuseBuffers(common.PyTablesTestCase):
    # Test that checks for possible reuse of buffers coming
    # from last row in the sorted part of indexes
    nelem = 1221
    numpy.random.seed(1); random.seed(1)

    class Record(IsDescription):
        id1 = Int16Col()

    def test00_lrucache(self):
        filename = tempfile.mktemp(".h5")
        fp = openFile(filename, 'w', NODE_CACHE_SLOTS=64)
        ta = fp.createTable('/', 'table', self.Record, filters=Filters(1))
        id1 = numpy.random.randint(0, 2**15, self.nelem)
        ta.append([id1])

        ta.cols.id1.createIndex()

        for i in xrange(self.nelem):
            nrow = random.randint(0, self.nelem-1)
            value = id1[nrow]
            idx = ta.getWhereList('id1 == %s' % value)
            self.assertTrue(len(idx) > 0 ,
                            "idx--> %s %s %s %s" % (idx, i, nrow, value))
            self.assertTrue(nrow in idx,
                            "nrow not found: %s != %s, %s" % (idx, nrow, value))

        fp.close()
        os.remove(filename)


    def test01_nocache(self):
        filename = tempfile.mktemp(".h5")
        fp = openFile(filename, 'w', NODE_CACHE_SLOTS=0)
        ta = fp.createTable('/', 'table', self.Record, filters=Filters(1))
        id1 = numpy.random.randint(0, 2**15, self.nelem)
        ta.append([id1])

        ta.cols.id1.createIndex()

        for i in xrange(self.nelem):
            nrow = random.randint(0, self.nelem-1)
            value = id1[nrow]
            idx = ta.getWhereList('id1 == %s' % value)
            self.assertTrue(len(idx) > 0 ,
                            "idx--> %s %s %s %s" % (idx, i, nrow, value))
            self.assertTrue(nrow in idx,
                            "nrow not found: %s != %s, %s" % (idx, nrow, value))

        fp.close()
        os.remove(filename)


    def test02_dictcache(self):
        filename = tempfile.mktemp(".h5")
        fp = openFile(filename, 'w', NODE_CACHE_SLOTS=-64)
        ta = fp.createTable('/', 'table', self.Record, filters=Filters(1))
        id1 = numpy.random.randint(0, 2**15, self.nelem)
        ta.append([id1])

        ta.cols.id1.createIndex()

        for i in xrange(self.nelem):
            nrow = random.randint(0, self.nelem-1)
            value = id1[nrow]
            idx = ta.getWhereList('id1 == %s' % value)
            self.assertTrue(len(idx) > 0 ,
                            "idx--> %s %s %s %s" % (idx, i, nrow, value))
            self.assertTrue(nrow in idx,
                            "nrow not found: %s != %s, %s" % (idx, nrow, value))

        fp.close()
        os.remove(filename)


normal_tests = (
    "SV1aTestCase", "SV2aTestCase", "SV3aTestCase",
    )

heavy_tests = (
    # The next are too hard to be in the 'normal' suite
    "SV1bTestCase", "SV2bTestCase", "SV3bTestCase",
    "SV4aTestCase", "SV5aTestCase", "SV6aTestCase",
    "SV7aTestCase", "SV8aTestCase", "SV9aTestCase",
    "SV10aTestCase", "SV11aTestCase", "SV12aTestCase",
    "SV13aTestCase", "SV14aTestCase", "SV15aTestCase",
    # This are properly heavy
    "SV4bTestCase", "SV5bTestCase", "SV6bTestCase",
    "SV7bTestCase", "SV8bTestCase", "SV9bTestCase",
    "SV10bTestCase", "SV11bTestCase", "SV12bTestCase",
    "SV13bTestCase", "SV14bTestCase", "SV15bTestCase",
    )


# Base classes for the different type indexes.
class UltraLightITableMixin:
    kind = "ultralight"
class LightITableMixin:
    kind = "light"
class MediumITableMixin:
    kind = "medium"
class FullITableMixin:
    kind = "full"

# Parameters for indexed queries.
ckinds = ['UltraLight', 'Light', 'Medium', 'Full']
testlevels = ['Normal', 'Heavy']

# Indexed queries: ``[ULMF]I[NH]SVXYTestCase``, where:
#
# 1. U is for 'UltraLight', L for 'Light', M for 'Medium', F for 'Full' indexes
# 2. N is for 'Normal', H for 'Heavy' tests
def iclassdata():
    for ckind in ckinds:
        for ctest in normal_tests + heavy_tests:
            heavy = ctest in heavy_tests
            classname = '%sI%s%s' % (ckind[0], testlevels[heavy][0], ctest)
            # Uncomment the next one and comment the past one if one
            # don't want to include the methods (testing purposes only)
            ###cbasenames = ( '%sITableMixin' % ckind, "object")
            cbasenames = ( '%sITableMixin' % ckind, ctest)
            classdict = dict(heavy=heavy)
            yield (classname, cbasenames, classdict)


# Create test classes.
for (cname, cbasenames, cdict) in iclassdata():
    cbases = tuple(eval(cbase) for cbase in cbasenames)
    class_ = new.classobj(cname, cbases, cdict)
    exec '%s = class_' % cname


# -----------------------------

def suite():
    theSuite = unittest.TestSuite()

    niter = 1

    for n in range(niter):
        for cdata in iclassdata():
            class_ = eval(cdata[0])
            if not class_.heavy:
                suite_ = unittest.makeSuite(class_)
                theSuite.addTest(suite_)
            elif heavy:
                suite_ = unittest.makeSuite(class_)
                theSuite.addTest(suite_)
        theSuite.addTest(unittest.makeSuite(LastRowReuseBuffers))

    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
