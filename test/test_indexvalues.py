import unittest
import os
import tempfile
import warnings

from tables import *
from tables.Index import Index
from tables.IndexArray import calcChunksize
from test_all import verbose, allequal, niterHeavy

# The minimum number of rows that can be indexed
# Remember to change that if the number is changed in
# IndexArray._calcChunksize
minRowIndex = 10000

class Small(IsDescription):
    var1 = StringCol(length=4, dflt="")
    var2 = BoolCol(0)
    var3 = IntCol(0)
    var4 = FloatCol(0)

class SelectValuesTestCase(unittest.TestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 0
    buffersize = 0

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()

    def populateFile(self):
        group = self.rootgroup
        # Create an table
        title = "This is the IndexArray title"
        rowswritten = 0
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        table1 = self.fileh.createTable(group, 'table1', Small, title,
                                        filters, self.nrows)
        table2 = self.fileh.createTable(group, 'table2', Small, title,
                                        filters, self.nrows)
        # Select small value for table buffers
        if self.buffersize:
            # Change the buffersize by default
            table1._v_maxTuples = self.buffersize
        #table2._v_maxTuples = self.buffersize  # This is not necessary
        for i in range(0, self.nrows, self.nrep):
            for j in range(self.nrep):
                #print i,
                table1.row['var1'] = str(i)
                table2.row['var1'] = str(i)
                table1.row['var2'] = i % 2
                table2.row['var2'] = i % 2
                table1.row['var3'] = i
                table2.row['var3'] = i
                table1.row['var4'] = float(self.nrows - i - 1)
                table2.row['var4'] = float(self.nrows - i - 1)
                table1.row.append()
                table2.row.append()
        table1.flush()
        table2.flush()
        # Index all entries:
        indexrows = table1.cols.var1.createIndex(testmode=1)
        indexrows = table1.cols.var2.createIndex(testmode=1)
        indexrows = table1.cols.var3.createIndex(testmode=1)
        indexrows = table1.cols.var4.createIndex(testmode=1)
        if verbose:
            print "Number of written rows:", self.nrows
            print "Number of indexed rows:", indexrows
        
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "r")
            self.table1 = self.fileh.root.table1
            self.table2 = self.fileh.root.table1

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
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
        results1 = [p["var1"] for p in table1.where(il <= t1var1 <= sl)]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Superior & inferior limits:", il, sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Second selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where(il <= t1var1 < sl)]
        results2 = [p["var1"] for p in table2
                    if il <= p["var1"] < sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Third selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where(il < t1var1 <= sl)]
        results2 = [p["var1"] for p in table2
                    if il < p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Forth selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where(il < t1var1 < sl)]
        results2 = [p["var1"] for p in table2
                    if il < p["var1"] < sl]
        results1.sort(); results2.sort()
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

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
        results1 = [p["var1"] for p in table1.where(t1var1 < sl)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] < sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Second selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where(t1var1 <= sl)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] <= sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Third selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where(t1var1 > sl)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] > sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Fourth selection
        t1var1 = table1.cols.var1
        results1 = [p["var1"] for p in table1.where(t1var1 >= sl)]
        results2 = [p["var1"] for p in table2
                    if p["var1"] >= sl]
        results1.sort(); results2.sort()
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

    def test02a(self):
        """Checking selecting values from an Index (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02a..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2
        
        # Do some selections and check the results
        t1var2 = table1.cols.var2
        results1 = [p["var2"] for p in table1.where(t1var2 == 1)]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == 1]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

    def test02b(self):
        """Checking selecting values from an Index (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b..." % self.__class__.__name__

        table1 = self.fileh.root.table1
        table2 = self.fileh.root.table2

        # Do some selections and check the results
        t1var2 = table1.cols.var2
        results1 = [p["var2"] for p in table1.where(t1var2 == 0)]
        results2 = [p["var2"] for p in table2
                    if p["var2"] == 0]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

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
        results1 = [p["var3"] for p in table1.where(il <= t1col <= sl)]
        results2 = [p["var3"] for p in table2
                    if il <= p["var3"] <= sl]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Second selection
        results1 = [p["var3"] for p in table1.where(il <= t1col < sl)]
        results2 = [p["var3"] for p in table2
                    if il <= p["var3"] < sl]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Third selection
        results1 = [p["var3"] for p in table1.where(il < t1col <= sl)]
        results2 = [p["var3"] for p in table2
                    if il < p["var3"] <= sl]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Fourth selection
        results1 = [p["var3"] for p in table1.where(il < t1col < sl)]
        results2 = [p["var3"] for p in table2
                    if il < p["var3"] < sl]
        if verbose:
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

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
        results1 = [p["var3"] for p in table1.where(t1col < sl)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] < sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Second selection
        results1 = [p["var3"] for p in table1.where(t1col <= sl)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] <= sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Third selection
        results1 = [p["var3"] for p in table1.where(t1col > sl)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] > sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

        # Fourth selection
        results1 = [p["var3"] for p in table1.where(t1col >= sl)]
        results2 = [p["var3"] for p in table2
                    if p["var3"] >= sl]
        if verbose:
            print "Limit:", sl
#             print "Selection results (index):", results1
#             print "Should look like:", results2
            print "Length results:", len(results1)
            print "Should be:", len(results2)
        assert len(results1) == len(results2)
        assert results1 == results2

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
        results1 = [p["var4"] for p in table1.where(il <= t1col <= sl)]
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
        assert len(results1) == len(results2)
        assert results1.sort() == results2.sort()

        # Second selection
        results1 = [p["var4"] for p in table1.where(il <= t1col < sl)]
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
        assert len(results1) == len(results2)
        assert results1 == results2

        # Third selection
        results1 = [p["var4"] for p in table1.where(il < t1col <= sl)]
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
        assert len(results1) == len(results2)
        # sort lists (indexing does not guarantee that rows are returned in
        # order)
        assert results1 == results2
        
        # Fourth selection
        results1 = [p["var4"] for p in table1.where(il < t1col < sl)]
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
        assert len(results1) == len(results2)
        assert results1 == results2

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
        results1 = [p["var4"] for p in table1.where(t1col < sl)]
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
        assert len(results1) == len(results2)
        assert results1 == results2

        # Second selection
        results1 = [p["var4"] for p in table1.where(t1col <= sl)]
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
        assert len(results1) == len(results2)
        assert results1 == results2

        # Third selection
        results1 = [p["var4"] for p in table1.where(t1col > sl)]
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
        assert len(results1) == len(results2)
        assert results1 == results2

        # Fourth selection
        results1 = [p["var4"] for p in table1.where(t1col >= sl)]
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
        assert len(results1) == len(results2)
        assert results1 == results2


class SV1aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    buffersize = 1
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns
    reopen = 0
    nrep = ns
    il = 0
    sl = ns

class SV1bTestCase(SV1aTestCase):
    minRowIndex = 1000
    buffersize = 10

class SV2aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    buffersize = 2
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns
    reopen = 1
    nrep = 1
    il = 0
    sl = 2

class SV2bTestCase(SV2aTestCase):
    minRowIndex = 1000
    buffersize = 20

class SV3aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    buffersize = 3
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns*2-1
    reopen = 1
    nrep = 3
    il = 0
    sl = 3

class SV3bTestCase(SV3aTestCase):
    minRowIndex = 1000
    buffersize = 33

class SV4aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    buffersize = 10
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns*3
    reopen = 0
    nrep = 1
    #il = nrows-cs
    il = 0
    sl = nrows

class SV4bTestCase(SV4aTestCase):
    minRowIndex = 1000
    buffersize = 1000

class SV5aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns*5
    reopen = 0
    #print "minRowIndex, ns, cs-->", minRowIndex, ns, cs
    #nrep = (ns+1)
    nrep = 1
    il = 0
    sl = nrows

class SV5bTestCase(SV5aTestCase):
    minRowIndex = 1000

class SV6aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns*5-1
    reopen = 0
    nrep = cs+1
    il = -1
    sl = nrows

class SV6bTestCase(SV6aTestCase):
    minRowIndex = 1000

class SV7aTestCase(SelectValuesTestCase):
    minRowIndex = 1000
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns*5+1
    reopen = 0
    nrep = cs-1
    il = -10
    sl = nrows

class SV7bTestCase(SV7aTestCase):
    minRowIndex = 1000

class SV8aTestCase(SelectValuesTestCase):
    minRowIndex = 10
    ns, cs = calcChunksize(minRowIndex, testmode=1)
    nrows = ns*5+1
    reopen = 0
    nrep = cs-1
    il = 10
    sl = nrows-10

class SV8bTestCase(SV7aTestCase):
    minRowIndex = 1000

# -----------------------------

def suite():
    import sys
    theSuite = unittest.TestSuite()
    # Default is to run light benchmarks
    niterLight = 1
    niterHeavy = 1  # Uncomment this only if you are on a big machine!

    #theSuite.addTest(unittest.makeSuite(SV4TestCase))
    for n in range(niterLight):
        theSuite.addTest(unittest.makeSuite(SV1aTestCase))
        theSuite.addTest(unittest.makeSuite(SV2aTestCase))
        theSuite.addTest(unittest.makeSuite(SV3aTestCase))
        theSuite.addTest(unittest.makeSuite(SV4aTestCase))
    
    for n in range(niterHeavy):
        theSuite.addTest(unittest.makeSuite(SV1bTestCase))
        theSuite.addTest(unittest.makeSuite(SV2bTestCase))
        theSuite.addTest(unittest.makeSuite(SV3bTestCase))
        theSuite.addTest(unittest.makeSuite(SV4bTestCase))
        theSuite.addTest(unittest.makeSuite(SV5bTestCase))
        theSuite.addTest(unittest.makeSuite(SV6bTestCase))
        theSuite.addTest(unittest.makeSuite(SV7bTestCase))
        theSuite.addTest(unittest.makeSuite(SV8bTestCase))
        # The next are too hard to be above
        theSuite.addTest(unittest.makeSuite(SV5aTestCase))
        theSuite.addTest(unittest.makeSuite(SV6aTestCase))
        theSuite.addTest(unittest.makeSuite(SV7aTestCase))
        theSuite.addTest(unittest.makeSuite(SV8aTestCase))
    
    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
