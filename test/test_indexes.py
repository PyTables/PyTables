# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import unittest
import os
import tempfile
import warnings

from tables import *
from test_all import verbose, allequal

# The minimum number of rows that can be indexed
# Remember to change that if the number is changed in IndexArray._calcChunksize
minRowIndex = 1000

class Small(IsDescription):
    var1 = StringCol(length=4, dflt="")
    var2 = BoolCol(0)
    var3 = IntCol(0)
    var4 = FloatCol(0)

class BasicTestCase(unittest.TestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    nrows = minRowIndex

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        self.populateFile()
        # Close the file
        self.fileh.close()

    def populateFile(self):
        group = self.rootgroup
        # Create an table
        title = "This is the IndexArray title"
        rowswritten = 0
        filters = Filters(complevel = self.compress,
                          complib = self.complib,
                          shuffle = self.shuffle,
                          fletcher32 = self.fletcher32)
        table = self.fileh.createTable(group, 'table', Small, title,
                                       filters, self.nrows)
        for i in range(self.nrows):
            table.row['var1'] = str(i)
            # table.row['var2'] = i > 2
            table.row['var2'] = i % 2
            table.row['var3'] = i
            table.row['var4'] = float(self.nrows - i - 1)
            table.row.append()
        table.flush()
        # Index all entries:
        indexrows = table.cols.var1.createIndex()
        indexrows = table.cols.var2.createIndex()
        indexrows = table.cols.var3.createIndex()
        indexrows = table.cols.var4.createIndex()
        if verbose:
            print "Number of written rows:", self.nrows
            print "Number of indexed rows:", indexrows

        return

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    #----------------------------------------

    def test01_readIndex(self):
        """Checking reading an Index (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.nelemslice
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var1"] for p in table(where=table.cols.var1 == "1")]
        assert len(results) == 1

    def test02_readIndex(self):
        """Checking reading an Index (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_readIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var2.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.nelemslice
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var2"] for p in table(where=table.cols.var2 == 1)]
        if verbose:
            print "Selected values:", results
        assert len(results) == self.nrows // 2

    def test03_readIndex(self):
        """Checking reading an Index (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_readIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var3.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.nelemslice
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var3"] for p in table(where=1< table.cols.var3 < 10)]
        if verbose:
            print "Selected values:", results
        assert len(results) == 8

    def test04_readIndex(self):
        """Checking reading an Index (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_readIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.nelemslice
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var4"] for p in table(where=table.cols.var4 < 10)]
        if verbose:
            print "Selected values:", results
        assert len(results) == 10


class BasicReadTestCase(BasicTestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    nrows = minRowIndex

class ZlibReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    nrows = minRowIndex

class LZOReadTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    shuffle = 0
    fletcher32 = 0
    nrows = minRowIndex

class UCLReadTestCase(BasicTestCase):
    compress = 1
    complib = "ucl"
    shuffle = 0
    fletcher32 = 0
    nrows = minRowIndex

class ShuffleReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 0
    nrows = minRowIndex

class Fletcher32ReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 0
    fletcher32 = 1
    nrows = minRowIndex

class ShuffleFletcher32ReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 1
    nrows = minRowIndex

class OneHalfTestCase(BasicTestCase):
    nrows = minRowIndex+500

class UpperBoundTestCase(BasicTestCase):
    nrows = minRowIndex+1

class LowerBoundTestCase(BasicTestCase):
    nrows = minRowIndex*2-1

class WarningTestCase(unittest.TestCase):
    nrows = 100 # Small enough to raise the warning
    
    def test01(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        group = self.rootgroup
        # Create an table
        title = "This is the IndexArray title"
        rowswritten = 0
        table = self.fileh.createTable(group, 'table', Small, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # try to index one entry
        warnings.filterwarnings("error", category=UserWarning)
        try:
            indexrows = table.cols.var1.createIndex()
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched!"
                print value
        else:
            self.fail("expected an UserWarning")
        # Reset the warning
        warnings.filterwarnings("default", category=UserWarning)

        self.fileh.close()
        os.remove(self.file)
        

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    #theSuite.addTest(unittest.makeSuite(BasicReadTestCase))
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicReadTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibReadTestCase))
        theSuite.addTest(unittest.makeSuite(LZOReadTestCase))
        theSuite.addTest(unittest.makeSuite(UCLReadTestCase))
        theSuite.addTest(unittest.makeSuite(ShuffleReadTestCase))
        theSuite.addTest(unittest.makeSuite(Fletcher32ReadTestCase))
        theSuite.addTest(unittest.makeSuite(ShuffleFletcher32ReadTestCase))
        theSuite.addTest(unittest.makeSuite(OneHalfTestCase))
        theSuite.addTest(unittest.makeSuite(UpperBoundTestCase))
        theSuite.addTest(unittest.makeSuite(LowerBoundTestCase))
        theSuite.addTest(unittest.makeSuite(WarningTestCase))
    
    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
