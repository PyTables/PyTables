# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

import unittest
import os
import tempfile
import warnings

from tables import *
from tables.Index import Index
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
        
class Small2(IsDescription):
    _v_automatic_index__ = 0  # Not the default
    #_v_reindex__ = 1  # The default
    var1 = StringCol(length=4, dflt="", indexed=1)
    var2 = BoolCol(0, indexed=1)
    var3 = IntCol(0, indexed=1)
    var4 = FloatCol(0, indexed=0)

class Small3(IsDescription):
    #_v_automatic_index__ = 1  # The default
    _v_reindex__ = 0  # Not the default
    var1 = StringCol(length=4, dflt="", indexed=1)
    var2 = BoolCol(0, indexed=1)
    var3 = IntCol(0, indexed=1)
    var4 = FloatCol(0, indexed=0)

class AutomaticIndexingTestCase(unittest.TestCase):
    nrows = 10
    reopen = 1
    klass = Small2
    
    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        # Create an table
        title = "This is the IndexArray title"
        rowswritten = 0
        root = self.fileh.root
        self.table = self.fileh.createTable(root, 'table', self.klass, title,
                                            None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            self.table.row.append()
        self.table.flush()
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            self.table = self.fileh.root.table

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        
    def test01_checkattrs(self):
        "Checking indexing attributes (part1)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_checkattrs..." % self.__class__.__name__

        table = self.table
        if self.klass is Small:
            assert table.indexed == 0
        else:
            assert table.indexed == 1
        # Check that the var1, var2 and var3 (and only these) has been indexed
        if self.klass is Small:
            assert table.colindexed["var1"] == 0
            assert table.cols.var1.indexed == 0
            assert table.colindexed["var2"] == 0
            assert table.cols.var2.indexed == 0
            assert table.colindexed["var3"] == 0
            assert table.cols.var3.indexed == 0
            assert table.colindexed["var4"] == 0
            assert table.cols.var4.indexed == 0
        else:
            assert table.colindexed["var1"] == 1
            assert table.cols.var1.indexed == 1
            assert table.colindexed["var2"] == 1
            assert table.cols.var2.indexed == 1
            assert table.colindexed["var3"] == 1
            assert table.cols.var3.indexed == 1
            assert table.colindexed["var4"] == 0
            assert table.cols.var4.indexed == 0
                    
    def test02_checkattrs(self):
        "Checking indexing attributes (part2)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_checkattrs..." % self.__class__.__name__

        table = self.table
        # Check the policy parameters
        if verbose:
            print "automatic_index:", table.automatic_index
            print "reindex:", table.reindex
        # Check non-default values for index saving policy
        if self.klass is Small:
            assert not hasattr(table, "automatic_index")
            assert not hasattr(table, "reindex")
        elif self.klass is Small2:
            assert table.automatic_index == 0
            assert table.reindex == 1
        elif self.klass is Small3:
            assert table.automatic_index == 1
            assert table.reindex == 0
            
        # Check Index() objects exists and are properly placed
        if self.klass is Small:
            assert table.cols.var1.index == None
            assert table.cols.var2.index == None
            assert table.cols.var3.index == None
            assert table.cols.var4.index == None
        else:
            assert isinstance(table.cols.var1.index, Index)
            assert isinstance(table.cols.var2.index, Index)
            assert isinstance(table.cols.var3.index, Index)
            assert table.cols.var4.index == None
        
    def test03_checkcounters(self):
        "Checking indexing counters"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_checkcounters..." % self.__class__.__name__
        table = self.table
        # Check the counters for indexes
        if verbose:
            print "indexedrows:", table._indexedrows
            print "unsavedindexedrows:", table._unsavedindexedrows
            index = table.cols.var1.index
            indexedrows = index.nrows * index.nelemslice
            print "computed indexed rows:", indexedrows
        if self.klass is not Small:
            index = table.cols.var1.index
            indexedrows = index.nrows * index.nelemslice
            assert table._indexedrows == indexedrows
            assert table._unsavedindexedrows == self.nrows - indexedrows

    def test04_checknoauto(self):
        "Checking indexing counters (non-automatic mode)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_checknoauto..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.addRowsToIndex()
        # Check the counters for indexes
        if verbose:
            print "indexedrows:", table._indexedrows
            print "unsavedindexedrows:", table._unsavedindexedrows
            index = table.cols.var1.index
            indexedrows = index.nrows * index.nelemslice
            print "computed indexed rows:", indexedrows

        # No unindexated rows should remain
        index = table.cols.var1.index
        if self.klass is Small:
            assert index is None
        else:
            indexedrows = index.nrows * index.nelemslice
            assert table._indexedrows == indexedrows
            assert table._unsavedindexedrows == self.nrows - indexedrows

    def test05_checknoreindex(self):
        "Checking indexing counters (non-reindex mode)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_checknoreindex..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.addRowsToIndex()
        # No unidexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsavedindexedrows
        # Now, remove some rows:
        table.removeRows(3,5)
        # Check the counters for indexes
        if verbose:
            print "indexedrows:", table._indexedrows
            print "unsavedindexedrows:", table._unsavedindexedrows
            index = table.cols.var1.index
            indexedrows = index.nrows * index.nelemslice
            print "computed indexed rows:", indexedrows

        # Check the counters
        assert table.nrows == self.nrows - 2
        if self.klass is Small3:
            # The unsaved indexed rows counter should be unchanged
            assert table._indexedrows == indexedrows
            assert table._unsavedindexedrows == unsavedindexedrows
        elif self.klass is Small2:
            index = table.cols.var1.index
            indexedrows = index.nrows * index.nelemslice
            assert table._indexedrows == indexedrows
            assert table._unsavedindexedrows == self.nrows - indexedrows - 2

class AI1TestCase(AutomaticIndexingTestCase):
    nrows = 10
    reopen = 0
    klass = Small2
    
class AI2TestCase(AutomaticIndexingTestCase):
    nrows = 10
    reopen = 1
    klass = Small2
    
class AI3TestCase(AutomaticIndexingTestCase):
    nrows = 10
    reopen = 1
    klass = Small3
    
class AI4TestCase(AutomaticIndexingTestCase):
    nrows = 10
    reopen = 0
    klass = Small3
    
class AI5TestCase(AutomaticIndexingTestCase):
    nrows = 1000
    reopen = 0
    klass = Small2
    
class AI6TestCase(AutomaticIndexingTestCase):
    nrows = 1000
    reopen = 1
    klass = Small2

class AI7TestCase(AutomaticIndexingTestCase):
    nrows = 1000
    reopen = 0
    klass = Small3
    
class AI8TestCase(AutomaticIndexingTestCase):
    nrows = 1000
    reopen = 1
    klass = Small3
    
class AI9TestCase(AutomaticIndexingTestCase):
    nrows = 1000
    reopen = 0
    klass = Small
    
class AI10TestCase(AutomaticIndexingTestCase):
    nrows = 10
    reopen = 1
    klass = Small
    

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
        theSuite.addTest(unittest.makeSuite(AI1TestCase))
        theSuite.addTest(unittest.makeSuite(AI2TestCase))
        theSuite.addTest(unittest.makeSuite(AI3TestCase))
        theSuite.addTest(unittest.makeSuite(AI4TestCase))
        theSuite.addTest(unittest.makeSuite(AI5TestCase))
        theSuite.addTest(unittest.makeSuite(AI6TestCase))
        theSuite.addTest(unittest.makeSuite(AI7TestCase))
        theSuite.addTest(unittest.makeSuite(AI8TestCase))
        theSuite.addTest(unittest.makeSuite(AI9TestCase))
        theSuite.addTest(unittest.makeSuite(AI10TestCase))
    
    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
