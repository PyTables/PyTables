import unittest
import os
import tempfile
import warnings
import sys

from tables import *
from tables.Index import Index
from tables.indexes import calcChunksize, minRowIndex
from common import verbose, allequal, heavy, cleanup
# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

import numarray

class Small(IsDescription):
    var1 = StringCol(length=4, dflt="", pos=1)
    var2 = BoolCol(0, pos=2)
    var3 = IntCol(0, pos=3)
    var4 = FloatCol(0, pos=4)

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
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        self.filters = Filters(complevel = self.compress,
                               complib = self.complib,
                               shuffle = self.shuffle,
                               fletcher32 = self.fletcher32)
        table = self.fileh.createTable(group, 'table', Small, title,
                                       self.filters, self.nrows)
        for i in range(self.nrows):
            table.row['var1'] = str(i)
            # table.row['var2'] = i > 2
            table.row['var2'] = i % 2
            table.row['var3'] = i
            table.row['var4'] = float(self.nrows - i - 1)
            table.row.append()
        table.flush()
        # Index all entries:
        indexrows = table.cols.var1.createIndex(testmode=1)
        indexrows = table.cols.var2.createIndex(testmode=1)
        indexrows = table.cols.var3.createIndex(testmode=1)
        indexrows = table.cols.var4.createIndex(testmode=1)
        if verbose:
            print "Number of written rows:", self.nrows
            print "Number of indexed rows:", indexrows

        return

    def tearDown(self):
        self.fileh.close()
        #print "File %s not removed!" % self.file
        os.remove(self.file)
        cleanup(self)

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
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var1"] for p in table.where(table.cols.var1 == "1")]
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
            print "Rows in table:", table.nrows
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var2"] for p in table.where(table.cols.var2 == 1)]
        if verbose:
            print "Selected values:", results
        assert len(results) == self.nrows // 2

    def test03_readIndex(self):
        """Checking reading an Index (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_readIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var3.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var3"] for p in table.where(1< table.cols.var3 < 10)]
        if verbose:
            print "Selected values:", results
        assert len(results) == 8

    def test04_readIndex(self):
        """Checking reading an Index (float flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_readIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var4"] for p in table.where(table.cols.var4 < 10)]
        if verbose:
            print "Selected values:", results
        assert len(results) == 10

    def test05_getWhereList(self):
        """Checking reading an Index with getWhereList (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_getWhereList..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        rowList1 = table.getWhereList(table.cols.var1 < "10", "python")
        rowList2 = [p.nrow for p in table if p['var1'] < "10"]
        if verbose:
            print "Selected values:", rowList1
            print "Should look like:", rowList2
        assert len(rowList1) == len(rowList2)
        assert rowList1 == rowList2

    def test06_getWhereList(self):
        """Checking reading an Index with getWhereList (bool flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_getWhereList..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var2.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Rows in tables:", table.nrows
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        rowList1 = table.getWhereList(table.cols.var2 == 0, "numarray", True)
        rowList2 = [p.nrow for p in table if p['var2'] == 0]
        # Convert to a numarray object
        rowList2 = numarray.array(rowList2, numarray.Int64)
        if verbose:
            print "Selected values:", rowList1
            print "Should look like:", rowList2
        assert len(rowList1) == len(rowList2)
        assert allequal(rowList1, rowList2)

    def test07_getWhereList(self):
        """Checking reading an Index with getWhereList (int flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_getWhereList..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        rowList1 = table.getWhereList(table.cols.var3 < 15, "python", True)
        rowList2 = [p.nrow for p in table if p["var3"] < 15]
        if verbose:
            print "Selected values:", rowList1
            print "Should look like:", rowList2
        assert len(rowList1) == len(rowList2)
        assert rowList1 == rowList2

    def test08_getWhereList(self):
        """Checking reading an Index with getWhereList (float flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_getWhereList..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "r")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table._v_maxTuples
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        rowList1 = table.getWhereList(table.cols.var4 < 10, "python", True)
        rowList2 = [p.nrow for p in table if p['var4'] < 10]
        if verbose:
            print "Selected values:", rowList1
            print "Should look like:", rowList2
        assert len(rowList1) == len(rowList2)
        assert rowList1 == rowList2

    def test09a_removeIndex(self):
        """Checking removing an index"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09a_removeIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before deletion"
            print "var1 column:", table.cols.var1
        assert table.colindexed["var1"] == 1
        assert idxcol is not None

        # delete the index
        table.removeIndex(idxcol)
        if verbose:
            print "After deletion"
            print "var1 column:", table.cols.var1
        assert table.cols.var1.index is None
        assert table.colindexed["var1"] == 0

        # re-create the index again
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        if verbose:
            print "After re-creation"
            print "var1 column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1

    def test09b_removeIndex(self):
        """Checking removing an index (persistent version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09b_removeIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before deletion"
            print "var1 index column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1
        # delete the index
        table.removeIndex(idxcol)

        # close and reopen the file
        self.fileh.close()
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index

        if verbose:
            print "After deletion"
            print "var1 column:", table.cols.var1
        assert table.cols.var1.index is None
        assert table.colindexed["var1"] == 0

        # re-create the index again
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        if verbose:
            print "After re-creation"
            print "var1 column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1

    def test10a_moveIndex(self):
        """Checking moving a table with an index"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10a_moveIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before move"
            print "var1 column:", idxcol
        assert table.colindexed["var1"] == 1
        assert idxcol is not None

        # Create a new group called "agroup"
        agroup = self.fileh.createGroup("/", "agroup")

        # move the table to "agroup"
        table.move(agroup, "table2")
        if verbose:
            print "After move"
            print "var1 column:", idxcol
        assert table.cols.var1.index is not None
        assert table.colindexed["var1"] == 1

        # Some sanity checks
        rowList1 = table.getWhereList(table.cols.var1 < "10", "python")
        rowList2 = [p.nrow for p in table if p['var1'] < "10"]
        if verbose:
            print "Selected values:", rowList1
            print "Should look like:", rowList2
        assert len(rowList1) == len(rowList2)
        assert rowList1 == rowList2

    def test10b_moveIndex(self):
        """Checking moving a table with an index (persistent version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10b_moveIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before move"
            print "var1 index column:", idxcol
        assert idxcol is not None
        assert table.colindexed["var1"] == 1
        # Create a new group called "agroup"
        agroup = self.fileh.createGroup("/", "agroup")

        # move the table to "agroup"
        table.move(agroup, "table2")

        # close and reopen the file
        self.fileh.close()
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.agroup.table2
        idxcol = table.cols.var1.index

        if verbose:
            print "After move"
            print "var1 column:", idxcol
        assert table.cols.var1.index is not None
        assert table.colindexed["var1"] == 1

        # Some sanity checks
        rowList1 = table.getWhereList(table.cols.var1 < "10", "python")
        rowList2 = [p.nrow for p in table if p['var1'] < "10"]
        if verbose:
            print "Selected values:", rowList1
            print "Should look like:", rowList2
        assert len(rowList1) == len(rowList2)
        assert rowList1 == rowList2


    def test11a_removeTableWithIndex(self):
        """Checking removing a table with indexes"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11a_removeTableWithIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-only mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before deletion"
            print "var1 column:", table.cols.var1
        assert table.colindexed["var1"] == 1
        assert idxcol is not None

        # delete the table
        self.fileh.removeNode("/table")
        if verbose:
            print "After deletion"
        assert "table" not in self.fileh.root

        # re-create the table and the index again
        table = self.fileh.createTable("/", 'table', Small, "New table",
                                       self.filters, self.nrows)
        for i in range(self.nrows):
            table.row['var1'] = str(i)
            table.row['var2'] = i % 2
            table.row['var3'] = i
            table.row['var4'] = float(self.nrows - i - 1)
            table.row.append()
        table.flush()
        # Index all entries:
        indexrows = table.cols.var1.createIndex(testmode=1)
        indexrows = table.cols.var2.createIndex(testmode=1)
        indexrows = table.cols.var3.createIndex(testmode=1)
        indexrows = table.cols.var4.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        if verbose:
            print "After re-creation"
            print "var1 column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1

    def test11b_removeTableWithIndex(self):
        """Checking removing a table with indexes (persistent version 2)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11b_removeTableWithIndex..." % self.__class__.__name__

        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before deletion"
            print "var1 column:", table.cols.var1
        assert table.colindexed["var1"] == 1
        assert idxcol is not None

        # delete the table
        self.fileh.removeNode("/table")
        if verbose:
            print "After deletion"
        assert "table" not in self.fileh.root

        # close and reopen the file
        self.fileh.close()
        self.fileh = openFile(self.file, mode = "r+")

        # re-create the table and the index again
        table = self.fileh.createTable("/", 'table', Small, "New table",
                                       self.filters, self.nrows)
        for i in range(self.nrows):
            table.row['var1'] = str(i)
            table.row['var2'] = i % 2
            table.row['var3'] = i
            table.row['var4'] = float(self.nrows - i - 1)
            table.row.append()
        table.flush()
        # Index all entries:
        indexrows = table.cols.var1.createIndex(testmode=1)
        indexrows = table.cols.var2.createIndex(testmode=1)
        indexrows = table.cols.var3.createIndex(testmode=1)
        indexrows = table.cols.var4.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        if verbose:
            print "After re-creation"
            print "var1 column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1

    # Test provided by Andrew Straw
    def test11c_removeTableWithIndex(self):
        """Checking removing a table with indexes (persistent version 2)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11c_removeTableWithIndex..." % self.__class__.__name__

        class Distance(IsDescription):
            frame = Int32Col(pos=0, indexed=True)
            distance = FloatCol(pos=1)

        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode='w')
        table = self.fileh.createTable(self.fileh.root, 'distance_table', Distance)
        r = table.row
        for i in range(10):
            r['frame']=i
            r['distance']=float(i**2)
            r.append()
        table.flush()
        self.fileh.close()

        self.fileh = openFile(self.file, mode='r+')
        self.fileh.removeNode(self.fileh.root.distance_table)


class BasicReadTestCase(BasicTestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class ZlibReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class LZOReadTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    shuffle = 0
    fletcher32 = 0
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class BZIP2ReadTestCase(BasicTestCase):
    compress = 1
    complib = "bzip2"
    shuffle = 0
    fletcher32 = 0
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class ShuffleReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 0
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class Fletcher32ReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 0
    fletcher32 = 1
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class ShuffleFletcher32ReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 1
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss

class OneHalfTestCase(BasicTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss+ss//2

class UpperBoundTestCase(BasicTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss+1

class LowerBoundTestCase(BasicTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss*2-1

# This warning has non-sense now in the PyTables Pro version, as *all*
# the rows can be indexed there
class WarningTestCase(unittest.TestCase):
    nrows = 100 # Small enough to raise the warning

    def test01(self):
        "Checking the user warning for too few entries to index"
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        self.rootgroup = self.fileh.root
        group = self.rootgroup
        # Create a table
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

class DeepTableIndexTestCase(unittest.TestCase):
    nrows = minRowIndex

    def test01(self):
        "Checking the indexing of a table in a 2nd level hierarchy"
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.createGroup(self.fileh.root,"agroup")
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        table = self.fileh.createTable(group, 'table', Small, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        # Some sanity checks
        assert table.colindexed["var1"] == 1
        assert idxcol is not None
        assert idxcol.nelements == self.nrows

        self.fileh.close()
        os.remove(self.file)

    def test01b(self):
        "Checking the indexing of a table in 2nd level (persistent version)"
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.createGroup(self.fileh.root,"agroup")
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        table = self.fileh.createTable(group, 'table', Small, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        # Close and re-open this file
        self.fileh.close()
        self.fileh = openFile(self.file, "a")
        table = self.fileh.root.agroup.table
        idxcol = table.cols.var1.index
        # Some sanity checks
        assert table.colindexed["var1"] == 1
        assert idxcol is not None
        assert idxcol.nelements == self.nrows

        self.fileh.close()
        os.remove(self.file)

    def test02(self):
        "Checking the indexing of a table in a 4th level hierarchy"
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.createGroup(self.fileh.root,"agroup")
        group = self.fileh.createGroup(group,"agroup")
        group = self.fileh.createGroup(group,"agroup")
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        table = self.fileh.createTable(group, 'table', Small, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        # Some sanity checks
        assert table.colindexed["var1"] == 1
        assert idxcol is not None
        assert idxcol.nelements == self.nrows

        self.fileh.close()
        os.remove(self.file)

    def test02b(self):
        "Checking the indexing of a table in a 4th level (persistent version)"
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.createGroup(self.fileh.root,"agroup")
        group = self.fileh.createGroup(group,"agroup")
        group = self.fileh.createGroup(group,"agroup")
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        table = self.fileh.createTable(group, 'table', Small, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        # Close and re-open this file
        self.fileh.close()
        self.fileh = openFile(self.file, "a")
        table = self.fileh.root.agroup.agroup.agroup.table
        idxcol = table.cols.var1.index
        # Some sanity checks
        assert table.colindexed["var1"] == 1
        assert idxcol is not None
        assert idxcol.nelements == self.nrows

        self.fileh.close()
        os.remove(self.file)

    def test03(self):
        "Checking the indexing of a table in a 100th level hierarchy"
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        group = self.fileh.root
        for i in range(100):
            group = self.fileh.createGroup(group,"agroup")
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        table = self.fileh.createTable(group, 'table', Small, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex(testmode=1)
        idxcol = table.cols.var1.index
        # Some sanity checks
        assert table.colindexed["var1"] == 1
        assert idxcol is not None
        assert idxcol.nelements == self.nrows

        self.fileh.close()
        os.remove(self.file)

class NoAuto(IsDescription):
    _v_indexprops = IndexProps(auto=0)
    var1 = StringCol(length=4, dflt="", pos=1, indexed=1)
    var2 = BoolCol(0, indexed=1, pos = 2)
    var3 = IntCol(0, indexed=1, pos = 3)
    var4 = FloatCol(0, indexed=0, pos = 4)

class NoReindex(IsDescription):
    _v_indexprops = IndexProps(reindex=0)
    var1 = StringCol(length=4, dflt="", indexed=1, pos=1)
    var2 = BoolCol(0, indexed=1, pos=2)
    var3 = IntCol(0, indexed=1, pos=3)
    var4 = FloatCol(0, indexed=0, pos=4)

class ChangeFilters(IsDescription):
    _v_indexprops = IndexProps(filters=Filters(complevel=6, complib="zlib",
                                               shuffle=False, fletcher32=True))
    var1 = StringCol(length=4, dflt="", indexed=1, pos=1)
    var2 = BoolCol(0, indexed=1, pos=2)
    var3 = IntCol(0, indexed=1, pos=3)
    var4 = FloatCol(0, indexed=0, pos=4)


class AutomaticIndexingTestCase(unittest.TestCase):
    reopen = 1
    klass = NoAuto

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        # Create a table
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
        cleanup(self)

    def test01_attrs(self):
        "Checking indexing attributes (part1)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_attrs..." % self.__class__.__name__

        table = self.table
        if self.klass is Small:
            assert table.indexed == 0
        else:
            assert table.indexed == 1
        if self.klass is Small:
            assert table.colindexed["var1"] == 0
            assert table.cols.var1.index is None
            assert table.colindexed["var2"] == 0
            assert table.cols.var2.index is None
            assert table.colindexed["var3"] == 0
            assert table.cols.var3.index is None
            assert table.colindexed["var4"] == 0
            assert table.cols.var4.index is None
        else:
            # Check that the var1, var2 and var3 (and only these)
            # has been indexed
            assert table.colindexed["var1"] == 1
            assert table.cols.var1.index is not None
            assert table.colindexed["var2"] == 1
            assert table.cols.var2.index is not None
            assert table.colindexed["var3"] == 1
            assert table.cols.var3.index is not None
            assert table.colindexed["var4"] == 0
            assert table.cols.var4.index is None

    def test02_attrs(self):
        "Checking indexing attributes (part2)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_attrs..." % self.__class__.__name__

        table = self.table
        # Check the policy parameters
        if verbose:
            if table.indexed:
                print "indexprops:", table.indexprops
            else:
                print "Table is not indexed"
        # Check non-default values for index saving policy
        if self.klass is Small:
            assert table.indexprops is not None
        elif self.klass is NoAuto:
            assert table.indexprops.auto == False
            assert table.indexprops.reindex == True
            filters = Filters(complevel=1, complib="zlib",
                              shuffle=True, fletcher32=False)
            assert str(table.indexprops.filters) == str(filters)
        elif self.klass is NoReindex:
            assert table.indexprops.auto == True
            assert table.indexprops.reindex == False
            filters = Filters(complevel=1, complib="zlib",
                              shuffle=True, fletcher32=False)
            assert str(table.indexprops.filters) == str(filters)
        elif self.klass is ChangeFilters:
            assert table.indexprops.auto == True
            assert table.indexprops.reindex == True
            filters = Filters(complevel=6, complib="zlib",
                              shuffle=False, fletcher32=True)
            assert str(table.indexprops.filters) == str(filters)

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

    def test03_counters(self):
        "Checking indexing counters"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_counters..." % self.__class__.__name__
        table = self.table
        # Check the counters for indexes
        if verbose:
            if table.indexed:
                print "indexedrows:", table._indexedrows
                print "unsavedindexedrows:", table._unsaved_indexedrows
                index = table.cols.var1.index
                print "table rows:", table.nrows
                print "computed indexed rows:", index.nrows * index.slicesize
            else:
                print "Table is not indexed"
        if self.klass is not Small:
            index = table.cols.var1.index
            indexedrows = index.nelements
            assert table._indexedrows == indexedrows
            indexedrows = index.nelements
            assert table._unsaved_indexedrows == self.nrows - indexedrows

    def test04_noauto(self):
        "Checking indexing counters (non-automatic mode)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_noauto..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Check the counters for indexes
        if verbose:
            if table.indexed:
                print "indexedrows:", table._indexedrows
                print "unsavedindexedrows:", table._unsaved_indexedrows
                index = table.cols.var1.index
                print "computed indexed rows:", index.nelements
            else:
                print "Table is not indexed"

        # No unindexated rows should remain
        index = table.cols.var1.index
        if self.klass is Small:
            assert index is None
        else:
            indexedrows = index.nelements
            assert table._indexedrows == index.nelements
            assert table._unsaved_indexedrows == self.nrows - indexedrows

        # Check non-default values for index saving policy
        if self.klass is Small:
            assert table.indexprops is not None
        elif self.klass is NoAuto:
            assert table.indexprops.auto == False
            assert table.indexprops.reindex == True
            filters = Filters(complevel=1, complib="zlib",
                              shuffle=True, fletcher32=False)
            assert str(table.indexprops.filters) == str(filters)
        elif self.klass is NoReindex:
            assert table.indexprops.auto == True
            assert table.indexprops.reindex == False
            filters = Filters(complevel=1, complib="zlib",
                              shuffle=True, fletcher32=False)
            assert str(table.indexprops.filters) == str(filters)
        elif self.klass is ChangeFilters:
            assert table.indexprops.auto == True
            assert table.indexprops.reindex == True
            filters = Filters(complevel=6, complib="zlib",
                              shuffle=False, fletcher32=True)
            assert str(table.indexprops.filters) == str(filters)


    def test05_icounters(self):
        "Checking indexing counters (removeRows)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_icounters..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, remove some rows:
        table.removeRows(3,5)
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check the counters for indexes
        if verbose:
            if table.indexed:
                print "indexedrows:", table._indexedrows
                print "original indexedrows:", indexedrows
                print "unsavedindexedrows:", table._unsaved_indexedrows
                print "original unsavedindexedrows:", unsavedindexedrows
                index = table.cols.var1.index
                print "index dirty:", table.cols.var1.dirty
            else:
                print "Table is not indexed"

        # Check the counters
        assert table.nrows == self.nrows - 2
        if self.klass is NoReindex:
            # I'm not sure that the results below are what we want...
            # But I don't think this is going to be important
            # the important thing is that dirtiness is working right
            # Francesc Altet 2004-12-31
#             if self.reopen:
#                 assert table._indexedrows == indexedrows - 2
#                 assert table._unsaved_indexedrows == unsavedindexedrows + 2
#             else:
#                 assert table._indexedrows == indexedrows
            # The next values should be more consistent... 2005-01-03
#             assert table._indexedrows == 0
#             assert table._unsaved_indexedrows == table.nrows
            pass
        elif self.klass is NoAuto:
            index = table.cols.var1.index
            indexedrows = index.nelements
            assert table._indexedrows == indexedrows
            assert table._indexedrows == index.nelements
            assert table._unsaved_indexedrows == self.nrows - indexedrows - 2

        # Check non-default values for index saving policy
        if self.klass is Small:
            assert table.indexprops is not None
        elif self.klass is NoAuto:
            assert table.indexprops.auto == False
            assert table.indexprops.reindex == True
            filters = Filters(complevel=1, complib="zlib",
                              shuffle=True, fletcher32=False)
            assert str(table.indexprops.filters) == str(filters)
        elif self.klass is NoReindex:
            assert table.indexprops.auto == True
            assert table.indexprops.reindex == False
            filters = Filters(complevel=1, complib="zlib",
                              shuffle=False, fletcher32=False)
            assert str(table.indexprops.filters) == str(filters)
        elif self.klass is ChangeFilters:
            assert table.indexprops.auto == True
            assert table.indexprops.reindex == True
            filters = Filters(complevel=6, complib="zlib",
                              shuffle=False, fletcher32=True)
            assert str(table.indexprops.filters) == str(filters)


    def test06_dirty(self):
        "Checking dirty flags (removeRows action)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_dirty..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Now, remove some rows:
        table.removeRows(3,5)
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check the dirty flag for indexes
        if verbose:
            print "reindex flag:", table.indexprops.reindex
            for colname in table.colnames:
                print "dirty flag col %s: %s" % \
                      (colname, table.cols._f_col(colname).dirty)
        # Check the flags
        for colname in table.colnames:
            if table.cols._f_col(colname).index:
                if not table.indexprops.reindex:
                    assert table.cols._f_col(colname).dirty == True
                else:
                    assert table.cols._f_col(colname).dirty == False
            else:
                assert table.cols._f_col(colname).dirty == True

    def test07_noreindex(self):
        "Checking indexing counters (modifyRows, no-reindex mode)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_noreindex..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # No unindexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, modify just one row:
        table.modifyRows(3, None, 1, [["asa",0,3,3.1]])
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
        # Check the counters for indexes
        if verbose:
            if table.indexed:
                print "indexedrows:", table._indexedrows
                print "original indexedrows:", indexedrows
                print "unsavedindexedrows:", table._unsaved_indexedrows
                print "original unsavedindexedrows:", unsavedindexedrows
                index = table.cols.var1.index
                print "computed indexed rows:", index.nelements
            else:
                print "Table is not indexed"
        # Check the counters
        assert table.nrows == self.nrows
        if self.klass is NoReindex:
            # The unsaved indexed rows counter should be unchanged
#             assert table._indexedrows == indexedrows
#             assert table._unsaved_indexedrows == unsavedindexedrows
            # I'm not sure that the results below are what we want...
            # But I don't think this is going to be important
            # the important thing is that dirtiness is working right
            # Francesc Altet 2004-12-31
#             if self.reopen:
#                 assert table._indexedrows == indexedrows - 2
#                 assert table._unsaved_indexedrows == unsavedindexedrows + 2
#             else:
#                 assert table._indexedrows == indexedrows
            pass
        elif self.klass is NoAuto:
            index = table.cols.var1.index
            indexedrows = index.nelements
            assert table._indexedrows == indexedrows
            indexedrows = index.nelements
            assert table._indexedrows == indexedrows
            assert table._unsaved_indexedrows == self.nrows - indexedrows

        # Check the dirty flag for indexes
        if verbose:
            for colname in table.colnames:
                print "dirty flag col %s: %s" % \
                      (colname, table.cols._f_col(colname).dirty)
        for colname in table.colnames:
            if table.cols._f_col(colname).index:
                if not table.indexprops.reindex:
                    assert table.cols._f_col(colname).dirty == True
                else:
                    assert table.cols._f_col(colname).dirty == False
            else:
                assert table.cols._f_col(colname).dirty == True

    def test08_dirty(self):
        "Checking dirty flags (modifyColumns)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_dirty..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, modify a couple of rows:
        table.modifyColumns(1, columns=[["asa","asb"],[1.,2.]],
                            names=["var1", "var4"])
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table

        # Check the counters
        assert table.nrows == self.nrows
        if self.klass is NoReindex:
            # The unsaved indexed rows counter should be unchanged
            assert table._indexedrows == indexedrows
            assert table._unsaved_indexedrows == unsavedindexedrows
        elif self.klass is NoAuto:
            index = table.cols.var1.index
            indexedrows = index.nelements
            assert table._indexedrows == indexedrows
            assert table._unsaved_indexedrows == self.nrows - indexedrows

        # Check the dirty flag for indexes
        if verbose:
            for colname in table.colnames:
                print "dirty flag col %s: %s" % \
                      (colname, table.cols._f_col(colname).dirty)
        for colname in table.colnames:
            if table.cols._f_col(colname).index:
                if not table.indexprops.reindex:
                    if colname in ["var1"]:
                        assert table.cols._f_col(colname).dirty == True
                    else:
                        assert table.cols._f_col(colname).dirty == False
                else:
                    assert table.cols._f_col(colname).dirty == False
            else:
                assert table.cols._f_col(colname).dirty == True

    def test09_copyIndex(self):
        "Checking copy Index feature in copyTable (attrs)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_copyIndex..." % self.__class__.__name__
        table = self.table
        # Don't force a sync in indexes
        #table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, remove some rows to make columns dirty
        #table.removeRows(3,5)
        # Copy a Table to another location
        warnings.filterwarnings("ignore", category=UserWarning)
        table2 = table.copy("/", 'table2')
        warnings.filterwarnings("default", category=UserWarning)
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
            table2 = self.fileh.root.table2

        index1 = table.cols.var1.index
        index2 = table2.cols.var1.index
        if verbose:
            print "Copied index:", index2
            print "Original index:", index1
            if index1:
                print "Elements in copied index:", index2.nelements
                print "Elements in original index:", index1.nelements
        # Check the counters
        assert table.nrows == table2.nrows
        if table.indexed:
            assert table2.indexed
            assert table._indexedrows == table2._indexedrows
            assert table._unsaved_indexedrows == table2._unsaved_indexedrows
        if self.klass is Small:
            # No index: the index should not exist
            assert index1 is None
            assert index2 is None
        elif self.klass is NoAuto:
            # No auto: the index should exists, but be empty
            assert index2 is not None
            assert index2.nelements == 0
        elif self.klass is NoReindex:
            # Auto: the index should exists, and have elements
            assert index2 is not None
            assert index2.nelements == index1.nelements

        # Check the dirty flag for indexes
        if verbose:
            for colname in table2.colnames:
                print "dirty flag col %s: %s" % \
                      (colname, table2.cols._f_col(colname).dirty)
        for colname in table2.colnames:
            if table2.cols._f_col(colname).index:
                assert table2.cols._f_col(colname).dirty == False
            else:
                assert table2.cols._f_col(colname).dirty == True

    def test10_copyIndex(self):
        "Checking copy Index feature in copyTable (values)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_copyIndex..." % self.__class__.__name__
        table = self.table
        # Don't force a sync in indexes
        #table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, remove some rows to make columns dirty
        #table.removeRows(3,5)
        # Copy a Table to another location
        warnings.filterwarnings("ignore", category=UserWarning)
        table2 = table.copy("/", 'table2')
        warnings.filterwarnings("default", category=UserWarning)
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
            table2 = self.fileh.root.table2

        index1 = table.cols.var3.index
        index2 = table2.cols.var3.index
        if verbose:
            print "Copied index:", index2
            print "Original index:", index1
            if index1:
                print "Elements in copied index:", index2.nelements
                print "Elements in original index:", index1.nelements
                if index2.nelements > 10:
                    print "First 10 elements in copied index (sorted):\n", \
                          index2.sorted[0,:10]
                    print "First 10 elements in orig index (sorted):\n", \
                          index1.sorted[0,:10]
                    print "First 10 elements in copied index (indices):\n", \
                          index2.indices[0,:10]
                    print "First 10 elements in orig index (indices):\n", \
                          index1.indices[0,:10]
        if self.klass is NoReindex:
            # Auto: the index should exists, and have equal elements
            assert allequal(index2.sorted.read(), index1.sorted.read())
            # The next assertion cannot be guaranteed. Why?
            # sorting algorithm in numarray is not deterministic?
            #assert allequal(index2.indices.read(), index1.indices.read())

    def test11_copyIndex(self):
        "Checking copy Index feature in copyTable (dirty flags)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_copyIndex..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.klass is not Small:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, modify an indexed column and an unindexed one
        # to make the "var1" dirty
        table.modifyColumns(1, columns=[["asa","asb"],[1.,2.]],
                            names=["var1", "var4"])
        # Copy a Table to another location
        warnings.filterwarnings("ignore", category=UserWarning)
        table2 = table.copy("/", 'table2')
        warnings.filterwarnings("default", category=UserWarning)
        if self.reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, "a")
            table = self.fileh.root.table
            table2 = self.fileh.root.table2

        index1 = table.cols.var1.index
        index2 = table2.cols.var1.index
        if verbose:
            print "Copied index:", index2
            print "Original index:", index1
            if index1:
                print "Elements in copied index:", index2.nelements
                print "Elements in original index:", index1.nelements

        # Check the dirty flag for indexes
        if verbose:
            for colname in table2.colnames:
                print "dirty flag col %s: %s" % \
                      (colname, table2.cols._f_col(colname).dirty)
        for colname in table2.colnames:
            if table2.cols._f_col(colname).index:
                if table2.indexprops.reindex:
                    # All the destination columns should be non-dirty because
                    # the copy removes the dirty state and puts the
                    # index in a sane state
                    assert table.cols._f_col(colname).dirty == False
                    assert table2.cols._f_col(colname).dirty == False
            else:
                assert table2.cols._f_col(colname).dirty == True


# minRowIndex = 10000  # just if one wants more indexed rows to be checked
class AI1TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    klass = NoAuto

class AI2TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 1
    klass = NoAuto

class AI3TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 1
    klass = NoReindex

class AI4aTestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    klass = NoReindex

class AI4bTestCase(AutomaticIndexingTestCase):
    #nrows = 10012
    nrows = 112
    reopen = 1
    klass = NoReindex

class AI5TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=0)[0]
    nrows = ss*11-1
    reopen = 0
    klass = NoAuto

class AI6TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=0)[0]
    nrows = ss*21+1
    reopen = 1
    klass = NoAuto

class AI7TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=0)[0]
    nrows = ss*12-1
    reopen = 0
    klass = NoReindex

class AI8TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=0)[0]
    nrows = ss*15+100
    reopen = 1
    klass = NoReindex

class AI9TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, optlevel=1, testmode=1)[0]
    nrows = ss
    reopen = 0
    klass = Small

class AI10TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 1
    klass = Small

class AI11TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    klass = ChangeFilters

class AI12TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    klass = ChangeFilters


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    #heavy = 1  # Uncomment this only for testing purposes!

#     theSuite.addTest(unittest.makeSuite(BasicReadTestCase))
#     theSuite.addTest(unittest.makeSuite(AI5TestCase))
#     theSuite.addTest(unittest.makeSuite(AI6TestCase))
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicReadTestCase))
        theSuite.addTest(unittest.makeSuite(ZlibReadTestCase))
        theSuite.addTest(unittest.makeSuite(LZOReadTestCase))
        theSuite.addTest(unittest.makeSuite(BZIP2ReadTestCase))
        theSuite.addTest(unittest.makeSuite(ShuffleReadTestCase))
        theSuite.addTest(unittest.makeSuite(Fletcher32ReadTestCase))
        theSuite.addTest(unittest.makeSuite(ShuffleFletcher32ReadTestCase))
        theSuite.addTest(unittest.makeSuite(OneHalfTestCase))
        theSuite.addTest(unittest.makeSuite(UpperBoundTestCase))
        theSuite.addTest(unittest.makeSuite(LowerBoundTestCase))
        theSuite.addTest(unittest.makeSuite(AI1TestCase))
        theSuite.addTest(unittest.makeSuite(AI2TestCase))
        theSuite.addTest(unittest.makeSuite(AI9TestCase))
        theSuite.addTest(unittest.makeSuite(DeepTableIndexTestCase))
    if heavy:
        # These are too heavy for normal testing
        theSuite.addTest(unittest.makeSuite(AI3TestCase))
        theSuite.addTest(unittest.makeSuite(AI4aTestCase))
        theSuite.addTest(unittest.makeSuite(AI4bTestCase))
        theSuite.addTest(unittest.makeSuite(AI5TestCase))
        theSuite.addTest(unittest.makeSuite(AI6TestCase))
        theSuite.addTest(unittest.makeSuite(AI7TestCase))
        theSuite.addTest(unittest.makeSuite(AI8TestCase))
        theSuite.addTest(unittest.makeSuite(AI10TestCase))
        theSuite.addTest(unittest.makeSuite(AI11TestCase))
        theSuite.addTest(unittest.makeSuite(AI12TestCase))

    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
