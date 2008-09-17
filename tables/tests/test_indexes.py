import unittest
import os
import tempfile
import warnings
import sys
import copy

from tables import *
from tables.index import Index, defaultAutoIndex, defaultIndexFilters
from tables.idxutils import calcChunksize
from tables.tests.common import verbose, allequal, heavy, cleanup, \
     PyTablesTestCase, TempFileMixin
from tables.exceptions import OldIndexWarning

# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

import numpy


# Sensible parameters for indexing with small blocksizes
minRowIndex = 10
small_blocksizes = (96, 24, 6, 3)

class TDescr(IsDescription):
    var1 = StringCol(itemsize=4, dflt="", pos=1)
    var2 = BoolCol(dflt=0, pos=2)
    var3 = IntCol(dflt=0, pos=3)
    var4 = FloatCol(dflt=0, pos=4)

class BasicTestCase(PyTablesTestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    nrows = minRowIndex
    ss = small_blocksizes[2]

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
        table = self.fileh.createTable(group, 'table', TDescr, title,
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
        for col in table.colinstances.itervalues():
            indexrows = col.createIndex(_blocksizes=small_blocksizes)
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

    def test00_flushLastRow(self):
        """Checking flushing an Index incrementing only the last row."""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_flushLastRow..." % self.__class__.__name__

        # Open the HDF5 file in append mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        # Add just 3 rows more
        for i in range(3):
            table.row['var1'] = str(i)
            table.row.append()
        table.flush()  # redo the indexes
        idxcol = table.cols.var1.index
        if verbose:
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize
            print "Elements in last row:", idxcol.indicesLR[-1]

        # Do a selection
        results = [p["var1"] for p in table.where('var1 == "1"')]
        assert len(results) == 2

    def test00_update(self):
        """Checking automatic re-indexing after an update operation."""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_update..." % self.__class__.__name__

        # Open the HDF5 file in append mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        # Modify a couple of columns
        for i,row in enumerate(table.where("(var3>1) & (var3<5)")):
            row['var1'] = str(i)
            row['var3'] = i
            row.update()
        table.flush()  # redo the indexes
        idxcol1 = table.cols.var1.index
        idxcol3 = table.cols.var3.index
        if verbose:
            print "Dirtyness of var1 col:", idxcol1.dirty
            print "Dirtyness of var3 col:", idxcol3.dirty
        assert idxcol1.dirty == False
        assert idxcol3.dirty == False

        # Do a couple of selections
        results = [p["var1"] for p in table.where('var1 == "1"')]
        assert len(results) == 2
        results = [p["var3"] for p in table.where('var3 == 0')]
        assert len(results) == 2

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
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var1"] for p in table.where('var1 == "1"')]
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
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var2"] for p in table.where('var2 == True')]
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
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var3"] for p in table.where('(1<var3)&(var3<10)')]
        if verbose:
            print "Selected values:", results
        assert len(results) == min(10, table.nrows) - 2

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
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of rows in table:", table.nrows
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        results = [p["var4"] for p in table.where('var4 < 10')]
        #results = [p["var4"] for p in table.where('(1<var4)&(var4<10)')]
        if verbose:
            print "Selected values:", results
        assert len(results) == min(10, table.nrows)

    def test05_getWhereList(self):
        """Checking reading an Index with getWhereList (string flavor)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_getWhereList..." % self.__class__.__name__

        # Open the HDF5 file in read-write mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        table.flavor = "python"
        rowList1 = table.getWhereList('var1 < "10"')
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

        # Open the HDF5 file in read-write mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var2.index
        if verbose:
            print "Max rows in buf:", table.nrowsinbuf
            print "Rows in tables:", table.nrows
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        table.flavor = "numpy"
        rowList1 = table.getWhereList('var2 == False', sort=True)
        rowList2 = [p.nrow for p in table if p['var2'] == False]
        # Convert to a NumPy object
        rowList2 = numpy.array(rowList2, numpy.int64)
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

        # Open the HDF5 file in read-write mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        table.flavor = "python"
        rowList1 = table.getWhereList('var3 < 15', sort=True)
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

        # Open the HDF5 file in read-write mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var4.index
        if verbose:
            print "Max rows in buf:", table.nrowsinbuf
            print "Number of elements per slice:", idxcol.slicesize
            print "Chunk size:", idxcol.sorted.chunksize

        # Do a selection
        table.flavor = "python"
        rowList1 = table.getWhereList('var4 < 10', sort=True)
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

        # Open the HDF5 file in read-write mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before deletion"
            print "var1 column:", table.cols.var1
        assert table.colindexed["var1"] == 1
        assert idxcol is not None

        # delete the index
        table.cols.var1.removeIndex()
        if verbose:
            print "After deletion"
            print "var1 column:", table.cols.var1
        assert table.cols.var1.index is None
        assert table.colindexed["var1"] == 0

        # re-create the index again
        indexrows = table.cols.var1.createIndex(_blocksizes=small_blocksizes)
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

        # Open the HDF5 file in read-write mode
        self.fileh = openFile(self.file, mode = "a")
        table = self.fileh.root.table
        idxcol = table.cols.var1.index
        if verbose:
            print "Before deletion"
            print "var1 index column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1
        # delete the index
        table.cols.var1.removeIndex()

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
        indexrows = table.cols.var1.createIndex(_blocksizes=small_blocksizes)
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

        # Open the HDF5 file in read-write mode
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
        table.flavor = "python"
        rowList1 = table.getWhereList('var1 < "10"')
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

        # Open the HDF5 file in read-write mode
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
        table.flavor = "python"
        rowList1 = table.getWhereList('var1 < "10"')
        rowList2 = [p.nrow for p in table if p['var1'] < "10"]
        if verbose:
            print "Selected values:", rowList1, type(rowList1)
            print "Should look like:", rowList2, type(rowList2)
        assert len(rowList1) == len(rowList2)
        assert rowList1 == rowList2


    def test11a_removeTableWithIndex(self):
        """Checking removing a table with indexes"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11a_removeTableWithIndex..." % self.__class__.__name__

        # Open the HDF5 file in read-write mode
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
        table = self.fileh.createTable("/", 'table', TDescr, "New table",
                                       self.filters, self.nrows)
        for i in range(self.nrows):
            table.row['var1'] = str(i)
            table.row['var2'] = i % 2
            table.row['var3'] = i
            table.row['var4'] = float(self.nrows - i - 1)
            table.row.append()
        table.flush()
        # Index all entries:
        for col in table.colinstances.itervalues():
            indexrows = col.createIndex(_blocksizes=small_blocksizes)
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
        table = self.fileh.createTable("/", 'table', TDescr, "New table",
                                       self.filters, self.nrows)
        for i in range(self.nrows):
            table.row['var1'] = str(i)
            table.row['var2'] = i % 2
            table.row['var3'] = i
            table.row['var4'] = float(self.nrows - i - 1)
            table.row.append()
        table.flush()
        # Index all entries:
        for col in table.colinstances.itervalues():
            indexrows = col.createIndex(_blocksizes=small_blocksizes)
        idxcol = table.cols.var1.index
        if verbose:
            print "After re-creation"
            print "var1 column:", table.cols.var1
        assert idxcol is not None
        assert table.colindexed["var1"] == 1

    # Test provided by Andrew Straw
    def test11c_removeTableWithIndex(self):
        """Checking removing a table with indexes (persistent version 3)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11c_removeTableWithIndex..." % self.__class__.__name__

        class Distance(IsDescription):
            frame = Int32Col(pos=0)
            distance = FloatCol(pos=1)

        # Delete the old temporal file
        os.remove(self.file)

        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode='w')
        table = self.fileh.createTable(self.fileh.root, 'distance_table', Distance)
        table.cols.frame.createIndex(_blocksizes=small_blocksizes)
        r = table.row
        for i in range(10):
            r['frame']=i
            r['distance']=float(i**2)
            r.append()
        table.flush()
        self.fileh.close()

        self.fileh = openFile(self.file, mode='r+')
        self.fileh.removeNode(self.fileh.root.distance_table)


small_ss = small_blocksizes[2]
class BasicReadTestCase(BasicTestCase):
    compress = 0
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    nrows = small_ss

class ZlibReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 0
    fletcher32 = 0
    nrows = small_ss

class LZOReadTestCase(BasicTestCase):
    compress = 1
    complib = "lzo"
    shuffle = 0
    fletcher32 = 0
    nrows = small_ss

class BZIP2ReadTestCase(BasicTestCase):
    compress = 1
    complib = "bzip2"
    shuffle = 0
    fletcher32 = 0
    nrows = small_ss

class ShuffleReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 0
    nrows = small_ss

class Fletcher32ReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 0
    fletcher32 = 1
    nrows = small_ss

class ShuffleFletcher32ReadTestCase(BasicTestCase):
    compress = 1
    complib = "zlib"
    shuffle = 1
    fletcher32 = 1
    nrows = small_ss

class OneHalfTestCase(BasicTestCase):
    nrows = small_ss+small_ss//2

class UpperBoundTestCase(BasicTestCase):
    nrows = small_ss+1

class LowerBoundTestCase(BasicTestCase):
    nrows = small_ss*2-1


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
        table = self.fileh.createTable(group, 'table', TDescr, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex()
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
        table = self.fileh.createTable(group, 'table', TDescr, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex()
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
        table = self.fileh.createTable(group, 'table', TDescr, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex()
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
        table = self.fileh.createTable(group, 'table', TDescr, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex()
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
        table = self.fileh.createTable(group, 'table', TDescr, title,
                                       None, self.nrows)
        for i in range(self.nrows):
            # Fill rows with defaults
            table.row.append()
        table.flush()
        # Index some column
        indexrows = table.cols.var1.createIndex()
        idxcol = table.cols.var1.index
        # Some sanity checks
        assert table.colindexed["var1"] == 1
        assert idxcol is not None
        assert idxcol.nelements == self.nrows

        self.fileh.close()
        os.remove(self.file)


class IndexProps(object):
    def __init__(self, auto=defaultAutoIndex, filters=defaultIndexFilters):
        self.auto = auto
        self.filters = filters

DefaultProps = IndexProps()
NoAutoProps = IndexProps(auto=False)
ChangeFiltersProps = IndexProps(
    filters=Filters( complevel=6, complib="zlib",
                     shuffle=False, fletcher32=False ) )

class AutomaticIndexingTestCase(unittest.TestCase):
    reopen = 1
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

    def setUp(self):
        # Create an instance of an HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")
        # Create a table
        title = "This is the IndexArray title"
        rowswritten = 0
        root = self.fileh.root
        self.table = self.fileh.createTable(root, 'table', TDescr, title,
                                            None, self.nrows)
        self.table.autoIndex = self.iprops.auto
        for colname in self.colsToIndex:
            self.table.colinstances[colname].createIndex()
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
        if self.iprops is DefaultProps:
            assert table.indexed == 0
        else:
            assert table.indexed == 1
        if self.iprops is DefaultProps:
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
                print "index props:", table.autoIndex
            else:
                print "Table is not indexed"
        # Check non-default values for index saving policy
        if self.iprops is NoAutoProps:
            assert not table.autoIndex
        elif self.iprops is ChangeFiltersProps:
            assert table.autoIndex

        # Check Index() objects exists and are properly placed
        if self.iprops is DefaultProps:
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
        if self.iprops is not DefaultProps:
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
        if self.iprops is DefaultProps:
            assert index is None
        else:
            indexedrows = index.nelements
            assert table._indexedrows == index.nelements
            assert table._unsaved_indexedrows == self.nrows - indexedrows

        # Check non-default values for index saving policy
        if self.iprops is NoAutoProps:
            assert not table.autoIndex
        elif self.iprops is ChangeFiltersProps:
            assert table.autoIndex


    def test05_icounters(self):
        "Checking indexing counters (removeRows)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_icounters..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.iprops is not DefaultProps:
            indexedrows = table._indexedrows
            unsavedindexedrows = table._unsaved_indexedrows
        # Now, remove some rows:
        table.removeRows(2,4)
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
                print "index dirty:", table.cols.var1.index.dirty
            else:
                print "Table is not indexed"

        # Check the counters
        assert table.nrows == self.nrows - 2
        if self.iprops is NoAutoProps:
            assert table.cols.var1.index.dirty

        # Check non-default values for index saving policy
        if self.iprops is NoAutoProps:
            assert not table.autoIndex
        elif self.iprops is ChangeFiltersProps:
            assert table.autoIndex


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
            print "auto flag:", table.autoIndex
            for colname in table.colnames:
                if table.cols._f_col(colname).index:
                    print "dirty flag col %s: %s" % \
                          (colname, table.cols._f_col(colname).index.dirty)
        # Check the flags
        for colname in table.colnames:
            if table.cols._f_col(colname).index:
                if not table.autoIndex:
                    assert table.cols._f_col(colname).index.dirty == True
                else:
                    assert table.cols._f_col(colname).index.dirty == False

    def test07_noauto(self):
        "Checking indexing counters (modifyRows, no-auto mode)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_noauto..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # No unindexated rows should remain here
        if self.iprops is not DefaultProps:
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
        if self.iprops is NoAutoProps:
            assert table.cols.var1.index.dirty

        # Check the dirty flag for indexes
        if verbose:
            for colname in table.colnames:
                if table.cols._f_col(colname).index:
                    print "dirty flag col %s: %s" % \
                          (colname, table.cols._f_col(colname).index.dirty)
        for colname in table.colnames:
            if table.cols._f_col(colname).index:
                if not table.autoIndex:
                    assert table.cols._f_col(colname).index.dirty == True
                else:
                    assert table.cols._f_col(colname).index.dirty == False

    def test08_dirty(self):
        "Checking dirty flags (modifyColumns)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_dirty..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.iprops is not DefaultProps:
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
        if self.iprops is NoAutoProps:
            assert table.cols.var1.index.dirty

        # Check the dirty flag for indexes
        if verbose:
            for colname in table.colnames:
                if table.cols._f_col(colname).index:
                    print "dirty flag col %s: %s" % \
                          (colname, table.cols._f_col(colname).index.dirty)
        for colname in table.colnames:
            if table.cols._f_col(colname).index:
                if not table.autoIndex:
                    if colname in ["var1"]:
                        assert table.cols._f_col(colname).index.dirty == True
                    else:
                        assert table.cols._f_col(colname).index.dirty == False
                else:
                    assert table.cols._f_col(colname).index.dirty == False

    def test09_copyIndex(self):
        "Checking copy Index feature in copyTable (attrs)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_copyIndex..." % self.__class__.__name__
        table = self.table
        # Don't force a sync in indexes
        #table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.iprops is not DefaultProps:
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
        if self.iprops is DefaultProps:
            # No index: the index should not exist
            assert index1 is None
            assert index2 is None
        elif self.iprops is NoAutoProps:
            assert index2 is not None

        # Check the dirty flag for indexes
        if verbose:
            for colname in table2.colnames:
                if table2.cols._f_col(colname).index:
                    print "dirty flag col %s: %s" % \
                          (colname, table2.cols._f_col(colname).index.dirty)
        for colname in table2.colnames:
            if table2.cols._f_col(colname).index:
                assert table2.cols._f_col(colname).index.dirty == False

    def test10_copyIndex(self):
        "Checking copy Index feature in copyTable (values)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_copyIndex..." % self.__class__.__name__
        table = self.table
        # Don't force a sync in indexes
        #table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.iprops is not DefaultProps:
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

    def test11_copyIndex(self):
        "Checking copy Index feature in copyTable (dirty flags)"
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_copyIndex..." % self.__class__.__name__
        table = self.table
        # Force a sync in indexes
        table.flushRowsToIndex()
        # Non indexated rows should remain here
        if self.iprops is not DefaultProps:
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
                if table2.cols._f_col(colname).index:
                    print "dirty flag col %s: %s" % \
                          (colname, table2.cols._f_col(colname).index.dirty)
        for colname in table2.colnames:
            if table2.cols._f_col(colname).index:
                if table2.autoIndex:
                    # All the destination columns should be non-dirty because
                    # the copy removes the dirty state and puts the
                    # index in a sane state
                    assert table.cols._f_col(colname).index.dirty == False
                    assert table2.cols._f_col(colname).index.dirty == False


# minRowIndex = 10000  # just if one wants more indexed rows to be checked
class AI1TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI2TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 1
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI4bTestCase(AutomaticIndexingTestCase):
    #nrows = 10012
    nrows = 112
    reopen = 1
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI5TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, memlevel=1)
    nrows = ss*11-1
    reopen = 0
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI6TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, memlevel=1)
    nrows = ss*21+1
    reopen = 1
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI7TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, memlevel=1)
    nrows = ss*12-1
    #nrows = ss*1-1  # faster test
    reopen = 0
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI8TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, memlevel=1)
    nrows = ss*15+100
    #nrows = ss*1+100  # faster test
    reopen = 1
    iprops = NoAutoProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI9TestCase(AutomaticIndexingTestCase):
    sbs, bs, ss, cs = calcChunksize(minRowIndex, memlevel=1)
    nrows = ss
    reopen = 0
    iprops = DefaultProps
    colsToIndex = []

class AI10TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 1
    iprops = DefaultProps
    colsToIndex = []

class AI11TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    iprops = ChangeFiltersProps
    colsToIndex = ['var1', 'var2', 'var3']

class AI12TestCase(AutomaticIndexingTestCase):
    #nrows = 10002
    nrows = 102
    reopen = 0
    iprops = ChangeFiltersProps
    colsToIndex = ['var1', 'var2', 'var3']


class ManyNodesTestCase(PyTablesTestCase):

    def setUp(self):
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, "w")

    def test00(self):
        """Indexing many nodes in one single session (based on bug #26)"""
        IdxRecord = {
            'f0': Int8Col(),
            'f1': Int8Col(),
            'f2': Int8Col(),
            }
        h5 = self.fileh
        for qn in range(5):
            for sn in range(5):
                qchr = 'chr' + str(qn)
                name = 'chr' + str(sn)
                path = "/at/%s/pt" % (qchr)
                table = h5.createTable(path, name, IdxRecord, createparents=1)
                table.cols.f0.createIndex()
                table.cols.f1.createIndex()
                table.cols.f2.createIndex()
                table.row.append()
                table.flush()

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        cleanup(self)


class IndexPropsChangeTestCase(TempFileMixin, PyTablesTestCase):
    """Test case for changing index properties in a table."""

    class MyDescription(IsDescription):
        icol = IntCol()
    oldIndexProps = IndexProps()
    newIndexProps = IndexProps(auto=False, filters=Filters(complevel=9))

    def setUp(self):
        super(IndexPropsChangeTestCase, self).setUp()
        table = self.h5file.createTable('/', 'test', self.MyDescription)
        table.autoIndex = self.oldIndexProps.auto
        row = table.row
        for i in xrange(100):
            row['icol'] = i % 25
            row.append()
        table.flush()
        self.table = table

    def tearDown(self):
        super(IndexPropsChangeTestCase, self).tearDown()

    def test_attributes(self):
        """Storing index properties as table attributes."""
        for refprops in [self.oldIndexProps, self.newIndexProps]:
            self.assertEqual(self.table.autoIndex, refprops.auto)
            self.table.autoIndex = self.newIndexProps.auto

    def test_copyattrs(self):
        """Copying index properties attributes."""
        oldtable = self.table
        newtable = oldtable.copy('/', 'test2')
        self.assertEqual(oldtable.autoIndex, newtable.autoIndex)


class IndexFiltersTestCase(TempFileMixin, PyTablesTestCase):
    """Test case for setting index filters."""

    def setUp(self):
        super(IndexFiltersTestCase, self).setUp()
        description = {'icol': IntCol()}
        self.table = self.h5file.createTable('/', 'test', description)

    def test_createIndex(self):
        """Checking input parameters in new indexes."""
        # Different from default.
        argfilters = copy.copy(defaultIndexFilters)
        argfilters.shuffle = not defaultIndexFilters.shuffle

        # Different both from default and the previous one.
        idxfilters = copy.copy(defaultIndexFilters)
        idxfilters.shuffle = not defaultIndexFilters.shuffle
        idxfilters.fletcher32 = not defaultIndexFilters.fletcher32

        icol = self.table.cols.icol

        # First create
        icol.createIndex(kind='ultralight', optlevel=4)
        self.assertEqual(icol.index.kind, 'ultralight')
        self.assertEqual(icol.index.optlevel, 4)
        self.assertEqual(icol.index.filters, defaultIndexFilters)
        icol.removeIndex()

        # Second create
        icol.createIndex(kind='medium', optlevel=3, filters=argfilters)
        self.assertEqual(icol.index.kind, 'medium')
        self.assertEqual(icol.index.optlevel, 3)
        self.assertEqual(icol.index.filters, argfilters)
        icol.removeIndex()


    def test_reindex(self):
        """Checking input parameters in recomputed indexes."""
        icol = self.table.cols.icol
        icol.createIndex(kind='full', optlevel=5, filters=Filters(complevel=3))
        kind = icol.index.kind
        optlevel = icol.index.optlevel
        filters = icol.index.filters
        icol.reIndex()
        ni = icol.index
        if verbose:
            print "Old parameters: %s, %s, %s" % (kind, optlevel, filters)
            print "New parameters: %s, %s, %s" % (
                ni.kind, ni.optlevel, ni.filters)
        self.assertEqual(ni.kind, kind)
        self.assertEqual(ni.optlevel, optlevel)
        self.assertEqual(ni.filters, filters)




class OldIndexTestCase(PyTablesTestCase):

    def test1_x(self):
        """Check that files with 1.x indexes are recognized and warned."""
        fname = self._testFilename("idx-std-1.x.h5")
        f = openFile(fname)
        self.assertWarns(OldIndexWarning, f.getNode, "/table")
        f.close()


class CompletelySortedIndexTestCase(TempFileMixin, PyTablesTestCase):
    """Test case for testing a complete sort in a table."""

    class MyDescription(IsDescription):
        icol = IntCol()

    def setUp(self):
        super(CompletelySortedIndexTestCase, self).setUp()
        table = self.h5file.createTable('/', 'test', self.MyDescription)
        row = table.row
        nrows = 100
        for i in xrange(nrows):
            row['icol'] = nrows - i
            row.append()
        table.flush()
        self.table = table

    def test_completely_sorted_index(self):
        """Testing the is_completely_sorted_index property."""
        icol = self.table.cols.icol
        # A full index with maximum optlevel should always be completely sorted
        icol.createFullIndex(optlevel=9)
        idx = icol.index
        self.assertEqual(icol.is_index_completely_sorted, True)
        icol.removeIndex()
        # As the table is small, lesser optlevels should be able to
        # create a completely sorted index too.
        icol.createFullIndex(optlevel=6)
        self.assertEqual(icol.is_index_completely_sorted, True)
        icol.removeIndex()
        # Other kinds than full, should never return a CSI
        icol.createMediumIndex(optlevel=9)
        self.assertEqual(icol.is_index_completely_sorted, False)



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
        theSuite.addTest(unittest.makeSuite(IndexPropsChangeTestCase))
        theSuite.addTest(unittest.makeSuite(IndexFiltersTestCase))
        theSuite.addTest(unittest.makeSuite(OldIndexTestCase))
        theSuite.addTest(unittest.makeSuite(CompletelySortedIndexTestCase))
    if heavy:
        # These are too heavy for normal testing
        theSuite.addTest(unittest.makeSuite(AI4bTestCase))
        theSuite.addTest(unittest.makeSuite(AI5TestCase))
        theSuite.addTest(unittest.makeSuite(AI6TestCase))
        theSuite.addTest(unittest.makeSuite(AI7TestCase))
        theSuite.addTest(unittest.makeSuite(AI8TestCase))
        theSuite.addTest(unittest.makeSuite(AI10TestCase))
        theSuite.addTest(unittest.makeSuite(AI11TestCase))
        theSuite.addTest(unittest.makeSuite(AI12TestCase))
        theSuite.addTest(unittest.makeSuite(ManyNodesTestCase))

    return theSuite

if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
