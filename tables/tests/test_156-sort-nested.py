"""
Test case for issue #156: sorting nested tables causes AttributeError
on tables.table.Cols container.

"""

import unittest
import tempfile
import os

import tables

import numpy as np

class Issue156(tables.tests.common.PyTablesTestCase):
    def setUp(self):
        # create hdf5 file
        self.filename = tempfile.mktemp(".hdf5")
        self.file = tables.openFile(self.filename, mode="w")

        # create nested table
        class Foo(tables.IsDescription):
            frame = tables.UInt16Col()
            class Bar(tables.IsDescription):
                code = tables.UInt16Col()

        table = self.file.createTable('/', 'foo', Foo, filters=tables.Filters(3, 'zlib'), createparents=True)

        self.file.flush()

        # fill table with 10 random numbers
        for k in xrange(10):
            row = table.row
            row['frame'] = np.random.random_integers(0, 2**16-1)
            row['Bar/code'] = np.random.random_integers(0, 2**16-1)
            row.append()

        self.file.flush()


    def tearDown(self):
        self.file.close()
        os.remove(self.filename)


    def test_copysort1(self):
        # field to sort by
        field = 'frame'

        # copy table
        oldNode = self.file.getNode('/foo')
        # create completely sorted index on a main column
        oldNode.colinstances[field].createCSIndex()

        # this fails on ade2ba123efd267fd31
        try:
            newNode = oldNode.copy(newname='foo2', overwrite=True, sortby=field, checkCSI=True, propindexes=True)
        except AttributeError as e:
            self.fail("test_copysort1() raised AttributeError unexpectedly: \n"+str(e))

        # check column is sorted
        self.assertTrue( np.all( newNode.col(field) == sorted( oldNode.col(field) ) ) )
        # check index is available
        self.assertTrue( newNode.colindexes.has_key(field) )
        # check CSI was propagated
        self.assertTrue( newNode.colindexes[field].is_CSI )

    def test_copysort2(self):
        # field to sort by
        field = 'Bar/code'

        # copy table
        oldNode = self.file.getNode('/foo')
        # create completely sorted index on a main column
        oldNode.colinstances[field].createCSIndex()

        # this fails on ade2ba123efd267fd31
        try:
            newNode = oldNode.copy(newname='foo2', overwrite=True, sortby=field, checkCSI=True, propindexes=True)
        except AttributeError as e:
            self.fail("test_copysort1() raised AttributeError unexpectedly: \n"+str(e))

        # check column is sorted
        self.assertTrue( np.all( newNode.col(field) == sorted( oldNode.col(field) ) ) )
        # check index is available
        self.assertTrue( newNode.colindexes.has_key(field) )
        # check CSI was propagated
        self.assertTrue( newNode.colindexes[field].is_CSI )


if (__name__ == '__main__'):
    unittest.main()
