import os
import tempfile
import unittest

import numpy as np
import tables as tb
import tables.backend_h5py as tbh
import tables.core as tcore


def open_file(filename, mode="r", title=""):
    backend = tbh.open(filename, mode)
    fo = tcore.PyTableFile(backend=backend)
    fo.attrs['TITLE'] = title
    return fo


class Record(tb.IsDescription):
    var1 = tb.StringCol(itemsize=4)  # 4-character String
    var2 = tb.IntCol()      # integer
    var3 = tb.Int16Col()    # short integer
    var4 = tb.FloatCol()    # double (double-precision)
    var5 = tb.Float32Col()  # float  (single-precision)


class CreateFile(unittest.TestCase):

    def setUp(self):
        self.h5fname = tempfile.mktemp(prefix='test_createfile', suffix='.h5')
        self.h5file = open_file(self.h5fname, 'w', title='A test')
        self.root = self.h5file.root

    def tearDown(self):
        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)   # comment this for debugging purposes only

    def test_create_array(self):
        array = self.h5file.create_array(self.root, 'anarray',
                                         [1], "Array title")
        assert array[:] == np.array([1], dtype="int")

    def test_create_table(self):
        table = self.h5file.create_table(self.root, 'atable',
                                         Record, "Table title")
        row = ('abcd', 0, 1, 1.23, 1.34)
        table.append([row] * 10)

        assert len(table) == 10

    def test_create_group(self):
        group = self.h5file.create_group(self.root, 'agroup', "Group title")

        assert group._v_pathname in self.h5file
