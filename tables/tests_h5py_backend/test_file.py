import os
import tempfile
import unittest

import numpy as np
import tables as tb
import tables.backend_h5py as tbh
import tables.core as tcore



class TempFileMixin(object):
    open_mode = 'w'
    open_kwargs = {}

    def _getTempFileName(self):
        return tempfile.mktemp(prefix='file', suffix='.h5')

    def open_file(self, filename, mode="r", title="", **kwargs):
        backend = tbh.open(filename, mode)
        fo = tcore.PyTablesFile(backend=backend, **kwargs)
        fo.attrs['TITLE'] = title
        return fo

    def setUp(self):
        """Set ``h5file`` and ``h5fname`` instance attributes.

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, empty, temporary HDF5 file.

        """

        super().setUp()
        self.h5fname = self._getTempFileName()
        self.h5file = self.open_file(
            self.h5fname, self.open_mode, title="", **self.open_kwargs)
        self.root = self.h5file.root

    def tearDown(self):
        """Close ``h5file`` and remove ``h5fname``."""

        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)   # comment this for debugging purposes only
        super().tearDown()


class Record(tb.IsDescription):
    var1 = tb.StringCol(itemsize=4)  # 4-character String
    var2 = tb.IntCol()      # integer
    var3 = tb.Int16Col()    # short integer
    var4 = tb.FloatCol()    # double (double-precision)
    var5 = tb.Float32Col()  # float  (single-precision)


class CreateFile(TempFileMixin, unittest.TestCase):

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


class ReadTable(TempFileMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.table = self.h5file.create_table(self.root, 'atable',
                                              Record, "Table title")
        self.row = ('abcd', 0, 1, 1.23, 1.34)
        self.table.append([self.row] * 10)

    def test_read_all(self):
        readout = self.table[:]
        assert len(readout) == 10
        recarr = np.array([self.row]*10, dtype="S4,i4,i2,f8,f4")
        np.testing.assert_equal(readout['var1'], recarr['f0'])
        for f1, f2 in zip(('f0', 'f1', 'f2'), ('var1', 'var2', 'var3')):
            np.testing.assert_equal(readout[f2], recarr[f1])
        for f1, f2 in zip(('f3', 'f4'), ('var4', 'var5')):
            np.testing.assert_almost_equal(readout[f2], recarr[f1])

    def test_iter(self):
        for row in self.table:
            assert row == self.row


    def test_iterrows(self):
        for row in self.table.iterrows():
            assert row[:] == self.row

