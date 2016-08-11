from .common import TempFileMixin
import unittest

import numpy as np
import tables as tb


class Record(tb.IsDescription):
    var1 = tb.StringCol(itemsize=4)  # 4-character String
    var2 = tb.IntCol()      # integer
    var3 = tb.Int16Col()    # short integer


class Queries(TempFileMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.table = self.h5file.create_table(self.root, 'atable', Record,
                                              "Table title")
        for i in range(10):
            row = ('abcd', 0, i)
            self.table.append([row])

    def test_callable(self):
        result = list(r[:] for r in self.table.where(
            lambda r: r['var3'] >= 5))
        assert result[0] == (b'abcd', 0, 5)

    def test_inkernel(self):
        result = list(r[:] for r in self.table.where("var3 >= 5"))
        assert result[0] == (b'abcd', 0, 5)

    def test_inkernel_constant(self):
        c = 5
        result = list(r[:] for r in self.table.where("var3 >= c"))
        assert result[0] == (b'abcd', 0, 5)
