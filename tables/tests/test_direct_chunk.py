import sys

import numpy as np
import tables as tb
from tables.tests import common


class ArrayDirectChunkingTestCase(common.TempFileMixin, common.PyTablesTestCase):
    obj = np.arange(25, dtype='uint8')

    def setUp(self):
        super().setUp()
        self.array = self.h5file.create_array('/', 'array', self.obj)

    def test_chunk_info(self):
        self.assertRaises(tb.NotChunkedError,
                          self.array.chunk_info,
                          (0,) * self.array.ndim)

    def test_read_chunk(self):
        self.assertRaises(tb.NotChunkedError,
                          self.array.read_chunk,
                          (0,) * self.array.ndim)

    def test_read_chunk_out(self):
        arr = np.zeros(self.obj.shape, dtype=self.obj.dtype)
        self.assertRaises(tb.NotChunkedError,
                          self.array.read_chunk,
                          (0,) * self.array.ndim,
                          out=memoryview(arr))

    def test_write_chunk(self):
        arr = self.obj // 2
        self.assertRaises(tb.NotChunkedError,
                          self.array.write_chunk,
                          (0,) * self.array.ndim,
                          arr)


def suite():
    theSuite = common.unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(common.make_suite(ArrayDirectChunkingTestCase))

    return theSuite


if __name__ == '__main__':
    common.parse_argv(sys.argv)
    common.print_versions()
    common.unittest.main(defaultTest='suite')
