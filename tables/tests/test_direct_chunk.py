import itertools
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


# For enlargeable and non-enlargeable datasets.
class DirectChunkingTestCase(common.TempFileMixin, common.PyTablesTestCase):
    # Class attributes:
    shape: tuple[int, ...]
    chunkshape: tuple[int, ...]
    shuffle: bool
    obj: np.ndarray

    # Instance attributes:
    array: tb.Leaf  # set by ``setUp()`` and ``_reopen()``

    def modified(self, obj):
        # Return altered copy with same dtype and shape.
        raise NotImplementedError

    def iter_chunks(self):
        chunk_ranges = list(range(0, s, cs) for (s, cs)
                            in zip(self.shape, self.chunkshape))
        yield from itertools.product(*chunk_ranges)

    def test_chunk_info_aligned(self):
        chunk_size = np.prod(self.chunkshape) * self.obj.dtype.itemsize
        for chunk_start in self.iter_chunks():
            chunk_info = self.array.chunk_info(chunk_start)
            self.assertEqual(chunk_info.start, chunk_start)
            self.assertIsNotNone(chunk_info.offset)
            self.assertEqual(chunk_info.size, chunk_size)

    def test_chunk_info_unaligned(self):
        chunk_info_a = self.array.chunk_info((0,) * self.array.ndim)
        chunk_info_u = self.array.chunk_info((1,) * self.array.ndim)
        self.assertIsNotNone(chunk_info_a.start)
        self.assertEqual(chunk_info_a, chunk_info_u)

    def test_chunk_info_aligned_beyond(self):
        beyond = tuple((1 + s // cs) * cs for (s, cs)
                       in zip(self.shape, self.chunkshape))
        self.assertRaises(tb.ChunkError,
                          self.array.chunk_info,
                          beyond)
        try:
            self.array.chunk_info(beyond)
        except tb.NotChunkAlignedError as e:
            self.fail("wrong exception in aligned chunk info "
                      "beyond max shape: %r" % e)
        except tb.ChunkError:
            pass

    def test_chunk_info_unaligned_beyond(self):
        beyond = tuple(1 + (1 + s // cs) * cs for (s, cs)
                       in zip(self.shape, self.chunkshape))
        self.assertRaises(tb.ChunkError,
                          self.array.chunk_info,
                          beyond)
        try:
            self.array.chunk_info(beyond)
        except tb.NotChunkAlignedError as e:
            self.fail("wrong exception in unaligned chunk info "
                      "beyond max shape: %r" % e)
        except tb.ChunkError:
            pass

    def maybe_shuffle(self, bytes_):
        if not self.shuffle:
            return bytes_
        itemsize = self.obj.dtype.itemsize
        return b''.join(bytes_[d::itemsize] for d in range(itemsize))

    def test_read_chunk(self):
        # Extended to fit chunk boundaries.
        ext_obj = np.pad(self.obj, [(0, s % cs) for (s, cs)
                                    in zip(self.shape, self.chunkshape)])
        for chunk_start in self.iter_chunks():
            chunk = self.array.read_chunk(chunk_start)
            self.assertIsInstance(chunk, bytes)
            obj_slice = tuple(slice(s, s + cs) for (s, cs)
                              in zip(chunk_start, self.chunkshape))
            obj_bytes = self.maybe_shuffle(ext_obj[obj_slice].tobytes())
            self.assertEqual(chunk, obj_bytes)

    def test_read_chunk_out(self):
        # Extended to fit chunk boundaries.
        ext_obj = np.pad(self.obj, [(0, s % cs) for (s, cs)
                                    in zip(self.shape, self.chunkshape)])
        chunk_start = (0,) * self.obj.ndim
        chunk_size = np.prod(self.chunkshape) * self.obj.dtype.itemsize
        chunk_out = bytearray(chunk_size)
        chunk = self.array.read_chunk(chunk_start, out=chunk_out)
        self.assertIsInstance(chunk, memoryview)
        obj_slice = tuple(slice(s, s + cs) for (s, cs)
                          in zip(chunk_start, self.chunkshape))
        obj_bytes = self.maybe_shuffle(ext_obj[obj_slice].tobytes())
        self.assertEqual(chunk, obj_bytes)
        self.assertEqual(chunk_out, obj_bytes)

        chunk_out = bytearray(chunk_size - 1)  # too short
        self.assertRaises(ValueError,
                          self.array.read_chunk, chunk_start, out=chunk_out)

    def test_read_chunk_unaligned(self):
        self.assertRaises(tb.NotChunkAlignedError,
                          self.array.read_chunk,
                          (1,) * self.array.ndim)

    def test_read_chunk_beyond(self):
        beyond = tuple((1 + s // cs) * cs for (s, cs)
                       in zip(self.shape, self.chunkshape))
        self.assertRaises(tb.ChunkError,
                          self.array.read_chunk,
                          beyond)
        try:
            self.array.read_chunk(beyond)
        except tb.NoSuchChunkError as e:
            self.fail("wrong exception in chunk read "
                      "beyond max shape: %r" % e)
        except tb.ChunkError:
            pass

    def test_write_chunk(self):
        new_obj = self.modified(self.obj)
        # Extended to fit chunk boundaries.
        ext_obj = np.pad(new_obj, [(0, s % cs) for (s, cs)
                                   in zip(self.shape, self.chunkshape)])
        for chunk_start in self.iter_chunks():
            obj_slice = tuple(slice(s, s + cs) for (s, cs)
                              in zip(chunk_start, self.chunkshape))
            obj_bytes = self.maybe_shuffle(ext_obj[obj_slice].tobytes())
            self.array.write_chunk(chunk_start, obj_bytes)

        self._reopen()
        self.assertTrue(common.areArraysEqual(self.array[:], new_obj))

    def test_write_chunk_filtermask(self):
        no_shuffle_mask = 0x00000002  # to turn shuffle off

        chunk_start = (0,) * self.obj.ndim
        obj_slice = tuple(slice(s, s + cs) for (s, cs)
                          in zip(chunk_start, self.chunkshape))
        new_obj = self.obj.copy()
        new_obj[obj_slice] = self.modified(new_obj[obj_slice])
        obj_bytes = new_obj[obj_slice].tobytes()  # do not shuffle
        self.array.write_chunk(chunk_start, obj_bytes,
                               filter_mask=no_shuffle_mask)

        self._reopen()
        self.assertTrue(common.areArraysEqual(self.array[:], new_obj))

        chunk_info = self.array.chunk_info(chunk_start)
        self.assertEqual(chunk_info.filter_mask, no_shuffle_mask)

    def test_write_chunk_unaligned(self):
        self.assertRaises(tb.NotChunkAlignedError,
                          self.array.write_chunk,
                          (1,) * self.array.ndim,
                          b'foobar')

    def test_write_chunk_beyond(self):
        beyond = tuple((1 + s // cs) * cs for (s, cs)
                       in zip(self.shape, self.chunkshape))
        self.assertRaises(tb.ChunkError,
                          self.array.write_chunk,
                          beyond,
                          b'foobar')
        try:
            self.array.write_chunk(beyond, b'foobar')
        except tb.NoSuchChunkError as e:
            self.fail("wrong exception in chunk write "
                      "beyond max shape: %r" % e)
        except tb.ChunkError:
            pass


# For enlargeable datasets only.
class XDirectChunkingTestCase(DirectChunkingTestCase):
    def test_chunk_info_miss_extdim(self):
        # Next chunk in the enlargeable dimension.
        assert self.array.extdim == 0
        chunk_start = (((1 + self.shape[0] // self.chunkshape[0])
                        * self.chunkshape[0]),
                       *((0,) * (self.array.ndim - 1)))
        chunk_info = self.array.chunk_info(chunk_start)
        self.assertIsNone(chunk_info.start)
        self.assertIsNone(chunk_info.offset)

    def test_chunk_info_miss_noextdim(self):
        if self.array.ndim < 2:
            raise common.unittest.SkipTest

        # Next chunk in the first non-enlargeable dimension.
        assert self.array.extdim != 1
        chunk_start = (0,
                       ((1 + self.shape[1] // self.chunkshape[1])
                        * self.chunkshape[1]),
                       *((0,) * (self.array.ndim - 2)))
        self.assertRaises(tb.ChunkError,
                          self.array.chunk_info,
                          chunk_start)
        try:
            self.array.chunk_info(chunk_start)
        except tb.NoSuchChunkError as e:
            self.fail("wrong exception in missing chunk info "
                      "out of enlargeable dimension: %r" % e)
        except tb.ChunkError:
            pass

    def test_read_chunk_miss_extdim(self):
        # Next chunk in the enlargeable dimension.
        assert self.array.extdim == 0
        chunk_start = (((1 + self.shape[0] // self.chunkshape[0])
                        * self.chunkshape[0]),
                       *((0,) * (self.array.ndim - 1)))
        self.assertRaises(tb.NoSuchChunkError,
                          self.array.chunk_info,
                          chunk_start)

    def _test_write_chunk_missing(self, enlarge_first, shrink_after=False):
        # Enlarge array by two chunk rows,
        # copy first old chunk in first chunk of new last chunk row.
        assert self.array.extdim == 0
        chunk_start = (((1 + self.shape[0] // self.chunkshape[0])
                        * self.chunkshape[0] + self.chunkshape[0]),
                       *((0,) * (self.array.ndim - 1)))
        chunk = self.array.read_chunk((0,) * self.array.ndim)
        if enlarge_first:
            self.array.truncate(chunk_start[0] + self.chunkshape[0])
        self.array.write_chunk(chunk_start, chunk)
        if shrink_after:
            self.array.truncate(self.shape[0] + 1)
            self.array.truncate(self.shape[0] - 1)
        if not enlarge_first:
            self.array.truncate(chunk_start[0] + self.chunkshape[0])

        new_obj = self.obj.copy()
        new_obj.resize(self.array.shape)
        obj_slice = tuple(slice(s, s + cs) for (s, cs)
                          in zip(chunk_start, self.chunkshape))
        if enlarge_first or not shrink_after:
            new_obj[obj_slice] = new_obj[tuple(slice(0, cs)
                                               for cs in self.chunkshape)]

        self._reopen()
        self.assertTrue(common.areArraysEqual(self.array[:], new_obj))

    def test_write_chunk_missing0(self):
        return self._test_write_chunk_missing(enlarge_first=True)

    def test_write_chunk_missing1(self):
        return self._test_write_chunk_missing(enlarge_first=False)

    def test_write_chunk_missing2(self):
        return self._test_write_chunk_missing(enlarge_first=False,
                                              shrink_after=True)


class CArrayDirectChunkingTestCase(DirectChunkingTestCase):
    shape = (5, 5)
    chunkshape = (2, 2)  # 3 x 3 chunks, incomplete at right/bottom boundaries
    shuffle = True
    obj = np.arange(np.prod(shape), dtype='u2').reshape(shape)

    def setUp(self):
        super().setUp()
        self.array = self.h5file.create_carray(
            '/', 'carray', chunkshape=self.chunkshape, obj=self.obj,
            filters=tb.Filters(shuffle=self.shuffle))

    def _reopen(self):
        super()._reopen()
        self.array = self.h5file.root.carray

    def modified(self, obj):
        return obj * 2


class EArrayDirectChunkingTestCase(XDirectChunkingTestCase):
    shape = (5, 5)  # enlargeable along first dimension
    chunkshape = (2, 2)  # 3 x 3 chunks, incomplete at right/bottom boundaries
    shuffle = True
    obj = np.arange(np.prod(shape), dtype='u2').reshape(shape)

    def setUp(self):
        super().setUp()
        atom = tb.Atom.from_dtype(self.obj.dtype)
        shape = (0, *self.shape[1:])
        self.array = self.h5file.create_earray(
            '/', 'earray', atom, shape, chunkshape=self.chunkshape,
            filters=tb.Filters(shuffle=self.shuffle))
        self.array.append(self.obj)

    def _reopen(self):
        super()._reopen()
        self.array = self.h5file.root.earray

    def modified(self, obj):
        return obj * 2


class TableDirectChunkingTestCase(XDirectChunkingTestCase):
    shape = (5,)  # enlargeable along first dimension
    chunkshape = (2,)  # 3 chunks, incomplete at bottom boundary
    shuffle = True
    obj = np.array([(i, float(i)) for i in range(np.prod(shape))],
                   dtype='u4,f4')

    def setUp(self):
        super().setUp()
        desc, _ = tb.descr_from_dtype(self.obj.dtype)
        self.array = self.h5file.create_table(
            '/', 'table', desc, chunkshape=self.chunkshape,
            filters=tb.Filters(shuffle=self.shuffle))
        self.array.append(self.obj)

    def _reopen(self):
        super()._reopen()
        self.array = self.h5file.root.table

    def modified(self, obj):
        flat = obj.copy().reshape((np.prod(obj.shape),))
        fnames = flat.dtype.names
        for i in range(len(flat)):
            for f in fnames:
                flat[i][f] *= 2
        return flat.reshape(obj.shape)


def suite():
    theSuite = common.unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(common.make_suite(ArrayDirectChunkingTestCase))
        theSuite.addTest(common.make_suite(CArrayDirectChunkingTestCase))
        theSuite.addTest(common.make_suite(EArrayDirectChunkingTestCase))
        theSuite.addTest(common.make_suite(TableDirectChunkingTestCase))

    return theSuite


if __name__ == '__main__':
    common.parse_argv(sys.argv)
    common.print_versions()
    common.unittest.main(defaultTest='suite')
