# -*- coding: utf-8 -*-


import os
import sys
import tempfile
import warnings

import numpy
from numpy import testing as npt


import tables
from tables import Atom, ClosedNodeError, NoSuchNodeError
from tables.utils import byteorders
from tables.tests import common
from tables.tests.common import allequal
from tables.tests.common import unittest, test_filename
from tables.tests.common import PyTablesTestCase as TestCase


#warnings.resetwarnings()


class BasicTestCase(TestCase):
    """Basic test for all the supported typecodes present in numpy.

    All of them are included on pytables.

    """
    endiancheck = False

    def write_read(self, testarray):
        a = testarray
        if common.verbose:
            print('\n', '-=' * 30)
            print("Running test for array with type '%s'" % a.dtype.type,
                  end=' ')
            print("for class check:", self.title)

        # Create an instance of HDF5 file
        filename = tempfile.mktemp(".h5")
        try:
            with tables.open_file(filename, mode="w") as fileh:
                root = fileh.root

                # Create the array under root and name 'somearray'
                if self.endiancheck and a.dtype.kind != "S":
                    b = a.byteswap()
                    b.dtype = a.dtype.newbyteorder()
                    a = b

                fileh.create_array(root, 'somearray', a, "Some array")

            # Re-open the file in read-only mode
            with tables.open_file(filename, mode="r") as fileh:
                root = fileh.root

                # Read the saved array
                b = root.somearray.read()

                # Compare them. They should be equal.
                if common.verbose and not allequal(a, b):
                    print("Write and read arrays differ!")
                    # print("Array written:", a)
                    print("Array written shape:", a.shape)
                    print("Array written itemsize:", a.itemsize)
                    print("Array written type:", a.dtype.type)
                    # print("Array read:", b)
                    print("Array read shape:", b.shape)
                    print("Array read itemsize:", b.itemsize)
                    print("Array read type:", b.dtype.type)
                    if a.dtype.kind != "S":
                        print("Array written byteorder:", a.dtype.byteorder)
                        print("Array read byteorder:", b.dtype.byteorder)

                # Check strictly the array equality
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.shape, root.somearray.shape)
                if a.dtype.kind == "S":
                    self.assertEqual(root.somearray.atom.type, "string")
                else:
                    self.assertEqual(a.dtype.type, b.dtype.type)
                    self.assertEqual(a.dtype.type,
                                     root.somearray.atom.dtype.type)
                    abo = byteorders[a.dtype.byteorder]
                    bbo = byteorders[b.dtype.byteorder]
                    if abo != "irrelevant":
                        self.assertEqual(abo, root.somearray.byteorder)
                        self.assertEqual(bbo, sys.byteorder)
                        if self.endiancheck:
                            self.assertNotEqual(bbo, abo)

                obj = root.somearray
                self.assertEqual(obj.flavor, 'numpy')
                self.assertEqual(obj.shape, a.shape)
                self.assertEqual(obj.ndim, a.ndim)
                self.assertEqual(obj.chunkshape, None)
                if a.shape:
                    nrows = a.shape[0]
                else:
                    # scalar
                    nrows = 1

                self.assertEqual(obj.nrows, nrows)

                self.assertTrue(allequal(a, b))
        finally:
            # Then, delete the file
            os.remove(filename)

    def write_read_out_arg(self, testarray):
        a = testarray

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running test for array with type '%s'" % a.dtype.type,
                  end=' ')
            print("for class check:", self.title)

        # Create an instance of HDF5 file
        filename = tempfile.mktemp(".h5")
        try:
            with tables.open_file(filename, mode="w") as fileh:
                root = fileh.root

                # Create the array under root and name 'somearray'
                if self.endiancheck and a.dtype.kind != "S":
                    b = a.byteswap()
                    b.dtype = a.dtype.newbyteorder()
                    a = b

                fileh.create_array(root, 'somearray', a, "Some array")

            # Re-open the file in read-only mode
            with tables.open_file(filename, mode="r") as fileh:
                root = fileh.root

                # Read the saved array
                b = numpy.empty_like(a, dtype=a.dtype)
                root.somearray.read(out=b)

                # Check strictly the array equality
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.shape, root.somearray.shape)
                if a.dtype.kind == "S":
                    self.assertEqual(root.somearray.atom.type, "string")
                else:
                    self.assertEqual(a.dtype.type, b.dtype.type)
                    self.assertEqual(a.dtype.type,
                                     root.somearray.atom.dtype.type)
                    abo = byteorders[a.dtype.byteorder]
                    bbo = byteorders[b.dtype.byteorder]
                    if abo != "irrelevant":
                        self.assertEqual(abo, root.somearray.byteorder)
                        self.assertEqual(abo, bbo)
                        if self.endiancheck:
                            self.assertNotEqual(bbo, sys.byteorder)

                self.assertTrue(allequal(a, b))
        finally:
            # Then, delete the file
            os.remove(filename)

    def write_read_atom_shape_args(self, testarray):
        a = testarray
        atom = Atom.from_dtype(a.dtype)
        shape = a.shape
        byteorder = None

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running test for array with type '%s'" % a.dtype.type,
                  end=' ')
            print("for class check:", self.title)

        # Create an instance of HDF5 file
        filename = tempfile.mktemp(".h5")
        try:
            with tables.open_file(filename, mode="w") as fileh:
                root = fileh.root

                # Create the array under root and name 'somearray'
                if self.endiancheck and a.dtype.kind != "S":
                    b = a.byteswap()
                    b.dtype = a.dtype.newbyteorder()
                    if b.dtype.byteorder in ('>', '<'):
                        byteorder = byteorders[b.dtype.byteorder]
                    a = b

                ptarr = fileh.create_array(root, 'somearray',
                                           atom=atom, shape=shape,
                                           title="Some array",
                                           # specify the byteorder explicitly
                                           # since there is no way to deduce
                                           # it in this case
                                           byteorder=byteorder)
                self.assertEqual(shape, ptarr.shape)
                self.assertEqual(atom, ptarr.atom)
                ptarr[...] = a

            # Re-open the file in read-only mode
            with tables.open_file(filename, mode="r") as fileh:
                root = fileh.root

                # Read the saved array
                b = root.somearray.read()

                # Compare them. They should be equal.
                if common.verbose and not allequal(a, b):
                    print("Write and read arrays differ!")
                    # print("Array written:", a)
                    print("Array written shape:", a.shape)
                    print("Array written itemsize:", a.itemsize)
                    print("Array written type:", a.dtype.type)
                    # print("Array read:", b)
                    print("Array read shape:", b.shape)
                    print("Array read itemsize:", b.itemsize)
                    print("Array read type:", b.dtype.type)
                    if a.dtype.kind != "S":
                        print("Array written byteorder:", a.dtype.byteorder)
                        print("Array read byteorder:", b.dtype.byteorder)

                # Check strictly the array equality
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.shape, root.somearray.shape)
                if a.dtype.kind == "S":
                    self.assertEqual(root.somearray.atom.type, "string")
                else:
                    self.assertEqual(a.dtype.type, b.dtype.type)
                    self.assertEqual(a.dtype.type,
                                     root.somearray.atom.dtype.type)
                    abo = byteorders[a.dtype.byteorder]
                    bbo = byteorders[b.dtype.byteorder]
                    if abo != "irrelevant":
                        self.assertEqual(abo, root.somearray.byteorder)
                        self.assertEqual(bbo, sys.byteorder)
                        if self.endiancheck:
                            self.assertNotEqual(bbo, abo)

                obj = root.somearray
                self.assertEqual(obj.flavor, 'numpy')
                self.assertEqual(obj.shape, a.shape)
                self.assertEqual(obj.ndim, a.ndim)
                self.assertEqual(obj.chunkshape, None)
                if a.shape:
                    nrows = a.shape[0]
                else:
                    # scalar
                    nrows = 1

                self.assertEqual(obj.nrows, nrows)

                self.assertTrue(allequal(a, b))
        finally:
            # Then, delete the file
            os.remove(filename)

    def setup00_char(self):
        """Data integrity during recovery (character objects)"""

        if not isinstance(self.tupleChar, numpy.ndarray):
            a = numpy.array(self.tupleChar, dtype="S")
        else:
            a = self.tupleChar

        return a

    def test00_char(self):
        a = self.setup00_char()
        self.write_read(a)

    def test00_char_out_arg(self):
        a = self.setup00_char()
        self.write_read_out_arg(a)

    def test00_char_atom_shape_args(self):
        a = self.setup00_char()
        self.write_read_atom_shape_args(a)

    def test00b_char(self):
        """Data integrity during recovery (string objects)"""

        a = self.tupleChar

        filename = tempfile.mktemp(".h5")
        try:
            # Create an instance of HDF5 file
            with tables.open_file(filename, mode="w") as fileh:
                fileh.create_array(fileh.root, 'somearray', a, "Some array")

            # Re-open the file in read-only mode
            with tables.open_file(filename, mode="r") as fileh:
                # Read the saved array
                b = fileh.root.somearray.read()
                if isinstance(a, bytes):
                    self.assertEqual(type(b), bytes)
                    self.assertEqual(a, b)
                else:
                    # If a is not a python string, then it should be a list
                    # or ndarray
                    self.assertIn(type(b), [list, numpy.ndarray])
        finally:
            # Then, delete the file
            os.remove(filename)

    def test00b_char_out_arg(self):
        """Data integrity during recovery (string objects)"""

        a = self.tupleChar

        filename = tempfile.mktemp(".h5")
        try:
            # Create an instance of HDF5 file
            with tables.open_file(filename, mode="w") as fileh:
                fileh.create_array(fileh.root, 'somearray', a, "Some array")

            # Re-open the file in read-only mode
            with tables.open_file(filename, mode="r") as fileh:
                # Read the saved array
                b = numpy.empty_like(a)
                if fileh.root.somearray.flavor != 'numpy':
                    self.assertRaises(TypeError,
                                      lambda: fileh.root.somearray.read(out=b))
                else:
                    fileh.root.somearray.read(out=b)
                self.assertIsInstance(b, numpy.ndarray)
        finally:
            # Then, delete the file
            os.remove(filename)

    def test00b_char_atom_shape_args(self):
        """Data integrity during recovery (string objects)"""

        a = self.tupleChar

        filename = tempfile.mktemp(".h5")
        try:
            # Create an instance of HDF5 file
            with tables.open_file(filename, mode="w") as fileh:
                nparr = numpy.asarray(a)
                atom = Atom.from_dtype(nparr.dtype)
                shape = nparr.shape
                if nparr.dtype.byteorder in ('>', '<'):
                    byteorder = byteorders[nparr.dtype.byteorder]
                else:
                    byteorder = None

                ptarr = fileh.create_array(fileh.root, 'somearray',
                                           atom=atom, shape=shape,
                                           byteorder=byteorder,
                                           title="Some array")
                self.assertEqual(shape, ptarr.shape)
                self.assertEqual(atom, ptarr.atom)
                ptarr[...] = a

            # Re-open the file in read-only mode
            with tables.open_file(filename, mode="r") as fileh:
                # Read the saved array
                b = numpy.empty_like(a)
                if fileh.root.somearray.flavor != 'numpy':
                    self.assertRaises(TypeError,
                                      lambda: fileh.root.somearray.read(out=b))
                else:
                    fileh.root.somearray.read(out=b)
                self.assertIsInstance(b, numpy.ndarray)
        finally:
            # Then, delete the file
            os.remove(filename)

    def setup01_char_nc(self):
        """Data integrity during recovery (non-contiguous character objects)"""

        if not isinstance(self.tupleChar, numpy.ndarray):
            a = numpy.array(self.tupleChar, dtype="S")
        else:
            a = self.tupleChar
        if a.ndim == 0:
            b = a.copy()
        else:
            b = a[::2]
            # Ensure that this numpy string is non-contiguous
            if len(b) > 1:
                self.assertEqual(b.flags.contiguous, False)
        return b

    def test01_char_nc(self):
        b = self.setup01_char_nc()
        self.write_read(b)

    def test01_char_nc_out_arg(self):
        b = self.setup01_char_nc()
        self.write_read_out_arg(b)

    def test01_char_nc_atom_shape_args(self):
        b = self.setup01_char_nc()
        self.write_read_atom_shape_args(b)

    def test02_types(self):
        """Data integrity during recovery (numerical types)"""

        typecodes = ['int8', 'int16', 'int32', 'int64',
                     'uint8', 'uint16', 'uint32', 'uint64',
                     'float32', 'float64',
                     'complex64', 'complex128']

        for name in ('float16', 'float96', 'float128',
                     'complex192', 'complex256'):
            atomname = name.capitalize() + 'Atom'
            if hasattr(tables, atomname):
                typecodes.append(name)

        for typecode in typecodes:
            a = numpy.array(self.tupleInt, typecode)
            self.write_read(a)
            b = numpy.array(self.tupleInt, typecode)
            self.write_read_out_arg(b)
            c = numpy.array(self.tupleInt, typecode)
            self.write_read_atom_shape_args(c)

    def test03_types_nc(self):
        """Data integrity during recovery (non-contiguous numerical types)"""

        typecodes = ['int8', 'int16', 'int32', 'int64',
                     'uint8', 'uint16', 'uint32', 'uint64',
                     'float32', 'float64',
                     'complex64', 'complex128', ]

        for name in ('float16', 'float96', 'float128',
                     'complex192', 'complex256'):
            atomname = name.capitalize() + 'Atom'
            if hasattr(tables, atomname):
                typecodes.append(name)

        for typecode in typecodes:
            a = numpy.array(self.tupleInt, typecode)
            if a.ndim == 0:
                b1 = a.copy()
                b2 = a.copy()
                b3 = a.copy()
            else:
                b1 = a[::2]
                b2 = a[::2]
                b3 = a[::2]
                # Ensure that this array is non-contiguous
                if len(b1) > 1:
                    self.assertEqual(b1.flags.contiguous, False)
                if len(b2) > 1:
                    self.assertEqual(b2.flags.contiguous, False)
                if len(b3) > 1:
                    self.assertEqual(b3.flags.contiguous, False)
            self.write_read(b1)
            self.write_read_out_arg(b2)
            self.write_read_atom_shape_args(b3)


class Basic0DOneTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 1"
    tupleInt = 3
    tupleChar = b"3"
    endiancheck = True


class Basic0DTwoTestCase(BasicTestCase):
    # Scalar case
    title = "Rank-0 case 2"
    tupleInt = 33
    tupleChar = b"33"
    endiancheck = True


class Basic1DZeroTestCase(BasicTestCase):
    # This test case is not supported by PyTables (HDF5 limitations)
    # 1D case
    title = "Rank-1 case 0"
    tupleInt = ()
    tupleChar = ()
    endiancheck = False


class Basic1DOneTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 1"
    tupleInt = (3,)
    tupleChar = (b"a",)
    endiancheck = True


class Basic1DTwoTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 2"
    tupleInt = (3, 4)
    tupleChar = (b"aaa",)
    endiancheck = True


class Basic1DThreeTestCase(BasicTestCase):
    # 1D case
    title = "Rank-1 case 3"
    tupleInt = (3, 4, 5)
    tupleChar = (b"aaa", b"bbb",)
    endiancheck = True


class Basic2DOneTestCase(BasicTestCase):
    # 2D case
    title = "Rank-2 case 1"
    tupleInt = numpy.array(numpy.arange((4)**2))
    tupleInt.shape = (4,)*2
    tupleChar = numpy.array(["abc"]*3**2, dtype="S3")
    tupleChar.shape = (3,)*2
    endiancheck = True


class Basic2DTwoTestCase(BasicTestCase):
    # 2D case, with a multidimensional dtype
    title = "Rank-2 case 2"
    tupleInt = numpy.array(numpy.arange((4)), dtype=(numpy.int_, (4,)))
    tupleChar = numpy.array(["abc"]*3, dtype=("S3", (3,)))
    endiancheck = True


class Basic10DTestCase(BasicTestCase):
    # 10D case
    title = "Rank-10 test"
    tupleInt = numpy.array(numpy.arange((2)**10))
    tupleInt.shape = (2,)*10
    tupleChar = numpy.array(
        ["abc"]*2**10, dtype="S3")
    tupleChar.shape = (2,)*10
    endiancheck = True


class Basic32DTestCase(BasicTestCase):
    # 32D case (maximum)
    title = "Rank-32 test"
    tupleInt = numpy.array((32,))
    tupleInt.shape = (1,)*32
    tupleChar = numpy.array(["121"], dtype="S3")
    tupleChar.shape = (1,)*32


class ReadOutArgumentTests(common.TempFileMixin, TestCase):

    def setUp(self):
        super(ReadOutArgumentTests, self).setUp()
        self.size = 1000

    def create_array(self):
        array = numpy.arange(self.size, dtype='f8')
        disk_array = self.h5file.create_array('/', 'array', array)
        return array, disk_array

    def test_read_entire_array(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.empty((self.size, ), 'f8')
        disk_array.read(out=out_buffer)
        numpy.testing.assert_equal(out_buffer, array)

    def test_read_contiguous_slice1(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.arange(self.size, dtype='f8')
        out_buffer = numpy.random.permutation(out_buffer)
        out_buffer_orig = out_buffer.copy()
        start = self.size // 2
        disk_array.read(start=start, stop=self.size, out=out_buffer[start:])
        numpy.testing.assert_equal(out_buffer[start:], array[start:])
        numpy.testing.assert_equal(out_buffer[:start], out_buffer_orig[:start])

    def test_read_contiguous_slice2(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.arange(self.size, dtype='f8')
        out_buffer = numpy.random.permutation(out_buffer)
        out_buffer_orig = out_buffer.copy()
        start = self.size // 4
        stop = self.size - start
        disk_array.read(start=start, stop=stop, out=out_buffer[start:stop])
        numpy.testing.assert_equal(out_buffer[start:stop], array[start:stop])
        numpy.testing.assert_equal(out_buffer[:start], out_buffer_orig[:start])
        numpy.testing.assert_equal(out_buffer[stop:], out_buffer_orig[stop:])

    def test_read_non_contiguous_slice_contiguous_buffer(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.empty((self.size // 2, ), dtype='f8')
        disk_array.read(start=0, stop=self.size, step=2, out=out_buffer)
        numpy.testing.assert_equal(out_buffer, array[0:self.size:2])

    def test_read_non_contiguous_buffer(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.empty((self.size, ), 'f8')
        out_buffer_slice = out_buffer[0:self.size:2]
        # once Python 2.6 support is dropped, this could change
        # to assertRaisesRegexp to check exception type and message at once
        self.assertRaises(ValueError, disk_array.read, 0, self.size, 2,
                          out_buffer_slice)
        try:
            disk_array.read(0, self.size, 2, out_buffer_slice)
        except ValueError as exc:
            self.assertEqual('output array not C contiguous', str(exc))

    def test_buffer_too_small(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.empty((self.size // 2, ), 'f8')
        self.assertRaises(ValueError, disk_array.read, 0, self.size, 1,
                          out_buffer)
        try:
            disk_array.read(0, self.size, 1, out_buffer)
        except ValueError as exc:
            self.assertIn('output array size invalid, got', str(exc))

    def test_buffer_too_large(self):
        array, disk_array = self.create_array()
        out_buffer = numpy.empty((self.size + 1, ), 'f8')
        self.assertRaises(ValueError, disk_array.read, 0, self.size, 1,
                          out_buffer)
        try:
            disk_array.read(0, self.size, 1, out_buffer)
        except ValueError as exc:
            self.assertIn('output array size invalid, got', str(exc))


class SizeOnDiskInMemoryPropertyTestCase(common.TempFileMixin, TestCase):

    def setUp(self):
        super(SizeOnDiskInMemoryPropertyTestCase, self).setUp()
        self.array_size = (10, 10)
        self.array = self.h5file.create_array(
            '/', 'somearray', numpy.zeros(self.array_size, 'i4'))

    def test_all_zeros(self):
        self.assertEqual(self.array.size_on_disk, 10 * 10 * 4)
        self.assertEqual(self.array.size_in_memory, 10 * 10 * 4)


class UnalignedAndComplexTestCase(common.TempFileMixin, TestCase):
    """Basic test for all the supported typecodes present in numpy.

    Most of them are included on PyTables.

    """

    def setUp(self):
        super(UnalignedAndComplexTestCase, self).setUp()
        self.root = self.h5file.root

    def write_read(self, testArray):
        if common.verbose:
            print('\n', '-=' * 30)
            print("\nRunning test for array with type '%s'" %
                  testArray.dtype.type)

        # Create the array under root and name 'somearray'
        a = testArray
        if self.endiancheck:
            byteorder = {"little": "big", "big": "little"}[sys.byteorder]
        else:
            byteorder = sys.byteorder

        self.h5file.create_array(self.root, 'somearray', a, "Some array",
                                 byteorder=byteorder)

        if self.reopen:
            self._reopen()
            self.root = self.h5file.root

        # Read the saved array
        b = self.root.somearray.read()

        # Get an array to be compared in the correct byteorder
        c = a.newbyteorder(byteorder)

        # Compare them. They should be equal.
        if not allequal(c, b) and common.verbose:
            print("Write and read arrays differ!")
            print("Array written:", a)
            print("Array written shape:", a.shape)
            print("Array written itemsize:", a.itemsize)
            print("Array written type:", a.dtype.type)
            print("Array read:", b)
            print("Array read shape:", b.shape)
            print("Array read itemsize:", b.itemsize)
            print("Array read type:", b.dtype.type)

        # Check strictly the array equality
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape, self.root.somearray.shape)
        if a.dtype.byteorder != "|":
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.dtype, self.root.somearray.atom.dtype)
            self.assertEqual(byteorders[b.dtype.byteorder], sys.byteorder)
            self.assertEqual(self.root.somearray.byteorder, byteorder)

        self.assertTrue(allequal(c, b))

    def test01_signedShort_unaligned(self):
        """Checking an unaligned signed short integer array"""

        r = numpy.rec.array(b'a'*200, formats='i1,f4,i2', shape=10)
        a = r["f2"]
        # Ensure that this array is non-aligned
        self.assertEqual(a.flags.aligned, False)
        self.assertEqual(a.dtype.type, numpy.int16)
        self.write_read(a)

    def test02_float_unaligned(self):
        """Checking an unaligned single precision array"""

        r = numpy.rec.array(b'a'*200, formats='i1,f4,i2', shape=10)
        a = r["f1"]
        # Ensure that this array is non-aligned
        self.assertEqual(a.flags.aligned, 0)
        self.assertEqual(a.dtype.type, numpy.float32)
        self.write_read(a)

    def test03_byte_offset(self):
        """Checking an offsetted byte array"""

        r = numpy.arange(100, dtype=numpy.int8)
        r.shape = (10, 10)
        a = r[2]
        self.write_read(a)

    def test04_short_offset(self):
        """Checking an offsetted unsigned short int precision array"""

        r = numpy.arange(100, dtype=numpy.uint32)
        r.shape = (10, 10)
        a = r[2]
        self.write_read(a)

    def test05_int_offset(self):
        """Checking an offsetted integer array"""

        r = numpy.arange(100, dtype=numpy.int32)
        r.shape = (10, 10)
        a = r[2]
        self.write_read(a)

    def test06_longlongint_offset(self):
        """Checking an offsetted long long integer array"""

        r = numpy.arange(100, dtype=numpy.int64)
        r.shape = (10, 10)
        a = r[2]
        self.write_read(a)

    def test07_float_offset(self):
        """Checking an offsetted single precision array"""

        r = numpy.arange(100, dtype=numpy.float32)
        r.shape = (10, 10)
        a = r[2]
        self.write_read(a)

    def test08_double_offset(self):
        """Checking an offsetted double precision array"""

        r = numpy.arange(100, dtype=numpy.float64)
        r.shape = (10, 10)
        a = r[2]
        self.write_read(a)

    def test09_float_offset_unaligned(self):
        """Checking an unaligned and offsetted single precision array"""

        r = numpy.rec.array(b'a'*200, formats='i1,3f4,i2', shape=10)
        a = r["f1"][3]
        # Ensure that this array is non-aligned
        self.assertEqual(a.flags.aligned, False)
        self.assertEqual(a.dtype.type, numpy.float32)
        self.write_read(a)

    def test10_double_offset_unaligned(self):
        """Checking an unaligned and offsetted double precision array"""

        r = numpy.rec.array(b'a'*400, formats='i1,3f8,i2', shape=10)
        a = r["f1"][3]
        # Ensure that this array is non-aligned
        self.assertEqual(a.flags.aligned, False)
        self.assertEqual(a.dtype.type, numpy.float64)
        self.write_read(a)

    def test11_int_byteorder(self):
        """Checking setting data with different byteorder in a range
         (integer)"""

        # Save an array with the reversed byteorder on it
        a = numpy.arange(25, dtype=numpy.int32).reshape(5, 5)
        a = a.byteswap()
        a = a.newbyteorder()
        array = self.h5file.create_array(
            self.h5file.root, 'array', a, "byteorder (int)")
        # Read a subarray (got an array with the machine byteorder)
        b = array[2:4, 3:5]
        b = b.byteswap()
        b = b.newbyteorder()
        # Set this subarray back to the array
        array[2:4, 3:5] = b
        b = b.byteswap()
        b = b.newbyteorder()
        # Set this subarray back to the array
        array[2:4, 3:5] = b
        # Check that the array is back in the correct byteorder
        c = array[...]
        if common.verbose:
            print("byteorder of array on disk-->", array.byteorder)
            print("byteorder of subarray-->", b.dtype.byteorder)
            print("subarray-->", b)
            print("retrieved array-->", c)
        self.assertTrue(allequal(a, c))

    def test12_float_byteorder(self):
        """Checking setting data with different byteorder in a range (float)"""

        # Save an array with the reversed byteorder on it
        a = numpy.arange(25, dtype=numpy.float64).reshape(5, 5)
        a = a.byteswap()
        a = a.newbyteorder()
        array = self.h5file.create_array(
            self.h5file.root, 'array', a, "byteorder (float)")
        # Read a subarray (got an array with the machine byteorder)
        b = array[2:4, 3:5]
        b = b.byteswap()
        b = b.newbyteorder()
        # Set this subarray back to the array
        array[2:4, 3:5] = b
        b = b.byteswap()
        b = b.newbyteorder()
        # Set this subarray back to the array
        array[2:4, 3:5] = b
        # Check that the array is back in the correct byteorder
        c = array[...]
        if common.verbose:
            print("byteorder of array on disk-->", array.byteorder)
            print("byteorder of subarray-->", b.dtype.byteorder)
            print("subarray-->", b)
            print("retrieved array-->", c)
        self.assertTrue(allequal(a, c))


class ComplexNotReopenNotEndianTestCase(UnalignedAndComplexTestCase):
    endiancheck = False
    reopen = False


class ComplexReopenNotEndianTestCase(UnalignedAndComplexTestCase):
    endiancheck = False
    reopen = True


class ComplexNotReopenEndianTestCase(UnalignedAndComplexTestCase):
    endiancheck = True
    reopen = False


class ComplexReopenEndianTestCase(UnalignedAndComplexTestCase):
    endiancheck = True
    reopen = True


class GroupsArrayTestCase(common.TempFileMixin, TestCase):
    """This test class checks combinations of arrays with groups."""

    def test00_iterativeGroups(self):
        """Checking combinations of arrays with groups."""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test00_iterativeGroups..." %
                  self.__class__.__name__)

        # Get the root group
        group = self.h5file.root

        # Set the type codes to test
        # The typecodes below does expose an ambiguity that is reported in:
        # http://projects.scipy.org/scipy/numpy/ticket/283 and
        # http://projects.scipy.org/scipy/numpy/ticket/290
        typecodes = ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'f', 'd',
                     'F', 'D']
        if hasattr(tables, 'Float16Atom'):
            typecodes.append('e')
        if hasattr(tables, 'Float96Atom') or hasattr(tables, 'Float128Atom'):
            typecodes.append('g')
        if (hasattr(tables, 'Complex192Atom') or
                hasattr(tables, 'Complex256Atom')):
            typecodes.append('G')

        for i, typecode in enumerate(typecodes):
            a = numpy.ones((3,), typecode)
            dsetname = 'array_' + typecode
            if common.verbose:
                print("Creating dataset:", group._g_join(dsetname))
            self.h5file.create_array(group, dsetname, a, "Large array")
            group = self.h5file.create_group(group, 'group' + str(i))

        # Reopen the file
        self._reopen()

        # Get the root group
        group = self.h5file.root

        # Get the metadata on the previosly saved arrays
        for i in range(len(typecodes)):
            # Create an array for later comparison
            a = numpy.ones((3,), typecodes[i])
            # Get the dset object hanging from group
            dset = getattr(group, 'array_' + typecodes[i])
            # Get the actual array
            b = dset.read()
            if common.verbose:
                print("Info from dataset:", dset._v_pathname)
                print("  shape ==>", dset.shape, end=' ')
                print("  type ==> %s" % dset.atom.dtype)
                print("Array b read from file. Shape: ==>", b.shape, end=' ')
                print(". Type ==> %s" % b.dtype)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(allequal(a, b))

            # Iterate over the next group
            group = getattr(group, 'group' + str(i))

    def test01_largeRankArrays(self):
        """Checking creation of large rank arrays (0 < rank <= 32)
        It also uses arrays ranks which ranges until maxrank.
        """

        # maximum level of recursivity (deepest group level) achieved:
        # maxrank = 32 (for a effective maximum rank of 32)
        # This limit is due to HDF5 library limitations.
        minrank = 1
        maxrank = 32

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test01_largeRankArrays..." %
                  self.__class__.__name__)
            print("Maximum rank for tested arrays:", maxrank)

        group = self.h5file.root
        if common.verbose:
            print("Rank array writing progress: ", end=' ')
        for rank in range(minrank, maxrank + 1):
            # Create an array of integers, with incrementally bigger ranges
            a = numpy.ones((1,) * rank, numpy.int32)
            if common.verbose:
                print("%3d," % (rank), end=' ')
            self.h5file.create_array(group, "array", a, "Rank: %s" % rank)
            group = self.h5file.create_group(group, 'group' + str(rank))

        # Reopen the file
        self._reopen()

        group = self.h5file.root
        if common.verbose:
            print()
            print("Rank array reading progress: ")
        # Get the metadata on the previosly saved arrays
        for rank in range(minrank, maxrank + 1):
            # Create an array for later comparison
            a = numpy.ones((1,) * rank, numpy.int32)
            # Get the actual array
            b = group.array.read()
            if common.verbose:
                print("%3d," % (rank), end=' ')
            if common.verbose and not allequal(a, b):
                print("Info from dataset:", group.array._v_pathname)
                print("  Shape: ==>", group.array.shape, end=' ')
                print("  typecode ==> %c" % group.array.typecode)
                print("Array b read from file. Shape: ==>", b.shape, end=' ')
                print(". Type ==> %c" % b.dtype)

            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(allequal(a, b))

            # print(self.h5file)
            # Iterate over the next group
            group = self.h5file.get_node(group, 'group' + str(rank))

        if common.verbose:
            print()  # This flush the stdout buffer


class CopyTestCase(common.TempFileMixin, TestCase):

    def test01_copy(self):
        """Checking Array.copy() method."""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test01_copy..." % self.__class__.__name__)

        # Create an Array
        arr = numpy.array([[456, 2], [3, 457]], dtype='int16')
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', arr, "title array1")

        # Copy to another Array
        array2 = array1.copy('/', 'array2')

        if self.close:
            if common.verbose:
                print("(closing file version)")
            self._reopen()
            array1 = self.h5file.root.array1
            array2 = self.h5file.root.array2

        if common.verbose:
            print("array1-->", array1.read())
            print("array2-->", array2.read())
            # print("dirs-->", dir(array1), dir(array2))
            print("attrs array1-->", repr(array1.attrs))
            print("attrs array2-->", repr(array2.attrs))

        # Check that all the elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.title, array2.title)

    def test02_copy(self):
        """Checking Array.copy() method (where specified)"""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test02_copy..." % self.__class__.__name__)

        # Create an Array
        arr = numpy.array([[456, 2], [3, 457]], dtype='int16')
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', arr, "title array1")

        # Copy to another Array
        group1 = self.h5file.create_group("/", "group1")
        array2 = array1.copy(group1, 'array2')

        if self.close:
            if common.verbose:
                print("(closing file version)")
            self._reopen()
            array1 = self.h5file.root.array1
            array2 = self.h5file.root.group1.array2

        if common.verbose:
            print("array1-->", array1.read())
            print("array2-->", array2.read())
            # print("dirs-->", dir(array1), dir(array2))
            print("attrs array1-->", repr(array1.attrs))
            print("attrs array2-->", repr(array2.attrs))

        # Check that all the elements are equal
        self.assertTrue(allequal(array1.read(), array2.read()))

        # Assert other properties in array
        self.assertEqual(array1.nrows, array2.nrows)
        self.assertEqual(array1.flavor, array2.flavor)
        self.assertEqual(array1.atom.dtype, array2.atom.dtype)
        self.assertEqual(array1.title, array2.title)

    def test03_copy(self):
        """Checking Array.copy() method (checking title copying)"""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test04_copy..." % self.__class__.__name__)

        # Create an Array
        arr = numpy.array([[456, 2], [3, 457]], dtype='int16')
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', arr, "title array1")
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2
        # Copy it to another Array
        array2 = array1.copy('/', 'array2', title="title array2")

        if self.close:
            if common.verbose:
                print("(closing file version)")
            self._reopen()
            array1 = self.h5file.root.array1
            array2 = self.h5file.root.array2

        # Assert user attributes
        if common.verbose:
            print("title of destination array-->", array2.title)
        self.assertEqual(array2.title, "title array2")

    def test04_copy(self):
        """Checking Array.copy() method (user attributes copied)"""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test05_copy..." % self.__class__.__name__)

        # Create an Array
        arr = numpy.array([[456, 2], [3, 457]], dtype='int16')
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', arr, "title array1")
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2
        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=1)

        if self.close:
            if common.verbose:
                print("(closing file version)")
            self._reopen()
            array1 = self.h5file.root.array1
            array2 = self.h5file.root.array2

        if common.verbose:
            print("attrs array1-->", repr(array1.attrs))
            print("attrs array2-->", repr(array2.attrs))

        # Assert user attributes
        self.assertEqual(array2.attrs.attr1, "attr1")
        self.assertEqual(array2.attrs.attr2, 2)

    def test04b_copy(self):
        """Checking Array.copy() method (user attributes not copied)"""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test05b_copy..." % self.__class__.__name__)

        # Create an Array
        arr = numpy.array([[456, 2], [3, 457]], dtype='int16')
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', arr, "title array1")
        # Append some user attrs
        array1.attrs.attr1 = "attr1"
        array1.attrs.attr2 = 2
        # Copy it to another Array
        array2 = array1.copy('/', 'array2', copyuserattrs=0)

        if self.close:
            if common.verbose:
                print("(closing file version)")
            self._reopen()
            array1 = self.h5file.root.array1
            array2 = self.h5file.root.array2

        if common.verbose:
            print("attrs array1-->", repr(array1.attrs))
            print("attrs array2-->", repr(array2.attrs))

        # Assert user attributes
        self.assertEqual(hasattr(array2.attrs, "attr1"), 0)
        self.assertEqual(hasattr(array2.attrs, "attr2"), 0)


class CloseCopyTestCase(CopyTestCase):
    close = 1


class OpenCopyTestCase(CopyTestCase):
    close = 0


class CopyIndexTestCase(common.TempFileMixin, TestCase):

    def test01_index(self):
        """Checking Array.copy() method with indexes."""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test01_index..." % self.__class__.__name__)

        # Create a numpy
        r = numpy.arange(200, dtype='int32')
        r.shape = (100, 2)
        # Save it in a array:
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', r, "title array1")

        # Copy to another array
        array2 = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        if common.verbose:
            print("array1-->", array1.read())
            print("array2-->", array2.read())
            print("attrs array1-->", repr(array1.attrs))
            print("attrs array2-->", repr(array2.attrs))

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        self.assertTrue(allequal(r2, array2.read()))

        # Assert the number of rows in array
        if common.verbose:
            print("nrows in array2-->", array2.nrows)
            print("and it should be-->", r2.shape[0])
        self.assertEqual(r2.shape[0], array2.nrows)

    def test02_indexclosef(self):
        """Checking Array.copy() method with indexes (close file version)"""

        if common.verbose:
            print('\n', '-=' * 30)
            print("Running %s.test02_indexclosef..." % self.__class__.__name__)

        # Create a numpy
        r = numpy.arange(200, dtype='int32')
        r.shape = (100, 2)
        # Save it in a array:
        array1 = self.h5file.create_array(
            self.h5file.root, 'array1', r, "title array1")

        # Copy to another array
        array2 = array1.copy("/", 'array2',
                             start=self.start,
                             stop=self.stop,
                             step=self.step)
        # Close and reopen the file
        self._reopen()
        array1 = self.h5file.root.array1
        array2 = self.h5file.root.array2

        if common.verbose:
            print("array1-->", array1.read())
            print("array2-->", array2.read())
            print("attrs array1-->", repr(array1.attrs))
            print("attrs array2-->", repr(array2.attrs))

        # Check that all the elements are equal
        r2 = r[self.start:self.stop:self.step]
        self.assertTrue(allequal(r2, array2.read()))

        # Assert the number of rows in array
        if common.verbose:
            print("nrows in array2-->", array2.nrows)
            print("and it should be-->", r2.shape[0])
        self.assertEqual(r2.shape[0], array2.nrows)


class CopyIndex1TestCase(CopyIndexTestCase):
    start = 0
    stop = 7
    step = 1


class CopyIndex2TestCase(CopyIndexTestCase):
    start = 0
    stop = -1
    step = 1


class CopyIndex3TestCase(CopyIndexTestCase):
    start = 1
    stop = 7
    step = 1


class CopyIndex4TestCase(CopyIndexTestCase):
    start = 0
    stop = 6
    step = 1


class CopyIndex5TestCase(CopyIndexTestCase):
    start = 3
    stop = 7
    step = 1


class CopyIndex6TestCase(CopyIndexTestCase):
    start = 3
    stop = 6
    step = 2


class CopyIndex7TestCase(CopyIndexTestCase):
    start = 0
    stop = 7
    step = 10


class CopyIndex8TestCase(CopyIndexTestCase):
    start = 6
    stop = -1  # Negative values means starting from the end
    step = 1


class CopyIndex9TestCase(CopyIndexTestCase):
    start = 3
    stop = 4
    step = 1


class CopyIndex10TestCase(CopyIndexTestCase):
    start = 3
    stop = 4
    step = 2


class CopyIndex11TestCase(CopyIndexTestCase):
    start = -3
    stop = -1
    step = 2


class CopyIndex12TestCase(CopyIndexTestCase):
    start = -1   # Should point to the last element
    stop = None  # None should mean the last element (including it)
    step = 1


class GetItemTestCase(common.TempFileMixin, TestCase):

    def test00_single(self):
        """Single element access (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charList
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original first element:", a[0], type(a[0]))
            print("Read first element:", arr[0], type(arr[0]))
        self.assertTrue(allequal(a[0], arr[0]))
        self.assertEqual(type(a[0]), type(arr[0]))

    def test01_single(self):
        """Single element access (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original first element:", a[0], type(a[0]))
            print("Read first element:", arr[0], type(arr[0]))
        self.assertEqual(a[0], arr[0])
        self.assertEqual(type(a[0]), type(arr[0]))

    def test02_range(self):
        """Range element access (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4])
            print("Read elements:", arr[1:4])
        self.assertTrue(allequal(a[1:4], arr[1:4]))

    def test03_range(self):
        """Range element access (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4])
            print("Read elements:", arr[1:4])
        self.assertTrue(allequal(a[1:4], arr[1:4]))

    def test04_range(self):
        """Range element access, strided (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4:2])
            print("Read elements:", arr[1:4:2])
        self.assertTrue(allequal(a[1:4:2], arr[1:4:2]))

    def test05_range(self):
        """Range element access, strided (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4:2])
            print("Read elements:", arr[1:4:2])
        self.assertTrue(allequal(a[1:4:2], arr[1:4:2]))

    def test06_negativeIndex(self):
        """Negative Index element access (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original last element:", a[-1])
            print("Read last element:", arr[-1])
        self.assertTrue(allequal(a[-1], arr[-1]))

    def test07_negativeIndex(self):
        """Negative Index element access (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original before last element:", a[-2])
            print("Read before last element:", arr[-2])
        if isinstance(a[-2], numpy.ndarray):
            self.assertTrue(allequal(a[-2], arr[-2]))
        else:
            self.assertEqual(a[-2], arr[-2])

    def test08_negativeRange(self):
        """Negative range element access (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original last elements:", a[-4:-1])
            print("Read last elements:", arr[-4:-1])
        self.assertTrue(allequal(a[-4:-1], arr[-4:-1]))

    def test09_negativeRange(self):
        """Negative range element access (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        if common.verbose:
            print("Original last elements:", a[-4:-1])
            print("Read last elements:", arr[-4:-1])
        self.assertTrue(allequal(a[-4:-1], arr[-4:-1]))


class GI1NATestCase(GetItemTestCase, TestCase):
    title = "Rank-1 case 1"
    numericalList = numpy.array([3])
    numericalListME = numpy.array([3, 2, 1, 0, 4, 5, 6])
    charList = numpy.array(["3"], 'S')
    charListME = numpy.array(
        ["321", "221", "121", "021", "421", "521", "621"], 'S')


class GI1NAOpenTestCase(GI1NATestCase):
    close = 0


class GI1NACloseTestCase(GI1NATestCase):
    close = 1


class GI2NATestCase(GetItemTestCase):
    # A more complex example
    title = "Rank-1,2 case 2"
    numericalList = numpy.array([3, 4])
    numericalListME = numpy.array([[3, 2, 1, 0, 4, 5, 6],
                                   [2, 1, 0, 4, 5, 6, 7],
                                   [4, 3, 2, 1, 0, 4, 5],
                                   [3, 2, 1, 0, 4, 5, 6],
                                   [3, 2, 1, 0, 4, 5, 6]])

    charList = numpy.array(["a", "b"], 'S')
    charListME = numpy.array(
        [["321", "221", "121", "021", "421", "521", "621"],
         ["21", "21", "11", "02", "42", "21", "61"],
         ["31", "21", "12", "21", "41", "51", "621"],
         ["321", "221", "121", "021",
          "421", "521", "621"],
         ["3241", "2321", "13216",
          "0621", "4421", "5421", "a621"],
         ["a321", "s221", "d121", "g021", "b421", "5vvv21", "6zxzxs21"]], 'S')


class GI2NAOpenTestCase(GI2NATestCase):
    close = 0


class GI2NACloseTestCase(GI2NATestCase):
    close = 1


class SetItemTestCase(common.TempFileMixin, TestCase):

    def test00_single(self):
        """Single element update (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charList
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify a single element of a and arr:
        a[0] = b"b"
        arr[0] = b"b"

        # Get and compare an element
        if common.verbose:
            print("Original first element:", a[0])
            print("Read first element:", arr[0])
        self.assertTrue(allequal(a[0], arr[0]))

    def test01_single(self):
        """Single element update (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        a[0] = 333
        arr[0] = 333

        # Get and compare an element
        if common.verbose:
            print("Original first element:", a[0])
            print("Read first element:", arr[0])
        self.assertEqual(a[0], arr[0])

    def test02_range(self):
        """Range element update (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        a[1:3] = b"xXx"
        arr[1:3] = b"xXx"

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4])
            print("Read elements:", arr[1:4])
        self.assertTrue(allequal(a[1:4], arr[1:4]))

    def test03_range(self):
        """Range element update (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = slice(1, 3, None)
        rng = numpy.arange(a[s].size)*2 + 3
        rng.shape = a[s].shape
        a[s] = rng
        arr[s] = rng

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4])
            print("Read elements:", arr[1:4])
        self.assertTrue(allequal(a[1:4], arr[1:4]))

    def test04_range(self):
        """Range element update, strided (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = slice(1, 4, 2)
        a[s] = b"xXx"
        arr[s] = b"xXx"

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4:2])
            print("Read elements:", arr[1:4:2])
        self.assertTrue(allequal(a[1:4:2], arr[1:4:2]))

    def test05_range(self):
        """Range element update, strided (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = slice(1, 4, 2)
        rng = numpy.arange(a[s].size)*2 + 3
        rng.shape = a[s].shape
        a[s] = rng
        arr[s] = rng

        # Get and compare an element
        if common.verbose:
            print("Original elements:", a[1:4:2])
            print("Read elements:", arr[1:4:2])
        self.assertTrue(allequal(a[1:4:2], arr[1:4:2]))

    def test06_negativeIndex(self):
        """Negative Index element update (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = -1
        a[s] = b"xXx"
        arr[s] = b"xXx"

        # Get and compare an element
        if common.verbose:
            print("Original last element:", a[-1])
            print("Read last element:", arr[-1])
        self.assertTrue(allequal(a[-1], arr[-1]))

    def test07_negativeIndex(self):
        """Negative Index element update (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = -2
        a[s] = a[s]*2 + 3
        arr[s] = arr[s]*2 + 3

        # Get and compare an element
        if common.verbose:
            print("Original before last element:", a[-2])
            print("Read before last element:", arr[-2])
        if isinstance(a[-2], numpy.ndarray):
            self.assertTrue(allequal(a[-2], arr[-2]))
        else:
            self.assertEqual(a[-2], arr[-2])

    def test08_negativeRange(self):
        """Negative range element update (character types)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = slice(-4, -1, None)
        a[s] = b"xXx"
        arr[s] = b"xXx"

        # Get and compare an element
        if common.verbose:
            print("Original last elements:", a[-4:-1])
            print("Read last elements:", arr[-4:-1])
        self.assertTrue(allequal(a[-4:-1], arr[-4:-1]))

    def test09_negativeRange(self):
        """Negative range element update (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of a and arr:
        s = slice(-3, -1, None)
        rng = numpy.arange(a[s].size)*2 + 3
        rng.shape = a[s].shape
        a[s] = rng
        arr[s] = rng

        # Get and compare an element
        if common.verbose:
            print("Original last elements:", a[-4:-1])
            print("Read last elements:", arr[-4:-1])
        self.assertTrue(allequal(a[-4:-1], arr[-4:-1]))

    def test10_outOfRange(self):
        """Out of range update (numerical types)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen('a')
            arr = self.h5file.root.somearray

        # Modify elements of arr that are out of range:
        s = slice(1, a.shape[0]+1, None)
        s2 = slice(1, 1000, None)
        rng = numpy.arange(a[s].size)*2 + 3
        rng.shape = a[s].shape
        a[s] = rng
        rng2 = numpy.arange(a[s2].size)*2 + 3
        rng2.shape = a[s2].shape
        arr[s2] = rng2

        # Get and compare an element
        if common.verbose:
            print("Original last elements:", a[-4:-1])
            print("Read last elements:", arr[-4:-1])
        self.assertTrue(allequal(a[-4:-1], arr[-4:-1]))


class SI1NATestCase(SetItemTestCase, TestCase):
    title = "Rank-1 case 1"
    numericalList = numpy.array([3])
    numericalListME = numpy.array([3, 2, 1, 0, 4, 5, 6])
    charList = numpy.array(["3"], 'S')
    charListME = numpy.array(
        ["321", "221", "121", "021", "421", "521", "621"], 'S')


class SI1NAOpenTestCase(SI1NATestCase):
    close = 0


class SI1NACloseTestCase(SI1NATestCase):
    close = 1


class SI2NATestCase(SetItemTestCase):
    # A more complex example
    title = "Rank-1,2 case 2"
    numericalList = numpy.array([3, 4])
    numericalListME = numpy.array([[3, 2, 1, 0, 4, 5, 6],
                                   [2, 1, 0, 4, 5, 6, 7],
                                   [4, 3, 2, 1, 0, 4, 5],
                                   [3, 2, 1, 0, 4, 5, 6],
                                   [3, 2, 1, 0, 4, 5, 6]])

    charList = numpy.array(["a", "b"], 'S')
    charListME = numpy.array(
        [["321", "221", "121", "021", "421", "521", "621"],
         ["21", "21", "11", "02", "42", "21", "61"],
         ["31", "21", "12", "21", "41", "51", "621"],
         ["321", "221", "121", "021",
          "421", "521", "621"],
         ["3241", "2321", "13216",
          "0621", "4421", "5421", "a621"],
         ["a321", "s221", "d121", "g021", "b421", "5vvv21", "6zxzxs21"]], 'S')


class SI2NAOpenTestCase(SI2NATestCase):
    close = 0


class SI2NACloseTestCase(SI2NATestCase):
    close = 1


class GeneratorTestCase(common.TempFileMixin, TestCase):

    def test00a_single(self):
        """Testing generator access to Arrays, single elements (char)"""

        # Create the array under root and name 'somearray'
        a = self.charList
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print("Result of original iterator:", ga)
            print("Result of read generator:", garr)
        self.assertEqual(ga, garr)

    def test00b_me(self):
        """Testing generator access to Arrays, multiple elements (char)"""

        # Create the array under root and name 'somearray'
        a = self.charListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        ga = [i for i in a]
        garr = [i for i in arr]

        if common.verbose:
            print("Result of original iterator:", ga)
            print("Result of read generator:", garr)
        for i in range(len(ga)):
            self.assertTrue(allequal(ga[i], garr[i]))

    def test01a_single(self):
        """Testing generator access to Arrays, single elements (numeric)"""

        # Create the array under root and name 'somearray'
        a = self.numericalList
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print("Result of original iterator:", ga)
            print("Result of read generator:", garr)
        self.assertEqual(ga, garr)

    def test01b_me(self):
        """Testing generator access to Arrays, multiple elements (numeric)"""

        # Create the array under root and name 'somearray'
        a = self.numericalListME
        arr = self.h5file.create_array(
            self.h5file.root, 'somearray', a, "Some array")

        if self.close:
            self._reopen()
            arr = self.h5file.root.somearray

        # Get and compare an element
        ga = [i for i in a]
        garr = [i for i in arr]
        if common.verbose:
            print("Result of original iterator:", ga)
            print("Result of read generator:", garr)
        for i in range(len(ga)):
            self.assertTrue(allequal(ga[i], garr[i]))


class GE1NATestCase(GeneratorTestCase):
    title = "Rank-1 case 1"
    numericalList = numpy.array([3])
    numericalListME = numpy.array([3, 2, 1, 0, 4, 5, 6])
    charList = numpy.array(["3"], 'S')
    charListME = numpy.array(
        ["321", "221", "121", "021", "421", "521", "621"], 'S')


class GE1NAOpenTestCase(GE1NATestCase):
    close = 0


class GE1NACloseTestCase(GE1NATestCase):
    close = 1


class GE2NATestCase(GeneratorTestCase):
    # A more complex example
    title = "Rank-1,2 case 2"
    numericalList = numpy.array([3, 4])
    numericalListME = numpy.array([[3, 2, 1, 0, 4, 5, 6],
                                   [2, 1, 0, 4, 5, 6, 7],
                                   [4, 3, 2, 1, 0, 4, 5],
                                   [3, 2, 1, 0, 4, 5, 6],
                                   [3, 2, 1, 0, 4, 5, 6]])

    charList = numpy.array(["a", "b"], 'S')
    charListME = numpy.array(
        [["321", "221", "121", "021", "421", "521", "621"],
         ["21", "21", "11", "02", "42", "21", "61"],
         ["31", "21", "12", "21", "41", "51", "621"],
         ["321", "221", "121", "021",
          "421", "521", "621"],
         ["3241", "2321", "13216",
          "0621", "4421", "5421", "a621"],
         ["a321", "s221", "d121", "g021", "b421", "5vvv21", "6zxzxs21"]], 'S')


class GE2NAOpenTestCase(GE2NATestCase):
    close = 0


class GE2NACloseTestCase(GE2NATestCase):
    close = 1


class NonHomogeneousTestCase(common.TempFileMixin, TestCase):
    def test(self):
        """Test for creation of non-homogeneous arrays."""

        # This checks ticket #12.
        self.assertRaises((ValueError, TypeError),
                          self.h5file.create_array, '/', 'test', [1, [2, 3]])
        self.assertRaises(NoSuchNodeError, self.h5file.remove_node, '/test')


class TruncateTestCase(common.TempFileMixin, TestCase):
    def test(self):
        """Test for unability to truncate Array objects."""

        array1 = self.h5file.create_array('/', 'array1', [0, 2])
        self.assertRaises(TypeError, array1.truncate, 0)


class PointSelectionTestCase(common.TempFileMixin, TestCase):

    def setUp(self):
        super(PointSelectionTestCase, self).setUp()
        # Limits for selections
        self.limits = [
            (0, 1),  # just one element
            (20, -10),  # no elements
            (-10, 4),  # several elements
            (0, 10),   # several elements (again)
        ]

        # Create a sample array
        size = numpy.prod(self.shape)
        nparr = numpy.arange(size, dtype=numpy.int32).reshape(self.shape)
        self.nparr = nparr
        self.tbarr = self.h5file.create_array(self.h5file.root, 'array', nparr)

    def test01a_read(self):
        """Test for point-selections (read, boolean keys)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for value1, value2 in self.limits:
            key = (nparr >= value1) & (nparr < value2)
            if common.verbose:
                print("Selection to test:", key)
            a = nparr[key]
            b = tbarr[key]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables selections does not match.")

    def test01b_read(self):
        """Test for point-selections (read, integer keys)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for value1, value2 in self.limits:
            key = numpy.where((nparr >= value1) & (nparr < value2))
            if common.verbose:
                print("Selection to test:", key)
            a = nparr[key]
            b = tbarr[key]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables selections does not match.")

    def test01c_read(self):
        """Test for point-selections (read, float keys)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for value1, value2 in self.limits:
            key = numpy.where((nparr >= value1) & (nparr < value2))
            if common.verbose:
                print("Selection to test:", key)
            # a = nparr[key]
            fkey = numpy.array(key, "f4")
            self.assertRaises((IndexError, TypeError), tbarr.__getitem__, fkey)

    def test01d_read(self):
        nparr = self.nparr
        tbarr = self.tbarr

        for key in self.working_keyset:
            if nparr.ndim > 1:
                key = tuple(key)
            if common.verbose:
                print("Selection to test:", key)
            a = nparr[key]
            b = tbarr[key]
            npt.assert_array_equal(
                a, b, "NumPy array and PyTables selections does not match.")

    def test01e_read(self):
        tbarr = self.tbarr

        for key in self.not_working_keyset:
            if common.verbose:
                print("Selection to test:", key)

            self.assertRaises(IndexError, tbarr.__getitem__, key)

    def test02a_write(self):
        """Test for point-selections (write, boolean keys)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for value1, value2 in self.limits:
            key = (nparr >= value1) & (nparr < value2)
            if common.verbose:
                print("Selection to test:", key)
            s = nparr[key]
            nparr[key] = s * 2
            tbarr[key] = s * 2
            a = nparr[:]
            b = tbarr[:]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables modifications does not match.")

    def test02b_write(self):
        """Test for point-selections (write, integer keys)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for value1, value2 in self.limits:
            key = numpy.where((nparr >= value1) & (nparr < value2))
            if common.verbose:
                print("Selection to test:", key)
            s = nparr[key]
            nparr[key] = s * 2
            tbarr[key] = s * 2
            a = nparr[:]
            b = tbarr[:]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables modifications does not match.")

    def test02c_write(self):
        """Test for point-selections (write, integer values, broadcast)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for value1, value2 in self.limits:
            key = numpy.where((nparr >= value1) & (nparr < value2))
            if common.verbose:
                print("Selection to test:", key)
            # s = nparr[key]
            nparr[key] = 2   # force a broadcast
            tbarr[key] = 2   # force a broadcast
            a = nparr[:]
            b = tbarr[:]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables modifications does not match.")


class PointSelection0(PointSelectionTestCase):
    shape = (3,)
    working_keyset = [
        [0, 1],
        [0, -1],
    ]
    not_working_keyset = [
        [0, 3],
        [0, 4],
        [0, -4],
    ]


class PointSelection1(PointSelectionTestCase):
    shape = (5, 3, 3)
    working_keyset = [
        [(0, 0), (0, 1), (0, 0)],
        [(0, 0), (0, -1), (0, 0)],
    ]
    not_working_keyset = [
        [(0, 0), (0, 3), (0, 0)],
        [(0, 0), (0, 4), (0, 0)],
        [(0, 0), (0, -4), (0, 0)],
        [(0, 0), (0, -5), (0, 0)]
    ]


class PointSelection2(PointSelectionTestCase):
    shape = (7, 3)
    working_keyset = [
        [(0, 0), (0, 1)],
        [(0, 0), (0, -1)],
        [(0, 0), (0, -2)],
    ]
    not_working_keyset = [
        [(0, 0), (0, 3)],
        [(0, 0), (0, 4)],
        [(0, 0), (0, -4)],
        [(0, 0), (0, -5)],
    ]


class PointSelection3(PointSelectionTestCase):
    shape = (4, 3, 2, 1)
    working_keyset = [
        [(0, 0), (0, 1), (0, 0), (0, 0)],
        [(0, 0), (0, -1), (0, 0), (0, 0)],
    ]
    not_working_keyset = [
        [(0, 0), (0, 3), (0, 0), (0, 0)],
        [(0, 0), (0, 4), (0, 0), (0, 0)],
        [(0, 0), (0, -4), (0, 0), (0, 0)],
    ]


class PointSelection4(PointSelectionTestCase):
    shape = (1, 3, 2, 5, 6)
    working_keyset = [
        [(0, 0), (0, 1), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, -1), (0, 0), (0, 0), (0, 0)],
    ]
    not_working_keyset = [
        [(0, 0), (0, 3), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 4), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, -4), (0, 0), (0, 0), (0, 0)],
    ]


class FancySelectionTestCase(common.TempFileMixin, TestCase):

    def setUp(self):
        super(FancySelectionTestCase, self).setUp()

        M, N, O = self.shape

        # The next are valid selections for both NumPy and PyTables
        self.working_keyset = [
            ([1, 3], slice(1, N-1), 2),
            ([M-1, 1, 3, 2], slice(None), 2),  # unordered lists supported
            (slice(M), [N-1, 1, 0], slice(None)),
            (slice(1, M, 3), slice(1, N), [O-1, 1, 0]),
            (M-1, [2, 1], 1),
            (1, 2, 1),              # regular selection
            ([1, 2], -2, -1),     # negative indices
            ([1, -2], 2, -1),     # more negative indices
            ([1, -2], 2, Ellipsis),     # one ellipsis
            (Ellipsis, [1, 2]),    # one ellipsis
            (numpy.array(
                [1, -2], 'i4'), 2, -1),  # array 32-bit instead of list
            (numpy.array(
                [-1, 2], 'i8'), 2, -1),  # array 64-bit instead of list
        ]

        # Using booleans instead of ints is deprecated since numpy 1.8
        # Tests for keys that have to support the __index__ attribute
        #self.working_keyset.append(
        #    (False, True),  # equivalent to (0,1) ;-)
        #)

        # Valid selections for NumPy, but not for PyTables (yet)
        # The next should raise an IndexError
        self.not_working_keyset = [
            numpy.array([False, True], dtype="b1"),  # boolean arrays
            ([1, 2, 1], 2, 1),    # repeated values
            ([1, 2], 2, [1, 2]),  # several lists
            ([], 2, 1),         # empty selections
            (Ellipsis, [1, 2], Ellipsis),  # several ellipsis
            # Using booleans instead of ints is deprecated since numpy 1.8
            ([False, True]),    # boolean values with incompatible shape
        ]

        # The next should raise an IndexError in both NumPy and PyTables
        self.not_working_oob = [
            ([1, 2], 2, 1000),         # out-of-bounds selections
            ([1, 2], 2000, 1),         # out-of-bounds selections
        ]

        # The next should raise a IndexError in both NumPy and PyTables
        self.not_working_too_many = [
            ([1, 2], 2, 1, 1),
        ]

        # Create a sample array
        nparr = numpy.empty(self.shape, dtype=numpy.int32)
        data = numpy.arange(N * O, dtype=numpy.int32).reshape(N, O)
        for i in range(M):
            nparr[i] = data * i
        self.nparr = nparr
        self.tbarr = self.h5file.create_array(self.h5file.root, 'array', nparr)

    def test01a_read(self):
        """Test for fancy-selections (working selections, read)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for key in self.working_keyset:
            if common.verbose:
                print("Selection to test:", key)
            a = nparr[key]
            b = tbarr[key]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables selections does not match.")

    def test01b_read(self):
        """Test for fancy-selections (not working selections, read)."""

        # nparr = self.nparr
        tbarr = self.tbarr
        for key in self.not_working_keyset:
            if common.verbose:
                print("Selection to test:", key)
            # a = nparr[key]
            self.assertRaises(IndexError, tbarr.__getitem__, key)

    def test01c_read(self):
        """Test for fancy-selections (out-of-bound indexes, read)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for key in self.not_working_oob:
            if common.verbose:
                print("Selection to test:", key)
            self.assertRaises(IndexError, nparr.__getitem__, key)
            self.assertRaises(IndexError, tbarr.__getitem__, key)

    def test01d_read(self):
        """Test for fancy-selections (too many indexes, read)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for key in self.not_working_too_many:
            if common.verbose:
                print("Selection to test:", key)
            # ValueError for numpy 1.6.x and earlier
            # IndexError in numpy > 1.8.0
            self.assertRaises((ValueError, IndexError), nparr.__getitem__, key)
            self.assertRaises(IndexError, tbarr.__getitem__, key)

    def test02a_write(self):
        """Test for fancy-selections (working selections, write)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for key in self.working_keyset:
            if common.verbose:
                print("Selection to test:", key)
            s = nparr[key]
            nparr[key] = s * 2
            tbarr[key] = s * 2
            a = nparr[:]
            b = tbarr[:]
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables modifications does not match.")

    def test02b_write(self):
        """Test for fancy-selections (working selections, write, broadcast)."""

        nparr = self.nparr
        tbarr = self.tbarr
        for key in self.working_keyset:
            if common.verbose:
                print("Selection to test:", key)
            # s = nparr[key]
            nparr[key] = 2   # broadcast value
            tbarr[key] = 2   # broadcast value
            a = nparr[:]
            b = tbarr[:]
#             if common.verbose:
#                 print("NumPy modified array:", a)
#                 print("PyTables modifyied array:", b)
            self.assertTrue(
                numpy.alltrue(a == b),
                "NumPy array and PyTables modifications does not match.")


class FancySelection1(FancySelectionTestCase):
    shape = (5, 3, 3)  # Minimum values


class FancySelection2(FancySelectionTestCase):
    # shape = (5, 3, 3)  # Minimum values
    shape = (7, 3, 3)


class FancySelection3(FancySelectionTestCase):
    # shape = (5, 3, 3)  # Minimum values
    shape = (7, 4, 5)


class FancySelection4(FancySelectionTestCase):
    # shape = (5, 3, 3)  # Minimum values
    shape = (5, 3, 10)


class CopyNativeHDF5MDAtom(TestCase):

    def setUp(self):
        super(CopyNativeHDF5MDAtom, self).setUp()
        filename = test_filename("array_mdatom.h5")
        self.h5file = tables.open_file(filename, "r")
        self.arr = self.h5file.root.arr
        self.copy = tempfile.mktemp(".h5")
        self.copyh = tables.open_file(self.copy, mode="w")
        self.arr2 = self.arr.copy(self.copyh.root, newname="arr2")

    def tearDown(self):
        self.h5file.close()
        self.copyh.close()
        os.remove(self.copy)
        super(CopyNativeHDF5MDAtom, self).tearDown()

    def test01_copy(self):
        """Checking that native MD atoms are copied as-is"""

        self.assertEqual(self.arr.atom, self.arr2.atom)
        self.assertEqual(self.arr.shape, self.arr2.shape)

    def test02_reopen(self):
        """Checking that native MD atoms are copied as-is (re-open)"""

        self.copyh.close()
        self.copyh = tables.open_file(self.copy, mode="r")
        self.arr2 = self.copyh.root.arr2
        self.assertEqual(self.arr.atom, self.arr2.atom)
        self.assertEqual(self.arr.shape, self.arr2.shape)


class AccessClosedTestCase(common.TempFileMixin, TestCase):

    def setUp(self):
        super(AccessClosedTestCase, self).setUp()

        a = numpy.zeros((10, 10))
        self.array = self.h5file.create_array(self.h5file.root, 'array', a)

    def test_read(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.read)

    def test_getitem(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.__getitem__, 0)

    def test_setitem(self):
        self.h5file.close()
        self.assertRaises(ClosedNodeError, self.array.__setitem__, 0, 0)


class BroadcastTest(common.TempFileMixin, TestCase):

    def test(self):
        """Test correct broadcasting when the array atom is not scalar."""

        array_shape = (2, 3)
        element_shape = (3,)

        dtype = numpy.dtype((numpy.int, element_shape))
        atom = Atom.from_dtype(dtype)
        h5arr = self.h5file.create_array(self.h5file.root, 'array',
                                          atom=atom, shape=array_shape)

        size = numpy.prod(element_shape)
        nparr = numpy.arange(size).reshape(element_shape)

        h5arr[0] = nparr
        self.assertTrue(numpy.all(h5arr[0] == nparr))


class TestCreateArrayArgs(common.TempFileMixin, TestCase):
    where = '/'
    name = 'array'
    obj = numpy.array([[1, 2], [3, 4]])
    title = 'title'
    byteorder = None
    createparents = False
    atom = Atom.from_dtype(obj.dtype)
    shape = obj.shape

    def test_positional_args(self):
        self.h5file.create_array(self.where, self.name, self.obj, self.title)
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_positional_args_atom_shape(self):
        self.h5file.create_array(self.where, self.name, None, self.title,
                                 self.byteorder, self.createparents,
                                 self.atom, self.shape)
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(numpy.zeros_like(self.obj), nparr))

    def test_kwargs_obj(self):
        self.h5file.create_array(self.where, self.name, title=self.title,
                                 obj=self.obj)
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_atom_shape_01(self):
        ptarr = self.h5file.create_array(self.where, self.name,
                                         title=self.title,
                                         atom=self.atom, shape=self.shape)
        ptarr[...] = self.obj
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_atom_shape_02(self):
        ptarr = self.h5file.create_array(self.where, self.name,
                                         title=self.title,
                                         atom=self.atom, shape=self.shape)
        #ptarr[...] = self.obj
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(numpy.zeros_like(self.obj), nparr))

    def test_kwargs_obj_atom(self):
        ptarr = self.h5file.create_array(self.where, self.name,
                                         title=self.title,
                                         obj=self.obj,
                                         atom=self.atom)
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_shape(self):
        ptarr = self.h5file.create_array(self.where, self.name,
                                         title=self.title,
                                         obj=self.obj,
                                         shape=self.shape)
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_atom_shape(self):
        ptarr = self.h5file.create_array(self.where, self.name,
                                         title=self.title,
                                         obj=self.obj,
                                         atom=self.atom,
                                         shape=self.shape)
        self.h5file.close()

        self.h5file = tables.open_file(self.h5fname)
        ptarr = self.h5file.get_node(self.where, self.name)
        nparr = ptarr.read()

        self.assertEqual(ptarr.title, self.title)
        self.assertEqual(ptarr.shape, self.shape)
        self.assertEqual(ptarr.atom, self.atom)
        self.assertEqual(ptarr.atom.dtype, self.atom.dtype)
        self.assertTrue(allequal(self.obj, nparr))

    def test_kwargs_obj_atom_error(self):
        atom = Atom.from_dtype(numpy.dtype('complex'))
        #shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_array,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=atom)

    def test_kwargs_obj_shape_error(self):
        #atom = Atom.from_dtype(numpy.dtype('complex'))
        shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_array,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          shape=shape)

    def test_kwargs_obj_atom_shape_error_01(self):
        atom = Atom.from_dtype(numpy.dtype('complex'))
        #shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_array,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=atom,
                          shape=self.shape)

    def test_kwargs_obj_atom_shape_error_02(self):
        #atom = Atom.from_dtype(numpy.dtype('complex'))
        shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_array,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=self.atom,
                          shape=shape)

    def test_kwargs_obj_atom_shape_error_03(self):
        atom = Atom.from_dtype(numpy.dtype('complex'))
        shape = self.shape + self.shape
        self.assertRaises(TypeError,
                          self.h5file.create_array,
                          self.where,
                          self.name,
                          title=self.title,
                          obj=self.obj,
                          atom=atom,
                          shape=shape)


def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        # The scalar case test should be refined in order to work
        theSuite.addTest(unittest.makeSuite(Basic0DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic0DTwoTestCase))
        # theSuite.addTest(unittest.makeSuite(Basic1DZeroTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic1DThreeTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2DOneTestCase))
        theSuite.addTest(unittest.makeSuite(Basic2DTwoTestCase))
        theSuite.addTest(unittest.makeSuite(Basic10DTestCase))
        # The 32 dimensions case is tested on GroupsArray
        # theSuite.addTest(unittest.makeSuite(Basic32DTestCase))
        theSuite.addTest(unittest.makeSuite(ReadOutArgumentTests))
        theSuite.addTest(unittest.makeSuite(
            SizeOnDiskInMemoryPropertyTestCase))
        theSuite.addTest(unittest.makeSuite(GroupsArrayTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexNotReopenNotEndianTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexReopenNotEndianTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexNotReopenEndianTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexReopenEndianTestCase))
        theSuite.addTest(unittest.makeSuite(CloseCopyTestCase))
        theSuite.addTest(unittest.makeSuite(OpenCopyTestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex1TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex2TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex3TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex4TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex5TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex6TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex7TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex8TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex9TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex10TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex11TestCase))
        theSuite.addTest(unittest.makeSuite(CopyIndex12TestCase))
        theSuite.addTest(unittest.makeSuite(GI1NAOpenTestCase))
        theSuite.addTest(unittest.makeSuite(GI1NACloseTestCase))
        theSuite.addTest(unittest.makeSuite(GI2NAOpenTestCase))
        theSuite.addTest(unittest.makeSuite(GI2NACloseTestCase))
        theSuite.addTest(unittest.makeSuite(SI1NAOpenTestCase))
        theSuite.addTest(unittest.makeSuite(SI1NACloseTestCase))
        theSuite.addTest(unittest.makeSuite(SI2NAOpenTestCase))
        theSuite.addTest(unittest.makeSuite(SI2NACloseTestCase))
        theSuite.addTest(unittest.makeSuite(GE1NAOpenTestCase))
        theSuite.addTest(unittest.makeSuite(GE1NACloseTestCase))
        theSuite.addTest(unittest.makeSuite(GE2NAOpenTestCase))
        theSuite.addTest(unittest.makeSuite(GE2NACloseTestCase))
        theSuite.addTest(unittest.makeSuite(NonHomogeneousTestCase))
        theSuite.addTest(unittest.makeSuite(TruncateTestCase))
        theSuite.addTest(unittest.makeSuite(FancySelection1))
        theSuite.addTest(unittest.makeSuite(FancySelection2))
        theSuite.addTest(unittest.makeSuite(FancySelection3))
        theSuite.addTest(unittest.makeSuite(FancySelection4))
        theSuite.addTest(unittest.makeSuite(PointSelection0))
        theSuite.addTest(unittest.makeSuite(PointSelection1))
        theSuite.addTest(unittest.makeSuite(PointSelection2))
        theSuite.addTest(unittest.makeSuite(PointSelection3))
        theSuite.addTest(unittest.makeSuite(PointSelection4))
        theSuite.addTest(unittest.makeSuite(CopyNativeHDF5MDAtom))
        theSuite.addTest(unittest.makeSuite(AccessClosedTestCase))
        theSuite.addTest(unittest.makeSuite(TestCreateArrayArgs))
        theSuite.addTest(unittest.makeSuite(BroadcastTest))

    return theSuite


if __name__ == '__main__':
    common.parse_argv(sys.argv)
    common.print_versions()
    unittest.main(defaultTest='suite')
