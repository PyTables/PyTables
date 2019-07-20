# -*- coding: utf-8 -*-

import sys

import numpy

import tables
from tables import (
    Col, StringCol, Atom, StringAtom, Int16Atom, Int32Atom,
    FloatAtom, Float64Atom,
)
from tables.tests import common
from tables.tests.common import unittest, test_filename
from tables.tests.common import PyTablesTestCase as TestCase


# Test Record class
class Record(tables.IsDescription):
    var1 = StringCol(itemsize=4)  # 4-character String
    var2 = Col.from_kind('int')   # integer
    var3 = Col.from_kind('int', itemsize=2)  # short integer
    var4 = Col.from_kind('float')  # double (double-precision)
    var5 = Col.from_kind('float', itemsize=4)  # float  (single-precision)
    var6 = Col.from_kind('complex')  # double-precision
    var7 = Col.from_kind('complex', itemsize=8)  # single-precision
    if hasattr(tables, "Float16Atom"):
        var8 = Col.from_kind('float', itemsize=2)  # half-precision
    if hasattr(tables, "Float96Atom"):
        var9 = Col.from_kind('float', itemsize=12)  # extended-precision
    if hasattr(tables, "Float128Atom"):
        var10 = Col.from_kind('float', itemsize=16)  # extended-precision
    if hasattr(tables, "Complex192Atom"):
        var11 = Col.from_kind('complex', itemsize=24)  # extended-precision
    if hasattr(tables, "Complex256Atom"):
        var12 = Col.from_kind('complex', itemsize=32)  # extended-precision


class RangeTestCase(common.TempFileMixin, TestCase):
    title = "This is the table title"
    expectedrows = 100
    maxshort = 2 ** 15
    maxint = 2147483648   # (2 ** 31)
    compress = 0

    def setUp(self):
        super(RangeTestCase, self).setUp()
        self.rootgroup = self.h5file.root

        # Create a table
        self.table = self.h5file.create_table(self.rootgroup, 'table',
                                              Record, self.title)

    def test00_range(self):
        """Testing the range check."""

        rec = self.table.row

        # Save a record
        i = self.maxshort
        rec['var1'] = '%04d' % (i)
        rec['var2'] = i
        rec['var3'] = i
        rec['var4'] = float(i)
        rec['var5'] = float(i)
        rec['var6'] = float(i)
        rec['var7'] = complex(i, i)
        if hasattr(tables, "Float16Atom"):
            rec['var8'] = float(i)
        if hasattr(tables, "Float96Atom"):
            rec['var9'] = float(i)
        if hasattr(tables, "Float128Atom"):
            rec['var10'] = float(i)
        try:
            rec.append()
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print("\nGreat!, the next ValueError was catched!")
                print(value)
            pass
        else:
            if common.verbose:
                print(
                    "\nNow, the range overflow no longer issues a ValueError")

    def test01_type(self):
        """Testing the type check."""

        rec = self.table.row
        # Save a record
        i = self.maxshort
        rec['var1'] = '%04d' % (i)
        rec['var2'] = i
        rec['var3'] = i % self.maxshort
        rec['var5'] = float(i)

        with self.assertRaises(TypeError):
            rec['var4'] = "124c"

        rec['var6'] = float(i)
        rec['var7'] = complex(i, i)
        if hasattr(tables, "Float16Atom"):
            rec['var8'] = float(i)
        if hasattr(tables, "Float96Atom"):
            rec['var9'] = float(i)
        if hasattr(tables, "Float128Atom"):
            rec['var10'] = float(i)


# Check the dtype read-only attribute
class DtypeTestCase(common.TempFileMixin, TestCase):

    def test00a_table(self):
        """Check dtype accessor for Table objects."""

        a = self.h5file.create_table('/', 'table', Record)
        self.assertEqual(a.dtype, a.description._v_dtype)

    def test00b_column(self):
        """Check dtype accessor for Column objects."""

        a = self.h5file.create_table('/', 'table', Record)
        c = a.cols.var3
        self.assertEqual(c.dtype, a.description._v_dtype['var3'])

    def test01_array(self):
        """Check dtype accessor for Array objects."""

        a = self.h5file.create_array('/', 'array', [1, 2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test02_carray(self):
        """Check dtype accessor for CArray objects."""

        a = self.h5file.create_carray('/', 'array', atom=FloatAtom(),
                                      shape=[1, 2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test03_carray(self):
        """Check dtype accessor for EArray objects."""

        a = self.h5file.create_earray('/', 'array', atom=FloatAtom(),
                                      shape=[0, 2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test04_vlarray(self):
        """Check dtype accessor for VLArray objects."""

        a = self.h5file.create_vlarray('/', 'array', FloatAtom())
        self.assertEqual(a.dtype, a.atom.dtype)


class ReadFloatTestCase(common.TestFileMixin, TestCase):
    h5fname = test_filename("float.h5")
    nrows = 5
    ncols = 6

    def setUp(self):
        super(ReadFloatTestCase, self).setUp()
        x = numpy.arange(self.ncols)
        y = numpy.arange(self.nrows)
        y.shape = (self.nrows, 1)
        self.values = x + y

    def test01_read_float16(self):
        dtype = "float16"
        if hasattr(numpy, dtype):
            ds = getattr(self.h5file.root, dtype)
            self.assertNotIsInstance(ds, tables.UnImplemented)
            self.assertEqual(ds.shape, (self.nrows, self.ncols))
            self.assertEqual(ds.dtype, dtype)
            self.assertTrue(common.allequal(
                ds.read(), self.values.astype(dtype)))
        else:
            with self.assertWarns(UserWarning):
                ds = getattr(self.h5file.root, dtype)
            self.assertIsInstance(ds, tables.UnImplemented)

    def test02_read_float32(self):
        dtype = "float32"
        ds = getattr(self.h5file.root, dtype)
        self.assertNotIsInstance(ds, tables.UnImplemented)
        self.assertEqual(ds.shape, (self.nrows, self.ncols))
        self.assertEqual(ds.dtype, dtype)
        self.assertTrue(common.allequal(
            ds.read(), self.values.astype(dtype)))

    def test03_read_float64(self):
        dtype = "float64"
        ds = getattr(self.h5file.root, dtype)
        self.assertNotIsInstance(ds, tables.UnImplemented)
        self.assertEqual(ds.shape, (self.nrows, self.ncols))
        self.assertEqual(ds.dtype, dtype)
        self.assertTrue(common.allequal(
            ds.read(), self.values.astype(dtype)))

    def test04_read_longdouble(self):
        dtype = "longdouble"
        if hasattr(tables, "Float96Atom") or hasattr(tables, "Float128Atom"):
            ds = getattr(self.h5file.root, dtype)
            self.assertNotIsInstance(ds, tables.UnImplemented)
            self.assertEqual(ds.shape, (self.nrows, self.ncols))
            self.assertEqual(ds.dtype, dtype)
            self.assertTrue(common.allequal(
                ds.read(), self.values.astype(dtype)))

            if hasattr(tables, "Float96Atom"):
                self.assertEqual(ds.dtype, "float96")
            elif hasattr(tables, "Float128Atom"):
                self.assertEqual(ds.dtype, "float128")
        else:
            # XXX: check
            # the behavior depends on the HDF5 lib configuration
            try:
                with self.assertWarns(UserWarning):
                    ds = getattr(self.h5file.root, dtype)
                self.assertIsInstance(ds, tables.UnImplemented)
            except AssertionError:
                from tables.utilsextension import _broken_hdf5_long_double
                if not _broken_hdf5_long_double():
                    ds = getattr(self.h5file.root, dtype)
                    self.assertEqual(ds.dtype, "float64")

    def test05_read_quadprecision_float(self):
        # XXX: check
        try:
            with self.assertWarns(UserWarning):
                ds = self.h5file.root.quadprecision
            self.assertIsInstance(ds, tables.UnImplemented)
        except AssertionError:
            # NOTE: it would be nice to have some sort of message that warns
            #       against the potential precision loss: the quad-precision
            #       dataset actually uses 128 bits for each element, not just
            #       80 bits (longdouble)
            ds = self.h5file.root.quadprecision
            self.assertEqual(ds.dtype, "longdouble")


class AtomTestCase(TestCase):
    def test_init_parameters_01(self):
        atom1 = StringAtom(itemsize=12)
        atom2 = atom1.copy()
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))
        self.assertIsNot(atom1, atom2)

    def test_init_parameters_02(self):
        atom1 = StringAtom(itemsize=12)
        atom2 = atom1.copy(itemsize=100, shape=(2, 2))
        self.assertEqual(atom2,
                         StringAtom(itemsize=100, shape=(2, 2), dflt=b''))

    def test_init_parameters_03(self):
        atom1 = StringAtom(itemsize=12)
        self.assertRaises(TypeError, atom1.copy, foobar=42)

    def test_from_dtype_01(self):
        atom1 = Atom.from_dtype(numpy.dtype((numpy.int16, (2, 2))))
        atom2 = Int16Atom(shape=(2, 2), dflt=0)
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_dtype_02(self):
        atom1 = Atom.from_dtype(numpy.dtype('S5'), dflt=b'hello')
        atom2 = StringAtom(itemsize=5, shape=(), dflt=b'hello')
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_dtype_03(self):
        with self.assertWarns(Warning):
            atom1 = Atom.from_dtype(numpy.dtype('U5'), dflt=b'hello')
        atom2 = StringAtom(itemsize=5, shape=(), dflt=b'hello')
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_dtype_04(self):
        atom1 = Atom.from_dtype(numpy.dtype('float64'))
        atom2 = Float64Atom(shape=(), dflt=0.0)
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_kind_01(self):
        atom1 = Atom.from_kind('int', itemsize=2, shape=(2, 2))
        atom2 = Int16Atom(shape=(2, 2), dflt=0)
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_kind_02(self):
        atom1 = Atom.from_kind('int', shape=(2, 2))
        atom2 = Int32Atom(shape=(2, 2), dflt=0)
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_kind_03(self):
        atom1 = Atom.from_kind('int', shape=1)
        atom2 = Int32Atom(shape=(1,), dflt=0)
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_kind_04(self):
        atom1 = Atom.from_kind('string', itemsize=5, dflt=b'hello')
        atom2 = StringAtom(itemsize=5, shape=(), dflt=b'hello')
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))

    def test_from_kind_05(self):
        # ValueError: no default item size for kind ``string``
        self.assertRaises(ValueError, Atom.from_kind, 'string', dflt=b'hello')

    def test_from_kind_06(self):
        # ValueError: unknown kind: 'Float'
        self.assertRaises(ValueError, Atom.from_kind, 'Float')


def suite():
    import doctest
    import tables.atom

    theSuite = unittest.TestSuite()

    for i in range(1):
        theSuite.addTest(doctest.DocTestSuite(tables.atom))
        theSuite.addTest(unittest.makeSuite(AtomTestCase))
        theSuite.addTest(unittest.makeSuite(RangeTestCase))
        theSuite.addTest(unittest.makeSuite(DtypeTestCase))
        theSuite.addTest(unittest.makeSuite(ReadFloatTestCase))

    return theSuite


if __name__ == '__main__':
    common.parse_argv(sys.argv)
    common.print_versions()
    unittest.main(defaultTest='suite')
