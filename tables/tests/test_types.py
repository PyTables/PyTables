# -*- coding: utf-8 -*-

import sys
import unittest
import os

import numpy

from tables import *
from tables.tests import common

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

# Test Record class


class Record(IsDescription):
    var1 = StringCol(itemsize=4)  # 4-character String
    var2 = Col.from_kind('int')   # integer
    var3 = Col.from_kind('int', itemsize=2)  # short integer
    var4 = Col.from_kind('float')  # double (double-precision)
    var5 = Col.from_kind('float', itemsize=4)  # float  (single-precision)
    var6 = Col.from_kind('complex')  # double-precision
    var7 = Col.from_kind('complex', itemsize=8)  # single-precision
    if hasattr(numpy, "float16"):
        var8 = Col.from_kind('float', itemsize=2)  # half-precision
    if hasattr(numpy, "float96"):
        var9 = Col.from_kind('float', itemsize=12)  # extended-precision
    if hasattr(numpy, "float128"):
        var10 = Col.from_kind('float', itemsize=16)  # extended-precision
    if hasattr(numpy, "complex192"):
        var11 = Col.from_kind('complex', itemsize=24)  # extended-precision
    if hasattr(numpy, "complex256"):
        var12 = Col.from_kind('complex', itemsize=32)  # extended-precision


class RangeTestCase(unittest.TestCase):
    file = "test.h5"
    title = "This is the table title"
    expectedrows = 100
    maxshort = 2 ** 15
    maxint = 2147483648   # (2 ** 31)
    compress = 0

    def setUp(self):
        # Create an instance of HDF5 Table
        self.fileh = open_file(self.file, mode="w")
        self.rootgroup = self.fileh.root

        # Create a table
        self.table = self.fileh.create_table(self.rootgroup, 'table',
                                             Record, self.title)

    def tearDown(self):
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test00_range(self):
        """Testing the range check"""
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
        if hasattr(numpy, "float16"):
            rec['var8'] = float(i)
        if hasattr(numpy, "float96"):
            rec['var9'] = float(i)
        if hasattr(numpy, "float128"):
            rec['var10'] = float(i)
        try:
            rec.append()
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
            pass
        else:
            if common.verbose:
                print "\nNow, the range overflow no longer issues a ValueError"

    def test01_type(self):
        """Testing the type check"""
        rec = self.table.row
        # Save a record
        i = self.maxshort
        rec['var1'] = '%04d' % (i)
        rec['var2'] = i
        rec['var3'] = i % self.maxshort
        rec['var5'] = float(i)
        try:
            rec['var4'] = "124c"
        except TypeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next TypeError was catched!"
                print value
            pass
        else:
            print rec
            self.fail("expected a TypeError")
        rec['var6'] = float(i)
        rec['var7'] = complex(i, i)
        if hasattr(numpy, "float16"):
            rec['var8'] = float(i)
        if hasattr(numpy, "float96"):
            rec['var9'] = float(i)
        if hasattr(numpy, "float128"):
            rec['var10'] = float(i)


# Check the dtype read-only attribute
class DtypeTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test00a_table(self):
        """Check dtype accessor for Table objects"""
        a = self.h5file.create_table('/', 'table', Record)
        self.assertEqual(a.dtype, a.description._v_dtype)

    def test00b_column(self):
        """Check dtype accessor for Column objects"""
        a = self.h5file.create_table('/', 'table', Record)
        c = a.cols.var3
        self.assertEqual(c.dtype, a.description._v_dtype['var3'])

    def test01_array(self):
        """Check dtype accessor for Array objects"""
        a = self.h5file.create_array('/', 'array', [1, 2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test02_carray(self):
        """Check dtype accessor for CArray objects"""
        a = self.h5file.create_carray(
            '/', 'array', atom=FloatAtom(), shape=[1, 2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test03_carray(self):
        """Check dtype accessor for EArray objects"""
        a = self.h5file.create_earray(
            '/', 'array', atom=FloatAtom(), shape=[0, 2])
        self.assertEqual(a.dtype, a.atom.dtype)

    def test04_vlarray(self):
        """Check dtype accessor for VLArray objects"""
        a = self.h5file.create_vlarray('/', 'array', FloatAtom())
        self.assertEqual(a.dtype, a.atom.dtype)


class ReadFloatTestCase(common.PyTablesTestCase):
    filename = "float.h5"
    nrows = 5
    ncols = 6

    def setUp(self):
        self.fileh = open_file(self._testFilename(self.filename), mode="r")
        x = numpy.arange(self.ncols)
        y = numpy.arange(self.nrows)
        y.shape = (self.nrows, 1)
        self.values = x + y

    def tearDown(self):
        self.fileh.close()

    def test01_read_float16(self):
        dtype = "float16"
        if hasattr(numpy, dtype):
            ds = getattr(self.fileh.root, dtype)
            self.assertFalse(isinstance(ds, UnImplemented))
            self.assertEqual(ds.shape, (self.nrows, self.ncols))
            self.assertEqual(ds.dtype, dtype)
            self.assertTrue(common.allequal(
                ds.read(), self.values.astype(dtype)))
        else:
            ds = self.assertWarns(UserWarning, getattr, self.fileh.root, dtype)
            self.assertTrue(isinstance(ds, UnImplemented))

    def test02_read_float32(self):
        dtype = "float32"
        ds = getattr(self.fileh.root, dtype)
        self.assertFalse(isinstance(ds, UnImplemented))
        self.assertEqual(ds.shape, (self.nrows, self.ncols))
        self.assertEqual(ds.dtype, dtype)
        self.assertTrue(common.allequal(
            ds.read(), self.values.astype(dtype)))

    def test03_read_float64(self):
        dtype = "float64"
        ds = getattr(self.fileh.root, dtype)
        self.assertFalse(isinstance(ds, UnImplemented))
        self.assertEqual(ds.shape, (self.nrows, self.ncols))
        self.assertEqual(ds.dtype, dtype)
        self.assertTrue(common.allequal(
            ds.read(), self.values.astype(dtype)))

    def test04_read_longdouble(self):
        dtype = "longdouble"
        if hasattr(numpy, "float96") or hasattr(numpy, "float128"):
            ds = getattr(self.fileh.root, dtype)
            self.assertFalse(isinstance(ds, UnImplemented))
            self.assertEqual(ds.shape, (self.nrows, self.ncols))
            self.assertEqual(ds.dtype, dtype)
            self.assertTrue(common.allequal(
                ds.read(), self.values.astype(dtype)))

            if hasattr(numpy, "float96"):
                self.assertEqual(ds.dtype, "float96")
            elif hasattr(numpy, "float128"):
                self.assertEqual(ds.dtype, "float128")
        else:
            # XXX: check
            # ds = self.assertWarns(UserWarning,
            #                       getattr, self.fileh.root, dtype)
            # self.assertTrue(isinstance(ds, UnImplemented))

            ds = getattr(self.fileh.root, dtype)
            self.assertEqual(ds.dtype, "float64")

    def test05_read_quadprecision_float(self):
        # ds = self.assertWarns(UserWarning, getattr, self.fileh.root,
        #                     "quadprecision")
        # self.assertTrue(isinstance(ds, UnImplemented))

        # NOTE: it would be nice to have some sort of message that warns
        #       against the potential precision loss: the quad-precision
        #       dataset actually uses 128 bits for each element, not just
        #       80 bits (longdouble)
        ds = self.fileh.root.quadprecision
        self.assertEqual(ds.dtype, "longdouble")


class AtomTestCase(common.PyTablesTestCase):
    def test_init_parameters_01(self):
        atom1 = StringAtom(itemsize=12)
        atom2 = atom1.copy()
        self.assertEqual(atom1, atom2)
        self.assertEqual(str(atom1), str(atom2))
        self.assertFalse(atom1 is atom2)

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
        atom1 = Atom.from_dtype(numpy.dtype('Float64'))
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


#----------------------------------------------------------------------

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
    unittest.main(defaultTest='suite')
