"""
Test module for enumerated types under PyTables
===============================================

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2005-05-09
:License:  BSD
:Revision: $Id$
"""

import unittest
import operator

import tables
import common
from common import verbose


class CreateColTestCase(common.PyTablesTestCase):

    """Test creating enumerated column descriptions."""

    def _createCol(self, enum, dflt, dtype='UInt32', shape=1):
        """Create and check an enumerated column description."""

        enumcol = tables.EnumCol(enum, dflt, dtype=dtype, shape=shape)
        sameEnum = tables.Enum(enum)
        self.assertEqual(enumcol.stype, 'Enum')
        self.assertEqual(enumcol.type, dtype)
        # To avoid 'LongInt' vs 'Int' issues
        #self.assertEqual(enumcol.dflt, sameEnum[dflt])
        self.assertEqual(int(enumcol.dflt), int(sameEnum[dflt]))
        self.assertEqual(enumcol.shape, shape)
        self.assertEqual(enumcol.enum, sameEnum)


    def test00a_validFromEnum(self):
        """Describing an enumerated column from an enumeration."""

        self._verboseHeader()
        colors = tables.Enum(['red', 'green', 'blue'])
        self._createCol(colors, 'red')

    def test00b_validFromDict(self):
        """Describing an enumerated column from a dictionary."""

        self._verboseHeader()
        colors = {'red': 4, 'green': 2, 'blue': 1}
        self._createCol(colors, 'red')


    def test00c_validFromList(self):
        """Describing an enumerated column from a list."""

        self._verboseHeader()
        colors = ['red', 'green', 'blue']
        self._createCol(colors, 'red')


    def test00d_invalidFromType(self):
        """Describing an enumerated column from an invalid object."""

        self._verboseHeader()
        colors = 123
        self.assertRaises(TypeError, self._createCol, colors, 'red')


    def test01a_invalidDfltType(self):
        """Describing an enumerated column with an invalid default object."""

        self._verboseHeader()
        colors = tables.Enum({'red': 4, 'green': 2, 'blue': 1})
        self.assertRaises(TypeError, self._createCol, colors, colors.red)


    def test01b_invalidDfltName(self):
        """Describing an enumerated column with an invalid default name."""

        self._verboseHeader()
        colors = {'red': 4, 'green': 2, 'blue': 1}
        self.assertRaises(KeyError, self._createCol, colors, 'black')


    def test02a_validDtypeBroader(self):
        """Describing an enumerated column with a broader type."""

        self._verboseHeader()
        colors = {'red': 4, 'green': 2, 'blue': 1}
        self._createCol(colors, 'red', dtype='Int64')


    def test02b_invalidDtypeTooNarrow(self):
        """Describing an enumerated column with a too narrow type."""

        self._verboseHeader()
        colors = ['e%d' % i for i in range(300)]
        self.assertRaises(TypeError, self._createCol, colors, 'e0', 'UInt8')


    def test03a_validShapeMD(self):
        """Describing an enumerated column with multidimensional shape."""

        self._verboseHeader()
        colors = ['red', 'green', 'blue']
        self._createCol(colors, 'red', shape=(2,))


    def test04a_validReprEnum(self):
        """Checking the string representation of an enumeration."""

        self._verboseHeader()
        colors = tables.Enum(['red', 'green', 'blue'])
        enumcol = tables.EnumCol(colors, 'red', dtype='UInt32', shape=1)
        assert repr(enumcol) == \
"""EnumCol(Enum({'blue': 2, 'green': 1, 'red': 0}), 'red', dtype='UInt32', shape=1, pos=None, indexed=False)"""


    def test99a_nonIntEnum(self):
        """Describing an enumerated column of floats (not implemented)."""

        self._verboseHeader()
        colors = {'red': 1.0}
        self.assertRaises(NotImplementedError, self._createCol, colors, 'red')


    def test99b_nonIntDtype(self):
        """Describing an enumerated column encoded as floats (not implemented)."""

        self._verboseHeader()
        colors = ['red', 'green', 'blue']
        self.assertRaises(
            NotImplementedError, self._createCol, colors, 'red', 'Float64')


    def test99b_nonScalarEnum(self):
        """Describing an enumerated column of non-scalars (not implemented)."""

        self._verboseHeader()
        colors = {'red': (1, 2, 3)}
        self.assertRaises(NotImplementedError, self._createCol, colors, 'red')



class CreateAtomTestCase(common.PyTablesTestCase):

    """Test creating enumerated atoms."""

    def _createAtom(self, enum, dtype='UInt32', shape=1):
        """Create and check an enumerated atom."""

        enumatom = tables.EnumAtom(enum, dtype=dtype, shape=shape)
        sameEnum = tables.Enum(enum)
        self.assertEqual(enumatom.stype, 'Enum')
        self.assertEqual(enumatom.type, dtype)
        self.assertEqual(enumatom.shape, shape)
        self.assertEqual(enumatom.enum, sameEnum)


    def test00a_validFromEnum(self):
        """Describing an enumerated atom from an enumeration."""

        self._verboseHeader()
        colors = tables.Enum(['red', 'green', 'blue'])
        self._createAtom(colors)


    def test00b_validFromDict(self):
        """Describing an enumerated atom from a dictionary."""

        self._verboseHeader()
        colors = {'red': 4, 'green': 2, 'blue': 1}
        self._createAtom(colors)


    def test00c_validFromList(self):
        """Describing an enumerated atom from a list."""

        self._verboseHeader()
        colors = ['red', 'green', 'blue']
        self._createAtom(colors)


    def test00d_invalidFromType(self):
        """Describing an enumerated atom from an invalid object."""

        self._verboseHeader()
        colors = 123
        self.assertRaises(TypeError, self._createAtom, colors)


    def test02a_validDtypeBroader(self):
        """Describing an enumerated atom with a broader type."""

        self._verboseHeader()
        colors = {'red': 4, 'green': 2, 'blue': 1}
        self._createAtom(colors, dtype='Int64')


    def test02b_invalidDtypeTooNarrow(self):
        """Describing an enumerated atom with a too narrow type."""

        self._verboseHeader()
        colors = ['e%d' % i for i in range(300)]
        self.assertRaises(TypeError, self._createAtom, colors, 'UInt8')


    def test03a_validShapeMD(self):
        """Describing an enumerated atom with multidimensional shape."""

        self._verboseHeader()
        colors = ['red', 'green', 'blue']
        self._createAtom(colors, shape=(2,))


    def test99a_nonIntEnum(self):
        """Describing an enumerated atom of floats (not implemented)."""

        self._verboseHeader()
        colors = {'red': 1.0}
        self.assertRaises(NotImplementedError, self._createAtom, colors)


    def test99b_nonIntDtype(self):
        """Describing an enumerated atom encoded as a float (not implemented)."""

        self._verboseHeader()
        colors = ['red', 'green', 'blue']
        self.assertRaises(
            NotImplementedError, self._createAtom, colors, 'Float64')


    def test99b_nonScalarEnum(self):
        """Describing an enumerated atom of non-scalars (not implemented)."""

        self._verboseHeader()
        colors = {'red': (1, 2, 3)}
        self.assertRaises(NotImplementedError, self._createAtom, colors)



class EnumTableTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test tables with enumerated columns."""

    enum = tables.Enum({'red': 4, 'green': 2, 'blue': 1, 'black': 0})
    defaultName = 'black'
    valueInEnum = enum.red
    valueOutOfEnum = 1234
    enumType = 'UInt16'


    def _description(self, shape=1):
        class TestDescription(tables.IsDescription):
            rid = tables.IntCol(pos=0)
            rcolor = tables.EnumCol(
                self.enum, self.defaultName,
                dtype=self.enumType, shape=shape, pos=1)

        return TestDescription


    def test00a_reopen(self):
        """Reopening a file with tables using enumerated data."""

        self._verboseHeader()
        self.h5file.createTable(
            '/', 'test', self._description(), title=self._getMethodName())

        self._reopen()

        self.assertEqual(
            self.h5file.root.test.getEnum('rcolor'), self.enum,
            "Enumerated type was not restored correctly from disk.")


    def test00b_reopenMD(self):
        """
        Reopening a file with tables using enumerated multi-dimensional
        data.
        """

        self._verboseHeader()
        self.h5file.createTable(
            '/', 'test', self._description((2,)), title=self._getMethodName())

        self._reopen()

        self.assertEqual(
            self.h5file.root.test.getEnum('rcolor'), self.enum,
            "Enumerated type was not restored correctly from disk.")


    def test01_rowAppend(self):
        """Appending enumerated values using ``row.append()``."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._description(), title=self._getMethodName())

        appended = [
            (10, self.valueInEnum),
            (20, self.valueOutOfEnum)]

        row = tbl.row

        row['rid'] = appended[0][0]
        row['rcolor'] = appended[0][1]
        row.append()

        row['rid'] = appended[1][0]
        self.assertRaises(
            ValueError, operator.setitem, row, 'rcolor', appended[1][1])

        tbl.flush()
        read = tbl.read(flavor='python')
        if verbose:
            print "appended value:", appended[:-1]
            print "read value:", read
        self.assertEqual(
            appended[:-1], read, "Written and read values differ.")


    def test02_append(self):
        """Appending enumerated values using ``table.append()``."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._description(), title=self._getMethodName())

        appended = [
            (10, self.valueInEnum),
            (20, self.valueOutOfEnum)]

        tbl.append(appended)
        tbl.flush()
        read = tbl.read(flavor='python')
        if verbose:
            print "appended value:", appended
            print "read value:", read
        self.assertEqual(appended, read, "Written and read values differ.")


    def test03_setitem(self):
        """Changing enumerated values using ``table.__setitem__()``."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._description(), title=self._getMethodName())

        appended = [
            (10, self.valueInEnum),
            (20, self.valueInEnum)]
        tbl.append(appended)

        written = [
            (10, self.valueInEnum),
            (20, self.valueOutOfEnum)]
        tbl[:] = written
        read = tbl.read(flavor='python')
        if verbose:
            print "written value:", written
            print "read value:", read
        self.assertEqual(written, read, "Written and read values differ.")


    def test04_multidim(self):
        """Appending multi-dimensional enumerated data."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._description((2,)), title=self._getMethodName())

        appended = (
            (10, (self.valueInEnum, self.valueOutOfEnum)),
            (20, (self.valueInEnum, self.valueOutOfEnum)))

        row = tbl.row
        row['rid'] = appended[0][0]
        self.assertRaises(
            ValueError, operator.setitem, row, 'rcolor', appended[0][1])

        tbl.append(appended)
        tbl.flush()
        read = tbl.read(flavor='python')
        for i in range(len(appended)):
            self.assertEqual(appended[i][0], read[i][0],
                             "Written and read values differ.")
            self.assertEqual(appended[i][1][0], read[i][1][0],
                             "Written and read values differ.")
            self.assertEqual(appended[i][1][1], read[i][1][1],
                             "Written and read values differ.")


    def test05_where(self):
        """Searching enumerated data."""

        self._verboseHeader()
        tbl = self.h5file.createTable(
            '/', 'test', self._description(), title=self._getMethodName())

        appended = [
            (10, self.valueInEnum),
            (20, self.valueInEnum),
            (30, self.valueOutOfEnum)]
        tbl.append(appended)
        tbl.flush()

        #searched = tuple(tbl.where(tbl.cols.rcolor == self.valueInEnum))
        searched = [(row["rid"], row["rcolor"]) for row \
                    in tbl.where(tbl.cols.rcolor == self.valueInEnum)]
        common.verbosePrint(
            "* ``valueInEnum``: %s\n"
            "* ``rcolor`` column: ``%s``\n"
            "* ``searched``: %s\n"
            "* Should look like: %s\n"
            % (self.valueInEnum, tbl.cols.rcolor, searched, appended[:-1]))
        self.assertEqual(
            searched, appended[:-1], "Search returned incorrect results.")



class EnumEArrayTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test extendable arrays of enumerated values."""

    enum = tables.Enum({'red': 4, 'green': 2, 'blue': 1, 'black': 0})
    valueInEnum = enum.red
    valueOutOfEnum = 1234
    enumType = 'UInt16'


    def _atom(self, shape=(0,)):
        return tables.EnumAtom(
            self.enum, dtype=self.enumType, shape=shape, flavor='python')


    def test00a_reopen(self):
        """Reopening a file with extendable arrays using enumerated data."""

        self._verboseHeader()
        self.h5file.createEArray(
            '/', 'test', self._atom(), title=self._getMethodName())

        self._reopen()

        self.assertEqual(
            self.h5file.root.test.getEnum(), self.enum,
            "Enumerated type was not restored correctly from disk.")


    def test00b_reopenMD(self):
        """
        Reopening a file with extendable arrays using enumerated
        multi-dimensional data.
        """

        self._verboseHeader()
        self.h5file.createEArray(
            '/', 'test', self._atom((0, 2)), title=self._getMethodName())

        self._reopen()

        self.assertEqual(
            self.h5file.root.test.getEnum(), self.enum,
            "Enumerated type was not restored correctly from disk.")


    def test01_append(self):
        """Appending scalar elements of enumerated values."""

        self._verboseHeader()
        earr = self.h5file.createEArray(
            '/', 'test', self._atom(), title=self._getMethodName())

        appended = [self.valueInEnum, self.valueOutOfEnum]

        earr.append(appended)
        earr.flush()
        read = earr.read()
        self.assertEqual(appended, read, "Written and read values differ.")


    def test02_appendMD(self):
        """Appending multi-dimensional elements of enumerated values."""

        self._verboseHeader()
        earr = self.h5file.createEArray(
            '/', 'test', self._atom((0, 2)), title=self._getMethodName())

        appended = [
            [self.valueInEnum, self.valueOutOfEnum],
            [self.valueInEnum, self.valueOutOfEnum]]

        earr.append(appended)
        earr.flush()
        read = earr.read()
        self.assertEqual(appended, read, "Written and read values differ.")


    def test03_setitem(self):
        """Changing enumerated values using ``earray.__setitem__()``."""

        self._verboseHeader()
        earr = self.h5file.createEArray(
            '/', 'test', self._atom(), title=self._getMethodName())

        appended = (self.valueInEnum, self.valueInEnum)
        earr.append(appended)

        written = [self.valueInEnum, self.valueOutOfEnum]
        earr[:] = written
        read = earr.read()
        self.assertEqual(written, read, "Written and read values differ.")



class EnumVLArrayTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test variable-length arrays of enumerated values."""

    enum = tables.Enum({'red': 4, 'green': 2, 'blue': 1, 'black': 0})
    valueInEnum = enum.red
    valueOutOfEnum = 1234
    enumType = 'UInt16'


    def _atom(self, shape=1):
        return tables.EnumAtom(
            self.enum, dtype=self.enumType, shape=shape, flavor='python')


    def test00a_reopen(self):
        """Reopening a file with variable-length arrays using enumerated data."""

        self._verboseHeader()
        self.h5file.createVLArray(
            '/', 'test', self._atom(), title=self._getMethodName())

        self._reopen()

        self.assertEqual(
            self.h5file.root.test.getEnum(), self.enum,
            "Enumerated type was not restored correctly from disk.")


    def test00b_reopenMD(self):
        """
        Reopening a file with variable-length arrays using enumerated
        multi-dimensional data.
        """

        self._verboseHeader()
        self.h5file.createVLArray(
            '/', 'test', self._atom((2,)), title=self._getMethodName())

        self._reopen()

        self.assertEqual(
            self.h5file.root.test.getEnum(), self.enum,
            "Enumerated type was not restored correctly from disk.")


    def test01_append(self):
        """Appending scalar elements of enumerated values."""

        self._verboseHeader()
        vlarr = self.h5file.createVLArray(
            '/', 'test', self._atom(), title=self._getMethodName())

        appended = [
            [self.valueInEnum,],
            [self.valueInEnum, self.valueOutOfEnum]]

        vlarr.append(appended[0])
        vlarr.append(appended[1])
        vlarr.flush()
        read = vlarr.read()
        if verbose:
            print "appended value:", appended
            print "read value:", read
        self.assertEqual(appended, read, "Written and read values differ.")


    def test02_appendMD(self):
        """Appending multi-dimensional elements of enumerated values."""

        self._verboseHeader()
        vlarr = self.h5file.createVLArray(
            '/', 'test', self._atom((2,)), title=self._getMethodName())

        appended = [
            [[self.valueInEnum, self.valueInEnum],],
            [[self.valueInEnum, self.valueOutOfEnum],
             [self.valueInEnum, self.valueInEnum]]]

        vlarr.append(appended[0])
        vlarr.append(appended[1])
        vlarr.flush()
        read = vlarr.read()
        if verbose:
            print "appended value:", appended
            print "read value:", read
        self.assertEqual(appended, read, "Written and read values differ.")


    def test03_setitem(self):
        """Changing enumerated values using ``vlarray.__setitem__()``."""

        self._verboseHeader()
        vlarr = self.h5file.createVLArray(
            '/', 'test', self._atom(), title=self._getMethodName())

        appended = (self.valueInEnum, self.valueInEnum)
        vlarr.append(appended)

        written = [self.valueInEnum, self.valueOutOfEnum]
        vlarr[0] = written
        read = vlarr.read()
        if verbose:
            print "written value:", written
            print "read value:", read[0]
        self.assertEqual(written, read[0], "Written and read values differ.")



def suite():
    """Return a test suite consisting of all the test cases in the module."""

    # These two are for including Enum's doctests here.
    import doctest
    import tables.enum
    theSuite = unittest.TestSuite()
    niter = 1

    #theSuite.addTest(unittest.makeSuite(EnumTableTestCase))
    for i in range(niter):
        theSuite.addTest(doctest.DocTestSuite(tables.enum))
        theSuite.addTest(unittest.makeSuite(CreateColTestCase))
        theSuite.addTest(unittest.makeSuite(CreateAtomTestCase))
        theSuite.addTest(unittest.makeSuite(EnumTableTestCase))
        theSuite.addTest(unittest.makeSuite(EnumEArrayTestCase))
        theSuite.addTest(unittest.makeSuite(EnumVLArrayTestCase))

    return theSuite



if __name__ == '__main__':
    unittest.main(defaultTest='suite')



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
