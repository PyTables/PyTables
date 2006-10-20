"""
Test module for queries on datasets
===================================

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2006-10-19
:License:  BSD
:Revision: $Id$
"""

import new
import unittest

import numpy

import tables
import tables.tests.common as tests


row_period = 100
"""Maximum number of unique rows before they start cycling."""
md_shape = (2, 2)
"""Shape of multidimensional fields."""

_maxnvalue = row_period + numpy.product(md_shape) - 1
_strlen = int(numpy.log10(_maxnvalue-1)) + 1

_uint32nxtype = type(numpy.uint32(0).tolist())
"""Numexpr type of UInt32 numbers on this platform."""

type_info = {
    'Bool': (numpy.bool_, bool),
    'Int8': (numpy.int8, int), 'UInt8': (numpy.uint8, int),
    'Int16': (numpy.int16, int), 'UInt16': (numpy.uint16, int),
    'Int32': (numpy.int32, int), 'UInt32': (numpy.uint32, _uint32nxtype),
    'Int64': (numpy.int64, long), 'UInt64': (numpy.uint64, long),
    'Float32': (numpy.float32, float), 'Float64': (numpy.float32, float),
    'Complex32': (numpy.complex64, complex),
    'Complex64': (numpy.complex128, complex),
    'Time32': (numpy.int32, int), 'Time64': (numpy.float64, float),
    'Enum': (numpy.uint8, int),  # just for these tests
    'String': (numpy.dtype('S%s' % _strlen), str) }  # just for these tests
"""NumPy and Numexpr type for each PyTables type that will be tested."""

dtype_from_ptype = dict(
    (dtype, info[0]) for (dtype, info) in type_info.iteritems() )
"""Maps PyTables types to NumPy data types."""
nxtype_from_ptype = dict(
    (dtype, info[1]) for (dtype, info) in type_info.iteritems() )
"""Maps PyTables types to Numexpr data types."""

enum = tables.Enum(dict(('n%d' % i, i) for i in range(_maxnvalue)))
"""Enumerated type to be used in tests."""


def append_columns(classdict, shape=()):
    """
    Append a ``Col`` of each PyTables data type to the `classdict`.

    A column of a certain TYPE gets called ``cTYPE``.  The number of
    added columns is returned.
    """
    for (itype, ptype) in enumerate(sorted(type_info.iterkeys())):
        colpos = itype + 1
        colname = 'c%s' % ptype
        colclass = getattr(tables, '%sCol' % ptype)
        colargs, colkwargs = [], {'shape': shape, 'pos': colpos}
        if ptype == 'Enum':
            colargs = [enum, enum(0), dtype_from_ptype[ptype]] + colargs
        elif ptype == 'String':
            colkwargs.update({'length': dtype_from_ptype[ptype].itemsize})
        classdict[colname] = colclass(*colargs, **colkwargs)
    ncols = colpos
    return ncols

def nested_description(classname, pos, shape=()):
    """
    Return a nested column description with all PyTables data types.

    A column of a certain TYPE gets called ``cTYPE``.  The nested
    column will be placed in the position indicated by `pos`.
    """
    classdict = {}
    append_columns(classdict, shape=shape)
    classdict['_v_pos'] = pos
    return new.classobj(classname, (tables.IsDescription,), classdict)

def table_description(classname, nclassname, shape=()):
    """
    Return a table description for testing queries.

    The description consists of all PyTables data types, both in the
    top level and in the ``cNested`` nested column.  A column of a
    certain TYPE gets called ``cTYPE``.  An extra integer column
    ``cExtra`` is also provided.  If a `shape` is given, it will be
    used for all columns.
    """
    classdict = {}
    colpos = append_columns(classdict, shape)

    ndescr = nested_description(nclassname, colpos, shape=shape)
    classdict['cNested'] = ndescr
    colpos += 1

    extracol = tables.IntCol(shape=shape, pos=colpos)
    classdict['cExtra'] = extracol
    colpos += 1

    return new.classobj(classname, (tables.IsDescription,), classdict)

TableDescription = table_description(
    'TableDescription', 'NestedDescription' )
"""Unidimensional table description for testing queries."""

MDTableDescription = table_description(
    'MDTableDescription', 'MDNestedDescription', shape=md_shape )
"""Multidimensional table description for testing queries."""


def fill_table(table, shape, nrows):
    """
    Fill the given `table` with `nrows` rows of data.

    Values in the i-th row (where 0 <= i < `row_period`) for a
    multidimensional field with M elements span from i to i+M-1.  For
    subsequent rows, values repeat cyclically.

    The same goes for the ``cExtra`` column, but values repeat every
    `row_period`/2 rows.
    """
    size = int(numpy.product(shape))

    row, value = table.row, 0
    for nrow in xrange(nrows):
        data = numpy.arange(value, value + size).reshape(shape)
        for (ptype, dtype) in dtype_from_ptype.iteritems():
            colname = 'c%s' % ptype
            ncolname = 'cNested/%s' % colname
            if ptype == 'Bool':
                coldata = data > (row_period / 2)
            elif ptype == 'String':
                sformat = '%%0%dd' % dtype.itemsize
                sdata = [sformat % x for x in range(value, value + size)]
                coldata = numpy.array(sdata, dtype=dtype).reshape(shape)
            else:
                coldata = numpy.asarray(data, dtype=dtype)
            row[ncolname] = row[colname] = coldata
            row['cExtra'] = data % (row_period / 2)
        row.append()
        value += 1
        if value == row_period:
            value = 0
    table.flush()


class XXXTestCase(tests.TempFileMixin, tests.PyTablesTestCase):
    def setUp(self):
        super(XXXTestCase, self).setUp()
        t1 = self.h5file.createTable('/', 'test', TableDescription)
        fill_table(t1, (), 500)
        t2 = self.h5file.createTable('/', 'mdtest', MDTableDescription)
        fill_table(t2, md_shape, 500)

    def testXXX(self):
        print self.h5file.root.test.cols.cString[:]
        print self.h5file.root.mdtest.cols.cString[:]

if __name__ == '__main__':
    unittest.main()
