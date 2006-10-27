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


row_period = 50
"""Maximum number of unique rows before they start cycling."""
md_shape = (2, 2)
"""Shape of multidimensional fields."""

_maxnvalue = row_period + numpy.product(md_shape) - 1
_strlen = int(numpy.log10(_maxnvalue-1)) + 1

str_format = '%%0%dd' % _strlen
"""Format of string values."""

ptype_info = {
    'Bool': (numpy.bool_, bool),
    'Int8': (numpy.int8, int), 'UInt8': (numpy.uint8, int),
    'Int16': (numpy.int16, int), 'UInt16': (numpy.uint16, int),
    'Int32': (numpy.int32, int), 'UInt32': (numpy.uint32, long),
    'Int64': (numpy.int64, long), 'UInt64': (numpy.uint64, long),
    'Float32': (numpy.float32, float), 'Float64': (numpy.float32, float),
    'Complex32': (numpy.complex64, complex),
    'Complex64': (numpy.complex128, complex),
    'Time32': (numpy.int32, int), 'Time64': (numpy.float64, float),
    'Enum': (numpy.uint8, int),  # just for these tests
    'String': (numpy.dtype('S%s' % _strlen), str) }  # just for these tests
"""NumPy and Numexpr type for each PyTables type that will be tested."""

dtype_from_ptype = dict(
    (dtype, info[0]) for (dtype, info) in ptype_info.iteritems() )
"""Maps PyTables types to NumPy data types."""
nxtype_from_ptype = dict(
    (dtype, info[1]) for (dtype, info) in ptype_info.iteritems() )
"""Maps PyTables types to Numexpr data types."""

heavy_types = ['UInt8', 'Int16', 'UInt16', 'Float32', 'Complex32']
"""Python types to be tested only in heavy mode."""

if not tests.heavy:
    for ptype in heavy_types:
        for tdict in ptype_info, dtype_from_ptype, nxtype_from_ptype:
            del tdict[ptype]

enum = tables.Enum(dict(('n%d' % i, i) for i in range(_maxnvalue)))
"""Enumerated type to be used in tests."""


def append_columns(classdict, shape=()):
    """
    Append a ``Col`` of each PyTables data type to the `classdict`.

    A column of a certain TYPE gets called ``cTYPE``.  The number of
    added columns is returned.
    """
    for (itype, ptype) in enumerate(sorted(ptype_info.iterkeys())):
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


table_data = {}
"""Cached table data for a given shape and number of rows."""
# Data is cached because computing it row by row is quite slow.  Hop!

def fill_table(table, shape, nrows):
    """
    Fill the given `table` with `nrows` rows of data.

    Values in the i-th row (where 0 <= i < `row_period`) for a
    multidimensional field with M elements span from i to i+M-1.  For
    subsequent rows, values repeat cyclically.

    The same goes for the ``cExtra`` column, but values range from
    -`row_period`/2 to +`row_period`/2.
    """
    # Reuse already computed data if possible.
    tdata = table_data.get((shape, nrows))
    if tdata is not None:
        table.append(tdata)
        table.flush()
        return

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
                sdata = [str_format % x for x in range(value, value + size)]
                coldata = numpy.array(sdata, dtype=dtype).reshape(shape)
            else:
                coldata = numpy.asarray(data, dtype=dtype)
            row[ncolname] = row[colname] = coldata
            row['cExtra'] = data - (row_period / 2)
        row.append()
        value += 1
        if value == row_period:
            value = 0
    table.flush()

    # Make computed data reusable.
    tdata = table.read()
    table_data[(shape, nrows)] = tdata


class TableQueryTestCase(tests.TempFileMixin, tests.PyTablesTestCase):

    """
    Test querying table data.

    Sub-classes must define the following attributes:

    ``tableDescription``
        The description of the table to be created.
    ``shape``
        The shape of data fields in the table.
    ``nrows``
        The number of data rows to be generated for the table.

    Sub-classes may redefine the following attributes:

    ``indexed``
        Whether columns shall be indexed, if possible.  Default is not
        to index them.
    ``optlevel``
        The level of optimisation of column indexes.  Default is 0.
    """

    indexed = False
    optlevel = 0

    def createIndexes(self, colname, ncolname):
        if not self.indexed:
            return
        colinsts = self.table.colinstances
        colinsts[colname].createIndex(optlevel=self.optlevel, testmode=True)
        colinsts[ncolname].createIndex(optlevel=self.optlevel, testmode=True)

    def setUp(self):
        super(TableQueryTestCase, self).setUp()
        self.table = table = self.h5file.createTable(
            '/', 'test', self.tableDescription, expectedrows=self.nrows )
        fill_table(table, self.shape, self.nrows)

    ## XXX Need some standard checks on query usage.


operators = ['<', '==', '!=']
if tests.heavy:
    operators += ['<=', '>=', '>']
## XXX Need to check por operator pairs.
left_bound = row_period / 4
right_bound = row_period * 3 / 4

def create_test_method(ptype, op, extracond):
    colname = 'c%s' % ptype
    ncolname = 'cNested/%s' % colname
    cond = '%s %s bound' % (colname, op)
    if extracond:
        cond = '(%s) %s' % (cond, extracond)

    bound = right_bound
    if ptype == 'String':
        bound = str_format % bound
    condvars = {'bound': bound}

    def test_method(self):
        # Replace bitwise operators with their logical counterparts.
        pycond = cond
        for (ptop, pyop) in [('&', 'and'), ('|', 'or'), ('~', 'not')]:
            pycond = pycond.replace(ptop, pyop)
        pycond = compile(pycond, '<string>', 'eval')

        table = self.table

        # Create indexes on the columns to be queried.
        try:
            self.createIndexes(colname, ncolname)
        except (TypeError, NotImplementedError):
            return  # column can't be indexed, nothing new to test

        reflen = None
        ## XXX Better try to check retrieved data with ``readWhere()``.
        ## XXX Better yet, try all query methods in parallel!
        # Test that both simple and nested columns work as expected.
        # Because of the way the table is filled, results are the same.
        for acolname in [colname, ncolname]:
            # First the reference Python version.
            pylen, pyvars = 0, condvars.copy()
            for row in table:
                pyvars[colname] = row[acolname]
                pyvars['cExtra'] = row['cExtra']
                try:
                    isvalidrow = eval(pycond, {}, pyvars)
                except TypeError:
                    return  # type doesn't support operation, skip test
                if isvalidrow:
                    pylen += 1
            if reflen is None:  # initialise reference length
                reflen = pylen
            self.assertEqual(pylen, reflen)

            # Then the in-kernel or indexed version.
            ptvars = condvars.copy()
            ptvars[colname] = table.colinstances[acolname]
            ptvars['cExtra'] = table.colinstances['cExtra']
            ptlen = len([r for r in table.where(cond, ptvars)])
            self.assertEqual(ptlen, reflen)

    test_method.__doc__ = "Testing ``%s``." % cond
    return test_method

testn = 0
for ptype in ptype_info:
    for op in operators:
        for extracond in ['', '& ((cExtra+1) > 0)', '| ((cExtra+1) > 0)']:
            tmethod = create_test_method(ptype, op, extracond)
            tmethod.__name__ = 'test_a%04d' % testn
            dmethod = tests.verboseDecorator(tmethod)
            imethod = new.instancemethod(dmethod, None, TableQueryTestCase)
            setattr(TableQueryTestCase, tmethod.__name__, imethod)
            testn += 1


# Base classes for queries.
NX_BLOCK_SIZE1 = 128  # from ``interpreter.c`` in Numexpr
NX_BLOCK_SIZE2 = 8  # from ``interpreter.c`` in Numexpr

class SmallTableMixin:
    nrows = row_period * 2
    assert NX_BLOCK_SIZE2 < nrows < NX_BLOCK_SIZE1
    assert nrows % NX_BLOCK_SIZE2 != 0  # to have some residual rows
class BigTableMixin:
    nrows = row_period * 3
    assert nrows > NX_BLOCK_SIZE1 + NX_BLOCK_SIZE2
    assert nrows % NX_BLOCK_SIZE1 != 0
    assert nrows % NX_BLOCK_SIZE2 != 0  # to have some residual rows

class ScalarTableMixin:
    tableDescription = TableDescription
    shape = ()
class MDTableMixin:
    tableDescription = MDTableDescription
    shape = md_shape

table_sizes = ['Small']
if tests.heavy:
    table_sizes.append('Big')
table_ndims = ['Scalar']  # to enable multidimensional testing, include 'MD'
table_optvalues = [0, 1, 3]
if tests.heavy:
    table_optvalues += [6, 9]

# Non-indexed queries: ``[SB][SM]TQTestCase``.
def niclassdata():
    for size in table_sizes:
        for ndim in table_ndims:
            classname = '%s%sTQTestCase' % (size[0], ndim[0])
            cbasenames = ( '%sTableMixin' % size, '%sTableMixin' % ndim,
                           'TableQueryTestCase' )
            yield (classname, cbasenames, {})

# Indexed queries: ``[SB]SI[0139]TQTestCase``.
def iclassdata():
    for size in table_sizes:
        for optlevel in table_optvalues:
            classname = '%sSI%dTQTestCase' % (size[0], optlevel)
            cbasenames = ( '%sTableMixin' % size, 'ScalarTableMixin',
                           'TableQueryTestCase' )
            yield ( classname, cbasenames,
                    {'optlevel': optlevel, 'indexed': True} )

# Create test classes.
for cdatafunc in [niclassdata, iclassdata]:
    for (cname, cbasenames, cdict) in cdatafunc():
        cbases = tuple(eval(cbase) for cbase in cbasenames)
        class_ = new.classobj(cname, cbases, cdict)
        exec '%s = class_' % cname


def suite():
    """Return a test suite consisting of all the test cases in the module."""

    testSuite = unittest.TestSuite()

    niter = 1
    for i in range(niter):
        for cdatafunc in [niclassdata, iclassdata]:
            for cdata in cdatafunc():
                cname = cdata[0]
                testSuite.addTest(unittest.makeSuite(eval(cname)))

    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
