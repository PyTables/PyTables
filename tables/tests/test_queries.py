"""
Test module for queries on datasets
===================================

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2006-10-19
:License:  BSD
:Revision: $Id$
"""

import re
import new
import unittest

import numpy

import tables
import tables.tests.common as tests
from tables.tests.common import verbosePrint as vprint


# Data parameters
# ---------------
row_period = 50
"""Maximum number of unique rows before they start cycling."""
md_shape = (2, 2)
"""Shape of multidimensional fields."""

_maxnvalue = row_period + numpy.product(md_shape) - 1
_strlen = int(numpy.log10(_maxnvalue-1)) + 1

str_format = '%%0%dd' % _strlen
"""Format of string values."""


# Type information
# ----------------
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


# Table description
# -----------------
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


# Table data
# ----------
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


# Base test cases
# ---------------
class BaseTableQueryTestCase(tests.TempFileMixin, tests.PyTablesTestCase):

    """
    Base test case for querying tables.

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

    colNotIndexable_re = re.compile(r"\bcan not be indexed\b")
    condNotBoolean_re = re.compile(r"\bdoes not have a boolean type\b")

    def createIndexes(self, colname, ncolname):
        if not self.indexed:
            return
        try:
            vprint("* Indexing ``%s`` columns..." % colname, nonl=True)
            for acolname in [colname, ncolname]:
                acolumn = self.table.colinstances[acolname]
                acolumn.createIndex(optlevel=self.optlevel, testmode=True)
            vprint("ok.")
        except TypeError, te:
            if self.colNotIndexable_re.search(str(te)):
                vprint("can not be indexed.")
                raise tests.SkipTest  # can't be indexed, nothing new to test
            raise
        except NotImplementedError:
            vprint("not supported yet.")
            raise tests.SkipTest  # column does not support indexing yet

    def setUp(self):
        super(BaseTableQueryTestCase, self).setUp()
        self.table = table = self.h5file.createTable(
            '/', 'test', self.tableDescription, expectedrows=self.nrows )
        fill_table(table, self.shape, self.nrows)


class ScalarTableMixin:
    tableDescription = TableDescription
    shape = ()

class MDTableMixin:
    tableDescription = MDTableDescription
    shape = md_shape


# Test cases on query data
# ------------------------
operators = [None, '<', '==', '!=', ('<', '<=')]
"""Comparison operators to check with different types."""
if tests.heavy:
    operators += ['~', '<=', '>=', '>', ('>', '>=')]
left_bound = row_period / 4
"""Operand of left side operator in comparisons with operator pairs."""
right_bound = row_period * 3 / 4
"""Operand of right side operator in comparisons with operator pairs."""
extra_conditions = ['', '& ((cExtra+1) > 0)', '| ((cExtra+1) > 0)']
"""Extra conditions to append to comparison conditions."""

class TableDataTestCase(BaseTableQueryTestCase):
    """Base test case for querying table data."""

def create_test_method(ptype, op, extracond):
    dtype = dtype_from_ptype[ptype]

    # Compute the value of bounds.
    condvars = { 'bound': right_bound,
                 'lbound': left_bound,
                 'rbound': right_bound }
    for (bname, bvalue) in condvars.items():
        if ptype == 'String':
            bvalue = str_format % bvalue
        bvalue = nxtype_from_ptype[ptype](bvalue)
        condvars[bname] = bvalue

    # Compute the name of columns.
    colname = 'c%s' % ptype
    ncolname = 'cNested/%s' % colname

    # Compute the query condition.
    if not op:  # as is
        cond = colname
    elif op == '~':  # unary
        cond = '~(%s)' % colname
    elif op == '<':  # binary variable-constant
        cond = '%s %s %s' % (colname, op, repr(condvars['bound']))
    elif type(op) is tuple: # double binary variable-constant
        cond = ( '(lbound %s %s) & (%s %s rbound)'
                 % (op[0], colname, colname, op[1]) )
    else:  # binary variable-variable
        cond = '%s %s bound' % (colname, op)
    if extracond:
        cond = '(%s) %s' % (cond, extracond)

    def test_method(self):
        vprint("* Condition is ``%s``." % cond)
        # Replace bitwise operators with their logical counterparts.
        pycond = cond
        for (ptop, pyop) in [('&', 'and'), ('|', 'or'), ('~', 'not')]:
            pycond = pycond.replace(ptop, pyop)
        pycond = compile(pycond, '<string>', 'eval')

        table = self.table
        self.createIndexes(colname, ncolname)

        rownos, fvalues = None, None
        # Test that both simple and nested columns work as expected.
        # Knowing how the table is filled, results must be the same.
        for acolname in [colname, ncolname]:
            # First the reference Python version.
            pyrownos, pyfvalues, pyvars = [], [], condvars.copy()
            for row in table:
                pyvars[colname] = row[acolname]
                pyvars['cExtra'] = row['cExtra']
                try:
                    isvalidrow = eval(pycond, {}, pyvars)
                except TypeError:
                    vprint("* Python type does not support the operation.")
                    raise tests.SkipTest
                if isvalidrow:
                    pyrownos.append(row.nrow)
                    pyfvalues.append(row[acolname])
            pyrownos = numpy.array(pyrownos)  # row numbers already sorted
            pyfvalues = numpy.array(pyfvalues, dtype=dtype)
            pyfvalues.sort()
            vprint( "* %d rows selected by Python from ``%s``."
                    % (len(pyrownos), acolname) )
            if rownos is None:
                rownos = pyrownos  # initialise reference results
                fvalues = pyfvalues
            else:
                self.assert_(numpy.all(pyrownos == rownos))  # check
                self.assert_(numpy.all(pyfvalues == fvalues))

            # Then the in-kernel or indexed version.
            ptvars = condvars.copy()
            ptvars[colname] = table.colinstances[acolname]
            ptvars['cExtra'] = table.colinstances['cExtra']
            try:
                isidxq = table.willQueryUseIndexing(cond, ptvars)
                ptrownos = table.getWhereList(cond, condvars, sort=True)
                ptfvalues = table.readWhere(cond, condvars, field=acolname)
            except TypeError, te:
                if self.condNotBoolean_re.search(str(te)):
                    vprint("* Condition is not boolean.")
                    raise tests.SkipTest
                raise
            except NotImplementedError:
                vprint("* PyTables type does not support the operation.")
                raise tests.SkipTest
            ptfvalues.sort()  # row numbers already sorted
            vprint( "* %d rows selected by PyTables from ``%s``"
                    % (len(ptrownos), acolname), nonl=True )
            vprint("(indexing: %s)." % ["no", "yes"][bool(isidxq)])
            self.assert_(numpy.all(ptrownos == rownos))
            self.assert_(numpy.all(ptfvalues == fvalues))

    test_method.__doc__ = "Testing ``%s``." % cond
    return test_method

# Create individual tests.  You may restrict which tests are generated
# by replacing the sequences in the ``for`` statements.  For instance:
testn = 0
for ptype in ptype_info:  # for ptype in ['String']:
    for op in operators:  # for op in ['!=']:
        for extracond in extra_conditions:  # for extracond in ['']:
            tmethod = create_test_method(ptype, op, extracond)
            tmethod.__name__ = 'test_a%04d' % testn
            ptmethod = tests.pyTablesTest(tmethod)
            imethod = new.instancemethod(ptmethod, None, TableDataTestCase)
            setattr(TableDataTestCase, tmethod.__name__, imethod)
            testn += 1


# Base classes for non-indexed queries.
NX_BLOCK_SIZE1 = 128  # from ``interpreter.c`` in Numexpr
NX_BLOCK_SIZE2 = 8  # from ``interpreter.c`` in Numexpr

class SmallNITableMixin:
    nrows = row_period * 2
    assert NX_BLOCK_SIZE2 < nrows < NX_BLOCK_SIZE1
    assert nrows % NX_BLOCK_SIZE2 != 0  # to have some residual rows
class BigNITableMixin:
    nrows = row_period * 3
    assert nrows > NX_BLOCK_SIZE1 + NX_BLOCK_SIZE2
    assert nrows % NX_BLOCK_SIZE1 != 0
    assert nrows % NX_BLOCK_SIZE2 != 0  # to have some residual rows

# Parameters for non-indexed queries.
table_sizes = ['Small']
if tests.heavy:
    table_sizes += ['Big']
table_ndims = ['Scalar']  # to enable multidimensional testing, include 'MD'

# Non-indexed queries: ``[SB][SM]TDTestCase``.
def niclassdata():
    for size in table_sizes:
        for ndim in table_ndims:
            classname = '%s%sTDTestCase' % (size[0], ndim[0])
            cbasenames = ( '%sNITableMixin' % size, '%sTableMixin' % ndim,
                           'TableDataTestCase' )
            yield (classname, cbasenames, {})


# Base classes for indexed queries.
class SmallITableMixin:
    nrows = 50
class MediumITableMixin:
    nrows = 100
class BigITableMixin:
    nrows = 500

# Parameters for indexed queries.
itable_sizes = ['Small']
itable_optvalues = [0, 2, 3]
if tests.heavy:
    itable_sizes += ['Medium', 'Big']
    itable_optvalues += [7, 9]

# Indexed queries: ``[SMB]I[02379]TDTestCase``.
def iclassdata():
    for size in itable_sizes:
        for optlevel in itable_optvalues:
            classname = '%sI%dTDTestCase' % (size[0], optlevel)
            cbasenames = ( '%sITableMixin' % size, 'ScalarTableMixin',
                           'TableDataTestCase' )
            yield ( classname, cbasenames,
                    {'optlevel': optlevel, 'indexed': True} )


# Create test classes.
for cdatafunc in [niclassdata, iclassdata]:
    for (cname, cbasenames, cdict) in cdatafunc():
        cbases = tuple(eval(cbase) for cbase in cbasenames)
        class_ = new.classobj(cname, cbases, cdict)
        exec '%s = class_' % cname


# Test cases on query usage
# -------------------------
class BaseTableUsageTestCase(BaseTableQueryTestCase):
    nrows = row_period

_gvar = None
"""Use this when a global variable is needed."""

class ScalarTableUsageTestCase(ScalarTableMixin, BaseTableUsageTestCase):

    """
    Test case for query usage on scalar tables.

    This also tests for most usage errors and situations.
    """

    def test_empty_condition(self):
        """Using an empty condition."""
        self.assertRaises(SyntaxError, self.table.where, '')

    def test_syntax_error(self):
        """Using a condition with a syntax error."""
        self.assertRaises(SyntaxError, self.table.where, 'foo bar')

    def test_unsupported_object(self):
        """Using a condition with an unsupported object."""
        self.assertRaises(TypeError, self.table.where, '[]')
        self.assertRaises(TypeError, self.table.where, 'obj', {'obj': {}})
        self.assertRaises(TypeError, self.table.where, 'cBool < []')

    def test_unsupported_syntax(self):
        """Using a condition with unsupported syntax."""
        self.assertRaises(TypeError, self.table.where, 'cBool[0]')
        self.assertRaises(TypeError, self.table.where, 'cBool()')
        self.assertRaises(NameError, self.table.where, 'cBool.__init__')

    def test_no_column(self):
        """Using a condition with no participating columns."""
        self.assertRaises(ValueError, self.table.where, 'True')

    def test_foreign_column(self):
        """Using a condition with a column from other table."""
        table2 = self.h5file.createTable('/', 'other', self.tableDescription)
        self.assertRaises( ValueError, self.table.where,
                           'cInt32_a + cInt32_b > 0',
                           { 'cInt32_a': self.table.cols.cInt32,
                             'cInt32_b': table2.cols.cInt32 } )

    def test_unsupported_op(self):
        """Using a condition with unsupported operations on types."""
        NIE = NotImplementedError
        self.assertRaises(NIE, self.table.where, 'cComplex64 > 0j')
        self.assertRaises(NIE, self.table.where, 'cString + "a" > "abc"')

    def test_not_boolean(self):
        """Using a non-boolean condition."""
        self.assertRaises(TypeError, self.table.where, 'cInt32')

    def test_nested_col(self):
        """Using a condition with nested columns."""
        self.assertRaises(TypeError, self.table.where, 'cNested')

    def test_implicit_col(self):
        """Using implicit column names in conditions."""
        # If implicit columns didn't work, a ``NameError`` would be raised.
        self.assertRaises(TypeError, self.table.where, 'cInt32')
        # If overriding didn't work, no exception would be raised.
        self.assertRaises( TypeError, self.table.where,
                           'cBool', {'cBool': self.table.cols.cInt32} )

    def test_condition_vars(self):
        """Using condition variables in conditions."""

        # If condition variables didn't work, a ``NameError`` would be raised.
        self.assertRaises( NotImplementedError, self.table.where,
                           'cString > bound', {'bound': 0})

        def where_with_locals():
            bound = 'foo'  # this wouldn't cause an error
            self.table.where('cString > bound', {'bound': 0})
        self.assertRaises(NotImplementedError, where_with_locals)

        def where_with_globals():
            global _gvar
            _gvar = 'foo'  # this wouldn't cause an error
            try:
                self.table.where('cString > _gvar', {'_gvar': 0})
            finally:
                del _gvar  # to keep global namespace clean
        self.assertRaises(NotImplementedError, where_with_globals)

    def test_scopes(self):
        """Looking up different scopes for variables."""

        # Make sure the variable is not implicit.
        self.assertRaises(NameError, self.table.where, 'col')

        # First scope: dictionary of condition variables.
        self.assertRaises( TypeError, self.table.where,
                           'col', {'col': self.table.cols.cInt32} )

        # Second scope: local variables.
        def where_whith_locals():
            col = self.table.cols.cInt32
            self.table.where('col')
        self.assertRaises(TypeError, where_whith_locals)

        # Third scope: global variables.
        def where_with_globals():
            global _gvar
            _gvar = self.table.cols.cInt32
            try:
                self.table.where('_gvar')
            finally:
                del _gvar  # to keep global namespace clean
        self.assertRaises(TypeError, where_with_globals)

class MDTableUsageTestCase(MDTableMixin, BaseTableUsageTestCase):

    """Test case for query usage on multidimensional tables."""

    def test(self):
        """Using a condition on a multidimensional table."""
        # Easy: queries on multidimensional tables are not implemented yet!
        self.assertRaises(NotImplementedError, self.table.where, 'cBool')

class IndexedTableUsageTestCase(ScalarTableMixin, BaseTableUsageTestCase):

    """
    Test case for query usage on indexed tables.

    Indexing could be used in more cases, but it is expected to kick
    in at least in the cases tested here.
    """
    nrows = 50
    indexed = True

    conditions = []
    """List of conditions to be tested."""

    # Add boolean conditions.
    for _cbase in ['cBool', '~cBool']:
        conditions.append(_cbase)
        conditions.append('(%s) & (cExtra > 0)' % _cbase)
        conditions.append('(cExtra > 0) & (%s)' % _cbase)
    # Add simple numeric conditions.
    for _usevar in [False, True]:
        for _condt in ['cInt32 %(o)s %(v)s', '%(v)s %(o)s cInt32']:
            for _op in ['<', '<=', '==', '>=', '>']:
                if _usevar:
                    _cdict = {'o': _op, 'v': 'var'}
                else:
                    _cdict = {'o': _op, 'v': 0}
                conditions.append(_condt % _cdict)
    conditions.append('(cInt32 > 0) & (cExtra > 0)')
    conditions.append('(cExtra > 0) & (cInt32 > 0)')
    # Add double numeric conditions.
    for _cbase in ['(0<cInt32) & (cInt32<10)', '(10>cInt32) & (cInt32>0)']:
        conditions.append(_cbase)
        conditions.append('(%s) & (cExtra > 0)' % _cbase)
        conditions.append('(cExtra > 0) & (%s)' % _cbase)

    def setUp(self):
        super(IndexedTableUsageTestCase, self).setUp()
        self.table.cols.cBool.createIndex(testmode=True)
        self.table.cols.cInt32.createIndex(testmode=True)

    def test(self):
        """Using indexing in some queries."""
        willQueryUseIndexing = self.table.willQueryUseIndexing
        for condition in self.conditions:
            self.assert_( willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            tests.verbosePrint(
                "* Query with condition ``%s`` will use indexing."
                % condition )


# Main part
# ---------
def suite():
    """Return a test suite consisting of all the test cases in the module."""

    testSuite = unittest.TestSuite()

    niter = 1
    for i in range(niter):
        # Tests on query data.
        for cdatafunc in [niclassdata, iclassdata]:
            for cdata in cdatafunc():
                cname = cdata[0]
                testSuite.addTest(unittest.makeSuite(eval(cname)))
        # Tests on query usage.
        testSuite.addTest(unittest.makeSuite(ScalarTableUsageTestCase))
        testSuite.addTest(unittest.makeSuite(MDTableUsageTestCase))
        testSuite.addTest(unittest.makeSuite(IndexedTableUsageTestCase))

    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
