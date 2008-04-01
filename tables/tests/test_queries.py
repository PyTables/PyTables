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
from tables.utils import SizeType
from tables.tests import common
from common import verbosePrint as vprint


# Data parameters
# ---------------
row_period = 50
"""Maximum number of unique rows before they start cycling."""
md_shape = (2, 2)
"""Shape of multidimensional fields."""

_maxnvalue = row_period + numpy.prod(md_shape, dtype=SizeType) - 1
_strlen = int(numpy.log10(_maxnvalue-1)) + 1

str_format = '%%0%dd' % _strlen
"""Format of string values."""

small_blocksizes = (300, 60, 20, 5)
#small_blocksizes = (512, 128, 32, 4)   # for manual testing only
"""Sensible parameters for indexing with small blocksizes."""


# Type information
# ----------------
type_info = {
    'bool': (numpy.bool_, bool),
    'int8': (numpy.int8, int), 'uint8': (numpy.uint8, int),
    'int16': (numpy.int16, int), 'uint16': (numpy.uint16, int),
    'int32': (numpy.int32, int), 'uint32': (numpy.uint32, long),
    'int64': (numpy.int64, long), 'uint64': (numpy.uint64, long),
    'float32': (numpy.float32, float), 'float64': (numpy.float32, float),
    'complex64': (numpy.complex64, complex),
    'complex128': (numpy.complex128, complex),
    'time32': (numpy.int32, int), 'time64': (numpy.float64, float),
    'enum': (numpy.uint8, int),  # just for these tests
    'string': ('S%s' % _strlen, str) }  # just for these tests
"""NumPy and Numexpr type for each PyTables type that will be tested."""

sctype_from_type = dict( (type_, info[0])
                         for (type_, info) in type_info.iteritems() )
"""Maps PyTables types to NumPy scalar types."""
nxtype_from_type = dict( (type_, info[1])
                         for (type_, info) in type_info.iteritems() )
"""Maps PyTables types to Numexpr types."""

heavy_types = frozenset(['uint8', 'int16', 'uint16', 'float32', 'complex64'])
"""PyTables types to be tested only in heavy mode."""

enum = tables.Enum(dict(('n%d' % i, i) for i in range(_maxnvalue)))
"""Enumerated type to be used in tests."""


# Table description
# -----------------
def append_columns(classdict, shape=()):
    """
    Append a ``Col`` of each PyTables data type to the `classdict`.

    A column of a certain TYPE gets called ``c_TYPE``.  The number of
    added columns is returned.
    """
    heavy = common.heavy
    for (itype, type_) in enumerate(sorted(type_info.iterkeys())):
        if not heavy and type_ in heavy_types:
            continue  # skip heavy type in non-heavy mode
        colpos = itype + 1
        colname = 'c_%s' % type_
        if type_ == 'enum':
            base = tables.Atom.from_sctype(sctype_from_type[type_])
            col = tables.EnumCol(enum, enum(0), base, shape=shape, pos=colpos)
        else:
            sctype = sctype_from_type[type_]
            dtype = numpy.dtype((sctype, shape))
            col = tables.Col.from_dtype(dtype, pos=colpos)
        classdict[colname] = col
    ncols = colpos
    return ncols

def nested_description(classname, pos, shape=()):
    """
    Return a nested column description with all PyTables data types.

    A column of a certain TYPE gets called ``c_TYPE``.  The nested
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
    top level and in the ``c_nested`` nested column.  A column of a
    certain TYPE gets called ``c_TYPE``.  An extra integer column
    ``c_extra`` is also provided.  If a `shape` is given, it will be
    used for all columns.
    """
    classdict = {}
    colpos = append_columns(classdict, shape)

    ndescr = nested_description(nclassname, colpos, shape=shape)
    classdict['c_nested'] = ndescr
    colpos += 1

    extracol = tables.IntCol(shape=shape, pos=colpos)
    classdict['c_extra'] = extracol
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

    The same goes for the ``c_extra`` column, but values range from
    -`row_period`/2 to +`row_period`/2.
    """
    # Reuse already computed data if possible.
    tdata = table_data.get((shape, nrows))
    if tdata is not None:
        table.append(tdata)
        table.flush()
        return

    heavy = common.heavy
    size = int(numpy.prod(shape, dtype=SizeType))

    row, value = table.row, 0
    for nrow in xrange(nrows):
        data = numpy.arange(value, value + size).reshape(shape)
        for (type_, sctype) in sctype_from_type.iteritems():
            if not heavy and type_ in heavy_types:
                continue  # skip heavy type in non-heavy mode
            colname = 'c_%s' % type_
            ncolname = 'c_nested/%s' % colname
            if type_ == 'bool':
                coldata = data > (row_period / 2)
            elif type_ == 'string':
                sdata = [str_format % x for x in range(value, value + size)]
                coldata = numpy.array(sdata, dtype=sctype).reshape(shape)
            else:
                coldata = numpy.asarray(data, dtype=sctype)
            row[ncolname] = row[colname] = coldata
            row['c_extra'] = data - (row_period / 2)
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
class BaseTableQueryTestCase(common.TempFileMixin, common.PyTablesTestCase):

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
            idxtype = self.idxtype
            vprint("* Indexing ``%s`` columns. Type: %s." % (colname, idxtype))
            for acolname in [colname, ncolname]:
                acolumn = self.table.colinstances[acolname]
                if idxtype == "Full":
                    acolumn.createFullIndex(optlevel=self.optlevel,
                                            _blocksizes=small_blocksizes,
                                            _testmode=True)
                elif idxtype == "Medium":
                    acolumn.createMediumIndex(optlevel=self.optlevel,
                                              _blocksizes=small_blocksizes,
                                              _testmode=True)
                elif idxtype == "Light":
                    acolumn.createLightIndex(optlevel=self.optlevel,
                                             _blocksizes=small_blocksizes,
                                             _testmode=True)
                elif idxtype == "UltraLight":
                    acolumn.createUltraLightIndex(optlevel=self.optlevel,
                                                  _blocksizes=small_blocksizes,
                                                  _testmode=True)
                else:
                    raise common.SkipTest(
                        "Indexing type ``%s`` is not supported yet." % idxtype )

        except TypeError, te:
            if self.colNotIndexable_re.search(str(te)):
                raise common.SkipTest(
                    "Columns of this type can not be indexed." )
            raise
        except tables.NoIndexingError:
            raise common.SkipTest("Indexing is not supported.")
        except NotImplementedError:
            raise common.SkipTest(
                "Indexing columns of this type is not supported yet." )

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
operators = [
    None, '~',
    '<', '<=', '==', '!=', '>=', '>',
    ('<', '<='), ('>', '>=') ]
"""Comparison operators to check with different types."""
heavy_operators = frozenset(['~', '<=', '>=', '>', ('>', '>=')])
"""Comparison operators to be tested only in heavy mode."""
left_bound = row_period / 4
"""Operand of left side operator in comparisons with operator pairs."""
right_bound = row_period * 3 / 4
"""Operand of right side operator in comparisons with operator pairs."""
extra_conditions = ['', '& ((c_extra+1) > 0)', '| ((c_extra+1) > 0)']
"""Extra conditions to append to comparison conditions."""

class TableDataTestCase(BaseTableQueryTestCase):
    """
    Base test case for querying table data.

    Automatically created test method names have the format
    ``test_XNNNN``, where ``NNNN`` is the zero-padded test number and
    ``X`` indicates whether the test belongs to the light (``l``) or
    heavy (``h``) set.
    """
    _testfmt_light = 'test_l%04d'
    _testfmt_heavy = 'test_h%04d'

def create_test_method(type_, op, extracond):
    sctype = sctype_from_type[type_]

    # Compute the value of bounds.
    condvars = { 'bound': right_bound,
                 'lbound': left_bound,
                 'rbound': right_bound }
    for (bname, bvalue) in condvars.items():
        if type_ == 'string':
            bvalue = str_format % bvalue
        bvalue = nxtype_from_type[type_](bvalue)
        condvars[bname] = bvalue

    # Compute the name of columns.
    colname = 'c_%s' % type_
    ncolname = 'c_nested/%s' % colname

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

        table_slice = dict(start=1, stop=table.nrows - 5, step=3)
        rownos, fvalues = None, None
        # Test that both simple and nested columns work as expected.
        # Knowing how the table is filled, results must be the same.
        for acolname in [colname, ncolname]:
            # First the reference Python version.
            pyrownos, pyfvalues, pyvars = [], [], condvars.copy()
            for row in table.iterrows(**table_slice):
                pyvars[colname] = row[acolname]
                pyvars['c_extra'] = row['c_extra']
                try:
                    isvalidrow = eval(pycond, {}, pyvars)
                except TypeError:
                    raise common.SkipTest(
                        "The Python type does not support the operation." )
                if isvalidrow:
                    pyrownos.append(row.nrow)
                    pyfvalues.append(row[acolname])
            pyrownos = numpy.array(pyrownos)  # row numbers already sorted
            pyfvalues = numpy.array(pyfvalues, dtype=sctype)
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
            ptvars['c_extra'] = table.colinstances['c_extra']
            try:
                isidxq = table.willQueryUseIndexing(cond, ptvars)
                # Query twice to trigger possible query result caching.
                ptrownos = [ table.getWhereList( cond, condvars, sort=True,
                                                 **table_slice )
                             for _ in range(2) ]
                ptfvalues = [ table.readWhere( cond, condvars, field=acolname,
                                               **table_slice )
                              for _ in range(2) ]
            except TypeError, te:
                if self.condNotBoolean_re.search(str(te)):
                    raise common.SkipTest("The condition is not boolean.")
                raise
            except NotImplementedError:
                raise common.SkipTest(
                    "The PyTables type does not support the operation." )
            for ptfvals in ptfvalues:  # row numbers already sorted
                ptfvals.sort()
            vprint( "* %d rows selected by PyTables from ``%s``"
                    % (len(ptrownos[0]), acolname), nonl=True )
            vprint("(indexing: %s)." % ["no", "yes"][bool(isidxq)])
            self.assert_(numpy.all(ptrownos[0] == rownos))
            self.assert_(numpy.all(ptfvalues[0] == fvalues))
            # The following test possible caching of query results.
            self.assert_(numpy.all(ptrownos[0] == ptrownos[1]))
            self.assert_(numpy.all(ptfvalues[0] == ptfvalues[1]))

    test_method.__doc__ = "Testing ``%s``." % cond
    return test_method

# Create individual tests.  You may restrict which tests are generated
# by replacing the sequences in the ``for`` statements.  For instance:
testn = 0
for type_ in type_info:  # for type_ in ['string']:
    for op in operators:  # for op in ['!=']:
        # Decide to which set the test belongs.
        heavy = type_ in heavy_types or op in heavy_operators
        if heavy:
            testfmt = TableDataTestCase._testfmt_heavy
            numfmt = ' [#H%d]'
        else:
            testfmt = TableDataTestCase._testfmt_light
            numfmt = ' [#L%d]'
        for extracond in extra_conditions:  # for extracond in ['']:
            tmethod = create_test_method(type_, op, extracond)
            # The test number is appended to the docstring to help
            # identify failing methods in non-verbose mode.
            tmethod.__name__ = testfmt % testn
            #tmethod.__doc__ += numfmt % testn
            tmethod.__doc__ += testfmt % testn
            ptmethod = common.pyTablesTest(tmethod)
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
table_sizes = ['Small', 'Big']
heavy_table_sizes = frozenset(['Big'])
table_ndims = ['Scalar']  # to enable multidimensional testing, include 'MD'

# Non-indexed queries: ``[SB][SM]TDTestCase``, where:
#
# 1. S is for small and B is for big size table.
#    Sizes are listed in `table_sizes`.
# 2. S is for scalar and M for multidimensional columns.
#    Dimensionalities are listed in `table_ndims`.
def niclassdata():
    for size in table_sizes:
        heavy = size in heavy_table_sizes
        for ndim in table_ndims:
            classname = '%s%sTDTestCase' % (size[0], ndim[0])
            cbasenames = ( '%sNITableMixin' % size, '%sTableMixin' % ndim,
                           'TableDataTestCase' )
            classdict = dict(heavy=heavy)
            yield (classname, cbasenames, classdict)


# Base classes for the different type index.
class UltraLightITableMixin:
    idxtype = "UltraLight"
class LightITableMixin:
    idxtype = "Light"
class MediumITableMixin:
    idxtype = "Medium"
class FullITableMixin:
    idxtype = "Full"

# Base classes for indexed queries.
class SmallSTableMixin:
    nrows = 50
class MediumSTableMixin:
    nrows = 100
class BigSTableMixin:
    nrows = 500

# Parameters for indexed queries.
idxtypes = ['UltraLight', 'Light', 'Medium', 'Full']
itable_sizes = ['Small', 'Medium', 'Big']
heavy_itable_sizes = frozenset(['Medium', 'Big'])
itable_optvalues = [0, 1, 3, 7, 9]
heavy_itable_optvalues = frozenset([0, 1, 7, 9])

# Indexed queries: ``[SMB]I[ULMF]O[01379]TDTestCase``, where:
#
# 1. S is for small, M for medium and B for big size table.
#    Sizes are listed in `itable_sizes`.
# 2. U is for 'UltraLight', L for 'Light', R for 'Medium', F for 'Full' indexes
#    Index types are listed in `idxtypes`.
# 3. 0 to 9 is the desired index optimization level.
#    Optimizations are listed in `itable_optvalues`.
def iclassdata():
    for idxtype in idxtypes:
        for size in itable_sizes:
            for optlevel in itable_optvalues:
                heavy = ( optlevel in heavy_itable_optvalues
                          or size in heavy_itable_sizes )
                classname = '%sI%sO%dTDTestCase' % (
                    size[0], idxtype[0], optlevel)
                cbasenames = ( '%sSTableMixin' % size,
                               '%sITableMixin' % idxtype,
                               'ScalarTableMixin',
                               'TableDataTestCase' )
                classdict = dict(heavy=heavy, optlevel=optlevel, indexed=True)
                yield (classname, cbasenames, classdict)


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
        self.assertRaises(TypeError, self.table.where, 'c_bool < []')

    def test_unsupported_syntax(self):
        """Using a condition with unsupported syntax."""
        self.assertRaises(TypeError, self.table.where, 'c_bool[0]')
        self.assertRaises(TypeError, self.table.where, 'c_bool()')
        self.assertRaises(NameError, self.table.where, 'c_bool.__init__')

    def test_no_column(self):
        """Using a condition with no participating columns."""
        self.assertRaises(ValueError, self.table.where, 'True')

    def test_foreign_column(self):
        """Using a condition with a column from other table."""
        table2 = self.h5file.createTable('/', 'other', self.tableDescription)
        self.assertRaises( ValueError, self.table.where,
                           'c_int32_a + c_int32_b > 0',
                           { 'c_int32_a': self.table.cols.c_int32,
                             'c_int32_b': table2.cols.c_int32 } )

    def test_unsupported_op(self):
        """Using a condition with unsupported operations on types."""
        NIE = NotImplementedError
        self.assertRaises(NIE, self.table.where, 'c_complex128 > 0j')
        self.assertRaises(NIE, self.table.where, 'c_string + "a" > "abc"')

    def test_not_boolean(self):
        """Using a non-boolean condition."""
        self.assertRaises(TypeError, self.table.where, 'c_int32')

    def test_nested_col(self):
        """Using a condition with nested columns."""
        self.assertRaises(TypeError, self.table.where, 'c_nested')

    def test_implicit_col(self):
        """Using implicit column names in conditions."""
        # If implicit columns didn't work, a ``NameError`` would be raised.
        self.assertRaises(TypeError, self.table.where, 'c_int32')
        # If overriding didn't work, no exception would be raised.
        self.assertRaises( TypeError, self.table.where,
                           'c_bool', {'c_bool': self.table.cols.c_int32} )
        # External variables do not override implicit columns.
        def where_with_locals():
            c_int32 = self.table.cols.c_bool  # this wouldn't cause an error
            self.table.where('c_int32')
        self.assertRaises(TypeError, where_with_locals)

    def test_condition_vars(self):
        """Using condition variables in conditions."""

        # If condition variables didn't work, a ``NameError`` would be raised.
        self.assertRaises( NotImplementedError, self.table.where,
                           'c_string > bound', {'bound': 0})

        def where_with_locals():
            bound = 'foo'  # this wouldn't cause an error
            self.table.where('c_string > bound', {'bound': 0})
        self.assertRaises(NotImplementedError, where_with_locals)

        def where_with_globals():
            global _gvar
            _gvar = 'foo'  # this wouldn't cause an error
            try:
                self.table.where('c_string > _gvar', {'_gvar': 0})
            finally:
                del _gvar  # to keep global namespace clean
        self.assertRaises(NotImplementedError, where_with_globals)

    def test_scopes(self):
        """Looking up different scopes for variables."""

        # Make sure the variable is not implicit.
        self.assertRaises(NameError, self.table.where, 'col')

        # First scope: dictionary of condition variables.
        self.assertRaises( TypeError, self.table.where,
                           'col', {'col': self.table.cols.c_int32} )

        # Second scope: local variables.
        def where_whith_locals():
            col = self.table.cols.c_int32
            self.table.where('col')
        self.assertRaises(TypeError, where_whith_locals)

        # Third scope: global variables.
        def where_with_globals():
            global _gvar
            _gvar = self.table.cols.c_int32
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
        self.assertRaises(NotImplementedError, self.table.where, 'c_bool')

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
    for _cbase in ['c_bool', '~c_bool']:
        conditions.append(_cbase)
        conditions.append('(%s) & (c_extra > 0)' % _cbase)
        conditions.append('(%s) & (c_bool == False)' % _cbase)
        conditions.append('(c_extra > 0) & (%s)' % _cbase)
    # Add simple numeric conditions.
    for _usevar in [False, True]:
        for _condt in ['c_int32 %(o)s %(v)s', '%(v)s %(o)s c_int32']:
            for _op in ['<', '<=', '==', '>=', '>']:
                if _usevar:
                    _cdict = {'o': _op, 'v': 'var'}
                else:
                    _cdict = {'o': _op, 'v': 0}
                conditions.append(_condt % _cdict)
    conditions.append('(c_int32 > 0) & (c_extra > 0)')
    conditions.append('(c_int32 > 0) & (c_bool == True)')
    conditions.append('(c_extra > 0) & (c_int32 > 0)')
    # Add double numeric conditions.
    for _cbase in ['(0<c_int32) & (c_int32<10)', '(10>c_int32) & (c_int32>0)']:
        conditions.append(_cbase)
        conditions.append('(%s) & (c_extra > 0)' % _cbase)
        conditions.append('(%s) & (c_bool == True)' % _cbase)
        conditions.append('(c_extra > 0) & (%s)' % _cbase)

    def setUp(self):
        super(IndexedTableUsageTestCase, self).setUp()
        self.table.cols.c_bool.createIndex(_blocksizes=small_blocksizes)
        self.table.cols.c_int32.createIndex(_blocksizes=small_blocksizes)
        self.willQueryUseIndexing = self.table.willQueryUseIndexing
        self.compileCondition = self.table._compileCondition
        self.requiredExprVars = self.table._requiredExprVars

    def test00(self):
        """Using indexing in some queries."""
        for condition in self.conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )


    def test01(self):
        """Using indexing in specific queries (set #1)."""
        conditions = [
            '(c_int32 > 0)',
            '(c_int32 > 0) & (c_bool == True)',
            '(c_int32 > 0) & (c_extra > 0)']
        for condition in conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )
            condvars = self.requiredExprVars(condition, None)
            compiled = self.compileCondition(condition, condvars)
            self.assert_( compiled.index_limits == [0],
                          "wrong limits in condition ``%s``"
                          % condition)
            self.assert_( compiled.index_operators == ['gt'],
                          "wrong limits in condition ``%s``"
                          % condition)

    def test02(self):
        """Using indexing in specific queries (set #2)."""
        conditions = [
            '(c_int32 > 0) & (c_int32 < 5)',
            '(c_int32 > 0) & (c_int32 < 5) & (c_bool == True)',
            '(c_int32 > 0) & (c_int32 < 5) & (c_extra > 0)']
        for condition in conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )
            condvars = self.requiredExprVars(condition, None)
            compiled = self.compileCondition(condition, condvars)
            self.assert_( compiled.index_limits == [0,5],
                          "wrong limits in condition ``%s``"
                          % condition)
            self.assert_( compiled.index_operators == ['gt','lt'],
                          "wrong limits in condition ``%s``"
                          % condition)

    def test03(self):
        """Using indexing in specific queries (set #3)."""
        conditions = [
            '(c_bool == True)',
            '(c_bool == True) & (c_extra > 0)',
            '(c_bool == True) & (c_int32 > 0) & (c_int32 < 5)']
        for condition in conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )
            condvars = self.requiredExprVars(condition, None)
            compiled = self.compileCondition(condition, condvars)
            self.assert_( compiled.index_limits == [True],
                          "wrong limits in condition ``%s``"
                          % condition)
            self.assert_( compiled.index_operators == ['eq'],
                          "wrong limits in condition ``%s``"
                          % condition)

    def test04(self):
        """Using indexing in specific queries (set #4)."""
        conditions = [
            '(c_int32 >= 1) & (c_int32 < 2)',
            '(c_int32 >= 1) & (c_int32 < 2) & (c_int32 > 0) & (c_int32 < 5)']
        for condition in conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )
            condvars = self.requiredExprVars(condition, None)
            compiled = self.compileCondition(condition, condvars)
            self.assert_( compiled.index_limits == [1,2],
                          "wrong limits in condition ``%s``"
                          % condition)
            self.assert_( compiled.index_operators == ['ge','lt'],
                          "wrong limits in condition ``%s``"
                          % condition)

    def test05(self):
        """Using indexing in specific queries (set #5)."""
        conditions = [
            '(c_extra > 0) & (c_bool == True)']
        for condition in conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )
            condvars = self.requiredExprVars(condition, None)
            compiled = self.compileCondition(condition, condvars)
            self.assert_( compiled.index_limits == [True],
                          "wrong limits in condition ``%s``"
                          % condition)
            self.assert_( compiled.index_operators == ['eq'],
                          "wrong limits in condition ``%s``"
                          % condition)

    def test06(self):
        """Using indexing in specific queries (set #6)."""
        conditions = [
            '(c_extra > 0) & (c_int32 > 0) & (c_int32 < 5)']
        for condition in conditions:
            self.assert_( self.willQueryUseIndexing(condition, {'var': 0}),
                          "query with condition ``%s`` should use indexing"
                          % condition )
            vprint(
                "* Query with condition ``%s`` will use indexing."
                % condition )
            condvars = self.requiredExprVars(condition, None)
            compiled = self.compileCondition(condition, condvars)
            self.assert_( compiled.index_limits == [0,5],
                          "wrong limits in condition ``%s``"
                          % condition)
            self.assert_( compiled.index_operators == ['gt','lt'],
                          "wrong limits in condition ``%s``"
                          % condition)



# Main part
# ---------
def suite():
    """Return a test suite consisting of all the test cases in the module."""

    testSuite = unittest.TestSuite()

    cdatafuncs = [niclassdata]  # non-indexing data tests
    if tables.is_pro:
        cdatafuncs.append(iclassdata)  # indexing data tests

    heavy = common.heavy
    # Choose which tests to run in classes with autogenerated tests.
    if heavy:
        autoprefix = 'test'  # all tests
    else:
        autoprefix = 'test_l'  # only light tests

    niter = 1
    for i in range(niter):
        # Tests on query data.
        for cdatafunc in cdatafuncs:
            for cdata in cdatafunc():
                class_ = eval(cdata[0])
                if heavy or not class_.heavy:
                    suite_ = unittest.makeSuite(class_, prefix=autoprefix)
                    testSuite.addTest(suite_)
        # Tests on query usage.
        testSuite.addTest(unittest.makeSuite(ScalarTableUsageTestCase))
        testSuite.addTest(unittest.makeSuite(MDTableUsageTestCase))
        if tables.is_pro:
            testSuite.addTest(unittest.makeSuite(IndexedTableUsageTestCase))

    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
