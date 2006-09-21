"""
Utility functions and classes for supporting query conditions.

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2006-09-19
:License:  BSD
:Revision: $Id$

Classes:

`SplittedCondition`
    Container for an splitted condition.

Functions:

`split_condition`
    Split a condition into indexable and non-indexable parts.
`call_on_recarr`
    Evaluate a function over a record array.
"""

from tables.numexpr.expressions import bestConstantType
from tables.numexpr.compiler import stringToExpression, numexpr


class SplittedCondition(object):
    """Container for an splitted condition."""
    def __init__(self, idxvar, idxops, idxlims, resfunc, resparams):
        self.index_variable = idxvar
        self.index_operators = idxops
        self.index_limits = idxlims
        self.residual_function = resfunc
        self.residual_parameters = resparams


def _get_indexable_cmp(exprnode, indexedcols):
    """
    Get the indexable variable-constant comparison in `exprnode`.

    A tuple of (variable, operation, constant) is returned if
    `exprnode` is a variable-constant (or constant-variable)
    comparison, and the variable is in `indexedcols`.  A normal
    variable can also be used instead of a constant: a tuple with its
    name will appear instead of its value.

    Otherwise, the values in the tuple are ``None``.
    """
    not_indexable = (None, None, None)
    turncmp = { 'lt': 'gt',
                'le': 'ge',
                'eq': 'eq',
                'ge': 'le',
                'gt': 'lt', }

    def get_cmp(var, const, op):
        var_value, const_value = var.value, const.value
        if ( var.astType == 'variable' and var_value in indexedcols
             and const.astType in ['constant', 'variable'] ):
            if const.astType == 'variable':
                const_value = (const_value, )
            return (var_value, op, const_value)
        return None

    # Check node type.
    if exprnode.astType != 'op':
        return not_indexable
    cmpop = exprnode.value
    if cmpop not in turncmp:
        return not_indexable

    # Look for a variable-constant comparison in both directions.
    left, right = exprnode.children
    cmp_ = get_cmp(left, right, cmpop)
    if cmp_:  return cmp_
    cmp_ = get_cmp(right, left, turncmp[cmpop])
    if cmp_:  return cmp_

    return not_indexable

def _split_expression(exprnode, indexedcols):
    """
    Split an expression into indexable and non-indexable parts.

    Looks for variable-constant comparisons in the expression node
    `exprnode` involving variables in `indexedcols`.  The *topmost*
    comparison of comparison pair is splitted apart from the rest of
    the expression (the residual expression) and the resulting tuple
    (indexed_variable, operators, limits, residual_expr) is returned.
    Thus (for indexed column *c1*):

    * 'c1 > 0' -> ('c1',['gt'],[0],None)
    * '(0 < c1) & (c1 <= 1)' -> ('c1',['gt','le'],[0,1],None)
    * '(0 < c1) & (c1 <= 1) & (c2 > 2)' -> ('c1',['gt','le'],[0,1],#c2>2#)

    * 'c2 > 2' -> (None,[],[],#c2>2#)
    * '(c2 > 2) & (c1 <= 1)' -> ('c1',['le'],[1],#c2>2#)
    * '(0 < c1) & (c1 <= 1) & (c2 > 2)' -> ('c1',['gt','le'],[0,1],#c2>2#)

    * '(c2 > 2) & (0 < c1) & (c1 <= 1)' -> ('c1',['le'],[1],#(c2>2)&(c1>0)#)
    * '(c2 > 2) & ((0 < c1) & (c1 <= 1))' -> ('c1',['gt','le'],[0,1],#c2>2#)

    * '(0 < c1) & (c2 > 2) & (c1 <= 1)' -> ('c1',['le'],[1],#(c1>0)&(c2>2)#)
    * '(0 < c1) & ((c2 > 2) & (c1 <= 1))' -> ('c1',['gt'],[0],#(c2>2)&(c1<=1)#)

    Expressions such as '0 < c1 <= 1' do not work as expected.
    """
    not_indexable =  (None, [], [], exprnode)

    # Indexable variable-constant comparison.
    idxcmp = _get_indexable_cmp(exprnode, indexedcols)
    if idxcmp[0]:
        return (idxcmp[0], [idxcmp[1]], [idxcmp[2]], None)

    # Only conjunctions of comparisons may be indexable.
    if exprnode.astType != 'op' or exprnode.value != 'and':
        return not_indexable

    left, right = exprnode.children
    lcolvar, lop, llim = _get_indexable_cmp(left, indexedcols)
    rcolvar, rop, rlim = _get_indexable_cmp(right, indexedcols)

    # Conjunction of indexable VC comparisons.
    if lcolvar and rcolvar and lcolvar == rcolvar:
        return (lcolvar, [lop, rop], [llim, rlim], None)

    # Indexable VC comparison on one side only.
    for (colvar, op, lim, other) in [ (lcolvar, lop, llim, right),
                                      (rcolvar, rop, rlim, left), ]:
        if colvar:
            return (colvar, [op], [lim], other)

    # Recursion: conjunction of indexable expression and other.
    for (this, other) in [(left, right), (right, left)]:
        colvar, ops, lims, res = _split_expression(this, indexedcols)
        if res:  # (IDX & RES) & OTHER <=> IDX & (RES & OTHER)
            other = res & other
        if colvar:
            return (colvar, ops, lims, other)

    # Can not use indexed column.
    return not_indexable

_nxTypeFromColumn = {
    'Bool': bool,
    'Int8': int,
    'Int16': int,
    'Int32': int,
    'Int64': long,
    'UInt8': int,
    'UInt16': int,
    'UInt32': long,
    'UInt64': long,
    'Float32': float,
    'Float64': float,
    'Complex32': complex,
    'Complex64': complex,
    'CharType': str, }

def _get_variable_names(expression):
    """Return the list of variable names in the Numexpr `expression`."""
    names = []
    stack = [expression]
    while stack:
        node = stack.pop()
        if node.astType == 'variable':
            names.append(node.value)
        elif hasattr(node, 'children'):
            stack.extend(node.children)
    return names

def split_condition(condition, condvars, table):
    """
    Split a condition into indexable and non-indexable parts.

    Looks for variable-constant comparisons in the condition string
    `condition` involving indexed columns in `condvars`.  The
    *topmost* comparison of comparison pair is splitted apart from the
    rest of the condition (the residual condition) and the resulting
    `SplittedCondition` is returned.  Thus (for indexed column *c1*):

    * 'c1>0' -> ('c1', ['gt'], [0], None, [])
    * '(0<c1) & (c1<=1)' -> ('c1', ['gt', 'le'], [0, 1], None, [])
    * '(0<c1) & (c1<=1) & (c2>2)' -> ('c1',['gt','le'],[0,1],{c2>2},['c2'])

    * 'c2>2' -> (None, [], [],'(c2>2)')
    * '(c2>2) & (c1<=1)' -> ('c1', ['le'], [1], {c2>2}, ['c2'])
    * '(0<c1) & (c1<=1) & (c2>2)' -> ('c1',['gt','le'],[0,1],{c2>2},['c2'])

    * '(c2>2) & (0<c1) & (c1<=1)' -> ('c1',['le'],[1],{(c2>2)&(c1>0)},['c2','c1'])
    * '(c2>2) & ((0<c1) & (c1<=1))' -> ('c1',['gt','le'],[0,1],{c2>2},['c2'])

    * '(0<c1) & (c2>2) & (c1<=1)' -> ('c1',['le'],[1],{(c1>0)&(c2>2)},['c1','c2'])
    * '(0<c1) & ((c2>2) & (c1<=1))' -> ('c1',['gt'],[0],{(c2>2)&(c1<=1)},['c2','c1'])

    Expressions such as '0 < c1 <= 1' do not work as expected.  The
    ``residual_condition`` is a Numexpr function object, and the list
    ``residual_params`` indicates the order of its parameters.  The
    `table` argument refers to a table where a condition cache can be
    looked up and updated.
    """
    tblfile = table._v_file
    tblpath = table._v_pathname

    # Build the key for the condition cache.
    colnames, varnames = [], []
    colpaths, vartypes = [], []
    for (var, val) in condvars.items():
        if hasattr(val, 'pathname'):  # looks like a column
            colnames.append(var)
            colpaths.append(val.pathname)
            if val._tableFile is not tblfile or val._tablePath != tblpath:
                raise ValueError("variable ``%s`` refers to a column "
                                 "which is not part of table ``%s``"
                                 % (var, tblpath))
        else:
            varnames.append(var)
            vartypes.append(bestConstantType(val))  # expensive
    colnames, varnames = tuple(colnames), tuple(varnames)
    colpaths, vartypes = tuple(colpaths), tuple(vartypes)
    condkey = (condition, colnames, varnames, colpaths, vartypes)

    # Look up the condition in the condition cache.
    condcache = table._conditionCache
    splitted = condcache.get(condkey)
    if splitted:
        return splitted  # bingo!

    # Bad luck, the condition must be parsed and splitted.

    # Extract types from *all* the given variables.
    typemap = dict(zip(varnames, vartypes))  # start with normal variables
    for colname in colnames:  # then add types of columns
        # Converting to a string may not be necessary when the
        # transition from numarray to NumPy is complete.
        coldtype = str(condvars[colname].type)
        typemap[colname] = _nxTypeFromColumn[coldtype]

    # Get the set of columns with usable indexes.
    can_use_index = lambda column: column.index and not column.dirty
    indexedcols = frozenset(
        colname for colname in colnames
        if can_use_index(condvars[colname]) )

    # Get the expression tree and split the indexable part out.
    expr = stringToExpression(condition, typemap, {})
    idxvar, idxops, idxlims, resexpr = _split_expression(expr, indexedcols)

    # Get the variable names used in the residual condition,
    # and check that they are defined in ``condvars``
    # (``idxvar`` needs not to be checked because it clearly is
    # in ``condvars`` and points to an existing column).
    # At the same time, build the signature of the residual condition.
    resfunc, resparams = None, []
    if resexpr:
        resvarnames, ressignature = _get_variable_names(resexpr), []
        for var in resvarnames:
            if var not in condvars:
                raise NameError("name ``%s`` is not defined" % var)
            ressignature.append((var, typemap[var]))
        resfunc = numexpr(resexpr, ressignature)
        resparams = resvarnames

    splitted = SplittedCondition(idxvar, idxops, idxlims, resfunc, resparams)

    # Store the splitted condition in the cache and return it.
    condcache[condkey] = splitted
    return splitted

def call_on_recarr(func, params, recarr, param2arg=None):
    """
    Call `func` with `params` over `recarr`.

    The `param2arg` function, when specified, is used to get an
    argument given a parameter name; otherwise, the parameter itself
    is used as an argument.  When the argument is a `Column` object,
    the proper column from `recarr` is used as its value.
    """
    args = []
    for param in params:
        if param2arg:
            arg = param2arg(param)
        else:
            arg = param
        if hasattr(arg, 'pathname'):  # looks like a column
            arg = recarr[arg.pathname]
        args.append(arg)
    return func(*args)
