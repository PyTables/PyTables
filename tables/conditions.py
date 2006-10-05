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

import re

from tables.numexpr.compiler import (
    typecode_to_kind, stringToExpression, numexpr )


class SplittedCondition(object):
    """Container for an splitted condition."""
    def __init__(self, idxvar, idxops, idxlims, resfunc, resparams):
        self.index_variable = idxvar
        self.index_operators = idxops
        self.index_limits = idxlims
        self.residual_function = resfunc
        self.residual_parameters = resparams

    def with_replaced_vars(self, condvars):
        """
        Replace index limit variables with their values.

        A new splitted condition is returned.  Values are taken from
        the `condvars` mapping and converted to Python scalars.
        """
        limit_values = []
        for idxlim in self.index_limits:
            if type(idxlim) is tuple:  # variable
                idxlim = condvars[idxlim[0]]  # look up value
                idxlim = idxlim.tolist()  # convert back to Python
            limit_values.append(idxlim)
        return SplittedCondition(
            self.index_variable, self.index_operators, limit_values,
            self.residual_function, self.residual_parameters )


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
    return list(set(names))  # remove repeated names

_no_matching_opcode = re.compile(r"[^a-z]([a-z]+)_([a-z]+)[^a-z]")
# E.g. "gt" and "bfc" from "couldn't find matching opcode for 'gt_bfc'".

def _unsupported_operation_error(exception):
    """
    Make the \"no matching opcode\" Numexpr `exception` more clear.

    A new exception of the same kind is returned.
    """
    message = exception.args[0]
    op, types = _no_matching_opcode.search(message).groups()
    newmessage = "unsupported operand types for *%s*: " % op
    newmessage += ', '.join([typecode_to_kind[t] for t in types[1:]])
    return exception.__class__(newmessage)

def split_condition(condition, typemap, indexedcols):
    """
    Split a condition into indexable and non-indexable parts.

    Looks for variable-constant comparisons in the condition string
    `condition` involving the indexed columns whose variable names
    appear in `indexedcols`.  The *topmost* comparison or comparison
    pair is splitted apart from the rest of the condition (the
    *residual condition*) and the resulting `SplittedCondition` is
    returned.  Thus (for indexed column *c1*):

    * 'c1>0' -> ('c1', ['gt'], [0], None, [])
    * '(0<c1)&(c1<=1)' -> ('c1', ['gt', 'le'], [0, 1], None, [])
    * '(0<c1)&(c1<=1)&(c2>2)' -> ('c1',['gt','le'],[0,1],{c2>2},['c2'])

    * 'c2>2' -> (None, [], [],'(c2>2)')
    * '(c2>2)&(c1<=1)' -> ('c1', ['le'], [1], {c2>2}, ['c2'])
    * '(0<c1)&(c1<=1)&(c2>2)' -> ('c1',['gt','le'],[0,1],{c2>2},['c2'])

    * '(c2>2)&(0<c1)&(c1<=1)' -> ('c1',['le'],[1],{(c2>2)&(c1>0)},['c2','c1'])
    * '(c2>2)&((0<c1)&(c1<=1))' -> ('c1',['gt','le'],[0,1],{c2>2},['c2'])

    * '(0<c1)&(c2>2)&(c1<=1)' -> ('c1',['le'],[1],{(c1>0)&(c2>2)},['c1','c2'])
    * '(0<c1)&((c2>2)&(c1<=1))'->('c1',['gt'],[0],{(c2>2)&(c1<=1)},['c2','c1'])

    Expressions such as '0 < c1 <= 1' do not work as expected.  The
    Numexpr types of *all* variables must be given in the `typemap`
    mapping.  The ``residual_condition`` of the ``SplittedCondition``
    instance is a Numexpr function object, and the ``residual_params``
    list indicates the order of its parameters.
    """

    def check_boolean(expr):
        if expr and expr.astKind != 'bool':
            raise TypeError( "condition ``%s`` does not have a boolean type"
                             % condition )

    # Get the expression tree and split the indexable part out.
    expr = stringToExpression(condition, typemap, {})
    check_boolean(expr)
    idxvar, idxops, idxlims, resexpr = _split_expression(expr, indexedcols)
    check_boolean(resexpr)

    # Get the variable names used in the residual condition.
    # At the same time, build the signature of the residual condition.
    resfunc, resparams = None, []
    if resexpr:
        resvarnames, ressignature = _get_variable_names(resexpr), []
        for var in resvarnames:
            ressignature.append((var, typemap[var]))
        try:
            resfunc = numexpr(resexpr, ressignature)
        except NotImplementedError, nie:
            # Try to make this Numexpr error less cryptic.
            raise _unsupported_operation_error(nie)
        resparams = resvarnames

    assert idxvar or resfunc, (
        "no usable indexed column and no residual condition "
        "after splitting search condition" )

    # This is more comfortable to handle about than a tuple.
    return SplittedCondition(idxvar, idxops, idxlims, resfunc, resparams)

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
            # It may be convenient to factor out this way of
            # descending nested fields into the ``__getitem__()``
            # method of a subclass of ``numpy.ndarray``.  -- ivb
            arg, field = recarr, arg.pathname
            for nestedfield in field.split('/'):
                arg = arg[nestedfield]
        args.append(arg)
    return func(*args)
