"""
Utility functions and classes for supporting query conditions (index).

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2007-02-12
:License:  XXX
:Revision: $Id$
"""

from tables.numexpr.compiler import expressionToAST, typeCompileAst
from tables._conditions_common import _unsupported_operation_error


def _check_indexable_cmp(getidxcmp):
    """
    Decorate `getidxcmp` to check the returned indexable comparison.

    This does some extra checking that Numexpr would perform later on
    the comparison if it was compiled within a complete condition.
    """
    def newfunc(exprnode, indexedcols):
        result = getidxcmp(exprnode, indexedcols)
        if result[0] is not None:
            try:
                typeCompileAst(expressionToAST(exprnode))
            except NotImplementedError, nie:
                # Try to make this Numexpr error less cryptic.
                raise _unsupported_operation_error(nie)
        return result
    newfunc.__name__ = getidxcmp.__name__
    newfunc.__doc__ = getidxcmp.__doc__
    return newfunc

@_check_indexable_cmp
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

    def is_indexed_boolean(node):
        return ( node.astType == 'variable'
                 and node.astKind == 'bool'
                 and node.value in indexedcols )

    # Boolean variables are indexable by themselves.
    if is_indexed_boolean(exprnode):
        return (exprnode.value, 'eq', True)
    # And so are negations of boolean variables.
    if exprnode.astType == 'op' and exprnode.value == 'invert':
        child = exprnode.children[0]
        if is_indexed_boolean(child):
            return (child.value, 'eq', False)

    # Check node type.  Only comparisons are indexable from now on.
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

    Right now only some of the *indexable comparisons* (i.e. the ones
    which can be translated into a one-piece range) which are always
    indexable *regardless of their limits*, are considered:

    * ``a <[=] x``, ``a == x`` and ``a >[=] x``
    * ``(a <[=] x) & (x <[=] b)`` and ``(a >[=] x) & (a >[=] b)``

    Particularly, the following indexable comparisons are *not
    considered*:

    * ``(a == x) & (x == b)`` and ``(a == x) & (x != b)``
    * ``(a == x) & (x >[=] b)``
    * ``(a >[=] x) & (x <[=]b)``, ``(a <[=] x) & (x >[=]b)``
    * ``(a >[=] x) | (x <[=]b)``, ``(a <[=] x) | (x >[=]b)``
    """
    not_indexable =  (None, [], [], exprnode)

    # Indexable variable-constant comparison.  Since comparisons like
    # ``a != x`` are not indexable by themselves, they will not appear
    # onwards.
    idxcmp = _get_indexable_cmp(exprnode, indexedcols)
    if idxcmp[0]:
        return (idxcmp[0], [idxcmp[1]], [idxcmp[2]], None)

    # Only conjunctions of comparisons are considered for the moment.
    # This excludes the indexable disjunctions
    # ``(a <[=] x) | (x >[=] b)`` and ``(a >[=] x) | (x <[=] b)``.
    if exprnode.astType != 'op' or exprnode.value != 'and':
        return not_indexable

    left, right = exprnode.children
    lcolvar, lop, llim = _get_indexable_cmp(left, indexedcols)
    rcolvar, rop, rlim = _get_indexable_cmp(right, indexedcols)

    # Use conjunction of indexable VC comparisons like
    # ``(a <[=] x) & (x <[=] b)`` or ``(a >[=] x) & (x >[=] b)``
    # as ``a <[=] x <[=] b``, for the moment.  This excludes the
    # indexable conjunctions ``(a == x) & (x >[=] b)``,
    # ``(a == x) & (x == b)`` and ``(a == x) & (x != b)``,
    # ``(a <[=] x) & (x >[=] b)`` and ``(a >[=] x) & (x <[=] b)``.
    if lcolvar and rcolvar and lcolvar == rcolvar:
        if lop in ['gt', 'ge'] and rop in ['lt', 'le']:  # l <= x <= r
            return (lcolvar, [lop, rop], [llim, rlim], None)  # l <= x <= r
        elif lop in ['lt', 'le'] and rop in ['gt', 'ge']:  # l >= x >= r
            return (rcolvar, [rop, lop], [rlim, llim], None)  # r <= x <= l

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
