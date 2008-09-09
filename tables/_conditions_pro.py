"""
Utility functions and classes for supporting query conditions (pro).

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2007-02-12
:License:  XXX
:Notes:    Heavily modified by Francesc Alted for multi-index support.
           2008-04-09
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
        # A negation of an expression will be returned as ``~child``.
        # The indexability of the negated expression will be decided later on.
        if child.astKind == "bool":
            return (child, 'invert', None)

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


def _get_idx_expr_recurse(exprnode, indexedcols, idxexprs, strexpr):
    """Here lives the actual implementation of the get_idx_expr() wrapper.

    'idxexprs' is a list of expressions in the form ``(var, (ops),
    (limits))``. 'strexpr' is the indexable expression in string
    format.  These parameters will be received empty (i.e. [], [''])
    for the first time and populated during the different recursive
    calls.  Finally, they are returned in the last level to the
    original wrapper.  If 'exprnode' is not indexable, it will return
    the tuple ([], ['']) so as to signal this.
    """
    not_indexable =  ([], [''])
    op_conv = { 'and': '&',
                'or': '|',
                'not': '~', }
    negcmp = { 'lt': 'ge',
               'le': 'gt',
               'ge': 'lt',
               'gt': 'le', }

    def fix_invert(idxcmp, exprnode, indexedcols):
        invert = False
        # Loop until all leading negations have been dealt with
        while idxcmp[1] == "invert":
            invert ^= True
            # The information about the negated node is in first position
            exprnode = idxcmp[0]
            idxcmp = _get_indexable_cmp(exprnode, indexedcols)
        return idxcmp, exprnode, invert

    # Indexable variable-constant comparison.
    idxcmp = _get_indexable_cmp(exprnode, indexedcols)
    idxcmp, exprnode, invert = fix_invert(idxcmp, exprnode, indexedcols)
    if idxcmp[0]:
        if invert:
            var, op, value = idxcmp
            if op == 'eq' and value in [True, False]:
                # ``var`` must be a boolean index.  Flip its value.
                value ^= True
            else:
                op = negcmp[op]
            expr = (var, (op,), (value,))
            invert = False
        else:
            expr = (idxcmp[0], (idxcmp[1],), (idxcmp[2],))
        return [expr]

    # For now negations of complex expressions will be not supported as
    # forming part of an indexable condition.  This might be supported in
    # the future.
    if invert:
        return not_indexable

    # Only conjunctions and disjunctions of comparisons are considered
    # for the moment.
    if exprnode.astType != 'op' or exprnode.value not in ['and', 'or']:
        return not_indexable

    left, right = exprnode.children
    # Get the expression at left
    lcolvar, lop, llim = _get_indexable_cmp(left, indexedcols)
    # Get the expression at right
    rcolvar, rop, rlim = _get_indexable_cmp(right, indexedcols)

    # Use conjunction of indexable VC comparisons like
    # ``(a <[=] x) & (x <[=] b)`` or ``(a >[=] x) & (x >[=] b)``
    # as ``a <[=] x <[=] b``, for the moment.
    op = exprnode.value
    if lcolvar and rcolvar and lcolvar == rcolvar and op == 'and':
        if lop in ['gt', 'ge'] and rop in ['lt', 'le']:  # l <= x <= r
            expr = (lcolvar, (lop, rop), (llim, rlim))
            return [expr]
        if lop in ['lt', 'le'] and rop in ['gt', 'ge']:  # l >= x >= r
            expr = (rcolvar, (rop, lop), (rlim, llim))
            return [expr]

    # Recursively get the expressions at the left and the right
    lexpr = _get_idx_expr_recurse(left, indexedcols, idxexprs, strexpr)
    rexpr = _get_idx_expr_recurse(right, indexedcols, idxexprs, strexpr)

    def add_expr(expr, idxexprs, strexpr):
        """Add a single expression to the list."""
        if type(expr) == list:
            # expr is a single expression
            idxexprs.append(expr[0])
            lenexprs = len(idxexprs)
            # Mutate the strexpr string
            if lenexprs == 1:
               strexpr[:] = ["e0"]
            else:
                strexpr[:] = [
                    "(%s %s e%d)" % (strexpr[0], op_conv[op], lenexprs-1) ]

    # Add expressions to the indexable list when they are and'ed, or
    # they are both indexable.
    if lexpr != not_indexable and (op == "and" or rexpr != not_indexable):
        add_expr(lexpr, idxexprs, strexpr)
        if rexpr != not_indexable:
            add_expr(rexpr, idxexprs, strexpr)
        return (idxexprs, strexpr)
    if rexpr != not_indexable and op == "and":
        add_expr(rexpr, idxexprs, strexpr)
        return (idxexprs, strexpr)

    # Can not use indexed column.
    return not_indexable


def _get_idx_expr(expr, indexedcols):
    """
    Extract an indexable expression out of `exprnode`.

    Looks for variable-constant comparisons in the expression node
    `exprnode` involving variables in `indexedcols`.

    It returns a tuple of (idxexprs, strexpr) where 'idxexprs' is a
    list of expressions in the form ``(var, (ops), (limits))`` and
    'strexpr' is the indexable expression in string format.

    Expressions such as ``0 < c1 <= 1`` do not work as expected.

    Right now only some of the *indexable comparisons* are considered:

    * ``a <[=] x``, ``a == x`` and ``a >[=] x``
    * ``(a <[=] x) & (y <[=] b)`` and ``(a == x) | (b == y)``
    * ``~(~c_bool)``, ``~~c_bool`` and ``~(~c_bool) & (c_extra != 2)``

    (where ``a``, ``b`` and ``c_bool`` are indexed columns, but
    ``c_extra`` is not)

    Particularly, the ``!=`` operator and negations of complex boolean
    expressions are *not considered* as valid candidates:

    * ``a != 1`` and  ``c_bool != False``
    * ``~((a > 0) & (c_bool))``
    """
    return _get_idx_expr_recurse(expr, indexedcols, [], [''])


