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

from tables.numexpr.compiler import stringToExpression, numexpr
from tables.utilsExtension import getNestedField
from tables._conditions_common import _unsupported_operation_error

try:
    from _conditions_pro import _split_expression
except ImportError:
    def _split_expression(exprnode, indexedcols):
        return (None, [], [], exprnode)


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
            arg = getNestedField(recarr, arg.pathname)
        args.append(arg)
    return func(*args)
