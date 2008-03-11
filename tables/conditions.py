"""
Utility functions and classes for supporting query conditions.

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2006-09-19
:License:  BSD
:Revision: $Id$

Classes:

`CompileCondition`
    Container for a compiled condition.

Functions:

`compile_condition`
    Compile a condition and extract usable index conditions.
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


class CompiledCondition(object):
    """Container for a compiled condition."""
    def __init__(self, func, params, idxvar, idxops, idxlims):
        self.function = func
        self.parameters = params
        self.index_variable = idxvar
        self.index_operators = idxops
        self.index_limits = idxlims

    def with_replaced_vars(self, condvars):
        """
        Replace index limit variables with their values.

        A new compiled condition is returned.  Values are taken from
        the `condvars` mapping and converted to Python scalars.
        """
        limit_values = []
        for idxlim in self.index_limits:
            if type(idxlim) is tuple:  # variable
                idxlim = condvars[idxlim[0]]  # look up value
                idxlim = idxlim.tolist()  # convert back to Python
            limit_values.append(idxlim)
        return CompiledCondition(
            self.function, self.parameters,
            self.index_variable, self.index_operators, limit_values )

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

def compile_condition(condition, typemap, indexedcols, copycols):
    """
    Compile a condition and extract usable index conditions.

    Looks for variable-constant comparisons in the `condition` string
    involving the indexed columns whose variable names appear in
    `indexedcols`.  The *topmost* comparison or comparison pair is
    used to extract usable index conditions, which are returned
    together with the compiled condition in a `CompiledCondition`.
    Thus, for an indexed column *c1* (*CC* is the compiled condition):

    * 'c1>0' -> (CC, ['c1'], 'c1', ['gt'], [0])
    * '(0<c1)&(c1<=1)' -> (CC, ['c1'], 'c1', ['gt', 'le'], [0, 1])
    * '(0<c1)&(c1<=1)&(c2>2)' -> (CC,['c2','c1'],'c1',['gt','le'],[0,1])

    * 'c2>2' -> (CC, ['c2'], None, [], [])
    * '(c2>2)&(c1<=1)' -> (CC, ['c2', 'c1'], 'c1', ['le'], [1])
    * '(0<c1)&(c1<=1)&(c2>2)' -> (CC,['c2','c1'],'c1',['gt','le'],[0,1])

    * '(c2>2)&(0<c1)&(c1<=1)' -> (CC, ['c2', 'c1'], 'c1', ['le'], [1])
    * '(c2>2)&((0<c1)&(c1<=1))' -> (CC,['c2','c1'],'c1',['gt','le'],[0,1])

    * '(0<c1)&(c2>2)&(c1<=1)' -> (CC, ['c2','c1'], 'c1', ['le'], [1])
    * '(0<c1)&((c2>2)&(c1<=1))' -> (CC, ['c2', 'c1'], 'c1', ['gt'], [0])

    Expressions such as '0 < c1 <= 1' do not work as expected.  The
    Numexpr types of *all* variables must be given in the `typemap`
    mapping.  The ``function`` of the resulting `CompiledCondition`
    instance is a Numexpr function object, and the ``parameters`` list
    indicates the order of its parameters.

    For columns whose variable names appear in `copycols`, an
    additional copy operation is inserted whenever the column is
    referenced.  This seems to accelerate access to unaligned,
    *unidimensional* arrays up to 2x (multidimensional arrays still
    need to be copied by `call_on_recarr()`.).
    """

    # Get the expression tree and extract index conditions.
    expr = stringToExpression(condition, typemap, {})
    if expr.astKind != 'bool':
        raise TypeError( "condition ``%s`` does not have a boolean type"
                         % condition )
    idxvar, idxops, idxlims, resexpr = _split_expression(expr, indexedcols)

    # Get the variable names used in the condition.
    # At the same time, build its signature.
    varnames = _get_variable_names(expr)
    signature = [(var, typemap[var]) for var in varnames]
    try:
        # See the comments in `tables.numexpr.evaluate()` for the
        # reasons of inserting copy operators for unaligned,
        # *unidimensional* arrays.
        func = numexpr(expr, signature, copy_args=copycols)
    except NotImplementedError, nie:
        # Try to make this Numexpr error less cryptic.
        raise _unsupported_operation_error(nie)
    params = varnames

    # This is more comfortable to handle about than a tuple.
    return CompiledCondition(func, params, idxvar, idxops, idxlims)

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
        # This is needed because the extension doesn't check for
        # unaligned arrays anymore. The reason for doing this is that,
        # for unaligned arrays, a pure copy() in Python is faster than
        # the equivalent in C. I'm not completely sure why.
        if not arg.flags.aligned and arg.ndim > 1:
            # See the comments in `tables.numexpr.evaluate()` for the
            # reasons of copying unaligned, *multidimensional* arrays.
            arg = arg.copy()
        args.append(arg)
    return func(*args)
