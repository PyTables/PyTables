"""
Utility functions and classes for supporting query conditions.

:Author:   Ivan Vilata i Balaguer
:Contact:  ivan@selidor.net
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

from tables.numexpr.necompiler import stringToExpression, NumExpr
from tables.utilsExtension import getNestedField
from tables._conditions_common import _unsupported_operation_error
from tables.utils import lazyattr

try:
    from _conditions_pro import _get_idx_expr
except ImportError:
    # Dummy version of get_idx_expr
    def _get_idx_expr(exprnode, indexedcols):
        return ([], [''])  # This tuple means "not_indexable"

class CompiledCondition(object):
    """Container for a compiled condition."""

    # Lazy attributes
    # ```````````````
    @lazyattr
    def index_variables(self):
        """The columns participating in the index expression."""
        idxexprs = self.index_expressions
        idxvars = []
        for expr in idxexprs:
            idxvar = expr[0]
            if idxvar not in idxvars:
                idxvars.append(idxvar)
        return frozenset(idxvars)


    def __init__(self, func, params, idxexprs, strexpr):
        self.function = func
        """The compiled function object corresponding to this condition."""
        self.parameters = params
        """A list of parameter names for this condition."""
        self.index_expressions = idxexprs
        """A list of expressions in the form ``(var, (ops), (limits))``."""
        self.string_expression = strexpr
        """The indexable expression in string format."""

    def __repr__(self):
        return ( "idxexprs: %s\nstrexpr: %s\nidxvars: %s"
                 % ( self.index_expressions, self.string_expression,
                     self.index_variables) )


    def with_replaced_vars(self, condvars):
        """
        Replace index limit variables with their values in-place.

        A new compiled condition is returned.  Values are taken from
        the `condvars` mapping and converted to Python scalars.
        """
        exprs = self.index_expressions
        exprs2 = []
        for expr in exprs:
            idxlims = expr[2]  # the limits are in third place
            limit_values = []
            for idxlim in idxlims:
                if type(idxlim) is tuple:  # variable
                    idxlim = condvars[idxlim[0]]  # look up value
                    idxlim = idxlim.tolist()  # convert back to Python
                limit_values.append(idxlim)
            # Add this replaced entry to the new exprs2
            var, ops, _ = expr
            exprs2.append((var, ops, tuple(limit_values)))
        # Create a new container for the converted values
        newcc = CompiledCondition(
            self.function, self.parameters, exprs2, self.string_expression )
        return newcc


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
    `indexedcols`.  The part of `condition` having usable indexes is
    returned as a compiled condition in a `CompiledCondition` container.

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
    idxexprs = _get_idx_expr(expr, indexedcols)
    # Post-process the answer
    if type(idxexprs) == list:
        # Simple expression
        strexpr = ['e0']
    else:
        # Complex expression
        idxexprs, strexpr = idxexprs
    # Get rid of the unneccessary list wrapper for strexpr
    strexpr = strexpr[0]

    # Get the variable names used in the condition.
    # At the same time, build its signature.
    varnames = _get_variable_names(expr)
    signature = [(var, typemap[var]) for var in varnames]
    try:
        # See the comments in `tables.numexpr.evaluate()` for the
        # reasons of inserting copy operators for unaligned,
        # *unidimensional* arrays.
        func = NumExpr(expr, signature, copy_args=copycols)
    except NotImplementedError, nie:
        # Try to make this Numexpr error less cryptic.
        raise _unsupported_operation_error(nie)
    params = varnames

    # This is more comfortable to handle about than a tuple.
    return CompiledCondition(func, params, idxexprs, strexpr)


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
