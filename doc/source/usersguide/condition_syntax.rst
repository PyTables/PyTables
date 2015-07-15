.. _condition_syntax:

Condition Syntax
================
.. currentmodule:: tables

Conditions in PyTables are used in methods related with in-kernel and indexed
searches such as :meth:`Table.where` or :meth:`Table.read_where`.
They are interpreted using Numexpr, a powerful package for achieving C-speed
computation of array operations (see :ref:`[NUMEXPR] <NUMEXPR>`).

A condition on a table is just a *string* containing a Python expression
involving *at least one column*, and maybe some constants and external
variables, all combined with algebraic operators and functions. The result of
a valid condition is always a *boolean array* of the same length as the
table, where the *i*-th element is true if the value of the expression on the
*i*-th row of the table evaluates to true

That is the reason why multidimensional fields in a table are not supported
in conditions, since the truth value of each resulting multidimensional
boolean value is not obvious.
Usually, a method using a condition will only consider the rows where the
boolean result is true.

For instance, the condition 'sqrt(x*x + y*y) < 1' applied on a table with x
and y columns consisting of floating point numbers results in a boolean array
where the *i*-th element is true if (unsurprisingly) the value of the square
root of the sum of squares of x and y is less than 1.
The sqrt() function works element-wise, the 1 constant is adequately
broadcast to an array of ones of the length of the table for evaluation, and
the *less than* operator makes the result a valid boolean array. A condition
like 'mycolumn' alone will not usually be valid, unless mycolumn is itself a
column of scalar, boolean values.

In the previous conditions, mycolumn, x and y are examples of *variables*
which are associated with columns.
Methods supporting conditions do usually provide their own ways of binding
variable names to columns and other values. You can read the documentation of
:meth:`Table.where` for more information on that. Also, please note that the
names None, True and False, besides the names of functions (see below) *can
not be overridden*, but you can always define other new names for the objects
you intend to use.

Values in a condition may have the following types:

- 8-bit boolean (bool).

- 32-bit signed integer (int).

- 64-bit signed integer (long).

- 32-bit, single-precision floating point number (float or float32).

- 64-bit, double-precision floating point number (double or float64).

- 2x64-bit, double-precision complex number (complex).

- Raw string of bytes (str).

Nevertheless, if the type passed is not among the above ones, it will be
silently upcasted, so you don't need to worry too much about passing
supported types, except for the Unsigned 64 bits integer, that cannot be
upcasted to any of the supported types.

However, the types in PyTables conditions are somewhat stricter than those of
Python. For instance, the *only* valid constants for booleans are True and
False, and they are *never* automatically cast to integers. The type
strengthening also affects the availability of operators and functions.
Beyond that, the usual type inference rules apply.

Conditions support the set of operators listed below:

- Logical operators: &, \|, ~.

- Comparison operators: <, <=, ==, !=, >=, >.

- Unary arithmetic operators: -.

- Binary arithmetic operators: +, -, \*, /, \**, %.

Types do not support all operators. Boolean values only support logical and
strict (in)equality comparison operators, while strings only support
comparisons, numbers do not work with logical operators, and complex
comparisons can only check for strict (in)equality. Unsupported operations
(including invalid castings) raise NotImplementedError exceptions.

You may have noticed the special meaning of the usually bitwise operators &,
| and ~. Because of the way Python handles the short-circuiting of logical
operators and the truth values of their operands, conditions must use the
bitwise operator equivalents instead.
This is not difficult to remember, but you must be careful because bitwise
operators have a *higher precedence* than logical operators. For instance,
'a and b == c' (*a is true AND b is equal to c*) is *not* equivalent to
'a & b == c' (*a AND b is equal to c)*. The safest way to avoid confusions is
to *use parentheses* around logical operators, like this: 'a & (b == c)'.
Another effect of short-circuiting is that expressions like '0 < x < 1' will
*not* work as expected; you should use '(0 < x) & (x < 1)'.

All of this may be solved if Python supported overloadable boolean operators
(see PEP 335) or some kind of non-shortcircuiting boolean operators (like C's
&&, || and !).

You can also use the following functions in conditions:

- where(bool, number1, number2):
  number - number1 if the bool condition is true, number2 otherwise.

- {sin,cos,tan}(float|complex):
  float|complex - trigonometric sine, cosine or tangent.

- {arcsin,arccos,arctan}(float|complex):
  float|complex - trigonometric inverse sine, cosine or tangent.

- arctan2(float1, float2):
  float - trigonometric inverse tangent of float1/float2.

- {sinh,cosh,tanh}(float|complex):
  float|complex - hyperbolic sine, cosine or tangent.

- {arcsinh,arccosh,arctanh}(float|complex):
  float|complex - hyperbolic inverse sine, cosine or tangent.

- {log,log10,log1p}(float|complex):
  float|complex - natural, base-10 and log(1+x) logarithms.

- {exp,expm1}(float|complex):
  float|complex - exponential and exponential minus one.

- sqrt(float|complex): float|complex - square root.

- abs(float|complex): float|complex - absolute value.

- {real,imag}(complex):
  float - real or imaginary part of complex.

- complex(float, float):
  complex - complex from real and imaginary parts.

