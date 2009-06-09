"""
Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

See:

http://code.google.com/p/numexpr/

for more info about it.

"""

import os.path
from tables.numexpr.expressions import E
from tables.numexpr.necompiler import NumExpr, disassemble, evaluate

import version

dirname = os.path.dirname(__file__)

__version__ = version.version

