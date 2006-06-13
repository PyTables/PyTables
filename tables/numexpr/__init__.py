from tables.numexpr.info import __doc__
from tables.numexpr.expressions import E
from tables.numexpr.compiler import numexpr, disassemble, evaluate

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    NumpyTest().test(level, verbosity)
