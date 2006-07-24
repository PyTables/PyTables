import unittest
import warnings
import numarray.dtype
from numarray import *
from numarray.strings import array as str_array
from numexpr import E, numexpr, evaluate

# These are simplifications from the functions in ``numpy.testing``.

def assert_array_equal(x, y):
    x, y = asarray(x), asarray(y)
    message = 'arrays are not equal'
    # Compare shapes.
    if shape(x) and shape(y):
        assert shape(x) == shape(y), message
    # Compare data.
    assert alltrue(ravel(x == y)), message

def assert_array_almost_equal(x,y,decimal=6):
    x = asarray(x)
    y = asarray(y)
    message = 'arrays are not almost equal'
    # Compare shapes.
    if shape(x) and shape(y):
        assert shape(x) == shape(y), message
    # Compare data.
    reduced = ravel(less_equal(around(abs(x - y), 6), 1e-6) == 1)
    assert alltrue(reduced), message

class NumexprTestCase(unittest.TestCase):
    def setUp(self):
        # Ugly hack to keep tests the same both with numarray and numpy.
        self._missing_dtypes = False
        dtypes = numarray.dtype._dtypes
        if int not in dtypes:
            self._missing_dtypes = True
            dtypes[int] = dtypes['int32']
            dtypes[long] = dtypes['int64']
            dtypes[float] = dtypes['float64']
            dtypes[complex] = dtypes['complex128']

    def tearDown(self):
        dtypes = numarray.dtype._dtypes
        if self._missing_dtypes:
            del dtypes[int]
            del dtypes[long]
            del dtypes[float]
            del dtypes[complex]

    def warn(self, message):
        warnings.warn(message)

class test_numexpr(NumexprTestCase):
    def check_simple(self):
        ex = 2.0 * E.a + 3.0 * E.b * E.c
        func = numexpr(ex, signature=[('a', float), ('b', float), ('c', float)])
        x = func(array([1., 2, 3]), array([4., 5, 6]), array([7., 8, 9]))
        assert_array_equal(x, array([  86.,  124.,  168.]))

    def check_simple_expr_small_array(self):
        func = numexpr(E.a)
        x = arange(100.0)
        y = func(x)
        assert_array_equal(x, y)

    def check_simple_expr(self):
        func = numexpr(E.a)
        x = arange(1e5)
        y = func(x)
        assert_array_equal(x, y)

    def check_rational_expr(self):
        func = numexpr((E.a + 2.0*E.b) / (1 + E.a + 4*E.b*E.b))
        a = arange(1e5)
        b = arange(1e5) * 0.1
        x = (a + 2*b) / (1 + a + 4*b*b)
        y = func(a, b)
        assert_array_equal(x, y)

class test_evaluate(NumexprTestCase):
    def check_simple(self):
        a = array([1., 2., 3.])
        b = array([4., 5., 6.])
        c = array([7., 8., 9.])
        x = evaluate("2*a + 3*b*c")
        assert_array_equal(x, array([  86.,  124.,  168.]))

    def check_simple_expr_small_array(self):
        x = arange(100.0)
        y = evaluate("x")
        assert_array_equal(x, y)

    def check_simple_expr(self):
        x = arange(1e5)
        y = evaluate("x")
        assert_array_equal(x, y)

    def check_rational_expr(self):
        a = arange(1e5)
        b = arange(1e5) * 0.1
        x = (a + 2*b) / (1 + a + 4*b*b)
        y = evaluate("(a + 2*b) / (1 + a + 4*b*b)")
        assert_array_equal(x, y)

    def check_complex_expr(self):
        def complex(a, b, complex=__builtins__.complex):
            c = zeros(a.shape, dtype=complex)
            c.real = a
            c.imag = b
            return c
        a = arange(1e4)
        b = arange(1e4)**1e-5
        z = a + 1j*b
        x = z.imag
        x = sin(complex(a, b)).real + z.imag
        y = evaluate("sin(complex(a, b)).real + z.imag")
        assert_array_almost_equal(x, y)

tests = [
('MISC', ['b*c+d*e',
          '2*a+3*b',
          'sinh(a)',
          '2*a + (cos(3)+5)*sinh(cos(b))',
          '2*a + arctan2(a, b)',
          'where(a, 2, b)',
          'where((a-10).real, a, 2)',
          'cos(1+1)',
          '1+1',
          '1',
          'cos(a2)',
          '(a+1)**0'])]
optests = []
for op in list('+-*/%') + ['**']:
    optests.append("(a+1) %s (b+3)" % op)
    optests.append("3 %s (b+3)" % op)
    optests.append("(a+1) %s 4" % op)
    optests.append("2 %s (b+3)" % op)
    optests.append("(a+1) %s 2" % op)
    optests.append("(a+1) %s -1" % op)
    optests.append("(a+1) %s 0.5" % op)

tests.append(('OPERATIONS', optests))
cmptests = []
for op in ['<', '<=', '==', '>=', '>', '!=']:
    cmptests.append("a/2+5 %s b" % op)
    cmptests.append("a/2+5 %s 7" % op)
    cmptests.append("7 %s b" % op)
tests.append(('COMPARISONS', cmptests))
func1tests = []
# ones_like() does not work with numarray.
for func in ['copy', 'sin', 'cos', 'tan', 'sqrt', 'sinh', 'cosh', 'tanh']:
    func1tests.append("a + %s(b+c)" % func)
tests.append(('1-ARG FUNCS', func1tests))
func2tests = []
# fmod() does not work with numarray.
for func in ['arctan2']:
    func2tests.append("a + %s(b+c, d+1)" % func)
    func2tests.append("a + %s(b+c, 1)" % func)
    func2tests.append("a + %s(1, d+1)" % func)
tests.append(('2-ARG FUNCS', func2tests))
powtests = []
for n in (-2.5, -1.5, -1.3, -.5, 0, 0.5, 1, 0.5, 1, 2.3, 2.5):
    powtests.append("(a+1)**%s" % n)
tests.append(('POW TESTS', powtests))

def equal(a, b, exact):
    if exact:
        return (shape(a) == shape(b)) and alltrue(ravel(a) == ravel(b))
    else:
        return (shape(a) == shape(b)) and (allclose(ravel(a), ravel(b)) or alltrue(ravel(a) == ravel(b))) # XXX report a bug?

class Skip(Exception): pass

class test_expressions(NumexprTestCase):
    def check_expressions(self):
        for test_scalar in [0,1,2]:
            for dtype in [int, long, float, complex]:
                array_size = 100
                a = arange(array_size, dtype=dtype)
                a2 = zeros([array_size, array_size], dtype=dtype)
                b = arange(array_size, dtype=dtype) / array_size
                c = arange(array_size, dtype=dtype)
                d = arange(array_size, dtype=dtype)
                e = arange(array_size, dtype=dtype)
                if dtype == complex:
                    a = a.real
                    for x in [a2, b, c, d, e]:
                        x += 1j
                        x *= 1+1j
                if test_scalar == 1:
                    a = asarray(a[array_size/2])
                if test_scalar == 2:
                    b = asarray(b[array_size/2])
                for optimization, exact in [('none', False), ('moderate', False), ('aggressive', False)]:
                    for section_name, section_tests in tests:
                        for expr in section_tests:
                            if dtype == complex and (
                                   '<' in expr or '>' in expr or '%' in expr
                                   or "arctan2" in expr or "fmod" in expr):
                                continue # skip complex comparisons
                            try:
                                try:
                                    npval = eval(expr)
                                except:
                                    raise Skip()
                                neval = evaluate(expr, optimization=optimization)
                                assert equal(npval, neval, exact), "%s (%s, %s, %s, %s)" % (expr, test_scalar, dtype.__name__, optimization, exact)
                            except Skip:
                                pass
                            except AssertionError:
                                raise
                            except NotImplementedError:
                                self.warn('%r not implemented for %s' % (expr,dtype.__name__))
                            except:
                                self.warn('numexpr error for expression %r' % (expr,))
                                raise

class test_int32_int64(NumexprTestCase):
    def check_small_long(self):
        # Small longs should not be downgraded to ints.
        res = evaluate('42L')
        assert_array_equal(res, 42)
        self.assertEqual(res.dtype.name, 'int64')

    def check_big_int(self):
        # Big ints should be promoted to longs.
        # This test may only fail under 64-bit platforms.
        res = evaluate('2**40')
        assert_array_equal(res, 2**40)
        self.assertEqual(res.dtype.name, 'int64')

    def check_long_constant_promotion(self):
        int32array = arange(100, dtype='int32')
        res = int32array * 2
        res32 = evaluate('int32array * 2')
        res64 = evaluate('int32array * 2L')
        assert_array_equal(res, res32)
        assert_array_equal(res, res64)
        self.assertEqual(res32.dtype.name, 'int32')
        self.assertEqual(res64.dtype.name, 'int64')

    def check_int64_array_promotion(self):
        int32array = arange(100, dtype='int32')
        int64array = arange(100, dtype='int64')
        respy = int32array * int64array
        resnx = evaluate('int32array * int64array')
        assert_array_equal(respy, resnx)
        self.assertEqual(resnx.dtype.name, 'int64')

class test_strings(NumexprTestCase):
    str_array1 = str_array(['foo', 'bar', '', '  '])
    str_array2 = str_array(['foo', '', 'x', ' '])
    str_constant = 'doodoo'

    def check_compare_array(self):
        sarr1 = self.str_array1
        sarr2 = self.str_array2
        expr = 'sarr1 >= sarr2'
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def check_compare_variable(self):
        sarr = self.str_array1
        svar = self.str_constant
        expr = 'sarr >= svar'
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def check_compare_constant(self):
        sarr = self.str_array1
        expr = 'sarr >= %r' % self.str_constant
        res1 = eval(expr)
        res2 = evaluate(expr)
        assert_array_equal(res1, res2)

    def check_add_string_array(self):
        sarr1 = self.str_array1
        sarr2 = self.str_array2
        expr = 'sarr1 + sarr2'
        self.assert_missing_op('add_sss', expr, locals())

    def check_add_numeric_array(self):
        sarr = self.str_array1
        narr = arange(len(sarr))
        expr = 'sarr >= narr'
        self.assert_missing_op('ge_bsi', expr, locals())

    def assert_missing_op(self, op, expr, local_dict):
        msg = "expected NotImplementedError regarding '%s'" % op
        try:
            evaluate(expr, local_dict)
        except NotImplementedError, nie:
            if "'%s'" % op not in nie.args[0]:
                self.fail(msg)
        else:
            self.fail(msg)

def suite():
    the_suite = unittest.TestSuite()
    the_suite.addTest(unittest.makeSuite(test_numexpr, prefix='check'))
    the_suite.addTest(unittest.makeSuite(test_evaluate, prefix='check'))
    the_suite.addTest(unittest.makeSuite(test_expressions, prefix='check'))
    the_suite.addTest(unittest.makeSuite(test_int32_int64, prefix='check'))
    the_suite.addTest(unittest.makeSuite(test_strings, prefix='check'))
    return the_suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
