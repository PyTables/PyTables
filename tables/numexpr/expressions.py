__all__ = ['E']

import sys
import operator

import numarray as num


# XXX Is there any reason to keep Expression around?
class Expression(object):
    def __init__(self):
        object.__init__(self)

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        else:
            return VariableNode(name, 'float')

E = Expression()


def get_context():
    """Context used to evaluate expression. Typically overridden in compiler."""
    return {}

def get_optimization():
    return get_context().get('optimization', 'none')


# helper functions for creating __magic__ methods

def ophelper(f):
    def func(*args):
        args = list(args)
        for i, x in enumerate(args):
            if isinstance(x, (bool, int, long, float, complex)):
                args[i] = x = ConstantNode(x)
            if not isinstance(x, ExpressionNode):
                return NotImplemented
        return f(*args)
    return func

def all_constant(args):
    """Return true if args are all constant. Convert scalars to ConstantNodes."""
    for x in args:
        if not isinstance(x, ConstantNode):
            return False
    return True

kind_rank = ['bool', 'int', 'long', 'float', 'complex', 'none']
def common_kind(nodes):
    n = -1
    for x in nodes:
        n = max(n, kind_rank.index(x.astKind))
    return kind_rank[n]

def binop(opname, reversed=False, kind=None):
    @ophelper
    def operation(self, other):
        if reversed:
            self, other = other, self
        if all_constant([self, other]):
            return ConstantNode(getattr(self.value, "__%s__" % opname)(other.value))
        else:
            return OpNode(opname, (self, other), kind=kind)
    return operation

def func(func, minkind=None):
    @ophelper
    def function(*args):
        if all_constant(args):
            return ConstantNode(func(*[x.value for x in args]))
        kind = common_kind(args)
        if minkind and kind_rank.index(minkind) > kind_rank.index(kind):
            kind = minkind
        if hasattr(func, "__name__"):
            # numpy or python functions
            func_name = func.__name__
        else:
            # some numarray or Numeric ufuncs don't provide a __name__
            s = str(func)
            func_name = s[s.find("'")+1:s.rfind("'")]
        return FuncNode(func_name, args, kind)
    return function

@ophelper
def where_func(a, b, c):
    if isinstance(a, ConstantNode):
        raise ValueError("too many dimensions")
    if all_constant([a,b,c]):
        return ConstantNode(num.where(a, b, c))
    return FuncNode('where', [a,b,c])

@ophelper
def div_op(a, b):
    if get_optimization() in ('moderate', 'aggressive'):
        if isinstance(b, ConstantNode) and (a.astKind == b.astKind) and a.astKind in ('float', 'complex'):
            return OpNode('mul', [a, ConstantNode(1./b.value)])
    return OpNode('div', [a,b])


@ophelper
def pow_op(a, b):
    if all_constant([a,b]):
        return ConstantNode(a**b)
    if isinstance(b, ConstantNode):
        x = b.value
        if get_optimization() == 'aggressive':
            RANGE = 50 # Approximate break even point with pow(x,y)
            # Optimize all integral and half integral powers in [-RANGE, RANGE]
            # Note: for complex numbers RANGE could be larger.
            if (int(2*x) == 2*x) and (-RANGE <= abs(x) <= RANGE):
                n = int(abs(x))
                ishalfpower = int(abs(2*x)) % 2
                def multiply(x, y):
                    if x is None: return y
                    return OpNode('mul', [x, y])
                r = None
                p = a
                mask = 1
                while True:
                    if (n & mask):
                        r = multiply(r, p)
                    mask <<= 1
                    if mask > n:
                        break
                    p = OpNode('mul', [p,p])
                if ishalfpower:
                    kind = common_kind([a])
                    if kind in ('int', 'long'): kind = 'float'
                    r = multiply(r, OpNode('sqrt', [a], kind))
                if r is None:
                    r = OpNode('ones_like', [a])
                if x < 0:
                    r = OpNode('div', [ConstantNode(1), r])
                return r
        if get_optimization() in ('moderate', 'aggressive'):
            if x == -1:
                return OpNode('div', [ConstantNode(1),a])
            if x == 0:
                return FuncNode('ones_like', [a])
            if x == 0.5:
                kind = a.astKind
                if kind in ('int', 'long'): kind = 'float'
                return FuncNode('sqrt', [a], kind=kind)
            if x == 1:
                return a
            if x == 2:
                return OpNode('mul', [a,a])
    return OpNode('pow', [a,b])


functions = {
    'copy' : func(num.copy),
    'sin' : func(num.sin, 'float'),
    'cos' : func(num.cos, 'float'),
    'tan' : func(num.tan, 'float'),
    'sqrt' : func(num.sqrt, 'float'),

    'sinh' : func(num.sinh, 'float'),
    'cosh' : func(num.cosh, 'float'),
    'tanh' : func(num.tanh, 'float'),

    'arctan2' : func(num.arctan2, 'float'),

    'where' : where_func,

    'complex' : func(complex, 'complex'),
    }

# Functions only supported in numpy
if num.__name__ == 'numpy':
    functions['ones_like'] = func(num.ones_like)
    functions['fmod'] = func(num.fmod, 'float')  # Not well-supported in numarray (?)


class ExpressionNode(object):
    astType = 'generic'

    def __init__(self, value=None, kind=None, children=None):
        object.__init__(self)
        self.value = value
        if kind is None:
            kind = 'none'
        self.astKind = kind
        if children is None:
            self.children = ()
        else:
            self.children = tuple(children)

    def get_real(self):
        if self.astType == 'constant':
            return ConstantNode(complex(self.value).real)
        return OpNode('real', (self,), 'float')
    real = property(get_real)

    def get_imag(self):
        if self.astType == 'constant':
            return ConstantNode(complex(self.value).imag)
        return OpNode('imag', (self,), 'float')
    imag = property(get_imag)

    def __str__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__,
                                   self.value, self.astKind, self.children)
    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return OpNode('neg', (self,))
    def __invert__(self):
        return OpNode('invert', (self,))
    def __pos__(self):
        return self

    __add__ = __radd__ = binop('add')
    __sub__ = binop('sub')
    __rsub__ = binop('sub', reversed=True)
    __mul__ = __rmul__ = binop('mul')
    __div__ =  div_op
    __rdiv__ = binop('div', reversed=True)
    __pow__ = pow_op
    __rpow__ = binop('pow', reversed=True)
    __mod__ = binop('mod')
    __rmod__ = binop('mod', reversed=True)

    __and__ = binop('and', kind='bool')
    __or__ = binop('or', kind='bool')

    __gt__ = binop('gt', kind='bool')
    __ge__ = binop('ge', kind='bool')
    __eq__ = binop('eq', kind='bool')
    __ne__ = binop('ne', kind='bool')
    __lt__ = binop('gt', reversed=True, kind='bool')
    __le__ = binop('ge', reversed=True, kind='bool')

class LeafNode(ExpressionNode):
    leafNode = True

class VariableNode(LeafNode):
    astType = 'variable'
    def __init__(self, value=None, kind=None, children=None):
        LeafNode.__init__(self, value=value, kind=kind)
    def topython(self):
        return self.value


max_int32 = 2147483647
min_int32 = -max_int32 - 1
def normalizeConstant(x):
    # ``long`` objects are kept as is to allow the user to force
    # promotion of results by using long constants, e.g. by operating
    # a 32-bit array with a long (64-bit) constant.
    if isinstance(x, long):
        return long(x)
    # Moreover, constants needing more than 32 bits are always
    # considered ``long``, *regardless of the platform*, so we can
    # clearly tell 32- and 64-bit constants apart.
    if isinstance(x, int) and not (min_int32 <= x <= max_int32):
        return long(x)
    # ``long`` is not explicitly needed since ``int`` automatically
    # returns longs when needed (since Python 2.3).
    for converter in bool, int, float, complex:
        try:
            y = converter(x)
        except StandardError, err:
            continue
        if x == y:
            return y

def getKind(x):
    return {bool : 'bool',
            int : 'int',
            long : 'long',
            float : 'float',
            complex : 'complex'}[type(normalizeConstant(x))]

class ConstantNode(LeafNode):
    astType = 'constant'
    def __init__(self, value=None, children=None):
        kind = getKind(value)
        LeafNode.__init__(self, value=value, kind=kind)
    def __neg__(self):
        return ConstantNode(-self.value)
    def __invert__(self):
        return ConstantNode(~self.value)
    def topython(self):
        return '%s' % (self.value,)

class OpNode(ExpressionNode):
    astType = 'op'
    def __init__(self, opcode=None, args=None, kind=None):
        if (kind is None) and (args is not None):
            kind = common_kind(args)
        ExpressionNode.__init__(self, value=opcode, kind=kind, children=args)

    def topython(self):
        children = self.children
        assert 0 < len(children) < 3
        opstr = {
            'neg': '-',
            'invert': '~',
            'add': '+',
            'sub': '-',
            'mul': '*',
            'div': '/',
            'pow': '**',
            'mod': '%',
            'and': '&',
            'or': '|',
            'gt': '>',
            'ge': '>=',
            'eq': '==',
            'ne': '!=',
            'le': '<=',
            'lt': '<', }[self.value]

        left = children[0].topython()
        if len(children) == 1:
            return '%s(%s)' % (opstr, left)
        right = children[1].topython()
        return '(%s%s%s)' % (left, opstr, right)

class FuncNode(OpNode):
    def __init__(self, opcode=None, args=None, kind=None):
        if (kind is None) and (args is not None):
            kind = common_kind(args)
        OpNode.__init__(self, opcode, args, kind)

    def topython(self):
        args = ','.join(c.topython() for c in self.children)
        return '%s(%s)' % (self.opcode, args)
