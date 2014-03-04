from __future__ import print_function
from tables.indexesextension import keysort
import numpy
from time import time

N = 1000 * 1000
rnd = numpy.random.randint(N, size=N)

for dtype1 in ('S6', 'b1',
               'i1', 'i2', 'i4', 'i8',
               'u1', 'u2', 'u4', 'u8', 'f4', 'f8'):
    for dtype2 in ('u4', 'i8'):
        print("dtype array1, array2-->", dtype1, dtype2)
        a = numpy.array(rnd, dtype1)
        b = numpy.arange(N, dtype=dtype2)
        c = a.copy()

        t1 = time()
        d = c.argsort()
        # c.sort()
        # e=c
        e = c[d]
        f = b[d]
        tref = time() - t1
        print("normal sort time-->", tref)

        t1 = time()
        keysort(a, b)
        tks = time() - t1
        print("keysort time-->", tks, "    %.2fx" % (tref / tks,))
        assert numpy.alltrue(a == e)
        #assert numpy.alltrue(b == d)
        assert numpy.alltrue(f == d)
