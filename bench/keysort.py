from tables.indexesExtension import keysort
import numpy
from time import time

N = 1000*1000
rnd=numpy.random.randint(N, size=N)

for dtype1 in ('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8'):
    for dtype2 in ('u4', 'i8'):
        print "dtype array1, array2-->", dtype1, dtype2
        a=numpy.array(rnd, dtype1)
        b=numpy.arange(N, dtype=dtype2)
        c=a.copy()

        t1=time()
        d=c.argsort()
        # c.sort()
        # e=c
        e=c[d]
        print "normal sort time-->", time()-t1

        t1=time()
        keysort(a, b)
        print "keysort time-->", time()-t1
        assert numpy.alltrue(a == e)
        assert numpy.alltrue(b == d)
