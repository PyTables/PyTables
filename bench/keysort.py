from time import perf_counter as clock

import numpy as np
import tables as tb

N = 1000 * 1000
rnd = np.random.randint(N, size=N)

for dtype1 in ('S6', 'b1',
               'i1', 'i2', 'i4', 'i8',
               'u1', 'u2', 'u4', 'u8', 'f4', 'f8'):
    for dtype2 in ('u4', 'i8'):
        print("dtype array1, array2-->", dtype1, dtype2)
        a = np.array(rnd, dtype1)
        b = np.arange(N, dtype=dtype2)
        c = a.copy()

        t1 = clock()
        d = c.argsort()
        # c.sort()
        # e=c
        e = c[d]
        f = b[d]
        tref = clock() - t1
        print("normal sort time-->", tref)

        t1 = clock()
        tb.indexesextension.keysort(a, b)
        tks = clock() - t1
        print("keysort time-->", tks, "    {:.2f}x".format(tref / tks))
        assert np.alltrue(a == e)
        #assert numpy.alltrue(b == d)
        assert np.alltrue(f == d)
