import numpy as np
import tables as tb
from time import time


# Number of rows in table
N = 10*1000

class Record(tb.IsDescription):
    var1 = tb.StringCol(itemsize=4)  # 4-character String
    var2 = tb.IntCol()      # integer
    var3 = tb.Int16Col()    # short integer


def create(filename):
    f = tb.open_file(filename, "w")
    t = f.create_table(f.root, "table", Record)
    for i in range(N):
        t.append([('abcd', 0, i)])
    f.close()


def bench(filename, condition):
    f = tb.open_file(filename, "r")
    t = f.root.table
    s = 0
    for r in t.where(condition):
        s += r['var3']
    return s


t0 = time()
create("inkernel.h5")
print("Time for creation:", round(time()-t0, 3))

t0 = time()
bench("inkernel.h5", "var3 < 3")
print("Time for inkernel query:", round(time()-t0, 3))

#t0 = time()
#bench("inkernel.h5", lambda r: r['var3'] < 3)
#print("Time for callable query:", round(time()-t0, 3))
