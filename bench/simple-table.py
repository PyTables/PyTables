#!/usr/bin/env python

import numpy as np
import tables as tb
from time import time

# Size of the buffer to be appended
M = 1_000_000
# Number of rows in table
N = 100 * M

filename = "simple-table-blosc2.h5"
filters = tb.Filters(9, "blosc2", shuffle=True)
#filters = tb.Filters(9, "zlib", shuffle=True)
#dt = np.dtype([('int32', np.int32), ('float32', np.float32, 10)])
dt = np.dtype([('int32', np.int32), ('float32', np.float32)])

a = np.fromiter(((i, i) for i in range(M)), dtype=dt)
#a = np.zeros(M, dtype=dt)

if 1:
    output_file = tb.open_file(filename, mode="w", PYTABLES_SYS_ATTRS=False)
    table = output_file.create_table("/", "test", dt, filters=filters, expectedrows=N)#, chunkshape=131072)
    chunkshape = table.chunkshape[0]
    print("chunkshape:", chunkshape)

    t0 = time()
    for i in range(0, N, M):
        table.append(a)
    #table.append(a[1:-1])
    table.flush()
    print(f"Time for storing: {time() - t0 : .3f}s")

    output_file.close()

print("Start reads:")

output_file = tb.open_file(filename)
table = output_file.root.test

t0 = time()
result = 0
# for i in range(0, N, 8192):
#   result += table[i]['int32']
# for row in table:
#     result += row['int32']
b = table[:]
print(f"Time for reading: {time() - t0 : .3f}s")
print("result:", result)

output_file.close()
