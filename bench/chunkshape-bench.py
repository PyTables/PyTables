#!/usr/bin/env python
# Benchmark the effect of chunkshapes in reading large datasets.
# You need at least PyTables 2.1 to run this!
# F. Alted

from time import perf_counter as clock
import numpy as np
import tables as tb

dim1, dim2 = 360, 6_109_666
rows_to_read = range(0, 360, 36)

print("=" * 32)
# Create the EArray
f = tb.open_file("/tmp/test.h5", "w")
a = f.create_earray(f.root, "a", tb.Float64Atom(), shape=(dim1, 0),
                    expectedrows=dim2)
print("Chunkshape for original array:", a.chunkshape)

# Fill the EArray
t1 = clock()
zeros = np.zeros((dim1, 1), dtype="float64")
for i in range(dim2):
    a.append(zeros)
tcre = clock() - t1
thcre = dim1 * dim2 * 8 / (tcre * 1024 * 1024)
print(f"Time to append {a.nrows} rows: {tcre:.3f} sec ({thcre:.1f} MB/s)")

# Read some row vectors from the original array
t1 = clock()
for i in rows_to_read:
    r1 = a[i, :]
tr1 = clock() - t1
thr1 = dim2 * len(rows_to_read) * 8 / (tr1 * 1024 * 1024)
print(f"Time to read ten rows in original array: {tr1:.3f} sec ({thr1:.1f} MB/s)")

print("=" * 32)
# Copy the array to another with a row-wise chunkshape
t1 = clock()
#newchunkshape = (1, a.chunkshape[0]*a.chunkshape[1])
newchunkshape = (1, a.chunkshape[0] * a.chunkshape[1] * 10)  # ten times larger
b = a.copy(f.root, "b", chunkshape=newchunkshape)
tcpy = clock() - t1
thcpy = dim1 * dim2 * 8 / (tcpy * 1024 * 1024)
print("Chunkshape for row-wise chunkshape array:", b.chunkshape)
print(f"Time to copy the original array: {tcpy:.3f} sec ({thcpy:.1f} MB/s)")

# Read the same ten rows from the new copied array
t1 = clock()
for i in rows_to_read:
    r2 = b[i, :]
tr2 = clock() - t1
thr2 = dim2 * len(rows_to_read) * 8 / (tr2 * 1024 * 1024)
print(f"Time to read with a row-wise chunkshape: {tr2:.3f} sec ({thr2:.1f} MB/s)")
print("=" * 32)
print(f"Speed-up with a row-wise chunkshape: {tr1 / tr2:.1f}")

f.close()
