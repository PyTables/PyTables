#!/usr/bin/env python
# Benchmark the effect of chunkshapes in reading large datasets.
# You need at least PyTables 2.1 to run this!
# F. Alted

from __future__ import print_function
import numpy
import tables
from time import time

dim1, dim2 = 360, 6109666
rows_to_read = range(0, 360, 36)

print("=" * 32)
# Create the EArray
f = tables.open_file("/tmp/test.h5", "w")
a = f.create_earray(f.root, "a", tables.Float64Atom(), shape=(dim1, 0),
                    expectedrows=dim2)
print("Chunkshape for original array:", a.chunkshape)

# Fill the EArray
t1 = time()
zeros = numpy.zeros((dim1, 1), dtype="float64")
for i in range(dim2):
    a.append(zeros)
tcre = round(time() - t1, 3)
thcre = round(dim1 * dim2 * 8 / (tcre * 1024 * 1024), 1)
print("Time to append %d rows: %s sec (%s MB/s)" % (a.nrows, tcre, thcre))

# Read some row vectors from the original array
t1 = time()
for i in rows_to_read:
    r1 = a[i, :]
tr1 = round(time() - t1, 3)
thr1 = round(dim2 * len(rows_to_read) * 8 / (tr1 * 1024 * 1024), 1)
print("Time to read ten rows in original array: %s sec (%s MB/s)" % (tr1,
                                                                     thr1))

print("=" * 32)
# Copy the array to another with a row-wise chunkshape
t1 = time()
#newchunkshape = (1, a.chunkshape[0]*a.chunkshape[1])
newchunkshape = (1, a.chunkshape[0] * a.chunkshape[1] * 10)  # ten times larger
b = a.copy(f.root, "b", chunkshape=newchunkshape)
tcpy = round(time() - t1, 3)
thcpy = round(dim1 * dim2 * 8 / (tcpy * 1024 * 1024), 1)
print("Chunkshape for row-wise chunkshape array:", b.chunkshape)
print("Time to copy the original array: %s sec (%s MB/s)" % (tcpy, thcpy))

# Read the same ten rows from the new copied array
t1 = time()
for i in rows_to_read:
    r2 = b[i, :]
tr2 = round(time() - t1, 3)
thr2 = round(dim2 * len(rows_to_read) * 8 / (tr2 * 1024 * 1024), 1)
print("Time to read with a row-wise chunkshape: %s sec (%s MB/s)" % (tr2,
                                                                     thr2))
print("=" * 32)
print("Speed-up with a row-wise chunkshape:", round(tr1 / tr2, 1))

f.close()
