#!/usr/bin/env python

"""Simple benchmark for testing chunkshapes and nrowsinbuf."""

from __future__ import print_function
import numpy
import tables
from time import time

L = 20
N = 2000
M = 30
complevel = 1

recarray = numpy.empty(shape=2, dtype='(2,2,2)i4,(2,3,3)f8,i4,i8')

f = tables.open_file("chunkshape.h5", mode="w")

# t = f.create_table(f.root, 'table', recarray, "mdim recarray")

# a0 = f.create_array(f.root, 'field0', recarray['f0'], "mdim int32 array")
# a1 = f.create_array(f.root, 'field1', recarray['f1'], "mdim float64 array")

# c0 = f.create_carray(f.root, 'cfield0',
#                     tables.Int32Atom(), (2,2,2),
#                     "mdim int32 carray")
# c1 = f.create_carray(f.root, 'cfield1',
#                     tables.Float64Atom(), (2,3,3),
#                     "mdim float64 carray")

f1 = tables.open_file("chunkshape1.h5", mode="w")
c1 = f.create_carray(f1.root, 'cfield1',
                     tables.Int32Atom(), (L, N, M),
                     "scalar int32 carray", tables.Filters(complevel=0))

t1 = time()
c1[:] = numpy.empty(shape=(L, 1, 1), dtype="int32")
print("carray1 populate time:", time() - t1)
f1.close()


f2 = tables.open_file("chunkshape2.h5", mode="w")
c2 = f.create_carray(f2.root, 'cfield2',
                     tables.Int32Atom(), (L, M, N),
                     "scalar int32 carray", tables.Filters(complevel))

t1 = time()
c2[:] = numpy.empty(shape=(L, 1, 1), dtype="int32")
print("carray2 populate time:", time() - t1)
f2.close()

f0 = tables.open_file("chunkshape0.h5", mode="w")
e0 = f.create_earray(f0.root, 'efield0',
                     tables.Int32Atom(), (0, L, M),
                     "scalar int32 carray", tables.Filters(complevel),
                     expectedrows=N)

t1 = time()
e0.append(numpy.empty(shape=(N, L, M), dtype="int32"))
print("earray0 populate time:", time() - t1)
f0.close()

f1 = tables.open_file("chunkshape1.h5", mode="w")
e1 = f.create_earray(f1.root, 'efield1',
                     tables.Int32Atom(), (L, 0, M),
                     "scalar int32 carray", tables.Filters(complevel),
                     expectedrows=N)

t1 = time()
e1.append(numpy.empty(shape=(L, N, M), dtype="int32"))
print("earray1 populate time:", time() - t1)
f1.close()


f2 = tables.open_file("chunkshape2.h5", mode="w")
e2 = f.create_earray(f2.root, 'efield2',
                     tables.Int32Atom(), (L, M, 0),
                     "scalar int32 carray", tables.Filters(complevel),
                     expectedrows=N)

t1 = time()
e2.append(numpy.empty(shape=(L, M, N), dtype="int32"))
print("earray2 populate time:", time() - t1)
f2.close()

# t1=time()
# c2[:] = numpy.empty(shape=(M, N), dtype="int32")
# print "carray populate time:", time()-t1

# f3 = f.create_carray(f.root, 'cfield3',
#                     tables.Float64Atom(), (3,),
#                     "scalar float64 carray", chunkshape=(32,))

# e2 = f.create_earray(f.root, 'efield2',
#                     tables.Int32Atom(), (0, M),
#                     "scalar int32 carray", expectedrows=N)
# t1=time()
# e2.append(numpy.empty(shape=(N, M), dtype="int32"))
# print "earray populate time:", time()-t1

# t1=time()
# c2._f_copy(newname='cfield2bis')
# print "carray copy time:", time()-t1
# t1=time()
# e2._f_copy(newname='efield2bis')
# print "earray copy time:", time()-t1

f.close()
