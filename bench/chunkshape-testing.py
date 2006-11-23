#!/usr/bin/env python

"""Simple benchmark for testing chunkshapes and maxTuples."""

import numpy, tables
from time import time

N = 20*1000
M = 30

recarray = numpy.empty(shape=2, dtype='(2,2,2)i4,(2,3,3)f8,i4,i8')

f = tables.openFile("table4.h5", mode = "w")

# t = f.createTable(f.root, 'table', recarray, "mdim recarray")

# a0 = f.createArray(f.root, 'field0', recarray['f0'], "mdim int32 array")
# a1 = f.createArray(f.root, 'field1', recarray['f1'], "mdim float64 array")

# c0 = f.createCArray(f.root, 'cfield0',
#                     tables.Int32Atom(), (2,2,2),
#                     "mdim int32 carray")
# c1 = f.createCArray(f.root, 'cfield1',
#                     tables.Float64Atom(), (2,3,3),
#                     "mdim float64 carray")
c2 = f.createCArray(f.root, 'cfield2',
                    tables.Int32Atom(), (M, N),
                    "scalar int32 carray")
# t1=time()
# c2[:] = numpy.empty(shape=(M, N), dtype="int32")
# print "carray populate time:", time()-t1

# f3 = f.createCArray(f.root, 'cfield3',
#                     tables.Float64Atom(), (3,),
#                     "scalar float64 carray", chunkshape=(32,))

# e2 = f.createEArray(f.root, 'efield2',
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
