#!/usr/bin/env python3

"""Example that shows how to easily save a variable number of atoms with a
VLArray."""

import numpy
import tables

N = 100
shape = (3, 3)

numpy.random.seed(10)  # For reproductible results
f = tables.open_file("vlarray3.h5", mode="w")
vlarray = f.create_vlarray(f.root, 'vlarray1',
                           tables.Float64Atom(shape=shape),
                           "ragged array of arrays")

k = 0
for i in range(N):
    l = []
    for j in range(numpy.random.randint(N)):
        l.append(numpy.random.randn(*shape))
        k += 1
    vlarray.append(l)

print("Total number of atoms:", k)
f.close()
