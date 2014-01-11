#!/usr/bin/env python

"""Example that shows how to easily save a variable number of atoms with a
VLArray."""

from __future__ import print_function
import numpy
import tables

N = 100
shape = (3, 3)

numpy.random.seed(10)  # For reproductible results
f = tables.open_file("vlarray4.h5", mode="w")
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
