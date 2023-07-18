#!/usr/bin/env python3

"""Example that shows how to easily save a variable number of atoms with a
VLArray."""

import numpy as np
import tables as tb

N = 100
shape = (3, 3)

np.random.seed(10)  # For reproductible results
f = tb.open_file("vlarray3.h5", mode="w")
vlarray = f.create_vlarray(f.root, 'vlarray1',
                           tb.Float64Atom(shape=shape),
                           "ragged array of arrays")

k = 0
for i in range(N):
    l = []
    for j in range(np.random.randint(N)):
        l.append(np.random.randn(*shape))
        k += 1
    vlarray.append(l)

print("Total number of atoms:", k)
f.close()
