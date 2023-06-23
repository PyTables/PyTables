import numpy as np
import tables as tb
import sys

fileName = 'carray1.h5'
shape = (200, 300)
atom = tb.UInt16Atom()
filters = tb.Filters(complevel=5, complib="blosc2:lz4")

h5f = tb.open_file(fileName, 'w')
a = np.arange(np.prod(shape), dtype="uint16").reshape(shape)

ca = h5f.create_carray(h5f.root, 'carray', atom, shape, filters=filters)
ca[...] = a
print("chunkshape: ", ca.chunkshape)
print("shape ", ca.shape)
print(ca.name)
# Fill a hyperslab in ``ca``.
# ca[10:60, 20:70] = np.ones((50, 50))
b = ca[...]
print(b)
print(b.shape)
h5f.close()
c = a[...]
print(c)
print(np.prod(c.shape))

assert np.array_equal(b, c)

# Re-open and read another hyperslab
# h5f = tb.open_file(fileName)
# print(h5f)
# print(h5f.root.carray[8:12, 18:22])
# h5f.close()
