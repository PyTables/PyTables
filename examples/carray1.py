import numpy as np
import tables as tb

fileName = 'carray1.h5'
shape = (200, 300)
atom = tb.UInt8Atom()
filters = tb.Filters(complevel=5, complib='zlib')

h5f = tb.open_file(fileName, 'w')
ca = h5f.create_carray(h5f.root, 'carray', atom, shape, filters=filters)
# Fill a hyperslab in ``ca``.
ca[10:60, 20:70] = np.ones((50, 50))
h5f.close()

# Re-open and read another hyperslab
h5f = tb.open_file(fileName)
print(h5f)
print(h5f.root.carray[8:12, 18:22])
h5f.close()
