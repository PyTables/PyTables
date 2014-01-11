from __future__ import print_function
import numpy
import tables

fileName = 'carray1.h5'
shape = (200, 300)
atom = tables.UInt8Atom()
filters = tables.Filters(complevel=5, complib='zlib')

h5f = tables.open_file(fileName, 'w')
ca = h5f.create_carray(h5f.root, 'carray', atom, shape, filters=filters)
# Fill a hyperslab in ``ca``.
ca[10:60, 20:70] = numpy.ones((50, 50))
h5f.close()

# Re-open and read another hyperslab
h5f = tables.open_file(fileName)
print(h5f)
print(h5f.root.carray[8:12, 18:22])
h5f.close()
