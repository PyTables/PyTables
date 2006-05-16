import numarray
import tables

fileName = 'carray1.h5'
shape = (200,300)
atom = tables.UInt8Atom(shape = (128,128))
filters = tables.Filters(complevel=5, complib='zlib')

h5f = tables.openFile(fileName,'w')
ca = h5f.createCArray(h5f.root, 'carray', shape, atom, filters=filters)
# Fill a hyperslab in ca
ca[10:60,20:70] = numarray.ones((50,50))  # Will be converted to UInt8 elements
h5f.close()

# Re-open a read another hyperslab
h5f = tables.openFile(fileName)
print h5f
print h5f.root.carray[8:12, 18:22]
h5f.close()
