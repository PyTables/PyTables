import tables
from Numeric import *   # or, from numarray import *

# Create a VLArray:
fileh = tables.openFile("vlarray1.h5", mode = "w")
vlarray = fileh.createVLArray(fileh.root, 'vlarray1',
                              tables.Int32Atom(flavor="Numeric"),
                              "ragged array of ints", compress = 1)
# Append some (variable length) rows
# All these different flavors are accepted:
vlarray.append(array([5, 6]))
vlarray.append(array([5, 6, 7], typecode='i'))
vlarray.append([5, 6, 9, 8])
vlarray.append(5, 6, 9, 10, 12)
print "Created VLArray:", repr(vlarray)

# Now, read it through an iterator
for x in vlarray:
    print vlarray.name+"["+str(vlarray.nrow)+"]-->", x

print "vlarray[3]-->", vlarray[3]
print "read-->", vlarray.read(start=3)[0]
print "vlarray[1:3]-->", vlarray[1:3:2]
print "read-->", vlarray.read(start=1, stop=3, step=2)
print "atom-->", vlarray.atom.type

# Close the file
fileh.close()
