import tables
from Numeric import *   # or, from numarray import *

# Create a VLArray:
fileh = tables.openFile("vlarray1.h5", mode = "w")
root = fileh.root
vlarray = fileh.createVLArray(root, 'vlarray1', tables.Int32Atom(),
                              "ragged array of ints")
vlarray.append(array([5, 6]))
vlarray.append(array([5, 6, 7]))
vlarray.append([5, 6, 9, 8])
vlarray.append(5, 6, 9, 10, 12)
print "Created VLArray:", repr(vlarray)
# Now, read it through an iterator
for x in vlarray:
    print vlarray.name+"["+str(vlarray.nrow)+"]-->", x
# Close the file
fileh.close()
