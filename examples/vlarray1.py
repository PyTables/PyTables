import tables
from Numeric import *   # or, from numarray import *

# Create a VLArray:
fileh = tables.openFile("vlarray1.h5", mode = "w")
vlarray = fileh.createVLArray(fileh.root, 'vlarray1',
                              tables.Int32Atom(flavor="Numeric"),
                              "ragged array of ints",
                              filters = tables.Filters(1))
# Append some (variable length) rows
# All these different flavors are accepted:
vlarray.append(array([5, 6]))
vlarray.append(array([5, 6, 7]))
vlarray.append([5, 6, 9, 8])
vlarray.append(5, 6, 9, 10, 12)

# Now, read it through an iterator
for x in vlarray:
    print vlarray.name+"["+str(vlarray.nrow)+"]-->", x

# Close the file
fileh.close()
