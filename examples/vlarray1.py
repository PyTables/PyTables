import tables
from Numeric import *   # or, from numarray import *

# Create a VLArray:
fileh = tables.openFile("vlarray1.h5", mode = "w")
vlarray = fileh.createVLArray(fileh.root, 'vlarray1',
                              tables.Int32Atom(shape=1, flavor="Numeric"),
                              "ragged array of ints",
                              filters = tables.Filters(1))
# Append some (variable length) rows:
vlarray.append(array([5, 6]))
vlarray.append(array([5, 6, 7]))
vlarray.append([5, 6, 9, 8])

# Now, read it through an iterator:
print "-->", vlarray.title
for x in vlarray:
    print vlarray.name+"["+str(vlarray.nrow)+"]-->", x

# Now, do the same with native python strings
vlarray2 = fileh.createVLArray(fileh.root, 'vlarray2',
                              tables.StringAtom(length=2, flavor="String"),
                              "ragged array of strings",
                              filters = tables.Filters(1))
# Append some (variable length) rows:
print "-->", vlarray2.title
vlarray2.append(["5", "66"])
vlarray2.append(["5", "6", "77"])
vlarray2.append(["5", "6", "9", "88"])

# Now, read it through an iterator:
for x in vlarray2:
    print vlarray2.name+"["+str(vlarray2.nrow)+"]-->", x


# Close the file
fileh.close()
