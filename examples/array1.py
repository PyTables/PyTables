from Numeric import *
from tables import *

# Open a new empty HDF5 file
fileh = openFile("array1.h5", mode = "w")
# Get the root group
root = fileh.root

# Create an array
a = array(['1', '2', '4'], "c")
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_c', a, "Character array")

# Create other
a = array([-1, 2, 4], "1")
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_1', a, "Signed byte array")

# This is amusing, just create another one ;-)
a = array([-1, 2, 4], "b")
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_b', a, "Unsigned byte array")

# Close the file
fileh.close()
