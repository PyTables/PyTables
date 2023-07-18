import numpy as np
import tables as tb

# Open a new empty HDF5 file
fileh = tb.open_file("attributes1.h5", mode="w", title="Testing attributes")
# Get the root group
root = fileh.root

# Create an array
a = np.array([1, 2, 4], np.int32)
# Save it on the HDF5 file
hdfarray = fileh.create_array(root, 'array', a, "Integer array")

# Assign user attributes

# A string
hdfarray.attrs.string = "This is an example"

# A Char
hdfarray.attrs.char = "1"

# An integer
hdfarray.attrs.int = 12

# A float
hdfarray.attrs.float = 12.32

# A generic object
hdfarray.attrs.object = {"a": 32.1, "b": 1, "c": [1, 2]}

# Close the file
fileh.close()
