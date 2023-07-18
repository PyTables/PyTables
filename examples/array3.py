import numpy as np
import tables as tb

# Open a new empty HDF5 file
fileh = tb.open_file("array3.h5", mode="w")
# Get the root group
root = fileh.root

# Create a large array
# a = reshape(array(range(2**16), "s"), (2,) * 16)
a = np.ones((2,) * 8, np.int8)
print("About to write array a")
print("  with shape: ==>", a.shape)
print("  and dtype: ==>", a.dtype)

# Save it on the HDF5 file
hdfarray = fileh.create_array(root, 'carray', a, "Large array")

# Get metadata on the previously saved array
print()
print("Info on the object:", repr(root.carray))

# Close the file
fileh.close()

# Open the previous HDF5 file in read-only mode
fileh = tb.open_file("array3.h5", mode="r")
# Get the root group
root = fileh.root

# Get metadata on the previously saved array
print()
print("Getting info on retrieved /carray object:", repr(root.carray))

# Get the actual array
# b = fileh.readArray("/carray")
# You can obtain the same result with:
b = root.carray.read()
print()
print("Array b read from file")
print("  with shape: ==>", b.shape)
print("  with dtype: ==>", b.dtype)
# print "  contents:", b

# Close the file
fileh.close()
