import numpy as np
import tables as tb

# Open a new empty HDF5 file
fileh = tb.open_file("array1.h5", mode="w")
# Get the root group
root = fileh.root

# Create an Array
a = np.array([-1, 2, 4], np.int16)
# Save it on the HDF5 file
hdfarray = fileh.create_array(root, 'array_1', a, "Signed short array")

# Create a scalar Array
a = np.array(4, np.int16)
# Save it on the HDF5 file
hdfarray = fileh.create_array(root, 'array_s', a, "Scalar signed short array")

# Create a 3-d array of floats
a = np.arange(120, dtype=np.float64).reshape(20, 3, 2)
# Save it on the HDF5 file
hdfarray = fileh.create_array(root, 'array_f', a, "3-D float array")

# Close the file
fileh.close()

# Open the file for reading
fileh = tb.open_file("array1.h5", mode="r")
# Get the root group
root = fileh.root

a = root.array_1.read()
print("Signed byte array -->", repr(a), a.shape)

print("Testing iterator (works even over scalar arrays):", end=' ')
arr = root.array_s
for x in arr:
    print("nrow-->", arr.nrow)
    print("Element-->", repr(x))

# print "Testing getitem:"
# for i in range(root.array_1.nrows):
#     print "array_1["+str(i)+"]", "-->", root.array_1[i]

print("array_f[:,2:3,2::2]", repr(root.array_f[:, 2:3, 2::2]))
print("array_f[1,2:]", repr(root.array_f[1, 2:]))
print("array_f[1]", repr(root.array_f[1]))

# Close the file
fileh.close()
