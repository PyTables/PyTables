from numarray import *
from numarray import strings
from tables import *

# Open a new empty HDF5 file
fileh = openFile("array1.h5", mode = "w")
# Get the root group
root = fileh.root

# Create an array
a = strings.array(['1', '2', '4'])
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_c', a, "Character array")

# Create other
a = array([-1, 2, 4], Int16)
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_1', a, "Signed byte array")

# Create an empty array
a = zeros((2,0,3),type=UInt8)
a = zeros((2,0,3),type=Bool)

#a = [[],[]]  # not supported
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_b', a, "Unsigned byte array",
                             compress = 1)
# Append an array to this table
hdfarray.append(array([[1,0,1],[0,0,1]], type=Bool, shape=(2,1,3)))
# hdfarray.append(array([[1,2,3],[3,2,1]], type=UInt8, shape=(2,1,3)))#
# hdfarray.append([[[1,2,3]],[[3,2,1]]])
# hdfarray.append(array([[1,2,3],[3,2,1]]*2, type=UInt8, shape=(2,2,3)))
# hdfarray.append(array([[4,5,6],[6,5,4]], type=UInt8, shape=(2,1,3)))
# hdfarray.append(array([[7,8,9],[9,8,7]], type=UInt8, shape=(2,1,3)))

# # Create an empty array
# a = zeros((2,0,0),type=UInt8)
# # Save it on the HDF5 file
# hdfarray = fileh.createArray(root, 'array_d', a, "Unsigned byte array",
#                              compress = 1)

# # Create an empty array
# a = array([], UInt8)
# a = zeros((0),type=UInt8)
# print "-->", repr(a), a.shape
# # Save it on the HDF5 file
# hdfarray = fileh.createArray(root, 'array_b', a, "Unsigned byte array",
#                              compress = 1)
# # Append an array to this table
# hdfarray.append(array([2], type=UInt8, shape=(1,)))
# hdfarray.append(array([5], type=UInt8, shape=(1,)))
# hdfarray.append(array([6], type=UInt8, shape=(1,)))

# Close the file
fileh.close()

# Open the file for reading
fileh = openFile("array1.h5", mode = "r")
# Get the root group
root = fileh.root

a = root.array_c.read()
print "Character array -->",repr(a), a.shape
a = root.array_1.read()
print "Signed byte array -->",repr(a), a.shape
a = root.array_b.read()
print "Empty array (yes, this is suported) -->",repr(a), a.shape

# Close the file
fileh.close()
