import sys
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
hdfarray = fileh.createArray(root, 'array_c', a, "Character array",
                             extdim=0)
hdfarray.append(strings.array(['c', 'b', 'c']))
# The next is legal:
hdfarray.append(strings.array(['c', 'b', 'c', 'd']))
# but these are not:
#hdfarray.append(strings.array([['c', 'b'], ['c', 'd']]))
#hdfarray.append(array([[1,2,3],[3,2,1]], type=UInt8, shape=(2,1,3)))

# Create other Array
a = array([-1, 2, 4], Int16)
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_1', a, "Signed short array")

# Create a scalar Array
a = array(4, Int16)
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_s', a, "Scalar signed short array")

# Create an empty array
a = zeros((2,0,3),type=UInt16)
hdfarray = fileh.createArray(root, 'array_e', a, "Unsigned short array")

# Create an enlargeable array
a = zeros((2,0,3),type=UInt8)

#a = [[],[]]  # not supported
#a = []  # supported (Int32 array)
# Save it on the HDF5 file
hdfarray = fileh.createArray(root, 'array_b', a, "Unsigned byte array",
                             compress = 1)
# Append an array to this table
hdfarray.append(array([[1,2,3],[3,2,1]], type=UInt8, shape=(2,1,3)))
hdfarray.append(array([[1,2,3],[3,2,1],[2,4,6],[6,4,2]],
                      type=UInt8, shape=(2,2,3))*2)
# The next should give a type error:
#hdfarray.append(array([[1,0,1],[0,0,1]], type=Bool, shape=(2,1,3)))

# # Create an empty array with two potentially enlargeable dimensions
# # that must generate an error
# a = zeros((2,0,0),type=UInt8)
# # Save it on the HDF5 file
# hdfarray = fileh.createArray(root, 'array_d', a, "Unsigned byte array")

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
a = root.array_e.read()
print "Empty array (yes, this is suported) -->",repr(a), a.shape
a = root.array_b.read(step=2)
print "Int8 array, even rows (step = 2) -->",repr(a), a.shape

print "Testing iterator:",
for x in root.array_b.iterrows(step=2):
    print "nrow-->", root.array_b.nrow
    print "Element-->",x

arr = root.array_s
for x in arr:
    print "nrow-->", arr.nrow
    print "Element-->", repr(x)

#sys.exit()

print "Testing getitem:"
for i in range(root.array_b.nrows):
    print "array_b["+str(i)+"]", "-->", root.array_b[i]
print "array_c[1:2]", repr(root.array_c[1:2])
print "array_c[1:3]", repr(root.array_c[1:3])
print "array_b[:]", root.array_b[:]

print repr(root.array_c)
# Close the file
fileh.close()
