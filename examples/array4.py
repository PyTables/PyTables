from numarray import *
from tables import *

basedim = 4
file = "array4.h5"
# Open a new empty HDF5 file
fileh = openFile(file, mode = "w")
# Get the root group
group = fileh.root
# Set the type codes to test
#typecodes = ['c', 'b', '1', 's', 'i', 'l', 'f', 'd']
typecodes = [Int8, UInt8, Int16, Int, Float32, Float]
i = 1
for typecode in typecodes:
    # Create an array of typecode, with incrementally bigger ranges
    a = ones((basedim,) * i, typecode)
    # Save it on the HDF5 file
    dsetname = 'array_' + str(typecode)
    hdfarray = fileh.createArray(group, dsetname, a, "Large array")
    print "Created dataset:", hdfarray
    # Create a new group
    group = fileh.createGroup(group, 'group' + str(i))
    # increment the range for next iteration
    i += 1

# Close the file
fileh.close()


# Open the previous HDF5 file in read-only mode
fileh = openFile(file, mode = "r")
# Get the root group
group = fileh.root
# Get the metadata on the previosly saved arrays
for i in range(1,len(typecodes)+1):
    # Create an array for later comparison
    a = ones((basedim,) * (i), typecodes[i-1])
    # Get the dset object hangin from group
    dset = getattr(group, 'array_' + str(typecodes[i-1]))
    print "Info from dataset:", repr(dset)
    # Read the actual data in array
    b = dset.read()
    print "Array b read from file. Shape ==>", b.shape,
    print ". Type ==> %s" % b.type()
    # Test if the original and read arrays are equal
    if allclose(a, b):
        print "Good: Read array is equal to the original"
    else:
        print "Error: Read array and the original differs!"
    # Iterate over the next group
    group = getattr(group, 'group' + str(i))

# Close the file
fileh.close()
