from numpy import *
from tables import *

basedim = 4
file = "array4.h5"
# Open a new empty HDF5 file
fileh = openFile(file, mode = "w")
# Get the root group
group = fileh.root
# Set the type codes to test
dtypes = [int8, uint8, int16, int, float32, float]
i = 1
for dtype in dtypes:
    # Create an array of dtype, with incrementally bigger ranges
    a = ones((basedim,) * i, dtype)
    # Save it on the HDF5 file
    dsetname = 'array_' + a.dtype.char
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
for i in range(len(dtypes)):
    # Create an array for later comparison
    a = ones((basedim,) * (i+1), dtypes[i])
    # Get the dset object hangin from group
    dset = getattr(group, 'array_' + a.dtype.char)
    print "Info from dataset:", repr(dset)
    # Read the actual data in array
    b = dset.read()
    print "Array b read from file. Shape ==>", b.shape,
    print ". Dtype ==> %s" % b.dtype
    # Test if the original and read arrays are equal
    if allclose(a, b):
        print "Good: Read array is equal to the original"
    else:
        print "Error: Read array and the original differs!"
    # Iterate over the next group
    group = getattr(group, 'group' + str(i+1))

# Close the file
fileh.close()
