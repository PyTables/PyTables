from Numeric import *
from tables import *

basedim = 4
file = "array4.h5"
# Open a new empty HDF5 file
fileh = openFile(name = file, mode = "w")
# Get the root group
group = fileh.root
# Set the type codes to test
typecodes = ["c", 'b', '1', 's', 'w', 'i', 'u', 'l', 'f', 'd']
i = 1
for typecode in typecodes:
    # Create an array of typecode, with incrementally bigger ranges
    a = ones((basedim,) * i, typecode)
    # Save it on the HDF5 file
    dsetname = 'array_' + typecode
    hdfarray = fileh.createArray(group, dsetname, a, "Large array")
    print "Created dataset:", hdfarray
    # Create a new group
    group = fileh.createGroup(group, 'group' + str(i))
    # increment the range for next iteration
    i += 1

# Close the file
fileh.close()


# Open the previous HDF5 file in read-only mode
fileh = openFile(name = file, mode = "r")
# Get the root group
group = fileh.root
# Get the metadata on the previosly saved arrays
for i in range(1,10):
    # Create an array for later comparison
    a = ones((basedim,) * (i), typecodes[i-1])
    # Get the dset object hangin from group
    dset = getattr(group, 'array_' + typecodes[i-1])
    # Read the actual data in array
    b = dset.read()
    print "Info from dataset:", dset._v_pathname
    print "  Shape: ==>", dset.shape, 
    print "  typecode ==> %c" % dset.typecode
    print "Array b read from file. Shape: ==>", b.shape,
    print ". Typecode ==> %c" % b.typecode()
    assert a == b
    # Iterate over the next group
    group = getattr(group, 'group' + str(i))

# Close the file
fileh.close()
