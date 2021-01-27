import numpy as np
import tables as tb

basedim = 4
file = "array4.h5"
# Open a new empty HDF5 file
fileh = tb.open_file(file, mode="w")
# Get the root group
group = fileh.root
# Set the type codes to test
dtypes = [np.int8, np.uint8, np.int16, int, np.float32, float]
for i, dtype in enumerate(dtypes, 1):
    # Create an array of dtype, with incrementally bigger ranges
    a = np.ones((basedim,) * i, dtype)
    # Save it on the HDF5 file
    dsetname = f'array_{a.dtype.char}'
    hdfarray = fileh.create_array(group, dsetname, a, "Large array")
    print(f"Created dataset: {hdfarray}")
    # Create a new group
    group = fileh.create_group(group, f'group{i}')

# Close the file
fileh.close()

# Open the previous HDF5 file in read-only mode
fileh = tb.open_file(file, mode="r")
# Get the root group
group = fileh.root
# Get the metadata on the previosly saved arrays
for i, dtype in enumerate(dtypes, 1):
    # Create an array for later comparison
    a = np.ones((basedim,) * i, dtype)
    # Get the dset object hangin from group
    dset = getattr(group, 'array_' + a.dtype.char)
    print(f"Info from dataset: {dset!r}")
    # Read the actual data in array
    b = dset.read()
    print(f"Array b read from file. Shape ==> {b.shape}. Dtype ==> {b.dtype}")
    # Test if the original and read arrays are equal
    if np.allclose(a, b):
        print("Good: Read array is equal to the original")
    else:
        print("Error: Read array and the original differs!")
    # Iterate over the next group
    group = getattr(group, f'group{i}')

# Close the file
fileh.close()
