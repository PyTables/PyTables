#!/usr/bin/env python3

"""Small example that shows how to work with variable length arrays of
different types, UNICODE strings and general Python objects included."""

import pickle
import numpy as np
import tables as tb

# Open a new empty HDF5 file
fileh = tb.open_file("vlarray2.h5", mode="w")
# Get the root group
root = fileh.root

# A test with VL length arrays:
vlarray = fileh.create_vlarray(root, 'vlarray1', tb.Int32Atom(),
                               "ragged array of ints")
vlarray.append(np.array([5, 6]))
vlarray.append(np.array([5, 6, 7]))
vlarray.append([5, 6, 9, 8])

# Test with lists of bidimensional vectors
vlarray = fileh.create_vlarray(root, 'vlarray2', tb.Int64Atom(shape=(2,)),
                               "Ragged array of vectors")
a = np.array([[1, 2], [1, 2]], dtype=np.int64)
vlarray.append(a)
vlarray.append(np.array([[1, 2], [3, 4]], dtype=np.int64))
vlarray.append(np.zeros(dtype=np.int64, shape=(0, 2)))
vlarray.append(np.array([[5, 6]], dtype=np.int64))
# This makes an error (shape)
# vlarray.append(array([[5], [6]], dtype=int64))
# This makes an error (type)
# vlarray.append(array([[5, 6]], dtype=uint64))

# Test with strings
vlarray = fileh.create_vlarray(root, 'vlarray3', tb.StringAtom(itemsize=3),
                               "Ragged array of strings")
vlarray.append(["123", "456", "3"])
vlarray.append(["456", "3"])
# This makes an error because of different string sizes than declared
# vlarray.append(["1234", "456", "3"])

# Python flavor
vlarray = fileh.create_vlarray(root, 'vlarray3b',
                               tb.StringAtom(itemsize=3),
                               "Ragged array of strings")
vlarray.flavor = "python"
vlarray.append(["123", "456", "3"])
vlarray.append(["456", "3"])

# Binary strings
vlarray = fileh.create_vlarray(root, 'vlarray4', tb.UInt8Atom(),
                               "pickled bytes")
data = pickle.dumps((["123", "456"], "3"))
vlarray.append(np.ndarray(buffer=data, dtype=np.uint8, shape=len(data)))

# The next is a way of doing the same than before
vlarray = fileh.create_vlarray(root, 'vlarray5', tb.ObjectAtom(),
                               "pickled object")
vlarray.append([["123", "456"], "3"])

# Boolean arrays are supported as well
vlarray = fileh.create_vlarray(root, 'vlarray6', tb.BoolAtom(),
                               "Boolean atoms")
# The next lines are equivalent...
vlarray.append([1, 0])
vlarray.append([1, 0, 3, 0])  # This will be converted to a boolean
# This gives a TypeError
# vlarray.append([1,0,1])

# Variable length strings
vlarray = fileh.create_vlarray(root, 'vlarray7', tb.VLStringAtom(),
                               "Variable Length String")
vlarray.append("asd")
vlarray.append("aaana")

# Unicode variable length strings
vlarray = fileh.create_vlarray(root, 'vlarray8', tb.VLUnicodeAtom(),
                               "Variable Length Unicode String")
vlarray.append("aaana")
vlarray.append("")   # The empty string
vlarray.append("asd")
vlarray.append("para\u0140lel")

# Close the file
fileh.close()

# Open the file for reading
fileh = tb.open_file("vlarray2.h5", mode="r")
# Get the root group
root = fileh.root

for object in fileh.list_nodes(root, "Leaf"):
    arr = object.read()
    print(object.name, "-->", arr)
    print("number of objects in this row:", len(arr))

# Close the file
fileh.close()
