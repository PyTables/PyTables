#!/usr/bin/env python

""" Small example that shows how to work with variable length arrays of
different types, UNICODE strings and general Python objects included. """

from numpy import *
from tables import *
import cPickle

# Open a new empty HDF5 file
fileh = openFile("vlarray2.h5", mode = "w")
# Get the root group
root = fileh.root

# A test with VL length arrays:
vlarray = fileh.createVLArray(root, 'vlarray1', Int32Atom(),
                              "ragged array of ints")
vlarray.append(array([5, 6]))
vlarray.append(array([5, 6, 7]))
vlarray.append([5, 6, 9, 8])

# Test with lists of bidimensional vectors
vlarray = fileh.createVLArray(root, 'vlarray2', Int64Atom(shape=(2,)),
                              "Ragged array of vectors")
a = array([[1,2],[1, 2]], dtype=int64)
vlarray.append(a)
vlarray.append(array([[1,2],[3, 4]], dtype=int64))
vlarray.append(zeros(dtype=int64, shape=(0,2)))
vlarray.append(array([[5, 6]], dtype=int64))
# This makes an error (shape)
#vlarray.append(array([[5], [6]], dtype=int64))
# This makes an error (type)
#vlarray.append(array([[5, 6]], dtype=uint64))

# Test with strings
vlarray = fileh.createVLArray(root, 'vlarray3', StringAtom(itemsize=3),
                               "Ragged array of strings")
vlarray.append(["123", "456", "3"])
vlarray.append(["456", "3"])
# This makes an error because of different string sizes than declared
#vlarray.append(["1234", "456", "3"])

# Python flavor
vlarray = fileh.createVLArray(root, 'vlarray3b', StringAtom(itemsize=3),
                              "Ragged array of strings")
vlarray.flavor = "python"
vlarray.append(["123", "456", "3"])
vlarray.append(["456", "3"])

# Binary strings
vlarray = fileh.createVLArray(root, 'vlarray4', UInt8Atom(),
                              "pickled bytes")
data = cPickle.dumps((["123", "456"], "3"))
vlarray.append(ndarray(buffer=data, dtype=uint8, shape=len(data)))

# The next is a way of doing the same than before
vlarray = fileh.createVLArray(root, 'vlarray5', ObjectAtom(),
                              "pickled object")
vlarray.append([["123", "456"], "3"])

# Boolean arrays are supported as well
vlarray = fileh.createVLArray(root, 'vlarray6', BoolAtom(),
                               "Boolean atoms")
# The next lines are equivalent...
vlarray.append([1,0])
vlarray.append([1,0,3,0])  # This will be converted to a boolean
# This gives a TypeError
#vlarray.append([1,0,1])

# Variable length strings
vlarray = fileh.createVLArray(root, 'vlarray7', VLStringAtom(),
                              "Variable Length String")
vlarray.append("asd")
vlarray.append("aaana")

# Unicode variable length strings
vlarray = fileh.createVLArray(root, 'vlarray8', VLUnicodeAtom(),
                               "Variable Length Unicode String")
vlarray.append(u"aaana")
vlarray.append(u"")   # The empty string
vlarray.append(u"asd")
vlarray.append(u"para\u0140lel")

# Close the file
fileh.close()

# Open the file for reading
fileh = openFile("vlarray2.h5", mode = "r")
# Get the root group
root = fileh.root

for object in fileh.listNodes(root, "Leaf"):
    arr = object.read()
    print object.name, "-->", arr
    print "number of objects in this row:", len(arr)

# Close the file
fileh.close()
