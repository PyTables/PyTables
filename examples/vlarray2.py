#!/usr/bin/env python
# -*- coding: latin-1 -*-

""" Small example that shows how to work with variable length arrays of
different types, UNICODE strings and general Python objects included. """

from numarray import *
from numarray import strings
from numarray import records
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
a = array([[1,2],[1, 2]], type=Int64)
vlarray.append(a)
vlarray.append(array([[1,2],[3, 4]], type=Int64))
vlarray.append(zeros(type=Int64, shape=(0,2)))
vlarray.append(array([[5, 6]], type=Int64))
# This makes an error (shape)
#vlarray.append(array([[5], [6]], type=Int64))
# This makes an error (type)
#vlarray.append(array([[5, 6]], type=UInt64))

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
vlarray.append(array(cPickle.dumps((["123", "456"], "3")),type=UInt8))

# In next example, the length of the array should be the same than before,
# but it is not: it is sligthly greater!. This should be investigated?
# However, both approachs seems to work well
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

# Unicode variable length strings (latin-1 encoding
vlarray = fileh.createVLArray(root, 'vlarray7', VLStringAtom(),
                              "Variable Length String")
vlarray.append(u"asd")
vlarray.append(u"aaañá")

# Unicode variable length strings (utf-8 encoding)
vlarray = fileh.createVLArray(root, 'vlarray8', VLStringAtom(),
                               "Variable Length String")
vlarray.append(u"aaañá")
vlarray.append(u"")   # The empty string
vlarray.append(u"asd")

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
