# Eh! python!, We are going to include isolatin characters here
# -*- coding: latin-1 -*-

from numarray import *
from numarray import strings
from numarray import records
from tables import *
import cPickle

# Open a new empty HDF5 file
fileh = openFile("vlarray1.h5", mode = "w")
# Get the root group
root = fileh.root

# A test with VL length arrays:
hdfarray = fileh.createVLArray(root, 'vlarray1', Int32Atom(),
                               "ragged array if ints")
hdfarray.append(array([5, 6]))
hdfarray.append(array([5, 6, 7]))
hdfarray.append([5, 6, 9, 8])
hdfarray.append(5, 6, 9, 10, 12)

# Test with lists of bidimensional vectors
hdfarray = fileh.createVLArray(root, 'vlarray2', Int64Atom(shape=(2,)),
                               "Ragged array of vectors")
a = array([[1,2],[1, 2]], type=Int64)
hdfarray.append(a)
hdfarray.append(array([[1,2],[3, 4]], type=Int64))
hdfarray.append(array([[5, 6]], type=Int64))
# This makes an error (shape)
#hdfarray.append(array([[5], [6]], type=Int64))
# This makes an error (type)
#hdfarray.append(array([[5, 6]], type=UInt64))

# Test with strings
hdfarray = fileh.createVLArray(root, 'vlarray3', StringAtom(length=3),
                               "Ragged array of strings")
hdfarray.append(["123", "456", "3"])
hdfarray.append(["456", "3"])
# This makes an error because of different string sizes than declared
#hdfarray.append(["1234", "456", "3"])

# Binary strings
# hdfarray = fileh.createVLArray(root, 'vlarray4', UInt8Atom(),
#                                "pickled bytes")
# hdfarray.append(array(cPickle.dumps((["123", "456"], "3")),type=UInt8))

# Proper pas: fer que aco funcione...
# Aquest example deuria ser identicament igual al de dalt, doncs no ho es
# (la longitud es major!)
hdfarray = fileh.createVLArray(root, 'vlarray5', ObjectAtom(),
                               "pickled object")
hdfarray.append(["123", "456"], "3")
# Boolean arrays are supported as well
hdfarray = fileh.createVLArray(root, 'vlarray6', BoolAtom(),
                               "Boolean atoms")
# The next lines are equivalent...
hdfarray.append([1,0])
hdfarray.append(1,0,3,0)  # This will be converted to a boolean
# This gives a TypeError
#hdfarray.append([1,0,1])

# Unicode variable length strings (latin-1 encoding
hdfarray = fileh.createVLArray(root, 'vlarray7', VLString(),
                               "Variable Length String")
hdfarray.append(u"asd")
hdfarray.append(u"aaañá")

# Unicode variable length strings (utf-8 encoding)
hdfarray = fileh.createVLArray(root, 'vlarray8', VLString(),
                               "Variable Length String")
hdfarray.append(u"aaañá")
hdfarray.append(u"asd")

# Close the file
fileh.close()

# Open the file for reading
fileh = openFile("vlarray1.h5", mode = "r")
# Get the root group
root = fileh.root

# for object in fileh.listNodes(root, "Leaf"):
#     arr = object.read(step=3)
# #     for i in range(len(arr)):
# #         print object.name, "[", i, "]-->", arr[i]
#     print object.name, "-->", arr
#     print "number of objects in this row:", len(arr)

for object in fileh.listNodes(root, "Leaf"):
    i = 0
    #for atom in object.iterrows(step=2):
    #for atom in object(step=2):
    for atom in object:
        print object.name, "[", i, "]-->", atom
        i += 1
    print "number of objects in this row:", i

# Close the file
fileh.close()
