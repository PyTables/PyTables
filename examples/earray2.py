#!/usr/bin/env python

""" Small example that shows how to work with extendeable arrays of
different types, strings included. """

import sys
from numarray import *
from numarray import strings
from tables import *

# Open a new empty HDF5 file
filename = "earray2.h5"
fileh = openFile(filename, mode = "w")
# Get the root group
root = fileh.root

# Create an string atom
a = StringAtom(shape=(0,), length=1)
# Use it as a type for the enlargeable array
hdfarray = fileh.createEArray(root, 'array_c', a, "Character array")
hdfarray.append(strings.array(['a', 'b', 'c']))
# The next is legal:
hdfarray.append(strings.array(['c', 'b', 'c', 'd']))
# but these are not:
#hdfarray.append(strings.array([['c', 'b'], ['c', 'd']]))
#hdfarray.append(array([[1,2,3],[3,2,1]], type=UInt8, shape=(2,1,3)))

# Create an atom
#a = zeros((2,0,3),type=UInt16)
a = UInt16Atom((2,0,3))
hdfarray = fileh.createEArray(root, 'array_e', a, "Unsigned short array")

# Create an enlargeable array
#a = zeros((2,0,3),type=UInt8)
a = UInt8Atom((2,0,3))
#a = [[],[]]  # not supported
#a = []  # supported (Int32 array)
hdfarray = fileh.createEArray(root, 'array_b', a, "Unsigned byte array",
                              Filters(complevel = 1))
# Append an array to this table
hdfarray.append(array([[1,2,3],[3,2,1]], type=UInt8, shape=(2,1,3)))
hdfarray.append(array([[1,2,3],[3,2,1],[2,4,6],[6,4,2]],
                      type=UInt8, shape=(2,2,3))*2)
# The next should give a type error:
#hdfarray.append(array([[1,0,1],[0,0,1]], type=Bool, shape=(2,1,3)))

# Close the file
fileh.close()

# Open the file for reading
fileh = openFile(filename, mode = "r")
# Get the root group
root = fileh.root

a = root.array_c.read()
print "Character array -->",repr(a), a.shape
a = root.array_e.read()
print "Empty array (yes, this is suported) -->",repr(a), a.shape
a = root.array_b.read(step=2)
print "Int8 array, even rows (step = 2) -->",repr(a), a.shape

print "Testing iterator:",
#for x in root.array_b.iterrows(step=2):
for x in root.array_b:
    print "nrow-->", root.array_b.nrow
    print "Element-->",x

print "Testing getitem:"
for i in range(root.array_b.shape[0]):
    print "array_b["+str(i)+"]", "-->", root.array_b[i]
# The nrows counts the growing dimension, which is different from the
# first index
for i in range(root.array_b.nrows):
    print "array_b[:,"+str(i)+",:]", "-->", root.array_b[:,i,:]
print "array_c[1:2]", repr(root.array_c[1:2])
print "array_c[1:3]", repr(root.array_c[1:3])
print "array_b[:]", root.array_b[:]

print repr(root.array_c)
# Close the file
fileh.close()
