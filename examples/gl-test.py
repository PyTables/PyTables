#!/usr/bin/env python 

import sys

# import all public PyTables stuff
from tables import *
# numarray is here, so why don't work with it?
import numarray
# If you want to use chararrays (included in numarray distribution)
import chararray
# If you want to use recarrays (included in numarray distribution)
import recarray
# If you want to use Numeric objects
# Caveat: don't use "from Numeric import *" because its namespace may
# collide with numarray! It's better to fully qualify objects from
# both libraries.
import Numeric

# The name of our HDF5 filename
filename = "gl-test.h5"

# This is an example of a translation table between python names and
# HDF5 (persistent) names
#          Python name    Persistent name
trTable = {"detector":       "for",  # A reserved word
           "recarray2":      " 11 ",}   # A non-valid python variable name
    
# Open a file in "w"rite mode
h5file = openFile(filename, mode = "w", trTable=trTable)

# Create a new group under "/" (root)
gdetector = h5file.createGroup("/", 'detector')
print "new detector group ==>", gdetector

# Get actual data from table. We are interested in column pressure.
# Create a pressure 1-D array
pressure = Numeric.array([1.2,2.3,3.4,4.5,5.6],'d')

# Create a new group to hold new arrays
gcolumns = h5file.createGroup("/", "columns")
print "new columns group ==>", gcolumns

# Create a Numeric array with this info under '/columns'
h5file.createArray(gcolumns, 'pressure', pressure,
                   "Pressure column", atomic=0)
print "gcolumns.pressure type ==> ", gcolumns.pressure.typeclass

# Create another array
TDC = [1,2,3,4,5]  # 0.3 version accepts python lists if they are homogeneous

h5file.createArray('/columns', 'TDC', TDC, "TDCcount column")

# An example with character arrays
names = [ "Name: 1", "Name: 2", "Name: 3", "Name: 4", "Name: 5" ]
h5file.createArray('/columns', 'name', names, "Name column")
# This works even with homogeneous tuples or lists
print "gcolumns.name shape ==>", gcolumns.name.shape 
print "gcolumns.name type ==> ", gcolumns.name.typeclass

# A few table examples that may be useful
# Create a 2-dimensional Numarray with 5 rows
# Save a recarray object under detector. This will become a Table object.
records = [[1.2, [1,2,3], "Name: 1", 1],
           [2.3, [3,4,5], "Name: 2", 2],
           [3.4, [5,6,7], "Name: 3", 3],
           [4.5, [7,8,9], "Name: 4", 4],
           [5.6, [9,10,11], "Name: 5", 5]]
colnames= ["First","Second","Third", "Fourth"]
r0=recarray.array(records, names=colnames)
# Another manner to create the same recarray, but using columns, follows
# Here, you must use numarray objects only
array2d = numarray.array([[1,2,3],
                          [3,4,5],
                          [5,6,7],
                          [7,8,9],
                          [9,10,11]], numarray.Int16)
r1=recarray.array([numarray.array(pressure.tolist()),  # Numeric to numarray
                   array2d,
                   chararray.array(names),  # Char arrays are useful
                   numarray.array(TDC)],
                  names=colnames)

# r0 and r1 should hold the same data
recarrt = h5file.createTable("/detector", 'recarray0', r0, "RecArray example0")
recarrt = h5file.createTable("/detector", 'recarray1', r1, "RecArray example1")

# Another example (more tricky)
r2 = r1[::2]    # Get a subset of r1 recarray object
# Change the byteorder property (just to check that endianess awareness works)
r2._byteorder = {"little":"big","big":"little"}[r2._byteorder]
recarrt = h5file.createTable("/detector", 'recarray2', r2,
                             "Non-contiguous recarray and byteorder changed!")
print repr(recarrt)

# Close the file
h5file.close()

# Reopen it in append mode
# Check what happens if you don't pass the translation table argument
h5file = openFile(filename, "a", trTable=trTable)
#h5file = openFile(filename, "a")

# Ok. let's start browsing the tree from this filename
print "Reading info from filename:", h5file.filename
print

# Firstly, list all the groups on tree
print "Groups in file:"
for group in h5file.walkGroups("/"):
    print group
print

# List all the nodes (Group and Leaf objects) on tree
print "List of all nodes in file:"
print h5file

# And finally, only the Arrays (Array objects)
print "Arrays in file:"
for group in h5file.walkGroups("/"):
    for array in h5file.listNodes(group, classname = 'Array'):
	print array
print

# Get group /detector and print some info on it
detector = h5file.getNode("/detector")
print "detector object ==>", detector

# List only leaves on detector
print "Leaves in group", detector, ":"
for leaf in h5file.listNodes("/detector", 'Leaf'):
    print leaf
print

# List only tables on detector
print "Tables in group", detector, ":"
for leaf in h5file.listNodes("/detector", 'Table'):
    print leaf
print

# List only arrays on detector (there should be none!)
print "Arrays in group", detector, ":"
for leaf in h5file.listNodes("/detector", 'Array'):
    print leaf
print


# Get "/detector" Group object
group = h5file.getNode(h5file.root, "detector", classname = 'Group')
print "/detector ==>", group 

# Get "/detector/recarray0
table = h5file.getNode("/detector/recarray0", classname = 'Table')
print "/detector/table ==>", table 


# Get metadata from table
print "Object:", table
print "Table name:", table.name
print "Table title:", table.title
print "Rows saved on table: %d" % (table.nrows)

# Print table metainfo on object and columns
print repr(table)    

# Read arrays in /columns/names and /columns/pressure

# Get the object in "/columns pressure"
pressureObject = h5file.getNode("/columns", "pressure")

# Get some metadata on this object
print "Info on the object:", str(pressureObject)
print "  shape: ==>", pressureObject.shape
print "  title: ==>", pressureObject.title
print "  type ==> ", pressureObject.typeclass
print "  byteorder ==> ", pressureObject.byteorder

# Read the pressure actual data
#pressureArray = Numeric.array(pressureObject.read().tolist())
pressureArray = pressureObject.read()
print "  data type ==>", type(pressureArray)
print "  data ==>", pressureArray
print

# More or less the same info can be get with
print "Info on the object (compact form):", repr(pressureObject)

# Get the object in "/columns/names"
nameObject = h5file.root.columns.name

# Get some metadata on this object
print "Info on the object:", repr(nameObject)

# Read the 'name' actual data
nameArray = nameObject.read()
print "  data type ==>", type(nameArray)
print "  data ==>", nameArray

# Print the data for both arrays
print "Data on arrays name and pressure:"
for i in range(pressureObject.shape[0]):
    print "".join(nameArray[i]), "-->", pressureArray[i]
print

# Print a recarray in table form
table = h5file.root.detector.recarray1
print repr(table)
print "  contents:", table[:]
print
# Finish printing a recarray with byteorder inverted
print "Warning: The next recarray byteorder has been inverted intentionally!"
table = h5file.root.detector.recarray2
print repr(table)
print "  Data:", table[:]

# Close this file
h5file.close()

print """

Excellent! The test seems to succeed.
Now, have a look at the resulting '%s' file, specially for the
translated names and the new format of arrays on disk.""" % filename
