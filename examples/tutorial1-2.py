"""This example shows how to browse the object tree and enlarge tables.

Before to run this program you need to execute first tutorial1-1.py
that create the tutorial1.h5 file needed here.

"""


import sys
from numarray import *
from tables import *

# Filename to work with

filename="tutorial1.h5"

print
print	'-**-**-**-**- open the previous tutorial file -**-**-**-**-**-'

# Reopen the file in append mode
h5file = openFile(filename, "a")

# Print the object tree created from this filename
print "Object tree from filename:", h5file.filename
print h5file

print
print	'-**-**-**-**-**-**- traverse tree methods -**-**-**-**-**-**-**-'

# List all the nodes (Group and Leaf objects) on tree
print h5file

# List all the nodes (using File iterator) on tree
print "Nodes in file:"
for node in h5file:
    print node
print

# Now, only list all the groups on tree
print "Groups in file:"
for group in h5file.walkNodes(classname="Group"):
    print group
print

# List only the arrays hanging from /
print "Arrays in file (I):"
for group in h5file.walkGroups("/"):
    for array in h5file.listNodes(group, classname = 'Array'):
        print array

# This do the same result
print "Arrays in file (II):"
for array in h5file.walkNodes("/", "Array"):
    print array
print
# And finally, list only leafs on /detector group (there should be one!)
print "Leafs in group '/detector' (I):"
for leaf in h5file.listNodes("/detector", 'Leaf'):
    print leaf

# Other way using iterators and natural naming
print "Leafs in group '/detector' (II):"
for leaf in h5file.root.detector._f_walkNodes('Leaf'):
    print leaf



print
print	'-**-**-**-**-**-**- setting/getting object attributes -**-**--**-**-'

# Get a pointer to '/detector' and '/detector/readout' nodes
detector = h5file.root.detector
table = h5file.root.detector.readout

# Attach it a string (date) attribute
table.attrs.gath_date = "Wed, 06/12/2003 18:33"

# Attach a floating point attribute
table.attrs.temperature = 18.4
table.attrs.temp_scale = "Celsius"

# Attach a general object to the parent (/detector) group
detector._v_attrs.stuff = [5, (2.3, 4.5), "Integer and tuple"]

# Now, get the attributes
print "gath_date attribute of /detector/readout:", table.attrs.gath_date
print "temperature attribute of /detector/readout:", table.attrs.temperature
print "temp_scale attribute of /detector/readout:", table.attrs.temp_scale
print "stuff attribute in /detector:", detector._v_attrs.stuff
print

# Delete permanently the attribute gath_date of /detector/readout
print "Deleting /detector/readout gath_date attribute"
del table.attrs.gath_date

# Print a representation of all attributes in  /detector/table
print "AttributeSet instance in /detector/table:", repr(table.attrs)

# Get the (user) attributes of /detector/table
print "List of user attributes in /detector/table:", table.attrs._f_list()

# Get the (sys) attributes of /detector/table
print "List of user attributes in /detector/table:", table.attrs._f_list("sys")
print
# Rename an attribute
print "renaming 'temp_scale' attribute to 'tempScale'"
table.attrs._f_rename("temp_scale","tempScale")

# Try to rename a system attribute:
try:
    table.attrs._v_rename("VERSION", "version")
except:
    print "You can not rename a VERSION attribute: it is read only!."

# Get all the attributes of /detector/table
print "List of all attributes in /detector:", detector._v_attrs._f_list("all")

print
print	'-**-**-**-**-**-**- getting object metadata -**-**-**-**-**-**-'

# Get a pointer to '/detector/readout' data
table = h5file.root.detector.readout

# Get metadata from table
print "Object:", table
print "Table name:", table.name
print "Table title:", table.title
print "Number of rows in table:", table.nrows
print "Table variable names with their type and shape:"
for name in table.colnames:
    print name, ':= %s, %s' % (table.coltypes[name], table.colshapes[name])
print    

# Get the object in "/columns pressure"
pressureObject = h5file.getNode("/columns", "pressure")

# Get some metadata on this object
print "Info on the object:", repr(pressureObject)
print
print	'-**-**-**-**-**- reading actual data from arrays -**-**-**-**-**-**-'

# Read the 'pressure' actual data
pressureArray = pressureObject.read()

# Read the 'name' Array actual data
nameArray = h5file.root.columns.name.read()

# Check the kind of object we have created (they should be numarray arrays)
print "pressureArray is an object of type:", type(pressureArray)
print "nameArray is an object of type:", type(nameArray)
print

# Print the data for both arrays
print "Data on arrays nameArray and pressureArray:"
for i in range(pressureObject.shape[0]):
    print nameArray[i], "-->", pressureArray[i]

print
print	'-**-**-**-**-**- reading actual data from tables -**-**-**-**-**-**-'

# Create a shortcut to table object
table = h5file.root.detector.readout

# Read the 'energy' column of '/detector/readout'
print "Column 'energy' of '/detector/readout':\n", table["energy"]
print
# Read the 3rd row of '/detector/readout'
print "Third row of '/detector/readout':\n", table[2]
print
# Read the rows from 3 to 9 of row of '/detector/readout'
print "Rows from 3 to 9 of '/detector/readout':\n", table[2:9]

print
print	'-**-**-**-**- append records to existing table -**-**-**-**-**-'

# Get the object row from table
particle = table.row

# Append 5 new particles to table
for i in xrange(10, 15):
    particle['name']  = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256    
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i 
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)  # This exceeds long integer range
    particle.append()

# Flush this table
table.flush()

# Print the data using the table iterator:
for r in table:
    print "%-16s | %11.1f | %11.4g | %6d | %6d | %8d |" % \
          (r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'], 
           r['TDCcount'])

print
print "Total number of entries in resulting table:", table.nrows

print
print	'-**-**-**-**- remove records from a table -**-**-**-**-**-'

# Delete some rows on the Table (yes, rows can be removed!)
table.removeRows(5,10)

# Print some table columns, for comparison with array data
print "Some columns in final table:"
print
# Print the headers
print "%-16s | %11s | %11s | %6s | %6s | %8s |" % \
       ('name', 'pressure', 'energy', 'grid_i', 'grid_j', 
        'TDCcount')

print "%-16s + %11s + %11s + %6s + %6s + %8s +" % \
      ('-' * 16, '-' * 11, '-' * 11, '-' * 6, '-' * 6, '-' * 8)
# Print the data using the table iterator:
for r in table.iterrows():
    print "%-16s | %11.1f | %11.4g | %6d | %6d | %8d |" % \
          (r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'], 
           r['TDCcount'])

print
print "Total number of entries in final table:", table.nrows

# Close the file
h5file.close()
