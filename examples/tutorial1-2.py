"""This example shows how to browse the object tree and enlarge tables.

Before to run this program you need to execute first tutorial1-1.py
that create the tutorial1.h5 file needed here.

"""


import sys
from Numeric import *
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

# Now, only list all the groups on tree
print "Groups in file:"
for group in h5file.walkGroups("/"):
    print group
print

# List only the arrays (Array objects) on tree
print "Arrays in file:"
for group in h5file.walkGroups("/"):
    for array in h5file.listNodes(group, classname = 'Array'):
	print array
print

# And finally, list only tables on /detector group (there should be one!)
print "Tables in group '/detector':"
for table in h5file.listNodes("/detector", 'Leaf'):
    print table


print
print	'-**-**-**-**-**-**- getting object metadata -**-**-**-**-**-**-'

# Get a pointer to '/detector/readout' data
table = h5file.root.detector.readout

# Get metadata from table
print "Object:", table
print "Table name:", table.name
print "Table title:", table.title
print "Number of rows in table: %d" % (table.nrows)
print "Table variable names with their type and shape:"
for name in table.colnames:
    print "  ", name, ':=', table.coltypes[name], table.colshapes[name]
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

# Check the kind of object we have created (they should be Numeric arrays)
print "pressureArray is an object of type:", type(pressureArray)
print "nameArray is an object of type:", type(nameArray)
print

# Print the data for both arrays
print "Data on arrays nameArray and pressureArray:"
for i in range(pressureObject.shape[0]):
    print nameArray[i], "-->", pressureArray[i]

print
print	'-**-**-**-**- append records to existing table -**-**-**-**-**-'

# Create a shortcut to table object
table = h5file.root.detector.readout

# Get the object row from table
particle = table.row

# Append 5 new particles to table (yes, tables can be enlarged!)
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

# Print some table columns, for comparison with array data
print "Some columns on enlarged table:"
print
# Print the headers
print "%-16s | %11s | %11s | %6s | %6s | %8s |" % \
       ('name', 'pressure', 'energy', 'grid_i', 'grid_j', 
        'TDCcount')

print "%-16s + %11s + %11s + %6s + %6s + %8s +" % \
      ('-' * 16, '-' * 11, '-' * 11, '-' * 6, '-' * 6, '-' * 8)
# Print the data
for r in table.iterrows():
    print "%-16s | %11.1f | %11.4g | %6d | %6d | %8d |" % \
       (r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'], 
        r['TDCcount'])
       
print
print "Total numbers of entries after appending new rows:", table.nrows

# Close the file
h5file.close()
