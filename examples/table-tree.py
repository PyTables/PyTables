import sys

from Numeric import *
from tables import *

# Define a user record to caracterize some kind of particles
class Particle(IsRecord):
    name        = '16s'  # 16-character String
    TDCcount    = 'B'    # unsigned byte
    ADCcont     = 'H'    # unsigned short integer
    grid_i      = 'i'    # integer
    grid_j      = 'i'    # integer
    pressure    = 'f'    # float  (single-precision)
    temperature = 'd'    # double (double-precision)
    idnumber    = 'Q'    # unsigned long long (i.e. 64-bit integer)

# The name of our HDF5 filename
filename = "table-tree.h5"
    
# Open a file in "w"rite mode
h5file = openFile(filename, mode = "w")

# Create a new group under "/" (root)
group = h5file.createGroup("/", 'detector')

# Create one table on it
table = h5file.createTable(group, 'table', Particle(), "Title example")

# Create a shortcut to the table record object
particle = table.record

# Fill the table with 10 particles
for i in xrange(10):
    # First, assign the values to the Particle record
    particle.name  = 'Particle: %6d' % (i)
    particle.TDCcount = i % 256    
    particle.ADCcont = (i * 256) % (1 << 16)
    particle.grid_i = i 
    particle.grid_j = 10 - i
    particle.pressure = float(i*i)
    particle.temperature = float(i**2)
    particle.idnumber = i * (2 ** 34)  # This exceeds integer range
    # This injects the Record values.
    table.appendAsRecord(particle)      

# Flush the buffers for table
table.flush()

# Get actual data from table. We are interested in column pressure.
pressure = [ p.pressure for p in table.readAsRecords() ]
print "Last record ==>", p
print "Column pressure ==>", pressure
print "Total records in table ==> ", len(pressure)
print

# Create a new group to hold new arrays
gcolumns = h5file.createGroup("/", "columns")

# Create a Numeric array with this info under '/columns'
h5file.createArray(gcolumns, 'pressure', array(pressure), "Pressure column")

# Do the same with name column
names = [ p.name for p in table.readAsRecords() ]
h5file.createArray('/columns', 'name', array(names), "Name column")

# Close the file
h5file.close()


# Reopen it in append mode
h5file = openFile(filename, "a")

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

# Add code to test invalid variable names for node names
#print h5file.createGroup("/", '_ _pepe__')
#print h5file.root.__pepe__


# Get group /detector and print some info on it
detector = h5file.getNode("/detector")

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

# Get the "/detector/table
table = h5file.getNode("/detector/table", classname = 'Table')
print "/detector/table ==>", table 


# Get metadata from table
print "Object:", table
print "Table name:", table.name
print "Table title:", table.title
print "Rows saved on table: %d" % (table.nrows)

print "Variable names on table with their type:"
for i in range(len(table.varnames)):
    print "  ", table.varnames[i], ':=', table.vartypes[i] 
print    

# Read arrays in /columns/names and /columns/pressure

# Get the object in "/columns pressure"
pressureObject = h5file.getNode("/columns", "pressure")

# Get some metadata on this object
print "Info on the object:", pressureObject
print "  shape: ==>", pressureObject.shape
print "  title: ==>", pressureObject.title
print "  typecode ==> %c" % pressureObject.typecode
print

# Read the pressure actual data
pressureArray = pressureObject.read()

# Get the object in "/columns/names"
nameObject = h5file.root.columns.name

# Get some metadata on this object
print "Info on the object:", nameObject
print "  shape: ==>", nameObject.shape
print "  title: ==>", nameObject.title
print "  typecode ==> %c" % nameObject.typecode


# Read the 'name' actual data
nameArray = nameObject.read()

# Print the data for both arrays
print "Data on arrays name and pressure:"
for i in range(pressureObject.shape[0]):
    print "".join(nameArray[i]), "-->", pressureArray[i]
print


# Finally, append some new records to table
table = h5file.root.detector.table

# Get the object record from table.
# Be careful, if you want to add new records in an existent table
# you have to get this object first, and use it to feed
# the table with new records. This is because this record object has
# the table correct alignment and big/little-endian attributes.
particle = table.record
# Append 5 new particles to table (yes, tables can be enlarged!)
for i in xrange(10, 15):
    particle.name  = 'Particle: %6d' % (i)
    particle.TDCcount = i % 256    
    particle.ADCcont = (i * 256) % (1 << 16)
    particle.grid_i = i 
    particle.grid_j = 10 - i
    particle.pressure = float(i*i)
    particle.temperature = float(i**2)
    particle.idnumber = i * (2 ** 34)  # This exceeds integer range
    table.appendAsRecord(particle)
    # Faster way
    #table.appendAsValues((i * 256) % (1 << 16), i % 256, i, 10 - i, i * (2 **34), 
    #                     str("Particle: %6d" % i), float(i*i), float(i**2))

# Flush this table
table.flush()

print "Columns name and pressure on expanded table:"
# Print some table columns, for comparison with array data
for p in table.readAsRecords():
    print p.name, '-->', p.pressure
print

# Close this file
h5file.close()
