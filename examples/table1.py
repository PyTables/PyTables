from tables import *

class Particle(IsDescription):
    name        = StringCol(16)   # 16-character String
    lati        = IntCol()        # integer
    longi       = IntCol()        # integer
    pressure    = Float32Col()    # float  (single-precision)
    temperature = FloatCol()      # double (double-precision)

# Open a file in "w"rite mode
fileh = openFile("table1.h5", mode = "w")
# Create a new group
group = fileh.createGroup(fileh.root, "newgroup")

# Create a new table in newgroup group
table = fileh.createTable(group, 'table', Particle, "A table",1)
particle = table.row

# Fill the table with 10 particles
for i in xrange(10):
    # First, assign the values to the Particle record
    particle['name']  = 'Particle: %6d' % (i)
    particle['lati'] = i 
    particle['longi'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['temperature'] = float(i**2)
    # This injects the row values.
    particle.append()

# We need to flush the buffers in table in order to get an
# accurate number of records on it.
table.flush()

group = fileh.root.newgroup
print "Nodes under group", group,":"
for node in fileh.listNodes(group):
    print node
print

print "Leaves everywhere in file", fileh.filename,":"
for leaf in fileh(classname="Leaf"):
    print leaf
print

table = fileh.root.newgroup.table
print "Object:", table
print "Table name: %s. Table title: %s" % (table.name, table.title)
print "Rows saved on table: %d" % (table.nrows)

print "Variable names on table with their type:"
for name in table.colnames:
    print "  ", name, ':=', table.coltypes[name] 

print "Table contents:"
for row in table:
    print row
print "Associated recarray:"
print table.read()

# Finally, close the file
fileh.close()
