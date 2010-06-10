from tables import *

class Particle(IsDescription):
    name        = StringCol(16, pos=1)   # 16-character String
    lati        = Int32Col(pos=2)        # integer
    longi       = Int32Col(pos=3)        # integer
    pressure    = Float32Col(pos=4)      # float  (single-precision)
    temperature = Float64Col(pos=5)      # double (double-precision)

# Open a file in "w"rite mode
fileh = openFile("table1.h5", mode = "w")
# Create a new group
group = fileh.createGroup(fileh.root, "newgroup")

# Create a new table in newgroup group
table = fileh.createTable(group, 'table', Particle, "A table", Filters(1))
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

# Add a couple of user attrs
table.attrs.user_attr1=1.023
table.attrs.user_attr2="This is the second user attr"

# Append several rows in only one call
table.append([("Particle:     10", 10, 0, 10*10, 10**2),
              ("Particle:     11", 11, -1, 11*11, 11**2),
              ("Particle:     12", 12, -2, 12*12, 12**2)])

group = fileh.root.newgroup
print "Nodes under group", group,":"
for node in fileh.listNodes(group):
    print node
print

print "Leaves everywhere in file", fileh.filename,":"
for leaf in fileh.walkNodes(classname="Leaf"):
    print leaf
print

table = fileh.root.newgroup.table
print "Object:", table
print "Table name: %s. Table title: %s" % (table.name, table.title)
print "Rows saved on table: %d" % (table.nrows)

print "Variable names on table with their type:"
for name in table.colnames:
    print "  ", name, ':=', table.coldtypes[name]

print "Table contents:"
for row in table:
    print row[:]
print "Associated recarray:"
print table.read()

# Finally, close the file
fileh.close()
