from tables import *

class Particle(IsColDescr):
    name        = Col("CharType", 16)  # 16-character String
    lati        = Col("Int32", 1)    # integer
    longi       = Col("Int32", 1)    # integer
    pressure    = Col("Float32", 1)    # float  (single-precision)
    temperature = Col("Float64", 1)    # double (double-precision)

# Open a file in "w"rite mode
fileh = openFile("table1.h5", mode = "w")
# Create a new group
group = fileh.createGroup(fileh.root, "newgroup")

# Create a new table in newgroup group
table = fileh.createTable(group, 'table', Particle(), "Title example")
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

group = fileh.getNode(fileh.root, "newgroup", classname = 'Group')
print "Nodes under group", group,":"
for node in fileh.listNodes(group):
    print node
print

print "Leaves everywhere in file", fileh.filename,":"
for node in fileh.walkGroups():
    for leaf in fileh.listNodes(node, classname = 'Leaf'):
        print leaf
print

table = fileh.getNode("/newgroup/table", classname = 'Table')
print "Object:", table
print "Table name: %s. Table title: %s" % (table.name, table.title)
print "Rows saved on table: %d" % (table.nrows)

print "Variable names on table with their type:"
for name in table.colnames:
    print "  ", name, ':=', table.coltypes[name] 
    
# Finally, close the file
fileh.close()

