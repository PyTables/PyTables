from tables import *

class Particle(IsRecord):
    name        = '16s'  # 16-character String
    lati        = 'i'    # integer
    longi       = 'i'    # integer
    pressure    = 'f'    # float  (single-precision)
    temperature = 'd'    # double (double-precision)

# Open a file in "w"rite mode
fileh = openFile(name = "table1.h5", mode = "w")
# Create a new group
group = fileh.createGroup(fileh.root, "newgroup")

# Create a new table in newgroup group
particle = Particle()  # First, create an instance of the user Record
table = fileh.createTable(group, 'table', particle, "Title example")

# Fill the table with 10 particles
for i in xrange(10):
    # First, assign the values to the Particle record
    particle.name  = 'Particle: %6d' % (i)
    particle.lati = i 
    particle.longi = 10 - i
    particle.pressure = float(i*i)
    particle.temperature = float(i**2)
    # This injects the Record values. Both ways do that.
    #fileh.appendAsRecord(table, particle)      
    table.appendAsRecord(particle)      

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
print "Records saved on table: %d" % (table.nrecords)

print "Variable names on table with their type:"
for i in range(len(table.varnames)):
    print "  ", table.varnames[i], ':=', table.vartypes[i] 
    
# Finally, close the file
fileh.close()

