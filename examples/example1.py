from tables import File, IsRecord

class Particle(IsRecord):
    name        = '16s'  # 16-character String
    lati        = 'i'    # integer
    longi       = 'i'    # integer
    pressure    = 'f'    # float  (single-precision)
    temperature = 'd'    # double (double-precision)

# Open a file in "w"rite mode
fileh = File(name = "example1.h5", mode = "w")
# Get the HDF5 root group
root = fileh.getRootGroup()
# Create a new table
table = fileh.newTable(root, 'table', Particle(), "Title example")
#print "Table name ==>", table._v_name
# Get the record object associated with the table: all three ways are valid
#particle = table.record
particle = fileh.getRecordObject(table)  # This is really an accessor
#particle = fileh.getRecordObject("/table")
# Fill the table with 10 particles
for i in xrange(10):
    # First, assign the values to the Particle record
    particle.name  = 'Particle: %6d' % (i)
    particle.lati = i 
    particle.longi = 10 - i
    particle.pressure = float(i*i)
    particle.temperature = float(i**2)
    # This injects the Record values. Both ways do that.
    #table.appendRecord(particle)      
    fileh.appendRecord(table, particle)      

# Finally, close the file
fileh.close()

