from tables import *

class Record(IsRecord):
    field1        = '22s'  # 22-character String
    field2        = '20s'  # 20-character String

# Open a file in "w"rite mode
fileh = openFile(name = "objecttree.h5", mode = "w")
# Get the HDF5 root group
root = fileh.root

# Create the groups:
group1 = fileh.createGroup(root, "subgroup1")
group2 = fileh.createGroup(root, "subgroup2")

# Now, create a table in "subgroup0" group
table1 = fileh.createTable(root, "table0", Record())
# Create 2 new tables in subgroup1
table2 = fileh.createTable(group1, "table1", Record())
table3 = fileh.createTable("/subgroup1", "table2", Record())
# Create the last table in subgroup2
table4 = fileh.createTable("/subgroup2", "table3", Record())

# Now, fill the tables:
for table in (table1, table2, table3, table4):
    # Get the record object associated with the table:
    rec = table.record
    # Fill the table with 10 records
    for i in xrange(10):
        # First, assign the values to the Particle record
        rec.field1  = 'This is field1: %2d' % (i)
        rec.field2  = 'This is field2: %2d' % i 
        # This injects the Record values
        table.appendAsRecord(rec)      

    # Flush the table buffers
    table.flush()

# Finally, close the file (this also will flush all the remaining buffers!)
fileh.close()

