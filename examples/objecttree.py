import tables as tb


class Particle(tb.IsDescription):
    identity = tb.StringCol(itemsize=22, dflt=" ", pos=0)
                                                # character String
    idnumber = tb.Int16Col(dflt=1, pos=1)   # short integer
    speed = tb.Float32Col(dflt=1, pos=1)    # single-precision

# Open a file in "w"rite mode
fileh = tb.open_file("objecttree.h5", mode="w")
# Get the HDF5 root group
root = fileh.root

# Create the groups:
group1 = fileh.create_group(root, "group1")
group2 = fileh.create_group(root, "group2")

# Now, create an array in root group

# Currently PyTables arrays don't support Unicode strings,
# so we need to make sure we pass plain bytes

array1 = fileh.create_array(
    root, "array1", [b"string", b"array"], "String array")

# Create 2 new tables in group1
table1 = fileh.create_table(group1, "table1", Particle)
table2 = fileh.create_table("/group2", "table2", Particle)
# Create the last table in group2
array2 = fileh.create_array("/group1", "array2", [1, 2, 3, 4])

# Now, fill the tables:
for table in (table1, table2):
    # Get the record object associated with the table:
    row = table.row
    # Fill the table with 10 records
    for i in range(10):
        # First, assign the values to the Particle record
        row['identity'] = 'This is particle: %2d' % (i)
        row['idnumber'] = i
        row['speed'] = i * 2
        # This injects the Record values
        row.append()

    # Flush the table buffers
    table.flush()

# Finally, close the file (this also will flush all the remaining buffers!)
fileh.close()
