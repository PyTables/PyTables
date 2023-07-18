import numpy as np
import tables as tb


class Particle(tb.IsDescription):
    ADCcount = tb.Int16Col()                # signed short integer
    TDCcount = tb.UInt8Col()                # unsigned byte
    grid_i = tb.Int32Col()                  # integer
    grid_j = tb.Int32Col()                  # integer
    idnumber = tb.Int64Col()                # signed long long
    name = tb.StringCol(16, dflt="")        # 16-character String
    pressure = tb.Float32Col(shape=2)       # float  (single-precision)
    temperature = tb.Float64Col()           # double (double-precision)

Particle2 = {
    # You can also use any of the atom factories, i.e. the one which
    # accepts a PyTables type.
    "ADCcount": tb.Col.from_type("int16"),          # signed short integer
    "TDCcount": tb.Col.from_type("uint8"),          # unsigned byte
    "grid_i": tb.Col.from_type("int32"),            # integer
    "grid_j": tb.Col.from_type("int32"),            # integer
    "idnumber": tb.Col.from_type("int64"),          # signed long long
    "name": tb.Col.from_kind("string", 16),         # 16-character String
    "pressure": tb.Col.from_type("float32", (2,)),  # float
                                                        # (single-precision)
    "temperature": tb.Col.from_type("float64"),     # double
                                                        # (double-precision)
}

# The name of our HDF5 filename
filename = "table-tree.h5"

# Open a file in "w"rite mode
h5file = tb.open_file(filename, mode="w")

# Create a new group under "/" (root)
group = h5file.create_group("/", 'detector')

# Create one table on it
# table = h5file.create_table(group, 'table', Particle, "Title example")
# You can choose creating a Table from a description dictionary if you wish
table = h5file.create_table(group, 'table', Particle2, "Title example")

# Create a shortcut to the table record object
particle = table.row

# Fill the table with 10 particles
for i in range(10):
    # First, assign the values to the Particle record
    particle['name'] = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = [float(i * i), float(i * 2)]
    particle['temperature'] = float(i ** 2)
    particle['idnumber'] = i * (2 ** 34)  # This exceeds integer range
    # This injects the Record values.
    particle.append()

# Flush the buffers for table
table.flush()

# Get actual data from table. We are interested in column pressure.
pressure = [p['pressure'] for p in table.iterrows()]
print("Last record ==>", pressure)
print("Column pressure ==>", np.array(pressure))
print("Total records in table ==> ", len(pressure))
print()

# Create a new group to hold new arrays
gcolumns = h5file.create_group("/", "columns")
print("columns ==>", gcolumns, pressure)
# Create an array with this info under '/columns' having a 'list' flavor
h5file.create_array(gcolumns, 'pressure', pressure,
                    "Pressure column")
print("gcolumns.pressure type ==> ", gcolumns.pressure.atom.dtype)

# Do the same with TDCcount, but with a numpy object
TDC = [p['TDCcount'] for p in table.iterrows()]
print("TDC ==>", TDC)
print("TDC shape ==>", np.array(TDC).shape)
h5file.create_array('/columns', 'TDC', np.array(TDC), "TDCcount column")

# Do the same with name column
names = [p['name'] for p in table.iterrows()]
print("names ==>", names)
h5file.create_array('/columns', 'name', names, "Name column")
# This works even with homogeneous tuples or lists (!)
print("gcolumns.name shape ==>", gcolumns.name.shape)
print("gcolumns.name type ==> ", gcolumns.name.atom.dtype)

print("Table dump:")
for p in table.iterrows():
    print(p)

# Save a recarray object under detector
r = np.rec.array("a" * 300, formats='f4,3i4,a5,i2', shape=3)
recarrt = h5file.create_table("/detector", 'recarray', r, "RecArray example")
r2 = r[0:3:2]
# Change the byteorder property
recarrt = h5file.create_table("/detector", 'recarray2', r2,
                              "Non-contiguous recarray")
print(recarrt)
print()

print(h5file.root.detector.table.description)
# Close the file
h5file.close()

# sys.exit()

# Reopen it in append mode
h5file = tb.open_file(filename, "a")

# Ok. let's start browsing the tree from this filename
print("Reading info from filename:", h5file.filename)
print()

# Firstly, list all the groups on tree
print("Groups in file:")
for group in h5file.walk_groups("/"):
    print(group)
print()

# List all the nodes (Group and Leaf objects) on tree
print("List of all nodes in file:")
print(h5file)

# And finally, only the Arrays (Array objects)
print("Arrays in file:")
for array in h5file.walk_nodes("/", classname="Array"):
    print(array)
print()

# Get group /detector and print some info on it
detector = h5file.get_node("/detector")
print("detector object ==>", detector)

# List only leaves on detector
print("Leaves in group", detector, ":")
for leaf in h5file.list_nodes("/detector", 'Leaf'):
    print(leaf)
print()

# List only tables on detector
print("Tables in group", detector, ":")
for leaf in h5file.list_nodes("/detector", 'Table'):
    print(leaf)
print()

# List only arrays on detector (there should be none!)
print("Arrays in group", detector, ":")
for leaf in h5file.list_nodes("/detector", 'Array'):
    print(leaf)
print()

# Get "/detector" Group object
group = h5file.root.detector
print("/detector ==>", group)

# Get the "/detector/table
table = h5file.get_node("/detector/table")
print("/detector/table ==>", table)

# Get metadata from table
print("Object:", table)
print("Table name:", table.name)
print("Table title:", table.title)
print("Rows saved on table: %d" % (table.nrows))

print("Variable names on table with their type:")
for name in table.colnames:
    print("  ", name, ':=', table.coldtypes[name])
print()

# Read arrays in /columns/names and /columns/pressure

# Get the object in "/columns pressure"
pressureObject = h5file.get_node("/columns", "pressure")

# Get some metadata on this object
print("Info on the object:", pressureObject)
print("  shape ==>", pressureObject.shape)
print("  title ==>", pressureObject.title)
print("  type ==> ", pressureObject.atom.dtype)
print("  byteorder ==> ", pressureObject.byteorder)

# Read the pressure actual data
pressureArray = pressureObject.read()
print("  data type ==>", type(pressureArray))
print("  data ==>", pressureArray)
print()

# Get the object in "/columns/names"
nameObject = h5file.root.columns.name

# Get some metadata on this object
print("Info on the object:", nameObject)
print("  shape ==>", nameObject.shape)
print("  title ==>", nameObject.title)
print("  type ==> " % nameObject.atom.dtype)


# Read the 'name' actual data
nameArray = nameObject.read()
print("  data type ==>", type(nameArray))
print("  data ==>", nameArray)

# Print the data for both arrays
print("Data on arrays name and pressure:")
for i in range(pressureObject.shape[0]):
    print("".join(nameArray[i]), "-->", pressureArray[i])
print()


# Finally, append some new records to table
table = h5file.root.detector.table

# Append 5 new particles to table (yes, tables can be enlarged!)
particle = table.row
for i in range(10, 15):
    # First, assign the values to the Particle record
    particle['name'] = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = [float(i * i), float(i * 2)]
    particle['temperature'] = float(i ** 2)
    particle['idnumber'] = i * (2 ** 34)  # This exceeds integer range
    # This injects the Row values.
    particle.append()

# Flush this table
table.flush()

print("Columns name and pressure on expanded table:")
# Print some table columns, for comparison with array data
for p in table:
    print(p['name'], '-->', p['pressure'])
print()

# Put several flavors
oldflavor = table.flavor
print(table.read(field="ADCcount"))
table.flavor = "numpy"
print(table.read(field="ADCcount"))
table.flavor = oldflavor
print(table.read(0, 0, 1, "name"))
table.flavor = "python"
print(table.read(0, 0, 1, "name"))
table.flavor = oldflavor
print(table.read(0, 0, 2, "pressure"))
table.flavor = "python"
print(table.read(0, 0, 2, "pressure"))
table.flavor = oldflavor

# Several range selections
print("Extended slice in selection: [0:7:6]")
print(table.read(0, 7, 6))
print("Single record in selection: [1]")
print(table.read(1))
print("Last record in selection: [-1]")
print(table.read(-1))
print("Two records before the last in selection: [-3:-1]")
print(table.read(-3, -1))

# Print a recarray in table form
table = h5file.root.detector.recarray2
print("recarray2:", table)
print("  nrows:", table.nrows)
print("  byteorder:", table.byteorder)
print("  coldtypes:", table.coldtypes)
print("  colnames:", table.colnames)

print(table.read())
for p in table.iterrows():
    print(p['f1'], '-->', p['f2'])
print()

result = [rec['f1'] for rec in table if rec.nrow < 2]
print(result)

# Test the File.rename_node() method
# h5file.rename_node(h5file.root.detector.recarray2, "recarray3")
h5file.rename_node(table, "recarray3")
# Delete a Leaf from the HDF5 tree
h5file.remove_node(h5file.root.detector.recarray3)
# Delete the detector group and its leaves recursively
# h5file.remove_node(h5file.root.detector, recursive=1)
# Create a Group and then remove it
h5file.create_group(h5file.root, "newgroup")
h5file.remove_node(h5file.root, "newgroup")
h5file.rename_node(h5file.root.columns, "newcolumns")

print(h5file)

# Close this file
h5file.close()
