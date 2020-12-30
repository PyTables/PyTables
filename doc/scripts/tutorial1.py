"""Small but quite comprehensive example showing the use of PyTables.

The program creates an output file, 'tutorial1.h5'.  You can view it
with any HDF5 generic utility.

"""

import os
import sys
import traceback

SECTION = "I HAVE NO TITLE"


def tutsep():
    print('----8<----', SECTION, '----8<----')


def tutprint(obj):
    tutsep()
    print(obj)


def tutrepr(obj):
    tutsep()
    print(repr(obj))


def tutexc():
    tutsep()
    traceback.print_exc(file=sys.stdout)


SECTION = "Importing tables objects"
import numpy as np
import tables as tb

SECTION = "Declaring a Column Descriptor"


# Define a user record to characterize some kind of particles
class Particle(tb.IsDescription):
    name = tb.StringCol(16)  # 16-character String
    idnumber = tb.Int64Col()  # Signed 64-bit integer
    ADCcount = tb.UInt16Col()  # Unsigned short integer
    TDCcount = tb.UInt8Col()  # unsigned byte
    grid_i = tb.Int32Col()  # integer
    grid_j = tb.IntCol()  # integer (equivalent to Int32Col)
    pressure = tb.Float32Col()  # float  (single-precision)
    energy = tb.FloatCol()  # double (double-precision)


SECTION = "Creating a PyTables file from scratch"
# Open a file in "w"rite mode
h5file = tb.open_file('tutorial1.h5', mode="w", title="Test file")

SECTION = "Creating a new group"
# Create a new group under "/" (root)
group = h5file.create_group("/", 'detector', 'Detector information')

SECTION = "Creating a new table"
# Create one table on it
table = h5file.create_table(group, 'readout', Particle, "Readout example")

tutprint(h5file)
tutrepr(h5file)

# Get a shortcut to the record object in table
particle = table.row

# Fill the table with 10 particles
for i in range(10):
    particle['name'] = f'Particle: {i:6d}'
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i * i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)
    # Insert a new particle record
    particle.append()

# Flush the buffers for table
table.flush()

SECTION = "Reading (and selecting) data in a table"
# Read actual data from table. We are interested in collecting pressure values
# on entries where TDCcount field is greater than 3 and pressure less than 50
table = h5file.root.detector.readout
pressure = [
    x['pressure']
    for x in table
    if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50
]

tutrepr(pressure)

# Read also the names with the same cuts
names = [
    x['name']
    for x in table
    if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50
]

tutrepr(names)

SECTION = "Creating new array objects"
gcolumns = h5file.create_group(h5file.root, "columns", "Pressure and Name")

tutrepr(
    h5file.create_array(gcolumns, 'pressure', np.array(pressure),
                        "Pressure column selection")
)

tutrepr(
    h5file.create_array('/columns', 'name', names, "Name column selection")
)

tutprint(h5file)

SECTION = "Closing the file and looking at its content"
# Close the file
h5file.close()

tutsep()
os.system('h5ls -rd tutorial1.h5')
tutsep()
os.system('ptdump tutorial1.h5')

"""This example shows how to browse the object tree and enlarge tables.

Before to run this program you need to execute first tutorial1-1.py
that create the tutorial1.h5 file needed here.

"""

SECTION = "Traversing the object tree"
# Reopen the file in append mode
h5file = tb.open_file("tutorial1.h5", "a")

# Print the object tree created from this filename
# List all the nodes (Group and Leaf objects) on tree
tutprint(h5file)

# List all the nodes (using File iterator) on tree
tutsep()
for node in h5file:
    print(node)

# Now, only list all the groups on tree
tutsep()
for group in h5file.walk_groups("/"):
    print(group)

# List only the arrays hanging from /
tutsep()
for group in h5file.walk_groups("/"):
    for array in h5file.list_nodes(group, classname='Array'):
        print(array)

# This gives the same result
tutsep()
for array in h5file.walk_nodes("/", "Array"):
    print(array)

# And finally, list only leafs on /detector group (there should be one!)
# Other way using iterators and natural naming
tutsep()
for leaf in h5file.root.detector('Leaf'):
    print(leaf)

SECTION = "Setting and getting user attributes"
# Get a pointer to '/detector/readout'
table = h5file.root.detector.readout

# Attach it a string (date) attribute
table.attrs.gath_date = "Wed, 06/12/2003 18:33"

# Attach a floating point attribute
table.attrs.temperature = 18.4
table.attrs.temp_scale = "Celsius"

# Get a pointer to '/detector'
detector = h5file.root.detector
# Attach a general object to the parent (/detector) group
detector._v_attrs.stuff = [5, (2.3, 4.5), "Integer and tuple"]

# Now, get the attributes
tutrepr(table.attrs.gath_date)
tutrepr(table.attrs.temperature)
tutrepr(table.attrs.temp_scale)
tutrepr(detector._v_attrs.stuff)

# Delete permanently the attribute gath_date of /detector/readout
del table.attrs.gath_date

# Print a representation of all attributes in  /detector/table
tutrepr(table.attrs)

# Get the (user) attributes of /detector/table
tutprint(table.attrs._f_list("user"))

# Get the (sys) attributes of /detector/table
tutprint(table.attrs._f_list("sys"))

# Rename an attribute
table.attrs._f_rename("temp_scale", "tempScale")
tutprint(table.attrs._f_list())

# Try to rename a system attribute:
try:
    table.attrs._f_rename("VERSION", "version")
except:
    tutexc()

h5file.flush()
tutsep()
os.system('h5ls -vr tutorial1.h5/detector/readout')

SECTION = "Getting object metadata"
# Get metadata from table
tutsep()
print("Object:", table)
tutsep()
print("Table name:", table.name)
tutsep()
print("Table title:", table.title)
tutsep()
print("Number of rows in table:", table.nrows)
tutsep()
print("Table variable names with their type and shape:")
tutsep()
for name in table.colnames:
    print(f'{name}:= {table.coltypes[name]}, {table.colshapes[name]}')

tutprint(table.__doc__)

# Get the object in "/columns pressure"
pressureObject = h5file.get_node("/columns", "pressure")

# Get some metadata on this object
tutsep()
print(f"Info on the object: {pressureObject!r}")
tutsep()
print(f" shape: ==> {pressureObject.shape}")
tutsep()
print(f" title: ==> {pressureObject.title}")
tutsep()
print(f" type: ==> {pressureObject.type}")

SECTION = "Reading data from Array objects"
# Read the 'pressure' actual data
pressureArray = pressureObject.read()
tutrepr(pressureArray)
tutsep()
print(f"pressureArray is an object of type: {type(pressureArray)}")

# Read the 'name' Array actual data
nameArray = h5file.root.columns.name.read()
tutrepr(nameArray)
print(f"nameArray is an object of type: {type(nameArray)}")

# Print the data for both arrays
tutprint("Data on arrays nameArray and pressureArray:")
tutsep()
for i in range(pressureObject.shape[0]):
    print(f"{nameArray[i]} --> {pressureArray[i]}")
tutrepr(pressureObject.name)

SECTION = "Appending data to an existing table"
# Create a shortcut to table object
table = h5file.root.detector.readout
# Get the object row from table
particle = table.row

# Append 5 new particles to table
for i in range(10, 15):
    particle['name'] = f'Particle: {i:6d}'
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i * i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)  # This exceeds long integer range
    particle.append()

# Flush this table
table.flush()

# Print the data using the table iterator:
tutsep()
for r in table:
    print(f"{r['name']:<16s} | {r['pressure']:11.1f} | {r['energy']:11.4g} | "
          f"{r['grid_i']:6d} | {r['grid_j']:6d} | {r['TDCcount']:8d} |")

# Delete some rows on the Table (yes, rows can be removed!)
tutrepr(table.remove_rows(5, 10))

# Close the file
h5file.close()
