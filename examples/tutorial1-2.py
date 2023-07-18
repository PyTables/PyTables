"""This example shows how to browse the object tree and enlarge tables.

Before to run this program you need to execute first tutorial1-1.py
that create the tutorial1.h5 file needed here.

"""

import tables as tb

print()
print('-**-**-**-**- open the previous tutorial file -**-**-**-**-**-')

# Reopen the file in append mode
h5file = tb.open_file("tutorial1.h5", "a")

# Print the object tree created from this filename
print("Object tree from filename:", h5file.filename)
print(h5file)

print()
print('-**-**-**-**-**-**- traverse tree methods -**-**-**-**-**-**-**-')

# List all the nodes (Group and Leaf objects) on tree
print(h5file)

# List all the nodes (using File iterator) on tree
print("Nodes in file:")
for node in h5file:
    print(node)
print()

# Now, only list all the groups on tree
print("Groups in file:")
for group in h5file.walk_groups():
    print(group)
print()

# List only the arrays hanging from /
print("Arrays in file (I):")
for group in h5file.walk_groups("/"):
    for array in h5file.list_nodes(group, classname='Array'):
        print(array)

# This do the same result
print("Arrays in file (II):")
for array in h5file.walk_nodes("/", "Array"):
    print(array)
print()
# And finally, list only leafs on /detector group (there should be one!)
print("Leafs in group '/detector' (I):")
for leaf in h5file.list_nodes("/detector", 'Leaf'):
    print(leaf)

# Other way using iterators and natural naming
print("Leafs in group '/detector' (II):")
for leaf in h5file.root.detector._f_walknodes('Leaf'):
    print(leaf)


print()
print('-**-**-**-**-**-**- setting/getting object attributes -**-**--**-**-')

# Get a pointer to '/detector/readout' node
table = h5file.root.detector.readout
# Attach it a string (date) attribute
table.attrs.gath_date = "Wed, 06/12/2003 18:33"
# Attach a floating point attribute
table.attrs.temperature = 18.4
table.attrs.temp_scale = "Celsius"

# Get a pointer to '/detector' node
detector = h5file.root.detector
# Attach a general object to the parent (/detector) group
detector._v_attrs.stuff = [5, (2.3, 4.5), "Integer and tuple"]

# Now, get the attributes
print("gath_date attribute of /detector/readout:", table.attrs.gath_date)
print("temperature attribute of /detector/readout:", table.attrs.temperature)
print("temp_scale attribute of /detector/readout:", table.attrs.temp_scale)
print("stuff attribute in /detector:", detector._v_attrs.stuff)
print()

# Delete permanently the attribute gath_date of /detector/readout
print("Deleting /detector/readout gath_date attribute")
del table.attrs.gath_date

# Print a representation of all attributes in  /detector/table
print("AttributeSet instance in /detector/table:", repr(table.attrs))

# Get the (user) attributes of /detector/table
print("List of user attributes in /detector/table:", table.attrs._f_list())

# Get the (sys) attributes of /detector/table
print("List of user attributes in /detector/table:",
      table.attrs._f_list("sys"))
print()
# Rename an attribute
print("renaming 'temp_scale' attribute to 'tempScale'")
table.attrs._f_rename("temp_scale", "tempScale")
print(table.attrs._f_list())

# Try to rename a system attribute:
try:
    table.attrs._f_rename("VERSION", "version")
except:
    print("You can not rename a VERSION attribute: it is read only!.")

print()
print('-**-**-**-**-**-**- getting object metadata -**-**-**-**-**-**-')

# Get a pointer to '/detector/readout' data
table = h5file.root.detector.readout

# Get metadata from table
print("Object:", table)
print("Table name:", table.name)
print("Table title:", table.title)
print("Number of rows in table:", table.nrows)
print("Table variable names with their type and shape:")
for name in table.colnames:
    print(name, ':= {}, {}'.format(table.coldtypes[name],
                               table.coldtypes[name].shape))
print()

# Get the object in "/columns pressure"
pressureObject = h5file.get_node("/columns", "pressure")

# Get some metadata on this object
print("Info on the object:", repr(pressureObject))
print("  shape: ==>", pressureObject.shape)
print("  title: ==>", pressureObject.title)
print("  atom: ==>", pressureObject.atom)
print()
print('-**-**-**-**-**- reading actual data from arrays -**-**-**-**-**-**-')

# Read the 'pressure' actual data
pressureArray = pressureObject.read()
print(repr(pressureArray))
# Check the kind of object we have created (it should be a numpy array)
print("pressureArray is an object of type:", type(pressureArray))

# Read the 'name' Array actual data
nameArray = h5file.root.columns.name.read()
# Check the kind of object we have created (it should be a numpy array)
print("nameArray is an object of type:", type(nameArray))

print()

# Print the data for both arrays
print("Data on arrays nameArray and pressureArray:")
for i in range(pressureObject.shape[0]):
    print(nameArray[i], "-->", pressureArray[i])

print()
print('-**-**-**-**-**- reading actual data from tables -**-**-**-**-**-**-')

# Create a shortcut to table object
table = h5file.root.detector.readout

# Read the 'energy' column of '/detector/readout'
print("Column 'energy' of '/detector/readout':\n", table.cols.energy)
print()
# Read the 3rd row of '/detector/readout'
print("Third row of '/detector/readout':\n", table[2])
print()
# Read the rows from 3 to 9 of row of '/detector/readout'
print("Rows from 3 to 9 of '/detector/readout':\n", table[2:9])

print()
print('-**-**-**-**- append records to existing table -**-**-**-**-**-')

# Get the object row from table
table = h5file.root.detector.readout
particle = table.row

# Append 5 new particles to table
for i in range(10, 15):
    particle['name'] = 'Particle: %6d' % (i)
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
for r in table:
    print("%-16s | %11.1f | %11.4g | %6d | %6d | %8d |" %
          (r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'],
           r['TDCcount']))

print()
print("Total number of entries in resulting table:", table.nrows)

print()
print('-**-**-**-**- modify records of a table -**-**-**-**-**-')

# Single cells
print("First row of readout table.")
print("Before modif-->", table[0])
table.cols.TDCcount[0] = 1
print("After modifying first row of TDCcount-->", table[0])
table.cols.energy[0] = 2
print("After modifying first row of energy-->", table[0])

# Column slices
table.cols.TDCcount[2:5] = [2, 3, 4]
print("After modifying slice [2:5] of ADCcount-->", table[0:5])
table.cols.energy[1:9:3] = [2, 3, 4]
print("After modifying slice [1:9:3] of energy-->", table[0:9])

# Modifying complete Rows
table.modify_rows(start=1, step=3,
                  rows=[(1, 2, 3.0, 4, 5, 6, 'Particle:   None', 8.0),
                        (2, 4, 6.0, 8, 10, 12, 'Particle: None*2', 16.0)])
print("After modifying the complete third row-->", table[0:5])

# Modifying columns inside table iterators
for row in table.where('TDCcount <= 2'):
    row['energy'] = row['TDCcount'] * 2
    row.update()
print("After modifying energy column (where TDCcount <=2)-->", table[0:4])

print()
print('-**-**-**-**- modify elements of an array -**-**-**-**-**-')

print("pressure array")
pressureObject = h5file.root.columns.pressure
print("Before modif-->", pressureObject[:])
pressureObject[0] = 2
print("First modif-->", pressureObject[:])
pressureObject[1:3] = [2.1, 3.5]
print("Second modif-->", pressureObject[:])
pressureObject[::2] = [1, 2]
print("Third modif-->", pressureObject[:])

print("name array")
nameObject = h5file.root.columns.name
print("Before modif-->", nameObject[:])
nameObject[0] = ['Particle:   None']
print("First modif-->", nameObject[:])
nameObject[1:3] = ['Particle:      0', 'Particle:      1']
print("Second modif-->", nameObject[:])
nameObject[::2] = ['Particle:     -3', 'Particle:     -5']
print("Third modif-->", nameObject[:])

print()
print('-**-**-**-**- remove records from a table -**-**-**-**-**-')

# Delete some rows on the Table (yes, rows can be removed!)
table.remove_rows(5, 10)

# Print some table columns, for comparison with array data
print("Some columns in final table:")
print()
# Print the headers
print("%-16s | %11s | %11s | %6s | %6s | %8s |" %
     ('name', 'pressure', 'energy', 'grid_i', 'grid_j',
      'TDCcount'))

print("%-16s + %11s + %11s + %6s + %6s + %8s +" %
      ('-' * 16, '-' * 11, '-' * 11, '-' * 6, '-' * 6, '-' * 8))
# Print the data using the table iterator:
for r in table.iterrows():
    print("%-16s | %11.1f | %11.4g | %6d | %6d | %8d |" %
          (r['name'], r['pressure'], r['energy'], r['grid_i'], r['grid_j'],
           r['TDCcount']))

print()
print("Total number of entries in final table:", table.nrows)

# Close the file
h5file.close()
