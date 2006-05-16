"""Small example showing the use of nested types in PyTables.

The program creates an output file, 'nested-tut.h5'.  You can view it
with ptdump or any HDF5 generic utility.

:Author: F. Altet
:Date: 2005/06/10
"""

from tables import *

# An example of an enumerated structure
colors = Enum(['red', 'green', 'blue'])

        #'-**-**-**-**- The sample nested class description  -**-**-**-**-**-'

class Info(IsDescription):
    """A sub-structure of Test"""
    _v_pos = 2   # The position in the whole structure
    name = StringCol(10)
    value = Float64Col(pos=0)

class NestedDescr(IsDescription):
    """A description that has several nested columns"""
    color = EnumCol(colors, 'red', dtype='UInt32', indexed=1) # indexed column
    info1 = Info()
    class info2(IsDescription):
        _v_pos = 1
        name = StringCol(10)
        value = Float64Col(pos=0)
        class info3(IsDescription):
            x = FloatCol(1)
            y = UInt8Col(1)

print
print   '-**-**-**-**-**-**- file creation  -**-**-**-**-**-**-**-'

filename = "nested-tut.h5"

print "Creating file:", filename
fileh = openFile(filename, "w")

print
print   '-**-**-**-**-**- nested table creation  -**-**-**-**-**-'

table = fileh.createTable(fileh.root, 'table', NestedDescr)

# Fill the table with some rows
row = table.row
for i in range(10):
    row['color'] = colors[['red', 'green', 'blue'][i%3]]
    row['info1/name'] = "name1-%s" % i
    row['info2/name'] = "name2-%s" % i
    row['info2/info3/y'] =  i
    # All the rest will be filled with defaults
    row.append()

table.flush()  # flush the row buffer to disk

nra = table[::4]
# Append some additional rows
table.append(nra)

# Create a new table
table2 = fileh.createTable(fileh.root, 'table2', nra)

print
print   '-**-**-**-**-**-**- table data reading & selection  -**-**-**-**-**-'

# Read the data
print
print "**** table data contents:\n", table[:]

print
print "**** table.info2 data contents:\n", table.cols.info2[1:5]

print
print "**** table.info2.info3 data contents:\n", table.cols.info2.info3[1:5]

# Read also the info2/name values with color == colors.red
names = [ x['info2/name'] for x in table if x['color'] == colors.red ]

print
print "**** info2/name elements satisfying color == 'red':", names

print
print   '-**-**-**-**-**-**- table metadata  -**-**-**-**-**-'

# Read description metadata
print
print "**** table description (short):\n", table.description
print
print "**** info2 sub-structure description:\n", table.description.info2
print
print "**** table representation (long form):\n", `table`

# Remember to always close the file
fileh.close()
