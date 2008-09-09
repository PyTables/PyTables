"""Small example showing the use of nested types in PyTables.

The program creates an output file, 'nested-tut.h5'.  You can view it
with ptdump or any HDF5 generic utility.

:Author: F. Alted
:Date: 2005/06/10
"""

import numpy

from tables import *

        #'-**-**-**-**- The sample nested class description  -**-**-**-**-**-'

class Info(IsDescription):
    """A sub-structure of Test"""
    _v_pos = 2   # The position in the whole structure
    name = StringCol(10)
    value = Float64Col(pos=0)

colors = Enum(['red', 'green', 'blue'])

class NestedDescr(IsDescription):
    """A description that has several nested columns"""
    color = EnumCol(colors, 'red', base='uint32')
    info1 = Info()
    class info2(IsDescription):
        _v_pos = 1
        name = StringCol(10)
        value = Float64Col(pos=0)
        class info3(IsDescription):
            x = Float64Col(dflt=1)
            y = UInt8Col(dflt=1)

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
print repr(table.nrows)

nra = table[::4]
print repr(nra)
# Append some additional rows
table.append(nra)
print repr(table.nrows)

# Create a new table
table2 = fileh.createTable(fileh.root, 'table2', nra)
print repr(table2[:])

# Read also the info2/name values with color == colors.red
names = [ x['info2/name'] for x in table if x['color'] == colors.red ]

print
print "**** info2/name elements satisfying color == 'red':", repr(names)

print
print   '-**-**-**-**-**-**- table data reading & selection  -**-**-**-**-**-'

# Read the data
print
print "**** table data contents:\n", table[:]

print
print "**** table.info2 data contents:\n", repr(table.cols.info2[1:5])

print
print "**** table.info2.info3 data contents:\n", repr(table.cols.info2.info3[1:5])

print "**** _f_col() ****"
print repr(table.cols._f_col('info2'))
print repr(table.cols._f_col('info2/info3/y'))

print
print   '-**-**-**-**-**-**- table metadata  -**-**-**-**-**-'

# Read description metadata
print
print "**** table description (short):\n", repr(table.description)
print
print "**** more from manual, period ***"
print repr(table.description.info1)
print repr(table.description.info2.info3)
print repr(table.description._v_nestedNames)
print repr(table.description.info1._v_nestedNames)
print
print "**** now some for nested records, take that ****"
print repr(table.description._v_nestedDescr)
print repr(numpy.rec.array(None, shape=0,
                           dtype=table.description._v_nestedDescr))
print repr(numpy.rec.array(None, shape=0,
                           dtype=table.description.info2._v_nestedDescr))
# NumPy recarrays doesn't have the machinery to understand the idiom below,
# please use the above form instead.
###print repr(numpy.rec.array(None, shape=1,
###           names=table.description._v_nestedNames,
###           formats=table.description._v_nestedFormats))
from tables import nra
print repr(nra.array(None, descr=table.description._v_nestedDescr))
print repr(nra.array(None, names=table.description._v_nestedNames,
                     formats=table.description._v_nestedFormats))
print
print "**** and some iteration over descriptions, too ****"
for coldescr in table.description._f_walk():
    print "column-->",coldescr
print
print "**** info2 sub-structure description:\n", table.description.info2
print
print "**** table representation (long form):\n", `table`

# Remember to always close the file
fileh.close()
