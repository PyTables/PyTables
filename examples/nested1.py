# Example to show how nested types can be dealed with PyTables
# F. Alted 2005/05/27

import random
from tables import *

fileout = "nested1.h5"

# An example of enumerated structure
colors = Enum(['red', 'green', 'blue'])

def read(file):
    fileh = openFile(file, "r")

    print "table (short)-->", fileh.root.table
    print "table (long)-->", `fileh.root.table`
    print "table (contents)-->", `fileh.root.table[:]`

    fileh.close()

def write(file, desc, indexed):
    fileh = openFile(file, "w")
    table = fileh.createTable(fileh.root, 'table', desc)
    for colname in indexed:
        table.colinstances[colname].createIndex()

    row = table.row
    for i in range(10):
        row['x'] = i
        row['y'] = 10.2-i
        row['z'] = i
        row['color'] = colors[random.choice(['red', 'green', 'blue'])]
        row['info/name'] = "name%s" % i
        row['info/info2/info3/z4'] =  i
        # All the rest will be filled with defaults
        row.append()

    fileh.close()

# The sample nested class description

class Info(IsDescription):
    _v_pos = 2
    Name = UInt32Col()
    Value = Float64Col()

class Test(IsDescription):
    """A description that has several columns"""
    x = Int32Col(shape=2, dflt=0, pos=0)
    y = Float64Col(dflt=1.2, shape=(2,3))
    z = UInt8Col(dflt=1)
    color = EnumCol(colors, 'red', base='uint32', shape=(2,))
    Info = Info()
    class info(IsDescription):
        _v_pos = 1
        name = StringCol(10)
        value = Float64Col(pos=0)
        y2 = Float64Col(dflt=1, shape=(2,3), pos=1)
        z2 = UInt8Col(dflt=1)
        class info2(IsDescription):
            y3 = Float64Col(dflt=1, shape=(2,3))
            z3 = UInt8Col(dflt=1)
            name = StringCol(10)
            value = EnumCol(colors, 'blue', base='uint32', shape=(1,))
            class info3(IsDescription):
                name = StringCol(10)
                value = Time64Col()
                y4 = Float64Col(dflt=1, shape=(2,3))
                z4 = UInt8Col(dflt=1)

# Write the file and read it
write(fileout, Test, ['info/info2/z3'])
read(fileout)
print "You can have a look at '%s' output file now." % fileout
