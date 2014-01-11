# Example to show how nested types can be dealed with PyTables
# F. Alted 2005/05/27

from __future__ import print_function
import random
import tables

fileout = "nested1.h5"

# An example of enumerated structure
colors = tables.Enum(['red', 'green', 'blue'])


def read(file):
    fileh = tables.open_file(file, "r")

    print("table (short)-->", fileh.root.table)
    print("table (long)-->", repr(fileh.root.table))
    print("table (contents)-->", repr(fileh.root.table[:]))

    fileh.close()


def write(file, desc, indexed):
    fileh = tables.open_file(file, "w")
    table = fileh.create_table(fileh.root, 'table', desc)
    for colname in indexed:
        table.colinstances[colname].create_index()

    row = table.row
    for i in range(10):
        row['x'] = i
        row['y'] = 10.2 - i
        row['z'] = i
        row['color'] = colors[random.choice(['red', 'green', 'blue'])]
        row['info/name'] = "name%s" % i
        row['info/info2/info3/z4'] = i
        # All the rest will be filled with defaults
        row.append()

    fileh.close()

# The sample nested class description


class Info(tables.IsDescription):
    _v_pos = 2
    Name = tables.UInt32Col()
    Value = tables.Float64Col()


class Test(tables.IsDescription):
    """A description that has several columns."""

    x = tables.Int32Col(shape=2, dflt=0, pos=0)
    y = tables.Float64Col(dflt=1.2, shape=(2, 3))
    z = tables.UInt8Col(dflt=1)
    color = tables.EnumCol(colors, 'red', base='uint32', shape=(2,))
    Info = Info()

    class info(tables.IsDescription):
        _v_pos = 1
        name = tables.StringCol(10)
        value = tables.Float64Col(pos=0)
        y2 = tables.Float64Col(dflt=1, shape=(2, 3), pos=1)
        z2 = tables.UInt8Col(dflt=1)

        class info2(tables.IsDescription):
            y3 = tables.Float64Col(dflt=1, shape=(2, 3))
            z3 = tables.UInt8Col(dflt=1)
            name = tables.StringCol(10)
            value = tables.EnumCol(colors, 'blue', base='uint32', shape=(1,))

            class info3(tables.IsDescription):
                name = tables.StringCol(10)
                value = tables.Time64Col()
                y4 = tables.Float64Col(dflt=1, shape=(2, 3))
                z4 = tables.UInt8Col(dflt=1)

# Write the file and read it
write(fileout, Test, ['info/info2/z3'])
read(fileout)
print("You can have a look at '%s' output file now." % fileout)
