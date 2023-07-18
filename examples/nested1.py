# Example to show how nested types can be dealed with PyTables
# F. Alted 2005/05/27

import random
import tables as tb

fileout = "nested1.h5"

# An example of enumerated structure
colors = tb.Enum(['red', 'green', 'blue'])


def read(file):
    fileh = tb.open_file(file, "r")

    print("table (short)-->", fileh.root.table)
    print("table (long)-->", repr(fileh.root.table))
    print("table (contents)-->", repr(fileh.root.table[:]))

    fileh.close()


def write(file, desc, indexed):
    fileh = tb.open_file(file, "w")
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


class Info(tb.IsDescription):
    _v_pos = 2
    Name = tb.UInt32Col()
    Value = tb.Float64Col()


class Test(tb.IsDescription):
    """A description that has several columns."""

    x = tb.Int32Col(shape=2, dflt=0, pos=0)
    y = tb.Float64Col(dflt=1.2, shape=(2, 3))
    z = tb.UInt8Col(dflt=1)
    color = tb.EnumCol(colors, 'red', base='uint32', shape=(2,))
    Info = Info()

    class info(tb.IsDescription):
        _v_pos = 1
        name = tb.StringCol(10)
        value = tb.Float64Col(pos=0)
        y2 = tb.Float64Col(dflt=1, shape=(2, 3), pos=1)
        z2 = tb.UInt8Col(dflt=1)

        class info2(tb.IsDescription):
            y3 = tb.Float64Col(dflt=1, shape=(2, 3))
            z3 = tb.UInt8Col(dflt=1)
            name = tb.StringCol(10)
            value = tb.EnumCol(colors, 'blue', base='uint32', shape=(1,))

            class info3(tb.IsDescription):
                name = tb.StringCol(10)
                value = tb.Time64Col()
                y4 = tb.Float64Col(dflt=1, shape=(2, 3))
                z4 = tb.UInt8Col(dflt=1)

# Write the file and read it
write(fileout, Test, ['info/info2/z3'])
read(fileout)
print("You can have a look at '%s' output file now." % fileout)
