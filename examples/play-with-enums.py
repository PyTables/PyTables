# Example on using enumerated types under PyTables.
# This file is intended to be run in an interactive Python session,
# since it contains some statements that raise exceptions.
# To run it, paste it as the input of ``python``.



def COMMENT(string):
    pass


COMMENT("**** Usage of the ``Enum`` class. ****")

COMMENT("Create an enumeration of colors with automatic concrete values.")
import tables as tb
colorList = ['red', 'green', 'blue', 'white', 'black']
colors = tb.Enum(colorList)

COMMENT("Take a look at the name-value pairs.")
print("Colors:", [v for v in colors])

COMMENT("Access values as attributes.")
print("Value of 'red' and 'white':", (colors.red, colors.white))
print("Value of 'yellow':", colors.yellow)

COMMENT("Access values as items.")
print("Value of 'red' and 'white':", (colors['red'], colors['white']))
print("Value of 'yellow':", colors['yellow'])

COMMENT("Access names.")
print("Name of value %s:" % colors.red, colors(colors.red))
print("Name of value 1234:", colors(1234))


COMMENT("**** Enumerated columns. ****")

COMMENT("Create a new PyTables file.")
h5f = tb.open_file('enum.h5', 'w')

COMMENT("This describes a ball extraction.")


class BallExt(tb.IsDescription):
    ballTime = tb.Time32Col()
    ballColor = tb.EnumCol(colors, 'black', base='uint8')

COMMENT("Create a table of ball extractions.")
tbl = h5f.create_table(
    '/', 'extractions', BallExt, title="Random ball extractions")

COMMENT("Simulate some ball extractions.")
import time
import random
now = time.time()
row = tbl.row
for i in range(10):
    row['ballTime'] = now + i
    row['ballColor'] = colors[random.choice(colorList)]  # notice this
    row.append()

COMMENT("Try to append an invalid value.")
row['ballTime'] = now + 42
row['ballColor'] = 1234

tbl.flush()

COMMENT("Now print them!")
for r in tbl:
    ballTime = r['ballTime']
    ballColor = colors(r['ballColor'])  # notice this
    print("Ball extracted on %d is of color %s." % (ballTime, ballColor))


COMMENT("**** Enumerated arrays. ****")

COMMENT("This describes a range of working days.")
workingDays = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5}
dayRange = tb.EnumAtom(workingDays, 'Mon', base='uint16', shape=(0, 2))

COMMENT("Create an EArray of day ranges within a week.")
earr = h5f.create_earray('/', 'days', dayRange, title="Working day ranges")
earr.flavor = 'python'

COMMENT("Throw some day ranges in.")
wdays = earr.get_enum()
earr.append([(wdays.Mon, wdays.Fri), (wdays.Wed, wdays.Fri)])

COMMENT("The append method does not check values!")
earr.append([(wdays.Mon, 1234)])

COMMENT("Print the values.")
for (d1, d2) in earr:
    print("From %s to %s (%d days)." % (wdays(d1), wdays(d2), d2 - d1 + 1))

COMMENT("Close the PyTables file and remove it.")
from pathlib import Path
h5f.close()
Path('enum.h5').unlink()
