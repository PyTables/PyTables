# Example on using enumerated types under PyTables.
# This file is intended to be run in an interactive Python session,
# since it contains some statements that raise exceptions.
# To run it, paste it as the input of ``python``.

import time
import random
from pathlib import Path

import tables as tb


def COMMENT(string):  # noqa: N802
    pass


COMMENT("**** Usage of the ``Enum`` class. ****")

COMMENT("Create an enumeration of colors with automatic concrete values.")

color_list = ["red", "green", "blue", "white", "black"]
colors = tb.Enum(color_list)

COMMENT("Take a look at the name-value pairs.")
print("Colors:", colors)

COMMENT("Access values as attributes.")
print("Value of 'red' and 'white':", (colors.red, colors.white))
print("Value of 'yellow':", colors.yellow)

COMMENT("Access values as items.")
print("Value of 'red' and 'white':", (colors["red"], colors["white"]))
print("Value of 'yellow':", colors["yellow"])

COMMENT("Access names.")
print("Name of value %s:" % colors.red, colors(colors.red))
print("Name of value 1234:", colors(1234))


COMMENT("**** Enumerated columns. ****")

COMMENT("Create a new PyTables file.")
h5f = tb.open_file("enum.h5", "w")

COMMENT("This describes a ball extraction.")


class BallExt(tb.IsDescription):
    ball_time = tb.Time32Col()
    ball_color = tb.EnumCol(colors, "black", base="uint8")


COMMENT("Create a table of ball extractions.")
tbl = h5f.create_table(
    "/", "extractions", BallExt, title="Random ball extractions"
)

COMMENT("Simulate some ball extractions.")

now = time.time()
row = tbl.row
for i in range(10):
    row["ballTime"] = now + i
    row["ballColor"] = colors[random.choice(color_list)]  # notice this
    row.append()

COMMENT("Try to append an invalid value.")
row["ballTime"] = now + 42
row["ballColor"] = 1234

tbl.flush()

COMMENT("Now print them!")
for r in tbl:
    ball_time = r["ballTime"]
    ball_color = colors(r["ballColor"])  # notice this
    print("Ball extracted on %d is of color %s." % (ball_time, ball_color))


COMMENT("**** Enumerated arrays. ****")

COMMENT("This describes a range of working days.")
working_days = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5}
day_range = tb.EnumAtom(working_days, "Mon", base="uint16", shape=(0, 2))

COMMENT("Create an EArray of day ranges within a week.")
earr = h5f.create_earray("/", "days", day_range, title="Working day ranges")
earr.flavor = "python"

COMMENT("Throw some day ranges in.")
wdays = earr.get_enum()
earr.append([(wdays.Mon, wdays.Fri), (wdays.Wed, wdays.Fri)])

COMMENT("The append method does not check values!")
earr.append([(wdays.Mon, 1234)])

COMMENT("Print the values.")
for d1, d2 in earr:
    print("From %s to %s (%d days)." % (wdays(d1), wdays(d2), d2 - d1 + 1))

COMMENT("Close the PyTables file and remove it.")

h5f.close()
Path("enum.h5").unlink()
