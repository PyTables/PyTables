"""Small example of do/undo capability with PyTables."""

import tables as tb

# Create an HDF5 file
fileh = tb.open_file("tutorial3-1.h5", "w", title="Undo/Redo demo 1")

         #'-**-**-**-**-**-**- enable undo/redo log  -**-**-**-**-**-**-**-'
fileh.enable_undo()

# Create a new array
one = fileh.create_array('/', 'anarray', [3, 4], "An array")
# Mark this point
fileh.mark()
# Create a new array
another = fileh.create_array('/', 'anotherarray', [4, 5], "Another array")
# Now undo the past operation
fileh.undo()
# Check that anotherarray does not exist in the object tree but anarray does
assert "/anarray" in fileh
assert "/anotherarray" not in fileh
# Unwind once more
fileh.undo()
# Check that anarray does not exist in the object tree
assert "/anarray" not in fileh
assert "/anotherarray" not in fileh
# Go forward up to the next marker
fileh.redo()
# Check that anarray has come back to life in a sane state
assert "/anarray" in fileh
assert fileh.root.anarray.read() == [3, 4]
assert fileh.root.anarray.title == "An array"
assert fileh.root.anarray == one
# But anotherarray is not here yet
assert "/anotherarray" not in fileh
# Now, go rewind up to the end
fileh.redo()
assert "/anarray" in fileh
# Check that anotherarray has come back to life in a sane state
assert "/anotherarray" in fileh
assert fileh.root.anotherarray.read() == [4, 5]
assert fileh.root.anotherarray.title == "Another array"
assert fileh.root.anotherarray == another

         #'-**-**-**-**-**-**- disable undo/redo log  -**-**-**-**-**-**-**-'
fileh.disable_undo()

# Close the file
fileh.close()
