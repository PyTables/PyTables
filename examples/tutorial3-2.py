"""A more complex example of do/undo capability with PyTables.

Here, names has been assigned to the marks, and jumps are done between
marks.

"""

import tables as tb

# Create an HDF5 file
fileh = tb.open_file('tutorial3-2.h5', 'w', title='Undo/Redo demo 2')

         #'-**-**-**-**-**-**- enable undo/redo log  -**-**-**-**-**-**-**-'
fileh.enable_undo()

# Start undoable operations
fileh.create_array('/', 'otherarray1', [3, 4], 'Another array 1')
fileh.create_group('/', 'agroup', 'Group 1')
# Create a 'first' mark
fileh.mark('first')
fileh.create_array('/agroup', 'otherarray2', [4, 5], 'Another array 2')
fileh.create_group('/agroup', 'agroup2', 'Group 2')
# Create a 'second' mark
fileh.mark('second')
fileh.create_array('/agroup/agroup2', 'otherarray3', [5, 6], 'Another array 3')
# Create a 'third' mark
fileh.mark('third')
fileh.create_array('/', 'otherarray4', [6, 7], 'Another array 4')
fileh.create_array('/agroup', 'otherarray5', [7, 8], 'Another array 5')

# Now go to mark 'first'
fileh.goto('first')
assert '/otherarray1' in fileh
assert '/agroup' in fileh
assert '/agroup/agroup2' not in fileh
assert '/agroup/otherarray2' not in fileh
assert '/agroup/agroup2/otherarray3' not in fileh
assert '/otherarray4' not in fileh
assert '/agroup/otherarray5' not in fileh
# Go to mark 'third'
fileh.goto('third')
assert '/otherarray1' in fileh
assert '/agroup' in fileh
assert '/agroup/agroup2' in fileh
assert '/agroup/otherarray2' in fileh
assert '/agroup/agroup2/otherarray3' in fileh
assert '/otherarray4' not in fileh
assert '/agroup/otherarray5' not in fileh
# Now go to mark 'second'
fileh.goto('second')
assert '/otherarray1' in fileh
assert '/agroup' in fileh
assert '/agroup/agroup2' in fileh
assert '/agroup/otherarray2' in fileh
assert '/agroup/agroup2/otherarray3' not in fileh
assert '/otherarray4' not in fileh
assert '/agroup/otherarray5' not in fileh
# Go to the end
fileh.goto(-1)
assert '/otherarray1' in fileh
assert '/agroup' in fileh
assert '/agroup/agroup2' in fileh
assert '/agroup/otherarray2' in fileh
assert '/agroup/agroup2/otherarray3' in fileh
assert '/otherarray4' in fileh
assert '/agroup/otherarray5' in fileh
# Check that objects have come back to life in a sane state
assert fileh.root.otherarray1.read() == [3, 4]
assert fileh.root.agroup.otherarray2.read() == [4, 5]
assert fileh.root.agroup.agroup2.otherarray3.read() == [5, 6]
assert fileh.root.otherarray4.read() == [6, 7]
assert fileh.root.agroup.otherarray5.read() == [7, 8]


         #'-**-**-**-**-**-**- disable undo/redo log  -**-**-**-**-**-**-**-'
fileh.disable_undo()

# Close the file
fileh.close()
