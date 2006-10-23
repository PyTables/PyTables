"""Yet another couple of examples on do/undo feauture."""

import tables

def setUp(filename):
    # Create an HDF5 file
    fileh = tables.openFile(filename, mode = "w", title="Undo/Redo demo")
    # Create some nodes in there
    fileh.createGroup("/", "agroup", "Group 1")
    fileh.createGroup("/agroup", "agroup2", "Group 2")
    fileh.createArray("/", "anarray", [1,2], "Array 1")
    # Enable undo/redo.
    fileh.enableUndo()
    return fileh

def tearDown(fileh):
    # Disable undo/redo.
    fileh.disableUndo()
    # Close the file
    fileh.close()

def demo_6times3marks():
    """Checking with six ops and three marks"""

    # Initialize the data base with some nodes
    fileh = setUp("undo-redo-6times3marks.h5")

    # Create a new array
    fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
    fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
    # Put a mark
    fileh.mark()
    fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
    fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
    # Put a mark
    fileh.mark()
    fileh.createArray('/', 'otherarray5', [7,8], "Another array 5")
    fileh.createArray('/', 'otherarray6', [8,9], "Another array 6")
    # Unwind just one mark
    fileh.undo()
    assert "/otherarray1" in fileh
    assert "/otherarray2" in fileh
    assert "/otherarray3" in fileh
    assert "/otherarray4" in fileh
    assert "/otherarray5" not in fileh
    assert "/otherarray6" not in fileh
    # Unwind another mark
    fileh.undo()
    assert "/otherarray1" in fileh
    assert "/otherarray2" in fileh
    assert "/otherarray3" not in fileh
    assert "/otherarray4" not in fileh
    assert "/otherarray5" not in fileh
    assert "/otherarray6" not in fileh
    # Unwind all marks
    fileh.undo()
    assert "/otherarray1" not in fileh
    assert "/otherarray2" not in fileh
    assert "/otherarray3" not in fileh
    assert "/otherarray4" not in fileh
    assert "/otherarray5" not in fileh
    assert "/otherarray6" not in fileh
    # Redo until the next mark
    fileh.redo()
    assert "/otherarray1" in fileh
    assert "/otherarray2" in fileh
    assert "/otherarray3" not in fileh
    assert "/otherarray4" not in fileh
    assert "/otherarray5" not in fileh
    assert "/otherarray6" not in fileh
    # Redo until the next mark
    fileh.redo()
    assert "/otherarray1" in fileh
    assert "/otherarray2" in fileh
    assert "/otherarray3" in fileh
    assert "/otherarray4" in fileh
    assert "/otherarray5" not in fileh
    assert "/otherarray6" not in fileh
    # Redo until the end
    fileh.redo()
    assert "/otherarray1" in fileh
    assert "/otherarray2" in fileh
    assert "/otherarray3" in fileh
    assert "/otherarray4" in fileh
    assert "/otherarray5" in fileh
    assert "/otherarray6" in fileh

    # Tear down the file
    tearDown(fileh)

def demo_manyops():
    """Checking many operations together """

    # Initialize the data base with some nodes
    fileh = setUp("undo-redo-manyops.h5")

    # Create an array
    array2 = fileh.createArray(fileh.root, 'anarray3',
                                    [3], "Array title 3")
    # Create a group
    array2 = fileh.createGroup(fileh.root, 'agroup3',
                                    "Group title 3")
    # /anarray => /agroup/agroup3/
    newNode = fileh.copyNode('/anarray3', '/agroup/agroup2')
    newNode = fileh.copyChildren('/agroup', '/agroup3', recursive=1)
    # rename anarray
    array4 = fileh.renameNode('/anarray', 'anarray4')
    # Move anarray
    newNode = fileh.copyNode('/anarray3', '/agroup')
    # Remove anarray4
    fileh.removeNode('/anarray4')
    # Undo the actions
    fileh.undo()
    assert '/anarray4' not in fileh
    assert '/anarray3' not in fileh
    assert '/agroup/agroup2/anarray3' not in fileh
    assert '/agroup3' not in fileh
    assert '/anarray4' not in fileh
    assert '/anarray' in fileh

    # Redo the actions
    fileh.redo()
    # Check that the copied node exists again in the object tree.
    assert '/agroup/agroup2/anarray3' in fileh
    assert '/agroup/anarray3' in fileh
    assert '/agroup3/agroup2/anarray3' in fileh
    assert '/agroup3/anarray3' not in fileh
    assert fileh.root.agroup.anarray3 is newNode
    assert '/anarray' not in fileh
    assert '/anarray4' not in fileh

    # Tear down the file
    tearDown(fileh)


if __name__ == '__main__':

    # run demos
    demo_6times3marks()
    demo_manyops()
