"""Yet another couple of examples on do/undo feauture."""

import tables as tb


def setUp(filename):
    # Create an HDF5 file
    fileh = tb.open_file(filename, mode="w", title="Undo/Redo demo")
    # Create some nodes in there
    fileh.create_group("/", "agroup", "Group 1")
    fileh.create_group("/agroup", "agroup2", "Group 2")
    fileh.create_array("/", "anarray", [1, 2], "Array 1")
    # Enable undo/redo.
    fileh.enable_undo()
    return fileh


def tearDown(fileh):
    # Disable undo/redo.
    fileh.disable_undo()
    # Close the file
    fileh.close()


def demo_6times3marks():
    """Checking with six ops and three marks."""

    # Initialize the data base with some nodes
    fileh = setUp("undo-redo-6times3marks.h5")

    # Create a new array
    fileh.create_array('/', 'otherarray1', [3, 4], "Another array 1")
    fileh.create_array('/', 'otherarray2', [4, 5], "Another array 2")
    # Put a mark
    fileh.mark()
    fileh.create_array('/', 'otherarray3', [5, 6], "Another array 3")
    fileh.create_array('/', 'otherarray4', [6, 7], "Another array 4")
    # Put a mark
    fileh.mark()
    fileh.create_array('/', 'otherarray5', [7, 8], "Another array 5")
    fileh.create_array('/', 'otherarray6', [8, 9], "Another array 6")
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
    """Checking many operations together."""

    # Initialize the data base with some nodes
    fileh = setUp("undo-redo-manyops.h5")

    # Create an array
    fileh.create_array(fileh.root, 'anarray3', [3], "Array title 3")
    # Create a group
    fileh.create_group(fileh.root, 'agroup3', "Group title 3")
    # /anarray => /agroup/agroup3/
    new_node = fileh.copy_node('/anarray3', '/agroup/agroup2')
    new_node = fileh.copy_children('/agroup', '/agroup3', recursive=1)
    # rename anarray
    fileh.rename_node('/anarray', 'anarray4')
    # Move anarray
    new_node = fileh.copy_node('/anarray3', '/agroup')
    # Remove anarray4
    fileh.remove_node('/anarray4')
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
    assert fileh.root.agroup.anarray3 is new_node
    assert '/anarray' not in fileh
    assert '/anarray4' not in fileh

    # Tear down the file
    tearDown(fileh)


if __name__ == '__main__':

    # run demos
    demo_6times3marks()
    demo_manyops()
