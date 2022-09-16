import numpy as np
import pytest

import tables as tb


class TestOpenFileFailure:
    def test01_open_file(self, h5path):
        """Checking opening of a non existing file."""
        open_files = len(tb.file._open_files)
        with pytest.raises(IOError):
            h5f = tb.open_file(h5path)
            h5f.close()
        assert open_files == len(tb.file._open_files)

    def test02_open_file(self, h5path):
        """Checking opening of an existing non HDF5 file."""
        open_files = len(tb.file._open_files)
        # create a dummy file
        h5path.write_bytes(b"")
        # Try to open the dummy file
        with pytest.raises(tb.HDF5ExtError):
            h5f = tb.open_file(h5path)
            h5f.close()
        assert open_files == len(tb.file._open_files)

    def test03_open_file(self, h5path):
        """Checking opening of an existing file with invalid mode."""
        # See gh-318
        # create a dummy file
        h5f = tb.open_file(h5path, "w")
        h5f.close()
        # Try to open the dummy file
        with pytest.raises(ValueError):
            tb.open_file(h5path, "ab")


class OpenFileTestCase:
    h5file = None
    open_kwargs = None

    @pytest.fixture(scope="function")
    def h5f(self, h5path, request):
        h5f = tb.open_file(h5path, "w", title=request.node.name)
        root = h5f.root

        # Create an array
        h5f.create_array(root, "array", [1, 2], title="Array example")
        h5f.create_table(root, "table", {"var1": tb.IntCol()}, "Table example")
        root._v_attrs.testattr = 41

        # Create another array object
        h5f.create_array(root, "anarray", [1], "Array title")
        h5f.create_table(root, "atable", {"var1": tb.IntCol()}, "Table title")

        # Create a group object
        group = h5f.create_group(root, "agroup", "Group title")
        group._v_attrs.testattr = 42

        # Create a some objects there
        array1 = h5f.create_array(
            group, "anarray1", [1, 2, 3, 4, 5, 6, 7], "Array title 1"
        )
        array1.attrs.testattr = 42
        h5f.create_array(group, "anarray2", [2], "Array title 2")
        h5f.create_table(
            group, "atable1", {"var1": tb.IntCol()}, "Table title 1"
        )
        ra = np.rec.array([(1, 11, "a")], formats="u1,f4,a1")
        h5f.create_table(group, "atable2", ra, "Table title 2")

        # Create a lonely group in first level
        h5f.create_group(root, "agroup2", "Group title 2")

        # Create a new group in the second level
        group3 = h5f.create_group(group, "agroup3", "Group title 3")

        # Create a new group in the third level
        h5f.create_group(group3, "agroup4", "Group title 4")

        # Create an array in the root with the same name as one in 'agroup'
        h5f.create_array(root, "anarray1", [1, 2], title="Array example")
        self.h5file = h5f

        yield h5f

        self.h5file.close()

    def _reopen(self, mode="r", **kwargs):
        """Reopen ``h5file`` in the specified ``mode``."""
        self.h5file.close()
        self.h5file = tb.open_file(
            self.h5file.filename, mode, **kwargs, **self.open_kwargs
        )
        return self.h5file

    def test00_newFile(self, h5f):
        """Checking creation of a new file."""
        h5f.create_array(h5f.root, "array_new", [1, 2], title="Array example")
        # Get the CLASS attribute of the arr object
        class_ = h5f.root.array.attrs.CLASS
        assert class_.capitalize() == "Array"

    def test00_newFile_unicode_filename(self, h5path):
        with tb.open_file(h5path, "w") as h5f:
            assert h5f

    def test00_newFile_np_str_filename(self, h5path):
        with tb.open_file(np.str_(h5path), "w") as h5f:
            assert h5f

    def test00_newFile_np_unicode_filename(self, h5path):
        with tb.open_file(np.unicode_(h5path), "w") as h5f:
            assert h5f

    def test01_openFile(self, h5f):
        """Checking opening of an existing file."""

        # Open the old HDF5 file
        h5f = self._reopen()

        # Get the CLASS attribute of the arr object
        title = h5f.root.array.get_attr("TITLE")

        assert title == "Array example"

    def test02_appendFile(self, h5f):
        """Checking appending objects to an existing file."""

        # Append a new array to the existing file
        h5f = self._reopen(mode="r+")
        h5f.create_array(h5f.root, "array2", [3, 4], title="Title example 2")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Get the CLASS attribute of the arr object
        assert h5f.root.array2.get_attr("TITLE") == "Title example 2"

    def test02b_appendFile2(self, h5f):
        """Checking appending objects to an existing file ("a" version)"""

        # Append a new array to the existing file
        h5f = self._reopen(mode="a")
        h5f.create_array(h5f.root, "array2", [3, 4], title="Title example 2")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Get the CLASS attribute of the arr object
        assert h5f.root.array2.get_attr("TITLE") == "Title example 2"

    # Begin to raise errors...

    def test03_appendErrorFile(self, h5f):
        """Checking appending objects to an existing file in "w" mode."""

        # Append a new array to the existing file but in write mode
        # so, the existing file should be deleted!
        h5f = self._reopen(mode="w")
        h5f.create_array(h5f.root, "array2", [3, 4], title="Title example 2")

        # Open this file in read-only mode
        h5f = self._reopen()

        with pytest.raises(LookupError):
            # Try to get the 'array' object in the old existing file
            h5f.root.array

    def test04a_openErrorFile(self):
        """Checking opening a non-existing file for reading"""

        with pytest.raises(IOError):
            tb.open_file(
                "nonexistent.h5",
                mode="r",
                **self.open_kwargs,
            )

    def test04b_alternateRootFile(self, h5f):
        """Checking alternate root access to the object tree."""

        # Open the existent HDF5 file
        h5f = self._reopen(root_uep="/agroup")

        # Get the CLASS attribute of the arr object
        title = h5f.root.anarray1.get_attr("TITLE")

        # Get the node again, as this can trigger errors in some situations
        assert h5f.root.anarray1 is not None
        assert title == "Array title 1"

    # This test works well, but HDF5 emits a series of messages that
    # may loose the user. It is better to deactivate it.
    def notest04c_alternateRootFile(self, h5f):
        """Checking non-existent alternate root access to the object tree"""

        with pytest.raises(RuntimeError):
            self._reopen(root_uep="/nonexistent")

    def test05a_removeGroupRecursively(self, h5f):
        """Checking removing a group recursively."""

        # Delete a group with leafs
        h5f = self._reopen(mode="r+")

        with pytest.raises(tb.NodeError):
            h5f.remove_node(h5f.root.agroup)

        # This should work now
        h5f.remove_node(h5f.root, "agroup", recursive=1)

        # Open this file in read-only mode
        h5f = self._reopen()

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.root.agroup

        # Try to get a child of the removed object
        with pytest.raises(LookupError):
            h5f.get_node("/agroup/agroup3")

    def test05b_removeGroupRecursively(self, h5f):
        """Checking removing a group recursively and access to it
        immediately."""

        # Delete a group with leafs
        h5f = self._reopen(mode="r+")

        with pytest.raises(tb.NodeError):
            h5f.remove_node(h5f.root, "agroup")

        # This should work now
        h5f.remove_node(h5f.root, "agroup", recursive=1)

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.root.agroup

        # Try to get a child of the removed object
        with pytest.raises(LookupError):
            h5f.get_node("/agroup/agroup3")

    def test06_removeNodeWithDel(self, h5f):
        """Checking removing a node using ``__delattr__()``"""

        h5f = self._reopen(mode="r+")

        with pytest.raises(AttributeError):
            # This should fail because there is no *Python attribute*
            # called ``agroup``.
            del h5f.root.agroup

    def test06a_removeGroup(self, h5f):
        """Checking removing a lonely group from an existing file."""

        h5f = self._reopen(mode="r+")

        h5f.remove_node(h5f.root, "agroup2")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.root.agroup2

    def test06b_removeLeaf(self, h5f):
        """Checking removing Leaves from an existing file."""

        h5f = self._reopen(mode="r+")
        h5f.remove_node(h5f.root, "anarray")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.root.anarray

    def test06c_removeLeaf(self, h5f):
        """Checking removing Leaves and access it immediately."""

        h5f = self._reopen(mode="r+")
        h5f.remove_node(h5f.root, "anarray")

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.root.anarray

    def test06d_removeLeaf(self, h5f):
        """Checking removing a non-existent node"""

        h5f = self._reopen(mode="r+")

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.remove_node(h5f.root, "nonexistent")

    def test06e_removeTable(self, h5f):
        """Checking removing Tables from an existing file."""

        h5f = self._reopen(mode="r+")
        h5f.remove_node(h5f.root, "atable")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Try to get the removed object
        with pytest.raises(LookupError):
            h5f.root.atable

    def test07_renameLeaf(self, h5f):
        """Checking renaming a leave and access it after a close/open."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.anarray, "anarray2")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Ensure that the new name exists
        array_ = h5f.root.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/anarray2"
        assert array_._v_depth == 1

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.anarray

    def test07b_renameLeaf(self, h5f):
        """Checking renaming Leaves and accesing them immediately."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.anarray, "anarray2")

        # Ensure that the new name exists
        array_ = h5f.root.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/anarray2"
        assert array_._v_depth == 1

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.anarray

    def test07c_renameLeaf(self, h5f):
        """Checking renaming Leaves and modify attributes after that."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.anarray, "anarray2")
        array_ = h5f.root.anarray2
        array_.attrs.TITLE = "hello"

        # Ensure that the new attribute has been written correctly
        assert array_.title == "hello"
        assert array_.attrs.TITLE == "hello"

    def test07d_renameLeaf(self, h5f):
        """Checking renaming a Group under a nested group."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.agroup.anarray2, "anarray3")

        # Ensure that we can access n attributes in the new group
        node = h5f.root.agroup.anarray3
        assert node._v_title == "Array title 2"

    def test08_renameToExistingLeaf(self, h5f):
        """Checking renaming a node to an existing name."""

        h5f = self._reopen(mode="r+")

        # Try to get the previous object with the old name
        with pytest.raises(tb.NodeError):
            h5f.rename_node(h5f.root.anarray, "array")

        # Now overwrite the destination node.
        anarray = h5f.root.anarray
        h5f.rename_node(anarray, "array", overwrite=True)
        assert "/anarray" not in h5f
        assert h5f.root.array is anarray

    def test08b_renameToNotValidNaturalName(self, h5f):
        """Checking renaming a node to a non-valid natural name"""

        h5f = self._reopen(mode="r+")

        # Try to get the previous object with the old name
        with pytest.warns(tb.NaturalNameWarning):
            h5f.rename_node(h5f.root.anarray, "array 2")

    def test09_renameGroup(self, h5f):
        """Checking renaming a Group and access it after a close/open."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.agroup, "agroup3")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Ensure that the new name exists
        group = h5f.root.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/agroup3"

        # The children of this group also must be accessible through the
        # new name path
        group2 = h5f.get_node("/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/agroup3/agroup3"

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.agroup

        # Try to get a child with the old pathname
        with pytest.raises(LookupError):
            h5f.get_node("/agroup/agroup3")

    def test09b_renameGroup(self, h5f):
        """Checking renaming a Group and access it immediately."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.agroup, "agroup3")

        # Ensure that the new name exists
        group = h5f.root.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/agroup3"

        # The children of this group also must be accessible through the
        # new name path
        group2 = h5f.get_node("/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/agroup3/agroup3"

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.agroup

        # Try to get a child with the old pathname
        with pytest.raises(LookupError):
            h5f.get_node("/agroup/agroup3")

    def test09c_renameGroup(self, h5f):
        """Checking renaming a Group and modify attributes afterwards."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.agroup, "agroup3")

        # Ensure that we can modify attributes in the new group
        group = h5f.root.agroup3
        group._v_attrs.TITLE = "Hello"
        assert group._v_title == "Hello"
        assert group._v_attrs.TITLE == "Hello"

    def test09d_renameGroup(self, h5f):
        """Checking renaming a Group under a nested group."""

        h5f = self._reopen(mode="r+")
        h5f.rename_node(h5f.root.agroup.agroup3, "agroup4")

        # Ensure that we can access n attributes in the new group
        group = h5f.root.agroup.agroup4
        assert group._v_title == "Group title 3"

    def test09e_renameGroup(self, h5f):
        """Checking renaming a Group with nested groups in the LRU cache."""
        # This checks for ticket #126.

        h5f = self._reopen(mode="r+")

        # Load intermediate groups and keep a nested one alive.
        g = h5f.root.agroup.agroup3.agroup4
        assert g is not None
        h5f.rename_node("/", name="agroup", newname="agroup_")

        # see ticket #126
        assert "/agroup_/agroup4" not in h5f

        assert "/agroup" not in h5f
        for newpath in [
            "/agroup_",
            "/agroup_/agroup3",
            "/agroup_/agroup3/agroup4",
        ]:
            assert newpath in h5f
            assert newpath == h5f.get_node(newpath)._v_pathname

    def test10_moveLeaf(self, h5f):
        """Checking moving a leave and access it after a close/open."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")
        h5f.move_node(h5f.root.anarray, newgroup, "anarray2")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Ensure that the new name exists
        array_ = h5f.root.newgroup.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/newgroup/anarray2"
        assert array_._v_depth == 2

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.anarray

    def test10b_moveLeaf(self, h5f):
        """Checking moving a leave and access it without a close/open."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")
        h5f.move_node(h5f.root.anarray, newgroup, "anarray2")

        # Ensure that the new name exists
        array_ = h5f.root.newgroup.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/newgroup/anarray2"
        assert array_._v_depth == 2

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.anarray

    def test10c_moveLeaf(self, h5f):
        """Checking moving Leaves and modify attributes after that."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")
        h5f.move_node(h5f.root.anarray, newgroup, "anarray2")
        array_ = h5f.root.newgroup.anarray2
        array_.attrs.TITLE = "hello"

        # Ensure that the new attribute has been written correctly
        assert array_.title == "hello"
        assert array_.attrs.TITLE == "hello"

    def test10d_moveToExistingLeaf(self, h5f):
        """Checking moving a leaf to an existing name."""

        h5f = self._reopen(mode="r+")

        # Try to get the previous object with the old name
        with pytest.raises(tb.NodeError):
            h5f.move_node(h5f.root.anarray, h5f.root, "array")

    def test10_2_moveTable(self, h5f):
        """Checking moving a table and access it after a close/open."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")
        h5f.move_node(h5f.root.atable, newgroup, "atable2")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Ensure that the new name exists
        table_ = h5f.root.newgroup.atable2
        assert table_.name == "atable2"
        assert table_._v_pathname == "/newgroup/atable2"
        assert table_._v_depth == 2

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.atable

    def test10_2b_moveTable(self, h5f):
        """Checking moving a table and access it without a close/open."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")
        h5f.move_node(h5f.root.atable, newgroup, "atable2")

        # Ensure that the new name exists
        table_ = h5f.root.newgroup.atable2
        assert table_.name == "atable2"
        assert table_._v_pathname == "/newgroup/atable2"
        assert table_._v_depth == 2

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.atable

    def test10_2b_bis_moveTable(self, h5f):
        """Checking moving a table and use cached row without a close/open."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")

        # Cache the Row attribute prior to the move
        row = h5f.root.atable.row
        h5f.move_node(h5f.root.atable, newgroup, "atable2")

        # Ensure that the new name exists
        table_ = h5f.root.newgroup.atable2
        assert table_.name == "atable2"
        assert table_._v_pathname == "/newgroup/atable2"
        assert table_._v_depth == 2

        # Ensure that cache Row attribute has been updated
        row = table_.row
        assert table_._v_pathname == row.table._v_pathname
        nrows = table_.nrows

        # Add a new row just to make sure that this works
        row.append()
        table_.flush()
        assert table_.nrows == nrows + 1

    def test10_2c_moveTable(self, h5f):
        """Checking moving tables and modify attributes after that."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group("/", "newgroup")
        h5f.move_node(h5f.root.atable, newgroup, "atable2")
        table_ = h5f.root.newgroup.atable2
        table_.attrs.TITLE = "hello"

        # Ensure that the new attribute has been written correctly
        assert table_.title == "hello"
        assert table_.attrs.TITLE == "hello"

    def test10_2d_moveToExistingTable(self, h5f):
        """Checking moving a table to an existing name."""

        h5f = self._reopen(mode="r+")

        # Try to get the previous object with the old name
        with pytest.raises(tb.NodeError):
            h5f.move_node(h5f.root.atable, h5f.root, "table")

    def test10_2e_moveToExistingTableOverwrite(self, h5f):
        """Checking moving a table to an existing name, overwriting it."""

        h5f = self._reopen(mode="r+")

        srcNode = h5f.root.atable
        h5f.move_node(srcNode, h5f.root, "table", overwrite=True)
        dstNode = h5f.root.table

        assert srcNode is dstNode

    def test11_moveGroup(self, h5f):
        """Checking moving a Group and access it after a close/open."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group(h5f.root, "newgroup")
        h5f.move_node(h5f.root.agroup, newgroup, "agroup3")

        # Open this file in read-only mode
        h5f = self._reopen()

        # Ensure that the new name exists
        group = h5f.root.newgroup.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/newgroup/agroup3"
        assert group._v_depth == 2

        # The children of this group must also be accessible through the
        # new name path
        group2 = h5f.get_node("/newgroup/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/newgroup/agroup3/agroup3"
        assert group2._v_depth == 3

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.agroup

        # Try to get a child with the old pathname
        with pytest.raises(LookupError):
            h5f.get_node("/agroup/agroup3")

    def test11b_moveGroup(self, h5f):
        """Checking moving a Group and access it immediately."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group(h5f.root, "newgroup")
        h5f.move_node(h5f.root.agroup, newgroup, "agroup3")

        # Ensure that the new name exists
        group = h5f.root.newgroup.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/newgroup/agroup3"
        assert group._v_depth == 2

        # The children of this group must also be accessible through the
        # new name path
        group2 = h5f.get_node("/newgroup/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/newgroup/agroup3/agroup3"
        assert group2._v_depth == 3

        # Try to get the previous object with the old name
        with pytest.raises(LookupError):
            h5f.root.agroup

        # Try to get a child with the old pathname
        with pytest.raises(LookupError):
            h5f.get_node("/agroup/agroup3")

    def test11c_moveGroup(self, h5f):
        """Checking moving a Group and modify attributes afterwards."""

        h5f = self._reopen(mode="r+")
        newgroup = h5f.create_group(h5f.root, "newgroup")
        h5f.move_node(h5f.root.agroup, newgroup, "agroup3")

        # Ensure that we can modify attributes in the new group
        group = h5f.root.newgroup.agroup3
        group._v_attrs.TITLE = "Hello"
        group._v_attrs.hola = "Hello"
        assert group._v_title == "Hello"
        assert group._v_attrs.TITLE == "Hello"
        assert group._v_attrs.hola == "Hello"

    def test11d_moveToExistingGroup(self, h5f):
        """Checking moving a group to an existing name."""

        h5f = self._reopen(mode="r+")

        # Try to get the previous object with the old name
        with pytest.raises(tb.NodeError):
            h5f.move_node(h5f.root.agroup, h5f.root, "agroup2")

    def test11e_moveToExistingGroupOverwrite(self, h5f):
        """Checking moving a group to an existing name, overwriting it."""

        h5f = self._reopen(mode="r+")

        # agroup2 -> agroup
        srcNode = h5f.root.agroup2
        h5f.move_node(srcNode, h5f.root, "agroup", overwrite=True)
        dstNode = h5f.root.agroup

        assert srcNode is dstNode

    def test12a_moveNodeOverItself(self, h5f):
        """Checking moving a node over itself."""

        h5f = self._reopen(mode="r+")

        # array -> array
        srcNode = h5f.root.array
        h5f.move_node(srcNode, h5f.root, "array")
        dstNode = h5f.root.array

        assert srcNode is dstNode

    def test12b_moveGroupIntoItself(self, h5f):
        """Checking moving a group into itself."""

        h5f = self._reopen(mode="r+")
        with pytest.raises(tb.NodeError):
            # agroup2 -> agroup2/
            h5f.move_node(h5f.root.agroup2, h5f.root.agroup2)

    def test13a_copyLeaf(self, h5f):
        """Copying a leaf."""

        h5f = self._reopen(mode="r+")

        # array => agroup2/
        new_node = h5f.copy_node(h5f.root.array, h5f.root.agroup2)
        dstNode = h5f.root.agroup2.array

        assert new_node is dstNode

    def test13b_copyGroup(self, h5f):
        """Copying a group."""

        h5f = self._reopen(mode="r+")

        # agroup2 => agroup/
        new_node = h5f.copy_node(h5f.root.agroup2, h5f.root.agroup)
        dstNode = h5f.root.agroup.agroup2

        assert new_node is dstNode

    def test13c_copyGroupSelf(self, h5f):
        """Copying a group into itself."""

        h5f = self._reopen(mode="r+")

        # agroup2 => agroup2/
        new_node = h5f.copy_node(h5f.root.agroup2, h5f.root.agroup2)
        dstNode = h5f.root.agroup2.agroup2

        assert new_node is dstNode

    def test13d_copyGroupRecursive(self, h5f):
        """Recursively copying a group."""

        h5f = self._reopen(mode="r+")

        # agroup => agroup2/
        new_node = h5f.copy_node(
            h5f.root.agroup, h5f.root.agroup2, recursive=True
        )
        dstNode = h5f.root.agroup2.agroup

        assert new_node is dstNode
        assert dstNode.anarray1 is not None
        assert dstNode.anarray2 is not None
        assert dstNode.agroup3 is not None

    def test13e_copyRootRecursive(self, h5f):
        """Recursively copying the root group into the root of another file."""

        h5f = self._reopen(mode="r+")
        with tb.open_file(
            h5f.filename.with_stem("two"),
            mode="w",
            **self.open_kwargs,
        ) as h5f2:
            # h5f.root => h5f2.root
            new_node = h5f.copy_node(h5f.root, h5f2.root, recursive=True)
            dstNode = h5f2.root

            assert new_node is dstNode
            assert "/agroup" in h5f2
            assert "/agroup/anarray1" in h5f2
            assert "/agroup/agroup3" in h5f2

    def test13f_copyRootRecursive(self, h5f):
        """Recursively copying the root group into a group in another file."""

        h5f = self._reopen(mode="r+")
        with tb.open_file(
            h5f.filename.with_stem("two"), mode="w", **self.open_kwargs
        ) as h5f2:
            h5f2.create_group("/", "agroup2")

            # fileh.root => h5f2.root.agroup2
            new_node = h5f.copy_node(
                h5f.root, h5f2.root.agroup2, recursive=True
            )
            dstNode = h5f2.root.agroup2

            assert new_node is dstNode
            assert "/agroup2/agroup" in h5f2
            assert "/agroup2/agroup/anarray1" in h5f2
            assert "/agroup2/agroup/agroup3" in h5f2

    def test13g_copyRootItself(self, h5f):
        """Recursively copying the root group into itself."""

        h5f = self._reopen(mode="r+")
        assert h5f.root is not None

        # h5f.root => h5f.root
        with pytest.raises(IOError):
            h5f.copy_node(h5f.root, h5f.root, recursive=True)

    def test14a_copyNodeExisting(self, h5f):
        """Copying over an existing node."""

        h5f = self._reopen(mode="r+")

        with pytest.raises(tb.NodeError):
            # agroup2 => agroup
            h5f.copy_node(h5f.root.agroup2, newname="agroup")

    def test14b_copyNodeExistingOverwrite(self, h5f):
        """Copying over an existing node, overwriting it."""

        h5f = self._reopen(mode="r+")

        # agroup2 => agroup
        new_node = h5f.copy_node(
            h5f.root.agroup2, newname="agroup", overwrite=True
        )
        dstNode = h5f.root.agroup

        assert new_node is dstNode

    def test14b2_copyNodeExistingOverwrite(self, h5f):
        """Copying over an existing node in other file, overwriting it."""

        h5f = self._reopen(mode="r+")

        with tb.open_file(
            h5f.filename.with_stem("two"), mode="w", **self.open_kwargs
        ) as h5f2:

            # file1:/anarray1 => h5fname2:/anarray1
            new_node = h5f.copy_node(
                h5f.root.agroup.anarray1, newparent=h5f2.root
            )
            # file1:/ => h5fname2:/
            new_node = h5f.copy_node(
                h5f.root, h5f2.root, overwrite=True, recursive=True
            )
            dstNode = h5f2.root

            assert new_node is dstNode

    def test14c_copyNodeExistingSelf(self, h5f):
        """Copying over self."""

        h5f = self._reopen(mode="r+")

        with pytest.raises(tb.NodeError):
            # agroup => agroup
            h5f.copy_node(h5f.root.agroup, newname="agroup")

    def test14d_copyNodeExistingOverwriteSelf(self, h5f):
        """Copying over self, trying to overwrite."""

        h5f = self._reopen(mode="r+")

        with pytest.raises(tb.NodeError):
            # agroup => agroup
            h5f.copy_node(h5f.root.agroup, newname="agroup", overwrite=True)

    def test14e_copyGroupSelfRecursive(self, h5f):
        """Recursively copying a group into itself."""

        h5f = self._reopen(mode="r+")

        with pytest.raises(tb.NodeError):
            # agroup => agroup/
            h5f.copy_node(h5f.root.agroup, h5f.root.agroup, recursive=True)

    def test15a_oneStepMove(self, h5f):
        """Moving and renaming a node in a single action."""

        h5f = self._reopen(mode="r+")

        # anarray1 -> agroup/array
        srcNode = h5f.root.anarray1
        h5f.move_node(srcNode, h5f.root.agroup, "array")
        dstNode = h5f.root.agroup.array

        assert srcNode is dstNode

    def test15b_oneStepCopy(self, h5f):
        """Copying and renaming a node in a single action."""

        h5f = self._reopen(mode="r+")

        # anarray1 => agroup/array
        new_node = h5f.copy_node(h5f.root.anarray1, h5f.root.agroup, "array")
        dstNode = h5f.root.agroup.array

        assert new_node is dstNode

    def test16a_fullCopy(self, h5f):
        """Copying full data and user attributes."""

        h5f = self._reopen(mode="r+")

        # agroup => groupcopy
        srcNode = h5f.root.agroup
        new_node = h5f.copy_node(srcNode, newname="groupcopy", recursive=True)
        dstNode = h5f.root.groupcopy

        assert new_node is dstNode
        assert srcNode._v_attrs.testattr == dstNode._v_attrs.testattr
        assert (
            srcNode.anarray1.attrs.testattr == dstNode.anarray1.attrs.testattr
        )
        assert srcNode.anarray1.read() == dstNode.anarray1.read()

    def test16b_partialCopy(self, h5f):
        """Copying partial data and no user attributes."""

        h5f = self._reopen(mode="r+")

        # agroup => groupcopy
        srcNode = h5f.root.agroup
        new_node = h5f.copy_node(
            srcNode,
            newname="groupcopy",
            recursive=True,
            copyuserattrs=False,
            start=0,
            stop=5,
            step=2,
        )
        dstNode = h5f.root.groupcopy

        assert new_node is dstNode
        assert not hasattr(dstNode._v_attrs, "testattr")
        assert not hasattr(dstNode.anarray1.attrs, "testattr")
        assert srcNode.anarray1.read()[0:5:2] == dstNode.anarray1.read()

    def test16c_fullCopy(self, h5f):
        """Copying full data and user attributes (from file to file)."""

        h5f = self._reopen(mode="r+")

        with tb.open_file(
            h5f.filename.with_stem("two"), mode="w", **self.open_kwargs
        ) as h5f2:

            # file1:/ => h5fname2:groupcopy
            srcNode = h5f.root
            new_node = h5f.copy_node(
                srcNode, h5f2.root, newname="groupcopy", recursive=True
            )
            dstNode = h5f2.root.groupcopy

            assert new_node is dstNode
            assert srcNode._v_attrs.testattr == dstNode._v_attrs.testattr
            assert (
                srcNode.agroup.anarray1.attrs.testattr
                == dstNode.agroup.anarray1.attrs.testattr
            )
            assert (
                srcNode.agroup.anarray1.read()
                == dstNode.agroup.anarray1.read()
            )

    def test17a_CopyChunkshape(self, h5f):
        """Copying dataset with a chunkshape."""

        h5f = self._reopen(mode="r+")
        srcTable = h5f.root.table
        newTable = h5f.copy_node(srcTable, newname="tablecopy", chunkshape=11)

        assert newTable.chunkshape == (11,)
        assert srcTable.chunkshape != newTable.chunkshape

    def test17b_CopyChunkshape(self, h5f):
        """Copying dataset with a chunkshape with 'keep' value."""

        h5f = self._reopen(mode="r+")
        srcTable = h5f.root.table
        newTable = h5f.copy_node(
            srcTable, newname="tablecopy", chunkshape="keep"
        )

        assert srcTable.chunkshape == newTable.chunkshape

    def test17c_CopyChunkshape(self, h5f):
        """Copying dataset with a chunkshape with 'auto' value."""

        h5f = self._reopen(mode="r+")
        srcTable = h5f.root.table
        newTable = h5f.copy_node(srcTable, newname="tablecopy", chunkshape=11)
        newTable2 = h5f.copy_node(
            newTable, newname="tablecopy2", chunkshape="auto"
        )

        assert srcTable.chunkshape == newTable2.chunkshape

    def test18_closedRepr(self, h5f):
        """Representing a closed node as a string."""

        h5f = self._reopen()

        for node in [h5f.root.agroup, h5f.root.anarray]:
            node._f_close()
            assert "closed" in str(node)
            assert "closed" in repr(node)

    def test19_fileno(self, h5f):
        """Checking that the 'fileno()' method works."""

        # Open the old HDF5 file
        h5f = self._reopen(mode="r")

        # Get the file descriptor for this file
        fd = h5f.fileno()
        assert fd >= 0


class TestNodeCacheOpenFile(OpenFileTestCase):
    open_kwargs = {"node_cache_slots": tb.parameters.NODE_CACHE_SLOTS}


class TestNoNodeCacheOpenFile(OpenFileTestCase):
    open_kwargs = {"node_cache_slots": 0}


class TestDictNodeCacheOpenFile(OpenFileTestCase):
    open_kwargs = {"node_cache_slots": -tb.parameters.NODE_CACHE_SLOTS}
