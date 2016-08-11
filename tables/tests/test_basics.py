import os
import shutil
import tempfile
import unittest

import numpy as np
import tables as tb

from .common import TempFileMixin


def test_file_has_a_title(pytables_file):
    assert pytables_file.title == 'test file'


def test_array_has_a_class(array):
    assert array.attrs.CLASS == 'ARRAY'


class OpenFileTestCase(TempFileMixin):
    def test01_openFile(self):
        """Checking opening of an existing file."""

        # Open the old HDF5 file
        self._reopen()

        # Get the CLASS attribute of the arr object
        title = self.h5file.root.array.attrs["TITLE"]

        self.assertEqual(title, "Array example")

    def test02_appendFile(self):
        """Checking appending objects to an existing file."""

        # Append a new array to the existing file
        self._reopen(mode="r+", )
        self.h5file.create_array(self.h5file.root, 'array2', [3, 4],
                                 title="Title example 2")

        # Open this file in read-only mode
        self._reopen()

        # Get the CLASS attribute of the arr object
        title = self.h5file.root.array2.attrs["TITLE"]

        self.assertEqual(title, "Title example 2")

    def test02b_appendFile2(self):
        """Checking appending objects to an existing file ("a" version)"""

        # Append a new array to the existing file
        self._reopen(mode="a", )
        self.h5file.create_array(self.h5file.root, 'array2', [3, 4],
                                 title="Title example 2")

        # Open this file in read-only mode
        self._reopen()

        # Get the CLASS attribute of the arr object
        title = self.h5file.root.array2.attrs['TITLE']

        self.assertEqual(title, "Title example 2")

    # Begin to raise errors...

    def test03_appendErrorFile(self):
        """Checking appending objects to an existing file in "w" mode."""

        # Append a new array to the existing file but in write mode
        # so, the existing file should be deleted!
        self._reopen(mode="w")
        self.h5file.create_array(self.h5file.root, 'array2', [3, 4],
                                 title="Title example 2")

        # Open this file in read-only mode
        self._reopen()

        with self.assertRaises(LookupError):
            # Try to get the 'array' object in the old existing file
            self.h5file.root.array

    def test04a_openErrorFile(self):
        """Checking opening a non-existing file for reading"""

        with self.assertRaises(IOError):
            tb.open_file("nonexistent.h5", mode="r")

    def test05a_removeGroup(self):
        """Checking removing a group recursively."""

        # Delete a group with leafs
        self._reopen(mode='r+')

        self.h5file.remove_node(self.h5file.root.agroup)

        # Open this file in read-only mode
        self._reopen()

        # Try to get the removed object
        with self.assertRaises(LookupError):
            self.h5file.root.agroup

        # Try to get a child of the removed object
        with self.assertRaises(LookupError):
            self.h5file["/agroup/agroup3"]

    def test06_removeNodeWithDel(self):
        """Checking removing a node using ``__delattr__()``"""

        self._reopen(mode="r+", )

        with self.assertRaises(AttributeError):
            # This should fail because there is no *Python attribute*
            # called ``agroup``.
            del self.h5file.root.agroup

    def test06a_removeGroup(self):
        """Checking removing a lonely group from an existing file."""

        self._reopen(mode="r+")

        self.h5file.remove_node(self.h5file.root, 'agroup2')

        # Open this file in read-only mode
        self._reopen()

        # Try to get the removed object
        with self.assertRaises(LookupError):
            self.h5file.root.agroup2

    def test06b_removeLeaf(self):
        """Checking removing Leaves from an existing file."""

        self._reopen(mode="r+")
        self.h5file.remove_node(self.h5file.root, 'anarray')

        # Open this file in read-only mode
        self._reopen()

        # Try to get the removed object
        with self.assertRaises(LookupError):
            self.h5file.root.anarray

    def test06c_removeLeaf(self):
        """Checking removing Leaves and access it immediately."""

        self._reopen(mode="r+")
        self.h5file.remove_node(self.h5file.root, 'anarray')

        # Try to get the removed object
        with self.assertRaises(LookupError):
            self.h5file.root.anarray

    def test06d_removeLeaf(self):
        """Checking removing a non-existent node"""

        self._reopen(mode="r+")

        # Try to get the removed object
        with self.assertRaises(LookupError):
            self.h5file.remove_node(self.h5file.root, 'nonexistent')

    def test06e_removeTable(self):
        """Checking removing Tables from an existing file."""

        self._reopen(mode="r+")
        self.h5file.remove_node(self.h5file.root, 'atable')

        # Open this file in read-only mode
        self._reopen()

        # Try to get the removed object
        with self.assertRaises(LookupError):
            self.h5file.root.atable

    def test07_renameLeaf(self):
        """Checking renaming a leave and access it after a close/open."""

        self._reopen(mode="r+")
        self.h5file.rename_node(self.h5file.root.anarray, 'anarray2')

        # Open this file in read-only mode
        self._reopen()

        # Ensure that the new name exists
        array_ = self.h5file.root.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/anarray2")
        self.assertEqual(array_._v_depth, 1)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.anarray

    def test07b_renameLeaf(self):
        """Checking renaming Leaves and accesing them immediately."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.anarray, 'anarray2')

        # Ensure that the new name exists
        array_ = self.h5file.root.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/anarray2")
        self.assertEqual(array_._v_depth, 1)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.anarray

    def test07c_renameLeaf(self):
        """Checking renaming Leaves and modify attributes after that."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.anarray, 'anarray2')
        array_ = self.h5file.root.anarray2
        array_.attrs.TITLE = "hello"

        # Ensure that the new attribute has been written correctly
        self.assertEqual(array_.title, "hello")
        self.assertEqual(array_.attrs.TITLE, "hello")

    def test07d_renameLeaf(self):
        """Checking renaming a Group under a nested group."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.agroup.anarray2, 'anarray3')

        # Ensure that we can access n attributes in the new group
        node = self.h5file.root.agroup.anarray3
        self.assertEqual(node._v_title, "Array title 2")

    def test08_renameToExistingLeaf(self):
        """Checking renaming a node to an existing name."""

        self._reopen(mode="r+", )

        # Try to get the previous object with the old name
        with self.assertRaises(NodeError):
            self.h5file.rename_node(self.h5file.root.anarray, 'array')

        # Now overwrite the destination node.
        anarray = self.h5file.root.anarray
        self.h5file.rename_node(anarray, 'array', overwrite=True)
        self.assertTrue('/anarray' not in self.h5file)
        self.assertTrue(self.h5file.root.array is anarray)

    def test08b_renameToNotValidNaturalName(self):
        """Checking renaming a node to a non-valid natural name"""

        self._reopen(mode="r+", )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=NaturalNameWarning)

            # Try to get the previous object with the old name
            with self.assertRaises(NaturalNameWarning):
                self.h5file.rename_node(self.h5file.root.anarray, 'array 2')

    def test09_renameGroup(self):
        """Checking renaming a Group and access it after a close/open."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.agroup, 'agroup3')

        # Open this file in read-only mode
        self._reopen()

        # Ensure that the new name exists
        group = self.h5file.root.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/agroup3")

        # The children of this group also must be accessible through the
        # new name path
        group2 = self.h5file.get_node("/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/agroup3/agroup3")

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.agroup

        # Try to get a child with the old pathname
        with self.assertRaises(LookupError):
            self.h5file.get_node("/agroup/agroup3")

    def test09b_renameGroup(self):
        """Checking renaming a Group and access it immediately."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.agroup, 'agroup3')

        # Ensure that the new name exists
        group = self.h5file.root.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/agroup3")

        # The children of this group also must be accessible through the
        # new name path
        group2 = self.h5file.get_node("/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/agroup3/agroup3")

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.agroup

        # Try to get a child with the old pathname
        with self.assertRaises(LookupError):
            self.h5file.get_node("/agroup/agroup3")

    def test09c_renameGroup(self):
        """Checking renaming a Group and modify attributes afterwards."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.agroup, 'agroup3')

        # Ensure that we can modify attributes in the new group
        group = self.h5file.root.agroup3
        group._v_attrs.TITLE = "Hello"
        self.assertEqual(group._v_title, "Hello")
        self.assertEqual(group._v_attrs.TITLE, "Hello")

    def test09d_renameGroup(self):
        """Checking renaming a Group under a nested group."""

        self._reopen(mode="r+", )
        self.h5file.rename_node(self.h5file.root.agroup.agroup3, 'agroup4')

        # Ensure that we can access n attributes in the new group
        group = self.h5file.root.agroup.agroup4
        self.assertEqual(group._v_title, "Group title 3")

    def test09e_renameGroup(self):
        """Checking renaming a Group with nested groups in the LRU cache."""
        # This checks for ticket #126.

        self._reopen(mode="r+", )

        # Load intermediate groups and keep a nested one alive.
        g = self.h5file.root.agroup.agroup3.agroup4
        self.assertTrue(g is not None)
        self.h5file.rename_node('/', name='agroup', newname='agroup_')

        # see ticket #126
        self.assertTrue('/agroup_/agroup4' not in self.h5file)

        self.assertTrue('/agroup' not in self.h5file)
        for newpath in ['/agroup_', '/agroup_/agroup3',
                        '/agroup_/agroup3/agroup4']:
            self.assertTrue(newpath in self.h5file)
            self.assertEqual(
                newpath, self.h5file.get_node(newpath)._v_pathname)

    def test10_moveLeaf(self):
        """Checking moving a leave and access it after a close/open."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")
        self.h5file.move_node(self.h5file.root.anarray, newgroup, 'anarray2')

        # Open this file in read-only mode
        self._reopen()

        # Ensure that the new name exists
        array_ = self.h5file.root.newgroup.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/newgroup/anarray2")
        self.assertEqual(array_._v_depth, 2)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.anarray

    def test10b_moveLeaf(self):
        """Checking moving a leave and access it without a close/open."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")
        self.h5file.move_node(self.h5file.root.anarray, newgroup, 'anarray2')

        # Ensure that the new name exists
        array_ = self.h5file.root.newgroup.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/newgroup/anarray2")
        self.assertEqual(array_._v_depth, 2)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.anarray

    def test10c_moveLeaf(self):
        """Checking moving Leaves and modify attributes after that."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")
        self.h5file.move_node(self.h5file.root.anarray, newgroup, 'anarray2')
        array_ = self.h5file.root.newgroup.anarray2
        array_.attrs.TITLE = "hello"

        # Ensure that the new attribute has been written correctly
        self.assertEqual(array_.title, "hello")
        self.assertEqual(array_.attrs.TITLE, "hello")

    def test10d_moveToExistingLeaf(self):
        """Checking moving a leaf to an existing name."""

        self._reopen(mode="r+", )

        # Try to get the previous object with the old name
        with self.assertRaises(NodeError):
            self.h5file.move_node(
                self.h5file.root.anarray, self.h5file.root, 'array')

    def test10_2_moveTable(self):
        """Checking moving a table and access it after a close/open."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")
        self.h5file.move_node(self.h5file.root.atable, newgroup, 'atable2')

        # Open this file in read-only mode
        self._reopen()

        # Ensure that the new name exists
        table_ = self.h5file.root.newgroup.atable2
        self.assertEqual(table_.name, "atable2")
        self.assertEqual(table_._v_pathname, "/newgroup/atable2")
        self.assertEqual(table_._v_depth, 2)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.atable

    def test10_2b_moveTable(self):
        """Checking moving a table and access it without a close/open."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")
        self.h5file.move_node(self.h5file.root.atable, newgroup, 'atable2')

        # Ensure that the new name exists
        table_ = self.h5file.root.newgroup.atable2
        self.assertEqual(table_.name, "atable2")
        self.assertEqual(table_._v_pathname, "/newgroup/atable2")
        self.assertEqual(table_._v_depth, 2)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.atable

    def test10_2b_bis_moveTable(self):
        """Checking moving a table and use cached row without a close/open."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")

        # Cache the Row attribute prior to the move
        row = self.h5file.root.atable.row
        self.h5file.move_node(self.h5file.root.atable, newgroup, 'atable2')

        # Ensure that the new name exists
        table_ = self.h5file.root.newgroup.atable2
        self.assertEqual(table_.name, "atable2")
        self.assertEqual(table_._v_pathname, "/newgroup/atable2")
        self.assertEqual(table_._v_depth, 2)

        # Ensure that cache Row attribute has been updated
        row = table_.row
        self.assertEqual(table_._v_pathname, row.table._v_pathname)
        nrows = table_.nrows

        # Add a new row just to make sure that this works
        row.append()
        table_.flush()
        self.assertEqual(table_.nrows, nrows + 1)

    def test10_2c_moveTable(self):
        """Checking moving tables and modify attributes after that."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group("/", "newgroup")
        self.h5file.move_node(self.h5file.root.atable, newgroup, 'atable2')
        table_ = self.h5file.root.newgroup.atable2
        table_.attrs.TITLE = "hello"

        # Ensure that the new attribute has been written correctly
        self.assertEqual(table_.title, "hello")
        self.assertEqual(table_.attrs.TITLE, "hello")

    def test10_2d_moveToExistingTable(self):
        """Checking moving a table to an existing name."""

        self._reopen(mode="r+", )

        # Try to get the previous object with the old name
        with self.assertRaises(NodeError):
            self.h5file.move_node(self.h5file.root.atable, self.h5file.root,
                                  'table')

    def test10_2e_moveToExistingTableOverwrite(self):
        """Checking moving a table to an existing name, overwriting it."""

        self._reopen(mode="r+", )

        srcNode = self.h5file.root.atable
        self.h5file.move_node(srcNode, self.h5file.root, 'table',
                              overwrite=True)
        dstNode = self.h5file.root.table

        self.assertTrue(srcNode is dstNode)

    def test11_moveGroup(self):
        """Checking moving a Group and access it after a close/open."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group(self.h5file.root, 'newgroup')
        self.h5file.move_node(self.h5file.root.agroup, newgroup, 'agroup3')

        # Open this file in read-only mode
        self._reopen()

        # Ensure that the new name exists
        group = self.h5file.root.newgroup.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/newgroup/agroup3")
        self.assertEqual(group._v_depth, 2)

        # The children of this group must also be accessible through the
        # new name path
        group2 = self.h5file.get_node("/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_depth, 3)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.agroup

        # Try to get a child with the old pathname
        with self.assertRaises(LookupError):
            self.h5file.get_node("/agroup/agroup3")

    def test11b_moveGroup(self):
        """Checking moving a Group and access it immediately."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group(self.h5file.root, 'newgroup')
        self.h5file.move_node(self.h5file.root.agroup, newgroup, 'agroup3')

        # Ensure that the new name exists
        group = self.h5file.root.newgroup.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/newgroup/agroup3")
        self.assertEqual(group._v_depth, 2)

        # The children of this group must also be accessible through the
        # new name path
        group2 = self.h5file.get_node("/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_depth, 3)

        # Try to get the previous object with the old name
        with self.assertRaises(LookupError):
            self.h5file.root.agroup

        # Try to get a child with the old pathname
        with self.assertRaises(LookupError):
            self.h5file.get_node("/agroup/agroup3")

    def test11c_moveGroup(self):
        """Checking moving a Group and modify attributes afterwards."""

        self._reopen(mode="r+", )
        newgroup = self.h5file.create_group(self.h5file.root, 'newgroup')
        self.h5file.move_node(self.h5file.root.agroup, newgroup, 'agroup3')

        # Ensure that we can modify attributes in the new group
        group = self.h5file.root.newgroup.agroup3
        group._v_attrs.TITLE = "Hello"
        group._v_attrs.hola = "Hello"
        self.assertEqual(group._v_title, "Hello")
        self.assertEqual(group._v_attrs.TITLE, "Hello")
        self.assertEqual(group._v_attrs.hola, "Hello")

    def test11d_moveToExistingGroup(self):
        """Checking moving a group to an existing name."""

        self._reopen(mode="r+", )

        # Try to get the previous object with the old name
        with self.assertRaises(NodeError):
            self.h5file.move_node(self.h5file.root.agroup, self.h5file.root,
                                  'agroup2')

    def test11e_moveToExistingGroupOverwrite(self):
        """Checking moving a group to an existing name, overwriting it."""

        self._reopen(mode="r+", )

        # agroup2 -> agroup
        srcNode = self.h5file.root.agroup2
        self.h5file.move_node(srcNode, self.h5file.root, 'agroup',
                              overwrite=True)
        dstNode = self.h5file.root.agroup

        self.assertTrue(srcNode is dstNode)

    def test12a_moveNodeOverItself(self):
        """Checking moving a node over itself."""

        self._reopen(mode="r+", )

        # array -> array
        srcNode = self.h5file.root.array
        self.h5file.move_node(srcNode, self.h5file.root, 'array')
        dstNode = self.h5file.root.array

        self.assertTrue(srcNode is dstNode)

    def test12b_moveGroupIntoItself(self):
        """Checking moving a group into itself."""

        self._reopen(mode="r+", )
        with self.assertRaises(NodeError):
            # agroup2 -> agroup2/
            self.h5file.move_node(self.h5file.root.agroup2,
                                  self.h5file.root.agroup2)

    def test13a_copyLeaf(self):
        """Copying a leaf."""

        self._reopen(mode="r+", )

        # array => agroup2/
        new_node = self.h5file.copy_node(self.h5file.root.array,
                                         self.h5file.root.agroup2)
        dstNode = self.h5file.root.agroup2.array

        self.assertTrue(new_node is dstNode)

    def test13b_copyGroup(self):
        """Copying a group."""

        self._reopen(mode="r+", )

        # agroup2 => agroup/
        new_node = self.h5file.copy_node(self.h5file.root.agroup2,
                                         self.h5file.root.agroup)
        dstNode = self.h5file.root.agroup.agroup2

        self.assertTrue(new_node is dstNode)

    def test13c_copyGroupSelf(self):
        """Copying a group into itself."""

        self._reopen(mode="r+", )

        # agroup2 => agroup2/
        new_node = self.h5file.copy_node(self.h5file.root.agroup2,
                                         self.h5file.root.agroup2)
        dstNode = self.h5file.root.agroup2.agroup2

        self.assertTrue(new_node is dstNode)

    def test13d_copyGroupRecursive(self):
        """Recursively copying a group."""

        self._reopen(mode="r+", )

        # agroup => agroup2/
        new_node = self.h5file.copy_node(
            self.h5file.root.agroup, self.h5file.root.agroup2, recursive=True)
        dstNode = self.h5file.root.agroup2.agroup

        self.assertTrue(new_node is dstNode)
        dstChild1 = dstNode.anarray1
        self.assertTrue(dstChild1 is not None)
        dstChild2 = dstNode.anarray2
        self.assertTrue(dstChild2 is not None)
        dstChild3 = dstNode.agroup3
        self.assertTrue(dstChild3 is not None)

    def test13e_copyRootRecursive(self):
        """Recursively copying the root group into the root of another file."""

        self._reopen(mode="r+", )
        h5fname2 = tempfile.mktemp(".h5")
        h5file2 = tb.open_file(
            h5fname2, mode="w", )
        try:
            # h5file.root => h5file2.root
            new_node = self.h5file.copy_node(
                self.h5file.root, h5file2.root, recursive=True)
            dstNode = h5file2.root

            self.assertTrue(new_node is dstNode)
            self.assertTrue("/agroup" in h5file2)
            self.assertTrue("/agroup/anarray1" in h5file2)
            self.assertTrue("/agroup/agroup3" in h5file2)

        finally:
            h5file2.close()
            os.remove(h5fname2)

    def test13f_copyRootRecursive(self):
        """Recursively copying the root group into a group in another file."""

        self._reopen(mode="r+", )
        h5fname2 = tempfile.mktemp(".h5")
        h5file2 = tb.open_file(
            h5fname2, mode="w", )
        try:
            h5file2.create_group('/', 'agroup2')

            # fileh.root => h5file2.root.agroup2
            new_node = self.h5file.copy_node(
                self.h5file.root, h5file2.root.agroup2, recursive=True)
            dstNode = h5file2.root.agroup2

            self.assertTrue(new_node is dstNode)
            self.assertTrue("/agroup2/agroup" in h5file2)
            self.assertTrue("/agroup2/agroup/anarray1" in h5file2)
            self.assertTrue("/agroup2/agroup/agroup3" in h5file2)

        finally:
            h5file2.close()
            os.remove(h5fname2)

    def test13g_copyRootItself(self):
        """Recursively copying the root group into itself."""

        self._reopen(mode="r+", )
        agroup2 = self.h5file.root
        self.assertTrue(agroup2 is not None)

        # h5file.root => h5file.root
        self.assertRaises(IOError, self.h5file.copy_node,
                          self.h5file.root, self.h5file.root, recursive=True)

    def test14a_copyNodeExisting(self):
        """Copying over an existing node."""

        self._reopen(mode="r+", )

        with self.assertRaises(NodeError):
            # agroup2 => agroup
            self.h5file.copy_node(self.h5file.root.agroup2, newname='agroup')

    def test14b_copyNodeExistingOverwrite(self):
        """Copying over an existing node, overwriting it."""

        self._reopen(mode="r+", )

        # agroup2 => agroup
        new_node = self.h5file.copy_node(self.h5file.root.agroup2,
                                         newname='agroup', overwrite=True)
        dstNode = self.h5file.root.agroup

        self.assertTrue(new_node is dstNode)

    def test14b2_copyNodeExistingOverwrite(self):
        """Copying over an existing node in other file, overwriting it."""

        self._reopen(mode="r+", )

        h5fname2 = tempfile.mktemp(".h5")
        h5file2 = tb.open_file(
            h5fname2, mode="w", )

        try:
            # file1:/anarray1 => h5fname2:/anarray1
            new_node = self.h5file.copy_node(self.h5file.root.agroup.anarray1,
                                             newparent=h5file2.root)
            # file1:/ => h5fname2:/
            new_node = self.h5file.copy_node(self.h5file.root, h5file2.root,
                                             overwrite=True, recursive=True)
            dstNode = h5file2.root

            self.assertTrue(new_node is dstNode)
        finally:
            h5file2.close()
            os.remove(h5fname2)

    def test14c_copyNodeExistingSelf(self):
        """Copying over self."""

        self._reopen(mode="r+", )

        with self.assertRaises(NodeError):
            # agroup => agroup
            self.h5file.copy_node(self.h5file.root.agroup, newname='agroup')

    def test14d_copyNodeExistingOverwriteSelf(self):
        """Copying over self, trying to overwrite."""

        self._reopen(mode="r+", )

        with self.assertRaises(NodeError):
            # agroup => agroup
            self.h5file.copy_node(
                self.h5file.root.agroup, newname='agroup', overwrite=True)

    def test14e_copyGroupSelfRecursive(self):
        """Recursively copying a group into itself."""

        self._reopen(mode="r+", )

        with self.assertRaises(NodeError):
            # agroup => agroup/
            self.h5file.copy_node(self.h5file.root.agroup,
                                  self.h5file.root.agroup, recursive=True)

    def test15a_oneStepMove(self):
        """Moving and renaming a node in a single action."""

        self._reopen(mode="r+", )

        # anarray1 -> agroup/array
        srcNode = self.h5file.root.anarray1
        self.h5file.move_node(srcNode, self.h5file.root.agroup, 'array')
        dstNode = self.h5file.root.agroup.array

        self.assertTrue(srcNode is dstNode)

    def test15b_oneStepCopy(self):
        """Copying and renaming a node in a single action."""

        self._reopen(mode="r+", )

        # anarray1 => agroup/array
        new_node = self.h5file.copy_node(
            self.h5file.root.anarray1, self.h5file.root.agroup, 'array')
        dstNode = self.h5file.root.agroup.array

        self.assertTrue(new_node is dstNode)

    def test16a_fullCopy(self):
        """Copying full data and user attributes."""

        self._reopen(mode="r+", )

        # agroup => groupcopy
        srcNode = self.h5file.root.agroup
        new_node = self.h5file.copy_node(
            srcNode, newname='groupcopy', recursive=True)
        dstNode = self.h5file.root.groupcopy

        self.assertTrue(new_node is dstNode)
        self.assertEqual(srcNode._v_attrs.testattr, dstNode._v_attrs.testattr)
        self.assertEqual(
            srcNode.anarray1.attrs.testattr, dstNode.anarray1.attrs.testattr)
        self.assertEqual(srcNode.anarray1.read(), dstNode.anarray1.read())

    def test16b_partialCopy(self):
        """Copying partial data and no user attributes."""

        self._reopen(mode="r+", )

        # agroup => groupcopy
        srcNode = self.h5file.root.agroup
        new_node = self.h5file.copy_node(
            srcNode, newname='groupcopy',
            recursive=True, copyuserattrs=False,
            start=0, stop=5, step=2)
        dstNode = self.h5file.root.groupcopy

        self.assertTrue(new_node is dstNode)
        self.assertFalse(hasattr(dstNode._v_attrs, 'testattr'))
        self.assertFalse(hasattr(dstNode.anarray1.attrs, 'testattr'))
        self.assertEqual(srcNode.anarray1.read()[
                         0:5:2], dstNode.anarray1.read())

    def test16c_fullCopy(self):
        """Copying full data and user attributes (from file to file)."""

        self._reopen(mode="r+", )

        h5fname2 = tempfile.mktemp(".h5")
        h5file2 = tb.open_file(
            h5fname2, mode="w", )

        try:
            # file1:/ => h5fname2:groupcopy
            srcNode = self.h5file.root
            new_node = self.h5file.copy_node(
                srcNode, h5file2.root, newname='groupcopy', recursive=True)
            dstNode = h5file2.root.groupcopy

            self.assertTrue(new_node is dstNode)
            self.assertEqual(srcNode._v_attrs.testattr,
                             dstNode._v_attrs.testattr)
            self.assertEqual(
                srcNode.agroup.anarray1.attrs.testattr,
                dstNode.agroup.anarray1.attrs.testattr)
            self.assertEqual(srcNode.agroup.anarray1.read(),
                             dstNode.agroup.anarray1.read())
        finally:
            h5file2.close()
            os.remove(h5fname2)

    def test17a_CopyChunkshape(self):
        """Copying dataset with a chunkshape."""

        self._reopen(mode="r+", )
        srcTable = self.h5file.root.table
        newTable = self.h5file.copy_node(
            srcTable, newname='tablecopy', chunkshape=11)

        self.assertEqual(newTable.chunkshape, (11,))
        self.assertNotEqual(srcTable.chunkshape, newTable.chunkshape)

    def test17b_CopyChunkshape(self):
        """Copying dataset with a chunkshape with 'keep' value."""

        self._reopen(mode="r+", )
        srcTable = self.h5file.root.table
        newTable = self.h5file.copy_node(
            srcTable, newname='tablecopy', chunkshape='keep')

        self.assertEqual(srcTable.chunkshape, newTable.chunkshape)

    def test17c_CopyChunkshape(self):
        """Copying dataset with a chunkshape with 'auto' value."""

        self._reopen(mode="r+", )
        srcTable = self.h5file.root.table
        newTable = self.h5file.copy_node(
            srcTable, newname='tablecopy', chunkshape=11)
        newTable2 = self.h5file.copy_node(
            newTable, newname='tablecopy2', chunkshape='auto')

        self.assertEqual(srcTable.chunkshape, newTable2.chunkshape)

    def test18_closedRepr(self):
        """Representing a closed node as a string."""

        self._reopen()

        for node in [self.h5file.root.agroup, self.h5file.root.anarray]:
            node._f_close()
            self.assertTrue('closed' in str(node))
            self.assertTrue('closed' in repr(node))

    def test19_fileno(self):
        """Checking that the 'fileno()' method works."""

        # Open the old HDF5 file
        self._reopen(mode="r", )

        # Get the file descriptor for this file
        fd = self.h5file.fileno()
        self.assertTrue(fd >= 0)


