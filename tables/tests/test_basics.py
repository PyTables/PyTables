# -*- coding: utf-8 -*-

import os
import shutil
import sys
import unittest
import tempfile
import warnings
import subprocess

try:
    import multiprocessing as mp
    multiprocessing_imported = True
except ImportError:
    multiprocessing_imported = False

import numpy

import tables
import tables.flavor
from tables import *
from tables.flavor import all_flavors, array_of_flavor
from tables.tests import common
from tables.parameters import NODE_CACHE_SLOTS
from tables.description import descr_from_dtype, dtype_from_descr

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


class OpenFileFailureTestCase(common.PyTablesTestCase):
    def setUp(self):
        import tables.file

        self.N = len(tables.file._open_files)
        self.open_files = tables.file._open_files

    def test01_openFile(self):
        """Checking opening of a non existing file"""

        filename = tempfile.mktemp(".h5")
        try:
            fileh = open_file(filename)
            fileh.close()
        except IOError:
            self.assertEqual(self.N, len(self.open_files))
        else:
            self.fail("IOError exception not raised")

    def test02_openFile(self):
        """Checking opening of an existing non HDF5 file"""

        # create a dummy file
        filename = tempfile.mktemp(".h5")
        open(filename, 'wb').close()

        # Try to open the dummy file
        try:
            try:
                fileh = tables.open_file(filename)
                fileh.close()
            except HDF5ExtError:
                self.assertEqual(self.N, len(self.open_files))
            else:
                self.fail("HDF5ExtError exception not raised")
        finally:
            os.remove(filename)


class OpenFileTestCase(common.PyTablesTestCase):

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        fileh = open_file(self.file, mode="w", title="File title",
                          node_cache_slots=self.nodeCacheSlots)
        root = fileh.root
        # Create an array
        fileh.create_array(root, 'array', [1, 2], title="Array example")
        fileh.create_table(root, 'table', {'var1': IntCol()}, "Table example")
        root._v_attrs.testattr = 41

        # Create another array object
        fileh.create_array(root, 'anarray', [1], "Array title")
        fileh.create_table(root, 'atable', {'var1': IntCol()}, "Table title")

        # Create a group object
        group = fileh.create_group(root, 'agroup',
                                   "Group title")
        group._v_attrs.testattr = 42

        # Create a some objects there
        array1 = fileh.create_array(group, 'anarray1',
                                    [1, 2, 3, 4, 5, 6, 7], "Array title 1")
        array1.attrs.testattr = 42
        fileh.create_array(group, 'anarray2', [2], "Array title 2")
        fileh.create_table(group, 'atable1', {
                           'var1': IntCol()}, "Table title 1")
        ra = numpy.rec.array([(1, 11, 'a')], formats='u1,f4,a1')
        fileh.create_table(group, 'atable2', ra, "Table title 2")

        # Create a lonely group in first level
        fileh.create_group(root, 'agroup2', "Group title 2")

        # Create a new group in the second level
        group3 = fileh.create_group(group, 'agroup3', "Group title 3")

        # Create a new group in the third level
        fileh.create_group(group3, 'agroup4', "Group title 4")

        # Create an array in the root with the same name as one in 'agroup'
        fileh.create_array(root, 'anarray1', [1, 2],
                           title="Array example")

        fileh.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)
        common.cleanup(self)

    def test00_newFile(self):
        """Checking creation of a new file"""

        # Create an HDF5 file
        file = tempfile.mktemp(".h5")
        fileh = open_file(
            file, mode="w", node_cache_slots=self.nodeCacheSlots)
        fileh.create_array(fileh.root, 'array', [1, 2],
                           title="Array example")
        # Get the CLASS attribute of the arr object
        class_ = fileh.root.array.attrs.CLASS

        # Close and delete the file
        fileh.close()
        os.remove(file)

        self.assertEqual(class_.capitalize(), "Array")

    def test00_newFile_unicode_filename(self):
        temp_dir = tempfile.mkdtemp()
        file_path = unicode(os.path.join(temp_dir, 'test.h5'))
        with open_file(file_path, 'w') as fileh:
            self.assertTrue(fileh, File)
        shutil.rmtree(temp_dir)

    def test00_newFile_numpy_str_filename(self):
        temp_dir = tempfile.mkdtemp()
        file_path = numpy.str_(os.path.join(temp_dir, 'test.h5'))
        with open_file(file_path, 'w') as fileh:
            self.assertTrue(fileh, File)
        shutil.rmtree(temp_dir)

    def test00_newFile_numpy_unicode_filename(self):
        temp_dir = tempfile.mkdtemp()
        file_path = numpy.unicode_(os.path.join(temp_dir, 'test.h5'))
        with open_file(file_path, 'w') as fileh:
            self.assertTrue(fileh, File)
        shutil.rmtree(temp_dir)

    def test01_openFile(self):
        """Checking opening of an existing file"""

        # Open the old HDF5 file
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Get the CLASS attribute of the arr object
        title = fileh.root.array.get_attr("TITLE")

        self.assertEqual(title, "Array example")
        fileh.close()

    def test02_appendFile(self):
        """Checking appending objects to an existing file"""

        # Append a new array to the existing file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.create_array(fileh.root, 'array2', [3, 4],
                           title="Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Get the CLASS attribute of the arr object
        title = fileh.root.array2.get_attr("TITLE")

        self.assertEqual(title, "Title example 2")
        fileh.close()

    def test02b_appendFile2(self):
        """Checking appending objects to an existing file ("a" version)"""

        # Append a new array to the existing file
        fileh = open_file(
            self.file, mode="a", node_cache_slots=self.nodeCacheSlots)
        fileh.create_array(fileh.root, 'array2', [3, 4],
                           title="Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Get the CLASS attribute of the arr object
        title = fileh.root.array2.get_attr("TITLE")

        self.assertEqual(title, "Title example 2")
        fileh.close()

    # Begin to raise errors...

    def test03_appendErrorFile(self):
        """Checking appending objects to an existing file in "w" mode"""

        # Append a new array to the existing file but in write mode
        # so, the existing file should be deleted!
        fileh = open_file(
            self.file, mode="w", node_cache_slots=self.nodeCacheSlots)
        fileh.create_array(fileh.root, 'array2', [3, 4],
                           title="Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)

        try:
            # Try to get the 'array' object in the old existing file
            fileh.root.array
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test04a_openErrorFile(self):
        """Checking opening a non-existing file for reading"""

        try:
            open_file("nonexistent.h5", mode="r",
                      node_cache_slots=self.nodeCacheSlots)
        except IOError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IOError was catched!"
                print value
        else:
            self.fail("expected an IOError")

    def test04b_alternateRootFile(self):
        """Checking alternate root access to the object tree"""

        # Open the existent HDF5 file
        fileh = open_file(self.file, mode="r", root_uep="/agroup",
                          node_cache_slots=self.nodeCacheSlots)
        # Get the CLASS attribute of the arr object
        if common.verbose:
            print "\nFile tree dump:", fileh
        title = fileh.root.anarray1.get_attr("TITLE")
        # Get the node again, as this can trigger errors in some situations
        anarray1 = fileh.root.anarray1
        self.assertTrue(anarray1 is not None)

        self.assertEqual(title, "Array title 1")
        fileh.close()

    # This test works well, but HDF5 emits a series of messages that
    # may loose the user. It is better to deactivate it.
    def notest04c_alternateRootFile(self):
        """Checking non-existent alternate root access to the object tree"""

        try:
            open_file(self.file, mode="r", root_uep="/nonexistent",
                      node_cache_slots=self.nodeCacheSlots)
        except RuntimeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected an IOError")

    def test05a_removeGroupRecursively(self):
        """Checking removing a group recursively"""

        # Delete a group with leafs
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        try:
            fileh.remove_node(fileh.root.agroup)
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected a NodeError")

        # This should work now
        fileh.remove_node(fileh.root, 'agroup', recursive=1)

        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Try to get the removed object
        try:
            fileh.root.agroup
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child of the removed object
        try:
            fileh.get_node("/agroup/agroup3")
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test05b_removeGroupRecursively(self):
        """Checking removing a group recursively and access to it
        immediately"""

        if common.verbose:
            print '\n', '-=' * 30
            print ("Running %s.test05b_removeGroupRecursively..." %
                   self.__class__.__name__)

        # Delete a group with leafs
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        try:
            fileh.remove_node(fileh.root, 'agroup')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected a NodeError")

        # This should work now
        fileh.remove_node(fileh.root, 'agroup', recursive=1)

        # Try to get the removed object
        try:
            fileh.root.agroup
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child of the removed object
        try:
            fileh.get_node("/agroup/agroup3")
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06_removeNodeWithDel(self):
        """Checking removing a node using ``__delattr__()``"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        try:
            # This should fail because there is no *Python attribute*
            # called ``agroup``.
            del fileh.root.agroup
        except AttributeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next AttributeError was catched!"
                print value
        else:
            self.fail("expected an AttributeError")

        fileh.close()

    def test06a_removeGroup(self):
        """Checking removing a lonely group from an existing file"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.remove_node(fileh.root, 'agroup2')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Try to get the removed object
        try:
            fileh.root.agroup2
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06b_removeLeaf(self):
        """Checking removing Leaves from an existing file"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.remove_node(fileh.root, 'anarray')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Try to get the removed object
        try:
            fileh.root.anarray
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06c_removeLeaf(self):
        """Checking removing Leaves and access it immediately"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.remove_node(fileh.root, 'anarray')

        # Try to get the removed object
        try:
            fileh.root.anarray
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06d_removeLeaf(self):
        """Checking removing a non-existent node"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # Try to get the removed object
        try:
            fileh.remove_node(fileh.root, 'nonexistent')
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06e_removeTable(self):
        """Checking removing Tables from an existing file"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.remove_node(fileh.root, 'atable')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Try to get the removed object
        try:
            fileh.root.atable
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test07_renameLeaf(self):
        """Checking renaming a leave and access it after a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.anarray, 'anarray2')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Ensure that the new name exists
        array_ = fileh.root.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/anarray2")
        self.assertEqual(array_._v_depth, 1)
        # Try to get the previous object with the old name
        try:
            fileh.root.anarray
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test07b_renameLeaf(self):
        """Checking renaming Leaves and accesing them immediately"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.anarray, 'anarray2')

        # Ensure that the new name exists
        array_ = fileh.root.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/anarray2")
        self.assertEqual(array_._v_depth, 1)
        # Try to get the previous object with the old name
        try:
            fileh.root.anarray
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test07c_renameLeaf(self):
        """Checking renaming Leaves and modify attributes after that"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.anarray, 'anarray2')
        array_ = fileh.root.anarray2
        array_.attrs.TITLE = "hello"
        # Ensure that the new attribute has been written correctly
        self.assertEqual(array_.title, "hello")
        self.assertEqual(array_.attrs.TITLE, "hello")
        fileh.close()

    def test07d_renameLeaf(self):
        """Checking renaming a Group under a nested group"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.agroup.anarray2, 'anarray3')

        # Ensure that we can access n attributes in the new group
        node = fileh.root.agroup.anarray3
        self.assertEqual(node._v_title, "Array title 2")
        fileh.close()

    def test08_renameToExistingLeaf(self):
        """Checking renaming a node to an existing name"""

        # Open this file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        # Try to get the previous object with the old name
        try:
            fileh.rename_node(fileh.root.anarray, 'array')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        # Now overwrite the destination node.
        anarray = fileh.root.anarray
        fileh.rename_node(anarray, 'array', overwrite=True)
        self.assertTrue('/anarray' not in fileh)
        self.assertTrue(fileh.root.array is anarray)
        fileh.close()

    def test08b_renameToNotValidNaturalName(self):
        """Checking renaming a node to a non-valid natural name"""

        # Open this file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        warnings.filterwarnings("error", category=NaturalNameWarning)
        # Try to get the previous object with the old name
        try:
            fileh.rename_node(fileh.root.anarray, 'array 2')
        except NaturalNameWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NaturalNameWarning was catched!"
                print value
        else:
            self.fail("expected an NaturalNameWarning")
        # Reset the warning
        warnings.filterwarnings("default", category=NaturalNameWarning)
        fileh.close()

    def test09_renameGroup(self):
        """Checking renaming a Group and access it after a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.agroup, 'agroup3')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Ensure that the new name exists
        group = fileh.root.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/agroup3")
        # The children of this group also must be accessible through the
        # new name path
        group2 = fileh.get_node("/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/agroup3/agroup3")
        # Try to get the previous object with the old name
        try:
            fileh.root.agroup
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            fileh.get_node("/agroup/agroup3")
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test09b_renameGroup(self):
        """Checking renaming a Group and access it immediately"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.agroup, 'agroup3')

        # Ensure that the new name exists
        group = fileh.root.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/agroup3")
        # The children of this group also must be accessible through the
        # new name path
        group2 = fileh.get_node("/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/agroup3/agroup3")
        # Try to get the previous object with the old name
        try:
            fileh.root.agroup
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            fileh.get_node("/agroup/agroup3")
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test09c_renameGroup(self):
        """Checking renaming a Group and modify attributes afterwards"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.agroup, 'agroup3')

        # Ensure that we can modify attributes in the new group
        group = fileh.root.agroup3
        group._v_attrs.TITLE = "Hello"
        self.assertEqual(group._v_title, "Hello")
        self.assertEqual(group._v_attrs.TITLE, "Hello")
        fileh.close()

    def test09d_renameGroup(self):
        """Checking renaming a Group under a nested group"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        fileh.rename_node(fileh.root.agroup.agroup3, 'agroup4')

        # Ensure that we can access n attributes in the new group
        group = fileh.root.agroup.agroup4
        self.assertEqual(group._v_title, "Group title 3")
        fileh.close()

    def test09e_renameGroup(self):
        """Checking renaming a Group with nested groups in the LRU cache"""
        # This checks for ticket #126.

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        # Load intermediate groups and keep a nested one alive.
        g = fileh.root.agroup.agroup3.agroup4
        self.assertTrue(g is not None)
        fileh.rename_node('/', name='agroup', newname='agroup_')
        self.assertTrue('/agroup_/agroup4' not in fileh)  # see ticket #126
        self.assertTrue('/agroup' not in fileh)
        for newpath in ['/agroup_', '/agroup_/agroup3',
                        '/agroup_/agroup3/agroup4']:
            self.assertTrue(newpath in fileh)
            self.assertEqual(newpath, fileh.get_node(newpath)._v_pathname)
        fileh.close()

    def test10_moveLeaf(self):
        """Checking moving a leave and access it after a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        fileh.move_node(fileh.root.anarray, newgroup, 'anarray2')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Ensure that the new name exists
        array_ = fileh.root.newgroup.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/newgroup/anarray2")
        self.assertEqual(array_._v_depth, 2)
        # Try to get the previous object with the old name
        try:
            fileh.root.anarray
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test10b_moveLeaf(self):
        """Checking moving a leave and access it without a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        fileh.move_node(fileh.root.anarray, newgroup, 'anarray2')

        # Ensure that the new name exists
        array_ = fileh.root.newgroup.anarray2
        self.assertEqual(array_.name, "anarray2")
        self.assertEqual(array_._v_pathname, "/newgroup/anarray2")
        self.assertEqual(array_._v_depth, 2)
        # Try to get the previous object with the old name
        try:
            fileh.root.anarray
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test10c_moveLeaf(self):
        """Checking moving Leaves and modify attributes after that"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        fileh.move_node(fileh.root.anarray, newgroup, 'anarray2')
        array_ = fileh.root.newgroup.anarray2
        array_.attrs.TITLE = "hello"
        # Ensure that the new attribute has been written correctly
        self.assertEqual(array_.title, "hello")
        self.assertEqual(array_.attrs.TITLE, "hello")
        fileh.close()

    def test10d_moveToExistingLeaf(self):
        """Checking moving a leaf to an existing name"""

        # Open this file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        # Try to get the previous object with the old name
        try:
            fileh.move_node(fileh.root.anarray, fileh.root, 'array')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test10_2_moveTable(self):
        """Checking moving a table and access it after a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        fileh.move_node(fileh.root.atable, newgroup, 'atable2')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Ensure that the new name exists
        table_ = fileh.root.newgroup.atable2
        self.assertEqual(table_.name, "atable2")
        self.assertEqual(table_._v_pathname, "/newgroup/atable2")
        self.assertEqual(table_._v_depth, 2)
        # Try to get the previous object with the old name
        try:
            fileh.root.atable
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test10_2b_moveTable(self):
        """Checking moving a table and access it without a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        fileh.move_node(fileh.root.atable, newgroup, 'atable2')

        # Ensure that the new name exists
        table_ = fileh.root.newgroup.atable2
        self.assertEqual(table_.name, "atable2")
        self.assertEqual(table_._v_pathname, "/newgroup/atable2")
        self.assertEqual(table_._v_depth, 2)
        # Try to get the previous object with the old name
        try:
            fileh.root.atable
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test10_2b_bis_moveTable(self):
        """Checking moving a table and use cached row without a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        # Cache the Row attribute prior to the move
        row = fileh.root.atable.row
        fileh.move_node(fileh.root.atable, newgroup, 'atable2')

        # Ensure that the new name exists
        table_ = fileh.root.newgroup.atable2
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
        fileh.close()

    def test10_2c_moveTable(self):
        """Checking moving tables and modify attributes after that"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group("/", "newgroup")
        fileh.move_node(fileh.root.atable, newgroup, 'atable2')
        table_ = fileh.root.newgroup.atable2
        table_.attrs.TITLE = "hello"
        # Ensure that the new attribute has been written correctly
        self.assertEqual(table_.title, "hello")
        self.assertEqual(table_.attrs.TITLE, "hello")
        fileh.close()

    def test10_2d_moveToExistingTable(self):
        """Checking moving a table to an existing name"""

        # Open this file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        # Try to get the previous object with the old name
        try:
            fileh.move_node(fileh.root.atable, fileh.root, 'table')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test10_2e_moveToExistingTableOverwrite(self):
        """Checking moving a table to an existing name, overwriting it"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        srcNode = fileh.root.atable
        fileh.move_node(srcNode, fileh.root, 'table', overwrite=True)
        dstNode = fileh.root.table

        self.assertTrue(srcNode is dstNode)
        fileh.close()

    def test11_moveGroup(self):
        """Checking moving a Group and access it after a close/open"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group(fileh.root, 'newgroup')
        fileh.move_node(fileh.root.agroup, newgroup, 'agroup3')
        fileh.close()

        # Open this file in read-only mode
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Ensure that the new name exists
        group = fileh.root.newgroup.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/newgroup/agroup3")
        self.assertEqual(group._v_depth, 2)
        # The children of this group must also be accessible through the
        # new name path
        group2 = fileh.get_node("/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_depth, 3)
        # Try to get the previous object with the old name
        try:
            fileh.root.agroup
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            fileh.get_node("/agroup/agroup3")
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test11b_moveGroup(self):
        """Checking moving a Group and access it immediately"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group(fileh.root, 'newgroup')
        fileh.move_node(fileh.root.agroup, newgroup, 'agroup3')
        # Ensure that the new name exists
        group = fileh.root.newgroup.agroup3
        self.assertEqual(group._v_name, "agroup3")
        self.assertEqual(group._v_pathname, "/newgroup/agroup3")
        self.assertEqual(group._v_depth, 2)
        # The children of this group must also be accessible through the
        # new name path
        group2 = fileh.get_node("/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_name, "agroup3")
        self.assertEqual(group2._v_pathname, "/newgroup/agroup3/agroup3")
        self.assertEqual(group2._v_depth, 3)
        # Try to get the previous object with the old name
        try:
            fileh.root.agroup
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            fileh.get_node("/agroup/agroup3")
        except LookupError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test11c_moveGroup(self):
        """Checking moving a Group and modify attributes afterwards"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        newgroup = fileh.create_group(fileh.root, 'newgroup')
        fileh.move_node(fileh.root.agroup, newgroup, 'agroup3')

        # Ensure that we can modify attributes in the new group
        group = fileh.root.newgroup.agroup3
        group._v_attrs.TITLE = "Hello"
        group._v_attrs.hola = "Hello"
        self.assertEqual(group._v_title, "Hello")
        self.assertEqual(group._v_attrs.TITLE, "Hello")
        self.assertEqual(group._v_attrs.hola, "Hello")
        fileh.close()

    def test11d_moveToExistingGroup(self):
        """Checking moving a group to an existing name"""

        # Open this file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        # Try to get the previous object with the old name
        try:
            fileh.move_node(fileh.root.agroup, fileh.root, 'agroup2')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test11e_moveToExistingGroupOverwrite(self):
        """Checking moving a group to an existing name, overwriting it"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup2 -> agroup
        srcNode = fileh.root.agroup2
        fileh.move_node(srcNode, fileh.root, 'agroup', overwrite=True)
        dstNode = fileh.root.agroup

        self.assertTrue(srcNode is dstNode)
        fileh.close()

    def test12a_moveNodeOverItself(self):
        """Checking moving a node over itself"""

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # array -> array
        srcNode = fileh.root.array
        fileh.move_node(srcNode, fileh.root, 'array')
        dstNode = fileh.root.array

        self.assertTrue(srcNode is dstNode)
        fileh.close()

    def test12b_moveGroupIntoItself(self):
        """Checking moving a group into itself"""

        # Open this file
        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        try:
            # agroup2 -> agroup2/
            fileh.move_node(fileh.root.agroup2, fileh.root.agroup2)
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test13a_copyLeaf(self):
        "Copying a leaf."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # array => agroup2/
        new_node = fileh.copy_node(fileh.root.array, fileh.root.agroup2)
        dstNode = fileh.root.agroup2.array

        self.assertTrue(new_node is dstNode)
        fileh.close()

    def test13b_copyGroup(self):
        "Copying a group."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup2 => agroup/
        new_node = fileh.copy_node(fileh.root.agroup2, fileh.root.agroup)
        dstNode = fileh.root.agroup.agroup2

        self.assertTrue(new_node is dstNode)
        fileh.close()

    def test13c_copyGroupSelf(self):
        "Copying a group into itself."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup2 => agroup2/
        new_node = fileh.copy_node(fileh.root.agroup2, fileh.root.agroup2)
        dstNode = fileh.root.agroup2.agroup2

        self.assertTrue(new_node is dstNode)
        fileh.close()

    def test13d_copyGroupRecursive(self):
        "Recursively copying a group."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup => agroup2/
        new_node = fileh.copy_node(
            fileh.root.agroup, fileh.root.agroup2, recursive=True)
        dstNode = fileh.root.agroup2.agroup

        self.assertTrue(new_node is dstNode)
        dstChild1 = dstNode.anarray1
        self.assertTrue(dstChild1 is not None)
        dstChild2 = dstNode.anarray2
        self.assertTrue(dstChild2 is not None)
        dstChild3 = dstNode.agroup3
        self.assertTrue(dstChild3 is not None)
        fileh.close()

    def test13e_copyRootRecursive(self):
        "Recursively copying the root group into the root of another file."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(
            file2, mode="w", node_cache_slots=self.nodeCacheSlots)

        # fileh.root => fileh2.root
        new_node = fileh.copy_node(
            fileh.root, fileh2.root, recursive=True)
        dstNode = fileh2.root

        self.assertTrue(new_node is dstNode)
        self.assertTrue("/agroup" in fileh2)
        self.assertTrue("/agroup/anarray1" in fileh2)
        self.assertTrue("/agroup/agroup3" in fileh2)

        fileh.close()
        fileh2.close()
        os.remove(file2)

    def test13f_copyRootRecursive(self):
        "Recursively copying the root group into a group in another file."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(
            file2, mode="w", node_cache_slots=self.nodeCacheSlots)
        fileh2.create_group('/', 'agroup2')

        # fileh.root => fileh2.root.agroup2
        new_node = fileh.copy_node(
            fileh.root, fileh2.root.agroup2, recursive=True)
        dstNode = fileh2.root.agroup2

        self.assertTrue(new_node is dstNode)
        self.assertTrue("/agroup2/agroup" in fileh2)
        self.assertTrue("/agroup2/agroup/anarray1" in fileh2)
        self.assertTrue("/agroup2/agroup/agroup3" in fileh2)

        fileh.close()
        fileh2.close()
        os.remove(file2)

    def test13g_copyRootItself(self):
        "Recursively copying the root group into itself."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        agroup2 = fileh.root
        self.assertTrue(agroup2 is not None)

        # fileh.root => fileh.root
        self.assertRaises(IOError, fileh.copy_node,
                          fileh.root, fileh.root, recursive=True)
        fileh.close()

    def test14a_copyNodeExisting(self):
        "Copying over an existing node."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        try:
            # agroup2 => agroup
            fileh.copy_node(fileh.root.agroup2, newname='agroup')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test14b_copyNodeExistingOverwrite(self):
        "Copying over an existing node, overwriting it."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup2 => agroup
        new_node = fileh.copy_node(fileh.root.agroup2, newname='agroup',
                                   overwrite=True)
        dstNode = fileh.root.agroup

        self.assertTrue(new_node is dstNode)
        fileh.close()

    def test14b2_copyNodeExistingOverwrite(self):
        "Copying over an existing node in other file, overwriting it."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(
            file2, mode="w", node_cache_slots=self.nodeCacheSlots)

        # file1:/anarray1 => file2:/anarray1
        new_node = fileh.copy_node(fileh.root.agroup.anarray1,
                                   newparent=fileh2.root)
        # file1:/ => file2:/
        new_node = fileh.copy_node(fileh.root, fileh2.root,
                                   overwrite=True, recursive=True)
        dstNode = fileh2.root

        self.assertTrue(new_node is dstNode)
        fileh.close()
        fileh2.close()
        os.remove(file2)

    def test14c_copyNodeExistingSelf(self):
        "Copying over self."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        try:
            # agroup => agroup
            fileh.copy_node(fileh.root.agroup, newname='agroup')
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test14d_copyNodeExistingOverwriteSelf(self):
        "Copying over self, trying to overwrite."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        try:
            # agroup => agroup
            fileh.copy_node(
                fileh.root.agroup, newname='agroup', overwrite=True)
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test14e_copyGroupSelfRecursive(self):
        "Recursively copying a group into itself."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        try:
            # agroup => agroup/
            fileh.copy_node(
                fileh.root.agroup, fileh.root.agroup, recursive=True)
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test15a_oneStepMove(self):
        "Moving and renaming a node in a single action."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # anarray1 -> agroup/array
        srcNode = fileh.root.anarray1
        fileh.move_node(srcNode, fileh.root.agroup, 'array')
        dstNode = fileh.root.agroup.array

        self.assertTrue(srcNode is dstNode)
        fileh.close()

    def test15b_oneStepCopy(self):
        "Copying and renaming a node in a single action."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # anarray1 => agroup/array
        new_node = fileh.copy_node(
            fileh.root.anarray1, fileh.root.agroup, 'array')
        dstNode = fileh.root.agroup.array

        self.assertTrue(new_node is dstNode)
        fileh.close()

    def test16a_fullCopy(self):
        "Copying full data and user attributes."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup => groupcopy
        srcNode = fileh.root.agroup
        new_node = fileh.copy_node(
            srcNode, newname='groupcopy', recursive=True)
        dstNode = fileh.root.groupcopy

        self.assertTrue(new_node is dstNode)
        self.assertEqual(srcNode._v_attrs.testattr, dstNode._v_attrs.testattr)
        self.assertEqual(
            srcNode.anarray1.attrs.testattr, dstNode.anarray1.attrs.testattr)
        self.assertEqual(srcNode.anarray1.read(), dstNode.anarray1.read())
        fileh.close()

    def test16b_partialCopy(self):
        "Copying partial data and no user attributes."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        # agroup => groupcopy
        srcNode = fileh.root.agroup
        new_node = fileh.copy_node(
            srcNode, newname='groupcopy',
            recursive=True, copyuserattrs=False,
            start=0, stop=5, step=2)
        dstNode = fileh.root.groupcopy

        self.assertTrue(new_node is dstNode)
        self.assertFalse(hasattr(dstNode._v_attrs, 'testattr'))
        self.assertFalse(hasattr(dstNode.anarray1.attrs, 'testattr'))
        self.assertEqual(srcNode.anarray1.read()[
                         0:5:2], dstNode.anarray1.read())
        fileh.close()

    def test16c_fullCopy(self):
        "Copying full data and user attributes (from file to file)."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)

        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(
            file2, mode="w", node_cache_slots=self.nodeCacheSlots)

        # file1:/ => file2:groupcopy
        srcNode = fileh.root
        new_node = fileh.copy_node(
            srcNode, fileh2.root, newname='groupcopy', recursive=True)
        dstNode = fileh2.root.groupcopy

        self.assertTrue(new_node is dstNode)
        self.assertEqual(srcNode._v_attrs.testattr, dstNode._v_attrs.testattr)
        self.assertEqual(
            srcNode.agroup.anarray1.attrs.testattr,
            dstNode.agroup.anarray1.attrs.testattr)
        self.assertEqual(srcNode.agroup.anarray1.read(),
                         dstNode.agroup.anarray1.read())

        fileh.close()
        fileh2.close()
        os.remove(file2)

    def test17a_CopyChunkshape(self):
        "Copying dataset with a chunkshape."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        srcTable = fileh.root.table
        newTable = fileh.copy_node(
            srcTable, newname='tablecopy', chunkshape=11)

        self.assertEqual(newTable.chunkshape, (11,))
        self.assertNotEqual(srcTable.chunkshape, newTable.chunkshape)
        fileh.close()

    def test17b_CopyChunkshape(self):
        "Copying dataset with a chunkshape with 'keep' value."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        srcTable = fileh.root.table
        newTable = fileh.copy_node(
            srcTable, newname='tablecopy', chunkshape='keep')

        self.assertEqual(srcTable.chunkshape, newTable.chunkshape)
        fileh.close()

    def test17c_CopyChunkshape(self):
        "Copying dataset with a chunkshape with 'auto' value."

        fileh = open_file(
            self.file, mode="r+", node_cache_slots=self.nodeCacheSlots)
        srcTable = fileh.root.table
        newTable = fileh.copy_node(
            srcTable, newname='tablecopy', chunkshape=11)
        newTable2 = fileh.copy_node(
            newTable, newname='tablecopy2', chunkshape='auto')

        self.assertEqual(srcTable.chunkshape, newTable2.chunkshape)
        fileh.close()

    def test18_closedRepr(self):
        "Representing a closed node as a string."
        fileh = open_file(
            self.file, node_cache_slots=self.nodeCacheSlots)
        for node in [fileh.root.agroup, fileh.root.anarray]:
            node._f_close()
            self.assertTrue('closed' in str(node))
            self.assertTrue('closed' in repr(node))
        fileh.close()

    def test19_fileno(self):
        """Checking that the 'fileno()' method works"""

        # Open the old HDF5 file
        fileh = open_file(
            self.file, mode="r", node_cache_slots=self.nodeCacheSlots)
        # Get the file descriptor for this file
        fd = fileh.fileno()
        if common.verbose:
            print "Value of fileno():", fd
        self.assertTrue(fd >= 0)
        fileh.close()


class NodeCacheOpenFile(OpenFileTestCase):
    nodeCacheSlots = NODE_CACHE_SLOTS


class NoNodeCacheOpenFile(OpenFileTestCase):
    nodeCacheSlots = 0


class DictNodeCacheOpenFile(OpenFileTestCase):
    nodeCacheSlots = -NODE_CACHE_SLOTS


class CheckFileTestCase(common.PyTablesTestCase):

    def test00_isHDF5File(self):
        """Checking is_hdf5_file function (TRUE case)"""

        # Create a PyTables file (and by so, an HDF5 file)
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode="w")
        fileh.create_array(fileh.root, 'array', [
                           1, 2], title="Title example")

        # For this method to run, it needs a closed file
        fileh.close()

        # When file has an HDF5 format, always returns 1
        if common.verbose:
            print "\nisHDF5File(%s) ==> %d" % (file, is_hdf5_file(file))
        self.assertEqual(is_hdf5_file(file), 1)

        # Then, delete the file
        os.remove(file)

    def test01_isHDF5File(self):
        """Checking is_hdf5_file function (FALSE case)"""

        # Create a regular (text) file
        file = tempfile.mktemp(".h5")
        fileh = open(file, "w")
        fileh.write("Hello!")
        fileh.close()

        version = is_hdf5_file(file)
        # When file is not an HDF5 format, always returns 0 or
        # negative value
        self.assertTrue(version <= 0)

        # Then, delete the file
        os.remove(file)

    def test01x_isHDF5File_nonexistent(self):
        """Identifying a nonexistent HDF5 file."""
        self.assertRaises(IOError, is_hdf5_file, 'nonexistent')

    def test01x_isHDF5File_unreadable(self):
        """Identifying an unreadable HDF5 file."""

        if hasattr(os, 'getuid') and os.getuid() != 0:
            h5fname = tempfile.mktemp(suffix='.h5')
            open_file(h5fname, 'w').close()
            try:
                os.chmod(h5fname, 0)  # no permissions at all
                self.assertRaises(IOError, is_hdf5_file, h5fname)
            finally:
                os.remove(h5fname)

    def test02_isPyTablesFile(self):
        """Checking is_pytables_file function (TRUE case)"""

        # Create a PyTables file
        file = tempfile.mktemp(".h5")
        fileh = open_file(file, mode="w")
        fileh.create_array(fileh.root, 'array', [
                           1, 2], title="Title example")

        # For this method to run, it needs a closed file
        fileh.close()

        version = is_pytables_file(file)
        # When file has a PyTables format, always returns "1.0" string or
        # greater
        if common.verbose:
            print
            print "\nPyTables format version number ==> %s" % \
                version
        self.assertTrue(version >= "1.0")

        # Then, delete the file
        os.remove(file)

    def test03_isPyTablesFile(self):
        """Checking is_pytables_file function (FALSE case)"""

        # Create a regular (text) file
        file = tempfile.mktemp(".h5")
        fileh = open(file, "w")
        fileh.write("Hello!")
        fileh.close()

        version = is_pytables_file(file)
        # When file is not a PyTables format, always returns 0 or
        # negative value
        if common.verbose:
            print
            print "\nPyTables format version number ==> %s" % \
                version
        self.assertTrue(version is None)

        # Then, delete the file
        os.remove(file)

    def test04_openGenericHDF5File(self):
        """Checking opening of a generic HDF5 file"""

        # Open an existing generic HDF5 file
        fileh = open_file(self._testFilename("ex-noattr.h5"), mode="r")

        # Check for some objects inside

        # A group
        columns = fileh.get_node("/columns", classname="Group")
        self.assertEqual(columns._v_name, "columns")

        # An Array
        array_ = fileh.get_node(columns, "TDC", classname="Array")
        self.assertEqual(array_._v_name, "TDC")

        # (The new LRU code defers the appearance of a warning to this point).

        # Here comes an Array of H5T_ARRAY type
        ui = fileh.get_node(columns, "pressure", classname="Array")
        self.assertEqual(ui._v_name, "pressure")
        if common.verbose:
            print "Array object with type H5T_ARRAY -->", repr(ui)
            print "Array contents -->", ui[:]

        # A Table
        table = fileh.get_node("/detector", "table", classname="Table")
        self.assertEqual(table._v_name, "table")

        fileh.close()

    def test04b_UnImplementedOnLoading(self):
        """Checking failure loading resulting in an ``UnImplemented`` node"""

        ############### Note for developers ###############################
        # This test fails if you have the line:                           #
        # ##return ChildClass(self, childname)  # uncomment for debugging #
        # uncommented in Group.py!                                        #
        ###################################################################

        h5file = open_file(self._testFilename('smpl_unsupptype.h5'))
        try:
            node = self.assertWarns(
                UserWarning, h5file.get_node, '/CompoundChunked')
            self.assertTrue(isinstance(node, UnImplemented))
        finally:
            h5file.close()

    def test04c_UnImplementedScalar(self):
        """Checking opening of HDF5 files containing scalar dataset of
        UnImlemented type"""

        h5file = open_file(self._testFilename("scalar.h5"))
        try:
            node = self.assertWarns(
                UserWarning, h5file.get_node, '/variable length string')
            self.assertTrue(isinstance(node, UnImplemented))
        finally:
            h5file.close()

    def test05_copyUnimplemented(self):
        """Checking that an UnImplemented object cannot be copied"""

        # Open an existing generic HDF5 file
        fileh = open_file(self._testFilename("smpl_unsupptype.h5"), mode="r")
        ui = self.assertWarns(
            UserWarning, fileh.get_node, '/CompoundChunked')
        self.assertEqual(ui._v_name, 'CompoundChunked')
        if common.verbose:
            print "UnImplement object -->", repr(ui)

        # Check that it cannot be copied to another file
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(file2, mode="w")
        # Force the userwarning to issue an error
        warnings.filterwarnings("error", category=UserWarning)
        try:
            ui.copy(fileh2.root, "newui")
        except UserWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched:"
                print value
        else:
            self.fail("expected an UserWarning")

        # Reset the warnings
        # Be careful with that, because this enables all the warnings
        # on the rest of the tests!
        # warnings.resetwarnings()
        # better use:
        warnings.filterwarnings("default", category=UserWarning)

        # Delete the new (empty) file
        fileh2.close()
        os.remove(file2)

        fileh.close()

    # The next can be used to check the copy of Array objects with H5T_ARRAY
    # in the future
    def _test05_copyUnimplemented(self):
        """Checking that an UnImplemented object cannot be copied"""

        # Open an existing generic HDF5 file
        # We don't need to wrap this in a try clause because
        # it has already been tried and the warning will not happen again
        fileh = open_file(self._testFilename("ex-noattr.h5"), mode="r")
        # An unsupported object (the deprecated H5T_ARRAY type in
        # Array, from pytables 0.8 on)
        ui = fileh.get_node(fileh.root.columns, "pressure")
        self.assertEqual(ui._v_name, "pressure")
        if common.verbose:
            print "UnImplement object -->", repr(ui)

        # Check that it cannot be copied to another file
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(file2, mode="w")
        # Force the userwarning to issue an error
        warnings.filterwarnings("error", category=UserWarning)
        try:
            ui.copy(fileh2.root, "newui")
        except UserWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched:"
                print value
        else:
            self.fail("expected an UserWarning")

        # Reset the warnings
        # Be careful with that, because this enables all the warnings
        # on the rest of the tests!
        # warnings.resetwarnings()
        # better use:
        warnings.filterwarnings("default", category=UserWarning)

        # Delete the new (empty) file
        fileh2.close()
        os.remove(file2)

        fileh.close()


class PythonAttrsTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test interactions of Python attributes and child nodes."""

    def test00_attrOverChild(self):
        """Setting a Python attribute over a child node."""

        root = self.h5file.root

        # Create ``/test`` and overshadow it with ``root.test``.
        child = self.h5file.create_array(root, 'test', [1])
        attr = 'foobar'
        self.assertWarns(NaturalNameWarning,
                         setattr, root, 'test', attr)

        self.assertTrue(root.test is attr)
        self.assertTrue(root._f_get_child('test') is child)

        # Now bring ``/test`` again to light.
        del root.test

        self.assertTrue(root.test is child)

        # Now there is no *attribute* named ``test``.
        self.assertRaises(AttributeError,
                          delattr, root, 'test')

    def test01_childUnderAttr(self):
        """Creating a child node under a Python attribute."""

        h5file = self.h5file
        root = h5file.root

        # Create ``root.test`` and an overshadowed ``/test``.
        attr = 'foobar'
        root.test = attr
        self.assertWarns(NaturalNameWarning,
                         h5file.create_array, root, 'test', [1])
        child = h5file.get_node('/test')

        self.assertTrue(root.test is attr)
        self.assertTrue(root._f_get_child('test') is child)

        # Now bring ``/test`` again to light.
        del root.test

        self.assertTrue(root.test is child)

        # Now there is no *attribute* named ``test``.
        self.assertRaises(AttributeError,
                          delattr, root, 'test')

    def test02_nodeAttrInLeaf(self):
        """Assigning a ``Node`` value as an attribute to a ``Leaf``."""

        h5file = self.h5file

        array1 = h5file.create_array('/', 'array1', [1])
        array2 = h5file.create_array('/', 'array2', [1])

        # This may make the garbage collector work a little.
        array1.array2 = array2
        array2.array1 = array1

        # Check the assignments.
        self.assertTrue(array1.array2 is array2)
        self.assertTrue(array2.array1 is array1)
        self.assertRaises(NoSuchNodeError,  # ``/array1`` is not a group
                          h5file.get_node, '/array1/array2')
        self.assertRaises(NoSuchNodeError,  # ``/array2`` is not a group
                          h5file.get_node, '/array2/array3')

    def test03_nodeAttrInGroup(self):
        """Assigning a ``Node`` value as an attribute to a ``Group``."""

        h5file = self.h5file
        root = h5file.root

        array = h5file.create_array('/', 'array', [1])

        # Assign the array to a pair of attributes,
        # one of them overshadowing the original.
        root.arrayAlias = array
        self.assertWarns(NaturalNameWarning,
                         setattr, root, 'array', array)

        # Check the assignments.
        self.assertTrue(root.arrayAlias is array)
        self.assertTrue(root.array is array)
        self.assertRaises(NoSuchNodeError, h5file.get_node, '/arrayAlias')
        self.assertTrue(h5file.get_node('/array') is array)

        # Remove the attribute overshadowing the child.
        del root.array
        # Now there is no *attribute* named ``array``.
        self.assertRaises(AttributeError,
                          delattr, root, 'array')


class StateTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """
    Test that ``File`` and ``Node`` operations check their state (open
    or closed, readable or writable) before proceeding.
    """

    def test00_fileCopyFileClosed(self):
        """Test copying a closed file."""

        h5cfname = tempfile.mktemp(suffix='.h5')
        self.h5file.close()

        try:
            self.assertRaises(ClosedFileError,
                              self.h5file.copy_file, h5cfname)
        finally:
            if os.path.exists(h5cfname):
                os.remove(h5fcname)
                self.fail("a (maybe incomplete) copy "
                          "of a closed file was created")

    def test01_fileCloseClosed(self):
        """Test closing an already closed file."""

        self.h5file.close()

        try:
            self.h5file.close()
        except ClosedFileError:
            self.fail("could not close an already closed file")

    def test02_fileFlushClosed(self):
        """Test flushing a closed file."""

        self.h5file.close()
        self.assertRaises(ClosedFileError, self.h5file.flush)

    def test03_fileFlushRO(self):
        """Flushing a read-only file."""

        self._reopen('r')

        try:
            self.h5file.flush()
        except FileModeError:
            self.fail("could not flush a read-only file")

    def test04_fileCreateNodeClosed(self):
        """Test creating a node in a closed file."""

        self.h5file.close()
        self.assertRaises(ClosedFileError,
                          self.h5file.create_group, '/', 'test')

    def test05_fileCreateNodeRO(self):
        """Test creating a node in a read-only file."""

        self._reopen('r')
        self.assertRaises(FileModeError,
                          self.h5file.create_group, '/', 'test')

    def test06_fileRemoveNodeClosed(self):
        """Test removing a node from a closed file."""

        self.h5file.create_group('/', 'test')
        self.h5file.close()
        self.assertRaises(ClosedFileError,
                          self.h5file.remove_node, '/', 'test')

    def test07_fileRemoveNodeRO(self):
        """Test removing a node from a read-only file."""

        self.h5file.create_group('/', 'test')
        self._reopen('r')
        self.assertRaises(FileModeError,
                          self.h5file.remove_node, '/', 'test')

    def test08_fileMoveNodeClosed(self):
        """Test moving a node in a closed file."""

        self.h5file.create_group('/', 'test1')
        self.h5file.create_group('/', 'test2')
        self.h5file.close()
        self.assertRaises(ClosedFileError,
                          self.h5file.move_node, '/test1', '/', 'test2')

    def test09_fileMoveNodeRO(self):
        """Test moving a node in a read-only file."""

        self.h5file.create_group('/', 'test1')
        self.h5file.create_group('/', 'test2')
        self._reopen('r')
        self.assertRaises(FileModeError,
                          self.h5file.move_node, '/test1', '/', 'test2')

    def test10_fileCopyNodeClosed(self):
        """Test copying a node in a closed file."""

        self.h5file.create_group('/', 'test1')
        self.h5file.create_group('/', 'test2')
        self.h5file.close()
        self.assertRaises(ClosedFileError,
                          self.h5file.copy_node, '/test1', '/', 'test2')

    def test11_fileCopyNodeRO(self):
        """Test copying a node in a read-only file."""

        self.h5file.create_group('/', 'test1')
        self._reopen('r')
        self.assertRaises(FileModeError,
                          self.h5file.copy_node, '/test1', '/', 'test2')

    def test13_fileGetNodeClosed(self):
        """Test getting a node from a closed file."""

        self.h5file.create_group('/', 'test')
        self.h5file.close()
        self.assertRaises(ClosedFileError, self.h5file.get_node, '/test')

    def test14_fileWalkNodesClosed(self):
        """Test walking a closed file."""

        self.h5file.create_group('/', 'test1')
        self.h5file.create_group('/', 'test2')
        self.h5file.close()
        self.assertRaises(ClosedFileError, self.h5file.walk_nodes().next)

    def test15_fileAttrClosed(self):
        """Test setting and deleting a node attribute in a closed file."""

        self.h5file.create_group('/', 'test')
        self.h5file.close()
        self.assertRaises(ClosedFileError,
                          self.h5file.set_node_attr, '/test', 'foo', 'bar')
        self.assertRaises(ClosedFileError,
                          self.h5file.del_node_attr, '/test', 'foo')

    def test16_fileAttrRO(self):
        """Test setting and deleting a node attribute in a read-only file."""

        self.h5file.create_group('/', 'test')
        self.h5file.set_node_attr('/test', 'foo', 'foo')
        self._reopen('r')
        self.assertRaises(FileModeError,
                          self.h5file.set_node_attr, '/test', 'foo', 'bar')
        self.assertRaises(FileModeError,
                          self.h5file.del_node_attr, '/test', 'foo')

    def test17_fileUndoClosed(self):
        """Test undo operations in a closed file."""

        self.h5file.enable_undo()
        self.h5file.create_group('/', 'test2')
        self.h5file.close()
        self.assertRaises(ClosedFileError, self.h5file.is_undo_enabled)
        self.assertRaises(ClosedFileError, self.h5file.get_current_mark)
        self.assertRaises(ClosedFileError, self.h5file.undo)
        self.assertRaises(ClosedFileError, self.h5file.disable_undo)

    def test18_fileUndoRO(self):
        """Test undo operations in a read-only file."""

        self.h5file.enable_undo()
        self.h5file.create_group('/', 'test')
        self._reopen('r')
        self.assertEqual(self.h5file._undoEnabled, False)
        # self.assertRaises(FileModeError, self.h5file.undo)
        # self.assertRaises(FileModeError, self.h5file.disable_undo)

    def test19a_getNode(self):
        """Test getting a child of a closed node."""

        g1 = self.h5file.create_group('/', 'g1')
        g2 = self.h5file.create_group('/g1', 'g2')

        # Close this *object* so that it should not be used.
        g1._f_close()
        self.assertRaises(ClosedNodeError, g1._f_get_child, 'g2')

        # Getting a node by its closed object is not allowed.
        self.assertRaises(ClosedNodeError,
                          self.h5file.get_node, g1)

        # Going through that *node* should reopen it automatically.
        try:
            g2_ = self.h5file.get_node('/g1/g2')
        except ClosedNodeError:
            self.fail("closed parent group has not been reopened")

        # Already open nodes should be closed now, but not the new ones.
        self.assertTrue(g2._v_isopen is False,
                        "open child of closed group has not been closed")
        self.assertTrue(g2_._v_isopen is True,
                        "open child of closed group has not been closed")

        # And existing closed ones should remain closed, but not the new ones.
        g1_ = self.h5file.get_node('/g1')
        self.assertTrue(g1._v_isopen is False,
                        "already closed group is not closed anymore")
        self.assertTrue(g1_._v_isopen is True,
                        "newly opened group is still closed")

    def test19b_getNode(self):
        """Test getting a node that does not start with a slash ('/')."""

        # Create an array in the root
        self.h5file.create_array('/', 'array', [1, 2], title="Title example")

        # Get the array without specifying a leading slash
        self.assertRaises(NameError, self.h5file.get_node, "array")

    def test20_removeNode(self):
        """Test removing a closed node."""

        # This test is a little redundant once we know that ``File.get_node()``
        # will reload a closed node, but anyway...

        group = self.h5file.create_group('/', 'group')
        array = self.h5file.create_array('/group', 'array', [1])

        # The closed *object* can not be used.
        group._f_close()
        self.assertRaises(ClosedNodeError, group._f_remove)
        self.assertRaises(ClosedNodeError, self.h5file.remove_node, group)

        # Still, the *node* is reloaded when necessary.
        try:
            self.h5file.remove_node('/group', recursive=True)
        except ClosedNodeError:
            self.fail("closed node has not been reloaded")

        # Objects of descendent removed nodes
        # should have been automatically closed when removed.
        self.assertRaises(ClosedNodeError, array._f_remove)

        self.assertTrue('/group/array' not in self.h5file)  # just in case
        self.assertTrue('/group' not in self.h5file)  # just in case

    def test21_attrsOfNode(self):
        """Test manipulating the attributes of a closed node."""

        node = self.h5file.create_group('/', 'test')
        nodeAttrs = node._v_attrs

        nodeAttrs.test = attr = 'foo'

        node._f_close()
        self.assertRaises(ClosedNodeError, getattr, node, '_v_attrs')
        # The design of ``AttributeSet`` does not yet allow this test.
        ## self.assertRaises(ClosedNodeError, getattr, nodeAttrs, 'test')

        self.assertEqual(self.h5file.get_node_attr('/test', 'test'), attr)

    def test21b_attrsOfNode(self):
        """Test manipulating the attributes of a node in a read-only file."""

        self.h5file.create_group('/', 'test')
        self.h5file.set_node_attr('/test', 'test', 'foo')

        self._reopen('r')
        self.assertRaises(FileModeError,
                          self.h5file.set_node_attr, '/test', 'test', 'bar')

    def test22_fileClosesNode(self):
        """Test node closing because of file closing."""

        node = self.h5file.create_group('/', 'test')

        self.h5file.close()
        self.assertRaises(ClosedNodeError, getattr, node, '_v_attrs')

    def test23_reopenFile(self):
        """Testing reopening a file and closing it several times."""

        self.h5file.create_array('/', 'test', [1, 2, 3])
        self.h5file.close()

        file1 = open_file(self.h5fname, "r")
        self.assertEqual(file1.open_count, 1)
        file2 = open_file(self.h5fname, "r")
        self.assertEqual(file1.open_count, 2)
        self.assertEqual(file2.open_count, 2)
        if common.verbose:
            print "(file1) open_count:", file1.open_count
            print "(file1) test[1]:", file1.root.test[1]
        self.assertEqual(file1.root.test[1], 2)
        file1.close()
        self.assertEqual(file2.open_count, 1)
        if common.verbose:
            print "(file2) open_count:", file2.open_count
            print "(file2) test[1]:", file2.root.test[1]
        self.assertEqual(file2.root.test[1], 2)
        file2.close()


class FlavorTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """
    Test that setting, getting and changing the ``flavor`` attribute
    of a leaf works as expected.
    """

    array_data = numpy.arange(10)
    scalar_data = numpy.int32(10)

    def _reopen(self, mode='r'):
        super(FlavorTestCase, self)._reopen(mode)
        self.array = self.h5file.get_node('/array')
        self.scalar = self.h5file.get_node('/scalar')
        return True

    def setUp(self):
        super(FlavorTestCase, self).setUp()
        self.array = self.h5file.create_array('/', 'array', self.array_data)
        self.scalar = self.h5file.create_array('/', 'scalar', self.scalar_data)

    def tearDown(self):
        self.array = None
        super(FlavorTestCase, self).tearDown()

    def test00_invalid(self):
        """Setting an invalid flavor."""
        self.assertRaises(FlavorError, setattr, self.array, 'flavor', 'foo')

    def test01_readonly(self):
        """Setting a flavor in a read-only file."""
        self._reopen(mode='r')
        self.assertRaises(FileModeError,
                          setattr, self.array, 'flavor',
                          tables.flavor.internal_flavor)

    def test02_change(self):
        """Changing the flavor and reading data."""
        for flavor in all_flavors:
            self.array.flavor = flavor
            self.assertEqual(self.array.flavor, flavor)
            idata = array_of_flavor(self.array_data, flavor)
            odata = self.array[:]
            self.assertTrue(common.allequal(odata, idata, flavor))

    def test03_store(self):
        """Storing a changed flavor."""
        for flavor in all_flavors:
            self.array.flavor = flavor
            self.assertEqual(self.array.flavor, flavor)
            self._reopen(mode='r+')
            self.assertEqual(self.array.flavor, flavor)

    def test04_missing(self):
        """Reading a dataset of a missing flavor."""
        flavor = self.array.flavor  # default is internal
        self.array._v_attrs.FLAVOR = 'foobar'  # breaks flavor
        self._reopen(mode='r')
        idata = array_of_flavor(self.array_data, flavor)
        odata = self.assertWarns(FlavorWarning, self.array.read)
        self.assertTrue(common.allequal(odata, idata, flavor))

    def test05_delete(self):
        """Deleting the flavor of a dataset."""
        self.array.flavor = 'python'  # non-default
        self.assertEqual(self.array.flavor, 'python')
        self.assertEqual(self.array.attrs.FLAVOR, 'python')
        del self.array.flavor
        self.assertEqual(self.array.flavor, tables.flavor.internal_flavor)
        self.assertRaises(AttributeError, getattr, self.array.attrs, 'FLAVOR')

    def test06_copyDeleted(self):
        """Copying a node with a deleted flavor (see #100)."""
        snames = [node._v_name for node in [self.array, self.scalar]]
        dnames = ['%s_copy' % name for name in snames]
        for name in snames:
            node = self.h5file.get_node('/', name)
            del node.flavor
        # Check the copied flavors right after copying and after reopening.
        for fmode in ['r+', 'r']:
            self._reopen(fmode)
            for sname, dname in zip(snames, dnames):
                if fmode == 'r+':
                    snode = self.h5file.get_node('/', sname)
                    node = snode.copy('/', dname)
                elif fmode == 'r':
                    node = self.h5file.get_node('/', dname)
                self.assertEqual(node.flavor, tables.flavor.internal_flavor,
                                 "flavor of node ``%s`` is not internal: %r"
                                 % (node._v_pathname, node.flavor))

    def test07_restrict_flavors(self):
        # regression test for gh-163

        all_flavors = list(tables.flavor.all_flavors)
        alias_map = tables.flavor.alias_map.copy()
        converter_map = tables.flavor.converter_map.copy()
        identifier_map = tables.flavor.identifier_map.copy()
        description_map = tables.flavor.description_map.copy()

        try:
            tables.flavor.restrict_flavors(keep=[])
            self.assertTrue(len(tables.flavor.alias_map) < len(alias_map))
            self.assertTrue(len(
                tables.flavor.converter_map) < len(converter_map))
        finally:
            tables.flavor.all_flavors[:] = all_flavors[:]
            tables.flavor.alias_map.update(alias_map)
            tables.flavor.converter_map.update(converter_map)
            tables.flavor.identifier_map.update(identifier_map)
            tables.flavor.description_map.update(description_map)


class UnicodeFilename(common.PyTablesTestCase):
    unicode_prefix = u'para\u0140lel'

    def setUp(self):
        self.h5fname = tempfile.mktemp(prefix=self.unicode_prefix,
                                       suffix=".h5")
        self.h5file = tables.open_file(self.h5fname, "w")
        self.test = self.h5file.create_array('/', 'test', [1, 2])
        # So as to check the reading
        self.h5file.close()
        self.h5file = tables.open_file(self.h5fname, "r")

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.h5fname)

    def test01(self):
        """Checking creating a filename with Unicode chars."""

        test = self.h5file.root.test
        if common.verbose:
            print "Filename:", self.h5fname
            print "Array:", test[:]
            print "Should look like:", [1, 2]
        self.assertEqual(test[:], [1, 2], "Values does not match.")

    def test02(self):
        """Checking is_hdf5_file with a Unicode filename."""

        self.h5file.close()
        if common.verbose:
            print "Filename:", self.h5fname
            print "is_hdf5_file?:", tables.is_hdf5_file(self.h5fname)
        self.assertTrue(tables.is_hdf5_file(self.h5fname))

    def test03(self):
        """Checking is_pytables_file with a Unicode filename."""

        self.h5file.close()
        if common.verbose:
            print "Filename:", self.h5fname
            print "is_pytables_file?:", tables.is_pytables_file(self.h5fname)
        self.assertNotEqual(tables.is_pytables_file(self.h5fname), False)


class FilePropertyTestCase(common.PyTablesTestCase):
    def setUp(self):
        self.h5fname = tempfile.mktemp(".h5")
        self.h5file = None

    def tearDown(self):
        if self.h5file:
            self.h5file.close()

        if os.path.exists(self.h5fname):
            os.remove(self.h5fname)

    def test_get_filesize(self):
        data = numpy.zeros((2000, 2000))
        datasize = numpy.prod(data.shape) * data.dtype.itemsize

        self.h5file = open_file(self.h5fname, mode="w")
        self.h5file.create_array(self.h5file.root, 'array', data)
        h5_filesize = self.h5file.get_filesize()
        self.h5file.close()

        fs_filesize = os.stat(self.h5fname)[6]

        self.assertTrue(h5_filesize >= datasize)
        self.assertEqual(h5_filesize, fs_filesize)

    def test01_null_userblock_size(self):
        self.h5file = open_file(self.h5fname, mode="w")
        self.h5file.create_array(self.h5file.root, 'array', [1, 2])
        self.assertEqual(self.h5file.get_userblock_size(), 0)

    def test02_null_userblock_size(self):
        self.h5file = open_file(self.h5fname, mode="w")
        self.h5file.create_array(self.h5file.root, 'array', [1, 2])
        self.h5file.close()
        self.h5file = open_file(self.h5fname, mode="r")
        self.assertEqual(self.h5file.get_userblock_size(), 0)

    def test03_null_userblock_size(self):
        USER_BLOCK_SIZE = 0
        self.h5file = open_file(self.h5fname, mode="w",
                                user_block_size=USER_BLOCK_SIZE)
        self.h5file.create_array(self.h5file.root, 'array', [1, 2])
        self.assertEqual(self.h5file.get_userblock_size(), 0)

    def test01_userblock_size(self):
        USER_BLOCK_SIZE = 512
        self.h5file = open_file(self.h5fname, mode="w",
                                user_block_size=USER_BLOCK_SIZE)
        self.h5file.create_array(self.h5file.root, 'array', [1, 2])
        self.assertEqual(self.h5file.get_userblock_size(), USER_BLOCK_SIZE)

    def test02_userblock_size(self):
        USER_BLOCK_SIZE = 512
        self.h5file = open_file(self.h5fname, mode="w",
                                user_block_size=USER_BLOCK_SIZE)
        self.h5file.create_array(self.h5file.root, 'array', [1, 2])
        self.h5file.close()
        self.h5file = open_file(self.h5fname, mode="r")
        self.assertEqual(self.h5file.get_userblock_size(), USER_BLOCK_SIZE)

    def test_small_userblock_size(self):
        USER_BLOCK_SIZE = 12
        self.assertRaises(ValueError, open_file, self.h5fname, mode="w",
                          user_block_size=USER_BLOCK_SIZE)

    def test_invalid_userblock_size(self):
        USER_BLOCK_SIZE = 1025
        self.assertRaises(ValueError, open_file, self.h5fname, mode="w",
                          user_block_size=USER_BLOCK_SIZE)


# Test for reading a file that uses Blosc and created on a big-endian platform
class BloscBigEndian(common.PyTablesTestCase):

    def setUp(self):
        filename = self._testFilename("blosc_bigendian.h5")
        self.fileh = open_file(filename, "r")

    def tearDown(self):
        self.fileh.close()

    def test00_bigendian(self):
        """Checking compatibility with Blosc on big-endian machines."""

        # Check that we can read the contents without problems (nor warnings!)
        for dset_name in ('i1', 'i2', 'i4', 'i8'):
            a = numpy.arange(10, dtype=dset_name)
            dset = self.fileh.get_node('/'+dset_name)
            self.assertTrue(common.allequal(a, dset[:]),
                            "Error in big-endian data!")


# Case test for Blosc and subprocesses (via multiprocessing module)

# The worker function for the subprocess (needs to be here because Windows
# has problems pickling nested functions with the multiprocess module :-/)
def _worker(fn, qout=None):
    fp = tables.open_file(fn)
    if common.verbose:
        print "About to load: ", fn
    rows = fp.root.table.where('(f0 < 10)')
    if common.verbose:
        print "Got the iterator, about to iterate"
    next(rows)
    if common.verbose:
        print "Succeeded in one iteration\n"
    fp.close()

    if qout is not None:
        qout.put("Done")


class BloscSubprocess(common.PyTablesTestCase):
    def test_multiprocess(self):
        # From: Yaroslav Halchenko <debian@onerussian.com>
        # Subject: Skip the unittest on kFreeBSD and Hurd -- locking seems to
        #         be N/A
        #
        #  on kfreebsd /dev/shm is N/A
        #  on Hurd -- inter-process semaphore locking is N/A
        import platform

        if platform.system().lower() in ('gnu', 'gnu/kfreebsd'):
            raise common.SkipTest("multiprocessing module is not supported "
                                  "on Hurd/kFreeBSD")

        # Create a relatively large table with Blosc level 9 (large blocks)
        fn = tempfile.mktemp(prefix="multiproc-blosc9-", suffix=".h5")
        size = int(3e5)
        sa = numpy.fromiter(((i, i**2, i//3)
                             for i in xrange(size)), 'i4,i8,f8')
        fp = open_file(fn, 'w')
        fp.create_table(fp.root, 'table', sa,
                        filters=Filters(complevel=9, complib="blosc"),
                        chunkshape=(size // 3,))
        fp.close()

        if common.verbose:
            print "**** Running from main process:"
        _worker(fn)

        if common.verbose:
            print "**** Running from subprocess:"

        try:
            qout = mp.Queue()
        except OSError:
            print "Permission denied due to /dev/shm settings"
        else:
            ps = mp.Process(target=_worker, args=(fn, qout,))
            ps.daemon = True
            ps.start()

            result = qout.get()
            if common.verbose:
                print result

        os.remove(fn)


class HDF5ErrorHandling(common.PyTablesTestCase):

    def setUp(self):
        self._old_policy = tables.HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY

    def tearDown(self):
        tables.HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY = self._old_policy

    def test_silence_messages(self):
        code = """
import tables
tables.silence_hdf5_messages(False)
tables.silence_hdf5_messages()
try:
    tables.open_file(r'%s')
except tables.HDF5ExtError, e:
    pass
"""

        fn = tempfile.mktemp(prefix="hdf5-error-handling-", suffix=".py")
        fp = open(fn, 'w')
        try:
            fp.write(code % fn)
            fp.close()

            p = subprocess.Popen([sys.executable, fn],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            (stdout, stderr) = p.communicate()

            self.assertFalse("HDF5-DIAG" in stderr.decode('ascii'))
        finally:
            os.remove(fn)

    def test_enable_messages(self):
        code = """
import tables
tables.silence_hdf5_messages()
tables.silence_hdf5_messages(False)
try:
    tables.open_file(r'%s')
except tables.HDF5ExtError as e:
    pass
"""

        fn = tempfile.mktemp(prefix="hdf5-error-handling-", suffix=".py")
        fp = open(fn, 'w')
        try:
            fp.write(code % fn)
            fp.close()

            p = subprocess.Popen([sys.executable, fn],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            (stdout, stderr) = p.communicate()

            self.assertTrue("HDF5-DIAG" in stderr.decode('ascii'))
        finally:
            os.remove(fn)

    def _raise_exterror(self):
        filename = tempfile.mktemp(".h5")
        open(filename, 'wb').close()

        try:
            f = tables.open_file(filename)
            f.close()
        finally:
            os.remove(filename)

    def test_h5_backtrace_quiet(self):
        tables.HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY = True

        try:
            self._raise_exterror()
        except tables.HDF5ExtError, e:
            self.assertFalse(e.h5backtrace is None)
        else:
            self.fail("HDF5ExtError exception not raised")

    def test_h5_backtrace_verbose(self):
        tables.HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY = "VERBOSE"

        try:
            self._raise_exterror()
        except tables.HDF5ExtError, e:
            self.assertFalse(e.h5backtrace is None)
            msg = str(e)
            self.assertTrue(e.h5backtrace[-1][-1] in msg)
        else:
            self.fail("HDF5ExtError exception not raised")

    def test_h5_backtrace_ignore(self):
        tables.HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY = False

        try:
            self._raise_exterror()
        except tables.HDF5ExtError, e:
            self.assertTrue(e.h5backtrace is None)
        else:
            self.fail("HDF5ExtError exception not raised")


class TestDescription(common.PyTablesTestCase):
    def test_isdescription_inheritance(self):
        # Regression test for gh-65
        class TestDescParent(IsDescription):
            c = Int32Col()

        class TestDesc(TestDescParent):
            pass

        self.assertTrue('c' in TestDesc.columns)

    def test_descr_from_dtype(self):
        t = numpy.dtype([('col1', 'int16'), ('col2', float)])
        descr, byteorder = descr_from_dtype(t)

        self.assertTrue('col1' in descr._v_colobjects)
        self.assertTrue('col2' in descr._v_colobjects)
        self.assertEqual(len(descr._v_colobjects), 2)
        self.assertTrue(isinstance(descr._v_colobjects['col1'], Col))
        self.assertTrue(isinstance(descr._v_colobjects['col2'], Col))
        self.assertEqual(descr._v_colobjects['col1'].dtype, numpy.int16)
        self.assertEqual(descr._v_colobjects['col2'].dtype, float)

    def test_descr_from_dtype_rich_dtype(self):
        header = [(('timestamp', 't'), 'u4'),
                  (('unit (cluster) id', 'unit'), 'u2')]
        t = numpy.dtype(header)

        descr, byteorder = descr_from_dtype(t)
        self.assertEqual(len(descr._v_names), 2)
        self.assertEqual(sorted(descr._v_names), ['t', 'unit'])

    def test_dtype_from_descr_is_description(self):
        # See gh-152
        class TestDescParent(IsDescription):
            col1 = Int16Col()
            col2 = FloatCol()

        dtype = numpy.dtype([('col1', 'int16'), ('col2', float)])
        t = dtype_from_descr(TestDescParent)

        self.assertEqual(t, dtype)

    def test_dtype_from_descr_is_description_instance(self):
        # See gh-152
        class TestDescParent(IsDescription):
            col1 = Int16Col()
            col2 = FloatCol()

        dtype = numpy.dtype([('col1', 'int16'), ('col2', float)])
        t = dtype_from_descr(TestDescParent())

        self.assertEqual(t, dtype)

    def test_dtype_from_descr_description_instance(self):
        # See gh-152
        class TestDescParent(IsDescription):
            col1 = Int16Col()
            col2 = FloatCol()

        dtype = numpy.dtype([('col1', 'int16'), ('col2', float)])
        desctiption = Description(TestDescParent().columns)
        t = dtype_from_descr(desctiption)

        self.assertEqual(t, dtype)

    def test_dtype_from_descr_dict(self):
        # See gh-152
        dtype = numpy.dtype([('col1', 'int16'), ('col2', float)])
        t = dtype_from_descr({'col1': Int16Col(), 'col2': FloatCol()})

        self.assertEqual(t, dtype)

    def test_dtype_from_descr_invalid_type(self):
        # See gh-152
        self.assertRaises(ValueError, dtype_from_descr, [])

    def test_dtype_from_descr_byteorder(self):
        # See gh-152
        class TestDescParent(IsDescription):
            col1 = Int16Col()
            col2 = FloatCol()

        t = dtype_from_descr(TestDescParent, byteorder='>')

        self.assertEqual(t['col1'].byteorder, '>')
        self.assertEqual(t['col2'].byteorder, '>')

    def test_str_names(self):
        # see gh-42
        d = {'name': tables.Int16Col()}
        descr = Description(d)
        self.assertEqual(sorted(descr._v_names), sorted(d.keys()))
        self.assertTrue(isinstance(descr._v_dtype, numpy.dtype))
        self.assertTrue(sorted(descr._v_dtype.fields.keys()),
                        sorted(d.keys()))

    if sys.version_info[0] < 3:
        def test_unicode_names(self):
            # see gh-42
            # the name used is a valid ASCII identifier passed as unicode
            # string
            d = {unicode('name'): tables.Int16Col()}
            descr = Description(d)
            self.assertEqual(sorted(descr._v_names), sorted(d.keys()))
            self.assertTrue(isinstance(descr._v_dtype, numpy.dtype))
            keys = []
            for key in d.keys():
                if isinstance(key, unicode):
                    keys.append(key.encode())
                else:
                    keys.append(key)
            self.assertTrue(sorted(descr._v_dtype.fields.keys()), sorted(keys))


class TestAtom(common.PyTablesTestCase):
    def test_atom_attributes01(self):
        shape = (10, 10)
        a = Float64Atom(shape=shape)

        self.assertEqual(a.dflt, 0.)
        self.assertEqual(a.dtype, numpy.dtype((numpy.float64, shape)))
        self.assertEqual(a.itemsize, a.dtype.base.itemsize)
        self.assertEqual(a.kind, 'float')
        self.assertEqual(a.ndim, len(shape))
        # self.assertEqual(a.recarrtype, )
        self.assertEqual(a.shape, shape)
        self.assertEqual(a.size, a.itemsize * numpy.prod(shape))
        self.assertEqual(a.type, 'float64')

    def test_atom_copy01(self):
        shape = (10, 10)
        a = Float64Atom(shape=shape)
        aa = a.copy()
        self.assertEqual(aa.shape, shape)

    def test_atom_copy02(self):
        dflt = 2.0
        a = Float64Atom(dflt=dflt)
        aa = a.copy()
        self.assertEqual(aa.dflt, dflt)

    def test_atom_copy_override(self):
        shape = (10, 10)
        dflt = 2.0
        a = Float64Atom(shape=shape, dflt=dflt)
        aa = a.copy(dflt=-dflt)
        self.assertEqual(aa.shape, shape)
        self.assertNotEqual(aa.dflt, dflt)
        self.assertEqual(aa.dflt, -dflt)


class TestCol(common.PyTablesTestCase):
    def test_col_copy01(self):
        shape = (10, 10)
        c = Float64Col(shape=shape)
        cc = c.copy()
        self.assertEqual(cc.shape, shape)

    def test_col_copy02(self):
        dflt = 2.0
        c = Float64Col(dflt=dflt)
        cc = c.copy()
        self.assertEqual(cc.dflt, dflt)

    def test_col_copy_override(self):
        shape = (10, 10)
        dflt = 2.0
        pos = 3
        c = Float64Col(shape=shape, dflt=dflt, pos=pos)
        cc = c.copy(pos=2)
        self.assertEqual(cc.shape, shape)
        self.assertEqual(cc.dflt, dflt)
        self.assertNotEqual(cc._v_pos, pos)
        self.assertEqual(cc._v_pos, 2)


class TestSysattrCompatibility(common.PyTablesTestCase):

    def test_open_python2(self):
        filename = self._testFilename("python2.h5")
        fileh = open_file(filename, "r")
        self.assertTrue(fileh.isopen)
        fileh.close()

    def test_open_python3(self):
        filename = self._testFilename("python2.h5")
        fileh = open_file(filename, "r")
        self.assertTrue(fileh.isopen)
        fileh.close()


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    blosc_avail = which_lib_version("blosc") is not None

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(NodeCacheOpenFile))
        theSuite.addTest(unittest.makeSuite(NoNodeCacheOpenFile))
        theSuite.addTest(unittest.makeSuite(DictNodeCacheOpenFile))
        theSuite.addTest(unittest.makeSuite(CheckFileTestCase))
        theSuite.addTest(unittest.makeSuite(PythonAttrsTestCase))
        theSuite.addTest(unittest.makeSuite(StateTestCase))
        theSuite.addTest(unittest.makeSuite(FlavorTestCase))
        theSuite.addTest(unittest.makeSuite(FilePropertyTestCase))
        if blosc_avail:
            theSuite.addTest(unittest.makeSuite(BloscBigEndian))
        if multiprocessing_imported:
            theSuite.addTest(unittest.makeSuite(BloscSubprocess))
        theSuite.addTest(unittest.makeSuite(HDF5ErrorHandling))
        theSuite.addTest(unittest.makeSuite(TestDescription))
        theSuite.addTest(unittest.makeSuite(TestAtom))
        theSuite.addTest(unittest.makeSuite(TestCol))
        theSuite.addTest(unittest.makeSuite(TestSysattrCompatibility))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

## Local Variables:
## mode: python
## End:
