import sys
import unittest
import os
import tempfile
import warnings

from tables import *

from test_all import verbose, cleanup
# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup


class OpenFileTestCase(unittest.TestCase):

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, mode = "w", title="File title")
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

	# Create another array object
	array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
	# Create a group object
	group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        group._v_attrs.testattr = 42
	# Create a couple of objects there
	array1 = fileh.createArray(group, 'anarray1',
                                   [1,2,3,4,5,6,7], "Array title 1")
        array1.attrs.testattr = 42
	array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
	# Create a lonely group in first level
	group2 = fileh.createGroup(root, 'agroup2',
                                  "Group title 2")
        # Create a new group in the second level
	group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")

        # Create an array in the root with the same name as one in 'agroup'
        fileh.createArray(root, 'anarray1', [1,2],
                          title = "Title example")

        fileh.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)
        cleanup(self)

    def test00_newFile(self):
        """Checking creation of a new file"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_newFile..." % self.__class__.__name__

        # Create an HDF5 file
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        arr = fileh.createArray(fileh.root, 'array', [1,2],
                                title = "Title example")
        # Get the CLASS attribute of the arr object
        class_ = fileh.root.array.attrs.CLASS

        # Close and delete the file
        fileh.close()
        os.remove(file)

        assert class_.capitalize() == "Array"
        
    def test01_openFile(self):
        """Checking opening of an existing file"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_openFile..." % self.__class__.__name__

        # Open the old HDF5 file
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root.array.getAttr("TITLE")

        assert title == "Title example"
	fileh.close()
    
    def test01b_trMap(self):
        """Checking the translation table capability for reading"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b_trMap..." % self.__class__.__name__

        # Open the old HDF5 file
        trMap = {"pythonarray": "array"}
        fileh = openFile(self.file, mode = "r", trMap=trMap)
        # Get the array objects in the file
        array_ = fileh.getNode("/pythonarray")

        assert array_.name == "pythonarray"
        assert array_._v_hdf5name == "array"

        # This should throw an LookupError exception
        try:
            # Try to get the 'array' object in the old existing file
            array_ = fileh.getNode("/array")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
    
	fileh.close()
	
    def test01c_trMap(self):
        """Checking the translation table capability for writing"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_trMap..." % self.__class__.__name__

        # Create an HDF5 file
        file = tempfile.mktemp(".h5")
        trMap = {"pythonarray": "array"}
        fileh = openFile(file, mode = "w", trMap=trMap)
        arr = fileh.createArray(fileh.root, 'pythonarray', [1,2],
                                title = "Title example")

        # Get the array objects in the file
        array_ = fileh.getNode("/pythonarray")
        assert array_.name == "pythonarray"
        assert array_._v_hdf5name == "array"

        fileh.close()
        
        # Open the old HDF5 file (without the trMap parameter)
        fileh = openFile(self.file, mode = "r")
        # Get the array objects in the file
        array_ = fileh.getNode("/array")

        assert array_.name == "array"
        assert array_._v_hdf5name == "array"

        # This should throw an LookupError exception
        try:
            # Try to get the 'pythonarray' object in the old existing file
            array_ = fileh.getNode("/pythonarray")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
            
        fileh.close()
        # Remove the temporary file
        os.remove(file)
        

    def test02_appendFile(self):
        """Checking appending objects to an existing file"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_appendFile..." % self.__class__.__name__

        # Append a new array to the existing file
        fileh = openFile(self.file, mode = "r+")
        fileh.createArray(fileh.root, 'array2', [3,4],
                          title = "Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root.array2.getAttr("TITLE")

        assert title == "Title example 2"
        fileh.close()
        
    def test02b_appendFile2(self):
        """Checking appending objects to an existing file ("a" version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_appendFile2..." % self.__class__.__name__

        # Append a new array to the existing file
        fileh = openFile(self.file, mode = "a")
        fileh.createArray(fileh.root, 'array2', [3,4],
                          title = "Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root.array2.getAttr("TITLE")

        assert title == "Title example 2"
        fileh.close()
        
    # Begin to raise errors...
        
    def test03_appendErrorFile(self):
        """Checking appending objects to an existing file in "w" mode"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_appendErrorFile..." % self.__class__.__name__

        # Append a new array to the existing file but in write mode
        # so, the existing file should be deleted!
        fileh = openFile(self.file, mode = "w")
        fileh.createArray(fileh.root, 'array2', [3,4],
                          title = "Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")

        try:
            # Try to get the 'array' object in the old existing file
            arr = fileh.root.array
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()
        
    def test04a_openErrorFile(self):
        """Checking opening a non-existing file for reading"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04a_openErrorFile..." % self.__class__.__name__

        try:
            fileh = openFile("nonexistent.h5", mode = "r")
        except IOError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IOError was catched!"
                print value
        else:
            self.fail("expected an IOError")

    def test04b_alternateRootFile(self):
        """Checking alternate root access to the object tree"""
        
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04b_alternateRootFile..." % self.__class__.__name__

        # Open the existent HDF5 file
        fileh = openFile(self.file, mode = "r", rootUEP="/agroup")
        # Get the CLASS attribute of the arr object
        if verbose:
            print "\nFile tree dump:", fileh
        title = fileh.root.anarray1.getAttr("TITLE")

        assert title == "Array title 1"
	fileh.close()

    # This test works well, but HDF5 emits a series of messages that
    # may loose the user. It is better to deactivate it.
    def notest04c_alternateRootFile(self):
        """Checking non-existent alternate root access to the object tree"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04c_alternateRootFile..." % self.__class__.__name__

        try:
            fileh = openFile(self.file, mode = "r", rootUEP="/nonexistent")
        except RuntimeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected an IOError")

    def test05a_removeGroupRecursively(self):
        """Checking removing a group recursively"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05a_removeGroupRecursively..." % self.__class__.__name__

        # Delete a group with leafs
        fileh = openFile(self.file, mode = "r+")

        try:
            fileh.removeNode(fileh.root.agroup)
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected a NodeError")

        # This should work now
        fileh.removeNode(fileh.root, 'agroup', recursive=1)

        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Try to get the removed object
        try:
            object = fileh.root.agroup
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child of the removed object
        try:
            object = fileh.getNode("/agroup/agroup3")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test05b_removeGroupRecursively(self):
        """Checking removing a group recursively and access to it immediately"""
        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_removeGroupRecursively..." % self.__class__.__name__

        # Delete a group with leafs
        fileh = openFile(self.file, mode = "r+")

        try:
            fileh.removeNode(fileh.root, 'agroup')
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected a NodeError")

        # This should work now
        fileh.removeNode(fileh.root, 'agroup', recursive=1)

        # Try to get the removed object
        try:
            object = fileh.root.agroup
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child of the removed object
        try:
            object = fileh.getNode("/agroup/agroup3")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test05c_removeGroupRecursively(self):
        """Checking removing a group recursively (__delattr__ version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05c_removeGroupRecursively..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        try:
            del fileh.root.agroup
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected a NodeError")

        fileh.close()

    def test06a_removeGroup(self):
        """Checking removing a lonely group from an existing file"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06a_removeGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.removeNode(fileh.root, 'agroup2')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Try to get the removed object
        try:
            object = fileh.root.agroup2
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06a2_removeGroup(self):
        """Checking removing a lonely group from an existing file (__delattr__ version)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06a2_removeGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        del fileh.root.agroup2
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Try to get the removed object
        try:
            object = fileh.root.agroup2
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06b_removeLeaf(self):
        """Checking removing Leaves from an existing file"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06b_removeLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.removeNode(fileh.root, 'anarray')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Try to get the removed object
        try:
            object = fileh.root.anarray
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06c_removeLeaf(self):
        """Checking removing Leaves and access it immediately"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06c_removeLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.removeNode(fileh.root, 'anarray')
        
        # Try to get the removed object
        try:
            object = fileh.root.anarray
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test06d_removeLeaf(self):
        """Checking removing a non-existent node"""


        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test06d_removeLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        
        # Try to get the removed object
        try:
            fileh.removeNode(fileh.root, 'nonexistent')
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test07_renameLeaf(self):
        """Checking renaming a leave and access it after a close/open"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_renameLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.anarray, 'anarray2')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Ensure that the new name exists
        array_ = fileh.root.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/anarray2"
        assert array_._v_depth == 1
        # Try to get the previous object with the old name
        try:
            object = fileh.root.anarray
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test07b_renameLeaf(self):
        """Checking renaming Leaves and accesing them immediately"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07b_renameLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.anarray, 'anarray2')

        # Ensure that the new name exists
        array_ = fileh.root.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/anarray2"
        assert array_._v_depth == 1
        # Try to get the previous object with the old name
        try:
            object = fileh.root.anarray
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test07c_renameLeaf(self):
        """Checking renaming Leaves and modify attributes after that"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test07c_renameLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.anarray, 'anarray2')
        fileh.root.anarray2.attrs.TITLE = "hello"
        # Ensure that the new attribute has been written correctly
        array_ = fileh.root.anarray2
        assert array_.title == "hello"
        assert array_.attrs.TITLE == "hello"
        fileh.close()

    def test08_renameToExistingLeaf(self):
        """Checking renaming a node to an existing name"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_renameToExistingLeaf..." % self.__class__.__name__

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        # Try to get the previous object with the old name
        try:
            fileh.renameNode(fileh.root.anarray, 'array')
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test08b_renameToNotValidNaturalName(self):
        """Checking renaming a node to a non-valid natural name"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test08b_renameToNotValidNaturalName..." % self.__class__.__name__

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        warnings.filterwarnings("error", category=NaturalNameWarning)
        # Try to get the previous object with the old name
        try:
            fileh.renameNode(fileh.root.anarray, 'array 2')        
        except NaturalNameWarning:
            if verbose:
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

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_renameGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.agroup, 'agroup3')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Ensure that the new name exists
        group = fileh.root.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/agroup3"
        # The children of this group also must be accessible through the
        # new name path
        group2 = fileh.getNode("/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/agroup3/agroup3"
        # Try to get the previous object with the old name
        try:
            object = fileh.root.agroup
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            object = fileh.getNode("/agroup/agroup3")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test09b_renameGroup(self):
        """Checking renaming a Group and access it immediately"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09b_renameGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.agroup, 'agroup3')

        # Ensure that the new name exists
        group = fileh.root.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/agroup3"
        # The children of this group also must be accessible through the
        # new name path
        group2 = fileh.getNode("/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/agroup3/agroup3"
        # Try to get the previous object with the old name
        try:
            object = fileh.root.agroup
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            object = fileh.getNode("/agroup/agroup3")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test09c_renameGroup(self):
        """Checking renaming a Group and modify attributes afterwards"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test09c_renameGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.agroup, 'agroup3')

        # Ensure that we can modify attributes in the new group
        group = fileh.root.agroup3
        group._v_attrs.TITLE = "Hello"
        assert group._v_title == "Hello"
        assert group._v_attrs.TITLE == "Hello"
        fileh.close()

    def test10_moveLeaf(self):
        """Checking moving a leave and access it after a close/open"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_moveLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        newgroup = fileh.createGroup("/", "newgroup")
        fileh.moveNode(fileh.root.anarray, newgroup, 'anarray2')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Ensure that the new name exists
        array_ = fileh.root.newgroup.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/newgroup/anarray2"
        assert array_._v_depth == 2
        # Try to get the previous object with the old name
        try:
            object = fileh.root.anarray
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test10b_moveLeaf(self):
        """Checking moving a leave and access it without a close/open"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10b_moveLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        newgroup = fileh.createGroup("/", "newgroup")
        fileh.moveNode(fileh.root.anarray, newgroup, 'anarray2')

        # Ensure that the new name exists
        array_ = fileh.root.newgroup.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/newgroup/anarray2"
        assert array_._v_depth == 2
        # Try to get the previous object with the old name
        try:
            object = fileh.root.anarray
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test10c_moveLeaf(self):
        """Checking moving Leaves and modify attributes after that"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10c_moveLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        newgroup = fileh.createGroup("/", "newgroup")
        fileh.moveNode(fileh.root.anarray, newgroup, 'anarray2')
        fileh.root.newgroup.anarray2.attrs.TITLE = "hello"
        # Ensure that the new attribute has been written correctly
        array_ = fileh.root.newgroup.anarray2
        assert array_.title == "hello"
        assert array_.attrs.TITLE == "hello"
        fileh.close()

    def test10d_moveToExistingLeaf(self):
        """Checking moving a leaf to an existing name"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10d_moveToExistingLeaf..." % self.__class__.__name__

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        # Try to get the previous object with the old name
        try:
            fileh.moveNode(fileh.root.anarray, fileh.root, 'array')
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test10e_moveToExistingLeafOverwrite(self):
        """Checking moving a leaf to an existing name, overwriting it"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test10e_moveToExistingLeafOverwrite..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        srcNode = fileh.root.anarray
        fileh.moveNode(srcNode, fileh.root, 'array', overwrite = True)
        dstNode = fileh.root.array

        self.assert_(srcNode is dstNode)
        fileh.close()

    def test11_moveGroup(self):
        """Checking moving a Group and access it after a close/open"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_moveGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        newgroup = fileh.createGroup(fileh.root, 'newgroup')
        fileh.moveNode(fileh.root.agroup, newgroup, 'agroup3')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Ensure that the new name exists
        group = fileh.root.newgroup.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/newgroup/agroup3"
        assert group._v_depth == 2
        # The children of this group must also be accessible through the
        # new name path
        group2 = fileh.getNode("/newgroup/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/newgroup/agroup3/agroup3"
        assert group2._v_depth == 3
        # Try to get the previous object with the old name
        try:
            object = fileh.root.agroup
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            object = fileh.getNode("/agroup/agroup3")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test11b_moveGroup(self):
        """Checking moving a Group and access it immediately"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_moveGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        newgroup = fileh.createGroup(fileh.root, 'newgroup')
        fileh.moveNode(fileh.root.agroup, newgroup, 'agroup3')
        # Ensure that the new name exists
        group = fileh.root.newgroup.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/newgroup/agroup3"
        assert group._v_depth == 2
        # The children of this group must also be accessible through the
        # new name path
        group2 = fileh.getNode("/newgroup/agroup3/agroup3")
        assert group2._v_name == "agroup3"
        assert group2._v_pathname == "/newgroup/agroup3/agroup3"
        assert group2._v_depth == 3
        # Try to get the previous object with the old name
        try:
            object = fileh.root.agroup
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        # Try to get a child with the old pathname
        try:
            object = fileh.getNode("/agroup/agroup3")
        except LookupError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next LookupError was catched!"
                print value
        else:
            self.fail("expected an LookupError")
        fileh.close()

    def test11c_moveGroup(self):
        """Checking moving a Group and modify attributes afterwards"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11c_moveGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        newgroup = fileh.createGroup(fileh.root, 'newgroup')
        fileh.moveNode(fileh.root.agroup, newgroup, 'agroup3')

        # Ensure that we can modify attributes in the new group
        group = fileh.root.newgroup.agroup3
        group._v_attrs.TITLE = "Hello"
        group._v_attrs.hola = "Hello"
        assert group._v_title == "Hello"
        assert group._v_attrs.TITLE == "Hello"
        assert group._v_attrs.hola == "Hello"
        fileh.close()

    def test11d_moveToExistingGroup(self):
        """Checking moving a group to an existing name"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11d_moveToExistingGroup..." % self.__class__.__name__

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        # Try to get the previous object with the old name
        try:
            fileh.moveNode(fileh.root.agroup, fileh.root, 'agroup2')
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test11e_moveToExistingGroupOverwrite(self):
        """Checking moving a group to an existing name, overwriting it"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test11e_moveToExistingGroupOverwrite..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup2 -> agroup
        srcNode = fileh.root.agroup2
        fileh.moveNode(srcNode, fileh.root, 'agroup', overwrite = True)
        dstNode = fileh.root.agroup

        self.assert_(srcNode is dstNode)
        fileh.close()

    def test12a_moveNodeOverItself(self):
        """Checking moving a node over itself"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test12_moveNodeOverItself..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # array -> array
        srcNode = fileh.root.array
        fileh.moveNode(srcNode, fileh.root, 'array')
        dstNode = fileh.root.array

        self.assert_(srcNode is dstNode)
        fileh.close()

    def test12b_moveGroupIntoItself(self):
        """Checking moving a group into itself"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test12c_moveGroupIntoItself1..." % self.__class__.__name__

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        try:
            # agroup2 -> agroup2/
            fileh.moveNode(fileh.root.agroup2, fileh.root.agroup2)
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test13a_copyLeaf(self):
        "Copying a leaf."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13a_copyLeaf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # array => agroup2/
        newNode = fileh.copyNode(fileh.root.array, fileh.root.agroup2)
        dstNode = fileh.root.agroup2.array

        self.assert_(newNode is dstNode)
        fileh.close()

    def test13b_copyGroup(self):
        "Copying a group."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13b_copyGroup..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup2 => agroup/
        newNode = fileh.copyNode(fileh.root.agroup2, fileh.root.agroup)
        dstNode = fileh.root.agroup.agroup2

        self.assert_(newNode is dstNode)
        fileh.close()

    def test13c_copyGroupSelf(self):
        "Copying a group into itself."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13c_copyGroupSelf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup2 => agroup2/
        newNode = fileh.copyNode(fileh.root.agroup2, fileh.root.agroup2)
        dstNode = fileh.root.agroup2.agroup2

        self.assert_(newNode is dstNode)
        fileh.close()

    def test13d_copyGroupRecursive(self):
        "Recursively copying a group."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test13d_copyGroupRecursive..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup => agroup2/
        newNode = fileh.copyNode(
            fileh.root.agroup, fileh.root.agroup2, recursive = True)
        dstNode = fileh.root.agroup2.agroup

        self.assert_(newNode is dstNode)
        dstChild1 = dstNode.anarray1
        dstChild2 = dstNode.anarray2
        dstChild3 = dstNode.agroup3
        fileh.close()

    def test14a_copyNodeExisting(self):
        "Copying over an existing node."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test14a_copyNodeExisting..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        try:
            # agroup2 => agroup
            fileh.copyNode(fileh.root.agroup2, newname = 'agroup')
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test14b_copyNodeExistingOverwrite(self):
        "Copying over an existing node, overwriting it."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test14b_copyNodeExistingOverwrite..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup2 => agroup
        newNode = fileh.copyNode(fileh.root.agroup2, newname = 'agroup',
                                 overwrite = True)
        dstNode = fileh.root.agroup

        self.assert_(newNode is dstNode)
        fileh.close()

    def test14c_copyNodeExistingSelf(self):
        "Copying over self."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test14c_copyNodeExistingSelf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        try:
            # agroup => agroup
            fileh.copyNode(fileh.root.agroup, newname = 'agroup')
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test14d_copyNodeExistingOverwriteSelf(self):
        "Copying over self, trying to overwrite."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test14d_copyNodeExistingOverwriteSelf..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        try:
            # agroup => agroup
            fileh.copyNode(
                fileh.root.agroup, newname = 'agroup', overwrite = True)
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test14e_copyGroupSelfRecursive(self):
        "Recursively copying a group into itself."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test14e_copyGroupSelfRecursive..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")
        try:
            # agroup => agroup/
            fileh.copyNode(
                fileh.root.agroup, fileh.root.agroup, recursive = True)
        except NodeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NodeError was catched!"
                print value
        else:
            self.fail("expected an NodeError")
        fileh.close()

    def test15a_oneStepMove(self):
        "Moving and renaming a node in a single action."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test15a_oneStepMove..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # anarray1 -> agroup/array
        srcNode = fileh.root.anarray1
        fileh.moveNode(srcNode, fileh.root.agroup, 'array')
        dstNode = fileh.root.agroup.array

        self.assert_(srcNode is dstNode)
        fileh.close()

    def test15b_oneStepCopy(self):
        "Copying and renaming a node in a single action."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test15b_oneStepMove..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # anarray1 => agroup/array
        newNode = fileh.copyNode(
            fileh.root.anarray1, fileh.root.agroup, 'array')
        dstNode = fileh.root.agroup.array

        self.assert_(newNode is dstNode)
        fileh.close()

    def test16a_fullCopy(self):
        "Copying full data and user attributes."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test16a_fullCopy..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup => groupcopy
        srcNode = fileh.root.agroup
        newNode = fileh.copyNode(
            srcNode, newname = 'groupcopy', recursive = True)
        dstNode = fileh.root.groupcopy

        self.assert_(newNode is dstNode)
        self.assertEqual(srcNode._v_attrs.testattr, dstNode._v_attrs.testattr)
        self.assertEqual(
            srcNode.anarray1.attrs.testattr, dstNode.anarray1.attrs.testattr)
        self.assertEqual(srcNode.anarray1.read(), dstNode.anarray1.read())
        fileh.close()

    def test16b_partialCopy(self):
        "Copying partial data and no user attributes."

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test16b_partialCopy..." % self.__class__.__name__

        fileh = openFile(self.file, mode = "r+")

        # agroup => groupcopy
        srcNode = fileh.root.agroup
        newNode = fileh.copyNode(
            srcNode, newname = 'groupcopy',
            recursive = True, copyuserattrs = False,
            start = 0, stop = 5, step = 2)
        dstNode = fileh.root.groupcopy

        self.assert_(newNode is dstNode)
        self.assert_(not hasattr(dstNode._v_attrs, 'testattr'))
        self.assert_(not hasattr(dstNode.anarray1.attrs, 'testattr'))
        self.assertEqual(srcNode.anarray1.read()[0:5:2], dstNode.anarray1.read())
        fileh.close()


class CheckFileTestCase(unittest.TestCase):

    def test00_isHDF5(self):
        """Checking isHDF5 function (TRUE case)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_isHDF5..." % self.__class__.__name__
        # Create a PyTables file (and by so, an HDF5 file)
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        arr = fileh.createArray(fileh.root, 'array', [1,2],
                                    title = "Title example")
        # For this method to run, it needs a closed file
        fileh.close()
	
        # When file has an HDF5 format, always returns 1
        if verbose:
            print "\nisHDF5(%s) ==> %d" % (file, isHDF5(file))
        assert isHDF5(file) == 1
	
        # Then, delete the file
        os.remove(file)

    def test01_isHDF5File(self):
        """Checking isHDF5 function (FALSE case)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_isHDF5..." % self.__class__.__name__
        # Create a regular (text) file
        file = tempfile.mktemp(".h5")
        fileh = open(file, "w")
        fileh.write("Hello!")
        fileh.close()

	version = isHDF5(file)
        # When file is not an HDF5 format, always returns 0 or
        # negative value
        assert version <= 0
        
        # Then, delete the file
        os.remove(file)

    def test02_isPyTablesFile(self):
        """Checking isPyTablesFile function (TRUE case)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_isPyTablesFile..." % self.__class__.__name__
        # Create a PyTables file
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        arr = fileh.createArray(fileh.root, 'array', [1,2],
                                    title = "Title example")
        # For this method to run, it needs a closed file
        fileh.close()

	version = isPyTablesFile(file)
        # When file has a PyTables format, always returns "1.0" string or
        # greater
        if verbose:
            print
            print "\nPyTables format version number ==> %s" % \
              version
        assert version >= "1.0"

        # Then, delete the file
        os.remove(file)


    def test03_isPyTablesFile(self):
        """Checking isPyTablesFile function (FALSE case)"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_isPyTablesFile..." % self.__class__.__name__
            
        # Create a regular (text) file
        file = tempfile.mktemp(".h5")
        fileh = open(file, "w")
        fileh.write("Hello!")
        fileh.close()

	version = isPyTablesFile(file)
        # When file is not a PyTables format, always returns 0 or
        # negative value
        assert version <= 0
	
        # Then, delete the file
        os.remove(file)

    def test04_openGenericHDF5File(self):
        """Checking opening of a generic HDF5 file"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_openGenericHDF5File..." % self.__class__.__name__

        warnings.filterwarnings("error", category=UserWarning)
        # Open an existing generic HDF5 file
        try:
            fileh = openFile("ex-noattr.h5", mode = "r")
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched:"
                print value
            # Ignore the warning and actually open the file
            warnings.filterwarnings("ignore", category=UserWarning)
            fileh = openFile("ex-noattr.h5", mode = "r")
        else:
            self.fail("expected an UserWarning")
            
        # Reset the warnings
        # Be careful with that, because this enables all the warnings
        # on the rest of the tests!
        #warnings.resetwarnings()
        # better use:
        warnings.filterwarnings("default", category=UserWarning)

        # Check for some objects inside

        # A group
        columns = fileh.getNode("/columns", classname="Group")
        assert columns._v_name == "columns"
        
        # An Array
        array_ = fileh.getNode(columns, "TDC", classname="Array")
        assert array_._v_name == "TDC"

        # An unsupported object (the deprecated H5T_ARRAY type in
        # Array, from pytables 0.8 on)
        ui = fileh.getNode(columns, "pressure", classname="UnImplemented")
        assert ui._v_name == "pressure"
        if verbose:
            print "UnImplement object -->",repr(ui)

        # A Table
        table = fileh.getNode("/detector", "table", classname="Table")
        assert table._v_name == "table"
        
        fileh.close()

    def test05_copyUnimplemented(self):
        """Checking that an UnImplemented object cannot be copied"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_copyUnimplemented..." % self.__class__.__name__

        # Open an existing generic HDF5 file
        # We don't need to wrap this in a try clause because
        # it has already been tried and the warning will not happen again
        fileh = openFile("ex-noattr.h5", mode = "r")
        # An unsupported object (the deprecated H5T_ARRAY type in
        # Array, from pytables 0.8 on)
        ui = fileh.getNode(fileh.root.columns, "pressure")
        assert ui._v_name == "pressure"
        if verbose:
            print "UnImplement object -->",repr(ui)

        # Check that it cannot be copied to another file
        file2 = tempfile.mktemp(".h5")
        fileh2 = openFile(file2, mode = "w")
        # Force the userwarning to issue an error
        warnings.filterwarnings("error", category=UserWarning)
        try:
            ui.copy(fileh2.root, "newui")
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched:"
                print value
        else:
            self.fail("expected an UserWarning")

        # Reset the warnings
        # Be careful with that, because this enables all the warnings
        # on the rest of the tests!
        #warnings.resetwarnings()
        # better use:
        warnings.filterwarnings("default", category=UserWarning)

        # Delete the new (empty) file
        fileh2.close()
        os.remove(file2)

        fileh.close()


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(OpenFileTestCase))
    theSuite.addTest(unittest.makeSuite(CheckFileTestCase))

    return theSuite

 
if __name__ == '__main__':
    unittest.main( defaultTest='suite' )

## Local Variables:
## mode: python
## End:
