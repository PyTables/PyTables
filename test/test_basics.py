import sys
import unittest
import os
import tempfile
import warnings

from tables import *

from test_all import verbose

class OpenFileTestCase(unittest.TestCase):

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, mode = "w")
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
	# Create a couple of objects there
	array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
	array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
	# Create a lonely group in first level
	group2 = fileh.createGroup(root, 'agroup2',
                                  "Group title 2")
        # Create a new group in the second level
	group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")
                                            
        fileh.close()
        
    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)

    def test00_newFile(self):
        """Checking creation of a new file"""

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

        # Open the old HDF5 file
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root.array.getAttr("TITLE")

        assert title == "Title example"
	fileh.close()
    
    def test01b_trMap(self):
        """Checking the translation table capability for reading"""

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
        
    def test04_openErrorFile(self):
        """Checking opening a non-existing file for reading"""

        try:
            fileh = openFile("nonexistent.h5", mode = "r")
        except IOError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IOError was catched!"
                print value
        else:
            self.fail("expected an IOError")

    def test05a_removeGroupRecursively(self):
        """Checking removing a group recursively"""

        # Delete a group with leafs
        fileh = openFile(self.file, mode = "r+")
        
        warnings.filterwarnings("error", category=UserWarning)
        try:
            fileh.removeNode(fileh.root.agroup)
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched!"
                print value
        else:
            self.fail("expected an UserWarning")
        # Reset the warning
        warnings.filterwarnings("default", category=UserWarning)

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

        # Delete a group with leafs
        fileh = openFile(self.file, mode = "r+")
        
        warnings.filterwarnings("error", category=UserWarning)
        try:
            fileh.removeNode(fileh.root, 'agroup')
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched!"
                print value
        else:
            self.fail("expected an UserWarning")
        # Reset the warning
        warnings.filterwarnings("default", category=UserWarning)

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

        # Delete a group with leafs
        fileh = openFile(self.file, mode = "r+")
        
        # Delete a group recursively
        del fileh.root.agroup

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

    def test06a_removeGroup(self):
        """Checking removing a lonely group from an existing file"""

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

    def test06b_removeLeaf(self):
        """Checking removing Leaves from an existing file"""

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

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.anarray, 'anarray2')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Ensure that the new name exists
        array_ = fileh.root.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/anarray2"
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

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.anarray, 'anarray2')

        # Ensure that the new name exists
        array_ = fileh.root.anarray2
        assert array_.name == "anarray2"
        assert array_._v_pathname == "/anarray2"
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

    def test08_renameToExistingLeaf(self):
        """Checking renaming a node to an existing name"""

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        # Try to get the previous object with the old name
        try:
            fileh.renameNode(fileh.root.anarray, 'array')        
        except RuntimeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected an RuntimeError")
        fileh.close()

    def test08b_renameToNotValidName(self):
        """Checking renaming a node to a non-valid name"""

        # Open this file
        fileh = openFile(self.file, mode = "r+")
        # Try to get the previous object with the old name
        try:
            fileh.renameNode(fileh.root.anarray, 'array 2')        
        except NameError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected an NameError")
        fileh.close()

    def test09_renameGroup(self):
        """Checking renaming a Group and access it after a close/open"""

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.agroup, 'agroup3')
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Ensure that the new name exists
        group = fileh.root.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/agroup3"
        # The childs of this group also must be accessible through the
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

        fileh = openFile(self.file, mode = "r+")
        fileh.renameNode(fileh.root.agroup, 'agroup3')

        # Ensure that the new name exists
        group = fileh.root.agroup3
        assert group._v_name == "agroup3"
        assert group._v_pathname == "/agroup3"
        # The childs of this group also must be accessible through the
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

class CheckFileTestCase(unittest.TestCase):
    
    def test00_isHDF5File(self):
        """Checking isHDF5 function (TRUE case)"""
        
        # Create a PyTables file (and by so, an HDF5 file)
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, mode = "w")
        arr = fileh.createArray(fileh.root, 'array', [1,2],
                                    title = "Title example")
        # For this method to run, it needs a closed file
        fileh.close()
	
        # When file has an HDF5 format, always returns 1
        if verbose:
            print "\nisHDF5(%s) --> %d" % (file, isHDF5(file))
        assert isHDF5(file) == 1
	
        # Then, delete the file
        os.remove(file)

    def test01_isHDF5File(self):
        """Checking isHDF5 function (FALSE case)"""

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

        warnings.filterwarnings("error", category=UserWarning)
        # Open an existing generic HDF5 file
        try:
            fileh = openFile("ex-noattr.h5", mode = "r")
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched!"
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
        
        # A Table
        table = fileh.getNode("/detector", "table", classname="Table")
        assert table._v_name == "table"
        
        fileh.close()


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(OpenFileTestCase))
    theSuite.addTest(unittest.makeSuite(CheckFileTestCase))

    return theSuite

 
if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
