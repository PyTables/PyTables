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
        fileh.createArray(fileh.root, 'array', [1,2],
                          title = "Title example")
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
        class_ = fileh.root._f_getLeafAttrStr("array", "CLASS")

        fileh.close()
        os.remove(file)

        assert class_ == "ARRAY"
        
    def test01_openFile(self):
        """Checking opening of an existing file"""

        # Open the old HDF5 file
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root._f_getLeafAttrStr("array", "TITLE")

        assert title == "Title example"
    
    def test01b_trTable(self):
        """Checking the translation table capability for reading"""

        # Open the old HDF5 file
        trTable = {"pythonarray": "array"}
        fileh = openFile(self.file, mode = "r", trTable=trTable)
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
    
    def test01c_trTable(self):
        """Checking the translation table capability for writing"""

        # Create an HDF5 file
        file = tempfile.mktemp(".h5")
        trTable = {"pythonarray": "array"}
        fileh = openFile(file, mode = "w", trTable=trTable)
        arr = fileh.createArray(fileh.root, 'pythonarray', [1,2],
                                title = "Title example")

        # Get the array objects in the file
        array_ = fileh.getNode("/pythonarray")
        assert array_.name == "pythonarray"
        assert array_._v_hdf5name == "array"

        fileh.close()
        
        # Open the old HDF5 file (without the trTable parameter)
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
        title = fileh.root._f_getLeafAttrStr("array2", "TITLE")

        assert title == "Title example 2"

    def test022_appendFile2(self):
        """Checking appending objects to an existing file ("a" version)"""

        # Append a new array to the existing file
        fileh = openFile(self.file, mode = "a")
        fileh.createArray(fileh.root, 'array2', [3,4],
                          title = "Title example 2")
        fileh.close()

        # Open this file in read-only mode
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root._f_getLeafAttrStr("array2", "TITLE")

        assert title == "Title example 2"

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

        # Here, a RuntimeError should be raised!
        try:
            # Try to get the 'array' object in the old existing file
            arr = fileh.root.array
        except AttributeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next AttributeError was catched!"
                print value
        else:
            self.fail("expected an AttributeError")

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

#     # This check no longer applies because we allow any file extension
#     def test05_openErrorFile(self):
#         """Checking opening a non HDF5 file extension"""

#         warnings.filterwarnings("error", category=UserWarning)
#         try:
#             fileh = openFile("nonexistent", mode = "r")
#         except UserWarning:
#             if verbose:
#                 (type, value, traceback) = sys.exc_info()
#                 print "\nGreat!, the next UserWarning was catched!"
#                 print value
#         else:
#             self.fail("expected an UserWarning")
#         # Reset the warning
#         warnings.filterwarnings("default", category=UserWarning)

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
        assert version >= "1.0"
        if verbose:
            print
            print "PyTables format version number ==> %s" % \
              version
	
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
