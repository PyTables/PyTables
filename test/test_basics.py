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

        assert class_ == "ARRAY"
        
    def test01_openFile(self):
        """Checking opening of an existing file"""

        # Open the old HDF5 file
        fileh = openFile(self.file, mode = "r")
        # Get the CLASS attribute of the arr object
        title = fileh.root._f_getLeafAttrStr("array", "TITLE")

        assert title == "Title example"
    
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
        

    def test05_openErrorFile(self):
        """Checking opening a non HDF5 file extension"""

        warnings.filterwarnings("error", category=UserWarning)
        try:
            fileh = openFile("nonexistent", mode = "r")
        except UserWarning:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UserWarning was catched!"
                print value
        except IOError:
            # Just in case someone run the test with -O optimization flag!
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IOError was catched!"
                print value
        else:
            self.fail("expected an AssertionError or IOError")
        # Reset the warnings
        warnings.resetwarnings()

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


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(OpenFileTestCase))
    theSuite.addTest(unittest.makeSuite(CheckFileTestCase))

    return theSuite

 
if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
