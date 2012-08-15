import sys
import unittest
import os
import tempfile
import warnings
import subprocess

import numpy

import tables
import tables.flavor
from tables import *
from tables.flavor import all_flavors, array_of_flavor
from tables.tests import common
from tables.parameters import NODE_CACHE_SLOTS
from tables.description import descr_from_dtype, dtype_from_descr


class FileDriverTestCase(common.PyTablesTestCase):
    file=None
    DRIVER=None
    def test00_newFile(self):
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, mode = "w", title="File title", DRIVER=self.DRIVER)
        # Create an HDF5 file
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1, 2],
                          title = "Array example")
        fileh.createTable(root, 'table', {'var1':IntCol()}, "Table example")
        root._v_attrs.testattr = 41
        fileh.close()

	try:
            """Checking opening of an existing file"""
            # Open the old HDF5 file
            fileh = openFile(self.file, mode = "r", title="File title", DRIVER=self.DRIVER)
            # Get the CLASS attribute of the arr object
            title = fileh.root.array.getAttr("TITLE")

            self.assertEqual(title, "Array example")
            fileh.close()
        except:
		self.fail("Can't open file for reading with DRIVER="+self.DRIVER);
        finally:
                # Remove the temporary file
	        try:
        	    os.remove(self.file)
	        except OSError:
		    pass
	        self.fileh=None

class DefaultDriverTestCase(FileDriverTestCase):
    DRIVER=None
class SEC2DriverTestCase(FileDriverTestCase):
    DRIVER="H5FD_SEC2"
class STDIODriverTestCase(FileDriverTestCase):
    DRIVER="H5FD_STDIO"
class COREDriverTestCase(FileDriverTestCase):
    DRIVER="H5FD_CORE"
class CORE_INMEMORYDriverTestCase(FileDriverTestCase):
    DRIVER="H5FD_CORE_INMEMORY"
    def test00_newFile(self):
        fileh = openFile("in-memory", mode = "w", title="File title", DRIVER=self.DRIVER)
        # Create an HDF5 file
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1, 2],
                          title = "Array example")
        fileh.createTable(root, 'table', {'var1':IntCol()}, "Table example")
        root._v_attrs.testattr = 41
        fileh.close()
        
	image=fileh.getInMemoryFileContents()
	try:
            """Checking opening of an existing file"""
            # Open the old HDF5 file
            fileh = openFile(None, mode = "r", title="File title", DRIVER=self.DRIVER, H5FD_CORE_INMEMORY_IMAGE=image)
            # Get the CLASS attribute of the arr object
            title = fileh.root.array.getAttr("TITLE")

            self.assertEqual(title, "Array example")
            fileh.close()
        except:
		# This test can fail if libhdf5hl is not installed
		pass
        self.fileh=None

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(FileDriverTestCase))
    theSuite.addTest(unittest.makeSuite(SEC2DriverTestCase))
    theSuite.addTest(unittest.makeSuite(STDIODriverTestCase))
    theSuite.addTest(unittest.makeSuite(COREDriverTestCase))
    theSuite.addTest(unittest.makeSuite(CORE_INMEMORYDriverTestCase))
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )

## Local Variables:
## mode: python
## End:
