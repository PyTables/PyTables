""" This test unit checks object creation funtions, like openFile, createTable,
createArray or createGroup.
It also checks:
- name identifiers in tree objects
- title character limit for objects (255)
- limit in number in table fields (255)
"""

import sys
import unittest
import os
import re
import tempfile

from tables import *
# important objects to test
from tables import File, Group, Leaf, Table, Array

from test_all import verbose

class Record(IsRecord):
    var1 = Col("CharType", 4)   # 4-character String
    var2 = Col("Int32", 1)      # integer
    var3 = Col("Int16", 1)      # short integer. 
    var4 = Col("Float64", 1)    # double (double-precision)
    var5 = Col("Float32", 1)    # float  (single-precision)

class createTestCase(unittest.TestCase):
    
    file  = "test.h5"
    title = "This is the table title"
    expectedrows = 100
    maxshort = 2 ** 15
    maxint   = 2147483648   # (2 ** 31)
    compress = 0

    
    def setUp(self):
        # Create an instance of HDF5 Table
        self.fileh = openFile(self.file, mode = "w")
        self.root = self.fileh.root

	# Create a table object
	self.table = self.fileh.createTable(self.root, 'atable',
                                            Record(), "Table title")
	# Create an array object
	self.array = self.fileh.createArray(self.root, 'anarray',
                                            [1], "Array title")
	# Create a group object
	self.group = self.fileh.createGroup(self.root, 'agroup',
                                            "Group title")

    def tearDown(self):

        self.fileh.close()
        os.remove(self.file)
        # Delete references
        del self.fileh, self.root, self.table, self.array, self.group

    #----------------------------------------

    def test00_isClass(self):
        """Testing table creation"""
	assert isinstance(self.table, Table)
	assert isinstance(self.array, Array)
	assert isinstance(self.array, Leaf)
	assert isinstance(self.group, Group)

    def test01_overwriteNode(self):
        """Checking protection against node overwriting"""

        try:
            self.array = self.fileh.createArray(self.root, 'anarray',
                                                [1], "Array title")
        except NameError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected a NameError")
	
    def test02_syntaxname(self):
        """Checking syntax in object tree names"""

	# Now, try to attach an array to the object tree with
	# a not allowed Python variable name
        try:
            self.array = self.fileh.createArray(self.root, ' array',
                                                [1], "Array title")
        except NameError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected a NameError")
	    
	# another name error
        try:
            self.array = self.fileh.createArray(self.root, '$array',
                                                [1], "Array title")
        except NameError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected a NameError")

	# Finally, test a reserved word
        try:
            self.array = self.fileh.createArray(self.root, 'for',
                                                [1], "Array title")
        except NameError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected a NameError")

    def test03_titleLength(self):
        """Checking large title character length limit (1024)"""

	titlelength = 1024
	# Try to put a very long title on a group object
	group = self.fileh.createGroup(self.root, 'group',
                                       "t" * titlelength)
        assert group._f_getAttr('TITLE') == "t" * titlelength
	
	# Now, try with a table object
	table = self.fileh.createTable(self.root, 'table',
                                       Record(), "t" * titlelength)
	assert table.getAttr("TITLE") == "t" * titlelength
	    
	# Finally, try with an Array object
        arr = self.fileh.createArray(self.root, 'arr',
                                     [1], "t" * titlelength)
	assert arr.title == "t" * titlelength
	    
    def test03b_setAttributes(self):
        """Checking setting large string attributes (File methods)"""

	attrlength = 2048
	# Try to put a long string attribute on a group object
	attr = self.fileh.setAttrNode(self.root.agroup,
                                      "attr1", "p" * attrlength)
        assert self.fileh.getAttrNode(self.root.agroup, 'attr1') == \
               "p" * attrlength
	
	# Now, try with a Table object
	attr = self.fileh.setAttrNode(self.root.atable,
                                      "attr1", "a" * attrlength)
        assert self.fileh.getAttrNode(self.root.atable, 'attr1') == \
               "a" * attrlength
	    
	# Finally, try with an Array object
	attr = self.fileh.setAttrNode(self.root.anarray,
                                      "attr1", "n" * attrlength)
        assert self.fileh.getAttrNode(self.root.anarray, 'attr1') == \
               "n" * attrlength
	    
	    
    def test03c_setAttributes(self):
        """Checking setting large string attributes (Node methods)"""

	attrlength = 2048
	# Try to put a long string attribute on a group object
        self.root.agroup._f_setAttr('attr1', "p" * attrlength)
        assert self.root.agroup._f_getAttr('attr1') == "p" * attrlength
	
	# Now, try with a Table object
        self.root.atable.setAttr('attr1', "a" * attrlength)
	assert self.root.atable.getAttr("attr1") == "a" * attrlength
	    
	# Finally, try with an Array object
        self.root.anarray.setAttr('attr1', "n" * attrlength)
	assert self.root.anarray.getAttr("attr1") == "n" * attrlength
	    
	    
    def test04_maxFields(self):
	"Checking the maximum number of fields (255) in tables"

	# The number of fields for a table
	varnumber = 255

	varnames = []
	for i in range(varnumber):
	    varnames.append('int%d' % i)
            
	# Build a dictionary with the types as values and varnames as keys
	recordDict = {}
	i = 0
	for varname in varnames:
	    #recordDict[varname] = vartypes[i]
	    recordDict[varname] = Col("Int32", 1)
	    i += 1
            
	# Append this entry to indicate the alignment!
	recordDict['_v_align'] = "="
        
	# Now, create a table with this record object
	table = Table(recordDict, "MetaRecord instance")

	# Attach the table to object tree
	self.root.table = table
	row = table.row
	# Write 10 records
        for j in range(10):
            i = 0
            for varname in varnames:
                setattr(row, varname, i*j)
                i += 1
	    
            row.append()

        # write data on disk
	table.flush()
	
	# Read all the data as records 
	for recout in table.iterrows():
            pass

        # Compare the last input row and last output
        # They should be equal
        assert row == recout
	    
    def test05_maxFieldsExceeded(self):
        
	"Checking an excess (256) of the maximum number of fields in tables"

	# The number of fields for a table
	varnumber = 256

	varnames = []
	for i in range(varnumber):
	    varnames.append('int%d' % i)
            
	# Build a dictionary with the types as values and varnames as keys
	recordDict = {}
	i = 0
	for varname in varnames:
	    #recordDict[varname] = vartypes[i]
	    recordDict[varname] = Col("Int32", 1)
	    i += 1
            
	# Append this entry to indicate the alignment!
	recordDict['_v_align'] = "="
        
	# Now, create a table with this record object
	table = Table(recordDict, "MetaRecord instance")

	# Attach the table to object tree
        # Here, IndexError should be raised!
        try:
            self.root.table = table
        except IndexError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IndexError was catched!"
                print value
        else:
            self.fail("expected an IndexError")
	
        
#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    for i in range(1):
        theSuite.addTest(unittest.makeSuite(createTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
