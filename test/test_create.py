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

from Numeric import *
from tables import *
# important objects to test
from tables import File, Group, Leaf, Table, Array, metaIsRecord

from test_all import verbose

class Record(IsRecord):
    var1 = '4s'   # 4-character String
    var2 = 'i'    # integer
    var3 = 'h'    # short integer. This is chosen in this place for 
                  # discovery of alignment issues!
    var4 = 'd'    # double (double-precision)
    var5 = 'f'    # float  (single-precision)


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
	#self.table = createTable(Record(), "Table title")
	
	# Create an array object
	self.array = self.fileh.createArray(self.root, 'anarray',
                                            array([1]), "Array title")
	#self.array = createArray(array([1]), "Array title")
	
	# Create a group object
	self.group = self.fileh.createGroup(self.root, 'agroup',
                                            "Group title")
	#self.group = createGroup("Group title")
	

    def tearDown(self):

        self.fileh.close()
        os.remove(self.file)

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
                                                array([1]), "Array title")
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
                                                array([1]), "Array title")
        except SyntaxError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next SyntaxError was catched!"
                print value
        else:
            self.fail("expected a SyntaxError")
	    
	# another syntax error
        try:
            self.array = self.fileh.createArray(self.root, '$array',
                                                array([1]), "Array title")
        except SyntaxError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next SyntaxError was catched!"
                print value
        else:
            self.fail("expected a SyntaxError")

	# Finally, test a reserved word
        try:
            self.array = self.fileh.createArray(self.root, 'for',
                                                array([1]), "Array title")
        except SyntaxError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next SyntaxError was catched!"
                print value
        else:
            self.fail("expected a SyntaxError")

    def test03_titleLenght(self):
        """Checking title character length limit (255)"""
	
	# Try to put a very long title on a group object
	group = self.fileh.createGroup(self.root, 'group',
                                       "t" * 1000)
        # For Group, unlimited title lenght is supported
        assert group._f_getGroupAttrStr('TITLE') == "t" * 1000
	
	# Now, try with a table object
	# This supports titles until 255 characters. The rest is lost.
	table = self.fileh.createTable(self.root, 'table',
                                       Record(), "t" * 512)
	# getTableTitle can retrieve only the first 255 charactes
	assert table.getTitle() == "t" * 255
	    
	# Finally, try with an Array object
	# This supports titles until 255 characters. The rest is lost.
        arr = self.fileh.createArray(self.root, 'arr',
                                     array([1]), "t" * 512)
	# getTitle can retrieve only the first 255 charactes
	assert arr.getTitle() == "t" * 255
	    
    def test04_maxFields(self):
	"Checking the maximum number of fields (255) in tables"

	# The number of fields for a table
	varnumber = 255

	varnames = []
	for i in range(varnumber):
	    varnames.append('int%d' % i)
            
	# The format string for this record
	fmt = "@" + ("i" * varnumber)
        
	# Get the variable types
	vartypes = re.findall(r'(\d*\w)', fmt)
	
	# Build a dictionary with the types as values and varnames as keys
	recordDict = {}
	i = 0
	for varname in varnames:
	    recordDict[varname] = vartypes[i]
	    i += 1
            
	# Append this entry to indicate the alignment!
	recordDict['_v_align'] = fmt[0]
	
	# Create an instance record to host the record fields
	record = metaIsRecord("", (), recordDict)()
        
	# Now, create a table with this record object
	table = Table(record, "MetaRecord instance")

	# Attach the table to object tree
	self.root.table = table
	
	# Write 10 records
        for j in range(10):
            i = 0
            for varname in varnames:
                setattr(record, varname, i*j)
                i += 1
	    
            table.appendAsRecord(record)

        # write data on disk
	table.flush()
	
	# Read all the data as records 
	for recout in table.readAsRecords():
            pass

        # Compare the last input record and last output
        # They should be equal
        assert record == recout
	    
    def test05_maxFieldsExceeded(self):
        
	"Checking an excess (256) of the maximum number of fields in tables"

	# The number of fields for a table
	varnumber = 256

	varnames = []
	for i in range(varnumber):
	    varnames.append('int%d' % i)
            
	# The format string for this record
	fmt = "@" + ("i" * varnumber)
        
	# Get the variable types
	vartypes = re.findall(r'(\d*\w)', fmt)
	
	# Build a dictionary with the types as values and varnames as keys
	recordDict = {}
	i = 0
	for varname in varnames:
	    recordDict[varname] = vartypes[i]
	    i += 1
            
	# Append this entry to indicate the alignment!
	recordDict['_v_align'] = fmt[0]
	
	# Create an instance record to host the record fields
	record = metaIsRecord("", (), recordDict)()
        
	# Now, create a table with this record object
	table = Table(record, "MetaRecord instance")

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

    theSuite.addTest(unittest.makeSuite(createTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
