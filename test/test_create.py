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

class Record(IsDescription):
    var1 = StringCol(length=4)     # 4-character String
    var2 = IntCol()                # integer
    var3 = Int16Col()              # short integer
    var4 = FloatCol()              # double (double-precision)
    var5 = Float32Col()            # float  (single-precision)

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
                                            Record, "Table title")
	# Create an array object
	self.array = self.fileh.createArray(self.root, 'anarray',
                                            [1], "Array title")
	# Create a group object
	self.group = self.fileh.createGroup(self.root, 'agroup',
                                            "Group title")

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

    def test03a_titleAttr(self):
        """Checking the self.title attr in nodes"""

        # Close the opened file to destroy the object tree 
        self.fileh.close()
        # Open the file again to re-create the objects
        self.fileh = openFile(self.file,"r")

        # Now, test that self.title exists and is correct in all the nodes
        assert self.fileh.root.agroup._v_title == "Group title"
        assert self.fileh.root.atable.title == "Table title"
        assert self.fileh.root.anarray.title == "Array title"

    def test03b_titleLength(self):
        """Checking large title character length limit (1024)"""

	titlelength = 1024
	# Try to put a very long title on a group object
	group = self.fileh.createGroup(self.root, 'group',
                                       "t" * titlelength)
        assert group._v_title == "t" * titlelength
        assert group._f_getAttr('TITLE') == "t" * titlelength
	
	# Now, try with a table object
	table = self.fileh.createTable(self.root, 'table',
                                       Record, "t" * titlelength)
	assert table.title == "t" * titlelength
	assert table.getAttr("TITLE") == "t" * titlelength
	    
	# Finally, try with an Array object
        arr = self.fileh.createArray(self.root, 'arr',
                                     [1], "t" * titlelength)
	assert arr.title == "t" * titlelength
	assert arr.getAttr("TITLE") == "t" * titlelength
	    
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
	    recordDict[varname] = Col("Int32", 1)
	    i += 1
	# Append this entry to indicate the alignment!
	recordDict['_v_align'] = "="
	# Now, create a table with this record object
	table = Table(recordDict, "MetaRecord instance")
	# Attach the table to the object tree
	self.root.table = table
        # This works the same than above
	#table = self.fileh.createTable(self.root, 'table',
        #                               recordDict, "MetaRecord instance")
	row = table.row
	# Write 10 records
        for j in range(10):
            i = 0
            for varname in varnames:
                row[varname] = i*j
                i += 1
	    
            row.append()

        # write data on disk
	table.flush()

	# Read all the data as records 
	for recout in table.iterrows():
            pass

        # Compare the last input row and last output
        # they should be equal
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


class createAttrTestCase(unittest.TestCase):
    
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
                                            Record, "Table title")
	# Create an array object
	self.array = self.fileh.createArray(self.root, 'anarray',
                                            [1], "Array title")
	# Create a group object
	self.group = self.fileh.createGroup(self.root, 'agroup',
                                            "Group title")

    def tearDown(self):

        self.fileh.close()
        os.remove(self.file)

#---------------------------------------

    def test01_setAttributes(self):
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
	    
	    
    def test02_setAttributes(self):
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
	    
	    
    def test03_setAttributes(self):
        """Checking setting large string attributes (AttributeSet methods)"""

	attrlength = 2048
	# Try to put a long string attribute on a group object
        self.group._v_attrs.attr1 = "p" * attrlength
        assert self.group._v_attrs.attr1 == "p" * attrlength
	
	# Now, try with a Table object
        self.table.attrs.attr1 = "a" * attrlength
	assert self.table.attrs.attr1 == "a" * attrlength
	    
	# Finally, try with an Array object
        self.array.attrs.attr1 = "n" * attrlength
	assert self.array.attrs.attr1 == "n" * attrlength
	    
    def test04_listAttributes(self):
        """Checking listing attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        if verbose:
            print "Attribute list:", self.group._v_attrs._f_list()
        assert self.group._v_attrs._f_list("user") == \
               ["pq", "qr", "rs"]
        assert self.group._v_attrs._f_list("sys") == \
               ['CLASS', 'TITLE','VERSION']
        assert self.group._v_attrs._f_list("all") == \
               ['CLASS', 'TITLE', 'VERSION', "pq", "qr", "rs"]
	
	# Now, try with a Table object
        self.table.attrs.a = "1"
        self.table.attrs.c = "2"
        self.table.attrs.b = "3"
        if verbose:
            print "Attribute list:", self.table.attrs._f_list()
        assert self.table.attrs._f_list() == ["a", "b", "c"]
        assert self.table.attrs._f_list("sys") == \
               ['CLASS', 'FIELD_0_NAME', 'FIELD_1_NAME', 'FIELD_2_NAME',
                'FIELD_3_NAME', 'FIELD_4_NAME', 'TITLE','VERSION']
        assert self.table.attrs._f_list("readonly") == ['CLASS', 'VERSION']
        assert self.table.attrs._f_list("all") == \
               ['CLASS', 'FIELD_0_NAME', 'FIELD_1_NAME', 'FIELD_2_NAME',
                'FIELD_3_NAME', 'FIELD_4_NAME', 'TITLE', 'VERSION',
                "a", "b", "c"]
	    
	# Finally, try with an Array object
        self.array.attrs.k = "1"
        self.array.attrs.j = "2"
        self.array.attrs.i = "3"
        if verbose:
            print "Attribute list:", self.array.attrs._f_list()
        assert self.array.attrs._f_list() == ["i", "j", "k"]
        assert self.array.attrs._f_list("sys") == \
               ['CLASS', 'FLAVOR', 'TITLE', 'VERSION']
        assert self.array.attrs._f_list("all") == \
               ['CLASS', 'FLAVOR', 'TITLE', 'VERSION', "i", "j", "k"]

    def test05_removeAttributes(self):
        """Checking removing attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # delete an attribute
        del self.group._v_attrs.pq
        if verbose:
            print "Attribute list:", self.group._v_attrs._f_list()
        # Check the local attributes names
        assert self.group._v_attrs._f_list() == ["qr", "rs"]
        if verbose:
            print "Attribute list in disk:", self.group._v_attrs._g_listAttr()
        # Check the disk attribute names
        assert self.group._v_attrs._g_listAttr() == \
               ('TITLE', 'CLASS', 'VERSION', "qr", "rs")

        # delete an attribute (__delattr__ method)
        del self.group._v_attrs.qr
        if verbose:
            print "Attribute list:", self.group._v_attrs._f_list()
        # Check the local attributes names
        assert self.group._v_attrs._f_list() == ["rs"]
        if verbose:
            print "Attribute list in disk:", self.group._v_attrs._g_listAttr()
        # Check the disk attribute names
        assert self.group._v_attrs._g_listAttr() == \
               ('TITLE', 'CLASS', 'VERSION', "rs")

    def test06_removeAttributes(self):
        """Checking removing system attributes """

        # remove a system attribute
        try:
            if verbose:
                print "System attrs:", self.group._v_attrs._v_attrnamessys
            del self.group._v_attrs.CLASS
        except RuntimeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected a RuntimeError")

    def test07_renameAttributes(self):
        """Checking renaming attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # rename an attribute
        self.group._v_attrs._f_rename("pq", "op")
        if verbose:
            print "Attribute list:", self.group._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        assert self.group._v_attrs._f_list() == ["op", "qr", "rs"]
        if verbose:
            print "Attribute list in disk:", self.group._v_attrs._g_listAttr()
        # Check the disk attribute names (not sorted)
        assert self.group._v_attrs._g_listAttr() == \
               ('TITLE', 'CLASS', 'VERSION', "qr", "rs", "op")

    def test08_renameAttributes(self):
        """Checking renaming system attributes """

        # rename a system attribute
        try:
            self.group._v_attrs._f_rename("CLASS", "op")
        except RuntimeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next RuntimeError was catched!"
                print value
        else:
            self.fail("expected a RuntimeError")

    def test09_setIntAttributes(self):
        """Checking setting Int attributes"""

        # With a Table object
        self.table.attrs.pq = 1
        self.table.attrs.qr = 2
        self.table.attrs.rs = 3

        # Check the results
        if verbose:
            print "pq -->", self.table.attrs.pq
            print "qr -->", self.table.attrs.qr
            print "rs -->", self.table.attrs.rs
            
        assert self.table.attrs.pq == 1
        assert self.table.attrs.qr == 2
        assert self.table.attrs.rs == 3

    def test09b_setIntAttributes(self):
        """Checking setting Int (byte, short, int) attributes"""

        # With a Table object
        self.table.attrs._g_setAttrChar("pq", chr(1))
        self.table.attrs._v_attrnames.append("pq")
        self.table.attrs._v_attrnamesuser.append("pq")

        self.table.attrs._g_setAttrShort("qr", 2)
        self.table.attrs._v_attrnames.append("qr")
        self.table.attrs._v_attrnamesuser.append("qr")
        
        self.table.attrs._g_setAttrInt("rs", 3)
        self.table.attrs._v_attrnames.append("rs")
        self.table.attrs._v_attrnamesuser.append("rs")

        # Check the results
        if verbose:
            print "pq -->", self.table.attrs.pq
            print "qr -->", self.table.attrs.qr
            print "rs -->", self.table.attrs.rs
            
        assert self.table.attrs.pq == 1
        assert self.table.attrs.qr == 2
        assert self.table.attrs.rs == 3

    def test10_setFloatAttributes(self):
        """Checking setting Float (double) attributes"""

        # With a Table object
        self.table.attrs.pq = 1.0
        self.table.attrs.qr = 2.0
        self.table.attrs.rs = 3.0

        # Check the results
        if verbose:
            print "pq -->", self.table.attrs.pq
            print "qr -->", self.table.attrs.qr
            print "rs -->", self.table.attrs.rs
            
        assert self.table.attrs.pq == 1.0
        assert self.table.attrs.qr == 2.0
        assert self.table.attrs.rs == 3.0

    def test10b_setFloatAttributes(self):
        """Checking setting Float (float) attributes"""

        # With a Table object
        self.table.attrs._g_setAttrFloat("pq", 1.0)
        self.table.attrs._v_attrnames.append("pq")
        self.table.attrs._v_attrnamesuser.append("pq")

        self.table.attrs._g_setAttrFloat("qr", 2.0)
        self.table.attrs._v_attrnames.append("qr")
        self.table.attrs._v_attrnamesuser.append("qr")
        
        self.table.attrs._g_setAttrFloat("rs", 3.0)
        self.table.attrs._v_attrnames.append("rs")
        self.table.attrs._v_attrnamesuser.append("rs")

        # Check the results
        if verbose:
            print "pq -->", self.table.attrs.pq
            print "qr -->", self.table.attrs.qr
            print "rs -->", self.table.attrs.rs
            
        assert self.table.attrs.pq == 1.0
        assert self.table.attrs.qr == 2.0
        assert self.table.attrs.rs == 3.0

    def test11_setObjectAttributes(self):
        """Checking setting Object attributes"""

        # With a Table object
        self.table.attrs.pq = [1.0, 2]
        self.table.attrs.qr = (1,2)
        self.table.attrs.rs = {"ddf":32.1, "dsd":1}

        # Check the results
        if verbose:
            print "pq -->", self.table.attrs.pq
            print "qr -->", self.table.attrs.qr
            print "rs -->", self.table.attrs.rs
            
        assert self.table.attrs.pq == [1.0, 2]             
        assert self.table.attrs.qr == (1,2)
        assert self.table.attrs.rs == {"ddf":32.1, "dsd":1}

	
#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()

    for i in range(1):
        theSuite.addTest(unittest.makeSuite(createTestCase))
    for i in range(1):
        theSuite.addTest(unittest.makeSuite(createAttrTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
