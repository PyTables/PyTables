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
        """Checking large title character length limit (1023)"""

	titlelength = 1023
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

    def test06_maxColumnNameLengthExceeded(self):
	"Checking an excess (256) of the maximum length in column names"

	# Build a dictionary with the types as values and varnames as keys
	recordDict = {}
	recordDict["a"*255] = IntCol(1) 
	recordDict["b"*256] = IntCol(1) # Should trigger a NameError
        
	# Now, create a table with this record object
	table = Table(recordDict, "MetaRecord instance")

	# Attach the table to object tree
        # Here, IndexError should be raised!
        try:
            self.root.table = table
        except NameError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected an NameError")


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
	# Now, try with a Table object
	attr = self.fileh.setAttrNode(self.root.atable,
                                      "attr1", "a" * attrlength)
	# Finally, try with an Array object
	attr = self.fileh.setAttrNode(self.root.anarray,
                                      "attr1", "n" * attrlength)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.fileh.getAttrNode(self.root.agroup, 'attr1') == \
               "p" * attrlength
        assert self.fileh.getAttrNode(self.root.atable, 'attr1') == \
               "a" * attrlength
        assert self.fileh.getAttrNode(self.root.anarray, 'attr1') == \
               "n" * attrlength
	    
	    
    def test02_setAttributes(self):
        """Checking setting large string attributes (Node methods)"""

	attrlength = 2048
	# Try to put a long string attribute on a group object
        self.root.agroup._f_setAttr('attr1', "p" * attrlength)
	# Now, try with a Table object
        self.root.atable.setAttr('attr1', "a" * attrlength)
	    
	# Finally, try with an Array object
        self.root.anarray.setAttr('attr1', "n" * attrlength)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.root.agroup._f_getAttr('attr1') == "p" * attrlength
	assert self.root.atable.getAttr("attr1") == "a" * attrlength
	assert self.root.anarray.getAttr("attr1") == "n" * attrlength
	    
	    
    def test03_setAttributes(self):
        """Checking setting large string attributes (AttributeSet methods)"""

	attrlength = 2048
	# Try to put a long string attribute on a group object
        self.group._v_attrs.attr1 = "p" * attrlength
	# Now, try with a Table object
        self.table.attrs.attr1 = "a" * attrlength
	# Finally, try with an Array object
        self.array.attrs.attr1 = "n" * attrlength
        
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.root.agroup._v_attrs.attr1 == "p" * attrlength
	assert self.root.atable.attrs.attr1 == "a" * attrlength
	assert self.root.anarray.attrs.attr1 == "n" * attrlength
	    
    def test04_listAttributes(self):
        """Checking listing attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        if verbose:
            print "Attribute list:", self.group._v_attrs._f_list()

	# Now, try with a Table object
        self.table.attrs.a = "1"
        self.table.attrs.c = "2"
        self.table.attrs.b = "3"
        if verbose:
            print "Attribute list:", self.table.attrs._f_list()

	# Finally, try with an Array object
        self.array.attrs.k = "1"
        self.array.attrs.j = "2"
        self.array.attrs.i = "3"
        if verbose:
            print "Attribute list:", self.array.attrs._f_list()

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            
        assert self.root.agroup._v_attrs._f_list("user") == \
               ["pq", "qr", "rs"]
        assert self.root.agroup._v_attrs._f_list("sys") == \
               ['CLASS','FILTERS', 'TITLE','VERSION']
        assert self.root.agroup._v_attrs._f_list("all") == \
               ['CLASS','FILTERS', 'TITLE', 'VERSION', "pq", "qr", "rs"]

        assert self.root.atable.attrs._f_list() == ["a", "b", "c"]
        assert self.root.atable.attrs._f_list("sys") == \
               ['CLASS', 'FIELD_0_NAME', 'FIELD_1_NAME', 'FIELD_2_NAME',
                'FIELD_3_NAME', 'FIELD_4_NAME', 'NROWS', 'TITLE',
                'VERSION']
        assert self.root.atable.attrs._f_list("readonly") == \
               ['CLASS', 'NROWS', 'VERSION']
        assert self.root.atable.attrs._f_list("all") == \
               ['CLASS', 'FIELD_0_NAME', 'FIELD_1_NAME', 'FIELD_2_NAME',
                'FIELD_3_NAME', 'FIELD_4_NAME', 'NROWS', 'TITLE',
                'VERSION', "a", "b", "c"]
	    
        assert self.root.anarray.attrs._f_list() == ["i", "j", "k"]
        assert self.root.anarray.attrs._f_list("sys") == \
               ['CLASS', 'FLAVOR', 'TITLE', 'VERSION']
        assert self.root.anarray.attrs._f_list("all") == \
               ['CLASS', 'FLAVOR', 'TITLE', 'VERSION',
                "i", "j", "k"]

    def test05_removeAttributes(self):
        """Checking removing attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # delete an attribute
        del self.group._v_attrs.pq
        
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            
        if verbose:
            print "Attribute list:", self.root.agroup._v_attrs._f_list()
        # Check the local attributes names
        assert self.root.agroup._v_attrs._f_list() == ["qr", "rs"]
        if verbose:
            print "Attribute list in disk:", \
                  self.root.agroup._v_attrs._f_list("all")
        # Check the disk attribute names
        assert self.root.agroup._v_attrs._f_list("all") == \
               ['CLASS', 'FILTERS', 'TITLE', 'VERSION', "qr", "rs"]

        # delete an attribute (__delattr__ method)
        del self.root.agroup._v_attrs.qr
        if verbose:
            print "Attribute list:", self.root.agroup._v_attrs._f_list()
        # Check the local attributes names
        assert self.root.agroup._v_attrs._f_list() == ["rs"]
        if verbose:
            print "Attribute list in disk:", \
                  self.root.agroup._v_attrs._g_listAttr()
        # Check the disk attribute names
        assert self.root.agroup._v_attrs._f_list("all") == \
               ['CLASS', 'FILTERS', 'TITLE', 'VERSION', "rs"]

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

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            
        if verbose:
            print "Attribute list:", self.root.agroup._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        assert self.root.agroup._v_attrs._f_list() == ["op", "qr", "rs"]
        if verbose:
            print "Attribute list in disk:", self.root.agroup._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        assert self.root.agroup._v_attrs._f_list("all") == \
               ['CLASS', 'FILTERS', 'TITLE', 'VERSION', "op", "qr", "rs"]

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
            
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            
        assert self.root.atable.attrs.pq == 1
        assert self.root.atable.attrs.qr == 2
        assert self.root.atable.attrs.rs == 3

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
            
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.root.atable.attrs.pq == 1
        assert self.root.atable.attrs.qr == 2
        assert self.root.atable.attrs.rs == 3

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
            
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.root.atable.attrs.pq == 1.0
        assert self.root.atable.attrs.qr == 2.0
        assert self.root.atable.attrs.rs == 3.0

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
            
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.root.atable.attrs.pq == 1.0
        assert self.root.atable.attrs.qr == 2.0
        assert self.root.atable.attrs.rs == 3.0

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
            
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.root.atable.attrs.pq == [1.0, 2]             
        assert self.root.atable.attrs.qr == (1,2)
        assert self.root.atable.attrs.rs == {"ddf":32.1, "dsd":1}

    def test12_overwriteAttributes(self):
        """Checking overwriting attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # overwrite attributes
        self.group._v_attrs.pq = "4"
        self.group._v_attrs.qr = 2
        self.group._v_attrs.rs = [1,2,3]

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            
        if verbose:
            print "Value of Attribute pq:", self.root.agroup._v_attrs.pq
        # Check the local attributes names (alphabetically sorted)
        assert self.root.agroup._v_attrs.pq == "4"
        assert self.root.agroup._v_attrs.qr == 2
        assert self.root.agroup._v_attrs.rs == [1,2,3]
        if verbose:
            print "Attribute list in disk:", \
                  self.root.agroup._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        assert self.root.agroup._v_attrs._f_list("all") == \
               ['CLASS', 'FILTERS', 'TITLE', 'VERSION', "pq", "qr", "rs"]


class createAttrNotCloseTestCase(createAttrTestCase):
    close = 0

class createAttrCloseTestCase(createAttrTestCase):
    close = 1

class Record2(IsDescription):
    var1 = StringCol(length=4)    # 4-character String
    var2 = IntCol()               # integer
    var3 = Int16Col()             # short integer

class FiltersTreeTestCase(unittest.TestCase):
    title = "A title"
    nrows = 10

    def setUp(self):
        # Create a temporary file
        self.file = tempfile.mktemp(".h5")
        # Create an instance of HDF5 Table
        self.h5file = openFile(self.file, "w", filters=self.filters)
        self.populateFile()
            
    def populateFile(self):
        group = self.h5file.root
        # Create a tree with three levels of depth
        for j in range(5):
            # Create a table
            table = self.h5file.createTable(group, 'table1', Record2,
                                        title = self.title,
                                        filters = None)
            # Get the record object associated with the new table
            d = table.row 
            # Fill the table
            for i in xrange(self.nrows):
                d['var1'] = '%04d' % (self.nrows - i)
                d['var2'] = i 
                d['var3'] = i * 2
                d.append()      # This injects the Record values
            # Flush the buffer for this table
            table.flush()
            
            # Create a couple of arrays in each group
            var1List = [ x['var1'] for x in table.iterrows() ]
            var3List = [ x['var3'] for x in table.iterrows() ]

            self.h5file.createArray(group, 'array1', var1List, "col 1")
            self.h5file.createArray(group, 'array2', var3List, "col 3")

            # Create a couple of EArrays as well
            ea1 = self.h5file.createEArray(group, 'earray1',
                                           StringAtom(shape=(0,), length=4),
                                           "col 1")
            ea2 = self.h5file.createEArray(group, 'earray2',
                                           Int16Atom(shape=(0,)), "col 3")
            # And fill them with some values
            ea1.append(var1List)
            ea2.append(var3List)
            
            # Create a new group (descendant of group)
            if j == 1: # The second level
                group2 = self.h5file.createGroup(group, 'group'+str(j),
                                                 filters=self.gfilters)
            elif j == 2: # third level
                group2 = self.h5file.createGroup(group, 'group'+str(j))
            else:   # The rest of levels
                group2 = self.h5file.createGroup(group, 'group'+str(j),
                                                 filters=self.filters)
            # Iterate over this new group (group2)
            group = group2
    
    def tearDown(self):
        # Close the file
        if self.h5file.isopen:
            self.h5file.close()

        os.remove(self.file)

    #----------------------------------------

    def test00_checkFilters(self):
        "Checking inheritance of filters on trees (open file version)"

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_checkFilters..." % self.__class__.__name__

        # First level check
        if verbose:
            print "Test filter:", repr(self.filters)
            print "Filters in file:", repr(self.h5file.filters)

        assert repr(self.filters) == repr(self.h5file.filters)
        # The next nodes have to have the same filter properties as
        # self.filter
        nodelist = ['/table1', '/group0/earray1', '/group0']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                assert repr(self.filters) == repr(object._v_filters)
            else:
                assert repr(self.filters) == repr(object.filters)
                
        # Second and third level check
        group1 = self.h5file.root.group0.group1
        if verbose:
            print "Test filter:", repr(self.gfilters)
            print "Filters in file:", repr(group1._v_filters)

        repr(self.gfilters) == repr(group1._v_filters)
        # The next nodes have to have the same filter properties as
        # self.gfile
        nodelist = ['/group0/group1', '/group0/group1/earray1',
                    '/group0/group1/table1', '/group0/group1/group2/table1']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                assert repr(self.gfilters) == repr(object._v_filters)
            else:
                assert repr(self.gfilters) == repr(object.filters)

        # Fourth and fifth level check
        group3 = self.h5file.root.group0.group1.group2.group3
        if verbose:
            print "Test filter:", repr(self.filters)
            print "Filters in file:", repr(group3._v_filters)

        repr(self.filters) == repr(group3._v_filters)
        # The next nodes have to have the same filter properties as
        # self.filter
        nodelist = ['/group0/group1/group2/group3',
                    '/group0/group1/group2/group3/earray1',
                    '/group0/group1/group2/group3/table1',
                    '/group0/group1/group2/group3/group4']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                assert repr(self.filters) == repr(object._v_filters)
            else:
                assert repr(self.filters) == repr(object.filters)


        # Checking the special case for Arrays in which the compression
        # should always be the empty Filter()
        # The next nodes have to have the same filter properties as
        # Filter()
        nodelist = ['/array1',
                    '/group0/array1',
                    '/group0/group1/array1',
                    '/group0/group1/group2/array1',
                    '/group0/group1/group2/group3/array1']
        for node in nodelist:
            object = self.h5file.getNode(node)
            assert repr(Filters()) == repr(object.filters)

    def test01_checkFilters(self):
        "Checking inheritance of filters on trees (close file version)"

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_checkFilters..." % self.__class__.__name__

        # Close the file
        self.h5file.close()
        # And open it again
        self.h5file = openFile(self.file, "r")

        # First level check
        if verbose:
            print "Test filter:", repr(self.filters)
            print "Filters in file:", repr(self.h5file.filters)

        assert repr(self.filters) == repr(self.h5file.filters)
        # The next nodes have to have the same filter properties as
        # self.filter
        nodelist = ['/table1', '/group0/earray1', '/group0']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                assert repr(self.filters) == repr(object._v_filters)
            else:
                assert repr(self.filters) == repr(object.filters)

        # Second and third level check
        group1 = self.h5file.root.group0.group1
        if verbose:
            print "Test filter:", repr(self.gfilters)
            print "Filters in file:", repr(group1._v_filters)

        repr(self.gfilters) == repr(group1._v_filters)
        # The next nodes have to have the same filter properties as
        # self.gfile
        nodelist = ['/group0/group1', '/group0/group1/earray1',
                    '/group0/group1/table1', '/group0/group1/group2/table1']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                assert repr(self.gfilters) == repr(object._v_filters)
            else:
                assert repr(self.gfilters) == repr(object.filters)

        # Fourth and fifth level check
        group3 = self.h5file.root.group0.group1.group2.group3
        if verbose:
            print "Test filter:", repr(self.filters)
            print "Filters in file:", repr(group3._v_filters)

        repr(self.filters) == repr(group3._v_filters)
        # The next nodes have to have the same filter properties as
        # self.filter
        nodelist = ['/group0/group1/group2/group3',
                    '/group0/group1/group2/group3/earray1',
                    '/group0/group1/group2/group3/table1',
                    '/group0/group1/group2/group3/group4']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                assert repr(self.filters) == repr(object._v_filters)
            else:
                assert repr(self.filters) == repr(object.filters)

        # Checking the special case for Arrays in which the compression
        # should always be the empty Filter()
        # The next nodes have to have the same filter properties as
        # Filter()
        nodelist = ['/array1',
                    '/group0/array1',
                    '/group0/group1/array1',
                    '/group0/group1/group2/array1',
                    '/group0/group1/group2/group3/array1']
        for node in nodelist:
            object = self.h5file.getNode(node)
            assert repr(Filters()) == repr(object.filters)


class FiltersCase1(FiltersTreeTestCase):
    filters = Filters()
    gfilters = Filters(complevel=1)

class FiltersCase2(FiltersTreeTestCase):
    filters = Filters(complevel=1, complib="ucl")
    gfilters = Filters(complevel=1)

class FiltersCase3(FiltersTreeTestCase):
    filters = Filters(shuffle=1, complib="zlib")
    gfilters = Filters(complevel=1, shuffle=0, complib="lzo")

class FiltersCase4(FiltersTreeTestCase):
    filters = Filters(shuffle=1)
    gfilters = Filters(complevel=1, shuffle=0)

class FiltersCase5(FiltersTreeTestCase):
    filters = Filters(fletcher32=1)
    gfilters = Filters(complevel=1, shuffle=0)

	
#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(createTestCase))
        theSuite.addTest(unittest.makeSuite(createAttrNotCloseTestCase))
        theSuite.addTest(unittest.makeSuite(createAttrCloseTestCase))
        theSuite.addTest(unittest.makeSuite(FiltersCase1))
        theSuite.addTest(unittest.makeSuite(FiltersCase2))
        theSuite.addTest(unittest.makeSuite(FiltersCase3))
        theSuite.addTest(unittest.makeSuite(FiltersCase4))
        theSuite.addTest(unittest.makeSuite(FiltersCase5))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
