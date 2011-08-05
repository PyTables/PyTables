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
import tempfile
import warnings

from tables import *
# important objects to test
from tables import Group, Leaf, Table, Array
from tables.tests import common
from tables.parameters import MAX_COLUMNS

import tables

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

class Record(IsDescription):
    var1 = StringCol(itemsize=4)  # 4-character String
    var2 = IntCol()               # integer
    var3 = Int16Col()             # short integer
    var4 = FloatCol()             # double (double-precision)
    var5 = Float32Col()           # float  (single-precision)

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
        common.cleanup(self)

    #----------------------------------------

    def test00_isClass(self):
        """Testing table creation"""
        self.assertTrue(isinstance(self.table, Table))
        self.assertTrue(isinstance(self.array, Array))
        self.assertTrue(isinstance(self.array, Leaf))
        self.assertTrue(isinstance(self.group, Group))

    def test01_overwriteNode(self):
        """Checking protection against node overwriting"""

        try:
            self.array = self.fileh.createArray(self.root, 'anarray',
                                                [1], "Array title")
        except NodeError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NameError was catched!"
                print value
        else:
            self.fail("expected a NodeError")

    def test02_syntaxname(self):
        """Checking syntax in object tree names"""

        # Now, try to attach an array to the object tree with
        # a not allowed Python variable name
        warnings.filterwarnings("error", category=NaturalNameWarning)
        try:
            self.array = self.fileh.createArray(self.root, ' array',
                                                [1], "Array title")
        except NaturalNameWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NaturalNameWarning was catched!"
                print value
        else:
            self.fail("expected a NaturalNameWarning")

        # another name error
        try:
            self.array = self.fileh.createArray(self.root, '$array',
                                                [1], "Array title")
        except NaturalNameWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NaturalNameWarning was catched!"
                print value
        else:
            self.fail("expected a NaturalNameWarning")

        # Finally, test a reserved word
        try:
            self.array = self.fileh.createArray(self.root, 'for',
                                                [1], "Array title")
        except NaturalNameWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next NaturalNameWarning was catched!"
                print value
        else:
            self.fail("expected a NaturalNameWarning")
        # Reset the warning
        warnings.filterwarnings("default", category=NaturalNameWarning)

    def test03a_titleAttr(self):
        """Checking the self.title attr in nodes"""

        # Close the opened file to destroy the object tree
        self.fileh.close()
        # Open the file again to re-create the objects
        self.fileh = openFile(self.file,"r")

        # Now, test that self.title exists and is correct in all the nodes
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        self.assertEqual(self.fileh.root.atable.title, "Table title")
        self.assertEqual(self.fileh.root.anarray.title, "Array title")

    def test03b_titleLength(self):
        """Checking large title character length limit (1023)"""

        titlelength = 1023
        # Try to put a very long title on a group object
        group = self.fileh.createGroup(self.root, 'group',
                                       "t" * titlelength)
        self.assertEqual(group._v_title, "t" * titlelength)
        self.assertEqual(group._f_getAttr('TITLE'), "t" * titlelength)

        # Now, try with a table object
        table = self.fileh.createTable(self.root, 'table',
                                       Record, "t" * titlelength)
        self.assertEqual(table.title, "t" * titlelength)
        self.assertEqual(table.getAttr("TITLE"), "t" * titlelength)

        # Finally, try with an Array object
        arr = self.fileh.createArray(self.root, 'arr',
                                     [1], "t" * titlelength)
        self.assertEqual(arr.title, "t" * titlelength)
        self.assertEqual(arr.getAttr("TITLE"), "t" * titlelength)

    def test04_maxFields(self):
        "Checking a large number of fields in tables"

        # The number of fields for a table
        varnumber = MAX_COLUMNS

        varnames = []
        for i in range(varnumber):
            varnames.append('int%d' % i)

        # Build a dictionary with the types as values and varnames as keys
        recordDict = {}
        i = 0
        for varname in varnames:
            recordDict[varname] = Col.from_type("int32", dflt=1, pos=i)
            i += 1
        # Append this entry to indicate the alignment!
        recordDict['_v_align'] = "="
        table = self.fileh.createTable(self.root, 'table',
                                       recordDict, "MetaRecord instance")
        row = table.row
        listrows = []
        # Write 10 records
        for j in range(10):
            rowlist = []
            for i in range(len(table.colnames)):
                row[varnames[i]] = i*j
                rowlist.append(i*j)

            row.append()
            listrows.append(tuple(rowlist))

        # write data on disk
        table.flush()

        # Read all the data as a list
        listout = table.read().tolist()

        # Compare the input rowlist and output row list. They should
        # be equal.
        if common.verbose:
            print "Original row list:", listrows[-1]
            print "Retrieved row list:", listout[-1]
        self.assertEqual(listrows, listout)

    # The next limitation has been released. A warning is still there, though
    def test05_maxFieldsExceeded(self):
        "Checking an excess of the maximum number of fields in tables"

        # The number of fields for a table
        varnumber = MAX_COLUMNS + 1

        varnames = []
        for i in range(varnumber):
            varnames.append('int%d' % i)

        # Build a dictionary with the types as values and varnames as keys
        recordDict = {}
        i = 0
        for varname in varnames:
            recordDict[varname] = Col.from_type("int32", dflt=1)
            i += 1

        # Now, create a table with this record object
        # This way of creating node objects has been deprecated
        #table = Table(recordDict, "MetaRecord instance")

        # Attach the table to object tree
        warnings.filterwarnings("error", category=PerformanceWarning)
        # Here, a PerformanceWarning should be raised!
        try:
            table = self.fileh.createTable(self.root, 'table',
                                           recordDict, "MetaRecord instance")
        except PerformanceWarning:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next PerformanceWarning was catched!"
                print value
        else:
            self.fail("expected an PerformanceWarning")
        # Reset the warning
        warnings.filterwarnings("default", category=PerformanceWarning)

    # The next limitation has been released
    def _test06_maxColumnNameLengthExceeded(self):
        "Checking an excess (256) of the maximum length in column names"

        # Build a dictionary with the types as values and varnames as keys
        recordDict = {}
        recordDict["a"*255] = IntCol(dflt=1)
        recordDict["b"*256] = IntCol(dflt=1) # Should trigger a ValueError

        # Now, create a table with this record object
        # This way of creating node objects has been deprecated
        table = Table(recordDict, "MetaRecord instance")

        # Attach the table to object tree
        # Here, ValueError should be raised!
        try:
            table = self.fileh.createTable(self.root, 'table',
                                           recordDict, "MetaRecord instance")
        except ValueError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next ValueError was catched!"
                print value
        else:
            self.fail("expected an ValueError")

    def test06_noMaxColumnNameLength(self):
        "Checking unlimited length in column names"

        # Build a dictionary with the types as values and varnames as keys
        recordDict = {}
        recordDict["a"*255] = IntCol(dflt=1, pos=0)
        recordDict["b"*1024] = IntCol(dflt=1, pos=1) # Should work well

        # Attach the table to object tree
        # Here, IndexError should be raised!
        table = self.fileh.createTable(self.root, 'table',
                                       recordDict, "MetaRecord instance")
        self.assertEqual(table.colnames[0], "a"*255)
        self.assertEqual(table.colnames[1], "b"*1024)


class Record2(IsDescription):
    var1 = StringCol(itemsize=4)  # 4-character String
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
                                           StringAtom(itemsize=4), (0,),
                                           "col 1")
            ea2 = self.h5file.createEArray(group, 'earray2',
                                           Int16Atom(), (0,), "col 3")
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
        common.cleanup(self)

    #----------------------------------------

    def test00_checkFilters(self):
        "Checking inheritance of filters on trees (open file version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_checkFilters..." % self.__class__.__name__

        # First level check
        if common.verbose:
            print "Test filter:", repr(self.filters)
            print "Filters in file:", repr(self.h5file.filters)

        if self.filters == None:
            filters = Filters()
        else:
            filters = self.filters
        self.assertEqual(repr(filters), repr(self.h5file.filters))
        # The next nodes have to have the same filter properties as
        # self.filters
        nodelist = ['/table1', '/group0/earray1', '/group0']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                self.assertEqual(repr(filters), repr(object._v_filters))
            else:
                self.assertEqual(repr(filters), repr(object.filters))

        # Second and third level check
        group1 = self.h5file.root.group0.group1
        if self.gfilters == None:
            if self.filters == None:
                gfilters = Filters()
            else:
                gfilters = self.filters
        else:
            gfilters = self.gfilters
        if common.verbose:
            print "Test gfilter:", repr(gfilters)
            print "Filters in file:", repr(group1._v_filters)

        self.assertEqual(repr(gfilters), repr(group1._v_filters))
        # The next nodes have to have the same filter properties as
        # gfilters
        nodelist = ['/group0/group1', '/group0/group1/earray1',
                    '/group0/group1/table1', '/group0/group1/group2/table1']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                self.assertEqual(repr(gfilters), repr(object._v_filters))
            else:
                self.assertEqual(repr(gfilters), repr(object.filters))

        # Fourth and fifth level check
        if self.filters == None:
            # If None, the filters are inherited!
            if self.gfilters == None:
                filters = Filters()
            else:
                filters = self.gfilters
        else:
            filters = self.filters
        group3 = self.h5file.root.group0.group1.group2.group3
        if common.verbose:
            print "Test filter:", repr(filters)
            print "Filters in file:", repr(group3._v_filters)

        self.assertEqual(repr(filters), repr(group3._v_filters))
        # The next nodes have to have the same filter properties as
        # self.filter
        nodelist = ['/group0/group1/group2/group3',
                    '/group0/group1/group2/group3/earray1',
                    '/group0/group1/group2/group3/table1',
                    '/group0/group1/group2/group3/group4']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                self.assertEqual(repr(filters), repr(object._v_filters))
            else:
                self.assertEqual(repr(filters), repr(object.filters))


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
            self.assertEqual(repr(Filters()), repr(object.filters))

    def test01_checkFilters(self):
        "Checking inheritance of filters on trees (close file version)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_checkFilters..." % self.__class__.__name__

        # Close the file
        self.h5file.close()
        # And open it again
        self.h5file = openFile(self.file, "r")

        # First level check
        if self.filters == None:
            filters = Filters()
        else:
            filters = self.filters
        if common.verbose:
            print "Test filter:", repr(filters)
            print "Filters in file:", repr(self.h5file.filters)

        self.assertEqual(repr(filters), repr(self.h5file.filters))
        # The next nodes have to have the same filter properties as
        # self.filters
        nodelist = ['/table1', '/group0/earray1', '/group0']
        for node in nodelist:
            object_ = self.h5file.getNode(node)
            if isinstance(object_, Group):
                self.assertEqual(repr(filters), repr(object_._v_filters))
            else:
                self.assertEqual(repr(filters), repr(object_.filters))

        # Second and third level check
        group1 = self.h5file.root.group0.group1
        if self.gfilters == None:
            if self.filters == None:
                gfilters = Filters()
            else:
                gfilters = self.filters
        else:
            gfilters = self.gfilters
        if common.verbose:
            print "Test filter:", repr(gfilters)
            print "Filters in file:", repr(group1._v_filters)

        repr(gfilters) == repr(group1._v_filters)
        # The next nodes have to have the same filter properties as
        # gfilters
        nodelist = ['/group0/group1', '/group0/group1/earray1',
                    '/group0/group1/table1', '/group0/group1/group2/table1']
        for node in nodelist:
            object_ = self.h5file.getNode(node)
            if isinstance(object_, Group):
                self.assertEqual(repr(gfilters), repr(object_._v_filters))
            else:
                self.assertEqual(repr(gfilters), repr(object_.filters))

        # Fourth and fifth level check
        if self.filters == None:
            if self.gfilters == None:
                filters = Filters()
            else:
                filters = self.gfilters
        else:
            filters = self.filters
        group3 = self.h5file.root.group0.group1.group2.group3
        if common.verbose:
            print "Test filter:", repr(filters)
            print "Filters in file:", repr(group3._v_filters)

        repr(filters) == repr(group3._v_filters)
        # The next nodes have to have the same filter properties as
        # self.filters
        nodelist = ['/group0/group1/group2/group3',
                    '/group0/group1/group2/group3/earray1',
                    '/group0/group1/group2/group3/table1',
                    '/group0/group1/group2/group3/group4']
        for node in nodelist:
            object = self.h5file.getNode(node)
            if isinstance(object, Group):
                self.assertEqual(repr(filters), repr(object._v_filters))
            else:
                self.assertEqual(repr(filters), repr(object.filters))

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
            self.assertEqual(repr(Filters()), repr(object.filters))


class FiltersCase1(FiltersTreeTestCase):
    filters = Filters()
    gfilters = Filters(complevel=1)

class FiltersCase2(FiltersTreeTestCase):
    filters = Filters(complevel=1, complib="bzip2")
    gfilters = Filters(complevel=1)

class FiltersCase3(FiltersTreeTestCase):
    filters = Filters(shuffle=True, complib="zlib")
    gfilters = Filters(complevel=1, shuffle=False, complib="lzo")

class FiltersCase4(FiltersTreeTestCase):
    filters = Filters(shuffle=True)
    gfilters = Filters(complevel=1, shuffle=False)

class FiltersCase5(FiltersTreeTestCase):
    filters = Filters(fletcher32=True)
    gfilters = Filters(complevel=1, shuffle=False)

class FiltersCase6(FiltersTreeTestCase):
    filters = None
    gfilters = Filters(complevel=1, shuffle=False)

class FiltersCase7(FiltersTreeTestCase):
    filters = Filters(complevel=1)
    gfilters = None

class FiltersCase8(FiltersTreeTestCase):
    filters = None
    gfilters = None

class FiltersCase9(FiltersTreeTestCase):
    filters = Filters(shuffle=True, complib="zlib")
    gfilters = Filters(complevel=5, shuffle=True, complib="bzip2")

class FiltersCase10(FiltersTreeTestCase):
    filters = Filters(shuffle=False, complevel=1, complib="blosc")
    gfilters = Filters(complevel=5, shuffle=True, complib="blosc")

class CopyGroupTestCase(unittest.TestCase):
    title = "A title"
    nrows = 10

    def setUp(self):
        # Create a temporary file
        self.file = tempfile.mktemp(".h5")
        self.file2 = tempfile.mktemp(".h5")
        # Create the source file
        self.h5file = openFile(self.file, "w")
        # Create the destination
        self.h5file2 = openFile(self.file2, "w")
        self.populateFile()

    def populateFile(self):
        group = self.h5file.root
        # Add some user attrs:
        group._v_attrs.attr1 = "an string for root group"
        group._v_attrs.attr2 = 124
        # Create a tree
        for j in range(5):
            for i in range(2):
                # Create a new group (brother of group)
                group2 = self.h5file.createGroup(group, 'bgroup'+str(i),
                                                 filters=None)

                # Create a table
                table = self.h5file.createTable(group2, 'table1', Record2,
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

                # Add some user attrs:
                table.attrs.attr1 = "an string"
                table.attrs.attr2 = 234

                # Create a couple of arrays in each group
                var1List = [ x['var1'] for x in table.iterrows() ]
                var3List = [ x['var3'] for x in table.iterrows() ]

                self.h5file.createArray(group2, 'array1', var1List, "col 1")
                self.h5file.createArray(group2, 'array2', var3List, "col 3")

                # Create a couple of EArrays as well
                ea1 = self.h5file.createEArray(group2, 'earray1',
                                               StringAtom(itemsize=4), (0,),
                                               "col 1")
                ea2 = self.h5file.createEArray(group2, 'earray2',
                                               Int16Atom(), (0,), "col 3")
                # Add some user attrs:
                ea1.attrs.attr1 = "an string for earray"
                ea2.attrs.attr2 = 123
                # And fill them with some values
                ea1.append(var1List)
                ea2.append(var3List)

            # Create a new group (descendant of group)
            group3 = self.h5file.createGroup(group, 'group'+str(j),
                                             filters=None)
            # Iterate over this new group (group3)
            group = group3
            # Add some user attrs:
            group._v_attrs.attr1 = "an string for group"
            group._v_attrs.attr2 = 124

    def tearDown(self):
        # Close the file
        if self.h5file.isopen:
            self.h5file.close()
        if self.h5file2.isopen:
            self.h5file2.close()

        os.remove(self.file)
        os.remove(self.file2)
        common.cleanup(self)

    #----------------------------------------

    def test00_nonRecursive(self):
        "Checking non-recursive copy of a Group"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_nonRecursive..." % self.__class__.__name__


        # Copy a group non-recursively
        srcgroup = self.h5file.root.group0.group1
#         srcgroup._f_copyChildren(self.h5file2.root,
#                                recursive=False,
#                                filters=self.filters)
        self.h5file.copyChildren(srcgroup, self.h5file2.root,
                                 recursive=False, filters=self.filters)
        if self.close:
            # Close the destination file
            self.h5file2.close()
            # And open it again
            self.h5file2 = openFile(self.file2, "r")

        # Check that the copy has been done correctly
        dstgroup = self.h5file2.root
        nodelist1 = srcgroup._v_children.keys()
        nodelist2 = dstgroup._v_children.keys()
        # Sort the lists
        nodelist1.sort(); nodelist2.sort()
        if common.verbose:
            print "The origin node list -->", nodelist1
            print "The copied node list -->", nodelist2
        self.assertEqual(srcgroup._v_nchildren, dstgroup._v_nchildren)
        self.assertEqual(nodelist1, nodelist2)

    def test01_nonRecursiveAttrs(self):
        "Checking non-recursive copy of a Group (attributes copied)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_nonRecursiveAttrs..." % self.__class__.__name__

        # Copy a group non-recursively with attrs
        srcgroup = self.h5file.root.group0.group1
        srcgroup._f_copyChildren(self.h5file2.root,
                                 recursive=False,
                                 filters=self.filters,
                                 copyuserattrs = 1)
        if self.close:
            # Close the destination file
            self.h5file2.close()
            # And open it again
            self.h5file2 = openFile(self.file2, "r")

        # Check that the copy has been done correctly
        dstgroup = self.h5file2.root
        for srcnode in srcgroup:
            dstnode = getattr(dstgroup, srcnode._v_name)
            if isinstance(srcnode, Group):
                srcattrs = srcnode._v_attrs
                srcattrskeys = srcattrs._f_list("all")
                dstattrs = dstnode._v_attrs
                dstattrskeys = dstattrs._f_list("all")
            else:
                srcattrs = srcnode.attrs
                srcattrskeys = srcattrs._f_list("all")
                dstattrs = dstnode.attrs
                dstattrskeys = dstattrs._f_list("all")
            # Filters may differ, do not take into account
            if self.filters is not None:
                dstattrskeys.remove('FILTERS')
            # These lists should already be ordered
            if common.verbose:
                print "srcattrskeys for node %s: %s" %(srcnode._v_name,
                                                       srcattrskeys)
                print "dstattrskeys for node %s: %s" %(dstnode._v_name,
                                                       dstattrskeys)
            self.assertEqual(srcattrskeys, dstattrskeys)
            if common.verbose:
                print "The attrs names has been copied correctly"

            # Now, for the contents of attributes
            for srcattrname in srcattrskeys:
                srcattrvalue = str(getattr(srcattrs, srcattrname))
                dstattrvalue = str(getattr(dstattrs, srcattrname))
                self.assertEqual(srcattrvalue, dstattrvalue)
            if self.filters is not None:
                self.assertEqual(dstattrs.FILTERS, self.filters)

            if common.verbose:
                print "The attrs contents has been copied correctly"

    def test02_Recursive(self):
        "Checking recursive copy of a Group"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_Recursive..." % self.__class__.__name__


        # Create the destination node
        group = self.h5file2.root
        for groupname in self.dstnode.split("/"):
            if groupname:
                group = self.h5file2.createGroup(group, groupname)
        dstgroup = self.h5file2.getNode(self.dstnode)

        # Copy a group non-recursively
        srcgroup = self.h5file.getNode(self.srcnode)
        self.h5file.copyChildren(srcgroup, dstgroup,
                                 recursive=True,
                                 filters=self.filters)
        lenSrcGroup = len(srcgroup._v_pathname)
        if lenSrcGroup == 1:
            lenSrcGroup = 0  # Case where srcgroup == "/"
        if self.close:
            # Close the destination file
            self.h5file2.close()
            # And open it again
            self.h5file2 = openFile(self.file2, "r")
            dstgroup = self.h5file2.getNode(self.dstnode)

        # Check that the copy has been done correctly
        lenDstGroup = len(dstgroup._v_pathname)
        if lenDstGroup == 1:
            lenDstGroup = 0  # Case where dstgroup == "/"
        first = 1
        nodelist1 = []
        for node in srcgroup._f_walkNodes():
            if first:
                # skip the first group
                first = 0
                continue
            nodelist1.append(node._v_pathname[lenSrcGroup:])

        first = 1
        nodelist2 = []
        for node in dstgroup._f_walkNodes():
            if first:
                # skip the first group
                first = 0
                continue
            nodelist2.append(node._v_pathname[lenDstGroup:])

        if common.verbose:
            print "The origin node list -->", nodelist1
            print "The copied node list -->", nodelist2
        self.assertEqual(nodelist1, nodelist2)

    def test03_RecursiveFilters(self):
        "Checking recursive copy of a Group (cheking Filters)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_RecursiveFilters..." % self.__class__.__name__

        # Create the destination node
        group = self.h5file2.root
        for groupname in self.dstnode.split("/"):
            if groupname:
                group = self.h5file2.createGroup(group, groupname)
        dstgroup = self.h5file2.getNode(self.dstnode)

        # Copy a group non-recursively
        srcgroup = self.h5file.getNode(self.srcnode)
        srcgroup._f_copyChildren(dstgroup,
                                 recursive=True,
                                 filters=self.filters)
        lenSrcGroup = len(srcgroup._v_pathname)
        if lenSrcGroup == 1:
            lenSrcGroup = 0  # Case where srcgroup == "/"
        if self.close:
            # Close the destination file
            self.h5file2.close()
            # And open it again
            self.h5file2 = openFile(self.file2, "r")
            dstgroup = self.h5file2.getNode(self.dstnode)

        # Check that the copy has been done correctly
        lenDstGroup = len(dstgroup._v_pathname)
        if lenDstGroup == 1:
            lenDstGroup = 0  # Case where dstgroup == "/"
        first = 1
        nodelist1 = {}
        for node in srcgroup._f_walkNodes():
            if first:
                # skip the first group
                first = 0
                continue
            nodelist1[node._v_name] = node._v_pathname[lenSrcGroup:]

        first = 1
        for node in dstgroup._f_walkNodes():
            if first:
                # skip the first group
                first = 0
                continue
            if isinstance(node, Group):
                repr(node._v_filters) == repr(nodelist1[node._v_name])
            else:
                repr(node.filters) == repr(nodelist1[node._v_name])


class CopyGroupCase1(CopyGroupTestCase):
    close = 0
    filters = None
    srcnode = '/group0/group1'
    dstnode = '/'

class CopyGroupCase2(CopyGroupTestCase):
    close = 1
    filters = None
    srcnode = '/group0/group1'
    dstnode = '/'

class CopyGroupCase3(CopyGroupTestCase):
    close = 0
    filters = None
    srcnode = '/group0'
    dstnode = '/group2/group3'

class CopyGroupCase4(CopyGroupTestCase):
    close = 1
    filters = Filters(complevel=1)
    srcnode = '/group0'
    dstnode = '/group2/group3'

class CopyGroupCase5(CopyGroupTestCase):
    close = 0
    filters = Filters()
    srcnode = '/'
    dstnode = '/group2/group3'

class CopyGroupCase6(CopyGroupTestCase):
    close = 1
    filters = Filters(fletcher32=True)
    srcnode = '/group0'
    dstnode = '/group2/group3'

class CopyGroupCase7(CopyGroupTestCase):
    close = 0
    filters = Filters(complevel=1, shuffle=False)
    srcnode = '/'
    dstnode = '/'

class CopyGroupCase8(CopyGroupTestCase):
    close = 1
    filters = Filters(complevel=1, complib="lzo")
    srcnode = '/'
    dstnode = '/'

class CopyFileTestCase(unittest.TestCase):
    title = "A title"
    nrows = 10

    def setUp(self):
        # Create a temporary file
        self.file = tempfile.mktemp(".h5")
        self.file2 = tempfile.mktemp(".h5")
        # Create the source file
        self.h5file = openFile(self.file, "w")
        self.populateFile()

    def populateFile(self):
        group = self.h5file.root
        # Add some user attrs:
        group._v_attrs.attr1 = "an string for root group"
        group._v_attrs.attr2 = 124
        # Create a tree
        for j in range(5):
            for i in range(2):
                # Create a new group (brother of group)
                group2 = self.h5file.createGroup(group, 'bgroup'+str(i),
                                                 filters=None)

                # Create a table
                table = self.h5file.createTable(group2, 'table1', Record2,
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

                # Add some user attrs:
                table.attrs.attr1 = "an string"
                table.attrs.attr2 = 234

                # Create a couple of arrays in each group
                var1List = [ x['var1'] for x in table.iterrows() ]
                var3List = [ x['var3'] for x in table.iterrows() ]

                self.h5file.createArray(group2, 'array1', var1List, "col 1")
                self.h5file.createArray(group2, 'array2', var3List, "col 3")

                # Create a couple of EArrays as well
                ea1 = self.h5file.createEArray(group2, 'earray1',
                                               StringAtom(itemsize=4), (0,),
                                               "col 1")
                ea2 = self.h5file.createEArray(group2, 'earray2',
                                               Int16Atom(), (0,),
                                               "col 3")
                # Add some user attrs:
                ea1.attrs.attr1 = "an string for earray"
                ea2.attrs.attr2 = 123
                # And fill them with some values
                ea1.append(var1List)
                ea2.append(var3List)

            # Create a new group (descendant of group)
            group3 = self.h5file.createGroup(group, 'group'+str(j),
                                             filters=None)
            # Iterate over this new group (group3)
            group = group3
            # Add some user attrs:
            group._v_attrs.attr1 = "an string for group"
            group._v_attrs.attr2 = 124

    def tearDown(self):
        # Close the file
        if self.h5file.isopen:
            self.h5file.close()
        if hasattr(self, 'h5file2') and self.h5file2.isopen:
            self.h5file2.close()

        os.remove(self.file)
        if hasattr(self, 'file2') and os.path.exists(self.file2):
            os.remove(self.file2)
        common.cleanup(self)

    #----------------------------------------

    def test00_overwrite(self):
        "Checking copy of a File (overwriting file)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_overwrite..." % self.__class__.__name__

        # Create a temporary file
        file2h = open(self.file2, "w")
        file2h.close()
        # Copy the file to the destination
        self.h5file.copyFile(self.file2, title=self.title,
                             overwrite = 1,
                             copyuserattrs = 0,
                             filters = None)

        # Close the original file, if needed
        if self.close:
            self.h5file.close()
            # re-open it
            self.h5file = openFile(self.file, "r")

        # ...and open the destination file
        self.h5file2 = openFile(self.file2, "r")

        # Check that the copy has been done correctly
        srcgroup = self.h5file.root
        dstgroup = self.h5file2.root
        nodelist1 = srcgroup._v_children.keys()
        nodelist2 = dstgroup._v_children.keys()
        # Sort the lists
        nodelist1.sort(); nodelist2.sort()
        if common.verbose:
            print "The origin node list -->", nodelist1
            print "The copied node list -->", nodelist2
        self.assertEqual(srcgroup._v_nchildren, dstgroup._v_nchildren)
        self.assertEqual(nodelist1, nodelist2)
        self.assertEqual(self.h5file2.title, self.title)

    def test00a_srcdstequal(self):
        "Checking copy of a File (srcfile == dstfile)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00a_srcdstequal..." % self.__class__.__name__

        # Copy the file to the destination
        self.assertRaises(IOError, self.h5file.copyFile, self.h5file.filename)

    def test00b_firstclass(self):
        "Checking copy of a File (first-class function)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00b_firstclass..." % self.__class__.__name__

        # Close the temporary file
        self.h5file.close()

        # Copy the file to the destination
        copyFile(self.file, self.file2, title=self.title,
                 copyuserattrs = 0, filters = None, overwrite = 1)

        # ...and open the source and destination file
        self.h5file = openFile(self.file, "r")
        self.h5file2 = openFile(self.file2, "r")

        # Check that the copy has been done correctly
        srcgroup = self.h5file.root
        dstgroup = self.h5file2.root
        nodelist1 = srcgroup._v_children.keys()
        nodelist2 = dstgroup._v_children.keys()
        # Sort the lists
        nodelist1.sort(); nodelist2.sort()
        if common.verbose:
            print "The origin node list -->", nodelist1
            print "The copied node list -->", nodelist2
        self.assertEqual(srcgroup._v_nchildren, dstgroup._v_nchildren)
        self.assertEqual(nodelist1, nodelist2)
        self.assertEqual(self.h5file2.title, self.title)

    def test01_copy(self):
        "Checking copy of a File (attributes not copied)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copy..." % self.__class__.__name__

        # Copy the file to the destination
        self.h5file.copyFile(self.file2, title=self.title,
                             copyuserattrs = 0,
                             filters = self.filters)

        # Close the original file, if needed
        if self.close:
            self.h5file.close()
            # re-open it
            self.h5file = openFile(self.file, "r")

        # ...and open the destination file
        self.h5file2 = openFile(self.file2, "r")

        # Check that the copy has been done correctly
        srcgroup = self.h5file.root
        dstgroup = self.h5file2.root
        nodelist1 = srcgroup._v_children.keys()
        nodelist2 = dstgroup._v_children.keys()
        # Sort the lists
        nodelist1.sort(); nodelist2.sort()
        if common.verbose:
            print "The origin node list -->", nodelist1
            print "The copied node list -->", nodelist2
        self.assertEqual(srcgroup._v_nchildren, dstgroup._v_nchildren)
        self.assertEqual(nodelist1, nodelist2)
        #print "_v_attrnames-->", self.h5file2.root._v_attrs._v_attrnames
        #print "--> <%s,%s>" % (self.h5file2.title, self.title)
        self.assertEqual(self.h5file2.title, self.title)

        # Check that user attributes has not been copied
        for srcnode in srcgroup:
            dstnode = getattr(dstgroup, srcnode._v_name)
            srcattrs = srcnode._v_attrs
            srcattrskeys = srcattrs._f_list("sys")
            dstattrs = dstnode._v_attrs
            dstattrskeys = dstattrs._f_list("all")
            # Filters may differ, do not take into account
            if self.filters is not None:
                dstattrskeys.remove('FILTERS')
            # These lists should already be ordered
            if common.verbose:
                print "srcattrskeys for node %s: %s" %(srcnode._v_name,
                                                       srcattrskeys)
                print "dstattrskeys for node %s: %s" %(dstnode._v_name,
                                                       dstattrskeys)
            self.assertEqual(srcattrskeys, dstattrskeys)
            if common.verbose:
                print "The attrs names has been copied correctly"

            # Now, for the contents of attributes
            for srcattrname in srcattrskeys:
                srcattrvalue = str(getattr(srcattrs, srcattrname))
                dstattrvalue = str(getattr(dstattrs, srcattrname))
                self.assertEqual(srcattrvalue, dstattrvalue)
            if self.filters is not None:
                self.assertEqual(dstattrs.FILTERS, self.filters)

            if common.verbose:
                print "The attrs contents has been copied correctly"


    def test02_Attrs(self):
        "Checking copy of a File (attributes copied)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_Attrs..." % self.__class__.__name__


        # Copy the file to the destination
        self.h5file.copyFile(self.file2, title=self.title,
                             copyuserattrs = 1,
                             filters = self.filters)

        # Close the original file, if needed
        if self.close:
            self.h5file.close()
            # re-open it
            self.h5file = openFile(self.file, "r")

        # ...and open the destination file
        self.h5file2 = openFile(self.file2, "r")

        # Check that the copy has been done correctly
        srcgroup = self.h5file.root
        dstgroup = self.h5file2.root
        for srcnode in srcgroup:
            dstnode = getattr(dstgroup, srcnode._v_name)
            srcattrs = srcnode._v_attrs
            srcattrskeys = srcattrs._f_list("all")
            dstattrs = dstnode._v_attrs
            dstattrskeys = dstattrs._f_list("all")
            # These lists should already be ordered
            if common.verbose:
                print "srcattrskeys for node %s: %s" %(srcnode._v_name,
                                                       srcattrskeys)
                print "dstattrskeys for node %s: %s" %(dstnode._v_name,
                                                       dstattrskeys)
            # Filters may differ, do not take into account
            if self.filters is not None:
                dstattrskeys.remove('FILTERS')
            self.assertEqual(srcattrskeys, dstattrskeys)
            if common.verbose:
                print "The attrs names has been copied correctly"

            # Now, for the contents of attributes
            for srcattrname in srcattrskeys:
                srcattrvalue = str(getattr(srcattrs, srcattrname))
                dstattrvalue = str(getattr(dstattrs, srcattrname))
                self.assertEqual(srcattrvalue, dstattrvalue)
            if self.filters is not None:
                self.assertEqual(dstattrs.FILTERS, self.filters)

            if common.verbose:
                print "The attrs contents has been copied correctly"

class CopyFileCase1(CopyFileTestCase):
    close = 0
    title = "A new title"
    filters = None

class CopyFileCase2(CopyFileTestCase):
    close = 1
    title = "A new title"
    filters = None

class CopyFileCase3(CopyFileTestCase):
    close = 0
    title = "A new title"
    filters = Filters(complevel=1)

class CopyFileCase4(CopyFileTestCase):
    close = 1
    title = "A new title"
    filters = Filters(complevel=1)

class CopyFileCase5(CopyFileTestCase):
    close = 0
    title = "A new title"
    filters = Filters(fletcher32=True)

class CopyFileCase6(CopyFileTestCase):
    close = 1
    title = "A new title"
    filters = Filters(fletcher32=True)

class CopyFileCase7(CopyFileTestCase):
    close = 0
    title = "A new title"
    filters = Filters(complevel=1, complib="lzo")

class CopyFileCase8(CopyFileTestCase):
    close = 1
    title = "A new title"
    filters = Filters(complevel=1, complib="lzo")

class CopyFileCase10(unittest.TestCase):

    def test01_notoverwrite(self):
        "Checking copy of a File (checking not overwriting)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_notoverwrite..." % self.__class__.__name__


        # Create two empty files:
        file = tempfile.mktemp(".h5")
        fileh = openFile(file, "w")
        file2 = tempfile.mktemp(".h5")
        fileh2 = openFile(file2, "w")
        fileh2.close()  # close the second one
        # Copy the first into the second
        try:
            fileh.copyFile(file2, overwrite=False)
        except IOError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next IOError was catched!"
                print value
        else:
            self.fail("expected a IOError")


        # Delete files
        fileh.close()
        os.remove(file)
        os.remove(file2)


class GroupFiltersTestCase(common.TempFileMixin, common.PyTablesTestCase):
    filters = tables.Filters(complevel=4)  # something non-default

    def setUp(self):
        super(GroupFiltersTestCase, self).setUp()

        atom, shape = tables.IntAtom(), (1, 1)
        createGroup = self.h5file.createGroup
        createCArray = self.h5file.createCArray

        createGroup('/', 'implicit_no')
        createGroup('/implicit_no', 'implicit_no')
        createCArray( '/implicit_no/implicit_no', 'implicit_no',
                      atom=atom, shape=shape )
        createCArray( '/implicit_no/implicit_no', 'explicit_no',
                      atom=atom, shape=shape, filters=tables.Filters() )
        createCArray( '/implicit_no/implicit_no', 'explicit_yes',
                      atom=atom, shape=shape, filters=self.filters )

        createGroup('/', 'explicit_yes', filters=self.filters)
        createGroup('/explicit_yes', 'implicit_yes')
        createCArray( '/explicit_yes/implicit_yes', 'implicit_yes',
                      atom=atom, shape=shape )
        createCArray( '/explicit_yes/implicit_yes', 'explicit_yes',
                      atom=atom, shape=shape, filters=self.filters )
        createCArray( '/explicit_yes/implicit_yes', 'explicit_no',
                      atom=atom, shape=shape, filters=tables.Filters() )

    def _check_filters(self, h5file, filters=None):
        for node in h5file:
            # Get node filters.
            if hasattr(node, 'filters'):
                node_filters = node.filters
            else:
                node_filters = node._v_filters
            # Compare to given filters.
            if filters is not None:
                self.assertEqual(node_filters, filters)
                return
            # Guess filters to compare to by node name.
            if node._v_name.endswith('_no'):
                self.assertEqual(
                    node_filters, tables.Filters(),
                    "node ``%s`` should have no filters" % node._v_pathname )
            elif node._v_name.endswith('_yes'):
                self.assertEqual(
                    node_filters, self.filters,
                    "node ``%s`` should have filters" % node._v_pathname )

    def test00_propagate(self):
        """Filters propagating to children."""
        self._check_filters(self.h5file)

    def _test_copyFile(self, filters=None):
        copyfname = tempfile.mktemp(suffix='.h5')
        try:
            self.h5file.copyFile(copyfname, filters=filters)
            try:
                copyf = tables.openFile(copyfname)
                self._check_filters(copyf, filters=filters)
            finally:
                copyf.close()
        finally:
            os.remove(copyfname)

    def test01_copyFile(self):
        """Keeping filters when copying a file."""
        self._test_copyFile()

    def test02_copyFile_override(self):
        """Overriding filters when copying a file."""
        self._test_copyFile(self.filters)

    def _test_change(self, pathname, change_filters, new_filters):
        group = self.h5file.getNode(pathname)
        # Check expected current filters.
        old_filters = tables.Filters()
        if pathname.endswith('_yes'):
            old_filters = self.filters
        self.assertEqual(group._v_filters, old_filters)
        # Change filters.
        change_filters(group)
        self.assertEqual(group._v_filters, new_filters)
        # Get and check changed filters.
        if self._reopen():
            group = self.h5file.getNode(pathname)
        self.assertEqual(group._v_filters, new_filters)

    def test03_change(self):
        """Changing the filters of a group."""
        def set_filters(group):
            group._v_filters = self.filters
        self._test_change('/', set_filters, self.filters)

    def test04_delete(self):
        """Deleting the filters of a group."""
        def del_filters(group):
            del group._v_filters
        self._test_change('/explicit_yes', del_filters, tables.Filters())


class setBloscMaxThreads(common.TempFileMixin, common.PyTablesTestCase):
    filters = tables.Filters(complevel=4, complib="blosc")

    def test00(self):
        """Checking setBloscMaxThreads()"""
        nthreads_old = tables.setBloscMaxThreads(4)
        if common.verbose:
            print "Previous max threads:", nthreads_old
            print "Should be:", self.h5file.params['MAX_THREADS']
        self.assertEqual(nthreads_old, self.h5file.params['MAX_THREADS'])
        self.h5file.createCArray('/', 'some_array',
                                 atom=tables.Int32Atom(), shape=(3,3),
                                 filters = self.filters)
        nthreads_old = tables.setBloscMaxThreads(1)
        if common.verbose:
            print "Previous max threads:", nthreads_old
            print "Should be:", 4
        self.assertEqual(nthreads_old, 4)

    def test01(self):
        """Checking setBloscMaxThreads() (re-open)"""
        nthreads_old = tables.setBloscMaxThreads(4)
        self.h5file.createCArray('/', 'some_array',
                                 atom=tables.Int32Atom(), shape=(3,3),
                                 filters = self.filters)
        self._reopen()
        nthreads_old = tables.setBloscMaxThreads(4)
        if common.verbose:
            print "Previous max threads:", nthreads_old
            print "Should be:", self.h5file.params['MAX_THREADS']
        self.assertEqual(nthreads_old, self.h5file.params['MAX_THREADS'])



#----------------------------------------------------------------------

def suite():
    import doctest
    import tables.atom

    theSuite = unittest.TestSuite()
    niter = 1
    #common.heavy = 1 # Uncomment this only for testing purposes!

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(FiltersCase1))
        theSuite.addTest(unittest.makeSuite(FiltersCase2))
        theSuite.addTest(unittest.makeSuite(FiltersCase10))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase1))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase2))
        theSuite.addTest(unittest.makeSuite(CopyFileCase1))
        theSuite.addTest(unittest.makeSuite(CopyFileCase2))
        theSuite.addTest(unittest.makeSuite(GroupFiltersTestCase))
        theSuite.addTest(unittest.makeSuite(setBloscMaxThreads))
        theSuite.addTest(doctest.DocTestSuite(tables.filters))
    if common.heavy:
        theSuite.addTest(unittest.makeSuite(createTestCase))
        theSuite.addTest(unittest.makeSuite(FiltersCase3))
        theSuite.addTest(unittest.makeSuite(FiltersCase4))
        theSuite.addTest(unittest.makeSuite(FiltersCase5))
        theSuite.addTest(unittest.makeSuite(FiltersCase6))
        theSuite.addTest(unittest.makeSuite(FiltersCase7))
        theSuite.addTest(unittest.makeSuite(FiltersCase8))
        theSuite.addTest(unittest.makeSuite(FiltersCase9))
        theSuite.addTest(unittest.makeSuite(CopyFileCase3))
        theSuite.addTest(unittest.makeSuite(CopyFileCase4))
        theSuite.addTest(unittest.makeSuite(CopyFileCase5))
        theSuite.addTest(unittest.makeSuite(CopyFileCase6))
        theSuite.addTest(unittest.makeSuite(CopyFileCase7))
        theSuite.addTest(unittest.makeSuite(CopyFileCase8))
        theSuite.addTest(unittest.makeSuite(CopyFileCase10))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase3))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase4))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase5))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase6))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase7))
        theSuite.addTest(unittest.makeSuite(CopyGroupCase8))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
