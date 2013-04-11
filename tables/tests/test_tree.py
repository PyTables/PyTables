# -*- coding: utf-8 -*-

import sys
import warnings
import unittest
import os
import tempfile

from tables import *
# Next imports are only necessary for this test suite
from tables import Group, Leaf, Table, Array

from tables.tests import common

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

# Test Record class
class Record(IsDescription):
    var1 = StringCol(itemsize=4) # 4-character String
    var2 = IntCol()              # integer
    var3 = Int16Col()            # short integer
    var4 = FloatCol()            # double (double-precision)
    var5 = Float32Col()          # float  (single-precision)

class TreeTestCase(unittest.TestCase):
    mode  = "w"
    title = "This is the table title"
    expectedrows = 10
    appendrows = 5

    def setUp(self):
        # Create a temporary file
        self.file = tempfile.mktemp(".h5")
        # Create an instance of HDF5 Table
        self.h5file = open_file(self.file, self.mode, self.title)
        self.populateFile()
        self.h5file.close()

    def populateFile(self):
        group = self.h5file.root
        maxshort = 1 << 15
        #maxint   = 2147483647   # (2 ** 31 - 1)
        for j in range(3):
            # Create a table
            table = self.h5file.create_table(group, 'table'+str(j), Record,
                                        title = self.title,
                                        filters = None,
                                        expectedrows = self.expectedrows)
            # Get the record object associated with the new table
            d = table.row
            # Fill the table
            for i in xrange(self.expectedrows):
                d['var1'] = '%04d' % (self.expectedrows - i)
                d['var2'] = i
                d['var3'] = i % maxshort
                d['var4'] = float(i)
                d['var5'] = float(i)
                d.append()      # This injects the Record values
            # Flush the buffer for this table
            table.flush()

            # Create a couple of arrays in each group
            var1List = [ x['var1'] for x in table.iterrows() ]
            var4List = [ x['var4'] for x in table.iterrows() ]

            self.h5file.create_array(group, 'var1', var1List, "1")
            self.h5file.create_array(group, 'var4', var4List, "4")

            # Create a new group (descendant of group)
            group2 = self.h5file.create_group(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2

    def tearDown(self):
        # Close the file
        if self.h5file.isopen:
            self.h5file.close()

        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test00_getNode(self):
        "Checking the File.get_node() with string node names"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_getNode..." % self.__class__.__name__

        self.h5file = open_file(self.file, "r")
        nodelist = ['/', '/table0', '/group0/var1', '/group0/group1/var4']
        nodenames = []
        for node in nodelist:
            object = self.h5file.get_node(node)
            nodenames.append(object._v_pathname)

        self.assertEqual(nodenames, nodelist)
        if common.verbose:
            print "get_node(pathname) test passed"
        nodegroups = ['/', '/group0', '/group0/group1', '/group0/group1/group2']
        nodenames = ['var1', 'var4']
        nodepaths = []
        for group in nodegroups:
            for name in nodenames:
                try:
                    object = self.h5file.get_node(group, name)
                except LookupError:
                    pass
                else:
                    nodepaths.append(object._v_pathname)

        self.assertEqual(nodepaths,
                         ['/var1', '/var4',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "get_node(groupname, name) test passed"
        nodelist = ['/', '/group0', '/group0/group1', '/group0/group1/group2',
                    '/table0']
        nodenames = []
        groupobjects = []
        #warnings.filterwarnings("error", category=UserWarning)
        for node in nodelist:
            try:
                object = self.h5file.get_node(node, classname = 'Group')
            except LookupError:
                if common.verbose:
                    (type, value, traceback) = sys.exc_info()
                    print "\nGreat!, the next LookupError was catched!"
                    print value
            else:
                nodenames.append(object._v_pathname)
                groupobjects.append(object)

        self.assertEqual(nodenames,
                         ['/', '/group0', '/group0/group1',
                          '/group0/group1/group2'])
        if common.verbose:
            print "get_node(groupname, classname='Group') test passed"

        # Reset the warning
        #warnings.filterwarnings("default", category=UserWarning)

        nodenames = ['var1', 'var4']
        nodearrays = []
        for group in groupobjects:
            for name in nodenames:
                try:
                    object = self.h5file.get_node(group, name, 'Array')
                except:
                    pass
                else:
                    nodearrays.append(object._v_pathname)

        self.assertEqual(nodearrays,
                         ['/var1', '/var4',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/var1', '/group0/group1/var4'])
        if common.verbose:
            print "get_node(groupobject, name, classname='Array') test passed"

    def test01_getNodeClass(self):
        "Checking the File.get_node() with instances"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_getNodeClass..." % self.__class__.__name__

        self.h5file = open_file(self.file, "r")
        # This tree ways of get_node usage should return a table instance
        table = self.h5file.get_node("/group0/table1")
        self.assertTrue(isinstance(table, Table))
        table = self.h5file.get_node("/group0", "table1")
        self.assertTrue(isinstance(table, Table))
        table = self.h5file.get_node(self.h5file.root.group0, "table1")
        self.assertTrue(isinstance(table, Table))

        # This should return an array instance
        arr = self.h5file.get_node("/group0/var1")
        self.assertTrue(isinstance(arr, Array))
        self.assertTrue(isinstance(arr, Leaf))

        # And this a Group
        group = self.h5file.get_node("/group0", "group1", "Group")
        self.assertTrue(isinstance(group, Group))

    def test02_listNodes(self):
        "Checking the File.list_nodes() method"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_listNodes..." % self.__class__.__name__

        # Made the warnings to raise an error
        #warnings.filterwarnings("error", category=UserWarning)
        self.h5file = open_file(self.file, "r")

        self.assertRaises(TypeError,
                          self.h5file.list_nodes, '/', 'NoSuchClass')

        nodelist = ['/', '/group0', '/group0/table1', '/group0/group1/group2',
                    '/var1']
        nodenames = []
        objects = []
        for node in nodelist:
            try:
                objectlist = self.h5file.list_nodes(node)
            except:
                pass
            else:
                objects.extend(objectlist)
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0', '/table0', '/var1', '/var4',
                          '/group0/group1', '/group0/table1',
                          '/group0/var1', '/group0/var4'])
        if common.verbose:
            print "list_nodes(pathname) test passed"

        nodenames = []
        for node in objects:
            try:
                objectlist = self.h5file.list_nodes(node)
            except:
                pass
            else:
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0/group1', '/group0/table1',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/group2', '/group0/group1/table2',
                          '/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "list_nodes(groupobject) test passed"

        nodenames = []
        for node in objects:
            try:
                objectlist = self.h5file.list_nodes(node, 'Leaf')
            except TypeError:
                if common.verbose:
                    (type, value, traceback) = sys.exc_info()
                    print "\nGreat!, the next TypeError was catched!"
                    print value
            else:
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0/table1',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/table2',
                          '/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "list_nodes(groupobject, classname = 'Leaf') test passed"

        nodenames = []
        for node in objects:
            try:
                objectlist = self.h5file.list_nodes(node, 'Table')
            except TypeError:
                if common.verbose:
                    (type, value, traceback) = sys.exc_info()
                    print "\nGreat!, the next TypeError was catched!"
                    print value
            else:
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0/table1', '/group0/group1/table2'])

        if common.verbose:
            print "list_nodes(groupobject, classname = 'Table') test passed"

        # Reset the warning
        #warnings.filterwarnings("default", category=UserWarning)

    def test02b_iterNodes(self):
        "Checking the File.iter_nodes() method"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02b_iterNodes..." % self.__class__.__name__

        self.h5file = open_file(self.file, "r")

        self.assertRaises(TypeError,
                          self.h5file.list_nodes, '/', 'NoSuchClass')

        nodelist = ['/', '/group0', '/group0/table1', '/group0/group1/group2',
                    '/var1']
        nodenames = []
        objects = []
        for node in nodelist:
            try:
                objectlist = [o for o in self.h5file.iter_nodes(node)]
            except:
                pass
            else:
                objects.extend(objectlist)
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0', '/table0', '/var1', '/var4',
                          '/group0/group1', '/group0/table1',
                          '/group0/var1', '/group0/var4'])
        if common.verbose:
            print "iter_nodes(pathname) test passed"

        nodenames = []
        for node in objects:
            try:
                objectlist = [o for o in self.h5file.iter_nodes(node)]
            except:
                pass
            else:
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0/group1', '/group0/table1',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/group2', '/group0/group1/table2',
                          '/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "iter_nodes(groupobject) test passed"

        nodenames = []
        for node in objects:
            try:
                objectlist = [o for o in self.h5file.iter_nodes(node, 'Leaf')]
            except TypeError:
                if common.verbose:
                    (type, value, traceback) = sys.exc_info()
                    print "\nGreat!, the next TypeError was catched!"
                    print value
            else:
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0/table1',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/table2',
                          '/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "iter_nodes(groupobject, classname = 'Leaf') test passed"

        nodenames = []
        for node in objects:
            try:
                objectlist = [o for o in self.h5file.iter_nodes(node, 'Table')]
            except TypeError:
                if common.verbose:
                    (type, value, traceback) = sys.exc_info()
                    print "\nGreat!, the next TypeError was catched!"
                    print value
            else:
                for object in objectlist:
                    nodenames.append(object._v_pathname)

        self.assertEqual(nodenames,
                         ['/group0/table1', '/group0/group1/table2'])

        if common.verbose:
            print "iter_nodes(groupobject, classname = 'Table') test passed"

        # Reset the warning
        #warnings.filterwarnings("default", category=UserWarning)

    def test03_TraverseTree(self):
        "Checking the File.walk_groups() method"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_TraverseTree..." % self.__class__.__name__

        self.h5file = open_file(self.file, "r")
        groups = []
        tables = []
        arrays = []
        for group in self.h5file.walk_groups():
            groups.append(group._v_pathname)
            for table in self.h5file.list_nodes(group, 'Table'):
                tables.append(table._v_pathname)
            for arr in self.h5file.list_nodes(group, 'Array'):
                arrays.append(arr._v_pathname)

        self.assertEqual(groups,
                         ["/", "/group0", "/group0/group1",
                          "/group0/group1/group2"])

        self.assertEqual(tables,
                         ["/table0", "/group0/table1", "/group0/group1/table2"])

        self.assertEqual(arrays,
                         ['/var1', '/var4',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/var1', '/group0/group1/var4'])
        if common.verbose:
            print "walk_groups() test passed"

        groups = []
        tables = []
        arrays = []
        for group in self.h5file.walk_groups("/group0/group1"):
            groups.append(group._v_pathname)
            for table in self.h5file.list_nodes(group, 'Table'):
                tables.append(table._v_pathname)
            for arr in self.h5file.list_nodes(group, 'Array'):
                arrays.append(arr._v_pathname)

        self.assertEqual(groups,
                         ["/group0/group1", "/group0/group1/group2"])

        self.assertEqual(tables, ["/group0/group1/table2"])

        self.assertEqual(arrays, ['/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "walk_groups(pathname) test passed"

    def test04_walkNodes(self):
        "Checking File.walk_nodes"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_walkNodes..." % self.__class__.__name__

        self.h5file = open_file(self.file, "r")

        self.assertRaises(TypeError,
                          self.h5file.walk_nodes('/', 'NoSuchClass').next)

        groups = []
        tables = []
        tables2 = []
        arrays = []
        for group in self.h5file.walk_nodes(classname="Group"):
            groups.append(group._v_pathname)
            for table in group._f_iter_nodes(classname='Table'):
                tables.append(table._v_pathname)
        # Test the recursivity
        for table in self.h5file.root._f_walknodes('Table'):
            tables2.append(table._v_pathname)

        for arr in self.h5file.walk_nodes(classname='Array'):
            arrays.append(arr._v_pathname)

        self.assertEqual(groups,
                         ["/", "/group0", "/group0/group1",
                          "/group0/group1/group2"])
        self.assertEqual(tables,
                         ["/table0", "/group0/table1",
                          "/group0/group1/table2"])
        self.assertEqual(tables2,
                         ["/table0", "/group0/table1",
                           "/group0/group1/table2"])
        self.assertEqual(arrays,
                         ['/var1', '/var4',
                          '/group0/var1', '/group0/var4',
                          '/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "File.__iter__() and Group.__iter__ test passed"

        groups = []
        tables = []
        arrays = []
        for group in self.h5file.walk_nodes("/group0/group1", classname="Group"):
            groups.append(group._v_pathname)
            for table in group._f_walknodes('Table'):
                tables.append(table._v_pathname)
            for arr in self.h5file.walk_nodes(group, 'Array'):
                arrays.append(arr._v_pathname)

        self.assertEqual(groups,
                         ["/group0/group1", "/group0/group1/group2"])

        self.assertEqual(tables, ["/group0/group1/table2"])

        self.assertEqual(arrays, ['/group0/group1/var1', '/group0/group1/var4'])

        if common.verbose:
            print "walk_nodes(pathname, classname) test passed"


class DeepTreeTestCase(unittest.TestCase):
    """Checks for deep hierarchy levels in PyTables trees.
    """

    def setUp(self):
        # Here we put a more conservative limit to deal with more platforms
        # With maxdepth = 64 this test would take less than 40 MB
        # of main memory to run, which is quite reasonable nowadays.
        # With maxdepth = 1024 this test will take around 300 MB.
        if common.heavy:
            self.maxdepth = 256  # Takes around 60 MB of memory!
        else:
            self.maxdepth = 64  # This should be safe for most machines
        if common.verbose:
            print "Maximum depth tested :", self.maxdepth

        # Open a new empty HDF5 file
        self.file = tempfile.mktemp(".h5")
        fileh = open_file(self.file, mode="w")
        group = fileh.root
        if common.verbose:
            print "Depth writing progress: ",
        # Iterate until maxdepth
        for depth in range(self.maxdepth):
            # Save it on the HDF5 file
            if common.verbose:
                print "%3d," % (depth),
            # Create a couple of arrays here
            fileh.create_array(group, 'array', [1, 1], "depth: %d" % depth)
            fileh.create_array(group, 'array2', [1, 1], "depth: %d" % depth)
            # And also a group
            fileh.create_group(group, 'group2_' + str(depth))
            # Finally, iterate over a new group
            group = fileh.create_group(group, 'group' + str(depth))
        # Close the file
        fileh.close()

    def tearDown(self):
        os.remove(self.file)
        common.cleanup(self)

    def _check_tree(self, file):
        # Open the previous HDF5 file in read-only mode
        fileh = open_file(file, mode="r")
        group = fileh.root
        if common.verbose:
            print "\nDepth reading progress: ",
        # Get the metadata on the previosly saved arrays
        for depth in range(self.maxdepth):
            if common.verbose:
                print "%3d," % (depth),
            # Check the contents
            self.assertEqual(group.array[:], [1, 1])
            self.assertTrue("array2" in group)
            self.assertTrue("group2_"+str(depth) in group)
            # Iterate over the next group
            group = fileh.get_node(group, 'group' + str(depth))
        if common.verbose:
            print # This flush the stdout buffer
        fileh.close()

    def test00_deepTree(self):
        "Creation of a large depth object tree."
        self._check_tree(self.file)

    def test01a_copyDeepTree(self):
        "Copy of a large depth object tree."
        fileh = open_file(self.file, mode="r")
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(file2, mode="w")
        if common.verbose:
            print "\nCopying deep tree..."
        fileh.copy_node(fileh.root, fileh2.root, recursive = True)
        fileh.close()
        fileh2.close()
        self._check_tree(file2)
        os.remove(file2)

    def test01b_copyDeepTree(self):
        "Copy of a large depth object tree with small node cache."
        fileh = open_file(self.file, mode="r", node_cache_slots=10)
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(file2, mode="w", node_cache_slots=10)
        if common.verbose:
            print "\nCopying deep tree..."
        fileh.copy_node(fileh.root, fileh2.root, recursive = True)
        fileh.close()
        fileh2.close()
        self._check_tree(file2)
        os.remove(file2)

    def test01c_copyDeepTree(self):
        "Copy of a large depth object tree with no node cache."
        fileh = open_file(self.file, mode="r", node_cache_slots=0)
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(file2, mode="w", node_cache_slots=0)
        if common.verbose:
            print "\nCopying deep tree..."
        fileh.copy_node(fileh.root, fileh2.root, recursive = True)
        fileh.close()
        fileh2.close()
        self._check_tree(file2)
        os.remove(file2)

    def test01d_copyDeepTree(self):
        "Copy of a large depth object tree with static node cache."
        # Do not execute this in heavy mode
        if common.heavy:
            return
        fileh = open_file(self.file, mode = "r", node_cache_slots=-256)
        file2 = tempfile.mktemp(".h5")
        fileh2 = open_file(file2, mode = "w", node_cache_slots=-256)
        if common.verbose:
            print "\nCopying deep tree..."
        fileh.copy_node(fileh.root, fileh2.root, recursive = True)
        fileh.close()
        fileh2.close()
        self._check_tree(file2)
        os.remove(file2)


class WideTreeTestCase(unittest.TestCase):
    """Checks for maximum number of children for a Group.
    """

    def test00_Leafs(self):
        """Checking creation of large number of leafs (1024) per group

        Variable 'maxchildren' controls this check. PyTables support
        up to 4096 children per group, but this would take too much
        memory (up to 64 MB) for testing purposes (may be we can add a
        test for big platforms). A 1024 children run takes up to 30 MB.
        A 512 children test takes around 25 MB.
        """

        import time
        if common.heavy:
            maxchildren = 4096
        else:
            maxchildren = 256
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_wideTree..." % \
                  self.__class__.__name__
            print "Maximum number of children tested :", maxchildren
        # Open a new empty HDF5 file
        file = tempfile.mktemp(".h5")
        #file = "test_widetree.h5"

        a = [1, 1]
        fileh = open_file(file, mode = "w")
        if common.verbose:
            print "Children writing progress: ",
        for child in range(maxchildren):
            if common.verbose:
                print "%3d," % (child),
            fileh.create_array(fileh.root, 'array' + str(child),
                              a, "child: %d" % child)
        if common.verbose:
            print
        # Close the file
        fileh.close()

        t1 = time.time()
        a = [1, 1]
        # Open the previous HDF5 file in read-only mode
        fileh = open_file(file, mode = "r")
        if common.verbose:
            print "\nTime spent opening a file with %d arrays: %s s" % \
                  (maxchildren, time.time()-t1)
            print "\nChildren reading progress: ",
        # Get the metadata on the previosly saved arrays
        for child in range(maxchildren):
            if common.verbose:
                print "%3d," % (child),
            # Create an array for later comparison
            # Get the actual array
            array_ = getattr(fileh.root, 'array' + str(child))
            b = array_.read()
            # Arrays a and b must be equal
            self.assertEqual(a, b)
        if common.verbose:
            print # This flush the stdout buffer
        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)


    def test01_wideTree(self):
        """Checking creation of large number of groups (1024) per group

        Variable 'maxchildren' controls this check. PyTables support
        up to 4096 children per group, but this would take too much
        memory (up to 64 MB) for testing purposes (may be we can add a
        test for big platforms). A 1024 children run takes up to 30 MB.
        A 512 children test takes around 25 MB.
        """

        import time
        if common.heavy:
            # for big platforms!
            maxchildren = 4096
        else:
            # for standard platforms
            maxchildren = 256
        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_wideTree..." % \
                  self.__class__.__name__
            print "Maximum number of children tested :", maxchildren
        # Open a new empty HDF5 file
        file = tempfile.mktemp(".h5")
        #file = "test_widetree.h5"

        fileh = open_file(file, mode = "w")
        if common.verbose:
            print "Children writing progress: ",
        for child in range(maxchildren):
            if common.verbose:
                print "%3d," % (child),
            fileh.create_group(fileh.root, 'group' + str(child),
                              "child: %d" % child)
        if common.verbose:
            print
        # Close the file
        fileh.close()

        t1 = time.time()
        # Open the previous HDF5 file in read-only mode
        fileh = open_file(file, mode = "r")
        if common.verbose:
            print "\nTime spent opening a file with %d groups: %s s" % \
                  (maxchildren, time.time()-t1)
            print "\nChildren reading progress: ",
        # Get the metadata on the previosly saved arrays
        for child in range(maxchildren):
            if common.verbose:
                print "%3d," % (child),
            # Get the actual group
            group = getattr(fileh.root, 'group' + str(child))
            # Arrays a and b must be equal
            self.assertEqual(group._v_title, "child: %d" % child)
        if common.verbose:
            print # This flush the stdout buffer
        # Close the file
        fileh.close()
        # Then, delete the file
        os.remove(file)



class HiddenTreeTestCase(unittest.TestCase):

    """Check for hidden groups, leaves and hierarchies."""

    def setUp(self):
        self.h5fname = tempfile.mktemp('.h5')
        self.h5file = open_file(
            self.h5fname, 'w', title = "Test for hidden nodes")

        self.visible = []  # list of visible object paths
        self.hidden = []  # list of hidden object paths

        # Create some visible nodes: a, g, g/a1, g/a2, g/g, g/g/a.
        h5f = self.h5file
        h5f.create_array('/', 'a', [0]);
        g = h5f.create_group('/', 'g');
        h5f.create_array(g, 'a1', [0]);
        h5f.create_array(g, 'a2', [0]);
        g_g = h5f.create_group(g, 'g');
        h5f.create_array(g_g, 'a', [0]);

        self.visible.extend(['/a', '/g', '/g/a1', '/g/a2', '/g/g', '/g/g/a'])

        # Create some hidden nodes: _p_a, _p_g, _p_g/a, _p_g/_p_a, g/_p_a.
        h5f.create_array('/', '_p_a', [0]);
        hg = h5f.create_group('/', '_p_g');
        h5f.create_array(hg, 'a', [0]);
        h5f.create_array(hg, '_p_a', [0]);
        h5f.create_array(g, '_p_a', [0]);

        self.hidden.extend(
            ['/_p_a', '/_p_g', '/_p_g/a', '/_p_g/_p_a', '/g/_p_a'])


    def tearDown(self):
        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)


    # The test behind commented out because the .objects dictionary
    # has been removed (as well as .leaves and .groups)
    def _test00_objects(self):
        """Absence of hidden nodes in `File.objects`."""

        objects = self.h5file.objects

        warnings.filterwarnings('ignore', category=DeprecationWarning)

        for vpath in self.visible:
            self.assertTrue(vpath in objects,
                "Missing visible node ``%s`` from ``File.objects``." % vpath)
        for hpath in self.hidden:
            self.assertTrue(hpath not in objects,
                "Found hidden node ``%s`` in ``File.objects``." % hpath)

        warnings.filterwarnings('default', category=DeprecationWarning)


    # The test behind commented out because the .objects dictionary
    # has been removed (as well as .leaves and .groups)
    def _test00b_objects(self):
        """Object dictionaries conformance with ``walk_nodes()``."""

        def dictCheck(dictName, className):
            file_ = self.h5file

            objects = getattr(file_, dictName)
            walkPaths = [node._v_pathname
                         for node in file_.walk_nodes('/', className)]
            dictPaths = [path for path in objects]
            walkPaths.sort()
            dictPaths.sort()
            self.assertEqual(
                walkPaths, dictPaths,
                "nodes in ``%s`` do not match those from ``walk_nodes()``"
                % dictName)
            self.assertEqual(
                len(walkPaths), len(objects),
                "length of ``%s`` differs from that of ``walk_nodes()``"
                % dictName)

        warnings.filterwarnings('ignore', category=DeprecationWarning)

        dictCheck('objects', None)
        dictCheck('groups', 'Group')
        dictCheck('leaves', 'Leaf')

        warnings.filterwarnings('default', category=DeprecationWarning)


    def test01_getNode(self):
        """Node availability via `File.get_node()`."""

        h5f = self.h5file

        for vpath in self.visible:
            h5f.get_node(vpath)
        for hpath in self.hidden:
            h5f.get_node(hpath)


    def test02_walkGroups(self):
        """Hidden group absence in `File.walk_groups()`."""

        hidden = self.hidden

        for group in self.h5file.walk_groups('/'):
            pathname = group._v_pathname
            self.assertTrue(pathname not in hidden,
                            "Walked across hidden group ``%s``." % pathname)


    def test03_walkNodes(self):
        """Hidden node absence in `File.walk_nodes()`."""

        hidden = self.hidden

        for node in self.h5file.walk_nodes('/'):
            pathname = node._v_pathname
            self.assertTrue(pathname not in hidden,
                            "Walked across hidden node ``%s``." % pathname)


    def test04_listNodesVisible(self):
        """Listing visible nodes under a visible group (list_nodes)."""

        hidden = self.hidden

        for node in self.h5file.list_nodes('/g'):
            pathname = node._v_pathname
            self.assertTrue(pathname not in hidden,
                            "Listed hidden node ``%s``." % pathname)


    def test04b_listNodesVisible(self):
        """Listing visible nodes under a visible group (iter_nodes)."""

        hidden = self.hidden

        for node in self.h5file.iter_nodes('/g'):
            pathname = node._v_pathname
            self.assertTrue(pathname not in hidden,
                            "Listed hidden node ``%s``." % pathname)


    def test05_listNodesHidden(self):
        """Listing visible nodes under a hidden group (list_nodes)."""

        hidden = self.hidden

        node_to_find = '/_p_g/a'
        found_node = False
        for node in self.h5file.list_nodes('/_p_g'):
            pathname = node._v_pathname
            if pathname == node_to_find:
                found_node = True
            self.assertTrue(pathname in hidden,
                            "Listed hidden node ``%s``." % pathname)

        self.assertTrue(found_node,
                        "Hidden node ``%s`` was not listed." % node_to_find)


    def test05b_iterNodesHidden(self):
        """Listing visible nodes under a hidden group (iter_nodes)."""

        hidden = self.hidden

        node_to_find = '/_p_g/a'
        found_node = False
        for node in self.h5file.iter_nodes('/_p_g'):
            pathname = node._v_pathname
            if pathname == node_to_find:
                found_node = True
            self.assertTrue(pathname in hidden,
                            "Listed hidden node ``%s``." % pathname)

        self.assertTrue(found_node,
                        "Hidden node ``%s`` was not listed." % node_to_find)


    # The test behind commented out because the .objects dictionary
    # has been removed (as well as .leaves and .groups)
    def _test06_reopen(self):
        """Reopening a file with hidden nodes."""

        self.h5file.close()
        self.h5file = open_file(self.h5fname)
        self.test00_objects()


    def test07_move(self):
        """Moving a node between hidden and visible groups."""

        is_visible_node = self.h5file.is_visible_node

        self.assertFalse(is_visible_node('/_p_g/a'))
        self.h5file.move_node('/_p_g/a', '/g', 'a')
        self.assertTrue(is_visible_node('/g/a'))
        self.h5file.move_node('/g/a', '/_p_g', 'a')
        self.assertFalse(is_visible_node('/_p_g/a'))


    def test08_remove(self):
        """Removing a visible group with hidden children."""

        self.assertTrue('/g/_p_a' in self.h5file)
        self.h5file.root.g._f_remove(recursive=True)
        self.assertFalse('/g/_p_a' in self.h5file)



class CreateParentsTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """
    Test the ``createparents`` flag.

    These are mainly for the user interface.  More thorough tests on
    the workings of the flag can be found in the ``test_do_undo.py``
    module.
    """

    filters = Filters(complevel=4)  # simply non-default

    def setUp(self):
        super(CreateParentsTestCase, self).setUp()
        self.h5file.create_array('/', 'array', [1])
        self.h5file.create_group('/', 'group', filters=self.filters)

    def test00_parentType(self):
        """Using the right type of parent node argument."""

        h5file, root = self.h5file, self.h5file.root

        self.assertRaises( TypeError, h5file.create_array,
                           root.group, 'arr', [1], createparents=True )
        self.assertRaises( TypeError, h5file.copy_node,
                           '/array', root.group, createparents=True )
        self.assertRaises( TypeError, h5file.move_node,
                           '/array', root.group, createparents=True )
        self.assertRaises( TypeError, h5file.copy_children,
                           '/group', root, createparents=True )

    def test01_inside(self):
        """Placing a node inside a nonexistent child of itself."""
        self.assertRaises( NodeError, self.h5file.move_node,
                           '/group', '/group/foo/bar',
                           createparents=True )
        self.assertFalse('/group/foo' in self.h5file)
        self.assertRaises( NodeError, self.h5file.copy_node,
                           '/group', '/group/foo/bar',
                           recursive=True, createparents=True )
        self.assertFalse('/group/foo' in self.h5file)

    def test02_filters(self):
        """Propagating the filters of created parent groups."""
        self.h5file.create_group('/group/foo/bar', 'baz', createparents=True)
        self.assertTrue('/group/foo/bar/baz' in self.h5file)
        for group in self.h5file.walk_groups('/group'):
            self.assertEqual(self.filters, group._v_filters)



#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    # This counter is useful when detecting memory leaks
    niter = 1

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(TreeTestCase))
        theSuite.addTest(unittest.makeSuite(DeepTreeTestCase))
        theSuite.addTest(unittest.makeSuite(WideTreeTestCase))
        theSuite.addTest(unittest.makeSuite(HiddenTreeTestCase))
        theSuite.addTest(unittest.makeSuite(CreateParentsTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )






