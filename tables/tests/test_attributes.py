""" This test unit checks node atributes that are persistent (AttributeSet).
"""

import sys
import unittest
import os
import tempfile
import numpy
from numpy.testing import assert_array_equal, assert_almost_equal
from tables.parameters import NODE_CACHE_SLOTS

from tables import *
from tables.tests import common
from tables.tests.common import PyTablesTestCase
from tables.exceptions import DataTypeWarning

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup

class Record(IsDescription):
    var1 = StringCol(itemsize=4)  # 4-character String
    var2 = IntCol()               # integer
    var3 = Int16Col()             # short integer
    var4 = FloatCol()             # double (double-precision)
    var5 = Float32Col()           # float  (single-precision)

class CreateTestCase(unittest.TestCase):

    def setUp(self):
        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(
            self.file, mode = "w", NODE_CACHE_SLOTS=self.nodeCacheSlots)
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

#---------------------------------------

    def test01_setAttributes(self):
        """Checking setting large string attributes (File methods)"""

        attrlength = 2048
        # Try to put a long string attribute on a group object
        attr = self.fileh.setNodeAttr(self.root.agroup,
                                      "attr1", "p" * attrlength)
        # Now, try with a Table object
        attr = self.fileh.setNodeAttr(self.root.atable,
                                      "attr1", "a" * attrlength)
        # Finally, try with an Array object
        attr = self.fileh.setNodeAttr(self.root.anarray,
                                       "attr1", "n" * attrlength)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        self.assertEqual(self.fileh.getNodeAttr(self.root.agroup, 'attr1'),
                         "p" * attrlength)
        self.assertEqual(self.fileh.getNodeAttr(self.root.atable, 'attr1'),
                         "a" * attrlength)
        self.assertEqual(self.fileh.getNodeAttr(self.root.anarray, 'attr1'),
                         "n" * attrlength)

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
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        self.assertEqual(self.root.agroup._f_getAttr('attr1'), "p" * attrlength)
        self.assertEqual(self.root.atable.getAttr("attr1"), "a" * attrlength)
        self.assertEqual(self.root.anarray.getAttr("attr1"), "n" * attrlength)

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
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        # This should work even when the node cache is disabled
        self.assertEqual(self.root.agroup._v_attrs.attr1, "p" * attrlength)
        self.assertEqual(self.root.atable.attrs.attr1, "a" * attrlength)
        self.assertEqual(self.root.anarray.attrs.attr1, "n" * attrlength)

    def test04_listAttributes(self):
        """Checking listing attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        if common.verbose:
            print "Attribute list:", self.group._v_attrs._f_list()

        # Now, try with a Table object
        self.table.attrs.a = "1"
        self.table.attrs.c = "2"
        self.table.attrs.b = "3"
        if common.verbose:
            print "Attribute list:", self.table.attrs._f_list()

        # Finally, try with an Array object
        self.array.attrs.k = "1"
        self.array.attrs.j = "2"
        self.array.attrs.i = "3"
        if common.verbose:
            print "Attribute list:", self.array.attrs._f_list()

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        agroup = self.root.agroup
        self.assertEqual(agroup._v_attrs._f_list("user"), ["pq", "qr", "rs"])
        self.assertEqual(agroup._v_attrs._f_list("sys"),
                         ['CLASS', 'TITLE', 'VERSION'])
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "pq", "qr", "rs"])

        atable = self.root.atable
        self.assertEqual(atable.attrs._f_list(), ["a", "b", "c"])
        self.assertEqual(atable.attrs._f_list("sys"),
               ['CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'NROWS',
                'TITLE', 'VERSION'])
        self.assertEqual(atable.attrs._f_list("all"),
               ['CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'NROWS',
                'TITLE', 'VERSION',
                "a", "b", "c"])

        anarray = self.root.anarray
        self.assertEqual(anarray.attrs._f_list(), ["i", "j", "k"])
        self.assertEqual(anarray.attrs._f_list("sys"),
                         ['CLASS', 'FLAVOR', 'TITLE', 'VERSION'])
        self.assertEqual(anarray.attrs._f_list("all"),
                         ['CLASS', 'FLAVOR', 'TITLE', 'VERSION', "i", "j", "k"])

    def test05_removeAttributes(self):
        """Checking removing attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # delete an attribute
        del self.group._v_attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        agroup = self.root.agroup
        if common.verbose:
            print "Attribute list:", agroup._v_attrs._f_list()
        # Check the local attributes names
        self.assertEqual(agroup._v_attrs._f_list(), ["qr", "rs"])
        if common.verbose:
            print "Attribute list in disk:", \
                  agroup._v_attrs._f_list("all")
        # Check the disk attribute names
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "qr", "rs"])

        # delete an attribute (__delattr__ method)
        del agroup._v_attrs.qr
        if common.verbose:
            print "Attribute list:", agroup._v_attrs._f_list()
        # Check the local attributes names
        self.assertEqual(agroup._v_attrs._f_list(), ["rs"])
        if common.verbose:
            print "Attribute list in disk:", \
                  agroup._v_attrs._f_list()
        # Check the disk attribute names
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "rs"])

    def test05b_removeAttributes(self):
        """Checking removing attributes (using File.delNodeAttr()) """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # delete an attribute
        self.fileh.delNodeAttr(self.group, "pq")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        agroup = self.root.agroup
        if common.verbose:
            print "Attribute list:", agroup._v_attrs._f_list()
        # Check the local attributes names
        self.assertEqual(agroup._v_attrs._f_list(), ["qr", "rs"])
        if common.verbose:
            print "Attribute list in disk:", \
                  agroup._v_attrs._f_list("all")
        # Check the disk attribute names
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "qr", "rs"])

        # delete an attribute (File.delNodeAttr method)
        self.fileh.delNodeAttr(self.root, "qr", "agroup")
        if common.verbose:
            print "Attribute list:", agroup._v_attrs._f_list()
        # Check the local attributes names
        self.assertEqual(agroup._v_attrs._f_list(), ["rs"])
        if common.verbose:
            print "Attribute list in disk:", \
                  agroup._v_attrs._f_list()
        # Check the disk attribute names
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "rs"])

    def test06_removeAttributes(self):
        """Checking removing system attributes """

        # remove a system attribute
        if common.verbose:
            print "Before removing CLASS attribute"
            print "System attrs:", self.group._v_attrs._v_attrnamessys
        del self.group._v_attrs.CLASS
        self.assertEqual(self.group._v_attrs._f_list("sys"),
                         ['TITLE', 'VERSION'])
        if common.verbose:
            print "After removing CLASS attribute"
            print "System attrs:", self.group._v_attrs._v_attrnamessys

    def test07_renameAttributes(self):
        """Checking renaming attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # rename an attribute
        self.group._v_attrs._f_rename("pq", "op")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        agroup = self.root.agroup
        if common.verbose:
            print "Attribute list:", agroup._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        self.assertEqual(agroup._v_attrs._f_list(), ["op", "qr", "rs"])
        if common.verbose:
            print "Attribute list in disk:", agroup._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "op", "qr", "rs"])

    def test08_renameAttributes(self):
        """Checking renaming system attributes """

        if common.verbose:
            print "Before renaming CLASS attribute"
            print "All attrs:", self.group._v_attrs._v_attrnames
        # rename a system attribute
        self.group._v_attrs._f_rename("CLASS", "op")
        if common.verbose:
            print "After renaming CLASS attribute"
            print "All attrs:", self.group._v_attrs._v_attrnames

        # Check the disk attribute names (not sorted)
        agroup = self.root.agroup
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['TITLE', 'VERSION', "op"])

    def test09_overwriteAttributes(self):
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
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        agroup = self.root.agroup
        if common.verbose:
            print "Value of Attribute pq:", agroup._v_attrs.pq
        # Check the local attributes names (alphabetically sorted)
        self.assertEqual(agroup._v_attrs.pq, "4")
        self.assertEqual(agroup._v_attrs.qr, 2)
        self.assertEqual(agroup._v_attrs.rs, [1,2,3])
        if common.verbose:
            print "Attribute list in disk:", \
                  agroup._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        self.assertEqual(agroup._v_attrs._f_list("all"),
                         ['CLASS', 'TITLE', 'VERSION', "pq", "qr", "rs"])

    def test10a_copyAttributes(self):
        """Checking copying attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # copy all attributes from "/agroup" to "/atable"
        self.group._v_attrs._f_copy(self.root.atable)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        atable = self.root.atable
        if common.verbose:
            print "Attribute list:", atable._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        self.assertEqual(atable._v_attrs._f_list(), ["pq", "qr", "rs"])
        if common.verbose:
            print "Complete attribute list:", atable._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        self.assertEqual(atable._v_attrs._f_list("all"),
               ['CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'NROWS',
                'TITLE', 'VERSION',
                "pq", "qr", "rs"])

    def test10b_copyAttributes(self):
        """Checking copying attributes (copyNodeAttrs)"""

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # copy all attributes from "/agroup" to "/atable"
        self.fileh.copyNodeAttrs(self.group, self.root.atable)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        atable = self.root.atable
        if common.verbose:
            print "Attribute list:", atable._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        self.assertEqual(atable._v_attrs._f_list(), ["pq", "qr", "rs"])
        if common.verbose:
            print "Complete attribute list:", atable._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        self.assertEqual(atable._v_attrs._f_list("all"),
               ['CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'NROWS',
                'TITLE', 'VERSION',
                "pq", "qr", "rs"])


    def test10c_copyAttributes(self):
        """Checking copying attributes during group copies"""

        # With a Group object
        self.group._v_attrs['CLASS'] = "GROUP2"
        self.group._v_attrs['VERSION'] = "1.3"
        # copy "/agroup" to "/agroup2"
        self.fileh.copyNode(self.group, self.root, "agroup2")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        agroup2 = self.root.agroup2
        if common.verbose:
            print "Complete attribute list:", agroup2._v_attrs._f_list("all")
        self.assertEqual(agroup2._v_attrs['CLASS'], "GROUP2")
        self.assertEqual(agroup2._v_attrs['VERSION'], "1.3")


    def test10d_copyAttributes(self):
        """Checking copying attributes during leaf copies"""

        # With a Group object
        atable = self.root.atable
        atable._v_attrs['CLASS'] = "TABLE2"
        atable._v_attrs['VERSION'] = "1.3"
        # copy "/agroup" to "/agroup2"
        self.fileh.copyNode(atable, self.root, "atable2")

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+", NODE_CACHE_SLOTS=self.nodeCacheSlots)
            self.root = self.fileh.root

        atable2 = self.root.atable2
        if common.verbose:
            print "Complete attribute list:", atable2._v_attrs._f_list("all")
        self.assertEqual(atable2._v_attrs['CLASS'], "TABLE2")
        self.assertEqual(atable2._v_attrs['VERSION'], "1.3")


    def test11a_getitem(self):
        """Checking the __getitem__ interface."""

        attrs = self.group._v_attrs
        attrs.pq = "1"
        self.assertEqual(attrs['pq'], "1")


    def test11b_setitem(self):
        """Checking the __setitem__ interface."""

        attrs = self.group._v_attrs
        attrs['pq'] = "2"
        self.assertEqual(attrs['pq'], "2")


    def test11c_delitem(self):
        """Checking the __delitem__ interface."""

        attrs = self.group._v_attrs
        attrs.pq = "1"
        del attrs['pq']
        self.assertTrue('pq' not in attrs._f_list())


    def test11d_KeyError(self):
        """Checking that KeyError is raised in __getitem__/__delitem__."""

        attrs = self.group._v_attrs
        self.assertRaises(KeyError, attrs.__getitem__, 'pq')
        self.assertRaises(KeyError, attrs.__delitem__, 'pq')



class NotCloseCreate(CreateTestCase):
    close = False
    nodeCacheSlots = NODE_CACHE_SLOTS

class CloseCreate(CreateTestCase):
    close = True
    nodeCacheSlots = NODE_CACHE_SLOTS

class NoCacheNotCloseCreate(CreateTestCase):
    close = False
    nodeCacheSlots = 0

class NoCacheCloseCreate(CreateTestCase):
    close = True
    nodeCacheSlots = 0

class DictCacheNotCloseCreate(CreateTestCase):
    close = False
    nodeCacheSlots = -NODE_CACHE_SLOTS

class DictCacheCloseCreate(CreateTestCase):
    close = True
    nodeCacheSlots = -NODE_CACHE_SLOTS


class TypesTestCase(unittest.TestCase):

    def setUp(self):
        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w")
        self.root = self.fileh.root

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

#---------------------------------------

    def test00a_setBoolAttributes(self):
        """Checking setting Bool attributes (scalar, Python case)"""

        self.array.attrs.pq = True
        self.array.attrs.qr = False
        self.array.attrs.rs = True

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertEqual(self.root.anarray.attrs.pq, True)
        self.assertEqual(self.root.anarray.attrs.qr, False)
        self.assertEqual(self.root.anarray.attrs.rs, True)

    def test00b_setBoolAttributes(self):
        """Checking setting Bool attributes (scalar, NumPy case)"""

        self.array.attrs.pq = numpy.bool_(True)
        self.array.attrs.qr = numpy.bool_(False)
        self.array.attrs.rs = numpy.bool_(True)

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.root.anarray.attrs.pq, numpy.bool_))
        self.assertTrue(isinstance(self.root.anarray.attrs.qr, numpy.bool_))
        self.assertTrue(isinstance(self.root.anarray.attrs.rs, numpy.bool_))
        self.assertEqual(self.root.anarray.attrs.pq, True)
        self.assertEqual(self.root.anarray.attrs.qr, False)
        self.assertEqual(self.root.anarray.attrs.rs, True)

    def test00c_setBoolAttributes(self):
        """Checking setting Bool attributes (NumPy, 0-dim case)"""

        self.array.attrs.pq = numpy.array(True)
        self.array.attrs.qr = numpy.array(False)
        self.array.attrs.rs = numpy.array(True)

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertEqual(self.root.anarray.attrs.pq, True)
        self.assertEqual(self.root.anarray.attrs.qr, False)
        self.assertEqual(self.root.anarray.attrs.rs, True)

    def test00d_setBoolAttributes(self):
        """Checking setting Bool attributes (NumPy, multidim case)"""

        self.array.attrs.pq = numpy.array([True])
        self.array.attrs.qr = numpy.array([[False]])
        self.array.attrs.rs = numpy.array([[True, False],[True, False]])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.root.anarray.attrs.pq, numpy.array([True]))
        assert_array_equal(self.root.anarray.attrs.qr, numpy.array([[False]]))
        assert_array_equal(self.root.anarray.attrs.rs,
                           numpy.array([[True, False],[True, False]]))

    def test01a_setIntAttributes(self):
        """Checking setting Int attributes (scalar, Python case)"""

        self.array.attrs.pq = 1
        self.array.attrs.qr = 2
        self.array.attrs.rs = 3

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.root.anarray.attrs.pq, numpy.int_))
        self.assertTrue(isinstance(self.root.anarray.attrs.qr, numpy.int_))
        self.assertTrue(isinstance(self.root.anarray.attrs.rs, numpy.int_))
        self.assertEqual(self.root.anarray.attrs.pq, 1)
        self.assertEqual(self.root.anarray.attrs.qr, 2)
        self.assertEqual(self.root.anarray.attrs.rs, 3)

    def test01b_setIntAttributes(self):
        """Checking setting Int attributes (scalar, NumPy case)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype, numpy.array(1, dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray


        for dtype in checktypes:
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array(1, dtype=dtype))

    def test01c_setIntAttributes(self):
        """Checking setting Int attributes (unidimensional NumPy case)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype, numpy.array([1,2], dtype=dtype))

        # Check the results
        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            if common.verbose:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array([1,2], dtype=dtype))

    def test01d_setIntAttributes(self):
        """Checking setting Int attributes (unidimensional, non-contiguous)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for dtype in checktypes:
            arr = numpy.array([1,2,3,4], dtype=dtype)[::2]
            setattr(self.array.attrs, dtype, arr)

        # Check the results
        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            arr = numpy.array([1,2,3,4], dtype=dtype)[::2]
            if common.verbose:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)
            assert_array_equal(getattr(self.array.attrs, dtype), arr)

    def test01e_setIntAttributes(self):
        """Checking setting Int attributes (bidimensional NumPy case)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array([[1,2],[2,3]], dtype=dtype))

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        # Check the results
        for dtype in checktypes:
            if common.verbose:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array([[1,2],[2,3]], dtype=dtype))

    def test02a_setFloatAttributes(self):
        """Checking setting Float (double) attributes"""

        # Set some attrs
        self.array.attrs.pq = 1.0
        self.array.attrs.qr = 2.0
        self.array.attrs.rs = 3.0

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.root.anarray.attrs.pq, numpy.float_))
        self.assertTrue(isinstance(self.root.anarray.attrs.qr, numpy.float_))
        self.assertTrue(isinstance(self.root.anarray.attrs.rs, numpy.float_))
        self.assertTrue(self.root.anarray.attrs.pq, 1.0)
        self.assertTrue(self.root.anarray.attrs.qr, 2.0)
        self.assertTrue(self.root.anarray.attrs.rs, 3.0)

    def test02b_setFloatAttributes(self):
        """Checking setting Float attributes (scalar, NumPy case)"""

        checktypes = ['Float32', 'Float64']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array(1.1, dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            #assert getattr(self.array.attrs, dtype) == 1.1
            # In order to make Float32 tests pass. This is legal, not a trick.
            assert_almost_equal(getattr(self.array.attrs, dtype), 1.1)

    def test02c_setFloatAttributes(self):
        """Checking setting Float attributes (unidimensional NumPy case)"""

        checktypes = ['Float32', 'Float64']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array([1.1,2.1], dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array([1.1,2.1], dtype=dtype))

    def test02d_setFloatAttributes(self):
        """Checking setting Float attributes (unidimensional, non-contiguous)"""

        checktypes = ['Float32', 'Float64']

        for dtype in checktypes:
            arr = numpy.array([1.1,2.1,3.1,4.1], dtype=dtype)[1::2]
            setattr(self.array.attrs, dtype, arr)

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            arr = numpy.array([1.1,2.1,3.1,4.1], dtype=dtype)[1::2]
            assert_array_equal(getattr(self.array.attrs, dtype), arr)

    def test02e_setFloatAttributes(self):
        """Checking setting Int attributes (bidimensional NumPy case)"""

        checktypes = ['Float32', 'Float64']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array([[1.1,2.1],[2.1,3.1]], dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array([[1.1,2.1],[2.1,3.1]], dtype=dtype))

    def test03_setObjectAttributes(self):
        """Checking setting Object attributes"""

        # Set some attrs
        self.array.attrs.pq = [1.0, 2]
        self.array.attrs.qr = (1,2)
        self.array.attrs.rs = {"ddf":32.1, "dsd":1}

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertEqual(self.root.anarray.attrs.pq, [1.0, 2])
        self.assertEqual(self.root.anarray.attrs.qr, (1,2))
        self.assertEqual(self.root.anarray.attrs.rs, {"ddf":32.1, "dsd":1})

    def test04a_setStringAttributes(self):
        """Checking setting string attributes (scalar case)"""

        self.array.attrs.pq = 'foo'
        self.array.attrs.qr = 'bar'
        self.array.attrs.rs = 'baz'

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.root.anarray.attrs.pq, numpy.string_))
        self.assertTrue(isinstance(self.root.anarray.attrs.qr, numpy.string_))
        self.assertTrue(isinstance(self.root.anarray.attrs.rs, numpy.string_))
        self.assertEqual(self.root.anarray.attrs.pq, 'foo')
        self.assertEqual(self.root.anarray.attrs.qr, 'bar')
        self.assertEqual(self.root.anarray.attrs.rs, 'baz')

    def test04b_setStringAttributes(self):
        """Checking setting string attributes (unidimensional 1-elem case)"""

        self.array.attrs.pq = numpy.array(['foo'])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.root.anarray.attrs.pq,
                           numpy.array(['foo']))

    def test04c_setStringAttributes(self):
        """Checking setting string attributes (empty unidimensional 1-elem case)"""

        self.array.attrs.pq = numpy.array([''])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray
            if common.verbose:
                print "pq -->", self.array.attrs.pq

        assert_array_equal(self.root.anarray.attrs.pq,
                           numpy.array(['']))

    def test04d_setStringAttributes(self):
        """Checking setting string attributes (unidimensional 2-elem case)"""

        self.array.attrs.pq = numpy.array(['foo', 'bar3'])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.root.anarray.attrs.pq,
                           numpy.array(['foo', 'bar3']))

    def test04e_setStringAttributes(self):
        """Checking setting string attributes (empty unidimensional 2-elem case)"""

        self.array.attrs.pq = numpy.array(['', ''])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.root.anarray.attrs.pq,
                           numpy.array(['', '']))

    def test04f_setStringAttributes(self):
        """Checking setting string attributes (bidimensional 4-elem case)"""

        self.array.attrs.pq = numpy.array([['foo', 'foo2'],
                                           ['foo3', 'foo4']])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.root.anarray.attrs.pq,
                           numpy.array([['foo', 'foo2'],
                                        ['foo3', 'foo4']]))

    def test05a_setComplexAttributes(self):
        """Checking setting Complex (python) attributes"""

        # Set some attrs
        self.array.attrs.pq = 1.0+2j
        self.array.attrs.qr = 2.0+3j
        self.array.attrs.rs = 3.0+4j

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.root.anarray.attrs.pq, numpy.complex_))
        self.assertTrue(isinstance(self.root.anarray.attrs.qr, numpy.complex_))
        self.assertTrue(isinstance(self.root.anarray.attrs.rs, numpy.complex_))
        self.assertEqual(self.root.anarray.attrs.pq, 1.0+2j)
        self.assertEqual(self.root.anarray.attrs.qr, 2.0+3j)
        self.assertEqual(self.root.anarray.attrs.rs, 3.0+4j)

    def test05b_setComplexAttributes(self):
        """Checking setting Complex attributes (scalar, NumPy case)"""

        checktypes = ['complex64', 'complex128']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array(1.1+2j, dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            #assert getattr(self.array.attrs, dtype) == 1.1+2j
            # In order to make Complex32 tests pass.
            assert_almost_equal(getattr(self.array.attrs, dtype), 1.1+2j)

    def test05c_setComplexAttributes(self):
        """Checking setting Complex attributes (unidimensional NumPy case)"""

        checktypes = ['Complex32', 'Complex64']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array([1.1,2.1], dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array([1.1,2.1], dtype=dtype))

    def test05d_setComplexAttributes(self):
        """Checking setting Int attributes (bidimensional NumPy case)"""

        checktypes = ['Complex32', 'Complex64']

        for dtype in checktypes:
            setattr(self.array.attrs, dtype,
                    numpy.array([[1.1,2.1],[2.1,3.1]], dtype=dtype))

        # Check the results
        if common.verbose:
            for dtype in checktypes:
                print "type, value-->", dtype, getattr(self.array.attrs, dtype)

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for dtype in checktypes:
            assert_array_equal(getattr(self.array.attrs, dtype),
                               numpy.array([[1.1,2.1],[2.1,3.1]], dtype=dtype))

    def test06a_setUnicodeAttributes(self):
        """Checking setting unicode attributes (scalar case)"""

        self.array.attrs.pq = u'para\u0140lel'
        self.array.attrs.qr = u''                 # check #213
        self.array.attrs.rs = u'baz'

        # Check the results
        if common.verbose:
            if sys.platform != 'win32':
                # It seems that Windows cannot print this
                print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.array.attrs.pq, numpy.unicode_))
        self.assertTrue(isinstance(self.array.attrs.qr, numpy.unicode_))
        self.assertTrue(isinstance(self.array.attrs.rs, numpy.unicode_))
        self.assertEqual(self.array.attrs.pq, u'para\u0140lel')
        self.assertEqual(self.array.attrs.qr, u'')
        self.assertEqual(self.array.attrs.rs, u'baz')


    def test06b_setUnicodeAttributes(self):
        """Checking setting unicode attributes (unidimensional 1-elem case)"""

        self.array.attrs.pq = numpy.array([u'para\u0140lel'])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.array.attrs.pq,
                           numpy.array([u'para\u0140lel']))

    def test06c_setUnicodeAttributes(self):
        """Checking setting unicode attributes (empty unidimensional 1-elem case)"""

        # The next raises a `TypeError` when unpickled. See:
        # http://projects.scipy.org/numpy/ticket/1037
        #self.array.attrs.pq = numpy.array([u''])
        self.array.attrs.pq = numpy.array([u''], dtype="U1")

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray
            if common.verbose:
                print "pq -->", `self.array.attrs.pq`

        assert_array_equal(self.array.attrs.pq,
                           numpy.array([u''], dtype="U1"))

    def test06d_setUnicodeAttributes(self):
        """Checking setting unicode attributes (unidimensional 2-elem case)"""

        self.array.attrs.pq = numpy.array([u'para\u0140lel', u'bar3'])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.array.attrs.pq,
                           numpy.array([u'para\u0140lel', u'bar3']))

    def test06e_setUnicodeAttributes(self):
        """Checking setting unicode attributes (empty unidimensional 2-elem case)"""

        self.array.attrs.pq = numpy.array(['', ''], dtype="U1")

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.array.attrs.pq,
                           numpy.array(['', ''], dtype="U1"))

    def test06f_setUnicodeAttributes(self):
        """Checking setting unicode attributes (bidimensional 4-elem case)"""

        self.array.attrs.pq = numpy.array([[u'para\u0140lel', 'foo2'],
                                           ['foo3', u'para\u0140lel4']])

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert_array_equal(self.array.attrs.pq,
                           numpy.array([[u'para\u0140lel', 'foo2'],
                                        ['foo3', u'para\u0140lel4']]))

    def test07a_setRecArrayAttributes(self):
        """Checking setting RecArray (NumPy) attributes"""

        dt = numpy.dtype('i4,f8')
        # Set some attrs
        self.array.attrs.pq = numpy.zeros(2, dt)
        self.array.attrs.qr = numpy.ones((2,2), dt)
        self.array.attrs.rs = numpy.array([(1,2.)], dt)

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.array.attrs.pq, numpy.ndarray))
        self.assertTrue(isinstance(self.array.attrs.qr, numpy.ndarray))
        self.assertTrue(isinstance(self.array.attrs.rs, numpy.ndarray))
        assert_array_equal(self.array.attrs.pq, numpy.zeros(2, dt))
        assert_array_equal(self.array.attrs.qr, numpy.ones((2,2), dt))
        assert_array_equal(self.array.attrs.rs, numpy.array([(1,2.)], dt))

    def test07b_setRecArrayAttributes(self):
        """Checking setting nested RecArray (NumPy) attributes"""

        # Build a nested dtype
        dt = numpy.dtype([('f1', [('f1', 'i2'), ('f2', 'f8')])])
        # Set some attrs
        self.array.attrs.pq = numpy.zeros(2, dt)
        self.array.attrs.qr = numpy.ones((2,2), dt)
        self.array.attrs.rs = numpy.array([((1,2.),)], dt)

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.array.attrs.pq, numpy.ndarray))
        self.assertTrue(isinstance(self.array.attrs.qr, numpy.ndarray))
        self.assertTrue(isinstance(self.array.attrs.rs, numpy.ndarray))
        assert_array_equal(self.array.attrs.pq, numpy.zeros(2, dt))
        assert_array_equal(self.array.attrs.qr, numpy.ones((2,2), dt))
        assert_array_equal(self.array.attrs.rs, numpy.array([((1,2),)], dt))

    def test07c_setRecArrayAttributes(self):
        """Checking setting multidim nested RecArray (NumPy) attributes"""

        # Build a nested dtype
        dt = numpy.dtype([('f1', [('f1', 'i2', (2,)), ('f2', 'f8')])])
        # Set some attrs
        self.array.attrs.pq = numpy.zeros(2, dt)
        self.array.attrs.qr = numpy.ones((2,2), dt)
        self.array.attrs.rs = numpy.array([(([1,3],2.),)], dt)

        # Check the results
        if common.verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        self.assertTrue(isinstance(self.array.attrs.pq, numpy.ndarray))
        self.assertTrue(isinstance(self.array.attrs.qr, numpy.ndarray))
        self.assertTrue(isinstance(self.array.attrs.rs, numpy.ndarray))
        assert_array_equal(self.array.attrs.pq, numpy.zeros(2, dt))
        assert_array_equal(self.array.attrs.qr, numpy.ones((2,2), dt))
        assert_array_equal(self.array.attrs.rs, numpy.array([(([1,3],2),)], dt))

class NotCloseTypesTestCase(TypesTestCase):
    close = 0

class CloseTypesTestCase(TypesTestCase):
    close = 1


class NoSysAttrsTestCase(unittest.TestCase):

    def setUp(self):
        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(
            self.file, mode = "w", PYTABLES_SYS_ATTRS=False)
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

    def test00_listAttributes(self):
        """Checking listing attributes (no system attrs version)."""

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        if common.verbose:
            print "Attribute list:", self.group._v_attrs._f_list()

        # Now, try with a Table object
        self.table.attrs.a = "1"
        self.table.attrs.c = "2"
        self.table.attrs.b = "3"
        if common.verbose:
            print "Attribute list:", self.table.attrs._f_list()

        # Finally, try with an Array object
        self.array.attrs.k = "1"
        self.array.attrs.j = "2"
        self.array.attrs.i = "3"
        if common.verbose:
            print "Attribute list:", self.array.attrs._f_list()

        if self.close:
            if common.verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(
                self.file, mode = "r+")
            self.root = self.fileh.root

        agroup = self.root.agroup
        self.assertEqual(agroup._v_attrs._f_list("user"), ["pq", "qr", "rs"])
        self.assertEqual(agroup._v_attrs._f_list("sys"), [])
        self.assertEqual(agroup._v_attrs._f_list("all"), ["pq", "qr", "rs"])

        atable = self.root.atable
        self.assertEqual(atable.attrs._f_list(), ["a", "b", "c"])
        self.assertEqual(atable.attrs._f_list("sys"), [])
        self.assertEqual(atable.attrs._f_list("all"), ["a", "b", "c"])

        anarray = self.root.anarray
        self.assertEqual(anarray.attrs._f_list(), ["i", "j", "k"])
        self.assertEqual(anarray.attrs._f_list("sys"), [])
        self.assertEqual(anarray.attrs._f_list("all"), ["i", "j", "k"])

class NoSysAttrsNotClose(NoSysAttrsTestCase):
    close = False

class NoSysAttrsClose(NoSysAttrsTestCase):
    close = True



class SegFaultPythonTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test00_segfault(self):
        """Checking workaround for Python unpickle problem (see #253)."""

        self.h5file.root._v_attrs.trouble1 = "0"
        self.assertEqual(self.h5file.root._v_attrs.trouble1, "0")
        self.h5file.root._v_attrs.trouble2 = "0."
        self.assertEqual(self.h5file.root._v_attrs.trouble2, "0.")
        # Problem happens after reopening
        self._reopen()
        self.assertEqual(self.h5file.root._v_attrs.trouble1, "0")
        self.assertEqual(self.h5file.root._v_attrs.trouble2, "0.")
        if common.verbose:
            print "Great! '0' and '0.' values can be safely retrieved."



class UnsupportedAttrTypeTestCase(PyTablesTestCase):

    def test00_unsupportedType(self):
        """Checking file with unsupported type."""

        filename = self._testFilename('attr-u16.h5')
        fileh = openFile(filename)
        self.failUnlessWarns(DataTypeWarning, repr, fileh)
        fileh.close()


# Test for specific system attributes in EArray
class SpecificAttrsTestCase(common.TempFileMixin, common.PyTablesTestCase):

    def test00_earray(self):
        "Testing EArray specific attrs (create)."
        ea = self.h5file.createEArray('/', 'ea', Int32Atom(), (2,0,4))
        if common.verbose:
            print "EXTDIM-->", ea.attrs.EXTDIM
        self.assertEqual(ea.attrs.EXTDIM, 1)

    def test01_earray(self):
        "Testing EArray specific attrs (open)."
        ea = self.h5file.createEArray('/', 'ea', Int32Atom(), (0,1,4))
        self._reopen('r')
        ea = self.h5file.root.ea
        if common.verbose:
            print "EXTDIM-->", ea.attrs.EXTDIM
        self.assertEqual(ea.attrs.EXTDIM, 0)



#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(NotCloseCreate))
        theSuite.addTest(unittest.makeSuite(CloseCreate))
        theSuite.addTest(unittest.makeSuite(NoCacheNotCloseCreate))
        theSuite.addTest(unittest.makeSuite(NoCacheCloseCreate))
        theSuite.addTest(unittest.makeSuite(DictCacheNotCloseCreate))
        theSuite.addTest(unittest.makeSuite(DictCacheCloseCreate))
        theSuite.addTest(unittest.makeSuite(NotCloseTypesTestCase))
        theSuite.addTest(unittest.makeSuite(CloseTypesTestCase))
        theSuite.addTest(unittest.makeSuite(NoSysAttrsNotClose))
        theSuite.addTest(unittest.makeSuite(NoSysAttrsClose))
        theSuite.addTest(unittest.makeSuite(SegFaultPythonTestCase))
        theSuite.addTest(unittest.makeSuite(UnsupportedAttrTypeTestCase))
        theSuite.addTest(unittest.makeSuite(SpecificAttrsTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
