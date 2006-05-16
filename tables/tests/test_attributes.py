""" This test unit checks object creation functions, like openFile, createTable,
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
import warnings
import numarray
import numarray.strings

from tables import *

from common import verbose, heavy, cleanup, allequal
# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

class Record(IsDescription):
    var1 = StringCol(length=4)     # 4-character String
    var2 = IntCol()                # integer
    var3 = Int16Col()              # short integer
    var4 = FloatCol()              # double (double-precision)
    var5 = Float32Col()            # float  (single-precision)

class CreateTestCase(unittest.TestCase):

    def setUp(self):
        # Create an instance of HDF5 Table
        self.file = tempfile.mktemp(".h5")
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
        cleanup(self)

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
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        assert self.fileh.getNodeAttr(self.root.agroup, 'attr1') == \
               "p" * attrlength
        assert self.fileh.getNodeAttr(self.root.atable, 'attr1') == \
               "a" * attrlength
        assert self.fileh.getNodeAttr(self.root.anarray, 'attr1') == \
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
               ['CLASS','FILTERS', 'TITLE', 'VERSION']
        assert self.root.agroup._v_attrs._f_list("all") == \
               ['CLASS','FILTERS', 'TITLE', 'VERSION', "pq", "qr", "rs"]

        assert self.root.atable.attrs._f_list() == ["a", "b", "c"]
        assert self.root.atable.attrs._f_list("sys") == \
               ['AUTOMATIC_INDEX', 'CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'FILTERS_INDEX', 'FLAVOR', 'NROWS',
                'REINDEX', 'TITLE', 'VERSION']
        assert self.root.atable.attrs._f_list("all") == \
               ['AUTOMATIC_INDEX', 'CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'FILTERS_INDEX', 'FLAVOR', 'NROWS',
                'REINDEX', 'TITLE', 'VERSION',
                "a", "b", "c"]

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

    def test05b_removeAttributes(self):
        """Checking removing attributes (using File.delNodeAttr()) """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # delete an attribute
        self.fileh.delNodeAttr(self.group, "pq")

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

        # delete an attribute (File.delNodeAttr method)
        self.fileh.delNodeAttr(self.root, "qr", "agroup")
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
                print "local dict:", self.group._v_attrs.__dict__
            del self.group._v_attrs.CLASS
        except AttributeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next AttributeError was catched!"
                print value
        else:
            self.fail("expected a AttributeError")

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
        except AttributeError:
            if verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next AttributeError was catched!"
                print value
        else:
            self.fail("expected a AttributeError")

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

    def test10a_copyAttributes(self):
        """Checking copying attributes """

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # copy all attributes from "/agroup" to "/atable"
        self.group._v_attrs._f_copy(self.root.atable)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        if verbose:
            print "Attribute list:", self.root.atable._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        assert self.root.atable._v_attrs._f_list() == ["pq", "qr", "rs"]
        if verbose:
            print "Complete attribute list:", self.root.atable._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        assert self.root.atable._v_attrs._f_list("all") == \
               ['AUTOMATIC_INDEX', 'CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'FILTERS_INDEX', 'FLAVOR', 'NROWS',
                'REINDEX', 'TITLE', 'VERSION',
                "pq", "qr", "rs"]

    def test10b_copyAttributes(self):
        """Checking copying attributes (copyNodeAttrs)"""

        # With a Group object
        self.group._v_attrs.pq = "1"
        self.group._v_attrs.qr = "2"
        self.group._v_attrs.rs = "3"
        # copy all attributes from "/agroup" to "/atable"
        self.fileh.copyNodeAttrs(self.group, self.root.atable)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root

        if verbose:
            print "Attribute list:", self.root.atable._v_attrs._f_list()
        # Check the local attributes names (alphabetically sorted)
        assert self.root.atable._v_attrs._f_list() == ["pq", "qr", "rs"]
        if verbose:
            print "Complete attribute list:", self.root.atable._v_attrs._f_list("all")
        # Check the disk attribute names (not sorted)
        assert self.root.atable._v_attrs._f_list("all") == \
               ['AUTOMATIC_INDEX', 'CLASS',
                'FIELD_0_FILL', 'FIELD_0_NAME',
                'FIELD_1_FILL', 'FIELD_1_NAME',
                'FIELD_2_FILL', 'FIELD_2_NAME',
                'FIELD_3_FILL', 'FIELD_3_NAME',
                'FIELD_4_FILL', 'FIELD_4_NAME',
                'FILTERS_INDEX','FLAVOR', 'NROWS',
                'REINDEX', 'TITLE', 'VERSION',
                "pq", "qr", "rs"]

class NotCloseCreateTestCase(CreateTestCase):
    close = 0

class CloseCreateTestCase(CreateTestCase):
    close = 1


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
        cleanup(self)

#---------------------------------------

    def test01a_setIntAttributes(self):
        """Checking setting Int attributes (scalar case)"""

        self.array.attrs.pq = 1
        self.array.attrs.qr = 2
        self.array.attrs.rs = 3

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert self.root.anarray.attrs.pq == 1
        assert self.root.anarray.attrs.qr == 2
        assert self.root.anarray.attrs.rs == 3

    def test01b_setIntAttributes(self):
        """Checking setting Int attributes (scalar, numarray case)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for stype in checktypes:
            setattr(self.array.attrs, stype, numarray.array(1, type=stype))

        # Check the results
        if verbose:
            for stype in checktypes:
                print "type, value-->", stype, getattr(self.array.attrs, stype)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray


        for stype in checktypes:
            assert allequal(getattr(self.array.attrs, stype),
                            numarray.array(1, type=stype))

    def test01c_setIntAttributes(self):
        """Checking setting Int attributes (unidimensional numarray case)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for stype in checktypes:
            setattr(self.array.attrs, stype, numarray.array([1,2], type=stype))

        # Check the results
        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for stype in checktypes:
            if verbose:
                print "type, value-->", stype, getattr(self.array.attrs, stype)
            assert allequal(getattr(self.array.attrs, stype),
                            numarray.array([1,2], type=stype))

    def test01d_setIntAttributes(self):
        """Checking setting Int attributes (bidimensional numarray case)"""

        # 'UInt64' not supported on Win
        checktypes = ['Int8', 'Int16', 'Int32', 'Int64',
                      'UInt8', 'UInt16', 'UInt32']

        for stype in checktypes:
            setattr(self.array.attrs, stype,
                    numarray.array([[1,2],[2,3]], type=stype))

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        # Check the results
        for stype in checktypes:
            if verbose:
                print "type, value-->", stype, getattr(self.array.attrs, stype)
            assert allequal(getattr(self.array.attrs, stype),
                            numarray.array([[1,2],[2,3]], type=stype))

    def test02a_setFloatAttributes(self):
        """Checking setting Float (double) attributes"""

        # With a Table object
        self.array.attrs.pq = 1.0
        self.array.attrs.qr = 2.0
        self.array.attrs.rs = 3.0

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert self.root.anarray.attrs.pq == 1.0
        assert self.root.anarray.attrs.qr == 2.0
        assert self.root.anarray.attrs.rs == 3.0

    def test02b_setFloatAttributes(self):
        """Checking setting Float attributes (scalar, numarray case)"""

        checktypes = ['Float32', 'Float64']

        for stype in checktypes:
            setattr(self.array.attrs, stype,
                    numarray.array(1.1, type=stype))

        # Check the results
        if verbose:
            for stype in checktypes:
                print "type, value-->", stype, getattr(self.array.attrs, stype)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for stype in checktypes:
            #assert getattr(self.array.attrs, stype) == 1.1
            # In order to make Float32 tests pass. This is legal, not a trick.
            assert round(getattr(self.array.attrs, stype),6) == 1.1

    def test02c_setFloatAttributes(self):
        """Checking setting Float attributes (unidimensional numarray case)"""

        checktypes = ['Float32', 'Float64']

        for stype in checktypes:
            setattr(self.array.attrs, stype,
                    numarray.array([1.1,2.1], type=stype))

        # Check the results
        if verbose:
            for stype in checktypes:
                print "type, value-->", stype, getattr(self.array.attrs, stype)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for stype in checktypes:
            assert allequal(getattr(self.array.attrs, stype),
                            numarray.array([1.1,2.1], type=stype))

    def test02d_setFloatAttributes(self):
        """Checking setting Int attributes (bidimensional numarray case)"""

        checktypes = ['Float32', 'Float64']

        for stype in checktypes:
            setattr(self.array.attrs, stype,
                    numarray.array([[1.1,2.1],[2.1,3.1]], type=stype))

        # Check the results
        if verbose:
            for stype in checktypes:
                print "type, value-->", stype, getattr(self.array.attrs, stype)

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        for stype in checktypes:
            assert allequal(getattr(self.array.attrs, stype),
                            numarray.array([[1.1,2.1],[2.1,3.1]], type=stype))

    def test03_setObjectAttributes(self):
        """Checking setting Object attributes"""

        # With a Table object
        self.array.attrs.pq = [1.0, 2]
        self.array.attrs.qr = (1,2)
        self.array.attrs.rs = {"ddf":32.1, "dsd":1}

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert self.root.anarray.attrs.pq == [1.0, 2]
        assert self.root.anarray.attrs.qr == (1,2)
        assert self.root.anarray.attrs.rs == {"ddf":32.1, "dsd":1}

    def test04a_setStringAttributes(self):
        """Checking setting string attributes (scalar case)"""

        self.array.attrs.pq = 'foo'
        self.array.attrs.qr = 'bar'
        self.array.attrs.rs = 'baz'

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq
            print "qr -->", self.array.attrs.qr
            print "rs -->", self.array.attrs.rs

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert self.root.anarray.attrs.pq == 'foo'
        assert self.root.anarray.attrs.qr == 'bar'
        assert self.root.anarray.attrs.rs == 'baz'

    def test04b_setStringAttributes(self):
        """Checking setting string attributes (unidimensional 1-elem case)"""

        # Yes, there is no such thing as scalar character arrays.
        self.array.attrs.pq = numarray.strings.array(['foo'])

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert allequal(self.root.anarray.attrs.pq,
                        numarray.strings.array(['foo']))

    def test04c_setStringAttributes(self):
        """Checking setting string attributes (empty unidimensional 1-elem case)"""

        # Yes, there is no such thing as scalar character arrays.
        self.array.attrs.pq = numarray.strings.array([''])

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray
            if verbose:
                print "pq -->", self.array.attrs.pq

        assert allequal(self.root.anarray.attrs.pq,
                        numarray.strings.array(['']))

    def test04d_setStringAttributes(self):
        """Checking setting string attributes (unidimensional 2-elem case)"""

        self.array.attrs.pq = numarray.strings.array(['foo', 'bar3'])

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert allequal(self.root.anarray.attrs.pq,
                        numarray.strings.array(['foo', 'bar3']))

    def test04e_setStringAttributes(self):
        """Checking setting string attributes (empty unidimensional 2-elem case)"""

        self.array.attrs.pq = numarray.strings.array(['', ''])

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert allequal(self.root.anarray.attrs.pq,
                        numarray.strings.array(['', '']))

    def test04f_setStringAttributes(self):
        """Checking setting string attributes (bidimensional 4-elem case)"""

        self.array.attrs.pq = numarray.strings.array([['foo', 'foo2'],
                                                      ['foo3', 'foo4']])

        # Check the results
        if verbose:
            print "pq -->", self.array.attrs.pq

        if self.close:
            if verbose:
                print "(closing file version)"
            self.fileh.close()
            self.fileh = openFile(self.file, mode = "r+")
            self.root = self.fileh.root
            self.array = self.fileh.root.anarray

        assert allequal(self.root.anarray.attrs.pq,
                        numarray.strings.array([['foo', 'foo2'],
                                                ['foo3', 'foo4']]))


class NotCloseTypesTestCase(TypesTestCase):
    close = 0

class CloseTypesTestCase(TypesTestCase):
    close = 1


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    #heavy = 1 # Uncomment this only for testing purposes!

    #theSuite.addTest(unittest.makeSuite(NotCloseTypesTestCase))
    #theSuite.addTest(unittest.makeSuite(CloseCreateTestCase))
    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(NotCloseCreateTestCase))
        theSuite.addTest(unittest.makeSuite(CloseCreateTestCase))
        theSuite.addTest(unittest.makeSuite(NotCloseTypesTestCase))
        theSuite.addTest(unittest.makeSuite(CloseTypesTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
