import sys
import unittest
import os
import tempfile
import warnings

import tables
from tables import *
from tables.node import NotLoggedMixin
from tables.path import joinPath

from tables.tests import common

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


class BasicTestCase(unittest.TestCase):

    """Test for basic Undo/Redo operations."""

    _reopen = False
    """Whether to reopen the file at certain points."""

    def _doReopen(self):
        if self._reopen:
            self.fileh.close()
            self.fileh = openFile(self.file, mode='r+')


    def setUp(self):
        # Create an HDF5 file
        #self.file = "/tmp/test.h5"
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                  "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test00_simple(self):
        """Checking simple do/undo"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_simple..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray', [3,4], "Another array")
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/otherarray" not in self.fileh)
        self.assertEqual(self.fileh._curaction, 0)
        self.assertEqual(self.fileh._curmark, 0)
        # Redo the operation
        self._doReopen()
        self.fileh.redo()
        if common.verbose:
            print "Object tree after redo:", self.fileh
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/otherarray" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray.title, "Another array")
        self.assertEqual(self.fileh._curaction, 1)
        self.assertEqual(self.fileh._curmark, 0)

    def test01_twice(self):
        """Checking do/undo (twice operations intertwined)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_twice..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray', [3,4], "Another array")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operations
        self._doReopen()
        self.fileh.undo()
        self.assertTrue("/otherarray" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertEqual(self.fileh._curaction, 0)
        self.assertEqual(self.fileh._curmark, 0)
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/otherarray" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray.title, "Another array")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")
        self.assertEqual(self.fileh._curaction, 2)
        self.assertEqual(self.fileh._curmark, 0)

    def test02_twice2(self):
        """Checking twice ops and two marks"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_twice2..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray', [3,4], "Another array")
        # Put a mark
        self._doReopen()
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        self.assertEqual(self.fileh._curaction, 3)
        self.assertEqual(self.fileh._curmark, 1)
        # Unwind just one mark
        self.fileh.undo()
        self.assertTrue("/otherarray" in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertEqual(self.fileh._curaction, 2)
        self.assertEqual(self.fileh._curmark, 1)
        # Unwind another mark
        self.fileh.undo()
        self.assertEqual(self.fileh._curaction, 0)
        self.assertEqual(self.fileh._curmark, 0)
        self.assertTrue("/otherarray" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        # Redo until the next mark
        self.fileh.redo()
        self.assertTrue("/otherarray" in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self._doReopen()
        self.assertEqual(self.fileh._curaction, 2)
        self.assertEqual(self.fileh._curmark, 1)
        # Redo until the end
        self.fileh.redo()
        self.assertTrue("/otherarray" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray.title, "Another array")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")
        self.assertEqual(self.fileh._curaction, 3)
        self.assertEqual(self.fileh._curmark, 1)

    def test03_6times3marks(self):
        """Checking with six ops and three marks"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_6times3marks..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Put a mark
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        # Put a mark
        self._doReopen()
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray5', [7,8], "Another array 5")
        self.fileh.createArray('/', 'otherarray6', [8,9], "Another array 6")
        # Unwind just one mark
        self.fileh.undo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" in self.fileh)
        self.assertTrue("/otherarray5" not in self.fileh)
        self.assertTrue("/otherarray6" not in self.fileh)
        # Unwind another mark
        self.fileh.undo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" not in self.fileh)
        self.assertTrue("/otherarray6" not in self.fileh)
        # Unwind all marks
        self.fileh.undo()
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" not in self.fileh)
        self.assertTrue("/otherarray6" not in self.fileh)
        # Redo until the next mark
        self._doReopen()
        self.fileh.redo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" not in self.fileh)
        self.assertTrue("/otherarray6" not in self.fileh)
        # Redo until the next mark
        self.fileh.redo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" in self.fileh)
        self.assertTrue("/otherarray5" not in self.fileh)
        self.assertTrue("/otherarray6" not in self.fileh)
        # Redo until the end
        self.fileh.redo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" in self.fileh)
        self.assertTrue("/otherarray5" in self.fileh)
        self.assertTrue("/otherarray6" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray3.read(), [5,6])
        self.assertEqual(self.fileh.root.otherarray4.read(), [6,7])
        self.assertEqual(self.fileh.root.otherarray5.read(), [7,8])
        self.assertEqual(self.fileh.root.otherarray6.read(), [8,9])
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")
        self.assertEqual(self.fileh.root.otherarray3.title, "Another array 3")
        self.assertEqual(self.fileh.root.otherarray4.title, "Another array 4")
        self.assertEqual(self.fileh.root.otherarray5.title, "Another array 5")
        self.assertEqual(self.fileh.root.otherarray6.title, "Another array 6")

    def test04_6times3marksro(self):
        """Checking with six operations, three marks and do/undo in random order"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_6times3marksro..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Put a mark
        self.fileh.mark()
        self._doReopen()
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        # Unwind the previous mark
        self.fileh.undo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Put a mark in the middle of stack
        if common.verbose:
            print "All nodes:", self.fileh.walkNodes()
        self.fileh.mark()
        self._doReopen()
        self.fileh.createArray('/', 'otherarray5', [7,8], "Another array 5")
        self.fileh.createArray('/', 'otherarray6', [8,9], "Another array 6")
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" in self.fileh)
        self.assertTrue("/otherarray6" in self.fileh)
        # Unwind previous mark
        self.fileh.undo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" not in self.fileh)
        self.assertTrue("/otherarray6" not in self.fileh)
        # Redo until the last mark
        self.fileh.redo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" in self.fileh)
        self.assertTrue("/otherarray6" in self.fileh)
        # Redo until the next mark (non-existent, so no action)
        self._doReopen()
        self.fileh.redo()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        self.assertTrue("/otherarray5" in self.fileh)
        self.assertTrue("/otherarray6" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray5.read(), [7,8])
        self.assertEqual(self.fileh.root.otherarray6.read(), [8,9])
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")
        self.assertEqual(self.fileh.root.otherarray5.title, "Another array 5")
        self.assertEqual(self.fileh.root.otherarray6.title, "Another array 6")

    def test05_destructive(self):
        """Checking with a destructive action during undo"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05_destructive..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        # Put a mark
        self.fileh.mark()
        self._doReopen()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operation
        self.fileh.undo()
        # Do the destructive operation
        self._doReopen()
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        # Check objects
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray3.read(), [5,6])
        self.assertEqual(self.fileh.root.otherarray3.title, "Another array 3")

    def test05b_destructive(self):
        """Checking with a destructive action during undo (II)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05b_destructive..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        # Put a mark
        self._doReopen()
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operation
        self.fileh.undo()
        # Do the destructive operation
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        # Put a mark
        self._doReopen()
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        self.assertTrue("/otherarray4" in self.fileh)
        # Now undo the past operation
        self.fileh.undo()
        # Check objects
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray3.read(), [5,6])
        self.assertEqual(self.fileh.root.otherarray3.title, "Another array 3")
        self.assertTrue("/otherarray4" not in self.fileh)

    def test05c_destructive(self):
        """Checking with a destructive action during undo (III)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05c_destructive..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        # Put a mark
        self.fileh.mark()
        self._doReopen()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operation
        self.fileh.undo()
        # Do the destructive operation
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        # Put a mark
        self.fileh.mark()
        self._doReopen()
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        self.assertTrue("/otherarray4" in self.fileh)
        # Now unwind twice
        self.fileh.undo()
        self._doReopen()
        self.fileh.undo()
        # Check objects
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)

    def test05d_destructive(self):
        """Checking with a destructive action during undo (IV)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05d_destructive..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        # Put a mark
        self._doReopen()
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operation
        self.fileh.undo()
        # Do the destructive operation
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        # Put a mark
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        self.assertTrue("/otherarray4" in self.fileh)
        # Now, go to the first mark
        self._doReopen()
        self.fileh.undo(0)
        # Check objects
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)

    def test05e_destructive(self):
        """Checking with a destructive action during undo (V)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05e_destructive..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        # Put a mark
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operation
        self.fileh.undo()
        self._doReopen()
        # Do the destructive operation
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        # Now, unwind the actions
        self.fileh.undo(0)
        self._doReopen()
        # Check objects
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)

    def test05f_destructive(self):
        "Checking with a destructive creation of existing node during undo"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test05f_destructive..." % self.__class__.__name__

        self.fileh.enableUndo()
        self.fileh.createArray('/', 'newarray', [1])
        self.fileh.undo()
        self._doReopen()
        self.assertTrue('/newarray' not in self.fileh)
        newarr = self.fileh.createArray('/', 'newarray', [1])
        self.fileh.undo()
        self.assertTrue('/newarray' not in self.fileh)
        self._doReopen()
        self.fileh.redo()
        self.assertTrue('/newarray' in self.fileh)
        if not self._reopen:
            self.assertTrue(self.fileh.root.newarray is newarr)

    def test06_totalunwind(self):
        """Checking do/undo (total unwind)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test06_totalunwind..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray', [3,4], "Another array")
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operations
        self._doReopen()
        self.fileh.undo(0)
        self.assertTrue("/otherarray" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)

    def test07_totalrewind(self):
        """Checking do/undo (total rewind)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test07_totalunwind..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray', [3,4], "Another array")
        self.fileh.mark()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operations
        self.fileh.undo(0)
        # Redo all the operations
        self._doReopen()
        self.fileh.redo(-1)
        # Check that objects has come back to life in a sane state
        self.assertTrue("/otherarray" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray.title, "Another array")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")


    def test08_marknames(self):
        """Checking mark names"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_marknames..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        self.fileh.mark("first")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        self.fileh.mark("second")
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        self.fileh.mark("third")
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        # Now go to mark "first"
        self.fileh.undo("first")
        self._doReopen()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to mark "third"
        self.fileh.redo("third")
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Now go to mark "second"
        self.fileh.undo("second")
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to the end
        self._doReopen()
        self.fileh.redo(-1)
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" in self.fileh)
        # Check that objects has come back to life in a sane state
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray3.read(), [5,6])
        self.assertEqual(self.fileh.root.otherarray4.read(), [6,7])

    def test08_initialmark(self):
        """Checking initial mark"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test08_initialmark..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        initmid = self.fileh.getCurrentMark()
        # Create a new array
        self.fileh.createArray('/', 'otherarray', [3,4], "Another array")
        self.fileh.mark()
        self._doReopen()
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        # Now undo the past operations
        self.fileh.undo(initmid)
        self.assertTrue("/otherarray" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        # Redo all the operations
        self.fileh.redo(-1)
        self._doReopen()
        # Check that objects has come back to life in a sane state
        self.assertTrue("/otherarray" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray.title, "Another array")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")


    def test09_marknames(self):
        """Checking mark names (wrong direction)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test09_marknames..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        self.fileh.mark("first")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        self.fileh.mark("second")
        self._doReopen()
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        self.fileh.mark("third")
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        # Now go to mark "first"
        self.fileh.undo("first")
        # Try to undo up to mark "third"
        try:
            self.fileh.undo("third")
        except UndoRedoError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UndoRedoError was catched!"
                print value
        else:
            self.fail("expected an UndoRedoError")
        # Now go to mark "third"
        self.fileh.redo("third")
        self._doReopen()
        # Try to redo up to mark "second"
        try:
            self.fileh.redo("second")
        except UndoRedoError:
            if common.verbose:
                (type, value, traceback) = sys.exc_info()
                print "\nGreat!, the next UndoRedoError was catched!"
                print value
        else:
            self.fail("expected an UndoRedoError")
        # Final checks
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)

    def test10_goto(self):
        """Checking mark names (goto)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_goto..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        self._doReopen()
        self.fileh.mark("first")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        self.fileh.mark("second")
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        self._doReopen()
        self.fileh.mark("third")
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        # Now go to mark "first"
        self.fileh.goto("first")
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to mark "third"
        self.fileh.goto("third")
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Now go to mark "second"
        self._doReopen()
        self.fileh.goto("second")
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to the end
        self.fileh.goto(-1)
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" in self.fileh)
        # Check that objects has come back to life in a sane state
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray3.read(), [5,6])
        self.assertEqual(self.fileh.root.otherarray4.read(), [6,7])

    def test10_gotoint(self):
        """Checking mark sequential ids (goto)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test10_gotoint..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [3,4], "Another array 1")
        self.fileh.mark("first")
        self.fileh.createArray('/', 'otherarray2', [4,5], "Another array 2")
        self.fileh.mark("second")
        self._doReopen()
        self.fileh.createArray('/', 'otherarray3', [5,6], "Another array 3")
        self.fileh.mark("third")
        self.fileh.createArray('/', 'otherarray4', [6,7], "Another array 4")
        # Now go to mark "first"
        self.fileh.goto(1)
        self._doReopen()
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to beginning
        self.fileh.goto(0)
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to mark "third"
        self._doReopen()
        self.fileh.goto(3)
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Now go to mark "second"
        self.fileh.goto(2)
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        self.assertTrue("/otherarray4" not in self.fileh)
        # Go to the end
        self._doReopen()
        self.fileh.goto(-1)
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertTrue("/otherarray4" in self.fileh)
        # Check that objects has come back to life in a sane state
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.read(), [3,4])
        self.assertEqual(self.fileh.root.otherarray2.read(), [4,5])
        self.assertEqual(self.fileh.root.otherarray3.read(), [5,6])
        self.assertEqual(self.fileh.root.otherarray4.read(), [6,7])

    def test11_contiguous(self):
        "Creating contiguous marks"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test11_contiguous..." % self.__class__.__name__

        self.fileh.enableUndo()
        m1 = self.fileh.mark()
        m2 = self.fileh.mark()
        self.assertNotEqual(m1, m2)
        self._doReopen()
        self.fileh.undo(m1)
        self.assertEqual(self.fileh.getCurrentMark(), m1)
        self.fileh.redo(m2)
        self.assertEqual(self.fileh.getCurrentMark(), m2)
        self.fileh.goto(m1)
        self.assertEqual(self.fileh.getCurrentMark(), m1)
        self.fileh.goto(m2)
        self.assertEqual(self.fileh.getCurrentMark(), m2)
        self.fileh.goto(-1)
        self._doReopen()
        self.assertEqual(self.fileh.getCurrentMark(), m2)
        self.fileh.goto(0)
        self.assertEqual(self.fileh.getCurrentMark(), 0)

    def test12_keepMark(self):
        "Ensuring the mark is kept after an UNDO operation"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test12_keepMark..." % self.__class__.__name__

        self.fileh.enableUndo()
        arr1 = self.fileh.createArray('/', 'newarray1', [1])

        mid = self.fileh.mark()
        self._doReopen()
        self.fileh.undo()
        # We should have moved to the initial mark.
        self.assertEqual(self.fileh.getCurrentMark(), 0)
        # So /newarray1 should not be there.
        self.assertTrue('/newarray1' not in self.fileh)

    def test13_severalEnableDisable(self):
        "Checking that successive enable/disable Undo works"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test13_severalEnableDisable..." % self.__class__.__name__

        self.fileh.enableUndo()
        arr1 = self.fileh.createArray('/', 'newarray1', [1])
        self.fileh.undo()
        self._doReopen()
        # We should have moved to 'mid' mark, not the initial mark.
        self.assertEqual(self.fileh.getCurrentMark(), 0)
        # So /newarray1 should still be there.
        self.assertTrue('/newarray1' not in self.fileh)
        # Close this do/undo session
        self.fileh.disableUndo()
        # Do something
        arr2 = self.fileh.createArray('/', 'newarray2', [1])
        # Enable again do/undo
        self.fileh.enableUndo()
        arr3 = self.fileh.createArray('/', 'newarray3', [1])
        mid = self.fileh.mark()
        arr4 = self.fileh.createArray('/', 'newarray4', [1])
        self.fileh.undo()
        # We should have moved to 'mid' mark, not the initial mark.
        self.assertEqual(self.fileh.getCurrentMark(), mid)
        # So /newarray2 and /newarray3 should still be there.
        self.assertTrue('/newarray1' not in self.fileh)
        self.assertTrue('/newarray2' in self.fileh)
        self.assertTrue('/newarray3' in self.fileh)
        self.assertTrue('/newarray4' not in self.fileh)
        # Close this do/undo session
        self._doReopen()
        self.fileh.disableUndo()
        # Enable again do/undo
        self.fileh.enableUndo()
        arr3 = self.fileh.createArray('/', 'newarray1', [1])
        arr4 = self.fileh.createArray('/', 'newarray4', [1])
        # So /newarray2 and /newarray3 should still be there.
        self.assertTrue('/newarray1' in self.fileh)
        self.assertTrue('/newarray2' in self.fileh)
        self.assertTrue('/newarray3' in self.fileh)
        self.assertTrue('/newarray4' in self.fileh)
        self.fileh.undo()
        self._doReopen()
        self.assertTrue('/newarray1' not in self.fileh)
        self.assertTrue('/newarray2' in self.fileh)
        self.assertTrue('/newarray3' in self.fileh)
        self.assertTrue('/newarray4' not in self.fileh)
        # Close this do/undo session
        self.fileh.disableUndo()


class PersistenceTestCase(BasicTestCase):

    """Test for basic Undo/Redo operations with persistence."""

    _reopen = True


class createArrayTestCase(unittest.TestCase):
    "Test for createArray operations"

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00(self):
        """Checking one action"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [1,2], "Another array 1")
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/otherarray1" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertEqual(self.fileh.root.otherarray1.read(), [1,2])


    def test01(self):
        """Checking two actions"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [1,2], "Another array 1")
        self.fileh.createArray('/', 'otherarray2', [2,3], "Another array 2")
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")
        self.assertEqual(self.fileh.root.otherarray1.read(), [1,2])
        self.assertEqual(self.fileh.root.otherarray2.read(), [2,3])


    def test02(self):
        """Checking three actions"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [1,2], "Another array 1")
        self.fileh.createArray('/', 'otherarray2', [2,3], "Another array 2")
        self.fileh.createArray('/', 'otherarray3', [3,4], "Another array 3")
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/otherarray2" not in self.fileh)
        self.assertTrue("/otherarray3" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/otherarray2" in self.fileh)
        self.assertTrue("/otherarray3" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertEqual(self.fileh.root.otherarray2.title, "Another array 2")
        self.assertEqual(self.fileh.root.otherarray3.title, "Another array 3")
        self.assertEqual(self.fileh.root.otherarray1.read(), [1,2])
        self.assertEqual(self.fileh.root.otherarray2.read(), [2,3])
        self.assertEqual(self.fileh.root.otherarray3.read(), [3,4])

    def test03(self):
        """Checking three actions in different depth levels"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.createArray('/', 'otherarray1', [1,2], "Another array 1")
        self.fileh.createArray('/agroup', 'otherarray2', [2,3], "Another array 2")
        self.fileh.createArray('/agroup/agroup3', 'otherarray3', [3,4], "Another array 3")
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/otherarray1" not in self.fileh)
        self.assertTrue("/agroup/otherarray2" not in self.fileh)
        self.assertTrue("/agroup/agroup3/otherarray3" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/otherarray1" in self.fileh)
        self.assertTrue("/agroup/otherarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3/otherarray3" in self.fileh)
        self.assertEqual(self.fileh.root.otherarray1.title, "Another array 1")
        self.assertEqual(self.fileh.root.agroup.otherarray2.title,
                         "Another array 2")
        self.assertEqual(self.fileh.root.agroup.agroup3.otherarray3.title,
                         "Another array 3")
        self.assertEqual(self.fileh.root.otherarray1.read(), [1,2])
        self.assertEqual(self.fileh.root.agroup.otherarray2.read(), [2,3])
        self.assertEqual(self.fileh.root.agroup.agroup3.otherarray3.read(),
                         [3,4])


class createGroupTestCase(unittest.TestCase):
    "Test for createGroup operations"

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00(self):
        """Checking one action"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new group
        self.fileh.createGroup('/', 'othergroup1', "Another group 1")
        # Now undo the past operation
        self.fileh.undo()
        # Check that othergroup1 does not exist in the object tree
        self.assertTrue("/othergroup1" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that othergroup1 has come back to life in a sane state
        self.assertTrue("/othergroup1" in self.fileh)
        self.assertEqual(self.fileh.root.othergroup1._v_title,
                         "Another group 1")


    def test01(self):
        """Checking two actions"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new group
        self.fileh.createGroup('/', 'othergroup1', "Another group 1")
        self.fileh.createGroup('/', 'othergroup2', "Another group 2")
        # Now undo the past operation
        self.fileh.undo()
        # Check that othergroup does not exist in the object tree
        self.assertTrue("/othergroup1" not in self.fileh)
        self.assertTrue("/othergroup2" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that othergroup* has come back to life in a sane state
        self.assertTrue("/othergroup1" in self.fileh)
        self.assertTrue("/othergroup2" in self.fileh)
        self.assertEqual(self.fileh.root.othergroup1._v_title,
                         "Another group 1")
        self.assertEqual(self.fileh.root.othergroup2._v_title,
                         "Another group 2")


    def test02(self):
        """Checking three actions"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new group
        self.fileh.createGroup('/', 'othergroup1', "Another group 1")
        self.fileh.createGroup('/', 'othergroup2', "Another group 2")
        self.fileh.createGroup('/', 'othergroup3', "Another group 3")
        # Now undo the past operation
        self.fileh.undo()
        # Check that othergroup* does not exist in the object tree
        self.assertTrue("/othergroup1" not in self.fileh)
        self.assertTrue("/othergroup2" not in self.fileh)
        self.assertTrue("/othergroup3" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that othergroup* has come back to life in a sane state
        self.assertTrue("/othergroup1" in self.fileh)
        self.assertTrue("/othergroup2" in self.fileh)
        self.assertTrue("/othergroup3" in self.fileh)
        self.assertEqual(self.fileh.root.othergroup1._v_title,
                         "Another group 1")
        self.assertEqual(self.fileh.root.othergroup2._v_title,
                         "Another group 2")
        self.assertEqual(self.fileh.root.othergroup3._v_title,
                         "Another group 3")


    def test03(self):
        """Checking three actions in different depth levels"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new group
        self.fileh.createGroup('/', 'othergroup1', "Another group 1")
        self.fileh.createGroup('/othergroup1', 'othergroup2', "Another group 2")
        self.fileh.createGroup('/othergroup1/othergroup2', 'othergroup3', "Another group 3")
        # Now undo the past operation
        self.fileh.undo()
        # Check that othergroup* does not exist in the object tree
        self.assertTrue("/othergroup1" not in self.fileh)
        self.assertTrue("/othergroup1/othergroup2" not in self.fileh)
        self.assertTrue("/othergroup1/othergroup2/othergroup3" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that othergroup* has come back to life in a sane state
        self.assertTrue("/othergroup1" in self.fileh)
        self.assertTrue("/othergroup1/othergroup2" in self.fileh)
        self.assertTrue("/othergroup1/othergroup2/othergroup3" in self.fileh)
        self.assertEqual(self.fileh.root.othergroup1._v_title,
                         "Another group 1")
        self.assertEqual(self.fileh.root.othergroup1.othergroup2._v_title,
                         "Another group 2")
        self.assertEqual(
            self.fileh.root.othergroup1.othergroup2.othergroup3._v_title,
            "Another group 3")


minRowIndex = 10
def populateTable(where, name):
    "Create a table under where with name name"

    class Indexed(IsDescription):
        var1 = StringCol(itemsize=4, dflt="", pos=1)
        var2 = BoolCol(dflt=0, pos=2)
        var3 = IntCol(dflt=0, pos=3)
        var4 = FloatCol(dflt=0, pos=4)

    nrows = minRowIndex
    table = where._v_file.createTable(where, name, Indexed, "Indexed",
                                      None, nrows)
    for i in range(nrows):
        table.row['var1'] = str(i)
        # table.row['var2'] = i > 2
        table.row['var2'] = i % 2
        table.row['var3'] = i
        table.row['var4'] = float(nrows - i - 1)
        table.row.append()
    table.flush()

    # Index all entries:
    indexrows = table.cols.var1.createIndex()
    indexrows = table.cols.var2.createIndex()
    indexrows = table.cols.var3.createIndex()
    # Do not index the var4 column
    #indexrows = table.cols.var4.createIndex()
    if common.verbose:
        print "Number of written rows:", nrows
        print "Number of indexed rows:", table.cols.var1.index.nelements
        print "Number of indexed rows(2):", indexrows

class renameNodeTestCase(unittest.TestCase):
    "Test for renameNode operations"

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")
        # Create a table in root
        table = populateTable(self.fileh.root, 'table')

    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)

    def test00(self):
        """Checking renameNode (over Groups without children)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.renameNode('/agroup2', 'agroup3')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does not exist in the object tree
        self.assertTrue("/agroup2" in self.fileh)
        self.assertTrue("/agroup3" not in self.fileh)
        self.assertEqual(self.fileh.root.agroup2._v_title, "Group title 2")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/agroup2" not in self.fileh)
        self.assertTrue("/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup3._v_title, "Group title 2")

    def test01(self):
        """Checking renameNode (over Groups with children)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.renameNode('/agroup', 'agroup3')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does not exist in the object tree
        self.assertTrue("/agroup" in self.fileh)
        self.assertTrue("/agroup3" not in self.fileh)
        # Check that children are reachable
        self.assertTrue("/agroup/anarray1" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/agroup" not in self.fileh)
        self.assertTrue("/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup3._v_title, "Group title")
        # Check that children are reachable
        self.assertTrue("/agroup3/anarray1" in self.fileh)
        self.assertTrue("/agroup3/anarray2" in self.fileh)
        self.assertTrue("/agroup3/agroup3" in self.fileh)

    def test01b(self):
        """Checking renameNode (over Groups with children 2)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.renameNode('/agroup', 'agroup3')
        self.fileh.renameNode('/agroup3', 'agroup4')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does not exist in the object tree
        self.assertTrue("/agroup" in self.fileh)
        self.assertTrue("/agroup4" not in self.fileh)
        # Check that children are reachable
        self.assertTrue("/agroup/anarray1" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/agroup" not in self.fileh)
        self.assertTrue("/agroup4" in self.fileh)
        self.assertEqual(self.fileh.root.agroup4._v_title, "Group title")
        # Check that children are reachable
        self.assertTrue("/agroup4/anarray1" in self.fileh)
        self.assertTrue("/agroup4/anarray2" in self.fileh)
        self.assertTrue("/agroup4/agroup3" in self.fileh)

    def test02(self):
        """Checking renameNode (over Leaves)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.renameNode('/anarray', 'anarray2')
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/anarray" in self.fileh)
        self.assertTrue("/anarray2" not in self.fileh)
        self.assertEqual(self.fileh.root.anarray.title, "Array title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/anarray" not in self.fileh)
        self.assertTrue("/anarray2" in self.fileh)
        self.assertEqual(self.fileh.root.anarray2.title, "Array title")

    def test03(self):
        """Checking renameNode (over Tables)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.renameNode('/table', 'table2')
        # Now undo the past operation
        self.fileh.undo()
        # Check that table2 does not exist in the object tree
        self.assertTrue("/table" in self.fileh)
        table = self.fileh.root.table
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertTrue(table.cols.var4.index is None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertTrue("/table2" not in self.fileh)
        self.assertEqual(self.fileh.root.table.title, "Indexed")
        # Redo the operation
        self.fileh.redo()
        # Check that table2 has come back to life in a sane state
        self.assertTrue("/table" not in self.fileh)
        self.assertTrue("/table2" in self.fileh)
        self.assertEqual(self.fileh.root.table2.title, "Indexed")
        table = self.fileh.root.table2
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertTrue(table.cols.var4.index is None)


class moveNodeTestCase(unittest.TestCase):
    "Tests for moveNode operations"

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")
        # Create a table in root
        table = populateTable(self.fileh.root, 'table')

    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00(self):
        """Checking moveNode (over Leaf)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.moveNode('/anarray', '/agroup/agroup3')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does not exist in the object tree
        self.assertTrue("/anarray" in self.fileh)
        self.assertTrue("/agroup/agroup3/anarray" not in self.fileh)
        self.assertEqual(self.fileh.root.anarray.title, "Array title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/anarray" not in self.fileh)
        self.assertTrue("/agroup/agroup3/anarray" in self.fileh)
        self.assertEqual(self.fileh.root.agroup.agroup3.anarray.title,
                         "Array title")

    def test01(self):
        """Checking moveNode (over Groups with children)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.moveNode('/agroup', '/agroup2', 'agroup3')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does not exist in the object tree
        self.assertTrue("/agroup" in self.fileh)
        self.assertTrue("/agroup2/agroup3" not in self.fileh)
        # Check that children are reachable
        self.assertTrue("/agroup/anarray1" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/agroup" not in self.fileh)
        self.assertTrue("/agroup2/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup2.agroup3._v_title,
                         "Group title")
        # Check that children are reachable
        self.assertTrue("/agroup2/agroup3/anarray1" in self.fileh)
        self.assertTrue("/agroup2/agroup3/anarray2" in self.fileh)
        self.assertTrue("/agroup2/agroup3/agroup3" in self.fileh)

    def test01b(self):
        """Checking moveNode (over Groups with children 2)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.moveNode('/agroup', '/', 'agroup3')
        self.fileh.moveNode('/agroup3', '/agroup2', 'agroup4')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does not exist in the object tree
        self.assertTrue("/agroup" in self.fileh)
        self.assertTrue("/agroup2/agroup4" not in self.fileh)
        # Check that children are reachable
        self.assertTrue("/agroup/anarray1" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/agroup" not in self.fileh)
        self.assertTrue("/agroup2/agroup4" in self.fileh)
        self.assertEqual(self.fileh.root.agroup2.agroup4._v_title,
                         "Group title")
        # Check that children are reachable
        self.assertTrue("/agroup2/agroup4/anarray1" in self.fileh)
        self.assertTrue("/agroup2/agroup4/anarray2" in self.fileh)
        self.assertTrue("/agroup2/agroup4/agroup3" in self.fileh)

    def test02(self):
        """Checking moveNode (over Leaves)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.moveNode('/anarray', '/agroup2', 'anarray2')
        # Now undo the past operation
        self.fileh.undo()
        # Check that otherarray does not exist in the object tree
        self.assertTrue("/anarray" in self.fileh)
        self.assertTrue("/agroup2/anarray2" not in self.fileh)
        self.assertEqual(self.fileh.root.anarray.title, "Array title")
        # Redo the operation
        self.fileh.redo()
        # Check that otherarray has come back to life in a sane state
        self.assertTrue("/anarray" not in self.fileh)
        self.assertTrue("/agroup2/anarray2" in self.fileh)
        self.assertEqual(self.fileh.root.agroup2.anarray2.title, "Array title")

    def test03(self):
        """Checking moveNode (over Tables)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.moveNode('/table', '/agroup2', 'table2')
        # Now undo the past operation
        self.fileh.undo()
        # Check that table2 does not exist in the object tree
        self.assertTrue("/table" in self.fileh)
        self.assertTrue("/agroup2/table2" not in self.fileh)
        table = self.fileh.root.table
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertTrue(table.cols.var4.index is None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertEqual(self.fileh.root.table.title, "Indexed")
        # Redo the operation
        self.fileh.redo()
        # Check that table2 has come back to life in a sane state
        self.assertTrue("/table" not in self.fileh)
        self.assertTrue("/agroup2/table2" in self.fileh)
        self.assertEqual(self.fileh.root.agroup2.table2.title, "Indexed")
        table = self.fileh.root.agroup2.table2
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertTrue(table.cols.var4.index is None)


class removeNodeTestCase(unittest.TestCase):
    "Test for removeNode operations"

    def setUp(self):
        # Create an HDF5 file
        #self.file = "/tmp/test.h5"
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")
        # Create a table in root
        table = populateTable(self.fileh.root, 'table')


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00(self):
        """Checking removeNode (over Leaf)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Delete an existing array
        self.fileh.removeNode('/anarray')
        # Now undo the past operation
        self.fileh.undo()
        # Check that it does exist in the object tree
        self.assertTrue("/anarray" in self.fileh)
        self.assertEqual(self.fileh.root.anarray.title, "Array title")
        # Redo the operation
        self.fileh.redo()
        # Check that array has gone again
        self.assertTrue("/anarray" not in self.fileh)

    def test00b(self):
        """Checking removeNode (over several Leaves)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00b..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Delete a couple of arrays
        self.fileh.removeNode('/anarray')
        self.fileh.removeNode('/agroup/anarray2')
        # Now undo the past operation
        self.fileh.undo()
        # Check that arrays has come into life
        self.assertTrue("/anarray" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertEqual(self.fileh.root.anarray.title, "Array title")
        self.assertEqual(self.fileh.root.agroup.anarray2.title, "Array title 2")
        # Redo the operation
        self.fileh.redo()
        # Check that arrays has disappeared again
        self.assertTrue("/anarray" not in self.fileh)
        self.assertTrue("/agroup/anarray2" not in self.fileh)

    def test00c(self):
        """Checking removeNode (over Tables)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00c..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Create a new array
        self.fileh.removeNode('/table')
        # Now undo the past operation
        self.fileh.undo()
        # Check that table2 does not exist in the object tree
        self.assertTrue("/table" in self.fileh)
        table = self.fileh.root.table
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertTrue(table.cols.var4.index is None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertEqual(self.fileh.root.table.title, "Indexed")
        # Redo the operation
        self.fileh.redo()
        # Check that table2 has come back to life in a sane state
        self.assertTrue("/table" not in self.fileh)

    def test01(self):
        """Checking removeNode (over Groups with children)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Delete a group recursively
        self.fileh.removeNode('/agroup', recursive=1)
        # Now undo the past operation
        self.fileh.undo()
        # Check that parent and children has come into life in a sane state
        self.assertTrue("/agroup" in self.fileh)
        self.assertTrue("/agroup/anarray1" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        # Redo the operation
        self.fileh.redo()
        # Check that parent and children are not reachable
        self.assertTrue("/agroup" not in self.fileh)
        self.assertTrue("/agroup/anarray1" not in self.fileh)
        self.assertTrue("/agroup/anarray2" not in self.fileh)
        self.assertTrue("/agroup/agroup3" not in self.fileh)

    def test01b(self):
        """Checking removeNode (over Groups with children 2)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01b..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # Remove a couple of groups
        self.fileh.removeNode('/agroup', recursive=1)
        self.fileh.removeNode('/agroup2')
        # Now undo the past operation
        self.fileh.undo()
        # Check that they does exist in the object tree
        self.assertTrue("/agroup" in self.fileh)
        self.assertTrue("/agroup2" in self.fileh)
        # Check that children are reachable
        self.assertTrue("/agroup/anarray1" in self.fileh)
        self.assertTrue("/agroup/anarray2" in self.fileh)
        self.assertTrue("/agroup/agroup3" in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        # Redo the operation
        self.fileh.redo()
        # Check that groups does not exist again
        self.assertTrue("/agroup" not in self.fileh)
        self.assertTrue("/agroup2" not in self.fileh)
        # Check that children are not reachable
        self.assertTrue("/agroup/anarray1" not in self.fileh)
        self.assertTrue("/agroup/anarray2" not in self.fileh)
        self.assertTrue("/agroup/agroup3" not in self.fileh)


class copyNodeTestCase(unittest.TestCase):
    "Tests for copyNode and copyChildren operations"

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")
        # Create a table in root
        table = populateTable(self.fileh.root, 'table')


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00_copyLeaf(self):
        """Checking copyNode (over Leaves)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_copyLeaf..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # /anarray => /agroup/agroup3/
        newNode = self.fileh.copyNode('/anarray', '/agroup/agroup3')

        # Undo the copy.
        self.fileh.undo()
        # Check that the copied node does not exist in the object tree.
        self.assertTrue('/agroup/agroup3/anarray' not in self.fileh)

        # Redo the copy.
        self.fileh.redo()
        # Check that the copied node exists again in the object tree.
        self.assertTrue('/agroup/agroup3/anarray' in self.fileh)
        self.assertTrue(self.fileh.root.agroup.agroup3.anarray is newNode)


    def test00b_copyTable(self):
        """Checking copyNode (over Tables)"""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00b_copyTable..." % self.__class__.__name__

        # open the do/undo
        self.fileh.enableUndo()
        # /table => /agroup/agroup3/
        warnings.filterwarnings("ignore", category=UserWarning)
        table = self.fileh.copyNode(
            '/table', '/agroup/agroup3', propindexes=True)
        warnings.filterwarnings("default", category=UserWarning)
        self.assertTrue("/agroup/agroup3/table" in self.fileh)

        table = self.fileh.root.agroup.agroup3.table
        self.assertEqual(table.title, "Indexed")
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertTrue(table.cols.var4.index is None)
        # Now undo the past operation
        self.fileh.undo()
        table = self.fileh.root.table
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertTrue(table.cols.var4.index is None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        # Check that the copied node does not exist in the object tree.
        self.assertTrue("/agroup/agroup3/table" not in self.fileh)
        # Redo the operation
        self.fileh.redo()
        # Check that table has come back to life in a sane state
        self.assertTrue("/table" in self.fileh)
        self.assertTrue("/agroup/agroup3/table" in self.fileh)
        table = self.fileh.root.agroup.agroup3.table
        self.assertEqual(table.title, "Indexed")
        self.assertTrue(table.cols.var1.index is not None)
        self.assertTrue(table.cols.var2.index is not None)
        self.assertTrue(table.cols.var3.index is not None)
        self.assertEqual(table.cols.var1.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var2.index.nelements, minRowIndex)
        self.assertEqual(table.cols.var3.index.nelements, minRowIndex)
        self.assertTrue(table.cols.var4.index is None)

    def test01_copyGroup(self):
        "Copying a group (recursively)."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_copyGroup..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # /agroup => /acopy
        newNode = self.fileh.copyNode(
            '/agroup', newname = 'acopy', recursive = True)

        # Undo the copy.
        self.fileh.undo()
        # Check that the copied node does not exist in the object tree.
        self.assertTrue('/acopy' not in self.fileh)
        self.assertTrue('/acopy/anarray1' not in self.fileh)
        self.assertTrue('/acopy/anarray2' not in self.fileh)
        self.assertTrue('/acopy/agroup3' not in self.fileh)

        # Redo the copy.
        self.fileh.redo()
        # Check that the copied node exists again in the object tree.
        self.assertTrue('/acopy' in self.fileh)
        self.assertTrue('/acopy/anarray1' in self.fileh)
        self.assertTrue('/acopy/anarray2' in self.fileh)
        self.assertTrue('/acopy/agroup3' in self.fileh)
        self.assertTrue(self.fileh.root.acopy is newNode)


    def test02_copyLeafOverwrite(self):
        "Copying a leaf, overwriting destination."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_copyLeafOverwrite..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # /anarray => /agroup/agroup
        oldNode = self.fileh.root.agroup
        newNode = self.fileh.copyNode(
            '/anarray', newname = 'agroup', overwrite = True)

        # Undo the copy.
        self.fileh.undo()
        # Check that the copied node does not exist in the object tree.
        # Check that the overwritten node exists again in the object tree.
        self.assertTrue(self.fileh.root.agroup is oldNode)

        # Redo the copy.
        self.fileh.redo()
        # Check that the copied node exists again in the object tree.
        # Check that the overwritten node does not exist in the object tree.
        self.assertTrue(self.fileh.root.agroup is newNode)


    def test03_copyChildren(self):
        "Copying the children of a group."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copyChildren..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # /agroup/* => /agroup/
        self.fileh.copyChildren('/agroup', '/agroup2', recursive = True)

        # Undo the copy.
        self.fileh.undo()
        # Check that the copied nodes do not exist in the object tree.
        self.assertTrue('/agroup2/anarray1' not in self.fileh)
        self.assertTrue('/agroup2/anarray2' not in self.fileh)
        self.assertTrue('/agroup2/agroup3' not in self.fileh)

        # Redo the copy.
        self.fileh.redo()
        # Check that the copied nodes exist again in the object tree.
        self.assertTrue('/agroup2/anarray1' in self.fileh)
        self.assertTrue('/agroup2/anarray2' in self.fileh)
        self.assertTrue('/agroup2/agroup3' in self.fileh)


class ComplexTestCase(unittest.TestCase):
    "Tests for a mix of all operations"

    def setUp(self):
        # Create an HDF5 file
        #self.file = "/tmp/test.h5"
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(self.file, mode = "w", title="File title")
        fileh = self.fileh
        root = fileh.root
        # Create an array
        fileh.createArray(root, 'array', [1,2],
                          title = "Title example")

        # Create another array object
        array = fileh.createArray(root, 'anarray',
                                  [1], "Array title")
        # Create a group object
        group = fileh.createGroup(root, 'agroup',
                                  "Group title")
        # Create a couple of objects there
        array1 = fileh.createArray(group, 'anarray1',
                                   [2], "Array title 1")
        array2 = fileh.createArray(group, 'anarray2',
                                   [2], "Array title 2")
        # Create a lonely group in first level
        group2 = fileh.createGroup(root, 'agroup2',
                                   "Group title 2")
        # Create a new group in the second level
        group3 = fileh.createGroup(group, 'agroup3',
                                   "Group title 3")


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00(self):
        """Mix of createArray, createGroup, renameNone, moveNode, removeNode,
           copyNode and copyChildren."""

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # Create an array
        self.fileh.createArray(self.fileh.root, 'anarray3',
                               [1], "Array title 3")
        # Create a group
        array2 = self.fileh.createGroup(self.fileh.root, 'agroup3',
                                        "Group title 3")
        # /anarray => /agroup/agroup3/
        newNode = self.fileh.copyNode('/anarray3', '/agroup/agroup3')
        newNode = self.fileh.copyChildren('/agroup', '/agroup3', recursive=1)
        # rename anarray
        array4 = self.fileh.renameNode('/anarray', 'anarray4')
        # Move anarray
        newNode = self.fileh.copyNode('/anarray3', '/agroup')
        # Remove anarray4
        self.fileh.removeNode('/anarray4')
        # Undo the actions
        self.fileh.undo()
        self.assertTrue('/anarray4' not in self.fileh)
        self.assertTrue('/anarray3' not in self.fileh)
        self.assertTrue('/agroup/agroup3/anarray3' not in self.fileh)
        self.assertTrue('/agroup3' not in self.fileh)
        self.assertTrue('/anarray4' not in self.fileh)
        self.assertTrue('/anarray' in self.fileh)

        # Redo the actions
        self.fileh.redo()
        # Check that the copied node exists again in the object tree.
        self.assertTrue('/agroup/agroup3/anarray3' in self.fileh)
        self.assertTrue('/agroup/anarray3' in self.fileh)
        self.assertTrue('/agroup3/agroup3/anarray3' in self.fileh)
        self.assertTrue('/agroup3/anarray3' not in self.fileh)
        self.assertTrue(self.fileh.root.agroup.anarray3 is newNode)
        self.assertTrue('/anarray' not in self.fileh)
        self.assertTrue('/anarray4' not in self.fileh)

    def test01(self):
        "Test with multiple generations (Leaf case)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # remove /anarray
        self.fileh.removeNode('/anarray')
        # Create an array in the same place
        self.fileh.createArray(self.fileh.root, 'anarray',
                                        [2], "Array title 2")
        # remove the array again
        self.fileh.removeNode('/anarray')
        # Create an array
        array2 = self.fileh.createArray(self.fileh.root, 'anarray',
                                        [3], "Array title 3")
        # remove the array again
        self.fileh.removeNode('/anarray')
        # Create an array
        array2 = self.fileh.createArray(self.fileh.root, 'anarray',
                                        [4], "Array title 4")
        # Undo the actions
        self.fileh.undo()
        # Check that /anarray is in the correct state before redoing
        self.assertEqual(self.fileh.root.anarray.title, "Array title")
        self.assertEqual(self.fileh.root.anarray[:], [1])
        # Redo the actions
        self.fileh.redo()
        self.assertEqual(self.fileh.root.anarray.title, "Array title 4")
        self.assertEqual(self.fileh.root.anarray[:], [4])

    def test02(self):
        "Test with multiple generations (Group case)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # remove /agroup
        self.fileh.removeNode('/agroup2')
        # Create a group in the same place
        self.fileh.createGroup(self.fileh.root, 'agroup2', "Group title 22")
        # remove the group
        self.fileh.removeNode('/agroup2')
        # Create a group
        self.fileh.createGroup(self.fileh.root, 'agroup2', "Group title 3")
        # remove the group
        self.fileh.removeNode('/agroup2')
        # Create a group
        self.fileh.createGroup(self.fileh.root, 'agroup2', "Group title 4")
        # Create a child group
        self.fileh.createGroup(self.fileh.root.agroup2, 'agroup5',
                               "Group title 5")
        # Undo the actions
        self.fileh.undo()
        # Check that /agroup is in the state before enabling do/undo
        self.assertEqual(self.fileh.root.agroup2._v_title, "Group title 2")
        self.assertTrue('/agroup2' in self.fileh)
        # Redo the actions
        self.fileh.redo()
        self.assertEqual(self.fileh.root.agroup2._v_title, "Group title 4")
        self.assertEqual(self.fileh.root.agroup2.agroup5._v_title,
                         "Group title 5")

    def test03(self):
        "Test with multiple generations (Group case, recursive remove)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # remove /agroup
        self.fileh.removeNode('/agroup', recursive=1)
        # Create a group in the same place
        self.fileh.createGroup(self.fileh.root, 'agroup', "Group title 2")
        # remove the group
        self.fileh.removeNode('/agroup')
        # Create a group
        self.fileh.createGroup(self.fileh.root, 'agroup', "Group title 3")
        # remove the group
        self.fileh.removeNode('/agroup')
        # Create a group
        self.fileh.createGroup(self.fileh.root, 'agroup', "Group title 4")
        # Create a child group
        self.fileh.createGroup(self.fileh.root.agroup, 'agroup5',
                               "Group title 5")
        # Undo the actions
        self.fileh.undo()
        # Check that /agroup is in the state before enabling do/undo
        self.assertTrue('/agroup' in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title")
        self.assertTrue('/agroup/anarray1' in self.fileh)
        self.assertTrue('/agroup/anarray2' in self.fileh)
        self.assertTrue('/agroup/agroup3' in self.fileh)
        self.assertTrue('/agroup/agroup5' not in self.fileh)
        # Redo the actions
        self.fileh.redo()
        self.assertTrue('/agroup' in self.fileh)
        self.assertEqual(self.fileh.root.agroup._v_title, "Group title 4")
        self.assertTrue('/agroup/agroup5' in self.fileh)
        self.assertEqual(self.fileh.root.agroup.agroup5._v_title, "Group title 5")

    def test03b(self):
        "Test with multiple generations (Group case, recursive remove, case 2)"

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03b..." % self.__class__.__name__

        # Enable undo/redo.
        self.fileh.enableUndo()
        # Create a new group with a child
        self.fileh.createGroup(self.fileh.root, 'agroup3', "Group title 3")
        self.fileh.createGroup(self.fileh.root.agroup3, 'agroup4',
                               "Group title 4")
        # remove /agroup3
        self.fileh.removeNode('/agroup3', recursive=1)
        # Create a group in the same place
        self.fileh.createGroup(self.fileh.root, 'agroup3', "Group title 4")
        # Undo the actions
        self.fileh.undo()
        # Check that /agroup is in the state before enabling do/undo
        self.assertTrue('/agroup3' not in self.fileh)
        # Redo the actions
        self.fileh.redo()
        self.assertEqual(self.fileh.root.agroup3._v_title, "Group title 4")
        self.assertTrue('/agroup3' in self.fileh)
        self.assertTrue('/agroup/agroup4' not in self.fileh)



class AttributesTestCase(unittest.TestCase):
    "Tests for operation on attributes"

    def setUp(self):
        # Create an HDF5 file
        self.file = tempfile.mktemp(".h5")
        self.fileh = openFile(
            self.file, mode = "w", title = "Attribute operations")

        # Create an array.
        array = self.fileh.createArray('/', 'array', [1,2])

        # Set some attributes on it.
        attrs = array.attrs
        attrs.attr_1 = 10
        attrs.attr_2 = 20
        attrs.attr_3 = 30


    def tearDown(self):
        # Remove the temporary file
        self.fileh.close()
        os.remove(self.file)
        common.cleanup(self)


    def test00_setAttr(self):
        "Setting a nonexistent attribute."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test00_setAttr..." % self.__class__.__name__

        array = self.fileh.root.array
        attrs = array.attrs

        self.fileh.enableUndo()
        setattr(attrs, 'attr_0', 0)
        self.assertTrue('attr_0' in attrs)
        self.assertEqual(attrs.attr_0, 0)
        self.fileh.undo()
        self.assertTrue('attr_0' not in attrs)
        self.fileh.redo()
        self.assertTrue('attr_0' in attrs)
        self.assertEqual(attrs.attr_0, 0)


    def test01_setAttrExisting(self):
        "Setting an existing attribute."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_setAttrExisting..." % self.__class__.__name__

        array = self.fileh.root.array
        attrs = array.attrs

        self.fileh.enableUndo()
        setattr(attrs, 'attr_1', 11)
        self.assertTrue('attr_1' in attrs)
        self.assertEqual(attrs.attr_1, 11)
        self.fileh.undo()
        self.assertTrue('attr_1' in attrs)
        self.assertEqual(attrs.attr_1, 10)
        self.fileh.redo()
        self.assertTrue('attr_1' in attrs)
        self.assertEqual(attrs.attr_1, 11)


    def test02_delAttr(self):
        "Removing an attribute."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test02_delAttr..." % self.__class__.__name__

        array = self.fileh.root.array
        attrs = array.attrs

        self.fileh.enableUndo()
        delattr(attrs, 'attr_1')
        self.assertTrue('attr_1' not in attrs)
        self.fileh.undo()
        self.assertTrue('attr_1' in attrs)
        self.assertEqual(attrs.attr_1, 10)
        self.fileh.redo()
        self.assertTrue('attr_1' not in attrs)


    def test03_copyNodeAttrs(self):
        "Copying an attribute set."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test03_copyNodeAttrs..." % self.__class__.__name__

        rattrs = self.fileh.root._v_attrs
        rattrs.attr_0 = 0
        rattrs.attr_1 = 100

        array = self.fileh.root.array
        attrs = array.attrs

        self.fileh.enableUndo()
        attrs._f_copy(self.fileh.root)
        self.assertEqual(rattrs.attr_0, 0)
        self.assertEqual(rattrs.attr_1, 10)
        self.assertEqual(rattrs.attr_2, 20)
        self.assertEqual(rattrs.attr_3, 30)
        self.fileh.undo()
        self.assertEqual(rattrs.attr_0, 0)
        self.assertEqual(rattrs.attr_1, 100)
        self.assertTrue('attr_2' not in rattrs)
        self.assertTrue('attr_3' not in rattrs)
        self.fileh.redo()
        self.assertEqual(rattrs.attr_0, 0)
        self.assertEqual(rattrs.attr_1, 10)
        self.assertEqual(rattrs.attr_2, 20)
        self.assertEqual(rattrs.attr_3, 30)


    def test04_replaceNode(self):
        "Replacing a node with a rewritten attribute."

        if common.verbose:
            print '\n', '-=' * 30
            print "Running %s.test04_replaceNode..." % self.__class__.__name__

        array = self.fileh.root.array
        attrs = array.attrs

        self.fileh.enableUndo()
        attrs.attr_1 = 11
        self.fileh.removeNode('/array')
        arr = self.fileh.createArray('/', 'array', [1])
        arr.attrs.attr_1 = 12
        self.fileh.undo()
        self.assertTrue('attr_1' in self.fileh.root.array.attrs)
        self.assertEqual(self.fileh.root.array.attrs.attr_1, 10)
        self.fileh.redo()
        self.assertTrue('attr_1' in self.fileh.root.array.attrs)
        self.assertEqual(self.fileh.root.array.attrs.attr_1, 12)


class NotLoggedTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test not logged nodes."""

    class NotLoggedArray(NotLoggedMixin, Array):
        pass


    def test00_hierarchy(self):
        """Performing hierarchy operations on a not logged node."""

        self.h5file.createGroup('/', 'tgroup')
        self.h5file.enableUndo()

        # Node creation is not undone.
        arr = self.NotLoggedArray( self.h5file.root, 'test',
                                   [1], self._getMethodName() )
        self.h5file.undo()
        self.assertTrue('/test' in self.h5file)

        # Node movement is not undone.
        arr.move('/tgroup')
        self.h5file.undo()
        self.assertTrue('/tgroup/test' in self.h5file)

        # Node removal is not undone.
        arr.remove()
        self.h5file.undo()
        self.assertTrue('/tgroup/test' not in self.h5file)


    def test01_attributes(self):
        """Performing attribute operations on a not logged node."""

        arr = self.NotLoggedArray( self.h5file.root, 'test',
                                   [1], self._getMethodName() )
        self.h5file.enableUndo()

        # Attribute creation is not undone.
        arr._v_attrs.foo = 'bar'
        self.h5file.undo()
        self.assertEqual(arr._v_attrs.foo, 'bar')

        # Attribute change is not undone.
        arr._v_attrs.foo = 'baz'
        self.h5file.undo()
        self.assertEqual(arr._v_attrs.foo, 'baz')

        # Attribute removal is not undone.
        del arr._v_attrs.foo
        self.h5file.undo()
        self.assertRaises(AttributeError, getattr, arr._v_attrs, 'foo')


class CreateParentsTestCase(common.TempFileMixin, common.PyTablesTestCase):

    """Test the ``createparents`` flag."""

    def setUp(self):
        super(CreateParentsTestCase, self).setUp()
        g1 = self.h5file.createGroup('/', 'g1')
        g2 = self.h5file.createGroup(g1, 'g2')

    def existing(self, paths):
        """Return a set of the existing paths in `paths`."""
        return frozenset(path for path in paths if path in self.h5file)

    def basetest(self, doit, pre, post):
        pre()
        self.h5file.enableUndo()

        paths =  ['/g1', '/g1/g2', '/g1/g2/g3', '/g1/g2/g3/g4']
        for newpath in paths:
            before = self.existing(paths)
            doit(newpath)
            after = self.existing(paths)
            self.assertTrue(after.issuperset(before))

            self.h5file.undo()
            post(newpath)
            after = self.existing(paths)
            self.assertEqual(after, before)

    def test00_create(self):
        """Test creating a node."""
        def pre():
            pass
        def doit(newpath):
            self.h5file.createArray(newpath, 'array', [1], createparents=True)
            self.assertTrue(joinPath(newpath, 'array') in self.h5file)
        def post(newpath):
            self.assertTrue(joinPath(newpath, 'array') not in self.h5file)
        self.basetest(doit, pre, post)

    def test01_move(self):
        """Test moving a node."""
        def pre():
            self.h5file.createArray('/', 'array', [1])
        def doit(newpath):
            self.h5file.moveNode('/array', newpath, createparents=True)
            self.assertTrue('/array' not in self.h5file)
            self.assertTrue(joinPath(newpath, 'array') in self.h5file)
        def post(newpath):
            self.assertTrue('/array' in self.h5file)
            self.assertTrue(joinPath(newpath, 'array') not in self.h5file)
        self.basetest(doit, pre, post)

    def test02_copy(self):
        """Test copying a node."""
        def pre():
            self.h5file.createArray('/', 'array', [1])
        def doit(newpath):
            self.h5file.copyNode('/array', newpath, createparents=True)
            self.assertTrue(joinPath(newpath, 'array') in self.h5file)
        def post(newpath):
            self.assertTrue(joinPath(newpath, 'array') not in self.h5file)
        self.basetest(doit, pre, post)

    def test03_copyChildren(self):
        """Test copying the children of a group."""
        def pre():
            g = self.h5file.createGroup('/', 'group')
            self.h5file.createArray(g, 'array1', [1])
            self.h5file.createArray(g, 'array2', [1])
        def doit(newpath):
            self.h5file.copyChildren('/group', newpath, createparents=True)
            self.assertTrue(joinPath(newpath, 'array1') in self.h5file)
            self.assertTrue(joinPath(newpath, 'array2') in self.h5file)
        def post(newpath):
            self.assertTrue(joinPath(newpath, 'array1') not in self.h5file)
            self.assertTrue(joinPath(newpath, 'array2') not in self.h5file)
        self.basetest(doit, pre, post)


def suite():
    theSuite = unittest.TestSuite()
    niter = 1
    #common.heavy = 1  # uncomment this only for testing purposes

    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(BasicTestCase))
        theSuite.addTest(unittest.makeSuite(PersistenceTestCase))
        theSuite.addTest(unittest.makeSuite(createArrayTestCase))
        theSuite.addTest(unittest.makeSuite(createGroupTestCase))
        theSuite.addTest(unittest.makeSuite(renameNodeTestCase))
        theSuite.addTest(unittest.makeSuite(moveNodeTestCase))
        theSuite.addTest(unittest.makeSuite(removeNodeTestCase))
        theSuite.addTest(unittest.makeSuite(copyNodeTestCase))
        theSuite.addTest(unittest.makeSuite(AttributesTestCase))
        theSuite.addTest(unittest.makeSuite(ComplexTestCase))
        theSuite.addTest(unittest.makeSuite(NotLoggedTestCase))
        theSuite.addTest(unittest.makeSuite(CreateParentsTestCase))
    if common.heavy:
        pass

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )

## Local Variables:
## mode: python
## End:
