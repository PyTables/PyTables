# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: 2005-05-24
# Author: Ivan Vilata i Balaguer - ivan@selidor.net
#
# $Id$
#
########################################################################

"""Utilities for PyTables' test suites"""

import os
import sys
import time
import unittest
import tempfile
import warnings

import os.path

try:
    # collections.Callable is new in python 2.6
    from collections import Callable
except ImportError:
    is_callable = callable
else:
    def is_callable(x):
        return isinstance(x, Callable)

import numpy

import tables

verbose = False
"""Show detailed output of the testing process."""

heavy = False
"""Run all tests even when they take long to complete."""

show_memory = False
"""Show the progress of memory consumption."""

if 'verbose' in sys.argv:
    verbose = True
    sys.argv.remove('verbose')

if 'silent' in sys.argv:  # take care of old flag, just in case
    verbose = False
    sys.argv.remove('silent')

if '--heavy' in sys.argv:
    heavy = True
    sys.argv.remove('--heavy')


def verbosePrint(string, nonl=False):
    """Print out the `string` if verbose output is enabled."""
    if not verbose:
        return
    if nonl:
        print string,
    else:
        print string


def cleanup(klass):
    # klass.__dict__.clear()     # This is too hard. Don't do that
#    print "Class attributes deleted"
    for key in klass.__dict__:
        if not klass.__dict__[key].__class__.__name__ in ('instancemethod'):
            klass.__dict__[key] = None


def allequal(a, b, flavor="numpy"):
    """Checks if two numerical objects are equal"""

    # print "a-->", repr(a)
    # print "b-->", repr(b)
    if not hasattr(b, "shape"):
        # Scalar case
        return a == b

    if ((not hasattr(a, "shape") or a.shape == ()) and
            (not hasattr(b, "shape") or b.shape == ())):
        return a == b

    if a.shape != b.shape:
        if verbose:
            print "Shape is not equal:", a.shape, "!=", b.shape
        return 0

    # Way to check the type equality without byteorder considerations
    if hasattr(b, "dtype") and a.dtype.str[1:] != b.dtype.str[1:]:
        if verbose:
            print "dtype is not equal:", a.dtype, "!=", b.dtype
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if a[()] == b[()]:
            return 1
        else:
            if verbose:
                print "Shape is not equal:", a.shape, "!=", b.shape
            return 0

    # null arrays
    if a.size == 0:  # len(a) is not correct for generic shapes
        if b.size == 0:
            return 1
        else:
            if verbose:
                print "length is not equal"
                print "len(a.data) ==>", len(a.data)
                print "len(b.data) ==>", len(b.data)
            return 0

    # Multidimensional case
    result = (a == b)
    result = numpy.all(result)
    if not result and verbose:
        print "Some of the elements in arrays are not equal"

    return result


def areArraysEqual(arr1, arr2):
    """
    Are both `arr1` and `arr2` equal arrays?

    Arguments can be regular NumPy arrays, chararray arrays or
    structured arrays (including structured record arrays).
    They are checked for type and value equality.
    """

    t1 = type(arr1)
    t2 = type(arr2)

    if not ((hasattr(arr1, 'dtype') and arr1.dtype == arr2.dtype) or
            issubclass(t1, t2) or issubclass(t2, t1)):
        return False

    return numpy.all(arr1 == arr2)


def pyTablesTest(oldmethod):
    def newmethod(self, *args, **kwargs):
        self._verboseHeader()
        try:
            try:
                return oldmethod(self, *args, **kwargs)
            except SkipTest, se:
                if se.args:
                    msg = se.args[0]
                else:
                    msg = "<skipped>"
                verbosePrint("\nSkipped test: %s" % msg)
            except self.failureException, fe:
                if fe.args:
                    msg = fe.args[0]
                else:
                    msg = "<failed>"
                verbosePrint("\nTest failed: %s" % msg)
                raise
            except Exception, exc:
                cname = exc.__class__.__name__
                verbosePrint("\nError in test::\n\n  %s: %s" % (cname, exc))
                raise
        finally:
            verbosePrint('')  # separator line between tests
    newmethod.__name__ = oldmethod.__name__
    newmethod.__doc__ = oldmethod.__doc__
    return newmethod


class SkipTest(Exception):
    """When this exception is raised, the test is skipped successfully."""
    pass


class MetaPyTablesTestCase(type):

    """Metaclass for PyTables test case classes."""

    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/198078

    def __new__(class_, name, bases, dict_):
        newdict = {}
        for (aname, avalue) in dict_.iteritems():
            if is_callable(avalue) and aname.startswith('test'):
                avalue = pyTablesTest(avalue)
            newdict[aname] = avalue
        return type.__new__(class_, name, bases, newdict)


class PyTablesTestCase(unittest.TestCase):

    """Abstract test case with useful methods."""

    __metaclass__ = MetaPyTablesTestCase

    def _getName(self):
        """Get the name of this test case."""
        return self.id().split('.')[-2]

    def _getMethodName(self):
        """Get the name of the method currently running in the test case."""
        return self.id().split('.')[-1]

    def _verboseHeader(self):
        """Print a nice header for the current test method if verbose."""

        if verbose:
            name = self._getName()
            methodName = self._getMethodName()

            title = "Running %s.%s" % (name, methodName)
            print '%s\n%s' % (title, '-' * len(title))

    @classmethod
    def _testFilename(class_, filename):
        """
        Returns an absolute version of the `filename`, taking care of
        the location of the calling test case class.
        """
        modname = class_.__module__
        # When the definitive switch to ``setuptools`` is made,
        # this should definitely use the ``pkg_resouces`` API::
        #
        #   return pkg_resources.resource_filename(modname, filename)
        #
        modfile = sys.modules[modname].__file__
        dirname = os.path.dirname(modfile)
        return os.path.join(dirname, filename)

    def failUnlessWarns(self, warnClass, callableObj, *args, **kwargs):
        """
        Fail unless a warning of class `warnClass` is issued.

        This method will fail if no warning belonging to the given
        `warnClass` is issued when invoking `callableObj` with arguments
        `args` and keyword arguments `kwargs`.  Warnings of the
        `warnClass` are hidden, while others are shown.

        This method returns the value returned by the call to
        `callableObj`.
        """

        issued = [False]  # let's avoid scoping problems ;)

        # Save the original warning-showing function.
        showwarning = warnings.showwarning

        # This warning-showing function hides and takes note
        # of expected warnings and acts normally on others.
        def myShowWarning(message, category, filename, lineno,
                          file=None, line=None):
            if issubclass(category, warnClass):
                issued[0] = True
                verbosePrint(
                    "Great!  The following ``%s`` was caught::\n"
                    "\n"
                    "  %s\n"
                    "\n"
                    "In file ``%s``, line number %d.\n"
                    % (category.__name__, message, filename, lineno))
            else:
                showwarning(message, category, filename, lineno, file, line)

        # By forcing Python to always show warnings of the wanted class,
        # and replacing the warning-showing function with a tailored one,
        # we can check for *every* occurence of the warning.
        warnings.filterwarnings('always', category=warnClass)
        warnings.showwarning = myShowWarning
        try:
            # Run code and see what happens.
            ret = callableObj(*args, **kwargs)
        finally:
            # Restore the original warning-showing function
            # and warning filter.
            warnings.showwarning = showwarning
            warnings.filterwarnings('default', category=warnClass)

        if not issued[0]:
            raise self.failureException(
                "``%s`` was not issued" % warnClass.__name__)

        # We only get here if the call to `callableObj` was successful
        # and it issued the expected warning.
        return ret

    assertWarns = failUnlessWarns

    def failUnlessRaises(self, excClass, callableObj, *args, **kwargs):
        if not verbose:
            # Use the ordinary implementation from `unittest.TestCase`.
            return super(PyTablesTestCase, self).assertRaises(
                excClass, callableObj, *args, **kwargs)

        try:
            callableObj(*args, **kwargs)
        except excClass, exc:
            print (
                "Great!  The following ``%s`` was caught::\n"
                "\n"
                "  %s\n"
                % (exc.__class__.__name__, exc))
        else:
            raise self.failureException(
                "``%s`` was not raised" % excClass.__name__)

    assertRaises = failUnlessRaises

    def _checkEqualityGroup(self, node1, node2, hardlink=False):
        if verbose:
            print "Group 1:", node1
            print "Group 2:", node2
        if hardlink:
            self.assertTrue(node1._v_pathname != node2._v_pathname,
                            "node1 and node2 have the same pathnames.")
        else:
            self.assertTrue(node1._v_pathname == node2._v_pathname,
                "node1 and node2 does not have the same pathnames.")
        self.assertTrue(node1._v_children == node2._v_children,
                "node1 and node2 does not have the same children.")

    def _checkEqualityLeaf(self, node1, node2, hardlink=False):
        if verbose:
            print "Leaf 1:", node1
            print "Leaf 2:", node2
        if hardlink:
            self.assertTrue(node1._v_pathname != node2._v_pathname,
                "node1 and node2 have the same pathnames.")
        else:
            self.assertTrue(node1._v_pathname == node2._v_pathname,
                "node1 and node2 does not have the same pathnames.")
        self.assertTrue(areArraysEqual(node1[:], node2[:]),
            "node1 and node2 does not have the same values.")


class TempFileMixin:
    def setUp(self):
        """
        Set ``h5file`` and ``h5fname`` instance attributes.

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, empty, temporary HDF5 file.
        """

        self.h5fname = tempfile.mktemp(suffix='.h5')
        self.h5file = tables.open_file(
            self.h5fname, 'w', title=self._getName())

    def tearDown(self):
        """Close ``h5file`` and remove ``h5fname``."""

        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)   # comment this for debugging purposes only

    def _reopen(self, mode='r'):
        """Reopen ``h5file`` in the specified ``mode``.

        Returns a true or false value depending on whether the file was
        reopenend or not.  If not, nothing is changed.
        """

        self.h5file.close()
        self.h5file = tables.open_file(self.h5fname, mode)
        return True


class ShowMemTime(PyTablesTestCase):
    tref = time.time()
    """Test for showing memory and time consumption."""

    def test00(self):
        """Showing memory and time consumption."""

        # Obtain memory info (only for Linux 2.6.x)
        for line in open("/proc/self/status"):
            if line.startswith("VmSize:"):
                vmsize = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                vmrss = int(line.split()[1])
            elif line.startswith("VmData:"):
                vmdata = int(line.split()[1])
            elif line.startswith("VmStk:"):
                vmstk = int(line.split()[1])
            elif line.startswith("VmExe:"):
                vmexe = int(line.split()[1])
            elif line.startswith("VmLib:"):
                vmlib = int(line.split()[1])
        print "\nWallClock time:", time.time() - self.tref
        print "Memory usage: ******* %s *******" % self._getName()
        print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
        print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
        print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
