"""
Utilities for PyTables' test suites
===================================

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2005-05-24
:License:  BSD
:Revision: $Id$
"""

import unittest
import tempfile
import os
import os.path
import warnings
import sys
import popen2
import time

import numarray
import numarray.strings
import numarray.records
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


def verbosePrint(string):
    """Print out the `string` if verbose output is enabled."""
    if verbose: print string


def cleanup(klass):
    #klass.__dict__.clear()     # This is too hard. Don't do that
#    print "Class attributes deleted"
    for key in klass.__dict__.keys():
        if not klass.__dict__[key].__class__.__name__ in ('instancemethod'):
            klass.__dict__[key] = None


def allequal(a,b, flavor="numpy"):
    """Checks if two numerical objects are equal"""

    #print "a-->", repr(a)
    #print "b-->", repr(b)
    if not hasattr(b, "shape"):
        # Scalar case
        return a == b

    if flavor == "numeric":
        # Convert the parameters to numpy objects
        if a.typecode() == "c":
            a = numpy.array(a.tostring(), itemsize=1, shape=a.shape)
            b = numpy.array(b.tostring(), itemsize=1, shape=b.shape)
        else:
            a = numpy.asarray(a)
            b = numpy.asarray(b)

    elif flavor == "numarray":
        # Convert the parameters to numpy objects
        a = numpy.asarray(a)
        b = numpy.asarray(b)

    if ((not hasattr(a, "shape") or a.shape == ()) and
        (not hasattr(b, "shape") or b.shape == ())):
        return a == b

    if a.shape <> b.shape:
        if verbose:
            print "Shape is not equal:", a.shape, "<>", b.shape
        return 0

    if hasattr(b, "type") and a.type() <> b.type():
        if verbose:
            print "Type is not equal:", a.type(), "<>", b.type()
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if a[()] == b[()]:
            return 1
        else:
            if verbose:
                print "Shape is not equal:", a.shape, "<>", b.shape
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

    Arguments can be regular NumPy arrays, chararray arrays or record
    arrays (including nested record arrays).  They are checked for type
    and value equality.
    """

    t1 = type(arr1)
    t2 = type(arr2)

    if not ((t1 is t2) or issubclass(t1, t2) or issubclass(t2, t1)):
        return False

    return numpy.all(arr1 == arr2)


def testFilename(filename):
    """
    Returns an absolute version of the `filename`, taking care of
    the location of the test module.
    """
    # When the definitive switch to ``setuptools`` is made,
    # this should definitely use the ``pkg_resouces`` API::
    #
    #   return pkg_resources.resource_filename(__name__, filename)
    #
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, filename)



class PyTablesTestCase(unittest.TestCase):

    """Abstract test case with useful methods."""

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
            print '%s\n%s\n' % (title, '-'*len(title))


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
        def myShowWarning(message, category, filename, lineno, file=None):
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
                showwarning(message, category, filename, lineno, file)

        # Replace the original warning-showing function.
        warnings.showwarning = myShowWarning

        try:
            # Run code and see what happens.
            ret = callableObj(*args, **kwargs)
        finally:
            # Restore the original warning-showing function.
            warnings.showwarning = showwarning

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
            return super(PyTablesTestCase, self).failUnlessRaises(
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



class TempFileMixin:
    def setUp(self):
        """
        Set ``h5file`` and ``h5fname`` instance attributes.

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, empty, temporary HDF5 file.
        """

        self.h5fname = tempfile.mktemp(suffix='.h5')
        self.h5file = tables.openFile(
            self.h5fname, 'w', title=self._getName())


    def tearDown(self):
        """Close ``h5file`` and remove ``h5fname``."""

        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)


    def _reopen(self, mode='r'):
        """Reopen ``h5file`` in the specified ``mode``."""

        self.h5file.close()
        self.h5file = tables.openFile(self.h5fname, mode)



class ShowMemTime(PyTablesTestCase):
    tref = time.time()
    """Test for showing memory and time consumption."""

    def test00(self):
        """Showing memory and time consumption."""

        self._verboseHeader()

        # Build the command to obtain memory info (only for Linux 2.6.x)
        cmd = "cat /proc/%s/status" % os.getpid()
        sout, sin = popen2.popen2(cmd)
        for line in sout:
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
        sout.close()
        sin.close()
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
