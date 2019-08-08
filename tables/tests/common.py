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

"""Utilities for PyTables' test suites."""

import os
import re
import sys
import time
import locale
import platform
import tempfile
import warnings
from distutils.version import LooseVersion

from pkg_resources import resource_filename

import unittest

import numpy
import numexpr

import tables
from tables.utils import detect_number_of_cores
from tables.req_versions import min_blosc_bitshuffle_version

hdf5_version = LooseVersion(tables.hdf5_version)
blosc_version = LooseVersion(tables.which_lib_version("blosc")[1])


verbose = False
"""Show detailed output of the testing process."""

heavy = False
"""Run all tests even when they take long to complete."""

show_memory = False
"""Show the progress of memory consumption."""


def parse_argv(argv):
    global verbose, heavy

    if 'verbose' in argv:
        verbose = True
        argv.remove('verbose')

    if 'silent' in argv:  # take care of old flag, just in case
        verbose = False
        argv.remove('silent')

    if '--heavy' in argv:
        heavy = True
        argv.remove('--heavy')

    return argv


zlib_avail = tables.which_lib_version("zlib") is not None
lzo_avail = tables.which_lib_version("lzo") is not None
bzip2_avail = tables.which_lib_version("bzip2") is not None
blosc_avail = tables.which_lib_version("blosc") is not None


def print_heavy(heavy):
    if heavy:
        print("""Performing the complete test suite!""")
    else:
        print("""\
Performing only a light (yet comprehensive) subset of the test suite.
If you want a more complete test, try passing the --heavy flag to this script
(or set the 'heavy' parameter in case you are using tables.test() call).
The whole suite will take more than 4 hours to complete on a relatively
modern CPU and around 512 MB of main memory.""")
    print('-=' * 38)


def print_versions():
    """Print all the versions of software that PyTables relies on."""

    print('-=' * 38)
    print("PyTables version:    %s" % tables.__version__)
    print("HDF5 version:        %s" % tables.which_lib_version("hdf5")[1])
    print("NumPy version:       %s" % numpy.__version__)
    tinfo = tables.which_lib_version("zlib")
    if numexpr.use_vml:
        # Get only the main version number and strip out all the rest
        vml_version = numexpr.get_vml_version()
        vml_version = re.findall("[0-9.]+", vml_version)[0]
        vml_avail = "using VML/MKL %s" % vml_version
    else:
        vml_avail = "not using Intel's VML/MKL"
    print("Numexpr version:     %s (%s)" % (numexpr.__version__, vml_avail))
    if tinfo is not None:
        print("Zlib version:        %s (%s)" % (tinfo[1],
                                                "in Python interpreter"))
    tinfo = tables.which_lib_version("lzo")
    if tinfo is not None:
        print("LZO version:         %s (%s)" % (tinfo[1], tinfo[2]))
    tinfo = tables.which_lib_version("bzip2")
    if tinfo is not None:
        print("BZIP2 version:       %s (%s)" % (tinfo[1], tinfo[2]))
    tinfo = tables.which_lib_version("blosc")
    if tinfo is not None:
        blosc_date = tinfo[2].split()[1]
        print("Blosc version:       %s (%s)" % (tinfo[1], blosc_date))
        blosc_cinfo = tables.blosc_get_complib_info()
        blosc_cinfo = [
            "%s (%s)" % (k, v[1]) for k, v in sorted(blosc_cinfo.items())
        ]
        print("Blosc compressors:   %s" % ', '.join(blosc_cinfo))
        blosc_finfo = ['shuffle']
        if tinfo[1] >= min_blosc_bitshuffle_version:
            blosc_finfo.append('bitshuffle')
        print("Blosc filters:       %s" % ', '.join(blosc_finfo))
    try:
        from Cython import __version__ as cython_version
        print('Cython version:      %s' % cython_version)
    except:
        pass
    print('Python version:      %s' % sys.version)
    print('Platform:            %s' % platform.platform())
    #if os.name == 'posix':
    #    (sysname, nodename, release, version, machine) = os.uname()
    #    print('Platform:          %s-%s' % (sys.platform, machine))
    print('Byte-ordering:       %s' % sys.byteorder)
    print('Detected cores:      %s' % detect_number_of_cores())
    print('Default encoding:    %s' % sys.getdefaultencoding())
    print('Default FS encoding: %s' % sys.getfilesystemencoding())
    print('Default locale:      (%s, %s)' % locale.getdefaultlocale())
    print('-=' * 38)

    # This should improve readability whan tests are run by CI tools
    sys.stdout.flush()


def test_filename(filename):
    return resource_filename('tables.tests', filename)


def verbosePrint(string, nonl=False):
    """Print out the `string` if verbose output is enabled."""
    if not verbose:
        return
    if nonl:
        print(string, end=' ')
    else:
        print(string)


def allequal(a, b, flavor="numpy"):
    """Checks if two numerical objects are equal."""

    # print("a-->", repr(a))
    # print("b-->", repr(b))
    if not hasattr(b, "shape"):
        # Scalar case
        return a == b

    if ((not hasattr(a, "shape") or a.shape == ()) and
            (not hasattr(b, "shape") or b.shape == ())):
        return a == b

    if a.shape != b.shape:
        if verbose:
            print("Shape is not equal:", a.shape, "!=", b.shape)
        return 0

    # Way to check the type equality without byteorder considerations
    if hasattr(b, "dtype") and a.dtype.str[1:] != b.dtype.str[1:]:
        if verbose:
            print("dtype is not equal:", a.dtype, "!=", b.dtype)
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if a[()] == b[()]:
            return 1
        else:
            if verbose:
                print("Shape is not equal:", a.shape, "!=", b.shape)
            return 0

    # null arrays
    if a.size == 0:  # len(a) is not correct for generic shapes
        if b.size == 0:
            return 1
        else:
            if verbose:
                print("length is not equal")
                print("len(a.data) ==>", len(a.data))
                print("len(b.data) ==>", len(b.data))
            return 0

    # Multidimensional case
    result = (a == b)
    result = numpy.all(result)
    if not result and verbose:
        print("Some of the elements in arrays are not equal")

    return result


def areArraysEqual(arr1, arr2):
    """Are both `arr1` and `arr2` equal arrays?

    Arguments can be regular NumPy arrays, chararray arrays or
    structured arrays (including structured record arrays). They are
    checked for type and value equality.

    """

    t1 = type(arr1)
    t2 = type(arr2)

    if not ((hasattr(arr1, 'dtype') and arr1.dtype == arr2.dtype) or
            issubclass(t1, t2) or issubclass(t2, t1)):
        return False

    return numpy.all(arr1 == arr2)


# COMPATIBILITY: assertWarns is new in Python 3.2
# Code copied from the standard unittest.case module (Python 3.4)
if not hasattr(unittest.TestCase, 'assertWarns'):
    class _BaseTestCaseContext:
        def __init__(self, test_case):
            self.test_case = test_case

        def _raiseFailure(self, standardMsg):
            msg = self.test_case._formatMessage(self.msg, standardMsg)
            raise self.test_case.failureException(msg)

    class _AssertRaisesBaseContext(_BaseTestCaseContext):
        def __init__(self, expected, test_case, callable_obj=None,
                     expected_regex=None):
            _BaseTestCaseContext.__init__(self, test_case)
            self.expected = expected
            self.test_case = test_case
            if callable_obj is not None:
                try:
                    self.obj_name = callable_obj.__name__
                except AttributeError:
                    self.obj_name = str(callable_obj)
            else:
                self.obj_name = None
            if expected_regex is not None:
                expected_regex = re.compile(expected_regex)
            self.expected_regex = expected_regex
            self.msg = None

        def handle(self, name, callable_obj, args, kwargs):
            """
            If callable_obj is None, assertRaises/Warns is being used as a
            context manager, so check for a 'msg' kwarg and return self.
            If callable_obj is not None, call it passing args and kwargs.
            """
            if callable_obj is None:
                self.msg = kwargs.pop('msg', None)
                return self
            with self:
                callable_obj(*args, **kwargs)

    class _AssertWarnsContext(_AssertRaisesBaseContext):
        def __enter__(self):
            for v in list(sys.modules.values()):
                if getattr(v, '__warningregistry__', None):
                    v.__warningregistry__ = {}
            self.warnings_manager = warnings.catch_warnings(record=True)
            self.warnings = self.warnings_manager.__enter__()
            warnings.simplefilter("always", self.expected)
            return self

        def __exit__(self, exc_type, exc_value, tb):
            self.warnings_manager.__exit__(exc_type, exc_value, tb)
            if exc_type is not None:
                # let unexpected exceptions pass through
                return
            try:
                exc_name = self.expected.__name__
            except AttributeError:
                exc_name = str(self.expected)
            first_matching = None
            for m in self.warnings:
                w = m.message
                if not isinstance(w, self.expected):
                    continue
                if first_matching is None:
                    first_matching = w
                if (self.expected_regex is not None and
                        not self.expected_regex.search(str(w))):
                    continue
                # store warning for later retrieval
                self.warning = w
                self.filename = m.filename
                self.lineno = m.lineno
                return
            # Now we simply try to choose a helpful failure message
            if first_matching is not None:
                self._raiseFailure(
                    '"{0}" does not match "{1}"'.format(
                        self.expected_regex.pattern, str(first_matching)))
            if self.obj_name:
                self._raiseFailure("{0} not triggered by {1}".format(
                                   exc_name, self.obj_name))
            else:
                self._raiseFailure("{0} not triggered".format(exc_name))


class PyTablesTestCase(unittest.TestCase):
    def tearDown(self):
        super(PyTablesTestCase, self).tearDown()
        for key in self.__dict__:
            if self.__dict__[key].__class__.__name__ not in ('instancemethod'):
                self.__dict__[key] = None

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
            print('%s\n%s' % (title, '-' * len(title)))

    # COMPATIBILITY: assertWarns is new in Python 3.2
    if not hasattr(unittest.TestCase, 'assertWarns'):
        def assertWarns(self, expected_warning, callable_obj=None,
                        *args, **kwargs):
            context = _AssertWarnsContext(expected_warning, self, callable_obj)
            return context.handle('assertWarns', callable_obj, args, kwargs)

    def _checkEqualityGroup(self, node1, node2, hardlink=False):
        if verbose:
            print("Group 1:", node1)
            print("Group 2:", node2)
        if hardlink:
            self.assertTrue(
                node1._v_pathname != node2._v_pathname,
                "node1 and node2 have the same pathnames.")
        else:
            self.assertTrue(
                node1._v_pathname == node2._v_pathname,
                "node1 and node2 does not have the same pathnames.")
        self.assertTrue(
            node1._v_children == node2._v_children,
            "node1 and node2 does not have the same children.")

    def _checkEqualityLeaf(self, node1, node2, hardlink=False):
        if verbose:
            print("Leaf 1:", node1)
            print("Leaf 2:", node2)
        if hardlink:
            self.assertTrue(
                node1._v_pathname != node2._v_pathname,
                "node1 and node2 have the same pathnames.")
        else:
            self.assertTrue(
                node1._v_pathname == node2._v_pathname,
                "node1 and node2 does not have the same pathnames.")
        self.assertTrue(
            areArraysEqual(node1[:], node2[:]),
            "node1 and node2 does not have the same values.")


class TestFileMixin(object):
    h5fname = None
    open_kwargs = {}

    def setUp(self):
        super(TestFileMixin, self).setUp()
        self.h5file = tables.open_file(
            self.h5fname, title=self._getName(), **self.open_kwargs)

    def tearDown(self):
        """Close ``h5file``."""

        self.h5file.close()
        super(TestFileMixin, self).tearDown()


class TempFileMixin(object):
    open_mode = 'w'
    open_kwargs = {}

    def _getTempFileName(self):
        return tempfile.mktemp(prefix=self._getName(), suffix='.h5')

    def setUp(self):
        """Set ``h5file`` and ``h5fname`` instance attributes.

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, empty, temporary HDF5 file.

        """

        super(TempFileMixin, self).setUp()
        self.h5fname = self._getTempFileName()
        self.h5file = tables.open_file(
            self.h5fname, self.open_mode, title=self._getName(),
            **self.open_kwargs)

    def tearDown(self):
        """Close ``h5file`` and remove ``h5fname``."""

        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)   # comment this for debugging purposes only
        super(TempFileMixin, self).tearDown()

    def _reopen(self, mode='r', **kwargs):
        """Reopen ``h5file`` in the specified ``mode``.

        Returns a true or false value depending on whether the file was
        reopenend or not.  If not, nothing is changed.

        """

        self.h5file.close()
        self.h5file = tables.open_file(self.h5fname, mode, **kwargs)
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
        print("\nWallClock time:", time.time() - self.tref)
        print("Memory usage: ******* %s *******" % self._getName())
        print("VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss))
        print("VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk))
        print("VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib))


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
