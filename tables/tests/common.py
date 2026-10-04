"""Utilities for PyTables' test suites."""

import os
import tempfile
import unittest
from time import perf_counter as clock
from pathlib import Path

import numpy as np

import tables as tb
from tables.utilsextension import which_lib_version

verbose = os.environ.get("VERBOSE", "FALSE") == "TRUE"
"""Show detailed output of the testing process."""

heavy = False
"""Run all tests even when they take long to complete."""

show_memory = False
"""Show the progress of memory consumption."""


# zlib_avail = which_lib_version("zlib") is not None
lzo_avail = which_lib_version("lzo") is not None
bzip2_avail = which_lib_version("bzip2") is not None
blosc_avail = which_lib_version("blosc") is not None
blosc2_avail = which_lib_version("blosc2") is not None


def parse_argv(argv):
    global verbose, heavy

    if "verbose" in argv:
        verbose = True
        argv.remove("verbose")

    if "silent" in argv:  # take care of old flag, just in case
        verbose = False
        argv.remove("silent")

    if "--heavy" in argv:
        heavy = True
        argv.remove("--heavy")

    return argv


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
    print("-=" * 38)


def test_filename(filename):
    from importlib import resources

    return resources.files("tables.tests") / filename


def verbose_print(string, nonl=False):
    """Print out the `string` if verbose output is enabled."""
    if not verbose:
        return
    if nonl:
        print(string, end=" ")
    else:
        print(string)


def allequal(a, b, flavor="numpy"):
    """Checks if two numerical objects are equal."""

    # print("a-->", repr(a))
    # print("b-->", repr(b))
    if not hasattr(b, "shape"):
        # Scalar case
        return a == b

    if (not hasattr(a, "shape") or a.shape == ()) and (
        not hasattr(b, "shape") or b.shape == ()
    ):
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
        if verbose:
            print("Shape is not equal:", a.shape, "!=", b.shape)
        return 0

    # null arrays
    if a.size == 0:  # len(a) is not correct for generic shapes
        if b.size == 0:
            return 1
        if verbose:
            print("length is not equal")
            print("len(a.data) ==>", len(a.data))
            print("len(b.data) ==>", len(b.data))
        return 0

    # Multidimensional case
    result = a == b
    result = np.all(result)
    if not result and verbose:
        print("Some of the elements in arrays are not equal")

    return result


def areArraysEqual(arr1, arr2, *, check_type=True):
    """Are both `arr1` and `arr2` equal arrays?

    Arguments can be regular NumPy arrays, chararray arrays or
    structured arrays (including structured record arrays). They are
    checked for type and value equality.

    """

    t1 = type(arr1)
    t2 = type(arr2)

    if check_type and not (
        (hasattr(arr1, "dtype") and arr1.dtype == arr2.dtype)
        or issubclass(t1, t2)
        or issubclass(t2, t1)
    ):
        return False

    return np.all(arr1 == arr2)


class PyTablesTestCase(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        for key in self.__dict__:
            if self.__dict__[key].__class__.__name__ != "instancemethod":
                self.__dict__[key] = None

    def _getName(self):
        """Get the name of this test case."""
        return self.id().split(".")[-2]

    def _getMethodName(self):
        """Get the name of the method currently running in the test case."""
        return self.id().split(".")[-1]

    def _verboseHeader(self):
        """Print a nice header for the current test method if verbose."""

        if verbose:
            name = self._getName()
            methodName = self._getMethodName()

            title = f"Running {name}.{methodName}"
            print("{}\n{}".format(title, "-" * len(title)))

    def _checkEqualityGroup(self, node1, node2, hardlink=False):
        if verbose:
            print("Group 1:", node1)
            print("Group 2:", node2)
        if hardlink:
            self.assertTrue(
                node1._v_pathname != node2._v_pathname,
                "node1 and node2 have the same pathnames.",
            )
        else:
            self.assertTrue(
                node1._v_pathname == node2._v_pathname,
                "node1 and node2 does not have the same pathnames.",
            )
        self.assertTrue(
            node1._v_children == node2._v_children,
            "node1 and node2 does not have the same children.",
        )

    def _checkEqualityLeaf(self, node1, node2, hardlink=False):
        if verbose:
            print("Leaf 1:", node1)
            print("Leaf 2:", node2)
        if hardlink:
            self.assertTrue(
                node1._v_pathname != node2._v_pathname,
                "node1 and node2 have the same pathnames.",
            )
        else:
            self.assertTrue(
                node1._v_pathname == node2._v_pathname,
                "node1 and node2 does not have the same pathnames.",
            )
        self.assertTrue(
            areArraysEqual(node1[:], node2[:]),
            "node1 and node2 does not have the same values.",
        )


class TestFileMixin:
    h5fname = None
    open_kwargs = {}

    def setUp(self):
        super().setUp()
        self.h5file = tb.open_file(
            self.h5fname, title=self._getName(), **self.open_kwargs
        )

    def tearDown(self):
        """Close ``h5file``."""

        self.h5file.close()
        super().tearDown()


class TempFileMixin:
    open_mode = "w"
    open_kwargs = {}

    def _getTempFileName(self):
        return tempfile.mktemp(prefix=self._getName(), suffix=".h5")

    def setUp(self):
        """Set ``h5file`` and ``h5fname`` instance attributes.

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, empty, temporary HDF5 file.

        """

        super().setUp()
        self.h5fname = self._getTempFileName()
        self.h5file = tb.open_file(
            self.h5fname,
            self.open_mode,
            title=self._getName(),
            **self.open_kwargs,
        )

    def tearDown(self):
        """Close ``h5file`` and remove ``h5fname``."""

        self.h5file.close()
        self.h5file = None
        Path(self.h5fname).unlink()  # comment this for debug only
        super().tearDown()

    def _reopen(self, mode="r", **kwargs):
        """Reopen ``h5file`` in the specified ``mode``.

        Returns a true or false value depending on whether the file was
        reopenend or not.  If not, nothing is changed.

        """

        self.h5file.close()
        self.h5file = tb.open_file(self.h5fname, mode, **kwargs)
        return True


class ShowMemTime(PyTablesTestCase):
    tref = clock()
    """Test for showing memory and time consumption."""

    def test00(self):
        """Showing memory and time consumption."""

        # Obtain memory info (only for Linux 2.6.x)
        for line in Path("/proc/self/status").read_text().splitlines():
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
        print("\nWallClock time:", clock() - self.tref)
        print("Memory usage: ******* {self._getName()} *******")
        print(f"VmSize: {vmsize:>7} kB\tVmRSS: {vmrss:>7} kB")
        print(f"VmData: {vmdata:>7} kB\tVmStk: {vmstk:>7} kB")
        print(f"VmExe:  {vmexe:>7} kB\tVmLib: {vmlib:>7} kB")


def make_suite(test_case_class, *, prefix=None):
    loader = unittest.TestLoader()
    if prefix:
        loader.testMethodPrefix = prefix
    return loader.loadTestsFromTestCase(test_case_class)


_FIRST_MARKER = object()


def first(iterable, default=_FIRST_MARKER):
    for item in iterable:
        return item
    if default is _FIRST_MARKER:
        raise ValueError(
            "first() was called on an empty iterable, "
            "and no default value was provided."
        )
    return default
