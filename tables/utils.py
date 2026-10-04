"""Utility functions."""

from __future__ import annotations

import os
import re
import sys
import math
import locale
import weakref
import platform
import warnings
import contextlib
from time import perf_counter as clock
from typing import TYPE_CHECKING, Any, TextIO, Literal
from pathlib import Path
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .flavor import array_of_flavor

if TYPE_CHECKING:
    from .atom import Atom

# The map between byteorders in NumPy and PyTables
byteorders = {
    ">": "big",
    "<": "little",
    "=": sys.byteorder,
    "|": "irrelevant",
}

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = np.int64

copy_if_needed: bool | None

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    copy_if_needed = None
elif np.lib.NumpyVersion(np.__version__) < "1.28.0":
    copy_if_needed = False
else:
    # 2.0.0 dev versions, handle cases where copy may or may not exist
    try:
        np.array([1]).__array__(copy=None)  # type: ignore[call-overload]
        copy_if_needed = None
    except TypeError:
        copy_if_needed = False


def correct_byteorder(ptype: str, byteorder: str) -> str:
    """Fix the byteorder depending on the PyTables types."""
    if ptype in ["string", "bool", "int8", "uint8", "object"]:
        return "irrelevant"
    return byteorder


def is_idx(index: Any) -> bool:
    """Check if an object can work as an index or not."""
    if type(index) is int:
        return True
    if hasattr(index, "__index__"):
        # Exclude the array([idx]) as working as an index.  Fixes #303.
        if hasattr(index, "shape") and index.shape != ():
            return False
        try:
            index.__index__()
            if isinstance(index, bool):
                warnings.warn(
                    "using a boolean instead of an integer will result in an "
                    "error in the future",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return True
        except TypeError:
            return False
    elif any(
        [
            isinstance(index, np.integer),
            # For Python 2.4 one should test 0-dim and 1-dim,
            # 1-elem arrays as well
            isinstance(index, np.ndarray)
            and (index.shape == ())
            and index.dtype.str[1] == "i",
        ]
    ):
        return True

    return False


def idx2long(index: int | float | np.ndarray) -> int:
    """Convert a possible index into a long int."""
    try:
        if hasattr(index, "item"):
            return index.item()
        return int(index)
    except Exception as exc:
        raise TypeError("not an integer type.") from exc


# This is used in VLArray and EArray to produce NumPy object compliant
# with atom from a generic python type.  If copy is stated as True, it
# is assured that it will return a copy of the object and never the same
# object or a new one sharing the same memory.
def convert_to_np_atom(
    arr: npt.ArrayLike, atom: Atom, copy: bool | None = copy_if_needed
) -> np.ndarray:
    """Convert a generic object into a NumPy object compliant with atom."""
    # First, convert the object into a NumPy array
    nparr = array_of_flavor(arr, "numpy")
    # Copy of data if necessary for getting a contiguous buffer, or if
    # dtype is not the correct one.
    if atom.shape == ():
        # Scalar atom case
        nparr = np.array(nparr, dtype=atom.dtype, copy=copy)
    else:
        # Multidimensional atom case.  Addresses #133.
        # We need to use this strange way to obtain a dtype compliant
        # array because NumPy doesn't honor the shape of the dtype when
        # it is multidimensional.  See:
        # http://scipy.org/scipy/numpy/ticket/926
        # for details.
        # All of this is done just to taking advantage of the NumPy
        # broadcasting rules.
        newshape = nparr.shape[: -len(atom.dtype.shape)]
        nparr2 = np.empty(newshape, dtype=[("", atom.dtype)])
        nparr2["f0"][:] = nparr
        # Return a view (i.e. get rid of the record type)
        nparr = nparr2.view(atom.dtype)
    return nparr


# The next is used in Array, EArray and VLArray, and it is a bit more
# high level than convert_to_np_atom
def convert_to_np_atom2(obj: npt.ArrayLike, atom: Atom) -> np.ndarray:
    """Convert a generic object into a NumPy object compliant with atom."""
    # Check whether the object needs to be copied to make the operation
    # safe to in-place conversion.
    copy = True if atom.type in ["time64"] else copy_if_needed
    nparr = convert_to_np_atom(obj, atom, copy)
    # Finally, check the byteorder and change it if needed
    byteorder = byteorders[nparr.dtype.byteorder]
    if byteorder in ["little", "big"] and byteorder != sys.byteorder:
        # The byteorder needs to be fixed (a copy is made
        # so that the original array is not modified)
        nparr = nparr.byteswap()

    return nparr


def check_file_access(
    filename: str, mode: Literal["r", "w", "a", "r+"] = "r"
) -> None:
    """Check for file access in the specified `mode`.

    `mode` is one of the modes supported by `File` objects.  If the file
    indicated by `filename` can be accessed using that `mode`, the
    function ends successfully.  Else, an ``IOError`` is raised
    explaining the reason of the failure.

    All this paraphernalia is used to avoid the lengthy and scaring HDF5
    messages produced when there are problems opening a file.  No
    changes are ever made to the file system.

    """
    path = Path(filename).resolve()

    if mode == "r":
        # The file should be readable.
        if not os.access(path, os.F_OK):
            raise FileNotFoundError(f"``{path}`` does not exist")
        if not path.is_file():
            raise IsADirectoryError(f"``{path}`` is not a regular file")
        if not os.access(path, os.R_OK):
            raise PermissionError(
                f"file ``{path}`` exists but it can not be read"
            )
    elif mode == "w":
        if os.access(path, os.F_OK):
            # Since the file is not removed but replaced,
            # it must already be accessible to read and write operations.
            check_file_access(path, "r+")
        else:
            # A new file is going to be created,
            # so the directory should be writable.
            if not os.access(path.parent, os.F_OK):
                raise FileNotFoundError(f"``{path.parent}`` does not exist")
            if not path.parent.is_dir():
                raise NotADirectoryError(
                    f"``{path.parent}`` is not a directory"
                )
            if not os.access(path.parent, os.W_OK):
                raise PermissionError(
                    f"directory ``{path.parent}`` exists but it can not be "
                    f"written"
                )
    elif mode == "a":
        if os.access(path, os.F_OK):
            check_file_access(path, "r+")
        else:
            check_file_access(path, "w")
    elif mode == "r+":
        check_file_access(path, "r")
        if not os.access(path, os.W_OK):
            raise PermissionError(
                f"file ``{path}`` exists but it can not be written"
            )
    else:
        raise ValueError(f"invalid mode: {mode!r}")


def lazyattr(fget: Callable[[Any], Any]) -> property:
    """Create a *lazy attribute* from the result of `fget`.

    This function is intended to be used as a *method decorator*.  It
    returns a *property* which caches the result of calling the `fget`
    instance method.  The docstring of `fget` is used for the property
    itself.  For instance:

    >>> class MyClass(object):
    ...     @lazyattr
    ...     def attribute(self):
    ...         "Attribute description."
    ...         print("creating value")
    ...         return 10
    >>> type(MyClass.attribute)
    <class 'property'>
    >>> MyClass.attribute.__doc__
    'Attribute description.'
    >>> obj = MyClass()
    >>> obj.__dict__
    {}
    >>> obj.attribute
    creating value
    10
    >>> obj.__dict__
    {'attribute': 10}
    >>> obj.attribute
    10
    >>> del obj.attribute
    Traceback (most recent call last):
      ...
    AttributeError: ...

    .. warning::

        Please note that this decorator *changes the type of the
        decorated object* from an instance method into a property.

    """
    name = fget.__name__

    def newfget(self):
        mydict = self.__dict__
        if name in mydict:
            return mydict[name]
        mydict[name] = value = fget(self)
        return value

    return property(newfget, None, None, fget.__doc__)


def show_stats(explain: str, tref: float, encoding=None) -> float:
    """Show the used memory (only works for Linux 2.6.x)."""
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
    print(f"Memory usage: ******* {explain} *******")
    print(f"VmSize: {vmsize:>7} kB\tVmRSS: {vmrss:>7} kB")
    print(f"VmData: {vmdata:>7} kB\tVmStk: {vmstk:>7} kB")
    print(f"VmExe:  {vmexe:>7} kB\tVmLib: {vmlib:>7} kB")
    tnow = clock()
    print(f"WallClock time: {tnow - tref:.3f}")
    return tnow


# truncate data before calling __setitem__, to improve compression ratio
# this function is taken verbatim from netcdf4-python
def quantize(data: npt.ArrayLike, least_significant_digit: int):
    """Quantize data to improve compression.

    Data is quantized using around(scale*data)/scale, where scale is
    2**bits, and bits is determined from the least_significant_digit.

    For example, if least_significant_digit=1, bits will be 4.

    """
    exp = -least_significant_digit
    exp = math.floor(exp) if exp < 0 else math.ceil(exp)
    bits = math.ceil(math.log2(10**-exp))
    scale = 2**bits
    return np.around(scale * data) / scale


# Utilities to detect leaked instances.  See recipe 14.10 of the Python
# Cookbook by Martelli & Ascher.
tracked_classes: dict[str, list[weakref.ReferenceType]] = {}


def log_instance_creation(instance: Any, name: str | None = None) -> None:
    """Log instance creation."""
    if name is None:
        name = instance.__class__.__name__
        if name not in tracked_classes:
            tracked_classes[name] = []
        tracked_classes[name].append(weakref.ref(instance))


def string_to_classes(s: str) -> list[str]:
    """Return the list of tracked classes matching the input string."""
    if s == "*":
        return sorted(tracked_classes)
    return s.split()


def fetch_logged_instances(classes: str = "*") -> list[tuple[str, int]]:
    """Return the list of logged instances."""
    classnames = string_to_classes(classes)
    return [(cn, len(tracked_classes[cn])) for cn in classnames]


def count_logged_instances(classes: str, file: TextIO = sys.stdout) -> None:
    """Write to file the number of logged instances."""
    file.writelines(
        f"{classname}: {len(tracked_classes[classname])}\n"
        for classname in string_to_classes(classes)
    )


def list_logged_instances(classes: str, file: TextIO = sys.stdout) -> None:
    """Write to file the list of loggen instances."""
    for classname in string_to_classes(classes):
        file.write(f"\n{classname}:\n")
        for ref in tracked_classes[classname]:
            obj = ref()
            if obj is not None:
                file.write(f"    {obj!r}\n")


def dump_logged_instances(classes: str, file: TextIO = sys.stdout) -> None:
    """Dump the logged instances."""
    for classname in string_to_classes(classes):
        file.write(f"\n{classname}:\n")
        for ref in tracked_classes[classname]:
            obj = ref()
            if obj is not None:
                file.write(f"    {obj}:\n")
                file.writelines(
                    f"        {key:>20} : {value}\n"
                    for key, value in obj.__dict__.items()
                )


#
# A class useful for cache usage
#
class CacheDict(dict):
    """A dictionary that prevents itself from growing too much."""

    def __init__(self, maxentries: int) -> None:
        self.maxentries = maxentries
        super().__init__(self)

    def __setitem__(self, key: str, value: Any) -> None:
        # Protection against growing the cache too much
        if len(self) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = self.maxentries / 10
            for k in list(self)[:entries_to_remove]:
                super().__delitem__(k)
        super().__setitem__(key, value)


class NailedDict:
    """A dictionary which ignores its items when it has nails on it."""

    def __init__(self, maxentries: int) -> None:
        self.maxentries = maxentries
        self._cache: dict = {}
        self._nailcount = 0

    # Only a restricted set of dictionary methods are supported.  That
    # is why we buy instead of inherit.

    # The following are intended to be used by ``Table`` code changing
    # the set of usable indexes.

    def clear(self) -> None:
        """Clear the dictionary."""
        self._cache.clear()

    def nail(self) -> None:
        """Increase the nail count."""
        self._nailcount += 1

    def unnail(self) -> None:
        """Decrease the nail count."""
        self._nailcount -= 1

    # The following are intended to be used by ``Table`` code handling
    # conditions.

    def __contains__(self, key: Any) -> bool:
        if self._nailcount > 0:
            return False
        return key in self._cache

    def __getitem__(self, key: Any) -> Any:
        if self._nailcount > 0:
            raise KeyError(key)
        return self._cache[key]

    def get(self, key: Any, default: Any | None = None) -> Any:
        """Return the value for the specified key."""
        if self._nailcount > 0:
            return default
        return self._cache.get(key, default)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._nailcount > 0:
            return
        cache = self._cache
        # Protection against growing the cache too much
        if len(cache) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = max(self.maxentries // 10, 1)
            for k in list(cache)[:entries_to_remove]:
                del cache[k]
        cache[key] = value


def detect_number_of_cores() -> int:
    """Detect the number of cores on a system."""
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


@contextlib.contextmanager
def override_locale(category, locale_string):
    """Context manager to temporary override locales."""
    prev_locale_string = locale.setlocale(category)
    locale.setlocale(category, locale_string)
    yield
    locale.setlocale(category, prev_locale_string)


# locale.getdefaultlocale()
def getdefaultlocale():
    """Return the default locale.

    This is a replacement for the deprecated ``locale.getdefaultlocale()``.
    """
    with override_locale(locale.LC_ALL, ""):
        current_encoding = locale.getencoding()
        current_locale = locale.setlocale(locale.LC_ALL)
    return current_locale, current_encoding


def print_versions():
    """Print all the versions of software that PyTables relies on."""
    import numexpr as ne

    from ._version import __version__
    from .utilsextension import (
        which_lib_version,
    )
    from .utilsextension import (
        blosc_get_complib_info_ as blosc_get_complib_info,
    )
    from .utilsextension import (
        blosc2_get_complib_info_ as blosc2_get_complib_info,
    )

    print("-=" * 38)
    print(f"PyTables version:    {__version__}")
    print(f"HDF5 version:        {which_lib_version('hdf5')[1]}")
    print(f"NumPy version:       {np.__version__}")
    tinfo = which_lib_version("zlib")
    if ne.use_vml:
        # Get only the main version number and strip out all the rest
        vml_version = ne.get_vml_version()
        vml_version = re.findall("[0-9.]+", vml_version)[0]
        vml_avail = f"using VML/MKL {vml_version}"
    else:
        vml_avail = "not using Intel's VML/MKL"
    print(f"Numexpr version:     {ne.__version__} ({vml_avail})")
    if tinfo is not None:
        print(f"Zlib version:        {tinfo[1]} (in Python interpreter)")
    tinfo = which_lib_version("lzo")
    if tinfo is not None:
        print(f"LZO version:         {tinfo[1]} ({tinfo[2]})")
    tinfo = which_lib_version("bzip2")
    if tinfo is not None:
        print(f"BZIP2 version:       {tinfo[1]} ({tinfo[2]})")
    tinfo = which_lib_version("blosc")
    if tinfo is not None:
        blosc_date = tinfo[2].split()[1]
        print(f"Blosc version:       {tinfo[1]} ({blosc_date})")
        blosc_cinfo = blosc_get_complib_info()
        blosc_cinfo = [f"{k} ({v[1]})" for k, v in sorted(blosc_cinfo.items())]
        print("Blosc compressors:   {}".format(", ".join(blosc_cinfo)))
        blosc_finfo = ["shuffle", "bitshuffle"]
        print("Blosc filters:       {}".format(", ".join(blosc_finfo)))
    tinfo = which_lib_version("blosc2")
    if tinfo is not None:
        blosc2_date = tinfo[2].split()[1]
        print(f"Blosc2 version:      {tinfo[1]} ({blosc2_date})")
        blosc2_cinfo = blosc2_get_complib_info()
        blosc2_cinfo = [
            f"{k} ({v[1]})" for k, v in sorted(blosc2_cinfo.items())
        ]
        print("Blosc2 compressors:  {}".format(", ".join(blosc2_cinfo)))
        blosc2_finfo = ["shuffle", "bitshuffle"]
        print("Blosc2 filters:      {}".format(", ".join(blosc2_finfo)))
    try:
        from Cython import __version__ as cython_version

        print(f"Cython version:      {cython_version}")
    except ImportError:
        pass
    print(f"Python version:      {sys.version}")
    print(f"Platform:            {platform.platform()}")
    # if os.name == 'posix':
    #     (sysname, nodename, release, version, machine) = os.uname()
    #     print(f"Platform:          {sys.platform}-{machine}")
    print(f"Byte-ordering:       {sys.byteorder}")
    print(f"Detected cores:      {detect_number_of_cores()}")
    print(f"Default encoding:    {sys.getdefaultencoding()}")
    print(f"Default FS encoding: {sys.getfilesystemencoding()}")
    print(f"Default locale:      {getdefaultlocale()}")
    print("-=" * 38)

    # This should improve readability when tests are run by CI tools
    sys.stdout.flush()
