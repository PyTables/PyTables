# -*- coding: utf-8 -*-

########################################################################
#
#       License: BSD
#       Created: March 4, 2003
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Utility functions."""

import os
import sys
import warnings
import subprocess
import re
from time import time

import numpy

from .flavor import array_of_flavor

# The map between byteorders in NumPy and PyTables
byteorders = {
    '>': 'big',
    '<': 'little',
    '=': sys.byteorder,
    '|': 'irrelevant',
}

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = numpy.int64


def correct_byteorder(ptype, byteorder):
    """Fix the byteorder depending on the PyTables types."""

    if ptype in ['string', 'bool', 'int8', 'uint8', 'object']:
        return "irrelevant"
    else:
        return byteorder


def is_idx(index):
    """Checks if an object can work as an index or not."""

    if type(index) is int:
        return True
    elif hasattr(index, "__index__"):  # Only works on Python 2.5 (PEP 357)
        # Exclude the array([idx]) as working as an index.  Fixes #303.
        if (hasattr(index, "shape") and index.shape != ()):
            return False
        try:
            index.__index__()
            if isinstance(index, bool):
                warnings.warn(
                    'using a boolean instead of an integer will result in an '
                    'error in the future', DeprecationWarning, stacklevel=2)
            return True
        except TypeError:
            return False
    elif isinstance(index, numpy.integer):
        return True
    # For Python 2.4 one should test 0-dim and 1-dim, 1-elem arrays as well
    elif (isinstance(index, numpy.ndarray) and (index.shape == ()) and
          index.dtype.str[1] == 'i'):
        return True

    return False


def idx2long(index):
    """Convert a possible index into a long int."""

    try:
        return int(index)
    except:
        raise TypeError("not an integer type.")


# This is used in VLArray and EArray to produce NumPy object compliant
# with atom from a generic python type.  If copy is stated as True, it
# is assured that it will return a copy of the object and never the same
# object or a new one sharing the same memory.
def convert_to_np_atom(arr, atom, copy=False):
    """Convert a generic object into a NumPy object compliant with atom."""

    # First, convert the object into a NumPy array
    nparr = array_of_flavor(arr, 'numpy')
    # Copy of data if necessary for getting a contiguous buffer, or if
    # dtype is not the correct one.
    if atom.shape == ():
        # Scalar atom case
        nparr = numpy.array(nparr, dtype=atom.dtype, copy=copy)
    else:
        # Multidimensional atom case.  Addresses #133.
        # We need to use this strange way to obtain a dtype compliant
        # array because NumPy doesn't honor the shape of the dtype when
        # it is multidimensional.  See:
        # http://scipy.org/scipy/numpy/ticket/926
        # for details.
        # All of this is done just to taking advantage of the NumPy
        # broadcasting rules.
        newshape = nparr.shape[:-len(atom.dtype.shape)]
        nparr2 = numpy.empty(newshape, dtype=[('', atom.dtype)])
        nparr2['f0'][:] = nparr
        # Return a view (i.e. get rid of the record type)
        nparr = nparr2.view(atom.dtype)
    return nparr



# The next is used in Array, EArray and VLArray, and it is a bit more
# high level than convert_to_np_atom
def convert_to_np_atom2(object, atom):
    """Convert a generic object into a NumPy object compliant with atom."""

    # Check whether the object needs to be copied to make the operation
    # safe to in-place conversion.
    copy = atom.type in ['time64']
    nparr = convert_to_np_atom(object, atom, copy)
    # Finally, check the byteorder and change it if needed
    byteorder = byteorders[nparr.dtype.byteorder]
    if (byteorder in ['little', 'big'] and byteorder != sys.byteorder):
        # The byteorder needs to be fixed (a copy is made
        # so that the original array is not modified)
        nparr = nparr.byteswap()

    return nparr



def check_file_access(filename, mode='r'):
    """Check for file access in the specified `mode`.

    `mode` is one of the modes supported by `File` objects.  If the file
    indicated by `filename` can be accessed using that `mode`, the
    function ends successfully.  Else, an ``IOError`` is raised
    explaining the reason of the failure.

    All this paraphernalia is used to avoid the lengthy and scaring HDF5
    messages produced when there are problems opening a file.  No
    changes are ever made to the file system.

    """

    if mode == 'r':
        # The file should be readable.
        if not os.access(filename, os.F_OK):
            raise IOError("``%s`` does not exist" % (filename,))
        if not os.path.isfile(filename):
            raise IOError("``%s`` is not a regular file" % (filename,))
        if not os.access(filename, os.R_OK):
            raise IOError("file ``%s`` exists but it can not be read"
                          % (filename,))
    elif mode == 'w':
        if os.access(filename, os.F_OK):
            # Since the file is not removed but replaced,
            # it must already be accessible to read and write operations.
            check_file_access(filename, 'r+')
        else:
            # A new file is going to be created,
            # so the directory should be writable.
            parentname = os.path.dirname(filename)
            if not parentname:
                parentname = '.'
            if not os.access(parentname, os.F_OK):
                raise IOError("``%s`` does not exist" % (parentname,))
            if not os.path.isdir(parentname):
                raise IOError("``%s`` is not a directory" % (parentname,))
            if not os.access(parentname, os.W_OK):
                raise IOError("directory ``%s`` exists but it can not be "
                              "written" % (parentname,))
    elif mode == 'a':
        if os.access(filename, os.F_OK):
            check_file_access(filename, 'r+')
        else:
            check_file_access(filename, 'w')
    elif mode == 'r+':
        check_file_access(filename, 'r')
        if not os.access(filename, os.W_OK):
            raise IOError("file ``%s`` exists but it can not be written"
                          % (filename,))
    else:
        raise ValueError("invalid mode: %r" % (mode,))



def lazyattr(fget):
    """Create a *lazy attribute* from the result of `fget`.

    This function is intended to be used as a *method decorator*.  It
    returns a *property* which caches the result of calling the `fget`
    instance method.  The docstring of `fget` is used for the property
    itself.  For instance:

    >>> class MyClass(object):
    ...     @lazyattr
    ...     def attribute(self):
    ...         'Attribute description.'
    ...         print('creating value')
    ...         return 10
    ...
    >>> type(MyClass.attribute)
    <type 'property'>
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
    AttributeError: can't delete attribute

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


def show_stats(explain, tref, encoding=None):
    """Show the used memory (only works for Linux 2.6.x)."""

    if encoding is None:
        encoding = sys.getdefaultencoding()

    # Build the command to obtain memory info
    cmd = "cat /proc/%s/status" % os.getpid()
    sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in sout:
        line = line.decode(encoding)
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
    print("Memory usage: ******* %s *******" % explain)
    print("VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss))
    print("VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk))
    print("VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib))
    tnow = time()
    print("WallClock time:", round(tnow - tref, 3))
    return tnow


# truncate data before calling __setitem__, to improve compression ratio
# this function is taken verbatim from netcdf4-python
def quantize(data, least_significant_digit):
    """quantize data to improve compression.

    Data is quantized using around(scale*data)/scale, where scale is
    2**bits, and bits is determined from the least_significant_digit.

    For example, if least_significant_digit=1, bits will be 4.

    """

    precision = pow(10., -least_significant_digit)
    exp = numpy.log10(precision)
    if exp < 0:
        exp = int(numpy.floor(exp))
    else:
        exp = int(numpy.ceil(exp))
    bits = numpy.ceil(numpy.log2(pow(10., -exp)))
    scale = pow(2., bits)
    datout = numpy.around(scale * data) / scale

    return datout


# Utilities to detect leaked instances.  See recipe 14.10 of the Python
# Cookbook by Martelli & Ascher.
tracked_classes = {}
import weakref


def log_instance_creation(instance, name=None):
    if name is None:
        name = instance.__class__.__name__
        if name not in tracked_classes:
            tracked_classes[name] = []
        tracked_classes[name].append(weakref.ref(instance))



def string_to_classes(s):
    if s == '*':
        c = sorted(tracked_classes.keys())
        return c
    else:
        return s.split()


def fetch_logged_instances(classes="*"):
    classnames = string_to_classes(classes)
    return [(cn, len(tracked_classes[cn])) for cn in classnames]



def count_logged_instances(classes, file=sys.stdout):
    for classname in string_to_classes(classes):
        file.write("%s: %d\n" % (classname, len(tracked_classes[classname])))



def list_logged_instances(classes, file=sys.stdout):
    for classname in string_to_classes(classes):
        file.write('\n%s:\n' % classname)
        for ref in tracked_classes[classname]:
            obj = ref()
            if obj is not None:
                file.write('    %s\n' % repr(obj))



def dump_logged_instances(classes, file=sys.stdout):
    for classname in string_to_classes(classes):
        file.write('\n%s:\n' % classname)
        for ref in tracked_classes[classname]:
            obj = ref()
            if obj is not None:
                file.write('    %s:\n' % obj)
                for key, value in obj.__dict__.items():
                    file.write('        %20s : %s\n' % (key, value))



#
# A class useful for cache usage
#
class CacheDict(dict):
    """A dictionary that prevents itself from growing too much."""

    def __init__(self, maxentries):
        self.maxentries = maxentries
        super(CacheDict, self).__init__(self)

    def __setitem__(self, key, value):
        # Protection against growing the cache too much
        if len(self) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = self.maxentries / 10
            for k in list(self.keys())[:entries_to_remove]:
                super(CacheDict, self).__delitem__(k)
        super(CacheDict, self).__setitem__(key, value)


class NailedDict(object):
    """A dictionary which ignores its items when it has nails on it."""

    def __init__(self, maxentries):
        self.maxentries = maxentries
        self._cache = {}
        self._nailcount = 0

    # Only a restricted set of dictionary methods are supported.  That
    # is why we buy instead of inherit.

    # The following are intended to be used by ``Table`` code changing
    # the set of usable indexes.

    def clear(self):
        self._cache.clear()

    def nail(self):
        self._nailcount += 1

    def unnail(self):
        self._nailcount -= 1

    # The following are intended to be used by ``Table`` code handling
    # conditions.

    def __contains__(self, key):
        if self._nailcount > 0:
            return False
        return key in self._cache

    def __getitem__(self, key):
        if self._nailcount > 0:
            raise KeyError(key)
        return self._cache[key]

    def get(self, key, default=None):
        if self._nailcount > 0:
            return default
        return self._cache.get(key, default)

    def __setitem__(self, key, value):
        if self._nailcount > 0:
            return
        cache = self._cache
        # Protection against growing the cache too much
        if len(cache) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = max(self.maxentries // 10, 1)
            for k in list(cache.keys())[:entries_to_remove]:
                del cache[k]
        cache[key] = value


def detect_number_of_cores():
    """Detects the number of cores on a system.

    Cribbed from pp.

    """

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



# Main part
# =========
def _test():
    """Run ``doctest`` on this module."""

    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
