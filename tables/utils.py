########################################################################
#
#       License: BSD
#       Created: March 4, 2003
#       Author:  Francesc Alted - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Utility functions

"""

import os, os.path
import sys
import math

import numpy

from tables.flavor import array_of_flavor


# The map between byteorders in NumPy and PyTables
byteorders = {'>': 'big',
              '<': 'little',
              '=': sys.byteorder,
              '|': 'irrelevant'}

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = numpy.int64


def correct_byteorder(ptype, byteorder):
    "Fix the byteorder depending on the PyTables types."

    if ptype in ['string', 'bool', 'int8', 'uint8']:
        return "irrelevant"
    else:
        return byteorder


def is_idx(index):
    """Checks if an object can work as an index or not."""

    if type(index) in (int,long):
        return True
    elif hasattr(index, "__index__"):  # Only works on Python 2.5 on
        try:                           # (as per PEP 357)
            idx = index.__index__()
            return True
        except TypeError:
            return False
    elif isinstance(index, numpy.integer):
        return True
    # For Python 2.4 one should test 0-dim arrays as well
    elif (isinstance(index, numpy.ndarray) and
          index.shape == () and
          index.dtype.str[1] == 'i'):
        return True

    return False


def idx2long(index):
    """Convert a possible index into a long int"""

    if is_idx(index):
        return long(index)
    else:
        raise TypeError, "not an integer type."


# This is used in VLArray and EArray to produce NumPy object compliant
# with atom from a generic python type.  If copy is stated as True, it
# is assured that it will return a copy of the object and never the same
# object or a new one sharing the same memory.
def convertToNPAtom(arr, atom, copy=False):
    "Convert a generic object into a NumPy object compliant with atom."

    # First, convert the object to a NumPy array
    nparr = array_of_flavor(arr, 'numpy')

    # Get copies of data if necessary for getting a contiguous buffer,
    # or if dtype is not the correct one.
    basetype = atom.dtype.base
    if (copy or nparr.dtype != basetype):
        nparr = numpy.array(nparr, dtype=basetype)

    return nparr


# The next is used in Array, EArray and VLArray, and it is a bit more
# high level than convertToNPAtom
def convertToNPAtom2(object, atom):
    "Convert a generic object into a NumPy object compliant with atom."
    # Check whether the object needs to be copied to make the operation
    # safe to in-place conversion.
    copy = atom.type in ['time64']
    nparr = convertToNPAtom(object, atom, copy)
    # Finally, check the byteorder and change it if needed
    byteorder = byteorders[nparr.dtype.byteorder]
    if ( byteorder in ['little', 'big'] and byteorder != sys.byteorder ):
        # The byteorder needs to be fixed (a copy is made
        # so that the original array is not modified)
        nparr = nparr.byteswap()

    return nparr


def checkFileAccess(filename, mode='r'):
    """
    Check for file access in the specified `mode`.

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
            checkFileAccess(filename, 'r+')
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
                raise IOError("directory ``%s`` exists but it can not be written"
                              % (parentname,))
    elif mode == 'a':
        if os.access(filename, os.F_OK):
            checkFileAccess(filename, 'r+')
        else:
            checkFileAccess(filename, 'w')
    elif mode == 'r+':
        checkFileAccess(filename, 'r')
        if not os.access(filename, os.W_OK):
            raise IOError("file ``%s`` exists but it can not be written"
                          % (filename,))
    else:
        raise ValueError("invalid mode: %r" % (mode,))


def lazyattr(fget):
    """
    Create a *lazy attribute* from the result of `fget`.

    This function is intended to be used as a *method decorator*.  It
    returns a *property* which caches the result of calling the `fget`
    instance method.  The docstring of `fget` is used for the property
    itself.  For instance:

    >>> class MyClass(object):
    ...     @lazyattr
    ...     def attribute(self):
    ...         'Attribute description.'
    ...         print 'creating value'
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

    .. Warning:: Please note that this decorator *changes the type of
       the decorated object* from an instance method into a property.
    """
    name = fget.__name__
    def newfget(self):
        mydict = self.__dict__
        if name in mydict:
            return mydict[name]
        mydict[name] = value = fget(self)
        return value
    return property(newfget, None, None, fget.__doc__)


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
