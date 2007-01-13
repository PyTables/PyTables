########################################################################
#
#       License: BSD
#       Created: March 4, 2003
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Utility functions

"""

import re
import warnings
import keyword
import os, os.path
import sys

import numpy

import tables.utilsExtension
from tables.exceptions import NaturalNameWarning
from tables.flavor import array_of_flavor


# The map between byteorders in NumPy and PyTables
byteorders = {'>': 'big',
              '<': 'little',
              '=': sys.byteorder,
              '|': 'irrelevant'}


# Python identifier regular expression.
pythonIdRE = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')
# PyTables reserved identifier regular expression.
#   c: class variables
#   f: class public methods
#   g: class private methods
#   v: instance variables
reservedIdRE = re.compile('^_[cfgv]_')

# Nodes with a name *matching* this expression are considered hidden.
# For instance::
#
#   name -> visible
#   _i_name -> hidden
#
hiddenNameRE = re.compile('^_[pi]_')

# Nodes with a path *containing* this expression are considered hidden.
# For instance::
#
#   /a/b/c -> visible
#   /a/c/_i_x -> hidden
#   /a/_p_x/y -> hidden
#
hiddenPathRE = re.compile('/_[pi]_')


def checkNameValidity(name):
    """
    Check the validity of the `name` of an object.

    If the name is not valid, a ``ValueError`` is raised.  If it is
    valid but it can not be used with natural naming, a
    `NaturalNameWarning` is issued.
    """

    warnInfo = """\
you will not be able to use natural naming to access this object \
(but using ``getattr()`` will still work)"""

    if not isinstance(name, basestring):  # Python >= 2.3
        raise TypeError("object name is not a string: %r" % (name,))

    # Check whether `name` is a valid HDF5 name.
    # http://hdf.ncsa.uiuc.edu/HDF5/doc/UG/03_Model.html#Structure
    if name == '':
        raise ValueError("the empty string is not allowed as an object name")
    if name == '.':
        raise ValueError("``.`` is not allowed as an object name")
    if '/' in name:
        raise ValueError(
            "the ``/`` character is not allowed in object names: %r" % (name,))

    # Check whether `name` is a valid Python identifier.
    if not pythonIdRE.match(name):
        warnings.warn("""\
object name is not a valid Python identifier: %r; \
it does not match the pattern ``%s``; %s"""
                      % (name, pythonIdRE.pattern, warnInfo),
                      NaturalNameWarning)
        return

    # However, Python identifiers and keywords have the same form.
    if keyword.iskeyword(name):
        warnings.warn("object name is a Python keyword: %r; %s"
                      % (name, warnInfo), NaturalNameWarning)
        return

    # Still, names starting with reserved prefixes are not allowed.
    if reservedIdRE.match(name):
        raise ValueError("object name starts with a reserved prefix: %r; "
                         "it matches the pattern ``%s``"
                         % (name, reservedIdRE.pattern))

    # ``__members__`` is the only exception to that rule.
    if name == '__members__':
        raise ValueError("``__members__`` is not allowed as an object name")


def is_idx(index):
    """Checks if an object can work as an index or not."""

    if type(index) in (int,long):
        return True
    elif hasattr(index, "__index__"):  # Only works on Python 2.5 on (as per PEP 357)
        try:
            idx = index.__index__()
            return True
        except TypeError:
            return False
    elif isinstance(index, numpy.integer):
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
    if (copy or nparr.dtype <> basetype):
        nparr = numpy.array(nparr, dtype=basetype)

    return nparr


def joinPath(parentPath, name):
    """joinPath(parentPath, name) -> path.  Joins a canonical path with a name.

    Joins the given canonical path with the given child node name.
    """

    if parentPath == '/':
        pstr = '%s%s'
    else:
        pstr = '%s/%s'
    return pstr % (parentPath, name)


def splitPath(path):
    """splitPath(path) -> (parentPath, name).  Splits a canonical path.

    Splits the given canonical path into a parent path (without the trailing
    slash) and a node name.
    """

    lastSlash = path.rfind('/')
    ppath = path[:lastSlash]
    name = path[lastSlash+1:]

    if ppath == '':
        ppath = '/'

    return (ppath, name)


def isVisibleName(name):
    """Does this name make the named node a visible one?"""
    return hiddenNameRE.match(name) is None


def isVisiblePath(path):
    """Does this path make the named node a visible one?"""
    return hiddenPathRE.search(path) is None


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


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
