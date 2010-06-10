"""
Functionality related with node paths in a PyTables file.

:Author: Ivan Vilata i Balaguer
:Contact: ivan at selidor dot net
:License: BSD
:Created: January 15, 2007
:Revision: $Id$

Variables
=========

`__docformat`__
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
"""

# Imports
# =======
import re
import warnings
import keyword

from tables.exceptions import NaturalNameWarning


# Public variables
# ================
__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""


# Private variables
# =================
_pythonIdRE = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')
"""Python identifier regular expression."""

_reservedIdRE = re.compile('^_[cfgv]_')
"""
PyTables reserved identifier regular expression.

- c: class variables
- f: class public methods
- g: class private methods
- v: instance variables
"""

_hiddenNameRE = re.compile('^_[pi]_')
"""
Nodes with a name *matching* this expression are considered hidden.

For instance, ``name`` whould be visible while ``_i_name`` would not.
"""

_hiddenPathRE = re.compile('/_[pi]_')
"""
Nodes with a path *containing* this expression are considered hidden.

For instance, a node with a pathname like ``/a/b/c`` would be visible
while nodes with pathnames like ``/a/c/_i_x`` or ``/a/_p_x/y`` would
not.
"""


# Public functions
# ================
def checkNameValidity(name):
    """
    Check the validity of the `name` of an object.

    If the name is not valid, a ``ValueError`` is raised.  If it is
    valid but it can not be used with natural naming, a
    `NaturalNameWarning` is issued.
    """

    warnInfo = (
        "you will not be able to use natural naming to access this object; "
        "using ``getattr()`` will still work, though" )

    if not isinstance(name, basestring):  # Python >= 2.3
        raise TypeError("object name is not a string: %r" % (name,))

    # Check whether `name` is a valid HDF5 name.
    # http://hdfgroup.org/HDF5/doc/UG/03_Model.html#Structure
    if name == '':
        raise ValueError("the empty string is not allowed as an object name")
    if name == '.':
        raise ValueError("``.`` is not allowed as an object name")
    if '/' in name:
        raise ValueError( "the ``/`` character is not allowed "
                          "in object names: %r" % name )

    # Check whether `name` is a valid Python identifier.
    if not _pythonIdRE.match(name):
        warnings.warn( "object name is not a valid Python identifier: %r; "
                       "it does not match the pattern ``%s``; %s"
                       % (name, _pythonIdRE.pattern, warnInfo),
                       NaturalNameWarning )
        return

    # However, Python identifiers and keywords have the same form.
    if keyword.iskeyword(name):
        warnings.warn( "object name is a Python keyword: %r; %s"
                       % (name, warnInfo), NaturalNameWarning )
        return

    # Still, names starting with reserved prefixes are not allowed.
    if _reservedIdRE.match(name):
        raise ValueError( "object name starts with a reserved prefix: %r; "
                          "it matches the pattern ``%s``"
                          % (name, _reservedIdRE.pattern) )

    # ``__members__`` is the only exception to that rule.
    if name == '__members__':
        raise ValueError("``__members__`` is not allowed as an object name")


def joinPath(parentPath, name):
    """
    Join a *canonical* `parentPath` with a *non-empty* `name`.

    >>> joinPath('/', 'foo')
    '/foo'
    >>> joinPath('/foo', 'bar')
    '/foo/bar'
    >>> joinPath('/foo', '/foo2/bar')
    '/foo/foo2/bar'
    >>> joinPath('/foo', '/')
    '/foo'
    """

    if name.startswith('./'):  # Support relative paths (mainly for links)
        name = name[2:]
    if parentPath == '/' and name.startswith('/'):
        pstr = '%s' % name
    elif parentPath == '/' or name.startswith('/'):
        pstr = '%s%s' % (parentPath, name)
    else:
        pstr = '%s/%s' % (parentPath, name)
    if pstr.endswith('/'):
        pstr = pstr[:-1]
    return pstr


def splitPath(path):
    """
    Split a *canonical* `path` into a parent path and a node name.

    The result is returned as a tuple.  The parent path does not
    include a trailing slash.

    >>> splitPath('/')
    ('/', '')
    >>> splitPath('/foo/bar')
    ('/foo', 'bar')
    """

    lastSlash = path.rfind('/')
    ppath = path[:lastSlash]
    name = path[lastSlash+1:]

    if ppath == '':
        ppath = '/'

    return (ppath, name)


def isVisibleName(name):
    """Does this `name` make the named node a visible one?"""
    return _hiddenNameRE.match(name) is None


def isVisiblePath(path):
    """Does this `path` make the named node a visible one?"""
    return _hiddenPathRE.search(path) is None


# Main part
# =========
def _test():
    """Run ``doctest`` on this module."""
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
