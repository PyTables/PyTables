########################################################################
#
#       License: BSD
#       Created: October 1, 2002
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""
PyTables, hierarchical datasets in Python

:URL: http://www.pytables.org/

PyTables is a package for managing hierarchical datasets and designed
to efficiently cope with extremely large amounts of data.

You can support the development of PyTables by purchasing the
PyTables Professional edition available at
http://www.pytables.org/moin/PyTablesPro

Most Important Classes
======================

Nodes
~~~~~

Group, Table, Array, CArray, EArray, VLArray, UnImplemented

Declarative
~~~~~~~~~~~

IsDescription, {Type}Atom, {Type}Col

Helpers
~~~~~~~

File, Filters, Cols, Column


First Level Functions
=====================

openFile, copyFile, test,  print_versions, whichLibVersion,
isPyTablesFile, isHDF5File

Misc variables
==============

__version__, hdf5Version, is_pro

"""

import sys, os
if os.name == 'nt':
     module_path = os.path.abspath(os.path.dirname(__file__))
     os.environ['PATH'] = ';'.join((os.environ['PATH'], module_path))
     sys.path.append(module_path)


# Necessary imports to get versions stored on the Pyrex extension
from tables.utilsExtension import getPyTablesVersion, getHDF5Version

__version__ = getPyTablesVersion()
"""The PyTables version number."""
hdf5Version = getHDF5Version()
"""The underlying HDF5 library version number."""
is_pro = __version__.endswith('pro')
"""True for PyTables Professional edition, false otherwise."""

from tables.utilsExtension import (
    isHDF5File, isPyTablesFile, whichLibVersion, lrange,
    setBloscMaxThreads )

from tables.misc.enum import Enum
from tables.atom import *
from tables.flavor import restrict_flavors
from tables.description import *
from tables.filters import Filters

# Import the user classes from the proper modules
from tables.exceptions import *
from tables.file import File, openFile, copyFile
from tables.node import Node
from tables.group import Group
from tables.leaf import Leaf
from tables.table import Table, Cols, Column
from tables.array import Array
from tables.carray import CArray
from tables.earray import EArray
from tables.vlarray import VLArray
from tables.unimplemented import UnImplemented, Unknown
from tables.expression import Expr
from tables.tests import print_versions, test


# List here only the objects we want to be publicly available
__all__ = [
    # Exceptions and warnings:
    'HDF5ExtError',
    'ClosedNodeError', 'ClosedFileError', 'FileModeError',
    'NaturalNameWarning', 'NodeError', 'NoSuchNodeError',
    'UndoRedoError', 'UndoRedoWarning',
    'PerformanceWarning',
    'FlavorError', 'FlavorWarning',
    'FiltersWarning', 'DataTypeWarning',
    # Functions:
    'isHDF5File', 'isPyTablesFile', 'whichLibVersion',
    'copyFile', 'openFile', 'print_versions', 'test',
    'split_type', 'restrict_flavors', 'lrange',
    # Helper classes:
    'IsDescription', 'Description', 'Filters', 'Cols', 'Column',
    # Types:
    'Enum',
    # Atom types:
    'Atom', 'StringAtom', 'BoolAtom',
    'IntAtom', 'UIntAtom', 'Int8Atom', 'UInt8Atom', 'Int16Atom', 'UInt16Atom',
    'Int32Atom', 'UInt32Atom', 'Int64Atom', 'UInt64Atom',
    'FloatAtom', 'Float32Atom', 'Float64Atom',
    'ComplexAtom', 'Complex32Atom', 'Complex64Atom', 'Complex128Atom',
    'TimeAtom', 'Time32Atom', 'Time64Atom',
    'EnumAtom',
    'PseudoAtom', 'ObjectAtom', 'VLStringAtom', 'VLUnicodeAtom',
    # Column types:
    'Col', 'StringCol', 'BoolCol',
    'IntCol', 'UIntCol', 'Int8Col', 'UInt8Col', 'Int16Col', 'UInt16Col',
    'Int32Col', 'UInt32Col', 'Int64Col', 'UInt64Col',
    'FloatCol', 'Float32Col', 'Float64Col',
    'ComplexCol', 'Complex32Col', 'Complex64Col', 'Complex128Col',
    'TimeCol', 'Time32Col', 'Time64Col',
    'EnumCol',
    # Node classes:
    'Node', 'Group', 'Leaf', 'Table', 'Array', 'CArray', 'EArray', 'VLArray',
    'UnImplemented',
    # The File class:
    'File',
    # Expr class
    'Expr',
    ]
