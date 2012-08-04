# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: October 1, 2002
# Author: Francesc Alted - faltet@pytables.com
#
# $Id$
#
########################################################################

"""PyTables, hierarchical datasets in Python

:URL: http://www.pytables.org/

PyTables is a package for managing hierarchical datasets and designed
to efficiently cope with extremely large amounts of data.

"""


import sys, os
if os.name == 'nt':
    module_path = os.path.abspath(os.path.dirname(__file__))
    os.environ['PATH'] = ';'.join((os.environ['PATH'], module_path))
    sys.path.append(module_path)

# In order to improve diagnosis of a common Windows dependency
# issue, we explicitly test that we can load the HDF5 dll before
# loading tables.utilsExtensions.
if os.name == 'nt':
    import ctypes.util

    if not ctypes.util.find_library('hdf5dll.dll'):
        raise ImportError('Could not load "hdf5dll.dll", please ensure' +
                ' that it can be found in the system path')


# Necessary imports to get versions stored on the cython extension
from tables.utilsExtension import getPyTablesVersion, getHDF5Version


__version__ = getPyTablesVersion()
"""The PyTables version number."""
hdf5Version = getHDF5Version()
"""The underlying HDF5 library version number."""

from tables.utilsExtension import (
    isHDF5File, isPyTablesFile, whichLibVersion, lrange,
    setBloscMaxThreads, silenceHDF5Messages)

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
    'split_type', 'restrict_flavors', 'lrange', 'setBloscMaxThreads',
    'silenceHDF5Messages',
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
    'UnImplemented', 'Unknown',
    # The File class:
    'File',
    # Expr class
    'Expr',
    ]

if 'Float16Atom' in locals():
    # float16 is new in numpy 1.6.0
    __all__.extend(('Float16Atom', 'Float16Col'))
