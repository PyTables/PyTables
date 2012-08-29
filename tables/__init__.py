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


import os

# On Windows, pre-load the HDF5 DLLs into the process via Ctypes
# to improve diagnostics and avoid issues when loading DLLs during runtime.
if os.name == 'nt':
    import ctypes

    def _load_library(dllname, loadfunction, dllpaths=['']):
        """Load a DLL via ctypes load function. Return None on failure.

        By default, try to load the DLL from the current package directory
        first, then from the Windows DLL search path.

        """
        try:
            dllpaths = [os.path.abspath(os.path.dirname(__file__))] + dllpaths
        except NameError:
            pass  # PyPy and frozen distributions have no __file__ attribute
        for path in dllpaths:
            if path:
                # Temporarily add the path to the PATH environment variable
                # so Windows can find additional DLL dependencies.
                try:
                    oldenv = os.environ['PATH']
                    os.environ['PATH'] = path + ';' + oldenv
                except KeyError:
                    oldenv = None
            try:
                return loadfunction(os.path.join(path, dllname))
            except WindowsError:
                pass
            finally:
                if path and oldenv is not None:
                    os.environ['PATH'] = oldenv
        return None

    # In order to improve diagnosis of a common Windows dependency
    # issue, we explicitly test that we can load the HDF5 dll before
    # loading tables.utilsExtensions.
    if not _load_library('hdf5dll.dll', ctypes.cdll.LoadLibrary):
        raise ImportError(
            'Could not load "hdf5dll.dll", please ensure'
            ' that it can be found in the system path')

    # Some PyTables binary distributions place the dependency DLLs in the
    # tables package directory.
    # Lzo2.dll is loaded dynamically at runtime but can't be found because
    # the package directory is not in the Windows DLL search path.
    # This pre-loads lzo2.dll from the tables package directory.
    if not _load_library('lzo2.dll', ctypes.cdll.LoadLibrary):
        pass


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
