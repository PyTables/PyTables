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

"""PyTables, hierarchical datasets in Python.

:URL: http://www.pytables.org/

PyTables is a package for managing hierarchical datasets and designed
to efficiently cope with extremely large amounts of data.

"""


import os
import sys


# On Windows, pre-load the HDF5 DLLs into the process via Ctypes
# to improve diagnostics and avoid issues when loading DLLs during runtime.
# However, if running from a frozen app (via PyInstaller), skip the check.
if os.name == 'nt' and not getattr(sys, 'frozen', False):
    import ctypes

    def _load_library(dllname, loadfunction, dllpaths=('', )):
        """Load a DLL via ctypes load function. Return None on failure.

        By default, try to load the DLL from the current package
        directory first, then from the Windows DLL search path.

        Python >= 3.8 no longer searches PATH for DLLs. Skip PATH modification
        on Python >= 3.8.

        """
        try:
            dllpaths = (os.path.abspath(
                os.path.dirname(__file__)), ) + dllpaths
        except NameError:
            pass  # PyPy and frozen distributions have no __file__ attribute
        oldenv = None
        for path in dllpaths:
            if path and sys.version_info < (3, 8):
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
    # loading tables.utilsextensions.
    import sys
    hdf5_dlls = ['hdf5.dll', 'hdf5dll.dll', 'pytables_hdf5.dll']
    if hasattr(sys, 'gettotalrefcount'):  # running with debug interpreter
        hdf5_dlls = ['hdf5_D.dll', 'hdf5ddll.dll']
    for dll in hdf5_dlls:
        if _load_library(dll, ctypes.cdll.LoadLibrary):
            break
    else:
        if sys.version_info < (3, 8):
            dll_loc = 'can be found in the system path'
        else:
            dll_loc = 'is installed in the package folder'
        raise ImportError(
            'Could not load any of %s, please ensure that it %s.' % (hdf5_dlls, dll_loc))

    # Some PyTables binary distributions place the dependency DLLs in the
    # tables package directory.
    # The lzo2 and libbz2 DLLs are loaded dynamically at runtime but can't be
    # found because the package directory is not in the Windows DLL search
    # path.
    # This pre-loads lzo2 and libbz2 DLLs from the tables package directory.
    if not _load_library('lzo2.dll', ctypes.cdll.LoadLibrary):
        pass

    if not _load_library('libbz2.dll', ctypes.cdll.LoadLibrary):
        pass


# Necessary imports to get versions stored on the cython extension
from .utilsextension import (
    get_pytables_version, get_hdf5_version, blosc_compressor_list,
    blosc_compcode_to_compname_ as blosc_compcode_to_compname,
    blosc_get_complib_info_ as blosc_get_complib_info)


__version__ = get_pytables_version()
"""The PyTables version number."""

hdf5_version = get_hdf5_version()
"""The underlying HDF5 library version number.

.. versionadded:: 3.0

"""

hdf5Version = hdf5_version
"""The underlying HDF5 library version number.

.. deprecated:: 3.0

    hdf5Version is pending deprecation, use :data:`hdf5_version`
    instead.

"""

from .utilsextension import (is_hdf5_file, is_pytables_file,
                             which_lib_version, set_blosc_max_threads,
                             silence_hdf5_messages)

from .misc.enum import Enum
from .atom import *
from .flavor import restrict_flavors
from .description import *
from .filters import Filters

# Import the user classes from the proper modules
from .exceptions import *
from .file import File, open_file, copy_file
from .node import Node
from .group import Group
from .leaf import Leaf
from .table import Table, Cols, Column
from .array import Array
from .carray import CArray
from .earray import EArray
from .vlarray import VLArray
from .unimplemented import UnImplemented, Unknown
from .expression import Expr
from .tests import print_versions, test


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
    'is_hdf5_file', 'is_pytables_file', 'which_lib_version',
    'copy_file', 'open_file', 'print_versions', 'test',
    'split_type', 'restrict_flavors', 'set_blosc_max_threads',
    'silence_hdf5_messages',
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


from .utilsextension import _broken_hdf5_long_double
if not _broken_hdf5_long_double():
    if 'Float96Atom' in locals():
        __all__.extend(('Float96Atom', 'Float96Col'))
        __all__.extend(('Complex192Atom', 'Complex192Col'))    # XXX check

    if 'Float128Atom' in locals():
        __all__.extend(('Float128Atom', 'Float128Col'))
        __all__.extend(('Complex256Atom', 'Complex256Col'))    # XXX check

else:

    from . import atom as _atom
    from . import description as _description
    try:
        del _atom.Float96Atom, _atom.Complex192Col
        del _description.Float96Col, _description.Complex192Col
        _atom.all_types.discard('complex192')
        _atom.ComplexAtom._isizes.remove(24)
    except AttributeError:
        try:
            del _atom.Float128Atom, _atom.Complex256Atom
            del _description.Float128Col, _description.Complex256Col
            _atom.all_types.discard('complex256')
            _atom.ComplexAtom._isizes.remove(32)
        except AttributeError:
            pass
    del _atom, _description
del _broken_hdf5_long_double
