########################################################################
#
#       License: BSD
#       Created: October 1, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Import modules and functions needed by users.

To know what modules/functions are imported by a line like:

from tables import *

look at the __all__ variable that controls that.

Classes:

Functions:

Misc variables:

    __version__
    hdf5Version

"""

# Necessary imports to get versions stored on the Pyrex extension
from tables.utilsExtension import getPyTablesVersion, getHDF5Version

__version__ = getPyTablesVersion()
hdf5Version = getHDF5Version()

from tables.utilsExtension import isHDF5  # deprecated
from tables.utilsExtension import isHDF5File, isPyTablesFile, whichLibVersion

from tables.misc.enum import Enum
from tables.atom import *
from tables.IsDescription import *

# Import the user classes from the proper modules
from tables.exceptions import *
from tables.File import File, openFile, copyFile
from tables.Node import Node
from tables.Group import Group
from tables.Leaf import Leaf, Filters
from tables.Index import IndexProps
from tables.Table import Table, Cols, Column
from tables.Array import Array
from tables.CArray import CArray
from tables.EArray import EArray
from tables.VLArray import VLArray
from tables.UnImplemented import UnImplemented

# Import sub-packages
##import nodes


# List here only the objects we want to be publicly available
__all__ = [
    # Exceptions and warnings:
    'HDF5ExtError',
    'ClosedNodeError', 'ClosedFileError', 'FileModeError',
    'NaturalNameWarning', 'NodeError', 'NoSuchNodeError',
    'UndoRedoError', 'UndoRedoWarning',
    'PerformanceWarning',
    'FlavorError', 'FlavorWarning',
    # Functions:
    'isHDF5',  # deprecated
    'isHDF5File', 'isPyTablesFile', 'whichLibVersion', 'copyFile', 'openFile',
    # Helper classes:
    'IsDescription', 'Description', 'Filters', 'IndexProps', 'Cols', 'Column',
    # Types:
    'Enum',
    # Atom types:
    'split_type',
    'Atom', 'StringAtom', 'BoolAtom',
    'IntAtom', 'UIntAtom', 'Int8Atom', 'UInt8Atom', 'Int16Atom', 'UInt16Atom',
    'Int32Atom', 'UInt32Atom', 'Int64Atom', 'UInt64Atom',
    'FloatAtom', 'Float32Atom', 'Float64Atom',
    'ComplexAtom', 'Complex32Atom', 'Complex64Atom', 'Complex128Atom',
    'TimeAtom', 'Time32Atom', 'Time64Atom',
    'EnumAtom',
    'PseudoAtom', 'ObjectAtom', 'VLStringAtom',
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
    # Testing functions:
    #'createNestedType',
    ]
