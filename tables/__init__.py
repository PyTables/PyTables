########################################################################
#
#       License: BSD
#       Created: October 1, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/__init__.py,v $
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
    HDF5Version
    ExtVersion

"""

# Necessary imports to get versions stored on the Pyrex extension
from tables.hdf5Extension import\
     getPyTablesVersion, getExtVersion, getHDF5Version

__version__ = getPyTablesVersion()
ExtVersion  = getExtVersion()
HDF5Version = getHDF5Version()


from tables.hdf5Extension import isHDF5, isPyTablesFile, whichLibVersion

from tables.Atom import *
from tables.IsDescription import *

# Import the user classes from the proper modules
from tables.exceptions import *
from tables.File import File, openFile, copyFile
from tables.Node import Node
from tables.Group import Group
from tables.Leaf import Leaf, Filters
from tables.Index import IndexProps
from tables.Table import Table
from tables.VLTable import VLTable
from tables.Array import Array
from tables.EArray import EArray
from tables.VLArray import VLArray
from tables.UnImplemented import UnImplemented

# Import sub-packages
##import nodes


# List here only the objects we want to be publicly available
__all__ = [
    # Exceptions and warnings:
    'NaturalNameWarning', 'NodeError', 'NoSuchNodeError',
    'UndoRedoError', 'UndoRedoWarning',
    'PerformanceWarning',
    # Functions:
    'isHDF5', 'isPyTablesFile', 'whichLibVersion', 'copyFile', 'openFile',
    # Helper classes:
    'IsDescription', 'Description', 'Filters', 'IndexProps',
    # Atom types:
    'Atom', 'ObjectAtom', 'VLStringAtom', 'StringAtom', 'BoolAtom',
    'IntAtom', 'Int8Atom', 'UInt8Atom', 'Int16Atom', 'UInt16Atom',
    'Int32Atom', 'UInt32Atom', 'Int64Atom', 'UInt64Atom',
    'FloatAtom', 'Float32Atom', 'Float64Atom',
    'ComplexAtom', 'Complex32Atom', 'Complex64Atom',
    'TimeAtom', 'Time32Atom', 'Time64Atom',
    # Column types:
    'Col', 'BoolCol', 'StringCol',
    'IntCol', 'Int8Col', 'UInt8Col', 'Int16Col', 'UInt16Col',
    'Int32Col', 'UInt32Col', 'Int64Col', 'UInt64Col',
    'FloatCol', 'Float32Col', 'Float64Col',
    'ComplexCol', 'Complex32Col', 'Complex64Col',
    'TimeCol', 'Time32Col', 'Time64Col',
    # Node classes:
    'Node', 'Group', 'Leaf', 'Table', 'VLTable', 'Array', 'EArray', 'VLArray',
    'UnImplemented',
    ]
