########################################################################
#
#       License: BSD
#       Created: October 1, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/__init__.py,v $
#       $Id: __init__.py,v 1.19 2003/12/27 22:54:34 falted Exp $
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
from hdf5Extension import getHDF5Version, \
                          getPyTablesVersion, \
                          getExtVersion

__version__ = getPyTablesVersion()
HDF5Version = getHDF5Version()
ExtVersion  = getExtVersion()

# Import the user classes from the proper modules
from File import File, openFile
from Group import Group
from Leaf import Leaf
from Table import Table
from Array import Array
from EArray import EArray
from VLArray import *
                    
from IsDescription import *
from hdf5Extension import isHDF5, isPyTablesFile, whichLibVersion

# List here only the objects we want to be publicly available
__all__ = ["isHDF5", "isPyTablesFile", "whichLibVersion",
           "openFile", "IsDescription", "Description",
           "Col", "BoolCol", "StringCol",
           "IntCol", "Int8Col", "UInt8Col", "Int16Col", "UInt16Col",
           "Int32Col", "UInt32Col", "Int64Col", "UInt64Col",
           "FloatCol", "Float32Col", "Float64Col",
           "Atom", "ObjectAtom", "VLStringAtom", "StringAtom", "BoolAtom",
           "IntAtom", "Int8Atom", "UInt8Atom", "Int16Atom", "UInt16Atom",
           "Int32Atom", "UInt32Atom", "Int64Atom", "UInt64Atom",
           "FloatAtom", "Float32Atom", "Float64Atom"]
