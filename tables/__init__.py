########################################################################
#
#       License: BSD
#       Created: October 1, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/__init__.py,v $
#       $Id: __init__.py,v 1.7 2003/03/10 11:24:02 falted Exp $
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
from IsColDescr import IsColDescr, Col
from hdf5Extension import isHDF5, isPyTablesFile

#import recarray2

# List here only the objects we want to be publicly available
__all__ = [ "openFile", "isHDF5", "isPyTablesFile", "IsColDescr", "Col" ]
