########################################################################
#
#       Copyright:      LGPL
#       Created:        October 1, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/__init__.py,v $
#       $Id: __init__.py,v 1.2 2002/11/07 17:52:35 falted Exp $
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


"""

__version__ = "$Revision: 1.2 $"

# Import the user classes from the proper modules
from File import File, openFile
from Group import Group
from Leaf import Leaf
from Table import Table
from Array import Array
from IsRecord import metaIsRecord, IsRecord
from hdf5Extension import isHDF5, isPyTablesFile

# List here only the objects we want to be publicly available
__all__ = [ "openFile", "isPyTablesFile", "IsRecord", "isHDF5" ]
