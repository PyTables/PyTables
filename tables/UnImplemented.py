########################################################################
#
#       License: BSD
#       Created: January 14, 2004
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/UnImplemented.py,v $
#       $Id: UnImplemented.py,v 1.1 2004/01/14 10:39:14 falted Exp $
#
########################################################################

"""Here is defined the UnImplemented class.

See UnImplemented class docstring for more info.

Classes:

    UnImplemented

Misc variables:

    __version__


"""

__version__ = "$Revision: 1.1 $"

from Leaf import Leaf
import hdf5Extension

class UnImplemented(Leaf, hdf5Extension.UnImplemented, object):
    """Represent an unimplemented dataset in HDF5 file.

    If you want to see this kind of HDF5 dataset implemented in PyTables,
    please, contact the developers.

    """
    
    def __init__(self):
        """Create the UnImplemented instance."""
        # UnImplemented objects exist always (we don't create them)
        self._v_new = 0

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.shape, self.byteorder) = self._openUnImplemented()
        # Get info about existing filters
        self._g_getFilters()

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""
        # byteorder = %r
        return """%s
  NOTE: <The UnImplemented object represents a PyTables unimplemented
         dataset present in the '%s' HDF5 file.
         If you wanna see this kind of HDF5 dataset implemented in
         PyTables, please, contact the developers.>
""" % (str(self), self._v_file.filename)
