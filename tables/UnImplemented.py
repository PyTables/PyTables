########################################################################
#
#       License: BSD
#       Created: January 14, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/UnImplemented.py,v $
#       $Id$
#
########################################################################

"""Here is defined the UnImplemented class.

See UnImplemented class docstring for more info.

Classes:

    UnImplemented

Misc variables:

    __version__


"""

import warnings

import tables.hdf5Extension as hdf5Extension
from tables.Leaf import Leaf



__version__ = "$Revision: 1.7 $"



class UnImplemented(hdf5Extension.UnImplemented, Leaf):
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

        # All this will eventually end up in the node constructor.

        (self.shape, self.byteorder) = self._openUnImplemented()
        self.nrows = self.shape[0]


    def _g_copy(self, newParent, newName, recursive, **kwargs):
        """_g_copy(newParent, newName, recursive[, arg=value...]) -> None.  Does nothing.

        This method does nothing, but a UserWarning is issued.  Please note
        that this method does not return a new node, but None.
        """

        warnings.warn(
            "UnImplemented node %r does not know how to copy itself; skipping"
            % (self._v_pathname,))
        return None  # Can you see it?


    def _f_copy(self, newparent = None, newname = None,
                overwrite = False, recursive = False, **kwargs):
        """_f_copy(newparent, newname[, overwrite][, recursive][, arg=value...]) -> None.  Does nothing.

        This method does nothing, since UnImplemented nodes can not be copied.
        However, a UserWarning is issued.  Please note that this method does
        not return a new node, but None.
        """

        # This also does nothing but warn.
        self._g_copy(newparent, newname, recursive, **kwargs)
        return None  # Can you see it?


    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""
        # byteorder = %r
        return """%s
  NOTE: <The UnImplemented object represents a PyTables unimplemented
         dataset present in the '%s' HDF5 file.
         If you want to see this kind of HDF5 dataset implemented in
         PyTables, please, contact the developers.>
""" % (str(self), self._v_file.filename)
