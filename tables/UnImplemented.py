########################################################################
#
#       License: BSD
#       Created: January 14, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
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

from tables import hdf5Extension
from tables.Leaf import Leaf


__version__ = "$Revision$"



class UnImplemented(hdf5Extension.UnImplemented, Leaf):
    """Represent an unimplemented dataset in HDF5 file.

    If you want to see this kind of HDF5 dataset implemented in PyTables,
    please, contact the developers.

    """

    def __init__(self, parentNode, name):
        """Create the UnImplemented instance."""

        # UnImplemented objects always come from opening an existing node
        # (they can not be created).
        self._v_new = False
        """Is this the first time the node has been created?"""
        self.nrows = 0
        """The length of the first dimension of the data."""
        self.shape = (0,)
        """The shape of the stored data."""
        self.byteorder = None
        """
        The endianness of data in memory ('big', 'little' or
        'irrelevant').
        """

        super(UnImplemented, self).__init__(parentNode, name)


    def _g_open(self):
        """Get the metadata info for an array in file."""

        (self.shape, self.byteorder, objectID) = \
                     self._openUnImplemented()
        self.nrows = self.shape[0]
        return objectID


    def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
        """
        Do nothing.

        This method does nothing, but a UserWarning is issued.  Please note
        that this method does not return a new node, but None.
        """

        warnings.warn(
            "UnImplemented node %r does not know how to copy itself; skipping"
            % (self._v_pathname,))
        return None  # Can you see it?


    def _f_copy(self, newparent=None, newname=None,
                overwrite=False, recursive=False, createparents=False,
                **kwargs):
        """_f_copy(newparent, newname[, overwrite][, recursive][, createparents][, arg=value...]) -> None.  Does nothing.

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


# Non supported classes. These are listed here for backward compatibility
# with PyTables 0.9.x indexes

class OldIndexArray(UnImplemented):
    _c_classId = 'IndexArray'
