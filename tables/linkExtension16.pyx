################### #####################################################
#
#       License: BSD
#       Created: December 11, 2009
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id: linkExtension.pyx 4270 2009-12-02 19:42:05Z faltet $
#
########################################################################

"""Pyrex functions and classes for supporting links in HDF5.
"""

from tables.exceptions import HDF5ExtError

from hdf5Extension cimport Node

from definitions cimport \
     size_t, hid_t, herr_t, hbool_t, time_t, H5G_obj_t, \
     malloc, free, \
     H5G_UNKNOWN, H5G_GROUP, H5G_DATASET, H5G_TYPE, H5G_LINK


__version__ = "$Revision: 4270 $"


#----------------------------------------------------------------------

# External declarations

cdef extern from "H5Gpublic.h":

  cdef enum H5G_link_t:
    H5G_LINK_ERROR      = -1,
    H5G_LINK_HARD       = 0,
    H5G_LINK_SOFT       = 1

  ctypedef struct H5G_stat_t:
    unsigned long fileno[2]
    unsigned long objno[2]
    unsigned nlink
    H5G_obj_t type
    time_t mtime
    size_t linklen
    #H5O_stat_t ohdr            # Object header information

  herr_t H5Glink (hid_t file_id, H5G_link_t link_type,
                  char *current_name, char *new_name)

  herr_t H5Glink2(hid_t curr_loc_id, char *current_name, H5G_link_t link_type,
                  hid_t new_loc_id, char *new_name)

  herr_t H5Gunlink(hid_t file_id, char *name)

  herr_t H5Gget_objinfo(hid_t loc_id, char *name, hbool_t follow_link,
                        H5G_stat_t *statbuf)

  herr_t H5Gget_linkval(hid_t loc_id, char *name, size_t size, char *value)



#----------------------------------------------------------------------

# Helper functions

def _g_createHardLink(parentNode, name, targetNode):
  """Create a hard link in the file."""
  cdef herr_t ret

  ret = H5Glink2(targetNode._v_parent._v_objectID, targetNode._v_name,
                 H5G_LINK_HARD, parentNode._v_objectID, name)
  if ret < 0:
    raise HDF5ExtError("failed to create HDF5 hard link")


#----------------------------------------------------------------------

# Public classes

cdef class Link(Node):
  """Extension class from which all link extensions inherits."""

  def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
    """Private part for the _f_copy() method."""
    cdef object stats

    # Update statistics if needed.
    stats = kwargs.get('stats', None)
    if stats is not None:
      stats['links'] += 1



cdef class SoftLink(Link):
  """Extension class representing a soft link."""

  def _g_create(self):
    """Create the link in file."""
    cdef herr_t ret

    ret = H5Glink(self.parent_id, H5G_LINK_SOFT, self.target, self.name)
    if ret < 0:
      raise HDF5ExtError("failed to create HDF5 soft link")

    return 0  # Object ID is zero'ed, as HDF5 does not assign one for links


  def _g_open(self):
    """Open the link in file."""
    cdef herr_t ret
    cdef H5G_stat_t link_buff
    cdef size_t val_size
    cdef char *linkval    # most be enough for most uses

    ret = H5Gget_objinfo(self.parent_id, self.name, 0, &link_buff)
    if ret < 0:
      raise HDF5ExtError("failed to get info about soft link")

    val_size = link_buff.linklen
    linkval = <char *>malloc(val_size)

    ret = H5Gget_linkval(self.parent_id, self.name, val_size, linkval)
    if ret < 0:
      raise HDF5ExtError("failed to get target value")

    self.target = linkval

    # Release resources
    free(linkval)
    return 0  # Object ID is zero'ed, as HDF5 does not assign one for links


  def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
    """Private part for the _f_copy() method."""
    cdef herr_t ret

    # The only link that will be using this code is the soft link
    ret = H5Glink(newParent._v_objectID, H5G_LINK_SOFT, self.target, newName)
    if ret < 0:
      raise HDF5ExtError("failed to copy HDF5 link")

    # Update statistics
    super(SoftLink, self)._g_copy(newParent, newName, recursive,
                                  _log=True, **kwargs)

    return newParent._v_file.getNode(newParent, newName)




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
