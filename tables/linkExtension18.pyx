########################################################################
#
#       License: BSD
#       Created: November 25, 2009
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Pyrex functions and classes for supporting links in HDF5.
"""

from tables.exceptions import HDF5ExtError

from hdf5Extension cimport Node

from definitions cimport \
     H5P_DEFAULT, \
     size_t, hid_t, herr_t, hbool_t, int64_t, H5T_cset_t, haddr_t, \
     malloc, free


__version__ = "$Revision$"


#----------------------------------------------------------------------

# External declarations

cdef extern from "H5Lpublic.h":

  ctypedef enum H5L_type_t:
    H5L_TYPE_ERROR = (-1),       # Invalid link type id
    H5L_TYPE_HARD = 0,           # Hard link id
    H5L_TYPE_SOFT = 1,           # Soft link id
    H5L_TYPE_EXTERNAL = 64,      # External link id
    H5L_TYPE_MAX = 255           # Maximum link type id

  # Information struct for link (for H5Lget_info)
  cdef union _add_u:
    haddr_t address              # Address hard link points to
    size_t val_size              # Size of a soft link or UD link value

  ctypedef struct H5L_info_t:
    H5L_type_t     type          # Type of link
    hbool_t        corder_valid  # Indicate if creation order is valid
    int64_t        corder        # Creation order
    H5T_cset_t     cset          # Character set of link name
    _add_u         u             # Size of a soft link or UD link value

  # Operations with links
  herr_t H5Lcreate_hard(
    hid_t obj_loc_id, char *obj_name, hid_t link_loc_id, char *link_name,
    hid_t lcpl_id, hid_t lapl_id)

  herr_t H5Lcreate_soft(
    char *target_path, hid_t link_loc_id, char *link_name,
    hid_t lcpl_id, hid_t lapl_id)

  herr_t H5Lcreate_external(
    char *file_name, char *object_name, hid_t link_loc_id, char *link_name,
    hid_t lcpl_id, hid_t lapl_id)

  herr_t H5Lget_info(
    hid_t link_loc_id, char *link_name, H5L_info_t *link_buff,
    hid_t lapl_id)

  herr_t H5Lget_val(
    hid_t link_loc_id, char *link_name, void *linkval_buff, size_t size,
    hid_t lapl_id)

  herr_t H5Lunpack_elink_val(
    char *ext_linkval, size_t link_size, unsigned *flags,
    char **filename, char **obj_path)

  herr_t H5Lcopy(
    hid_t src_loc_id, char *src_name, hid_t dest_loc_id, char *dest_name,
    hid_t lcpl_id, hid_t lapl_id)



#----------------------------------------------------------------------

# Helper functions

def _getLinkClass(parent_id, name):
    """Guess the link class."""
    cdef herr_t ret
    cdef H5L_info_t link_buff
    cdef H5L_type_t link_type

    ret = H5Lget_info(parent_id, name, &link_buff, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to get info about link")

    link_type = link_buff.type
    if link_type == H5L_TYPE_SOFT:
      return "SoftLink"
    elif link_type == H5L_TYPE_EXTERNAL:
      return "ExternalLink"
    return "UnImplemented"


def _g_createHardLink(parentNode, name, targetNode):
  """Create a hard link in the file."""
  cdef herr_t ret

  ret = H5Lcreate_hard(targetNode._v_parent._v_objectID, targetNode._v_name,
                       parentNode._v_objectID, name,
                       H5P_DEFAULT, H5P_DEFAULT)
  if ret < 0:
    raise HDF5ExtError("failed to create HDF5 hard link")


#----------------------------------------------------------------------

# Public classes

cdef class Link(Node):
  """Extension class from which all link extensions inherits."""

  def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
    """Private part for the _f_copy() method."""
    cdef herr_t ret
    cdef object stats

    ret = H5Lcopy(self.parent_id, self.name, newParent._v_objectID, newName,
                  H5P_DEFAULT, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to copy HDF5 link")

    # Update statistics if needed.
    stats = kwargs.get('stats', None)
    if stats is not None:
      stats['links'] += 1

    return newParent._v_file.getNode(newParent, newName)



cdef class SoftLink(Link):
  """Extension class representing a soft link."""

  def _g_create(self):
    """Create the link in file."""
    cdef herr_t ret

    ret = H5Lcreate_soft(self.target, self.parent_id, self.name,
                         H5P_DEFAULT, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to create HDF5 soft link")

    return 0  # Object ID is zero'ed, as HDF5 does not assign one for links


  def _g_open(self):
    """Open the link in file."""
    cdef herr_t ret
    cdef H5L_info_t link_buff
    cdef size_t val_size
    cdef char *linkval

    ret = H5Lget_info(self.parent_id, self.name, &link_buff, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to get info about soft link")

    val_size = link_buff.u.val_size
    linkval = <char *>malloc(val_size)

    ret = H5Lget_val(self.parent_id, self.name, linkval, val_size, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to get target value")

    self.target = linkval

    # Release resources
    free(linkval)
    return 0  # Object ID is zero'ed, as HDF5 does not assign one for links



cdef class ExternalLink(Link):
  """Extension class representing an external link."""

  def _g_create(self):
    """Create the link in file."""
    cdef herr_t ret

    filename, target = self._get_filename_node()
    ret = H5Lcreate_external(filename, target, self.parent_id, self.name,
                             H5P_DEFAULT, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to create HDF5 external link")

    return 0  # Object ID is zero'ed, as HDF5 does not assign one for links


  def _g_open(self):
    """Open the link in file."""
    cdef herr_t ret
    cdef H5L_info_t link_buff
    cdef size_t val_size
    cdef char *linkval, *filename, *obj_path
    cdef unsigned flags

    ret = H5Lget_info(self.parent_id, self.name, &link_buff, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to get info about external link")

    val_size = link_buff.u.val_size
    linkval = <char *>malloc(val_size)

    ret = H5Lget_val(self.parent_id, self.name, linkval, val_size, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("failed to get target value")

    ret = H5Lunpack_elink_val(linkval, val_size, &flags, &filename, &obj_path)
    if ret < 0:
      raise HDF5ExtError("failed to unpack external link value")

    self.target = filename+':'+obj_path

    # Release resources
    free(linkval)
    return 0  # Object ID is zero'ed, as HDF5 does not assign one for links




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
