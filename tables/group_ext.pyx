from .exceptions import HDF5ExtError

from .definitions cimport H5L_TYPE_ERROR, H5L_TYPE_SOFT, H5L_TYPE_EXTERNAL, H5L_TYPE_HARD, H5Lmove
from .definitions cimport H5Dopen, H5Dclose
from .definitions cimport H5Fflush, H5F_SCOPE_GLOBAL
from .definitions cimport H5O_TYPE_UNKNOWN, H5O_TYPE_GROUP, H5O_TYPE_DATASET, H5O_TYPE_NAMED_DATATYPE
from .definitions cimport H5P_DEFAULT
from .definitions cimport H5Gcreate, H5Gopen, H5Gclose, Giterate
from .definitions cimport get_linkinfo, get_objinfo

from .hdf5extension cimport get_attribute_string_or_none

cdef class Group(Node):
  def _g_create(self):
    cdef hid_t ret
    cdef bytes encoded_name

    encoded_name = self.name.encode('utf-8')

    # @TODO: set property list --> utf-8

    # Create a new group
    ret = H5Gcreate(self.parent_id, encoded_name, H5P_DEFAULT, H5P_DEFAULT,
                    H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_open(self):
    cdef hid_t ret
    cdef bytes encoded_name

    encoded_name = self.name.encode('utf-8')

    ret = H5Gopen(self.parent_id, encoded_name, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("Can't open the group: '%s'." % self.name)
    self.group_id = ret
    return self.group_id

  def _g_get_objinfo(self, object h5name):
    """Check whether 'name' is a children of 'self' and return its type."""

    cdef int ret
    cdef object node_type
    cdef bytes encoded_name
    cdef char *cname

    encoded_name = h5name.encode('utf-8')
    # Get the C pointer
    cname = encoded_name

    ret = get_linkinfo(self.group_id, cname)
    if ret == -2 or ret == H5L_TYPE_ERROR:
      node_type = "NoSuchNode"
    elif ret == H5L_TYPE_SOFT:
      node_type = "SoftLink"
    elif ret == H5L_TYPE_EXTERNAL:
      node_type = "ExternalLink"
    elif ret == H5L_TYPE_HARD:
        ret = get_objinfo(self.group_id, cname)
        if ret == -2:
          node_type = "NoSuchNode"
        elif ret == H5O_TYPE_UNKNOWN:
          node_type = "Unknown"
        elif ret == H5O_TYPE_GROUP:
          node_type = "Group"
        elif ret == H5O_TYPE_DATASET:
          node_type = "Leaf"
        elif ret == H5O_TYPE_NAMED_DATATYPE:
          node_type = "NamedType"              # Not supported yet
        #else H5O_TYPE_LINK:
        #    # symbolic link
        #    raise RuntimeError('unexpected object type')
        else:
          node_type = "Unknown"
    return node_type

  def _g_list_group(self, parent):
    """Return a tuple with the groups and the leaves hanging from self."""

    cdef bytes encoded_name

    encoded_name = self.name.encode('utf-8')

    return Giterate(parent._v_objectid, self._v_objectid, encoded_name)


  def _g_get_gchild_attr(self, group_name, attr_name):
    """Return an attribute of a child `Group`.

    If the attribute does not exist, ``None`` is returned.

    """

    cdef hid_t gchild_id
    cdef object retvalue
    cdef bytes encoded_group_name
    cdef bytes encoded_attr_name

    encoded_group_name = group_name.encode('utf-8')
    encoded_attr_name = attr_name.encode('utf-8')

    # Open the group
    retvalue = None  # Default value
    gchild_id = H5Gopen(self.group_id, encoded_group_name, H5P_DEFAULT)
    if gchild_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" %
                         (group_name, self._v_pathname))
    retvalue = get_attribute_string_or_none(gchild_id, encoded_attr_name)
    # Close child group
    H5Gclose(gchild_id)

    return retvalue


  def _g_get_lchild_attr(self, leaf_name, attr_name):
    """Return an attribute of a child `Leaf`.

    If the attribute does not exist, ``None`` is returned.

    """

    cdef hid_t leaf_id
    cdef object retvalue
    cdef bytes encoded_leaf_name
    cdef bytes encoded_attr_name

    encoded_leaf_name = leaf_name.encode('utf-8')
    encoded_attr_name = attr_name.encode('utf-8')

    # Open the dataset
    leaf_id = H5Dopen(self.group_id, encoded_leaf_name, H5P_DEFAULT)
    if leaf_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" %
                         (leaf_name, self._v_pathname))
    retvalue = get_attribute_string_or_none(leaf_id, encoded_attr_name)
    # Close the dataset
    H5Dclose(leaf_id)
    return retvalue


  def _g_flush_group(self):
    # Close the group
    H5Fflush(self.group_id, H5F_SCOPE_GLOBAL)


  def _g_close_group(self):
    cdef int ret

    ret = H5Gclose(self.group_id)
    if ret < 0:
      raise HDF5ExtError("Problems closing the Group %s" % self.name)
    self.group_id = 0  # indicate that this group is closed


  def _g_move_node(self, hid_t oldparent, oldname, hid_t newparent, newname,
                   oldpathname, newpathname):
    cdef int ret
    cdef bytes encoded_oldname, encoded_newname

    encoded_oldname = oldname.encode('utf-8')
    encoded_newname = newname.encode('utf-8')

    ret = H5Lmove(oldparent, encoded_oldname, newparent, encoded_newname,
                  H5P_DEFAULT, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("Problems moving the node %s to %s" %
                         (oldpathname, newpathname) )
    return ret

