#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id$
#
########################################################################

"""Pyrex interface between several PyTables classes and HDF5 library.

Classes (type extensions):

    File
    AttributeSet
    Node
    Leaf
    Group
    Array
    VLArray
    UnImplemented

Functions:

Misc variables:

    __version__
"""

import sys
import os
import warnings
import cPickle

import numpy

from tables.misc.enum import Enum
from tables.exceptions import HDF5ExtError
from tables.utils import checkFileAccess, byteorders

from tables.utilsExtension import  \
     enumToHDF5, enumFromHDF5, getTypeEnum, \
     convertTime64, isHDF5File, isPyTablesFile

from lrucacheExtension cimport NodeCache

# Types, constants, functions, classes & other objects from everywhere
from definitions cimport  \
     strdup, malloc, free, \
     Py_ssize_t, PyObject_AsReadBuffer, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, PyString_AsString, \
     PyString_FromStringAndSize, PyDict_Contains, PyDict_GetItem, \
     Py_INCREF, Py_DECREF, \
     import_array, ndarray, dtype, \
     time_t, size_t, hid_t, herr_t, hsize_t, hvl_t, \
     H5T_class_t, \
     H5F_SCOPE_GLOBAL, H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, \
     H5P_DEFAULT, H5T_SGN_NONE, H5T_SGN_2, H5S_SELECT_SET, \
     H5get_libversion, H5check_version, H5Fcreate, H5Fopen, H5Fclose, \
     H5Fflush, H5Gcreate, H5Gopen, H5Gclose, H5Glink, H5Gunlink, H5Gmove, \
     H5Gmove2,  H5Dopen, H5Dclose, H5Dread, H5Dget_type, H5Dget_space, \
     H5Dvlen_reclaim, H5Adelete, H5Aget_num_attrs, H5Aget_name, H5Aopen_idx, \
     H5Aread, H5Aclose, H5Tclose, H5Pcreate, H5Pclose, \
     H5Pset_cache, H5Pset_sieve_buf_size, H5Pset_fapl_log, \
     H5Sselect_hyperslab, H5Screate_simple, H5Sget_simple_extent_ndims, \
     H5Sget_simple_extent_dims, H5Sclose, \
     H5ATTRset_attribute, H5ATTRset_attribute_string, \
     H5ATTRget_attribute, H5ATTRget_attribute_string, \
     H5ATTRfind_attribute, H5ATTRget_attribute_ndims, \
     H5ATTRget_attribute_info, \
     set_cache_size, Giterate, Aiterate, H5UIget_info, get_len_of_range, \
     convArrayType, getArrayType, get_order, set_order



# Include conversion tables
include "convtypetables.pxi"


__version__ = "$Revision$"


#-------------------------------------------------------------------

# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h":

  herr_t H5ARRAYmake(hid_t loc_id, char *dset_name, char *class_,
                     char *title, char *obversion,
                     int rank, hsize_t *dims, int extdim,
                     hid_t type_id, hsize_t *dims_chunk, void *fill_data,
                     int complevel, char  *complib, int shuffle,
                     int fletcher32, void *data)

  herr_t H5ARRAYappend_records(hid_t dataset_id, hid_t type_id,
                               int rank, hsize_t *dims_orig,
                               hsize_t *dims_new, int extdim, void *data )

  herr_t H5ARRAYwrite_records(hid_t dataset_id, hid_t type_id,
                              int rank, hsize_t *start, hsize_t *step,
                              hsize_t *count, void *data)

  herr_t H5ARRAYtruncate(hid_t dataset_id, int extdim, hsize_t size)

  herr_t H5ARRAYread(hid_t dataset_id, hid_t type_id,
                     hsize_t start, hsize_t nrows, hsize_t step,
                     int extdim, void *data)

  herr_t H5ARRAYreadSlice(hid_t dataset_id, hid_t type_id,
                          hsize_t *start, hsize_t *stop,
                          hsize_t *step, void *data)

  herr_t H5ARRAYreadIndex(hid_t dataset_id, hid_t type_id, int notequal,
                          hsize_t *start, hsize_t *stop, hsize_t *step,
                          void *data)

  herr_t H5ARRAYget_ndims(hid_t dataset_id, hid_t type_id, int *rank)

  herr_t H5ARRAYget_info(hid_t dataset_id, hid_t type_id, hsize_t *dims,
                         hsize_t *maxdims, hid_t *super_type_id,
                         H5T_class_t *super_class_id, char *byteorder)

  herr_t H5ARRAYget_chunkshape(hid_t dataset_id, int rank, hsize_t *dims_chunk)


# Functions for dealing with VLArray objects
cdef extern from "H5VLARRAY.h":

  herr_t H5VLARRAYmake( hid_t loc_id, char *dset_name, char *class_,
                        char *title, char *obversion,
                        int rank, int scalar, hsize_t *dims, hid_t type_id,
                        hsize_t chunk_size, void *fill_data, int complevel,
                        char *complib, int shuffle, int flecther32,
                        void *data)

  herr_t H5VLARRAYappend_records( hid_t dataset_id, hid_t type_id,
                                  int nobjects, hsize_t nrecords,
                                  void *data )

  herr_t H5VLARRAYmodify_records( hid_t dataset_id, hid_t type_id,
                                  hsize_t nrow, int nobjects,
                                  void *data )

  herr_t H5VLARRAYget_ndims( hid_t dataset_id, hid_t type_id, int *rank )

  herr_t H5VLARRAYget_info( hid_t dataset_id, hid_t type_id,
                            hsize_t *nrecords, hsize_t *base_dims,
                            hid_t *base_type_id, char *base_byteorder)



#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#---------------------------------------------------------------------------

# Helper functions

cdef object getshape(int rank, hsize_t *dims):
  "Return a shape (tuple) from a dims C array of rank dimensions."
  cdef int i
  cdef object shape

  shape = []
  for i from 0 <= i < rank:
    shape.append(dims[i])

  return tuple(shape)


cdef object splitPath(object path):
  """splitPath(path) -> (parentPath, name).  Splits a canonical path.

  Splits the given canonical path into a parent path (without the trailing
  slash) and a node name.
  """

  lastSlash = path.rfind('/')
  ppath = path[:lastSlash]
  name = path[lastSlash+1:]

  if ppath == '':
      ppath = '/'

  return (ppath, name)


# Helper function for quickly fetch an attribute string
def get_attribute_string_or_none(node_id, attr_name):
  """
  Returns a string attribute if it exists in node_id.

  It returns ``None`` in case it don't exists (or there have been problems
  reading it).
  """

  cdef char *attr_value
  cdef object retvalue

  attr_value = NULL
  retvalue = None   # Default value
  if H5ATTRfind_attribute(node_id, attr_name):
    ret = H5ATTRget_attribute_string(node_id, attr_name, &attr_value)
    retvalue = numpy.string_(attr_value)
    # Important to release attr_value, because it has been malloc'ed!
    if attr_value: free(<void *>attr_value)
    if ret < 0: return None

  return retvalue



# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef hid_t   access_plist
  cdef char    *name


  def __new__(self, object name, char *mode, char *title,
              object trTable, char *root, object filters,
              size_t metadataCacheSize, size_t nodeCacheSize):
    # Create a new file using default properties
    self.name = name
    self.mode = pymode = mode

    # These fields can be seen from Python.
    self._v_new = None  # this will be computed later
    """Is this file going to be created from scratch?"""
    self._isPTFile = True  # assume a PyTables file by default
    """Does this HDF5 file have a PyTables format?"""

    # After the following check we can be quite sure
    # that the file or directory exists and permissions are right.
    checkFileAccess(name, mode)

    assert pymode in ('r', 'r+', 'a', 'w'), \
           "an invalid mode string ``%s`` " \
           "passed the ``checkFileAccess()`` test; " \
           "please report this to the authors" % pymode

    # Should a new file be created?
    exists = os.path.exists(name)
    self._v_new = new = not (
      pymode in ('r', 'r+') or (pymode == 'a' and exists))

    # After the following check we know that the file exists
    # and it is an HDF5 file, or maybe a PyTables file.
    if not new:
      if not isPyTablesFile(name):
        if isHDF5File(name):
          # HDF5 but not PyTables.
          warnings.warn("file ``%s`` exists and it is an HDF5 file, " \
                        "but it does not have a PyTables format; " \
                        "I will try to do my best to guess what's there " \
                        "using HDF5 metadata" % name)
          self._isPTFile = False
        else:
          # The file is not even HDF5.
          raise IOError(
            "file ``%s`` exists but it is not an HDF5 file" % name)
      # Else the file is an ordinary PyTables file.

    if pymode == 'r':
      # Just a test for disabling metadata caching.
      ## access_plist = H5Pcreate(H5P_FILE_ACCESS)
      ## H5Pset_cache(access_plist, 0, 0, 0, 0.0)
      ## H5Pset_sieve_buf_size(access_plist, 0)
      ##self.file_id = H5Fopen(name, H5F_ACC_RDONLY, access_plist)
      self.file_id = H5Fopen(name, H5F_ACC_RDONLY, H5P_DEFAULT)
    elif pymode == 'r+':
      self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
    elif pymode == 'a':
      if exists:
        # A test for logging.
        ## access_plist = H5Pcreate(H5P_FILE_ACCESS)
        ## H5Pset_cache(access_plist, 0, 0, 0, 0.0)
        ## H5Pset_sieve_buf_size(access_plist, 0)
        ## H5Pset_fapl_log (access_plist, "test.log", H5FD_LOG_LOC_WRITE, 0)
        ## self.file_id = H5Fopen(name, H5F_ACC_RDWR, access_plist)
        self.file_id = H5Fopen(name, H5F_ACC_RDWR, H5P_DEFAULT)
      else:
        self.file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
    elif pymode == 'w':
      self.file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)

    # Set the cache size (only for HDF5 1.8.x)
    set_cache_size(self.file_id, metadataCacheSize)


  # Accessor definitions
  def _getFileId(self):
    return self.file_id


  # Optimised version in Pyrex of File._getNode
  # This is a try to see if I can get _getNode significantly faster than
  # the pure Python version, but I had no success because it is only marginally
  # faster (just a 5% or less). So, the best is not to use it, I think.
  # I'll let it here just in case more speed is needed (in the context of
  # benchmarks, most presumably) and optimization work goes back again.
  # I think that the problem could be that calling a Python method in Pyrex
  # is more costly than in Python itself, but this is only a guess.
  # F. Altet 2006-08-07
#   def _getNode(self, object nodePath):
#     cdef object aliveNodes, parentPath, pathTail, parentNode, node
#     cdef NodeCache deadNodes

#     # The root node is always at hand.
#     if nodePath == '/':
#       return self.root
#     else:
#       # Check quickly is nodePath is alive or dead (i.e. in memory)
#       aliveNodes = self._aliveNodes
#       #if nodePath in aliveNodes:
#       # We don't check for -1 as this should never fail
#       if PyDict_Contains(aliveNodes, nodePath):
#         # The parent node is in memory and alive, so get it.
#         node = aliveNodes[nodePath]
#         assert node is not None, \
#                "stale weak reference to dead node ``%s``" % parentPath
#         return node
#       deadNodes = <NodeCache>self._deadNodes
#       if deadNodes.contains(nodePath):
#         # The parent node is in memory but dead, so revive it.
#         node = self._g_reviveNode(nodePath)
#         # Call the post-revive hook
#         node._g_postReviveHook()
#         return node

#     # Walk up the hierarchy until a node in the path is in memory.
#     (parentPath, nodeName) = splitPath(nodePath)
#     pathTail = [nodeName]  # subsequent children below that node
#     while parentPath != '/':
#       if parentPath in aliveNodes:
#         # The parent node is in memory and alive, so get it.
#         parentNode = aliveNodes[parentPath]
#         assert parentNode is not None, \
#                "stale weak reference to dead node ``%s``" % parentPath
#         break
#       if deadNodes.contains(parentPath):
#         # The parent node is in memory but dead, so revive it.
#         parentNode = self._g_reviveNode(parentPath)
#         # Call the post-revive hook
#         parentNode._g_postReviveHook()
#         break
#       # Go up one level to try again.
#       (parentPath, nodeName) = splitPath(parentPath)
#       pathTail.insert(0, nodeName)
#     else:
#       # We hit the root node and no parent was in memory.
#       parentNode = self.root

#     # Walk down the hierarchy until the last child in the tail is loaded.
#     node = parentNode  # maybe `nodePath` was already in memory
#     for childName in pathTail:
#       # Load the node and use it as a parent for the next one in tail
#       # (it puts itself into life via `self._refNode()` when created).
#       if not isinstance(parentNode, Group):
#         # This is the root group
#         parentPath = parentNode._v_pathname
#         raise TypeError("node ``%s`` is not a group; "
#                         "it can not have a child named ``%s``"
#                         % (parentPath, childName))
#       node = parentNode._g_loadChild(childName)
#       parentNode = node

#     return node


#   cdef object _g_reviveNode(self, object nodePath):
#     """
#     Revive the node under `nodePath` and return it.

#     Moves the node under `nodePath` from the set of dead,
#     unreferenced nodes to the set of alive, referenced ones.
#     """
#     cdef object aliveNodes, node
#     cdef NodeCache deadNodes

#     assert nodePath in self._deadNodes, \
#            "trying to revive non-dead node ``%s``" % nodePath

#     # Take the node out of the limbo.
#     deadNodes = <NodeCache>self._deadNodes
#     node = deadNodes.cpop(nodePath)
#     # Make references to the node.
#     if nodePath != '/':
#       # The root group does not participate in alive/dead stuff.
#       aliveNodes = self._aliveNodes
#       assert nodePath not in aliveNodes, \
#              "file already has a node with path ``%s``" % nodePath
#       # Add the node to the set of referenced ones.
#       aliveNodes[nodePath] = node

#     return node


  def _flushFile(self, scope):
    # Close the file
    H5Fflush(self.file_id, scope)


  def _closeFile(self):
    # Close the file
    H5Fclose( self.file_id )
    self.file_id = 0    # Means file closed


  # This method is moved out of scope, until we provide code to delete
  # the memory booked by this extension types
  def __dealloc__(self):
    cdef int ret
    if self.file_id:
      # Close the HDF5 file because user didn't do that!
      ret = H5Fclose(self.file_id)
      if ret < 0:
        raise HDF5ExtError("Problems closing the file '%s'" % self.name)



cdef class AttributeSet:
  cdef hid_t   dataset_id
  cdef char    *name


  def _g_new(self, node):
    # Initialize the C attributes of Node object
    self.name =  PyString_AsString(node._v_hdf5name)
    # The dataset id of the node
    self.dataset_id = node._v_objectID


  def __g_listAttr(self):
    "Return a tuple with the attribute list"
    a = Aiterate(self.dataset_id)
    return a


  # The next is a re-implementation of Aiterate but in pure Pyrex
  def _g_listAttr(self):
    "Return a tuple with the attribute list"
    cdef int nattrs, i
    cdef hid_t attr_id
    cdef char attr_name[256]

    lattrs = []
    nattrs = H5Aget_num_attrs(self.dataset_id)
    for i from 0 <= i < nattrs:
      attr_id = H5Aopen_idx(self.dataset_id, i)
      H5Aget_name(attr_id, 256, attr_name)
      H5Aclose(attr_id)
      lattrs.append(attr_name)
    return lattrs


  # Get a system attribute (they should be only strings)
  def _g_getSysAttr(self, char *attrname):
    return get_attribute_string_or_none(self.dataset_id, attrname)


  # Set a system attribute (they should be only strings)
  def _g_setAttrStr(self, char *attrname, char *attrvalue):
    cdef int ret

    ret = H5ATTRset_attribute_string(self.dataset_id, attrname, attrvalue)
    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                         (attrname, self._v_node))


  def _g_setAttr(self, char *name, object value):
    """Save Python or NumPy objects as HDF5 attributes.

    Scalar Python objects, scalar NumPy & 0-dim NumPy objects will all be
    saved as H5T_SCALAR type.  N-dim NumPy objects will be saved as H5T_ARRAY
    type.
    """

    cdef int ret, i, rank
    cdef hid_t type_id
    cdef hsize_t *dims
    cdef ndarray ndv
    cdef dtype ndt
    cdef object byteorder

    # Convert a Python or NumPy scalar into a NumPy 0-dim ndarray
    if (type(value) in (bool, str, int, float, complex) or
        isinstance(value, numpy.generic)):
      value = numpy.array(value)

    # Check if value is a NumPy ndarray and of a supported type
    if (isinstance(value, numpy.ndarray) and
        value.dtype.kind in ('S', 'b', 'i', 'u', 'f', 'c')):
      # Get the associated native HDF5 type
      byteorder = byteorders[value.dtype.byteorder]
      ndt = <dtype>value.dtype
      type_id = convArrayType(ndt.type_num, ndt.elsize, byteorder)
      # Get dimensionality info
      ndv = <ndarray>value
      rank = ndv.nd
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      for i from 0 <= i < rank:
        dims[i] = ndv.dimensions[i]

      # Actually write the attribute
      ret = H5ATTRset_attribute(self.dataset_id, name, type_id,
                                rank, dims, ndv.data)
      if ret < 0:
        raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                           (name, self._v_node))

      # Release resources
      free(<void *>dims)
      H5Tclose(type_id)
    else:
      # Object cannot be natively represented in HDF5.
      # Convert this object to a null-terminated string
      # (binary pickles are not supported at this moment)
      value = cPickle.dumps(value, 0)
      ret = H5ATTRset_attribute_string(self.dataset_id, name, value)

    return


  # Get attributes
  def _g_getAttr(self, char *attrname):
    """Get HDF5 attributes and retrieve them as NumPy objects.

    H5T_SCALAR types will be retrieved as scalar NumPy.
    H5T_ARRAY types will be retrieved as ndarray NumPy objects.
    """

    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef hid_t mem_type, dset_id, type_id, type_id2
    cdef int rank, ret, enumtype
    cdef void *rbuf
    cdef ndarray ndvalue
    cdef char  byteorder[11]
    cdef object retvalue, shape

    dset_id = self.dataset_id
    dims = NULL

    ret = H5ATTRget_attribute_ndims(dset_id, attrname, &rank )
    if ret < 0:
      raise HDF5ExtError("Can't get ndims on attribute %s in node %s." %
                             (attrname, self.name))

    # Get the dimensional info
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    ret = H5ATTRget_attribute_info(dset_id, attrname, dims,
                                   &class_id, &type_size, &type_id)
    if ret < 0:
      raise HDF5ExtError("Can't get info on attribute %s in node %s." %
                               (attrname, self.name))

    # Check that class_id is a supported type for attributes
    if class_id not in (H5T_STRING, H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT,
                        H5T_COMPOUND):   # complex types are COMPOUND
      warnings.warn("""\
Type of attribute '%s' in node '%s' is not supported. Sorry about that!"""
                    % (attrname, self.name))
      return None

    # Get the dimensional info
    shape = getshape(rank, dims)
    # dims is not needed anymore
    free(<void *> dims)

    # Get the attribute NumPy type & type size
    type_size = getArrayType(type_id, &enumtype)
    # type_size is of type size_t (unsigned long), so cast it to a long first
    if <long>type_size < 0:
      # This class is not supported. Instead of raising a TypeError,
      # issue a warning explaining the problem. This will allow to continue
      # browsing native HDF5 files, while informing the user about the problem.
      warnings.warn("""\
Unsupported type for attribute '%s' in node '%s'. Offending HDF5 class: %d"""
                      % (attrname, self.name, class_id))
      return None

    # Get the dtype
    dtype = NPCodeToType[enumtype]
    if class_id == H5T_STRING:
      dtype = numpy.dtype((dtype, type_size))
    # Fix the byteorder
    get_order(type_id, byteorder)
    if byteorder in ('little', 'big'):
      dtype = numpy.dtype(dtype).newbyteorder(byteorder)
    # Get the container for data
    ndvalue = numpy.empty(dtype=dtype, shape=shape)
    # Get the pointer to the buffer data area
    rbuf = ndvalue.data

    # Actually read the attribute from disk
    ret = H5ATTRget_attribute(dset_id, attrname, type_id, rbuf)
    if ret < 0:
      raise HDF5ExtError("Attribute %s exists in node %s, but can't get it."\
                         % (attrname, self.name))
    H5Tclose(type_id)

    if rank > 0:    # multidimensional case
      retvalue = ndvalue
    else:
      retvalue = ndvalue[()]   # 0-dim ndarray becomes a NumPy scalar

    return retvalue


  def _g_remove(self, attrname):
    cdef int ret
    ret = H5Adelete(self.dataset_id, attrname)
    if ret < 0:
      raise HDF5ExtError("Attribute '%s' exists in node '%s', but cannot be deleted." \
                         % (attrname, self.name))


  def __dealloc__(self):
    self.dataset_id = 0



cdef class Node:
  # Instance variables declared in .pxd


  def _g_new(self, where, name, init):
    self.name = strdup(name)
    """The name of this node in its parent group."""
    self.parent_id = where._v_objectID
    """The identifier of the parent group."""


  def _g_delete(self):
    cdef int ret

    # Delete this node
    ret = H5Gunlink(self.parent_id, self.name)
    if ret < 0:
      raise HDF5ExtError("problems deleting the node ``%s``" % self.name)
    return ret


  def __dealloc__(self):
    free(<void *>self.name)



cdef class Group(Node):
  cdef hid_t   group_id


  def _g_create(self):
    cdef hid_t ret

    # Create a new group
    ret = H5Gcreate(self.parent_id, self.name, 0)
    if ret < 0:
      raise HDF5ExtError("Can't create the group %s." % self.name)
    self.group_id = ret
    return self.group_id


  def _g_open(self):
    cdef hid_t ret

    ret = H5Gopen(self.parent_id, self.name)
    if ret < 0:
      raise HDF5ExtError("Can't open the group: '%s'." % self.name)
    self.group_id = ret
    return self.group_id


  def _g_listGroup(self):
    # Return a tuple with the objects groups and objects dsets
    return Giterate(self.parent_id, self._v_objectID, self.name)


  def _g_getGChildAttr(self, char *group_name, char *attr_name):
    """
    Return an attribute of a child `Group`.

    If the attribute does not exist, ``None`` is returned.
    """

    cdef hid_t gchild_id
    cdef object retvalue

    # Open the group
    retvalue = None  # Default value
    gchild_id = H5Gopen(self.group_id, group_name)
    retvalue = get_attribute_string_or_none(gchild_id, attr_name)
    # Close child group
    H5Gclose(gchild_id)

    return retvalue


  def _g_getLChildAttr(self, char *leaf_name, char *attr_name):
    """
    Return an attribute of a child `Leaf`.

    If the attribute does not exist, ``None`` is returned.
    """

    cdef hid_t leaf_id
    cdef object retvalue

    # Open the dataset
    leaf_id = H5Dopen(self.group_id, leaf_name)
    retvalue = get_attribute_string_or_none(leaf_id, attr_name)
    # Close the dataset
    H5Dclose(leaf_id)
    return retvalue


  def _g_flushGroup(self):
    # Close the group
    H5Fflush(self.group_id, H5F_SCOPE_GLOBAL)


  def _g_closeGroup(self):
    cdef int ret

    ret = H5Gclose(self.group_id)
    if ret < 0:
      raise HDF5ExtError("Problems closing the Group %s" % self.name )
    self.group_id = 0  # indicate that this group is closed


  def _g_moveNode(self, hid_t oldparent, char *oldname,
                  hid_t newparent, char *newname,
                  char *oldpathname, char *newpathname):
    cdef int ret

    ret = H5Gmove2(oldparent, oldname, newparent, newname)
    if ret < 0:
      raise HDF5ExtError("Problems moving the node %s to %s" %
                         (oldpathname, newpathname) )
    return ret



cdef class Leaf(Node):
  # Instance variables declared in .pxd


  def _g_new(self, where, name, init):
    if init:
      # Put this info to 0 just when the class is initialized
      self.dataset_id = -1
      self.type_id = -1
      self.base_type_id = -1
    super(Leaf, self)._g_new(where, name, init)


  def _g_loadEnum(self):
    """_g_loadEnum() -> (Enum, npType)
    Load enumerated type associated with this array.

    This method loads the HDF5 enumerated type associated with this
    array.  It returns an `Enum` instance built from that, and the
    NumPy type used to encode it.
    """

    cdef hid_t enumId, type_id
    cdef char  byteorder[11]  # "irrelevant" fits well here

    if self._c_classId == "VLARRAY":
      # For VLArray objects, the interesting type is the base type
      type_id = self.base_type_id
    else:
      type_id = self.type_id
    # Get the enumerated type
    enumId = getTypeEnum(type_id)
    # Get the byteorder
    get_order(type_id, byteorder)
    # Get the Enum and NumPy types and close the HDF5 type.
    try:
      return enumFromHDF5(enumId, byteorder)
    finally:
      # (Yes, the ``finally`` clause *is* executed.)
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")


  def _g_flush(self):
    # Flush the dataset (in fact, the entire buffers in file!)
    if self.dataset_id >= 0:
        H5Fflush(self.dataset_id, H5F_SCOPE_GLOBAL)


  def _g_close(self):
    # Close dataset in HDF5 space
    # Release resources
    if self.type_id >= 0:
      H5Tclose(self.type_id)
    if self.base_type_id >= 0:
      H5Tclose(self.base_type_id)
    if self.dataset_id >= 0:
      H5Dclose(self.dataset_id)


cdef class Array(Leaf):
  # Instance variables declared in .pxd


  def _createArray(self, ndarray nparr, char *title):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef int enumtype
    cdef long itemsize
    cdef char *complib, *version, *class_
    cdef object type_, dtype, ptype, byteorder

    dtype = nparr.dtype.base
    # Get the ptype
    type_ = nparr.dtype.type
    try:
      if type_ == numpy.string_:
        ptype = "string"
      elif type_ == numpy.bool_:
        ptype = "bool"
      else:
        ptype = type_.__name__  # the PyTables string type
      enumtype = PTTypeToNPCode[ptype]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported right now. Sorry about that.
            """ % repr(type_)

    # Get the HDF5 type associated with this numpy type
    itemsize = dtype.itemsize
    byteorder = byteorders[dtype.byteorder]
    self.type_id = convArrayType(enumtype, itemsize, byteorder)
    if self.type_id < 0:
      raise TypeError, \
        """Type '%s' is not supported right now. Sorry about that.""" \
        % ptype

    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Allocate space for the dimension axis info and fill it
    self.rank = nparr.nd
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    for i from  0 <= i < self.rank:
      self.dims[i] = nparr.dimensions[i]
    # Save the array
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    self.dataset_id = H5ARRAYmake(self.parent_id, self.name, class_, title,
                                  version, self.rank, self.dims,
                                  self.extdim, self.type_id, NULL, NULL,
                                  self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    return (self.dataset_id, dtype, ptype)


  def _createEArray(self, char *title):
    cdef int i, enumtype
    cdef herr_t ret
    cdef void *rbuf
    cdef char *complib, *version, *class_
    cdef void *fill_value
    cdef int itemsize
    cdef object atom, kind, byteorder

    atom = self.atom
    itemsize = atom.itemsize
    try:
      enumtype = PTTypeToNPCode[atom.type]
    except KeyError:
      raise TypeError( "type ``%s`` is not supported right now; "
                       "sorry about that" % atom.type )

    kind = atom.kind
    if kind == "string":
      byteorder = "irrelevant"
    else:
      # Only support for creating objects in system byteorder
      byteorder = sys.byteorder

    if kind == 'enum':
      self.type_id = enumToHDF5(atom, byteorder)
    else:
      self.type_id = convArrayType(enumtype, itemsize, byteorder)
      if self.type_id < 0:
        raise TypeError( "type ``%s`` is not supported right now; "
                         "sorry about that" % atom.type )

    self.rank = len(self.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if self._v_chunkshape:
      self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      self.dims[i] = self.shape[i]
      if self._v_chunkshape:
        self.dims_chunk[i] = self._v_chunkshape[i]

    rbuf = NULL   # The data pointer. We don't have data to save initially
    # Manually convert some string values that can't be done automatically
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Setup the fill values
    fill_value = <void *>malloc(<size_t> itemsize)
    for i from  0 <= i < itemsize:
      (<char *>fill_value)[i] = 0

    # Create the EArray
    self.dataset_id = H5ARRAYmake(
      self.parent_id, self.name, class_, title, version,
      self.rank, self.dims, self.extdim, self.type_id, self.dims_chunk,
      fill_value, self.filters.complevel, complib,
      self.filters.shuffle, self.filters.fletcher32, rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    # Release resources
    free(fill_value)

    return self.dataset_id


  def _openArray(self):
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef char byteorder[11]  # "irrelevant" fits easily here
    cdef int i, enumtype
    cdef int extdim
    cdef hid_t base_type_id
    cdef herr_t ret
    cdef object shape, type_, dtype

    # Open the dataset (and keep it open)
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems opening dataset %s" % self.name)
    # Get the datatype handle (and keep it open)
    self.type_id = H5Dget_type(self.dataset_id)
    if self.type_id < 0:
      raise HDF5ExtError("Problems getting type id for dataset %s" % self.name)

    # Get the rank for this array object
    if H5ARRAYget_ndims(self.dataset_id, self.type_id, &self.rank) < 0:
      raise HDF5ExtError("Problems getting ndims!")
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    self.maxdims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type (of base class)
    ret = H5ARRAYget_info(self.dataset_id, self.type_id,
                          self.dims, self.maxdims,
                          &base_type_id, &class_id, byteorder)
    if ret < 0:
      raise HDF5ExtError("Unable to get array info.")

    self.extdim = -1  # default is non-chunked Array
    # Get the extendeable dimension (if any)
    for i from 0 <= i < self.rank:
      if self.maxdims[i] == -1:
        self.extdim = i
        break

    # Allocate space for the dimension chunking info
    self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if ((H5ARRAYget_chunkshape(self.dataset_id, self.rank,
                              self.dims_chunk)) < 0):
      if self.extdim >= 0 or self.__class__.__name__ == 'CArray':
        raise HDF5ExtError, "Problems getting the chunkshapes!"
    # Get the array type & size
    type_size = getArrayType(base_type_id, &enumtype)
    if type_size < 0:
      raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id

    H5Tclose(base_type_id)    # Release resources

    # Get the shape and chunkshapes as python tuples
    shape = getshape(self.rank, self.dims)
    chunkshapes = getshape(self.rank, self.dims_chunk)

    # Finally, get the dtype
    type_ = NPCodeToType.get(enumtype, "int32")
    if type_ == numpy.string_:
      dtype = numpy.dtype("S%s"%type_size)
    else:
      dtype = numpy.dtype(type_).newbyteorder(byteorder)

    return (self.dataset_id, dtype, NPCodeToPTType[enumtype],
            shape, chunkshapes)


  def _convertTypes(self, object nparr, int sense):
    """Converts time64 elements in 'nparr' between NumPy and HDF5 formats.

    NumPy to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to NumPy conversion is performed.
    The conversion is done in place, i.e. 'nparr' is modified.
    """

    # This should be generalised to support other type conversions.
    if self.type == 'time64':
      convertTime64(nparr, len(nparr), sense)


  def _append(self, ndarray nparr):
    cdef int ret, extdim
    cdef hsize_t *dims_arr
    cdef void *rbuf
    cdef object shape

    # Allocate space for the dimension axis info
    dims_arr = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
        dims_arr[i] = nparr.dimensions[i]

    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Convert some NumPy types to HDF5 before storing.
    self._convertTypes(nparr, 0)

    # Append the records
    extdim = self.extdim
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYappend_records(self.dataset_id, self.type_id, self.rank,
                                self.dims, dims_arr, extdim, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems appending the elements")

    free(dims_arr)
    # Update the new dimensionality
    shape = list(self.shape)
    shape[self.extdim] = self.dims[self.extdim]
    self.shape = tuple(shape)


  def _modify(self, ndarray startl, ndarray stepl, ndarray countl,
              ndarray nparr):
    cdef int ret
    cdef void *rbuf, *temp
    cdef hsize_t *start, *step, *count

    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Get the start, step and count values
    start = <hsize_t *>startl.data
    step = <hsize_t *>stepl.data
    count = <hsize_t *>countl.data

    # Convert some NumPy types to HDF5 before storing.
    self._convertTypes(nparr, 0)

    # Modify the elements:
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYwrite_records(self.dataset_id, self.type_id, self.rank,
                               start, step, count, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Internal error modifying the elements (H5ARRAYwrite_records returned errorcode -%i)"%(-ret))


  def _truncateArray(self, hsize_t size):
    cdef int extdim
    cdef hsize_t ret

    extdim = self.extdim
    if size >= self.shape[extdim]:
      return self.shape[extdim]

    ret = H5ARRAYtruncate(self.dataset_id, extdim, size)
    if ret < 0:
      raise HDF5ExtError("Problems truncating the EArray node.")

    # Update the new dimensionality
    shape = list(self.shape)
    shape[self.extdim] = size
    self.shape = tuple(shape)


  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step,
                 ndarray nparr):
    cdef herr_t ret
    cdef void *rbuf
    cdef hsize_t nrows
    cdef int extdim

    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Number of rows to read
    nrows = get_len_of_range(start, stop, step)
    if hasattr(self, "extdim"):
      extdim = self.extdim
    else:
      exdim = -1

    # Do the physical read
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYread(self.dataset_id, self.type_id, start, nrows, step,
                      extdim, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Convert some HDF5 types to NumPy after reading.
    self._convertTypes(nparr, 1)

    return


  def _g_readSlice(self, ndarray startl, ndarray stopl, ndarray stepl,
                   ndarray bufferl):
    cdef herr_t ret
    cdef hsize_t *start, *stop, *step
    cdef void *rbuf

    # Get the pointer to the buffer data area of startl, stopl and stepl arrays
    start = <hsize_t *>startl.data
    stop = <hsize_t *>stopl.data
    step = <hsize_t *>stepl.data
    # Get the pointer to the buffer data area
    rbuf = bufferl.data

    # Do the physical read
    ret = H5ARRAYreadSlice(self.dataset_id, self.type_id,
                           start, stop, step, rbuf)
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Convert some HDF5 types to NumPy after reading
    self._convertTypes(bufferl, 1)

    return


  def __dealloc__(self):
    if self.dims:
      free(<void *>self.dims)
    if self.maxdims:
      free(<void *>self.maxdims)
    if self.dims_chunk:
      free(self.dims_chunk)



cdef class VLArray(Leaf):
  # Instance variables
  cdef int     rank
  cdef hsize_t *dims
  cdef hsize_t nrecords
  cdef int     scalar


  def _createArray(self, char *title):
    cdef int i, enumtype
    cdef herr_t ret
    cdef void *rbuf
    cdef char *complib, *version, *class_
    cdef object type_, kind, byteorder, itemsize

    atom = self.atom
    if not hasattr(atom, 'size'):  # it is a pseudo-atom
      type_ = atom.base.type
      itemsize = atom.base.itemsize
    else:
      type_ = atom.type
      itemsize = atom.itemsize

    try:
      enumtype = PTTypeToNPCode[type_]
    except KeyError:
      raise TypeError( "type ``%s`` is not supported right now; "
                       "sorry about that" % type_ )

    kind = atom.kind
    if kind == "string":
      byteorder = "irrelevant"
    else:
      # Only support for creating objects in system byteorder
      byteorder = sys.byteorder

    if kind == 'enum':
      self.base_type_id = enumToHDF5(atom, byteorder)
    else:
      # Get the HDF5 type id
      self.base_type_id = convArrayType(enumtype, itemsize, byteorder)
      if self.base_type_id < 0:
        raise TypeError( "type ``%s`` is not supported right now; "
                         "orry about that." % type_)

    # Allocate space for the dimension axis info
    if atom.shape == ():
      self.rank = 1
      self.scalar = 1
    else:
      self.rank = len(atom.shape)
      self.scalar = 0

    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      if atom.shape == ():
        self.dims[i] = 1
      else:
        self.dims[i] = atom.shape[i]

    rbuf = NULL   # We don't have data to save initially

    # Manually convert some string values that can't be done automatically
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Create the vlarray
    self.dataset_id = H5VLARRAYmake(self.parent_id, self.name, class_, title,
                                    version, self.rank, self.scalar,
                                    self.dims, self.base_type_id,
                                    self._v_chunkshape[0], rbuf,
                                    self.filters.complevel, complib,
                                    self.filters.shuffle,
                                    self.filters.fletcher32,
                                    rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the VLArray.")
    self.nrecords = 0  # Initialize the number of records saved

    # Get the datatype handle (and keep it open)
    self.type_id = H5Dget_type(self.dataset_id)

    return self.dataset_id


  def _openArray(self):
    cdef char byteorder[11]  # "irrelevant" fits easily here
    cdef int i, enumtype
    cdef herr_t ret
    cdef hsize_t nrecords, chunksize
    cdef object shape, dtype, type_

    # Open the dataset (and keep it open)
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems opening dataset %s" % self.name)
    # Get the datatype handle (and keep it open)
    self.type_id = H5Dget_type(self.dataset_id)
    if self.type_id < 0:
      raise HDF5ExtError("Problems getting type id for dataset %s" % self.name)

    # Get the rank for the atom in the array object
    ret = H5VLARRAYget_ndims(self.dataset_id, self.type_id, &self.rank)
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type (of base class)
    H5VLARRAYget_info(self.dataset_id, self.type_id, &nrecords,
                      self.dims, &self.base_type_id, byteorder)

    # Get the array type & size
    self._basesize = getArrayType(self.base_type_id, &enumtype)
    if self._basesize < 0:
      raise TypeError, "The HDF5 class of object does not seem VLEN. Sorry!"

    # Get the type of the atomic type
    type_ = NPCodeToType.get(enumtype, None)
    if type_ == numpy.string_:
      self._atomicdtype = numpy.dtype("S%s"%self._basesize)
    else:
      self._atomicdtype = numpy.dtype(type_).newbyteorder(byteorder)
    self._atomictype = NPCodeToPTType[enumtype]

    # Get the size and shape of the atomic type
    self._atomicshape = getshape(self.rank, self.dims)
    self._atomicsize = self._basesize
    for i from 0 <= i < self.rank:
      self._atomicsize = self._atomicsize * self.dims[i]

    # Get the chunkshape (VLArrays are unidimensional entities)
    H5ARRAYget_chunkshape(self.dataset_id, 1, &chunksize)

    self.nrecords = nrecords  # Initialize the number of records saved
    return self.dataset_id, nrecords, (chunksize,)


  def _convertTypes(self, object nparr, int sense):
    """Converts Time64 elements in 'nparr' between NumPy and HDF5 formats.

    NumPy to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to NumPy conversion is performed.
    The conversion is done in place, i.e. 'nparr' is modified.
    """

    # This should be generalised to support other type conversions.
    if self._atomictype == 'time64':
      convertTime64(nparr, len(nparr), sense)


  def _append(self, ndarray nparr, int nobjects):
    cdef int ret
    cdef void *rbuf

    # Get the pointer to the buffer data area
    if nobjects:
      rbuf = nparr.data
      # Convert some NumPy types to HDF5 before storing.
      self._convertTypes(nparr, 0)
    else:
      rbuf = NULL

    # Append the records:
    Py_BEGIN_ALLOW_THREADS
    ret = H5VLARRAYappend_records(self.dataset_id, self.type_id,
                                  nobjects, self.nrecords, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems appending the records.")

    self.nrecords = self.nrecords + 1


  def _modify(self, hsize_t nrow, ndarray nparr, int nobjects):
    cdef int ret
    cdef void *rbuf

    # Get the pointer to the buffer data area
    rbuf = nparr.data
    if nobjects:
      # Convert some NumPy types to HDF5 before storing.
      self._convertTypes(nparr, 0)

    # Append the records:
    Py_BEGIN_ALLOW_THREADS
    ret = H5VLARRAYmodify_records(self.dataset_id, self.type_id,
                                  nrow, nobjects, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems modifying the record.")

    return nobjects


  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step):
    cdef int i
    cdef size_t vllen
    cdef herr_t ret
    cdef hvl_t *rdata
    cdef hsize_t nrows
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef object buf, nparr, shape, datalist, dtype

    # Compute the number of rows to read
    nrows = get_len_of_range(start, stop, step)
    if start + nrows > self.nrows:
      raise HDF5ExtError(
        "Asking for a range of rows exceeding the available ones!.")

    # Now, read the chunk of rows
    Py_BEGIN_ALLOW_THREADS
    # Allocate the necessary memory for keeping the row handlers
    rdata = <hvl_t *>malloc(<size_t>nrows*sizeof(hvl_t))
    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle
    mem_space_id = H5Screate_simple(1, &nrows, NULL)
    # Select the data to be read
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start, &step, &nrows, NULL)
    # Do the actual read
    ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                  H5P_DEFAULT, rdata)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError(
        "VLArray._readArray: Problems reading the array data.")

    datalist = []
    for i from 0 <= i < nrows:
      # Number of atoms in row
      vllen = rdata[i].len
      # Get the pointer to the buffer data area
      if vllen > 0:
        # Create a buffer to keep this info. It is important to do a
        # copy, because we will dispose the buffer memory later on by
        # calling the H5Dvlen_reclaim. PyString_FromStringAndSize does this.
        buf = PyString_FromStringAndSize(<char *>rdata[i].p,
                                         vllen*self._atomicsize)
      else:
        # Case where there is info with zero lentgh
        buf = None
      # Compute the shape for the read array
      shape = list(self._atomicshape)   # a copy is done: important!
      shape.insert(0, vllen)  # put the length at the beginning of the shape
      nparr = numpy.ndarray(buffer=buf, dtype=self._atomicdtype, shape=shape)
      # Set the writeable flag for this ndarray object
      nparr.flags.writeable = True
      # Convert some HDF5 types to NumPy after reading.
      self._convertTypes(nparr, 1)
      # Append this array to the output list
      datalist.append(nparr)

    # Release resources
    # Reclaim all the (nested) VL data
    ret = H5Dvlen_reclaim(self.type_id, mem_space_id, H5P_DEFAULT, rdata)
    if ret < 0:
      raise HDF5ExtError("VLArray._readArray: error freeing the data buffer.")
    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)
    # Free the amount of row pointers to VL row data
    free(rdata)

    return datalist


  def __dealloc__(self):
    if self.dims:
      free(<void *>self.dims)



cdef class UnImplemented(Leaf):


  def _openUnImplemented(self):
    cdef object shape
    cdef char byteorder[11]  # "irrelevant" fits easily here

    # Get info on dimensions
    shape = H5UIget_info(self.parent_id, self.name, byteorder)
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    return (shape, byteorder, self.dataset_id)


  def _g_close(self):
    H5Dclose(self.dataset_id)



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
