########################################################################
#
#       License: BSD
#       Created: September 21, 2002
#       Author:  Francesc Alted - faltet@pytables.com
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
from tables.utils import \
     checkFileAccess, byteorders, correct_byteorder, SizeType

from tables.atom import Atom

from tables.utilsExtension import \
     enumToHDF5, enumFromHDF5, getTypeEnum, isHDF5File, isPyTablesFile, \
     AtomToHDF5Type, AtomFromHDF5Type, loadEnum, HDF5ToNPExtType

from utilsExtension cimport malloc_dims, get_native_type


# Types, constants, functions, classes & other objects from everywhere
from definitions cimport  \
     memcpy, strdup, malloc, free, \
     Py_ssize_t, PyObject_AsReadBuffer, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, PyString_AsString, \
     PyString_FromStringAndSize, PyDict_Contains, PyDict_GetItem, \
     Py_INCREF, Py_DECREF, \
     import_array, ndarray, dtype, \
     time_t, size_t, uintptr_t, hid_t, herr_t, hsize_t, hvl_t, \
     H5G_GROUP, H5G_DATASET, H5G_stat_t, H5T_class_t, H5T_sign_t, \
     H5F_SCOPE_GLOBAL, H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, \
     H5P_DEFAULT, H5T_SGN_NONE, H5T_SGN_2, H5T_DIR_DEFAULT, H5S_SELECT_SET, \
     H5get_libversion, H5check_version, H5Fcreate, H5Fopen, H5Fclose, \
     H5Fflush, H5Fget_vfd_handle, \
     H5Gcreate, H5Gopen, H5Gclose, H5Glink, H5Gunlink, H5Gmove, \
     H5Gmove2, H5Gget_objinfo, \
     H5Dopen, H5Dclose, H5Dread, H5Dget_type, \
     H5Tget_native_type, H5Tget_super, H5Tget_class, H5Tcopy, H5Dget_space, \
     H5Dvlen_reclaim, H5Adelete, H5Aget_num_attrs, H5Aget_name, H5Aopen_idx, \
     H5Aread, H5Aclose, H5Tclose, H5Pcreate, H5Pclose, \
     H5Pset_cache, H5Pset_sieve_buf_size, H5Pset_fapl_log, \
     H5Sselect_hyperslab, H5Screate_simple, H5Sget_simple_extent_ndims, \
     H5Sget_simple_extent_dims, H5Sclose, \
     H5Tis_variable_str, H5Tget_sign, \
     H5ATTRset_attribute, H5ATTRset_attribute_string, \
     H5ATTRget_attribute, H5ATTRget_attribute_string, \
     H5ATTRfind_attribute, H5ATTRget_type_ndims, H5ATTRget_dims, \
     H5ARRAYget_ndims, H5ARRAYget_info, \
     set_cache_size, get_objinfo, Giterate, Aiterate, H5UIget_info, \
     get_len_of_range, get_order, set_order, \
     conv_float64_timeval32


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

  herr_t H5ARRAYget_chunkshape(hid_t dataset_id, int rank, hsize_t *dims_chunk)


# Functions for dealing with VLArray objects
cdef extern from "H5VLARRAY.h":

  herr_t H5VLARRAYmake( hid_t loc_id, char *dset_name, char *class_,
                        char *title, char *obversion,
                        int rank, hsize_t *dims, hid_t type_id,
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

cdef hsize_t *npy_malloc_dims(int rank, npy_intp *pdims):
  """Returns a malloced hsize_t dims from a npy_intp *pdims."""
  cdef int i
  cdef hsize_t *dims

  dims = NULL
  if rank > 0:
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    for i from 0 <= i < rank:
      dims[i] = pdims[i]
  return dims


cdef object getshape(int rank, hsize_t *dims):
  """Return a shape (tuple) from a dims C array of rank dimensions."""
  cdef int i
  cdef object shape

  shape = []
  for i from 0 <= i < rank:
    shape.append(SizeType(dims[i]))

  return tuple(shape)


# Helper function for quickly fetch an attribute string
cdef object get_attribute_string_or_none(node_id, attr_name):
  """Returns a string attribute if it exists in node_id.

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


# Get the numpy dtype scalar attribute from an HDF5 type as fast as possible
cdef object get_dtype_scalar(hid_t type_id, H5T_class_t class_id,
                             size_t itemsize):
  cdef H5T_sign_t sign
  cdef object stype

  if class_id == H5T_BITFIELD:
    stype = "b1"
  elif class_id ==  H5T_INTEGER:
    # Get the sign
    sign = H5Tget_sign(type_id)
    if (sign > 0):
      stype = "i%s" % (itemsize)
    else:
      stype = "u%s" % (itemsize)
  elif class_id ==  H5T_FLOAT:
    stype = "f%s" % (itemsize)
  elif class_id ==  H5T_STRING:
    if H5Tis_variable_str(type_id):
      raise TypeError("variable length strings are not supported yet")
    stype = "S%s" % (itemsize)

  return numpy.dtype(stype)



# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef hid_t   access_plist
  cdef char    *name


  def __cinit__(self, object name, char *mode, char *title,
                char *root, object filters,
                size_t metadataCacheSize, size_t nodeCacheSize):
    # Create a new file using default properties
    self.name = name
    self.mode = pymode = mode

    # These fields can be seen from Python.
    self._v_new = None  # this will be computed later
    # """Is this file going to be created from scratch?"""
    self._isPTFile = True  # assume a PyTables file by default
    # """Does this HDF5 file have a PyTables format?"""

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
          # I'm going to disable the next warning because
          # it should be enough to map unsupported objects to
          # UnImplemented class.
          # F. Alted 2007-02-14
#           warnings.warn("file ``%s`` is a valid HDF5 file, " \
#                         "but is not in PyTables format; " \
#                         "attempting to determine its contents " \
#                         "by using the HDF5 metadata" % name)
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


  def fileno(self):
    """Return the underlying OS integer file descriptor.

    This is needed for lower-level file interfaces, such as the ``fcntl``
    module.
    """

    cdef void *file_handle
    cdef uintptr_t *descriptor
    cdef herr_t err
    err = H5Fget_vfd_handle(self.file_id, H5P_DEFAULT, &file_handle)
    if err < 0:
      raise HDF5Error(
        "Problems getting file descriptor for file ``%s``", self.name)
    # Convert the 'void *file_handle' into an 'int *descriptor'
    descriptor = <uintptr_t *>file_handle
    return descriptor[0]


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
    self.name =  PyString_AsString(node._v_name)
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


  def _g_setAttr(self, char *name, object value):
    """Save Python or NumPy objects as HDF5 attributes.

    Scalar Python objects, scalar NumPy & 0-dim NumPy objects will all be
    saved as H5T_SCALAR type.  N-dim NumPy objects will be saved as H5T_ARRAY
    type.
    """

    cdef int ret
    cdef hid_t type_id
    cdef hsize_t *dims
    cdef ndarray ndv
    cdef object byteorder, baseatom

    # Convert a NumPy scalar into a NumPy 0-dim ndarray
    if isinstance(value, numpy.generic):
      value = numpy.array(value)

    # Check if value is a NumPy ndarray and of a supported type
    if (isinstance(value, numpy.ndarray) and
        value.dtype.kind in ('S', 'b', 'i', 'u', 'f', 'c')):
      # Get the associated native HDF5 type of the scalar type
      baseatom = Atom.from_dtype(value.dtype.base)
      byteorder = byteorders[value.dtype.byteorder]
      type_id = AtomToHDF5Type(baseatom, byteorder)
      # Get dimensionality info
      ndv = <ndarray>value
      dims = npy_malloc_dims(ndv.nd, ndv.dimensions)
      # Actually write the attribute
      ret = H5ATTRset_attribute(self.dataset_id, name, type_id,
                                ndv.nd, dims, ndv.data)
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
    cdef hid_t mem_type, dset_id, type_id, native_type
    cdef int rank, ret, enumtype
    cdef void *rbuf
    cdef char *str_value
    cdef ndarray ndvalue
    cdef object shape, stype_atom, shape_atom, retvalue

    dset_id = self.dataset_id
    dims = NULL

    ret = H5ATTRget_type_ndims(dset_id, attrname, &type_id, &class_id,
                               &type_size, &rank )
    if ret < 0:
      raise HDF5ExtError("Can't get type info on attribute %s in node %s." %
                         (attrname, self.name))

    # Call a fast function for scalar values and typical class types
    if (rank == 0 and class_id == H5T_STRING):
        ret = H5ATTRget_attribute_string(dset_id, attrname, &str_value)
        retvalue = numpy.string_(str_value)
        # Important to release attr_value, because it has been malloc'ed!
        if str_value: free(str_value)
        H5Tclose(type_id)
        return retvalue
    elif (rank == 0 and class_id in (H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT)):
      dtype = get_dtype_scalar(type_id, class_id, type_size)
      shape = ()
    else:
      # Attribute is multidimensional
      # Check that class_id is a supported type for attributes
      # [complex types are COMPOUND]
      if class_id not in (H5T_STRING, H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT,
                          H5T_COMPOUND, H5T_ARRAY):
        warnings.warn("""\
Type of attribute '%s' in node '%s' is not supported. Sorry about that!"""
                      % (attrname, self.name))
        return None

      # Get the dimensional info
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      ret = H5ATTRget_dims(dset_id, attrname, dims)
      if ret < 0:
        raise HDF5ExtError("Can't get dims info on attribute %s in node %s." %
                           (attrname, self.name))

      # Get the dimensional info
      shape = getshape(rank, dims)
      # dims is not needed anymore
      free(<void *> dims)
      # Get the atom from the type_id
      # This can be quite CPU consuming, and for attributes we just need
      # the numpy dtype
      #atom = AtomFromHDF5Type(type_id, issue_error=False)
      stype_atom, shape_atom = HDF5ToNPExtType(type_id, False)
      if not stype_atom:
        # This class is not supported. Instead of raising a TypeError, issue a
        # warning explaining the problem. This will allow to continue browsing
        # native HDF5 files, while informing the user about the problem.
        warnings.warn("""\
Unsupported type for attribute '%s' in node '%s'. Offending HDF5 class: %d"""
                      % (attrname, self.name, class_id))
        return None
      # Get the dtype
      dtype = numpy.dtype((stype_atom, shape_atom))

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    native_type_id = get_native_type(type_id)
    # Get the container for data
    ndvalue = numpy.empty(dtype=dtype, shape=shape)
    # Get the pointer to the buffer data area
    rbuf = ndvalue.data
    # Actually read the attribute from disk
    ret = H5ATTRget_attribute(dset_id, attrname, native_type_id, rbuf)
    if ret < 0:
      raise HDF5ExtError("Attribute %s exists in node %s, but can't get it."\
                         % (attrname, self.name))
    H5Tclose(native_type_id)
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
    # """The name of this node in its parent group."""
    self.parent_id = where._v_objectID
    # """The identifier of the parent group."""


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


  def _g_get_objinfo(self, object h5name):
    """Check whether 'name' is a children of 'self' and return its type. """
    cdef int ret
    cdef object node_type

    node_type = "Unknown"
    ret = get_objinfo(self.group_id, h5name)
    if ret == -2:
      node_type = "NoSuchNode"
    elif ret == H5G_GROUP:
      node_type = "Group"
    elif ret == H5G_DATASET:
      node_type = "Leaf"
    return node_type


  def _g_listGroup(self):
    """Return a tuple with the groups and the leaves hanging from self."""
    if get_objinfo(self.parent_id, ".") < 0:
      # Refresh the parent_id because the parent seems closed
      self.parent_id = self._v_parent._v_objectID
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
      self.disk_type_id = -1
    super(Leaf, self)._g_new(where, name, init)


  cdef _get_type_ids(self):
    """Get the disk and native HDF5 types associated with this leaf.

    It is guaranteed that both disk and native types are not the same
    descriptor (so that it is safe to close them separately).
    """
    cdef hid_t disk_type_id, native_type_id

    disk_type_id = H5Dget_type(self.dataset_id)
    native_type_id = get_native_type(disk_type_id)
    return (disk_type_id, native_type_id)


  cdef _convertTime64(self, ndarray nparr, int sense):
    """Converts a NumPy of Time64 elements between NumPy and HDF5 formats.

    NumPy to HDF5 conversion is performed when 'sense' is 0.  Otherwise, HDF5
    to NumPy conversion is performed.  The conversion is done in place,
    i.e. 'nparr' is modified.
    """

    cdef void *t64buf
    cdef long byteoffset, bytestride, nelements
    cdef hsize_t nrecords

    byteoffset = 0   # NumPy objects doesn't have an offset
    if nparr.shape == ():
      # 0-dim array does contain *one* element
      nrecords = 1
      bytestride = 8
    else:
      nrecords = len(nparr)
      bytestride = nparr.strides[0]  # supports multi-dimensional recarray
    nelements = nparr.size / nrecords
    t64buf = nparr.data

    conv_float64_timeval32(
      t64buf, byteoffset, bytestride, nrecords, nelements, sense)


  def _g_flush(self):
    # Flush the dataset (in fact, the entire buffers in file!)
    if self.dataset_id >= 0:
        H5Fflush(self.dataset_id, H5F_SCOPE_GLOBAL)


  def _g_close(self):
    # Close dataset in HDF5 space
    # Release resources
    if self.type_id >= 0:
      H5Tclose(self.type_id)
    if self.disk_type_id >= 0:
      H5Tclose(self.disk_type_id)
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
    cdef char *complib, *version, *class_
    cdef object dtype, atom

    dtype = nparr.dtype.base
    # Get the HDF5 type associated with this numpy type
    atom = Atom.from_dtype(dtype)
    self.disk_type_id = AtomToHDF5Type(atom, self.byteorder)

    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Allocate space for the dimension axis info and fill it
    self.rank = nparr.nd
    self.dims = npy_malloc_dims(nparr.nd, nparr.dimensions)
    # Save the array
    complib = PyString_AsString(self.filters.complib or '')
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    self.dataset_id = H5ARRAYmake(self.parent_id, self.name, class_, title,
                                  version, self.rank, self.dims,
                                  self.extdim, self.disk_type_id, NULL, NULL,
                                  self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    self.type_id = get_native_type(self.disk_type_id)

    return (self.dataset_id, atom)


  def _createCArray(self, char *title):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef char *complib, *version, *class_
    cdef void *fill_value
    cdef int itemsize
    cdef ndarray buf
    cdef object atom

    atom = self.atom
    itemsize = atom.itemsize
    self.disk_type_id = AtomToHDF5Type(atom, self.byteorder)

    self.rank = len(self.shape)
    self.dims = malloc_dims(self.shape)
    if self.chunkshape:
      self.dims_chunk = malloc_dims(self.chunkshape)

    rbuf = NULL   # The data pointer. We don't have data to save initially
    # Manually convert some string values that can't be done automatically
    complib = PyString_AsString(self.filters.complib or '')
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Set the fill values
    fill_value = <void *>malloc(<size_t> itemsize)
    buf = numpy.array(atom.dflt, dtype=atom.dtype)
    memcpy(fill_value, buf.data, itemsize)

    # Create the CArray/EArray
    self.dataset_id = H5ARRAYmake(
      self.parent_id, self.name, class_, title, version,
      self.rank, self.dims, self.extdim, self.disk_type_id, self.dims_chunk,
      fill_value, self.filters.complevel, complib,
      self.filters.shuffle, self.filters.fletcher32, rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    self.type_id = get_native_type(self.disk_type_id)

    # Release resources
    free(fill_value)

    return self.dataset_id


  def _openArray(self):
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef char byteorder[11]  # "irrelevant" fits easily here
    cdef int i
    cdef int extdim
    cdef herr_t ret
    cdef object shape, chunkshapes, atom

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems opening dataset %s" % self.name)
    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()
    # Get the atom for this type
    atom = AtomFromHDF5Type(self.disk_type_id)

    # Get the rank for this array object
    if H5ARRAYget_ndims(self.dataset_id, self.type_id, &self.rank) < 0:
      raise HDF5ExtError("Problems getting ndims!")
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    self.maxdims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type (of base class)
    ret = H5ARRAYget_info(self.dataset_id, self.disk_type_id,
                          self.dims, self.maxdims,
                          &class_id, byteorder)
    if ret < 0:
      raise HDF5ExtError("Unable to get array info.")

    # Get the extendable dimension (if any)
    self.extdim = -1  # default is non-extensible Array
    for i from 0 <= i < self.rank:
      if self.maxdims[i] == -1:
        self.extdim = i
        break

    # Get the shape as a python tuple
    shape = getshape(self.rank, self.dims)

    # Allocate space for the dimension chunking info
    self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if H5ARRAYget_chunkshape(self.dataset_id, self.rank, self.dims_chunk) < 0:
      # The Array class is not chunked!
      chunkshapes = None
    else:
      # Get the chunkshape as a python tuple
      chunkshapes = getshape(self.rank, self.dims_chunk)

    # Get the byteorder
    self.byteorder = correct_byteorder(atom.type, byteorder)

    return (self.dataset_id, atom, shape, chunkshapes)


  def _append(self, ndarray nparr):
    cdef int ret, extdim
    cdef hsize_t *dims_arr
    cdef void *rbuf
    cdef object shape

    # Allocate space for the dimension axis info
    dims_arr = npy_malloc_dims(self.rank, nparr.dimensions)
    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Convert some NumPy types to HDF5 before storing.
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 0)

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
    shape[self.extdim] = SizeType(self.dims[self.extdim])
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
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 0)

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
    self.dims[self.extdim] = size
    shape = list(self.shape)
    shape[self.extdim] = SizeType(size)
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

    if self.atom.kind == 'time':
      # Swap the byteorder by hand (this is not currently supported by HDF5)
      if H5Tget_order(self.type_id) != platform_byteorder:
        nparr.byteswap(True)

    # Convert some HDF5 types to NumPy after reading.
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 1)

    return


  def _g_readSlice(self, ndarray startl, ndarray stopl, ndarray stepl,
                   ndarray nparr):
    cdef herr_t ret
    cdef hsize_t *start, *stop, *step
    cdef void *rbuf

    # Get the pointer to the buffer data area of startl, stopl and stepl arrays
    start = <hsize_t *>startl.data
    stop = <hsize_t *>stopl.data
    step = <hsize_t *>stepl.data
    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Do the physical read
    ret = H5ARRAYreadSlice(self.dataset_id, self.type_id,
                           start, stop, step, rbuf)
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    if self.atom.kind == 'time':
      # Swap the byteorder by hand (this is not currently supported by HDF5)
      if H5Tget_order(self.type_id) != platform_byteorder:
        nparr.byteswap(True)

    # Convert some HDF5 types to NumPy after reading
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 1)

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
  cdef hsize_t nrecords

  def _createArray(self, char *title):
    cdef int rank
    cdef hsize_t *dims
    cdef herr_t ret
    cdef void *rbuf
    cdef char *complib, *version, *class_
    cdef object type_, itemsize, atom, scatom

    atom = self.atom
    if not hasattr(atom, 'size'):  # it is a pseudo-atom
      atom = atom.base

    # Get the HDF5 type of the *scalar* atom
    scatom = atom.copy(shape=())
    self.base_type_id = AtomToHDF5Type(scatom, self.byteorder)

    # Allocate space for the dimension axis info
    rank = len(atom.shape)
    dims = malloc_dims(atom.shape)

    rbuf = NULL   # We don't have data to save initially

    # Manually convert some string values that can't be done automatically
    complib = PyString_AsString(self.filters.complib or '')
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Create the vlarray
    self.dataset_id = H5VLARRAYmake(self.parent_id, self.name, class_, title,
                                    version, rank, dims, self.base_type_id,
                                    self.chunkshape[0], rbuf,
                                    self.filters.complevel, complib,
                                    self.filters.shuffle,
                                    self.filters.fletcher32,
                                    rbuf)
    if dims:
      free(<void *>dims)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the VLArray.")
    self.nrecords = 0  # Initialize the number of records saved

    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()

    return self.dataset_id


  def _openArray(self):
    cdef char byteorder[11]  # "irrelevant" fits easily here
    cdef int i, enumtype
    cdef int rank
    cdef hsize_t *dims
    cdef herr_t ret
    cdef hsize_t nrecords, chunksize
    cdef object shape, dtype, type_

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems opening dataset %s" % self.name)
    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()

    # Get the rank for the atom in the array object
    ret = H5VLARRAYget_ndims(self.dataset_id, self.type_id, &rank)
    # Allocate space for the dimension axis info
    dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    # Get info on dimensions & types (of base class)
    H5VLARRAYget_info(self.dataset_id, self.disk_type_id, &nrecords,
                      dims, &self.base_type_id, byteorder)
    # Get the shape for the base type
    shape = getshape(rank, dims)
    if dims:
      free(<void *>dims)

    # Get the scalar atom and the atom for this type
    scatom = AtomFromHDF5Type(self.base_type_id)
    atom = scatom.copy(shape=shape)

    # Get some properties of the atomic type
    self._atomicdtype = scatom.dtype
    self._atomictype = scatom.type
    self._atomicshape = atom.shape
    self._atomicsize = atom.size

    # Get the byteorder
    self.byteorder = correct_byteorder(atom.type, byteorder)

    # Get the chunkshape (VLArrays are unidimensional entities)
    H5ARRAYget_chunkshape(self.dataset_id, 1, &chunksize)

    self.nrecords = nrecords  # Initialize the number of records saved
    return self.dataset_id, SizeType(nrecords), (SizeType(chunksize),), atom


  def _append(self, ndarray nparr, int nobjects):
    cdef int ret
    cdef void *rbuf

    # Get the pointer to the buffer data area
    if nobjects:
      rbuf = nparr.data
      # Convert some NumPy types to HDF5 before storing.
      if self.atom.type == 'time64':
        self._convertTime64(nparr, 0)
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
      if self.atom.type == 'time64':
        self._convertTime64(nparr, 0)

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
    cdef object buf, nparr, shape, datalist

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
      if self.atom.kind == 'time':
        # Swap the byteorder by hand (this is not currently supported by HDF5)
        if H5Tget_order(self.type_id) != platform_byteorder:
          nparr.byteswap(True)
      # Convert some HDF5 types to NumPy after reading.
      if self.atom.type == 'time64':
        self._convertTime64(nparr, 1)
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



cdef class UnImplemented(Leaf):


  def _openUnImplemented(self):
    cdef object shape
    cdef char byteorder[11]  # "irrelevant" fits easily here

    # Get info on dimensions
    shape = H5UIget_info(self.parent_id, self.name, byteorder)
    shape = tuple(map(SizeType, shape))
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
