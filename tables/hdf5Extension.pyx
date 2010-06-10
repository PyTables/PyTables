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
from tables.exceptions import HDF5ExtError, DataTypeWarning

from tables.utils import \
     checkFileAccess, byteorders, correct_byteorder, SizeType

from tables.atom import Atom

from tables.description import descr_from_dtype, Description

from tables.utilsExtension import \
     enumToHDF5, enumFromHDF5, getTypeEnum, \
     encode_filename, isHDF5File, isPyTablesFile, \
     AtomToHDF5Type, AtomFromHDF5Type, loadEnum, \
     HDF5ToNPExtType, HDF5ToNPNestedType, createNestedType, \
     setBloscMaxThreads


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
     H5S_seloper_t, H5D_FILL_VALUE_UNDEFINED, \
     H5G_UNKNOWN, H5G_GROUP, H5G_DATASET, H5G_LINK, H5G_TYPE, \
     H5T_class_t, H5T_sign_t, H5T_NATIVE_INT, \
     H5F_SCOPE_GLOBAL, H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, \
     H5P_DEFAULT, H5P_FILE_ACCESS, \
     H5T_SGN_NONE, H5T_SGN_2, H5T_DIR_DEFAULT, \
     H5S_SELECT_SET, H5S_SELECT_AND, H5S_SELECT_NOTB, \
     H5get_libversion, H5check_version, H5Fcreate, H5Fopen, H5Fclose, \
     H5Fflush, H5Fget_vfd_handle, \
     H5Gcreate, H5Gopen, H5Gclose, H5Gunlink, H5Gmove, H5Gmove2, \
     H5Dopen, H5Dclose, H5Dread, H5Dwrite, H5Dget_type, \
     H5Dget_space, H5Dvlen_reclaim, \
     H5Tget_native_type, H5Tget_super, H5Tget_class, H5Tcopy, \
     H5Tclose, H5Tis_variable_str, H5Tget_sign, \
     H5Adelete, H5Aget_num_attrs, H5Aget_name, H5Aopen_idx, \
     H5Aread, H5Aclose, H5Pcreate, H5Pclose, \
     H5Pset_cache, H5Pset_sieve_buf_size, H5Pset_fapl_log, \
     H5Pset_fapl_core, \
     H5Sselect_all, H5Sselect_elements, H5Sselect_hyperslab, \
     H5Screate_simple, H5Sget_simple_extent_ndims, \
     H5Sget_simple_extent_dims, H5Sclose, \
     H5ATTRset_attribute, H5ATTRset_attribute_string, \
     H5ATTRget_attribute, H5ATTRget_attribute_string, \
     H5ATTRfind_attribute, H5ATTRget_type_ndims, H5ATTRget_dims, \
     H5ARRAYget_ndims, H5ARRAYget_info, \
     set_cache_size, get_objinfo, Giterate, Aiterate, H5UIget_info, \
     get_len_of_range, get_order, set_order, is_complex, \
     conv_float64_timeval32, truncate_dset


# Include conversion tables
include "convtypetables.pxi"


__version__ = "$Revision$"


#-------------------------------------------------------------------

# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h":

  herr_t H5ARRAYmake(hid_t loc_id, char *dset_name, char *obversion,
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

  herr_t H5ARRAYget_fill_value( hid_t dataset_id, hid_t type_id,
                                int *status, void *value)


# Functions for dealing with VLArray objects
cdef extern from "H5VLARRAY.h":

  herr_t H5VLARRAYmake( hid_t loc_id, char *dset_name, char *obversion,
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

  herr_t H5VLARRAYget_info( hid_t dataset_id, hid_t type_id,
                            hsize_t *nrecords, char *base_byteorder)



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
    if ret < 0:
      return None
    retvalue = numpy.string_(attr_value)
    # Important to release attr_value, because it has been malloc'ed!
    if attr_value:
      free(<void *>attr_value)

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

  # Try to get a NumPy type.  If this can't be done, return None.
  try:
    ntype = numpy.dtype(stype)
  except TypeError:
    ntype = None
  return ntype



# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef hid_t   access_plist
  cdef object  name


  def _g_new(self, name, pymode, **params):
    # Create a new file using default properties
    self.name = name

    # Encode the filename in case it is unicode
    encname = encode_filename(name)

    # These fields can be seen from Python.
    self._v_new = None  # this will be computed later
    # """Is this file going to be created from scratch?"""
    self._isPTFile = True  # assume a PyTables file by default
    # """Does this HDF5 file have a PyTables format?"""

    # After the following check we can be quite sure
    # that the file or directory exists and permissions are right.
    checkFileAccess(name, pymode)

    assert pymode in ('r', 'r+', 'a', 'w'), \
           "an invalid mode string ``%s`` " \
           "passed the ``checkFileAccess()`` test; " \
           "please report this to the authors" % pymode

    # Should a new file be created?
    exists = os.path.exists(name)
    self._v_new = new = not (
      pymode in ('r', 'r+') or (pymode == 'a' and exists))

    access_plist = H5Pcreate(H5P_FILE_ACCESS)
    # The line below uses the CORE driver for doing I/O from memory, not disk
    # In general it is a bad idea to do this because HDF5 will have to load
    # the contents of the file on disk prior to operate, which takes time and
    # resources.
    # F. Alted 2010-04-15
    #H5Pset_fapl_core(access_plist, 1024, 1)
    # Set parameters for chunk cache
    H5Pset_cache(access_plist, 0,
                 params['CHUNK_CACHE_NELMTS'],
                 params['CHUNK_CACHE_SIZE'],
                 params['CHUNK_CACHE_PREEMPT'])

    if pymode == 'r':
      self.file_id = H5Fopen(encname, H5F_ACC_RDONLY, access_plist)
    elif pymode == 'r+':
      self.file_id = H5Fopen(encname, H5F_ACC_RDWR, access_plist)
    elif pymode == 'a':
      if exists:
        # A test for logging.
        ## H5Pset_sieve_buf_size(access_plist, 0)
        ## H5Pset_fapl_log (access_plist, "test.log", H5FD_LOG_LOC_WRITE, 0)
        self.file_id = H5Fopen(encname, H5F_ACC_RDWR, access_plist)
      else:
        self.file_id = H5Fcreate(encname, H5F_ACC_TRUNC,
                                 H5P_DEFAULT, access_plist)
    elif pymode == 'w':
      self.file_id = H5Fcreate(encname, H5F_ACC_TRUNC,
                               H5P_DEFAULT, access_plist)

    # Set the cache size (only for HDF5 1.8.x)
    set_cache_size(self.file_id, params['METADATA_CACHE_SIZE'])

    # Set the maximum number of threads for Blosc
    setBloscMaxThreads(params['MAX_THREADS'])


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
      raise HDF5ExtError(
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
  cdef char    *name


  def _g_new(self, node):
    # Initialize the C attributes of Node object
    self.name =  PyString_AsString(node._v_name)


  def _g_listAttr(self, node):
    "Return a tuple with the attribute list"
    a = Aiterate(node._v_objectID)
    return a


  def _g_setAttr(self, node, char *name, object value):
    """Save Python or NumPy objects as HDF5 attributes.

    Scalar Python objects, scalar NumPy & 0-dim NumPy objects will all be
    saved as H5T_SCALAR type.  N-dim NumPy objects will be saved as H5T_ARRAY
    type.
    """

    cdef int ret
    cdef hid_t dset_id, type_id
    cdef hsize_t *dims
    cdef ndarray ndv
    cdef object byteorder, rabyteorder, baseatom

    # The dataset id of the node
    dset_id = node._v_objectID

    # Convert a NumPy scalar into a NumPy 0-dim ndarray
    if isinstance(value, numpy.generic):
      value = numpy.array(value)

    # Check if value is a NumPy ndarray and of a supported type
    if (isinstance(value, numpy.ndarray) and
        value.dtype.kind in ('V', 'S', 'b', 'i', 'u', 'f', 'c')):
      value = numpy.array(value)  # to get a contiguous array.  Fixes #270.
      if value.dtype.kind == 'V':
        description, rabyteorder = descr_from_dtype(value.dtype)
        byteorder = byteorders[rabyteorder]
        type_id = createNestedType(description, byteorder)
      else:
        # Get the associated native HDF5 type of the scalar type
        baseatom = Atom.from_dtype(value.dtype.base)
        byteorder = byteorders[value.dtype.byteorder]
        type_id = AtomToHDF5Type(baseatom, byteorder)
      # Get dimensionality info
      ndv = <ndarray>value
      dims = npy_malloc_dims(ndv.nd, ndv.dimensions)
      # Actually write the attribute
      ret = H5ATTRset_attribute(dset_id, name, type_id,
                                ndv.nd, dims, ndv.data)
      if ret < 0:
        raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                           (name, self._v_node))
      # Release resources
      free(<void *>dims)
      H5Tclose(type_id)
    else:
      # Object cannot be natively represented in HDF5.
      # Unicode attributes has to be pickled until we can definitely switch
      # to HDF5 1.8.x, where Unicode datatype is supported natively.
      if (isinstance(value, numpy.ndarray) and
          value.dtype.kind == 'U' and
          value.shape == ()):
        value = value[()]
      # Convert this object to a null-terminated string
      # (binary pickles are not supported at this moment)
      value = cPickle.dumps(value, 0)
      ret = H5ATTRset_attribute_string(dset_id, name, value)

    return


  # Get attributes
  def _g_getAttr(self, node, char *attrname):
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

    # The dataset id of the node
    dset_id = node._v_objectID
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
      if dtype is None:
        warnings.warn("""\
Unsupported type for attribute '%s' in node '%s'. Offending HDF5 class: %d"""
                      % (attrname, self.name, class_id), DataTypeWarning)
        self._v_unimplemented.append(attrname)
        return None
      shape = ()
    else:
      # General case

      # Get the NumPy dtype from the type_id
      try:
        stype_, shape_ = HDF5ToNPExtType(type_id, pure_numpy_types=True)
        dtype = numpy.dtype(stype_, shape_)
      except TypeError:
        # This class is not supported. Instead of raising a TypeError, issue a
        # warning explaining the problem. This will allow to continue browsing
        # native HDF5 files, while informing the user about the problem.
        warnings.warn("""\
Unsupported type for attribute '%s' in node '%s'. Offending HDF5 class: %d"""
                      % (attrname, self.name, class_id), DataTypeWarning)
        self._v_unimplemented.append(attrname)
        return None

      # Get the dimensional info
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      ret = H5ATTRget_dims(dset_id, attrname, dims)
      if ret < 0:
        raise HDF5ExtError("Can't get dims info on attribute %s in node %s." %
                           (attrname, self.name))
      shape = getshape(rank, dims)
      # dims is not needed anymore
      free(<void *> dims)

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


  def _g_remove(self, node, attrname):
    cdef int ret
    cdef hid_t dset_id

    # The dataset id of the node
    dset_id = node._v_objectID

    ret = H5Adelete(dset_id, attrname)
    if ret < 0:
      raise HDF5ExtError("Attribute '%s' exists in node '%s', but cannot be deleted." \
                         % (attrname, self.name))




cdef class Node:
  # Instance variables declared in .pxd


  def _g_new(self, where, name, init):
    self.name = strdup(name)
    # """The name of this node in its parent group."""
    self.parent_id = where._v_objectID
    # """The identifier of the parent group."""


  def _g_delete(self, parent):
    cdef int ret

    # Delete this node
    ret = H5Gunlink(parent._v_objectID, self.name)
    if ret < 0:
      raise HDF5ExtError("problems deleting the node ``%s``" % self.name)
    return ret


  def __dealloc__(self):
    free(<void *>self.name)
    self.parent_id = 0



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

    ret = get_objinfo(self.group_id, h5name)
    if ret == -2:
      node_type = "NoSuchNode"
    elif ret == H5G_UNKNOWN:
      node_type = "Unknown"
    elif ret == H5G_GROUP:
      node_type = "Group"
    elif ret == H5G_DATASET:
      node_type = "Leaf"
    elif ret == H5G_LINK:
      node_type = "SoftLink"
    elif ret == H5G_TYPE:
      node_type = "NamedType"              # Not supported yet
    else:
      node_type = "ExternalLink"
    return node_type


  def _g_listGroup(self, parent):
    """Return a tuple with the groups and the leaves hanging from self."""
    return Giterate(parent._v_objectID, self._v_objectID, self.name)


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
    if gchild_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" % \
                         (group_name, self._v_pathname))
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
    if leaf_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" % \
                         (leaf_name, self._v_pathname))
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


  def _g_truncate(self, hsize_t size):
    """Truncate a Leaf to `size` nrows."""
    cdef hsize_t ret

    ret = truncate_dset(self.dataset_id, self.maindim, size)
    if ret < 0:
      raise HDF5ExtError("Problems truncating the leaf: %s" % self)

    classname = self.__class__.__name__
    if classname in ('EArray', 'CArray'):
      # Update the new dimensionality
      self.dims[self.maindim] = size
      # Update the shape
      shape = list(self.shape)
      shape[self.maindim] = SizeType(size)
      self.shape = tuple(shape)
    elif classname in ('Table', 'VLArray'):
      self.nrows = size
    else:
      raise ValueError, "Unexpected classname:", classname


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


  def _createArray(self, ndarray nparr, char *title, object _atom):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef char *complib, *version, *class_
    cdef object dtype, atom, shape
    cdef ndarray dims

    # Get the HDF5 type associated with this numpy type
    shape = nparr.shape
    if _atom is None or _atom.shape == ():
      dtype = nparr.dtype.base
      atom = Atom.from_dtype(dtype)
    else:
      atom = _atom
      shape = shape[:-len(atom.shape)]
    self.disk_type_id = AtomToHDF5Type(atom, self.byteorder)

    # Allocate space for the dimension axis info and fill it
    dims = numpy.array(shape, dtype=numpy.intp)
    self.rank = len(shape)
    self.dims = npy_malloc_dims(self.rank, <npy_intp *>(dims.data))
    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Save the array
    complib = PyString_AsString(self.filters.complib or '')
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    self.dataset_id = H5ARRAYmake(self.parent_id, self.name, version,
                                  self.rank, self.dims,
                                  self.extdim, self.disk_type_id, NULL, NULL,
                                  self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    if self._v_file.params['PYTABLES_SYS_ATTRS']:
      # Set the conforming array attributes
      H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_ )
      H5ATTRset_attribute_string(self.dataset_id, "VERSION", version)
      H5ATTRset_attribute_string(self.dataset_id, "TITLE", title)

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    self.type_id = get_native_type(self.disk_type_id)

    return (self.dataset_id, shape, atom)


  def _createCArray(self, char *title):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef char *complib, *version, *class_
    cdef int itemsize
    cdef ndarray dflts
    cdef void *fill_data
    cdef ndarray extdim
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
    # Get the fill values
    if isinstance(atom.dflt, numpy.ndarray) or atom.dflt:
      dflts = numpy.array(atom.dflt, dtype=atom.dtype)
      fill_data = dflts.data
    else:
      dflts = numpy.zeros((), dtype=atom.dtype)
      fill_data = NULL
    if atom.shape == ():
      # The default is preferred as a scalar value instead of 0-dim array
      atom.dflt = dflts[()]
    else:
      atom.dflt = dflts

    # Create the CArray/EArray
    self.dataset_id = H5ARRAYmake(
      self.parent_id, self.name, version, self.rank,
      self.dims, self.extdim, self.disk_type_id, self.dims_chunk,
      fill_data, self.filters.complevel, complib,
      self.filters.shuffle, self.filters.fletcher32, rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    if self._v_file.params['PYTABLES_SYS_ATTRS']:
      # Set the conforming array attributes
      H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_ )
      H5ATTRset_attribute_string(self.dataset_id, "VERSION", version)
      H5ATTRset_attribute_string(self.dataset_id, "TITLE", title)
      if self.extdim >= 0:
        extdim = <ndarray>numpy.array([self.extdim], dtype="int32")
        # Attach the EXTDIM attribute in case of enlargeable arrays
        H5ATTRset_attribute(self.dataset_id, "EXTDIM", H5T_NATIVE_INT,
                            0, NULL, extdim.data)

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    self.type_id = get_native_type(self.disk_type_id)

    return self.dataset_id


  def _openArray(self):
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef char byteorder[11]  # "irrelevant" fits easily here
    cdef int i
    cdef int extdim
    cdef herr_t ret
    cdef object shape, chunkshapes, atom
    cdef int fill_status
    cdef ndarray dflts
    cdef void *fill_data

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" % \
                         (self.name, self._v_parent._v_pathname))
    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()
    # Get the atom for this type
    atom = AtomFromHDF5Type(self.disk_type_id)

    # Get the rank for this array object
    if H5ARRAYget_ndims(self.dataset_id, &self.rank) < 0:
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

    # Get the fill value
    dflts = numpy.zeros((), dtype=atom.dtype)
    fill_data = dflts.data
    H5ARRAYget_fill_value(self.dataset_id, self.type_id,
                          &fill_status, fill_data);
    if fill_status == H5D_FILL_VALUE_UNDEFINED:
      # This can only happen with datasets created with other libraries
      # than PyTables.
      dflts = None
    if dflts is not None and atom.shape == ():
      # The default is preferred as a scalar value instead of 0-dim array
      atom.dflt = dflts[()]
    else:
      atom.dflt = dflts

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
    Py_BEGIN_ALLOW_THREADS
    ret = H5ARRAYreadSlice(self.dataset_id, self.type_id,
                           start, stop, step, rbuf)
    Py_END_ALLOW_THREADS
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


  def _g_readCoords(self, ndarray coords, ndarray nparr):
    """Read coordinates in an already created NumPy array."""
    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode

    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle
    size = nparr.size
    mem_space_id = H5Screate_simple(1, &size, NULL)

    # Select the dataspace to be read
    H5Sselect_elements(space_id, H5S_SELECT_SET,
                       <size_t>size, <hsize_t *>coords.data)

    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Do the actual read
    Py_BEGIN_ALLOW_THREADS
    ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                  H5P_DEFAULT, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)

    if self.atom.kind == 'time':
      # Swap the byteorder by hand (this is not currently supported by HDF5)
      if H5Tget_order(self.type_id) != platform_byteorder:
        nparr.byteswap(True)

    # Convert some HDF5 types to NumPy after reading
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 1)

    return


  def perform_selection(self, space_id, start, count, step, idx, mode):
    """Performs a selection using start/count/step in the given axis.

    All other axes have their full range selected.  The selection is
    added to the current `space_id` selection using the given mode.

    Note: This is a backport from the h5py project.
    """
    cdef int select_mode
    cdef ndarray start_, count_, step_
    cdef hsize_t *startp, *countp, *stepp

    # Build arrays for the selection parameters
    startl, countl, stepl = [], [], []
    for i, x in enumerate(self.shape):
      if i != idx:
        startl.append(0)
        countl.append(x)
        stepl.append(1)
      else:
        startl.append(start)
        countl.append(count)
        stepl.append(step)
    start_ = numpy.array(startl, dtype="i8")
    count_ = numpy.array(countl, dtype="i8")
    step_ = numpy.array(stepl, dtype="i8")

    # Get the pointers to array data
    startp = <hsize_t *>start_.data
    countp = <hsize_t *>count_.data
    stepp = <hsize_t *>step_.data

    # Do the actual selection
    select_modes = {"AND": H5S_SELECT_AND, "NOTB": H5S_SELECT_NOTB}
    assert mode in select_modes
    select_mode = select_modes[mode]
    H5Sselect_hyperslab(space_id, <H5S_seloper_t>select_mode,
                        startp, stepp, countp, NULL)


  def _g_readSelection(self, object selection, ndarray nparr):
    """Read a selection in an already created NumPy array."""
    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode

    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle
    size = nparr.size
    mem_space_id = H5Screate_simple(1, &size, NULL)

    # Select the dataspace to be read
    # Start by selecting everything
    H5Sselect_all(space_id)
    # Now refine with outstanding selections
    for args in selection:
      self.perform_selection(space_id, *args)

    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Do the actual read
    Py_BEGIN_ALLOW_THREADS
    ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                  H5P_DEFAULT, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems reading the array data.")

    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)

    if self.atom.kind == 'time':
      # Swap the byteorder by hand (this is not currently supported by HDF5)
      if H5Tget_order(self.type_id) != platform_byteorder:
        nparr.byteswap(True)

    # Convert some HDF5 types to NumPy after reading
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 1)

    return


  def _g_writeSlice(self, ndarray startl, ndarray stepl, ndarray countl,
                    ndarray nparr):
    """Write a slice in an already created NumPy array."""
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

    return


  def _g_writeCoords(self, ndarray coords, ndarray nparr):
    """Write a selection in an already created NumPy array."""
    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode

    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle
    size = nparr.size
    mem_space_id = H5Screate_simple(1, &size, NULL)

    # Select the dataspace to be written
    H5Sselect_elements(space_id, H5S_SELECT_SET,
                       <size_t>size, <hsize_t *>coords.data)

    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Convert some NumPy types to HDF5 before storing.
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 0)

    # Do the actual write
    Py_BEGIN_ALLOW_THREADS
    ret = H5Dwrite(self.dataset_id, self.type_id, mem_space_id, space_id,
                   H5P_DEFAULT, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems writing the array data.")

    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)

    return


  def _g_writeSelection(self, object selection, ndarray nparr):
    """Write a selection in an already created NumPy array."""
    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode

    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle
    size = nparr.size
    mem_space_id = H5Screate_simple(1, &size, NULL)

    # Select the dataspace to be written
    # Start by selecting everything
    H5Sselect_all(space_id)
    # Now refine with outstanding selections
    for args in selection:
      self.perform_selection(space_id, *args)

    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Convert some NumPy types to HDF5 before storing.
    if self.atom.type == 'time64':
      self._convertTime64(nparr, 0)

    # Do the actual write
    Py_BEGIN_ALLOW_THREADS
    ret = H5Dwrite(self.dataset_id, self.type_id, mem_space_id, space_id,
                   H5P_DEFAULT, rbuf)
    Py_END_ALLOW_THREADS
    if ret < 0:
      raise HDF5ExtError("Problems writing the array data.")

    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)

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
    self.dataset_id = H5VLARRAYmake(self.parent_id, self.name, version,
                                    rank, dims, self.base_type_id,
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

    if self._v_file.params['PYTABLES_SYS_ATTRS']:
      # Set the conforming array attributes
      H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_ )
      H5ATTRset_attribute_string(self.dataset_id, "VERSION", version)
      H5ATTRset_attribute_string(self.dataset_id, "TITLE", title)

    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()

    return self.dataset_id


  def _openArray(self):
    cdef char byteorder[11]  # "irrelevant" fits easily here
    cdef int i, enumtype
    cdef int rank
    cdef herr_t ret
    cdef hsize_t nrecords, chunksize
    cdef object shape, dtype, type_

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, self.name)
    if self.dataset_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" % \
                         (self.name, self._v_parent._v_pathname))
    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()
    # Get the atom for this type
    atom = AtomFromHDF5Type(self.disk_type_id)

    # Get info on dimensions & types (of base class)
    H5VLARRAYget_info(self.dataset_id, self.disk_type_id, &nrecords,
                      byteorder)
    # Get some properties of the atomic type
    self._atomicdtype = atom.dtype
    self._atomictype = atom.type
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
      shape = list(self._atomicshape)
      shape.insert(0, vllen)  # put the length at the beginning of the shape
      nparr = numpy.ndarray(
        buffer=buf, dtype=self._atomicdtype.base, shape=shape)
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
