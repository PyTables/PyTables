# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: September 21, 2002
# Author:  Francesc Alted - faltet@pytables.com
#
# $Id$
#
########################################################################

"""Cython interface between several PyTables classes and HDF5 library.

Classes (type extensions):

    Node
    Leaf
    Group
    Array
    VLArray
    UnImplemented

Functions:

Misc variables:

"""

import os
import warnings
from collections import namedtuple

ObjInfo = namedtuple('ObjInfo', ['addr', 'rc'])

from cpython cimport PY_MAJOR_VERSION
if PY_MAJOR_VERSION < 3:
    import cPickle as pickle
else:
    import pickle

import numpy

from .exceptions import HDF5ExtError, DataTypeWarning

from .utils import (check_file_access, byteorders, correct_byteorder,
  SizeType)

from .atom import Atom

from .description import descr_from_dtype

from .utilsextension import (atom_to_hdf5_type, atom_from_hdf5_type, platform_byteorder)

# Types, constants, functions, classes & other objects from everywhere
from libc.stdlib cimport malloc, free
from libc.string cimport strdup, strlen
from numpy cimport import_array, ndarray, npy_intp
from cpython.bytes cimport (PyBytes_AsString, PyBytes_FromStringAndSize,
    PyBytes_Check)
from cpython.unicode cimport PyUnicode_DecodeUTF8

from .definitions cimport (const_char, uintptr_t, hid_t, herr_t, hsize_t, hvl_t,
  H5S_seloper_t, H5D_FILL_VALUE_UNDEFINED,
  H5O_TYPE_UNKNOWN, H5O_TYPE_GROUP, H5O_TYPE_DATASET, H5O_TYPE_NAMED_DATATYPE,
  H5L_TYPE_ERROR, H5L_TYPE_HARD, H5L_TYPE_SOFT, H5L_TYPE_EXTERNAL,
  H5T_class_t, H5T_sign_t, H5T_NATIVE_INT,
  H5T_cset_t, H5T_CSET_ASCII, H5T_CSET_UTF8,
  H5F_SCOPE_GLOBAL, H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR,
  H5P_DEFAULT, H5P_FILE_ACCESS, H5P_FILE_CREATE,
  H5S_SELECT_SET, H5S_SELECT_AND, H5S_SELECT_NOTB,
  H5Fcreate, H5Fopen, H5Fclose, H5Fflush, H5Fget_vfd_handle, H5Fget_filesize,
  H5Fget_create_plist,
  H5Gcreate, H5Gopen, H5Gclose, H5Ldelete, H5Lmove,
  H5Dopen, H5Dclose, H5Dread, H5Dwrite, H5Dget_type,
  H5Dget_space, H5Dvlen_reclaim, H5Dget_storage_size, H5Dvlen_get_buf_size,
  H5Tclose, H5Tis_variable_str, H5Tget_sign,
  H5Adelete, H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT, H5T_STRING, H5Tget_order,
  H5Pcreate, H5Pset_cache, H5Pclose, H5Pget_userblock, H5Pset_userblock,
  H5Pset_fapl_sec2, H5Pset_fapl_log, H5Pset_fapl_stdio, H5Pset_fapl_core,
  H5Pset_fapl_split,
  H5Sselect_all, H5Sselect_elements, H5Sselect_hyperslab,
  H5Screate_simple, H5Sclose,
  H5Oget_info, H5O_info_t,
  H5ARRAYget_ndims, H5ARRAYget_info,
  set_cache_size, get_objinfo, get_linkinfo, Giterate, Aiterate, H5UIget_info,
  get_len_of_range, conv_float64_timeval32, truncate_dset,
  H5_HAVE_DIRECT_DRIVER, pt_H5Pset_fapl_direct,
  H5_HAVE_WINDOWS_DRIVER, pt_H5Pset_fapl_windows,
  H5_HAVE_IMAGE_FILE, pt_H5Pset_file_image, pt_H5Fget_file_image,
  H5Tget_size, hobj_ref_t)

from .utilsextension cimport malloc_dims, get_native_type, cstr_to_pystr, load_reference, getshape, npy_malloc_dims

from .attributes_ext cimport H5ATTRset_attribute, H5ATTRset_attribute_string, H5ATTRget_attribute, H5ATTRget_attribute_string, H5ATTRget_attribute_vlen_string_array, H5ATTRfind_attribute, H5ATTRget_type_ndims, H5ATTRget_dims

# FIXME duplicated
cdef hid_t H5T_CSET_DEFAULT = 16

#-------------------------------------------------------------------


# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h" nogil:

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
cdef extern from "H5VLARRAY.h" nogil:

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

# Helper function for quickly fetch an attribute string
cdef object get_attribute_string_or_none(hid_t node_id, char* attr_name):
  """Returns a string/unicode attribute if it exists in node_id.

  It returns ``None`` in case it don't exists (or there have been problems
  reading it).

  """

  cdef char *attr_value
  cdef int cset = H5T_CSET_DEFAULT
  cdef object retvalue
  cdef hsize_t size

  attr_value = NULL
  retvalue = None   # Default value
  if H5ATTRfind_attribute(node_id, attr_name):
    size = H5ATTRget_attribute_string(node_id, attr_name, &attr_value, &cset)
    if size == 0:
      if cset == H5T_CSET_UTF8:
        retvalue = numpy.unicode_(u'')
      else:
        retvalue = numpy.bytes_(b'')
    elif cset == H5T_CSET_UTF8:
      if size == 1 and attr_value[0] == 0:
        # compatibility with PyTables <= 3.1.1
        retvalue = numpy.unicode_(u'')
      retvalue = PyUnicode_DecodeUTF8(attr_value, size, NULL)
      retvalue = numpy.unicode_(retvalue)
    else:
      retvalue = PyBytes_FromStringAndSize(attr_value, size)
      # AV: oct 2012
      # since now we use the string size got form HDF5 we have to stip
      # trailing zeros used for padding.
      # The entire process is quite odd but due to a bug (??) in the way
      # numpy arrays are pickled in python 3 we can't assume that
      # strlen(attr_value) is the actual length of the attibute
      # and numpy.bytes_(attr_value) can give a truncated pickle string
      retvalue = retvalue.rstrip(b'\x00')
      retvalue = numpy.bytes_(retvalue)

    # Important to release attr_value, because it has been malloc'ed!
    if attr_value:
      free(<void *>attr_value)

  return retvalue


cdef class Node:
  # Instance variables declared in .pxd

  def _g_new(self, where, name, init):
    self.name = name
    # """The name of this node in its parent group."""
    self.parent_id = where._v_objectid
    # """The identifier of the parent group."""

  def _g_delete(self, parent):
    cdef int ret
    cdef bytes encoded_name

    encoded_name = self.name.encode('utf-8')

    # Delete this node
    ret = H5Ldelete(parent._v_objectid, encoded_name, H5P_DEFAULT)
    if ret < 0:
      raise HDF5ExtError("problems deleting the node ``%s``" % self.name)
    return ret

  def __dealloc__(self):
    self.parent_id = 0

  def _get_obj_info(self):
    cdef herr_t ret = 0
    cdef H5O_info_t oinfo

    ret = H5Oget_info(self._v_objectid, &oinfo)
    if ret < 0:
      raise HDF5ExtError("Unable to get object info for '%s'" %
                         self. _v_pathname)

    return ObjInfo(oinfo.addr, oinfo.rc)


cdef class Group(Node):
  cdef hid_t   group_id

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



cdef class Leaf(Node):
  # Instance variables declared in .pxd

  def _get_storage_size(self):
      return H5Dget_storage_size(self.dataset_id)

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
    return disk_type_id, native_type_id

  cdef _convert_time64(self, ndarray nparr, int sense):
    """Converts a NumPy of Time64 elements between NumPy and HDF5 formats.

    NumPy to HDF5 conversion is performed when 'sense' is 0.  Otherwise, HDF5
    to NumPy conversion is performed.  The conversion is done in place,
    i.e. 'nparr' is modified.

    """

    cdef void *t64buf
    cdef long byteoffset, bytestride, nelements
    cdef hsize_t nrecords

    byteoffset = 0   # NumPy objects doesn't have an offset
    if (<object>nparr).shape == ():
      # 0-dim array does contain *one* element
      nrecords = 1
      bytestride = 8
    else:
      nrecords = len(nparr)
      bytestride = nparr.strides[0]  # supports multi-dimensional recarray
    nelements = <size_t>nparr.size / nrecords
    t64buf = nparr.data

    conv_float64_timeval32(
      t64buf, byteoffset, bytestride, nrecords, nelements, sense)

  # can't do since cdef'd

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
      raise ValueError("Unexpected classname: %s" % classname)

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

  def _create_array(self, ndarray nparr, object title, object atom):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef bytes complib, version, class_
    cdef object dtype_, atom_, shape
    cdef ndarray dims
    cdef bytes encoded_title, encoded_name
    cdef H5T_cset_t cset = H5T_CSET_ASCII

    encoded_title = title.encode('utf-8')
    encoded_name = self.name.encode('utf-8')

    # Get the HDF5 type associated with this numpy type
    shape = (<object>nparr).shape
    if atom is None or atom.shape == ():
      dtype_ = nparr.dtype.base
      atom_ = Atom.from_dtype(dtype_)
    else:
      atom_ = atom
      shape = shape[:-len(atom_.shape)]
    self.disk_type_id = atom_to_hdf5_type(atom_, self.byteorder)

    # Allocate space for the dimension axis info and fill it
    dims = numpy.array(shape, dtype=numpy.intp)
    self.rank = len(shape)
    self.dims = npy_malloc_dims(self.rank, <npy_intp *>(dims.data))
    # Get the pointer to the buffer data area
    strides = (<object>nparr).strides
    # When the object is not a 0-d ndarray and its strides == 0, that
    # means that the array does not contain actual data
    if strides != () and sum(strides) == 0:
      rbuf = NULL
    else:
      rbuf = nparr.data
    # Save the array
    complib = (self.filters.complib or '').encode('utf-8')
    version = self._v_version.encode('utf-8')
    class_ = self._c_classid.encode('utf-8')
    self.dataset_id = H5ARRAYmake(self.parent_id, encoded_name, version,
                                  self.rank, self.dims,
                                  self.extdim, self.disk_type_id, NULL, NULL,
                                  self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    if self._v_file.params['PYTABLES_SYS_ATTRS']:
      if PY_MAJOR_VERSION > 2:
        cset = H5T_CSET_UTF8
      # Set the conforming array attributes
      H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_,
                                 len(class_), cset)
      H5ATTRset_attribute_string(self.dataset_id, "VERSION", version,
                                 len(version), cset)
      H5ATTRset_attribute_string(self.dataset_id, "TITLE", encoded_title,
                                 len(encoded_title), cset)

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    self.type_id = get_native_type(self.disk_type_id)

    return self.dataset_id, shape, atom_


  def _create_carray(self, object title):
    cdef int i
    cdef herr_t ret
    cdef void *rbuf
    cdef bytes complib, version, class_
    cdef ndarray dflts
    cdef void *fill_data
    cdef ndarray extdim
    cdef object atom
    cdef bytes encoded_title, encoded_name

    encoded_title = title.encode('utf-8')
    encoded_name = self.name.encode('utf-8')

    atom = self.atom
    self.disk_type_id = atom_to_hdf5_type(atom, self.byteorder)

    self.rank = len(self.shape)
    self.dims = malloc_dims(self.shape)
    if self.chunkshape:
      self.dims_chunk = malloc_dims(self.chunkshape)

    rbuf = NULL   # The data pointer. We don't have data to save initially
    # Encode strings
    complib = (self.filters.complib or '').encode('utf-8')
    version = self._v_version.encode('utf-8')
    class_ = self._c_classid.encode('utf-8')

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
      self.parent_id, encoded_name, version, self.rank,
      self.dims, self.extdim, self.disk_type_id, self.dims_chunk,
      fill_data, self.filters.complevel, complib,
      self.filters.shuffle, self.filters.fletcher32, rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    if self._v_file.params['PYTABLES_SYS_ATTRS']:
      # Set the conforming array attributes
      H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_,
                                 len(class_), H5T_CSET_ASCII)
      H5ATTRset_attribute_string(self.dataset_id, "VERSION", version,
                                 len(version), H5T_CSET_ASCII)
      H5ATTRset_attribute_string(self.dataset_id, "TITLE", encoded_title,
                                 len(encoded_title), H5T_CSET_ASCII)
      if self.extdim >= 0:
        extdim = <ndarray>numpy.array([self.extdim], dtype="int32")
        # Attach the EXTDIM attribute in case of enlargeable arrays
        H5ATTRset_attribute(self.dataset_id, "EXTDIM", H5T_NATIVE_INT,
                            0, NULL, extdim.data)

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    self.type_id = get_native_type(self.disk_type_id)

    return self.dataset_id


  def _open_array(self):
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef char cbyteorder[11]  # "irrelevant" fits easily here
    cdef int i
    cdef int extdim
    cdef herr_t ret
    cdef object shape, chunkshapes, atom
    cdef int fill_status
    cdef ndarray dflts
    cdef void *fill_data
    cdef bytes encoded_name
    cdef str byteorder

    encoded_name = self.name.encode('utf-8')

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, encoded_name, H5P_DEFAULT)
    if self.dataset_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" %
                         (self.name, self._v_parent._v_pathname))
    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()
    # Get the atom for this type
    atom = atom_from_hdf5_type(self.type_id)

    # Get the rank for this array object
    if H5ARRAYget_ndims(self.dataset_id, &self.rank) < 0:
      raise HDF5ExtError("Problems getting ndims!")
    # Allocate space for the dimension axis info
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    self.maxdims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Get info on dimensions, class and type (of base class)
    ret = H5ARRAYget_info(self.dataset_id, self.disk_type_id,
                          self.dims, self.maxdims,
                          &class_id, cbyteorder)
    if ret < 0:
      raise HDF5ExtError("Unable to get array info.")

    byteorder = cstr_to_pystr(cbyteorder)

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

    # object arrays should not be read directly into memory
    if atom.dtype != numpy.object:
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

    return self.dataset_id, atom, shape, chunkshapes


  def _append(self, ndarray nparr):
    cdef int ret, extdim
    cdef hsize_t *dims_arr
    cdef void *rbuf
    cdef object shape

    if self.atom.kind == "reference":
      raise ValueError("Cannot append to the reference types")

    # Allocate space for the dimension axis info
    dims_arr = npy_malloc_dims(self.rank, nparr.shape)
    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Convert some NumPy types to HDF5 before storing.
    if self.atom.type == 'time64':
      self._convert_time64(nparr, 0)

    # Append the records
    extdim = self.extdim
    with nogil:
        ret = H5ARRAYappend_records(self.dataset_id, self.type_id, self.rank,
                                    self.dims, dims_arr, extdim, rbuf)

    if ret < 0:
      raise HDF5ExtError("Problems appending the elements")

    free(dims_arr)
    # Update the new dimensionality
    shape = list(self.shape)
    shape[self.extdim] = SizeType(self.dims[self.extdim])
    self.shape = tuple(shape)

  def _read_array(self, hsize_t start, hsize_t stop, hsize_t step,
                 ndarray nparr):
    cdef herr_t ret
    cdef void *rbuf
    cdef hsize_t nrows
    cdef int extdim
    cdef size_t item_size = H5Tget_size(self.type_id)
    cdef void * refbuf = NULL

    # Number of rows to read
    nrows = get_len_of_range(start, stop, step)

    # Get the pointer to the buffer data area
    if self.atom.kind == "reference":
      refbuf = malloc(nrows * item_size)
      rbuf = refbuf
    else:
      rbuf = nparr.data

    if hasattr(self, "extdim"):
      extdim = self.extdim
    else:
      extdim = -1

    # Do the physical read
    with nogil:
        ret = H5ARRAYread(self.dataset_id, self.type_id, start, nrows, step,
                          extdim, rbuf)

    try:
      if ret < 0:
        raise HDF5ExtError("Problems reading the array data.")

      # Get the pointer to the buffer data area
      if self.atom.kind == "reference":
        load_reference(self.dataset_id, <hobj_ref_t *>rbuf, item_size, nparr)
    finally:
      if refbuf:
        free(refbuf)
        refbuf = NULL

    if self.atom.kind == 'time':
      # Swap the byteorder by hand (this is not currently supported by HDF5)
      if H5Tget_order(self.type_id) != platform_byteorder:
        nparr.byteswap(True)

    # Convert some HDF5 types to NumPy after reading.
    if self.atom.type == 'time64':
      self._convert_time64(nparr, 1)

    return


  def _g_read_slice(self, ndarray startl, ndarray stopl, ndarray stepl,
                   ndarray nparr):
    cdef herr_t ret
    cdef hsize_t *start
    cdef hsize_t *stop
    cdef hsize_t *step
    cdef void *rbuf
    cdef size_t item_size = H5Tget_size(self.type_id)
    cdef void * refbuf = NULL

    # Get the pointer to the buffer data area of startl, stopl and stepl arrays
    start = <hsize_t *>startl.data
    stop = <hsize_t *>stopl.data
    step = <hsize_t *>stepl.data

    # Get the pointer to the buffer data area
    if self.atom.kind == "reference":
      refbuf = malloc(nparr.size * item_size)
      rbuf = refbuf
    else:
      rbuf = nparr.data

    # Do the physical read
    with nogil:
        ret = H5ARRAYreadSlice(self.dataset_id, self.type_id,
                               start, stop, step, rbuf)
    try:
      if ret < 0:
        raise HDF5ExtError("Problems reading the array data.")

      # Get the pointer to the buffer data area
      if self.atom.kind == "reference":
        load_reference(self.dataset_id, <hobj_ref_t *>rbuf, item_size, nparr)
    finally:
      if refbuf:
        free(refbuf)
        refbuf = NULL

    if self.atom.kind == 'time':
      # Swap the byteorder by hand (this is not currently supported by HDF5)
      if H5Tget_order(self.type_id) != platform_byteorder:
        nparr.byteswap(True)

    # Convert some HDF5 types to NumPy after reading
    if self.atom.type == 'time64':
      self._convert_time64(nparr, 1)

    return


  def _g_read_coords(self, ndarray coords, ndarray nparr):
    """Read coordinates in an already created NumPy array."""

    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode
    cdef size_t item_size = H5Tget_size(self.type_id)
    cdef void * refbuf = NULL

    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)
    # Create a memory dataspace handle
    size = nparr.size
    mem_space_id = H5Screate_simple(1, &size, NULL)

    # Select the dataspace to be read
    H5Sselect_elements(space_id, H5S_SELECT_SET,
                       <size_t>size, <hsize_t *>coords.data)

    # Get the pointer to the buffer data area
    if self.atom.kind == "reference":
      refbuf = malloc(nparr.size * item_size)
      rbuf = refbuf
    else:
      rbuf = nparr.data

    # Do the actual read
    with nogil:
        ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                      H5P_DEFAULT, rbuf)

    try:
      if ret < 0:
        raise HDF5ExtError("Problems reading the array data.")

      # Get the pointer to the buffer data area
      if self.atom.kind == "reference":
        load_reference(self.dataset_id, <hobj_ref_t *>rbuf, item_size, nparr)
    finally:
      if refbuf:
        free(refbuf)
        refbuf = NULL

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
      self._convert_time64(nparr, 1)

    return


  def perform_selection(self, space_id, start, count, step, idx, mode):
    """Performs a selection using start/count/step in the given axis.

    All other axes have their full range selected.  The selection is
    added to the current `space_id` selection using the given mode.

    Note: This is a backport from the h5py project.

    """

    cdef int select_mode
    cdef ndarray start_, count_, step_
    cdef hsize_t *startp
    cdef hsize_t *countp
    cdef hsize_t *stepp

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

  def _g_read_selection(self, object selection, ndarray nparr):
    """Read a selection in an already created NumPy array."""

    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode
    cdef size_t item_size = H5Tget_size(self.type_id)
    cdef void * refbuf = NULL

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
    if self.atom.kind == "reference":
      refbuf = malloc(nparr.size * item_size)
      rbuf = refbuf
    else:
      rbuf = nparr.data

    # Do the actual read
    with nogil:
        ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                      H5P_DEFAULT, rbuf)

    try:
      if ret < 0:
        raise HDF5ExtError("Problems reading the array data.")

      # Get the pointer to the buffer data area
      if self.atom.kind == "reference":
        load_reference(self.dataset_id, <hobj_ref_t *>rbuf, item_size, nparr)
    finally:
      if refbuf:
        free(refbuf)
        refbuf = NULL

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
      self._convert_time64(nparr, 1)

    return


  def _g_write_slice(self, ndarray startl, ndarray stepl, ndarray countl,
                    ndarray nparr):
    """Write a slice in an already created NumPy array."""

    cdef int ret
    cdef void *rbuf
    cdef void *temp
    cdef hsize_t *start
    cdef hsize_t *step
    cdef hsize_t *count

    if self.atom.kind == "reference":
      raise ValueError("Cannot write reference types yet")
    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Get the start, step and count values
    start = <hsize_t *>startl.data
    step = <hsize_t *>stepl.data
    count = <hsize_t *>countl.data

    # Convert some NumPy types to HDF5 before storing.
    if self.atom.type == 'time64':
      self._convert_time64(nparr, 0)

    # Modify the elements:
    with nogil:
        ret = H5ARRAYwrite_records(self.dataset_id, self.type_id, self.rank,
                                   start, step, count, rbuf)

    if ret < 0:
      raise HDF5ExtError("Internal error modifying the elements "
                "(H5ARRAYwrite_records returned errorcode -%i)" % (-ret))

    return


  def _g_write_coords(self, ndarray coords, ndarray nparr):
    """Write a selection in an already created NumPy array."""

    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode

    if self.atom.kind == "reference":
      raise ValueError("Cannot write reference types yet")
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
      self._convert_time64(nparr, 0)

    # Do the actual write
    with nogil:
        ret = H5Dwrite(self.dataset_id, self.type_id, mem_space_id, space_id,
                       H5P_DEFAULT, rbuf)

    if ret < 0:
      raise HDF5ExtError("Problems writing the array data.")

    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)

    return


  def _g_write_selection(self, object selection, ndarray nparr):
    """Write a selection in an already created NumPy array."""

    cdef herr_t ret
    cdef hid_t space_id
    cdef hid_t mem_space_id
    cdef hsize_t size
    cdef void *rbuf
    cdef object mode

    if self.atom.kind == "reference":
      raise ValueError("Cannot write reference types yet")
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
      self._convert_time64(nparr, 0)

    # Do the actual write
    with nogil:
        ret = H5Dwrite(self.dataset_id, self.type_id, mem_space_id, space_id,
                       H5P_DEFAULT, rbuf)

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

  def _create_array(self, object title):
    cdef int rank
    cdef hsize_t *dims
    cdef herr_t ret
    cdef void *rbuf
    cdef bytes complib, version, class_
    cdef object type_, itemsize, atom, scatom
    cdef bytes encoded_title, encoded_name
    cdef H5T_cset_t cset = H5T_CSET_ASCII

    encoded_title = title.encode('utf-8')
    encoded_name = self.name.encode('utf-8')

    atom = self.atom
    if not hasattr(atom, 'size'):  # it is a pseudo-atom
      atom = atom.base

    # Get the HDF5 type of the *scalar* atom
    scatom = atom.copy(shape=())
    self.base_type_id = atom_to_hdf5_type(scatom, self.byteorder)

    # Allocate space for the dimension axis info
    rank = len(atom.shape)
    dims = malloc_dims(atom.shape)

    rbuf = NULL   # We don't have data to save initially

    # Encode strings
    complib = (self.filters.complib or '').encode('utf-8')
    version = self._v_version.encode('utf-8')
    class_ = self._c_classid.encode('utf-8')

    # Create the vlarray
    self.dataset_id = H5VLARRAYmake(self.parent_id, encoded_name, version,
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
      if PY_MAJOR_VERSION > 2:
        cset = H5T_CSET_UTF8
      # Set the conforming array attributes
      H5ATTRset_attribute_string(self.dataset_id, "CLASS", class_,
                                 len(class_), cset)
      H5ATTRset_attribute_string(self.dataset_id, "VERSION", version,
                                 len(version), cset)
      H5ATTRset_attribute_string(self.dataset_id, "TITLE", encoded_title,
                                 len(encoded_title), cset)

    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()

    return self.dataset_id


  def _open_array(self):
    cdef char cbyteorder[11]  # "irrelevant" fits easily here
    cdef int i, enumtype
    cdef int rank
    cdef herr_t ret
    cdef hsize_t nrecords, chunksize
    cdef object shape, type_
    cdef bytes encoded_name
    cdef str byteorder

    encoded_name = self.name.encode('utf-8')

    # Open the dataset
    self.dataset_id = H5Dopen(self.parent_id, encoded_name, H5P_DEFAULT)
    if self.dataset_id < 0:
      raise HDF5ExtError("Non-existing node ``%s`` under ``%s``" %
                         (self.name, self._v_parent._v_pathname))
    # Get the datatype handles
    self.disk_type_id, self.type_id = self._get_type_ids()
    # Get the atom for this type
    atom = atom_from_hdf5_type(self.type_id)

    # Get info on dimensions & types (of base class)
    H5VLARRAYget_info(self.dataset_id, self.disk_type_id, &nrecords,
                      cbyteorder)

    byteorder = cstr_to_pystr(cbyteorder)

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
        self._convert_time64(nparr, 0)
    else:
      rbuf = NULL

    # Append the records:
    with nogil:
        ret = H5VLARRAYappend_records(self.dataset_id, self.type_id,
                                      nobjects, self.nrecords, rbuf)

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
        self._convert_time64(nparr, 0)

    # Append the records:
    with nogil:
        ret = H5VLARRAYmodify_records(self.dataset_id, self.type_id,
                                      nrow, nobjects, rbuf)

    if ret < 0:
      raise HDF5ExtError("Problems modifying the record.")

    return nobjects

  # Because the size of each "row" is unknown, there is no easy way to
  # calculate this value
  def _get_memory_size(self):
    cdef hid_t space_id
    cdef hsize_t size
    cdef herr_t ret

    if self.nrows == 0:
      size = 0
    else:
      # Get the dataspace handle
      space_id = H5Dget_space(self.dataset_id)
      # Return the size of the entire dataset
      ret = H5Dvlen_get_buf_size(self.dataset_id, self.type_id, space_id,
                                 &size)
      if ret < 0:
        size = -1

      # Terminate access to the dataspace
      H5Sclose(space_id)

    return size

  def _read_array(self, hsize_t start, hsize_t stop, hsize_t step):
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
        "Asking for a range of rows exceeding the available ones!.",
        h5bt=False)

    # Now, read the chunk of rows
    with nogil:
        # Allocate the necessary memory for keeping the row handlers
        rdata = <hvl_t *>malloc(<size_t>nrows*sizeof(hvl_t))
        # Get the dataspace handle
        space_id = H5Dget_space(self.dataset_id)
        # Create a memory dataspace handle
        mem_space_id = H5Screate_simple(1, &nrows, NULL)
        # Select the data to be read
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start, &step, &nrows,
                            NULL)
        # Do the actual read
        ret = H5Dread(self.dataset_id, self.type_id, mem_space_id, space_id,
                      H5P_DEFAULT, rdata)

    if ret < 0:
      raise HDF5ExtError(
        "VLArray._read_array: Problems reading the array data.")

    datalist = []
    for i from 0 <= i < nrows:
      # Number of atoms in row
      vllen = rdata[i].len
      # Get the pointer to the buffer data area
      if vllen > 0:
        # Create a buffer to keep this info. It is important to do a
        # copy, because we will dispose the buffer memory later on by
        # calling the H5Dvlen_reclaim. PyBytes_FromStringAndSize does this.
        buf = PyBytes_FromStringAndSize(<char *>rdata[i].p,
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
        self._convert_time64(nparr, 1)
      # Append this array to the output list
      datalist.append(nparr)

    # Release resources
    # Reclaim all the (nested) VL data
    ret = H5Dvlen_reclaim(self.type_id, mem_space_id, H5P_DEFAULT, rdata)
    if ret < 0:
      raise HDF5ExtError("VLArray._read_array: error freeing the data buffer.")
    # Terminate access to the memory dataspace
    H5Sclose(mem_space_id)
    # Terminate access to the dataspace
    H5Sclose(space_id)
    # Free the amount of row pointers to VL row data
    free(rdata)

    return datalist


  def get_row_size(self, row):
    """Return the total size in bytes of all the elements contained in a given row."""

    cdef hid_t space_id
    cdef hsize_t size
    cdef herr_t ret

    cdef hsize_t offset[1]
    cdef hsize_t count[1]

    if row >= self.nrows:
      raise HDF5ExtError(
        "Asking for a range of rows exceeding the available ones!.",
        h5bt=False)

    # Get the dataspace handle
    space_id = H5Dget_space(self.dataset_id)

    offset[0] = row
    count[0] = 1

    ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    if ret < 0:
      size = -1

    ret = H5Dvlen_get_buf_size(self.dataset_id, self.type_id, space_id, &size)
    if ret < 0:
      size = -1

    # Terminate access to the dataspace
    H5Sclose(space_id)

    return size


cdef class UnImplemented(Leaf):

  def _open_unimplemented(self):
    cdef object shape
    cdef char cbyteorder[11]  # "irrelevant" fits easily here
    cdef bytes encoded_name
    cdef str byteorder

    encoded_name = self.name.encode('utf-8')

    # Get info on dimensions
    shape = H5UIget_info(self.parent_id, encoded_name, cbyteorder)
    shape = tuple(map(SizeType, shape))
    self.dataset_id = H5Dopen(self.parent_id, encoded_name, H5P_DEFAULT)
    byteorder = cstr_to_pystr(cbyteorder)

    return (shape, byteorder, self.dataset_id)

  def _g_close(self):
    H5Dclose(self.dataset_id)


## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
