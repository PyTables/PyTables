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

from .utils import SizeType

from .atom import Atom

# Types, constants, functions, classes & other objects from everywhere
from libc.stdlib cimport free
from numpy cimport import_array, ndarray, npy_intp
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_DecodeUTF8

from .definitions cimport (const_char, uintptr_t, hid_t, herr_t, hsize_t, hvl_t, H5T_CSET_UTF8, H5F_SCOPE_GLOBAL, H5P_DEFAULT, H5Fflush, H5Ldelete, H5Dopen, H5Dclose, H5Dget_type, H5Dget_storage_size, H5Tclose, H5Oget_info, H5O_info_t, H5UIget_info, conv_float64_timeval32, truncate_dset)

from .utilsextension cimport get_native_type, cstr_to_pystr

from .attributes_ext cimport H5ATTRget_attribute_string, H5ATTRfind_attribute

# FIXME duplicated
cdef hid_t H5T_CSET_DEFAULT = 16

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
