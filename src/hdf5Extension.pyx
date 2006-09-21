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

from tables.exceptions import HDF5ExtError
from tables.enum import Enum
from tables.utils import checkFileAccess

from tables.utilsExtension import  \
     enumToHDF5, enumFromHDF5, getTypeEnum, \
     convertTime64, getLeafHDF5Type, isHDF5File, isPyTablesFile

from lrucacheExtension cimport NodeCache

# Types, constants, functions, classes & other objects from everywhere
from definitions cimport  \
     strdup, malloc, free, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, PyString_AsString, \
     PyString_FromStringAndSize, PyDict_Contains, PyDict_GetItem, \
     Py_INCREF, Py_DECREF, \
     import_array, ndarray, dtype, \
     time_t, size_t, hid_t, herr_t, hsize_t, hvl_t, \
     H5T_sign_t, H5T_class_t, \
     H5F_SCOPE_GLOBAL, H5F_ACC_TRUNC, H5F_ACC_RDONLY, H5F_ACC_RDWR, \
     H5P_DEFAULT, H5T_SGN_NONE, H5T_SGN_2, H5S_SELECT_SET, \
     H5get_libversion, H5check_version, H5Fcreate, H5Fopen, H5Fclose, \
     H5Fflush, H5Gcreate, H5Gopen, H5Gclose, H5Glink, H5Gunlink, H5Gmove, \
     H5Gmove2,  H5Dopen, H5Dclose, H5Dread, H5Dget_type, H5Dget_space, \
     H5Dvlen_reclaim, H5Adelete, H5Aget_num_attrs, H5Aget_name, H5Aopen_idx, \
     H5Aread, H5Aclose, H5Tclose, H5Tget_sign, H5Pcreate, H5Pclose, \
     H5Pset_cache, H5Pset_sieve_buf_size, H5Pset_fapl_log, \
     H5Sselect_hyperslab, H5Screate_simple, H5Sget_simple_extent_ndims, \
     H5Sget_simple_extent_dims, H5Sclose, \
     H5ATTRget_attribute_ndims, H5ATTRget_attribute_info, \
     H5ATTRset_attribute_string, H5ATTRset_attribute_string_CAarray, \
     H5ATTRset_attribute_numerical_NParray, H5ATTRget_attribute, \
     H5ATTRget_attribute_string, H5ATTRget_attribute_string_CAarray, \
     H5ATTR_find_attribute


# Include conversion tables
include "convtypetables.pxi"


__version__ = "$Revision$"


#-------------------------------------------------------------------

# Functions from HDF5 ARRAY (this is not part of HDF5 HL; it's private)
cdef extern from "H5ARRAY.h":

  herr_t H5ARRAYmake(hid_t loc_id, char *dset_name, char *class_,
                     char *title, char *flavor, char *obversion,
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

  herr_t H5ARRAYget_chunksize(hid_t dataset_id, int rank, hsize_t *dims_chunk)


# Functions for dealing with VLArray objects
cdef extern from "H5VLARRAY.h":

  herr_t H5VLARRAYmake( hid_t loc_id, char *dset_name, char *class_,
                        char *title, char *flavor, char *obversion,
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


# Function to convert HDF5 types from/to numpy types
cdef extern from "arraytypes.h":
  hid_t convArrayType(int fmt, size_t size, char *byteorder)
  size_t getArrayType(hid_t type_id, int *fmt)


# Helper routines
cdef extern from "utils.h":
  herr_t set_cache_size(hid_t file_id, size_t cache_size)
  object Giterate(hid_t parent_id, hid_t loc_id, char *name)
  object Aiterate(hid_t loc_id)
  object H5UIget_info(hid_t loc_id, char *name, char *byteorder)



#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#---------------------------------------------------------------------------



# Helper functions

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
  if H5ATTR_find_attribute(node_id, attr_name):
    ret = H5ATTRget_attribute_string(node_id, attr_name, &attr_value)
    if ret < 0: return None
    retvalue = attr_value
    # Important to release attr_value, because it has been malloc'ed!
    if attr_value: free(<void *>attr_value)
  return retvalue



# Type extensions declarations (these are subclassed by PyTables
# Python classes)

cdef class File:
  cdef hid_t   file_id
  cdef hid_t   access_plist
  cdef char    *name


  def __new__(self, char *name, char *mode, char *title,
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
  # XYX Comentat fins que no migrem a numpy (li cal l'extensio lrucache.pyx
  # que funciona amb numpy!
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
#         # The lines below doesn't work I don't know why!
#         #node = PyDict_GetItem(aliveNodes, nodePath)
#         #Py_INCREF(node)  # Because PyDict_GetItem returns a borrowed reference.
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
    #print "Destroying object File in Extension"
    if self.file_id:
      #print "Closing the HDF5 file", name," because user didn't do that!."
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


  # Get attributes
  def _g_getAttr(self, char *attrname):
    cdef hsize_t *dims, nelements
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef hid_t mem_type, dset_id
    cdef int rank
    cdef int ret, i
    cdef int enumtype
    cdef void *rbuf
    cdef hid_t type_id
    cdef H5T_sign_t sign #H5T_SGN_ERROR (-1), H5T_SGN_NONE (0), H5T_SGN_2 (1)
    cdef char *dsetname
    cdef ndarray ndvalue
    cdef object retvalue

    dset_id = self.dataset_id
    dsetname = self.name

    # Check if attribute exists
    if H5ATTR_find_attribute(dset_id, attrname) <= 0:
      # If the attribute does not exists, return None
      # and do not even warn the user
      return None

    ret = H5ATTRget_attribute_ndims(dset_id, attrname, &rank )
    if ret < 0:
      raise HDF5ExtError("Can't get ndims on attribute %s in node %s." %
                             (attrname, dsetname))

    # Allocate memory to collect the dimension of objects with dimensionality
    if rank > 0:
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
    else:
      dims = NULL

    ret = H5ATTRget_attribute_info(dset_id, attrname, dims,
                                   &class_id, &type_size, &type_id)
    if ret < 0:
      raise HDF5ExtError("Can't get info on attribute %s in node %s." %
                               (attrname, dsetname))
    # Get the attribute type
    type_size = getArrayType(type_id, &enumtype)
    # type_size is of type size_t (unsigned long), so cast it to a long first
    if <long>type_size < 0:
      # This class is not supported. Instead of raising a TypeError,
      # return a string explaining the problem. This will allow to continue
      # browsing native HDF5 files, while informing the user about the problem.
      #raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id
      H5Tclose(type_id)    # Release resources
      return "***Attribute error: HDF5 class %d not supported. Sorry!***" % \
             class_id
    sign = H5Tget_sign(type_id)
    H5Tclose(type_id)    # Release resources

    # Get the array shape
    shape = []
    for i from 0 <= i < rank:
      # The <int> cast avoids returning a Long integer
      shape.append(<int>dims[i])
    shape = tuple(shape)

    retvalue = None
    ndvalue = None
    dtype = NPCodeToType[enumtype]
    if class_id in (H5T_INTEGER, H5T_FLOAT):
      ndvalue = numpy.empty(dtype=dtype, shape=shape)
    elif class_id == H5T_STRING:
      dtype = numpy.dtype((dtype, type_size))
      ndvalue = numpy.empty(dtype=dtype, shape=shape)

    if <object>ndvalue is not None:
      # Get the pointer to the buffer data area
      rbuf = ndvalue.data

    if class_id == H5T_INTEGER:
      if sign == H5T_SGN_2:
        if type_size == 1:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_I8, rbuf)
        if type_size == 2:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_I16, rbuf)
        if type_size == 4:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_I32, rbuf)
        if type_size == 8:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_I64, rbuf)
      elif sign == H5T_SGN_NONE:
        if type_size == 1:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_U8, rbuf)
        if type_size == 2:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_U16, rbuf)
        if type_size == 4:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_U32, rbuf)
        if type_size == 8:
          ret = H5ATTRget_attribute(dset_id, attrname, H5T_STD_U64, rbuf)
      else:
        warnings.warn("""\
Type of attribute '%s' in node '%s' is not supported. Sorry about that!"""
                      % (attrname, dsetname))
        return None

    elif class_id == H5T_FLOAT:
      if type_size == 4:
        ret = H5ATTRget_attribute(dset_id, attrname, H5T_IEEE_F32, rbuf)
      if type_size == 8:
        ret = H5ATTRget_attribute(dset_id, attrname, H5T_IEEE_F64, rbuf)

    elif class_id == H5T_STRING:
      ret = H5ATTRget_attribute_string_CAarray(dset_id, attrname, <char *>rbuf)
      if rank == 0:
        # Scalar string attributes are returned as Python strings, while
        # multi-dimensional ones are returned as character arrays.
        retvalue = ndvalue.item()
    else:
      warnings.warn("""\
Type of attribute '%s' in node '%s' is not supported. Sorry about that!"""
                    % (attrname, dsetname))
      return None

    if dims:
      free(<void *> dims)

    # Check the return value of H5ATTRget_attribute_* call
    if ret < 0:
      raise HDF5ExtError("Attribute %s exists in node %s, but can't get it."\
                         % (attrname, dsetname))

    node = self._v_node

    # Check for multimensional attributes (if file.format_version > "1.4")
    if hasattr(node._v_file, "format_version"):
      format_version = node._v_file.format_version
    else:
      format_version = None


    if retvalue is not None:
      return retvalue
    else:
      return ndvalue


  def _g_setAttr(self, char *name, object value):
    cdef int ret
    cdef int valint
    cdef double valdouble
    cdef hid_t type_id
    cdef size_t rank
    cdef hsize_t *dims
    cdef void *data
    cdef int i
    cdef int itemsize
    cdef ndarray ndv
    cdef dtype ndt

    node = self._v_node

    ret = 0
    # Append this attribute on disk
    if isinstance(value, str):
      ret = H5ATTRset_attribute_string(self.dataset_id, name, value)
    elif isinstance(value, numpy.ndarray):
      ndv = <ndarray>value
      data = ndv.data
      ndt = <dtype>value.dtype
      rank = ndv.nd
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      for i from 0 <= i < rank:
        dims[i] = ndv.dimensions[i]
      if value.dtype.char == "S":
        itemsize = value.itemsize
        ret = H5ATTRset_attribute_string_CAarray(
          self.dataset_id, name, rank, dims, itemsize, data)
      elif ndt.type_num in NPCodeToHDF5.keys():
        type_id = NPCodeToHDF5[ndt.type_num]
        ret = H5ATTRset_attribute_numerical_NParray(self.dataset_id, name,
                                                    rank, dims, type_id, data)
      else:   # One should add complex support for numpy arrays
        pickledvalue = cPickle.dumps(value, 0)
        self._g_setAttrStr(name, pickledvalue)
      free(<void *>dims)
    else:
      # Convert this object to a null-terminated string
      # (binary pickles are not supported at this moment)
      pickledvalue = cPickle.dumps(value, 0)
      self._g_setAttrStr(name, pickledvalue)

    if ret < 0:
      raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                         (name, node))


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
    cdef char *byteorder
    cdef char *flavor, *complib, *version, *class_
    cdef object type_

    type_ = nparr.dtype.type
    try:
      enumtype = NPTypeToCode[type_]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported right now. Sorry about that.
            """ % repr(type_)

    # Get the HDF5 type associated with this numpy type
    itemsize = nparr.itemsize
    byteorder = PyString_AsString(self.byteorder)
    self.type_id = convArrayType(enumtype, itemsize, byteorder)
    if self.type_id < 0:
      raise TypeError, \
        """type '%s' is not supported right now. Sorry about that.""" \
        % type_

    # Get the pointer to the buffer data area
    rbuf = nparr.data
    # Allocate space for the dimension axis info and fill it
    self.rank = nparr.nd
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    for i from  0 <= i < self.rank:
      self.dims[i] = nparr.dimensions[i]
    # Save the array
    flavor = PyString_AsString(self.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    self.dataset_id = H5ARRAYmake(self.parent_id, self.name, class_, title,
                                  flavor, version, self.rank, self.dims,
                                  self.extdim, self.type_id, NULL, NULL,
                                  self.filters.complevel, complib,
                                  self.filters.shuffle,
                                  self.filters.fletcher32,
                                  rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    stype = numpy.typeNA[type_]
    return (self.dataset_id, type_, stype)


  def _createEArray(self, char *title):
    cdef int i, enumtype
    cdef herr_t ret
    cdef void *rbuf
    cdef char *byteorder
    cdef char *flavor, *complib, *version, *class_
    cdef void *fill_value
    cdef int itemsize

    atom = self.atom
    itemsize = atom.itemsize
    try:
      # Since Time columns have no NumPy type of their own,
      # a special case is made for them.
      stype = atom.stype
      if stype == 'Time32':
        enumtype = ord('t')
      elif stype == 'Time64':
        enumtype = ord('T')
      else:
        enumtype = NPTypeToCode[self.type]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported right now. Sorry about that.
            """ % repr(self.type)

    if stype == 'Enum':
      self.type_id = enumToHDF5(atom, self.byteorder)
    else:
      byteorder = PyString_AsString(self.byteorder)
      self.type_id = convArrayType(enumtype, itemsize, byteorder)
      if self.type_id < 0:
        raise TypeError, \
          """type '%s' is not supported right now. Sorry about that.""" \
      % self.type

    self.rank = len(self.shape)
    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if self._v_chunksize:
      self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      self.dims[i] = self.shape[i]
      if self._v_chunksize:
        self.dims_chunk[i] = self._v_chunksize[i]

    rbuf = NULL   # The data pointer. We don't have data to save initially
    # Manually convert some string values that can't be done automatically
    flavor = PyString_AsString(atom.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Setup the fill values
    fill_value = <void *>malloc(<size_t> itemsize)
    for i from  0 <= i < itemsize:
      (<char *>fill_value)[i] = 0

    # Create the EArray
    self.dataset_id = H5ARRAYmake(
      self.parent_id, self.name, class_, title, flavor, version,
      self.rank, self.dims, self.extdim, self.type_id, self.dims_chunk,
      fill_value, self.filters.complevel, complib,
      self.filters.shuffle, self.filters.fletcher32, rbuf)
    if self.dataset_id < 0:
      raise HDF5ExtError("Problems creating the %s." % self.__class__.__name__)

    # Release resources
    free(fill_value)

    return self.dataset_id


  def _loadEnum(self):
    """_loadEnum() -> (Enum, npType)
    Load enumerated type associated with this array.

    This method loads the HDF5 enumerated type associated with this
    array.  It returns an `Enum` instance built from that, and the
    NumPy type used to encode it.
    """

    cdef hid_t enumId

    enumId = getTypeEnum(self.type_id)

    # Get the Enum and NumPy types and close the HDF5 type.
    try:
      return enumFromHDF5(enumId)
    finally:
      # (Yes, the ``finally`` clause *is* executed.)
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")


  def _openArray(self):
    cdef size_t type_size, type_precision
    cdef H5T_class_t class_id
    cdef H5T_sign_t sign
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i, enumtype
    cdef int extdim
    cdef char *flavor
    cdef hid_t base_type_id
    cdef herr_t ret
    cdef object shape, type_

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

    flavor = "numpy"   # Default value
    if self._v_file._isPTFile:
      H5ATTRget_attribute_string(self.dataset_id, "FLAVOR", &flavor)
    self.flavor = flavor  # Gives class visibility to flavor

    # Allocate space for the dimension chunking info
    self.dims_chunk = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    if ((H5ARRAYget_chunksize(self.dataset_id, self.rank,
                              self.dims_chunk)) < 0):
      if self.extdim >= 0 or self.__class__.__name__ == 'CArray':
        raise HDF5ExtError, "Problems getting the chunksizes!"
    # Get the array type
    type_size = getArrayType(base_type_id, &enumtype)
    if type_size < 0:
      raise TypeError, "HDF5 class %d not supported. Sorry!" % class_id

    H5Tclose(base_type_id)    # Release resources

    # We had problems when creating Tuples directly with Pyrex!.
    # A bug report has been sent to Greg Ewing and here is his answer:
    """
    It's impossible to call PyTuple_SetItem and PyTuple_GetItem
    correctly from Pyrex, because they don't follow the standard
    reference counting protocol (PyTuple_GetItem returns a borrowed
    reference, and PyTuple_SetItem steals a reference).

    It's best to use Python constructs to create tuples if you
    can. Otherwise, you could create wrapppers for these functions in
    an external C file which provide standard reference counting
    behaviour.
    """
    # So, I've decided to create the shape tuple using Python constructs
    shape = []
    chunksizes = []
    for i from 0 <= i < self.rank:
      shape.append(self.dims[i])
      if self.dims_chunk:
        chunksizes.append(<int>self.dims_chunk[i])
    shape = tuple(shape)
    chunksizes = tuple(chunksizes)

    type_ = NPCodeToType.get(enumtype, None)
    return (self.dataset_id, type_, NPCodeToPTType[enumtype],
            shape, type_size, byteorder, chunksizes)


  def _convertTypes(self, object nparr, int sense):
    """Converts Time64 elements in 'nparr' between NumPy and HDF5 formats.

    NumPy to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to NumPy conversion is performed.
    The conversion is done in place, i.e. 'nparr' is modified.
    """

    # This should be generalised to support other type conversions.
    if self.stype == 'Time64':
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
    self.nrows = self.dims[self.extdim]


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
    cdef hsize_t extdim
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
    self.nrows = size


  def _readArray(self, hsize_t start, hsize_t stop, hsize_t step,
                 ndarray nparr):
    cdef herr_t ret
    cdef void *rbuf
    cdef hsize_t nrows
    cdef int extdim

    # Get the pointer to the buffer data area
    rbuf = nparr.data

    # Number of rows to read
    nrows = ((stop - start - 1) / step) + 1  # (stop-start)/step  do not work
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
    cdef char *byteorder
    cdef char *flavor, *complib, *version, *class_

    atom = self.atom
    type_ = atom.type
    stype = atom.stype
    try:
      # Since Time columns have no NumPy type of their own,
      # a special case is made for them.
      if stype == 'Time32':
        enumtype = ord('t')
      elif stype == 'Time64':
        enumtype = ord('T')
      else:
        enumtype = NPTypeToCode[type_]
    except KeyError:
      raise TypeError, \
            """Type class '%s' not supported right now. Sorry about that.
            """ % repr(type_)

    if stype == 'Enum':
      self.base_type_id = enumToHDF5(atom, self.byteorder)
    else:
      byteorder = PyString_AsString(self.byteorder)
      # Get the HDF5 type id
      self.base_type_id = convArrayType(enumtype, atom.itemsize, byteorder)
      if self.base_type_id < 0:
        raise TypeError, \
          """type '%s' is not supported right now. Sorry about that.""" \
      % type_

    # Allocate space for the dimension axis info
    if isinstance(atom.shape, int):
      self.rank = 1
      self.scalar = 1
    else:
      self.rank = len(atom.shape)
      self.scalar = 0

    self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    # Fill the dimension axis info with adequate info
    for i from  0 <= i < self.rank:
      if isinstance(atom.shape, int):
        self.dims[i] = atom.shape
      else:
        self.dims[i] = atom.shape[i]

    rbuf = NULL   # We don't have data to save initially

    # Manually convert some string values that can't be done automatically
    flavor = PyString_AsString(atom.flavor)
    complib = PyString_AsString(self.filters.complib)
    version = PyString_AsString(self._v_version)
    class_ = PyString_AsString(self._c_classId)
    # Create the vlarray
    self.dataset_id = H5VLARRAYmake(self.parent_id, self.name, class_, title,
                                    flavor, version, self.rank, self.scalar,
                                    self.dims, self.base_type_id,
                                    self._v_chunksize, rbuf,
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


  def _loadEnum(self):
    """_loadEnum() -> (Enum, npType)
    Load enumerated type associated with this array.

    This method loads the HDF5 enumerated type associated with this
    array.  It returns an `Enum` instance built from that, and the
    NumPy type used to encode it.
    """

    cdef hid_t typeId, rowTypeId, enumId

    # Get the enumerated type.
    enumId = getTypeEnum(self.base_type_id)

    # Get the Enum and NumPy types and close the HDF5 type.
    try:
      return enumFromHDF5(enumId)
    finally:
      # (Yes, the ``finally`` clause *is* executed.)
      if H5Tclose(enumId) < 0:
        raise HDF5ExtError("failed to close HDF5 enumerated type")


  def _openArray(self):
    cdef object shape
    cdef char byteorder[16]  # "non-relevant" fits easily here
    cdef int i, enumtype
    cdef herr_t ret
    cdef hsize_t nrecords
    cdef char *flavor

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
    if self.rank:
      self.dims = <hsize_t *>malloc(self.rank * sizeof(hsize_t))
    else:
      self.dims = NULL;
    # Get info on dimensions, class and type (of base class)
    H5VLARRAYget_info(self.dataset_id, self.type_id, &nrecords,
                      self.dims, &self.base_type_id, byteorder)
    flavor = "numpy"  # Default value
    if self._v_file._isPTFile:
      H5ATTRget_attribute_string(self.dataset_id, "FLAVOR", &flavor)
    self.flavor = flavor  # Gives class visibility to flavor
    self.byteorder = byteorder  # Gives class visibility to byteorder

    # Get the array type
    self._basesize = getArrayType(self.base_type_id, &enumtype)
    if self._basesize < 0:
      raise TypeError, "The HDF5 class of object does not seem VLEN. Sorry!"

    # Get the type of the atomic type
    self._atomictype = NPCodeToType.get(enumtype, None)
    self._atomicstype = NPCodeToPTType[enumtype]
    # Get the size and shape of the atomic type
    self._atomicsize = self._basesize
    if self.rank:
      shape = []
      for i from 0 <= i < self.rank:
        shape.append(self.dims[i])
        self._atomicsize = self._atomicsize * self.dims[i]
      shape = tuple(shape)
    else:
      # rank zero means a scalar
      shape = 1

    self._atomicshape = shape
    self.nrecords = nrecords  # Initialize the number of records saved
    return (self.dataset_id, nrecords)


  def _convertTypes(self, object nparr, int sense):
    """Converts Time64 elements in 'nparr' between NumPy and HDF5 formats.

    NumPy to HDF5 conversion is performed when 'sense' is 0.
    Otherwise, HDF5 to NumPy conversion is performed.
    The conversion is done in place, i.e. 'nparr' is modified.
    """

    # This should be generalised to support other type conversions.
    if self._atomicstype == 'Time64':
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
    nrows = ((stop - start - 1) / step) + 1  # (stop-start)/step  do not work
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
      if (isinstance(self._atomicshape, tuple)):
        shape = list(self._atomicshape)
        shape.insert(0, vllen)  # put the length at the beginning of the shape
      elif self._atomicshape > 1:
        shape = (vllen, self._atomicshape)
      else:
        # Case of scalars (self._atomicshape == 1)
        shape = (vllen,)
      if str(self._atomictype) == "CharType":
        dtype = numpy.dtype((numpy.string_, self._basesize))
      else:
        dtype = numpy.dtype(self._atomictype)
        # Set the same byteorder than on-disk
        dtype = dtype.newbyteorder(self.byteorder)
      nparr = numpy.array(buf, type=dtype, shape=shape)
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
    cdef char byteorder[16]  # "non-relevant" fits easily here

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
