import pickle
import warnings
import numpy as np

from .atom import Atom
from .description import descr_from_dtype
from .exceptions import HDF5ExtError, DataTypeWarning
from .utils import byteorders
from .utilsextension import atom_to_hdf5_type, hdf5_to_np_ext_type, create_nested_type

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_DecodeUTF8
from libc.stdlib cimport malloc, free
from libc.string cimport strlen

from numpy cimport ndarray, npy_intp

from .definitions cimport hid_t, hsize_t, herr_t
from .definitions cimport H5Adelete
from .definitions cimport H5T_ARRAY, H5T_CSET_UTF8, H5T_STRING, H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT, H5Tget_sign, H5Tclose, H5T_class_t, H5T_sign_t, H5Tis_variable_str
from .definitions cimport Aiterate
from .utilsextension cimport getshape, npy_malloc_dims, get_native_type

from .attributes_ext cimport H5ATTRget_attribute, H5ATTRget_type_ndims, H5ATTRget_attribute_string, H5ATTRget_dims, H5ATTRget_attribute_vlen_string_array, H5ATTRfind_attribute,H5ATTRset_attribute, H5ATTRset_attribute_string

# FIXME duplicated
cdef hid_t H5T_CSET_DEFAULT = 16

# Get the numpy dtype scalar attribute from an HDF5 type as fast as possible
cdef object get_dtype_scalar(hid_t type_id, H5T_class_t class_id,
                             size_t itemsize):
  cdef H5T_sign_t sign
  cdef object stype

  if class_id == H5T_BITFIELD:
    stype = "b1"
  elif class_id == H5T_INTEGER:
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
    ntype = np.dtype(stype)
  except TypeError:
    ntype = None
  return ntype


cdef class AttributeSet:
  cdef object name

  def _g_new(self, node):
    self.name = node._v_name

  def _g_list_attr(self, node):
    "Return a tuple with the attribute list"
    a = Aiterate(node._v_objectid)
    return a


  def _g_setattr(self, node, name, object value):
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
    cdef char* cname = NULL
    cdef bytes encoded_name
    cdef int cset = H5T_CSET_DEFAULT

    encoded_name = name.encode('utf-8')
    # get the C pointer
    cname = encoded_name

    # The dataset id of the node
    dset_id = node._v_objectid

    # Convert a NumPy scalar into a NumPy 0-dim ndarray
    if isinstance(value, np.generic):
      value = np.array(value)

    # Check if value is a NumPy ndarray and of a supported type
    if (isinstance(value, np.ndarray) and
        value.dtype.kind in ('V', 'S', 'b', 'i', 'u', 'f', 'c')):
      # get a contiguous array: fixes #270 and gh-176
      #value = np.ascontiguousarray(value)
      value = value.copy()
      if value.dtype.kind == 'V':
        description, rabyteorder = descr_from_dtype(value.dtype)
        byteorder = byteorders[rabyteorder]
        type_id = create_nested_type(description, byteorder)
      else:
        # Get the associated native HDF5 type of the scalar type
        baseatom = Atom.from_dtype(value.dtype.base)
        byteorder = byteorders[value.dtype.byteorder]
        type_id = atom_to_hdf5_type(baseatom, byteorder)
      # Get dimensionality info
      ndv = <ndarray>value
      dims = npy_malloc_dims(ndv.ndim, ndv.shape)
      # Actually write the attribute
      ret = H5ATTRset_attribute(dset_id, cname, type_id,
                                ndv.ndim, dims, ndv.data)
      if ret < 0:
        raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                           (name, self._v_node))
      # Release resources
      free(<void *>dims)
      H5Tclose(type_id)
    else:
      # Object cannot be natively represented in HDF5.
      if (isinstance(value, np.ndarray) and
          value.dtype.kind == 'U' and
          value.shape == ()):
        value = value[()].encode('utf-8')
        cset = H5T_CSET_UTF8
      else:
        # Convert this object to a null-terminated string
        # (binary pickles are not supported at this moment)
        value = pickle.dumps(value, 0)

      ret = H5ATTRset_attribute_string(dset_id, cname, value, len(value), cset)
      if ret < 0:
        raise HDF5ExtError("Can't set attribute '%s' in node:\n %s." %
                           (name, self._v_node))


  # Get attributes
  def _g_getattr(self, node, attrname):
    """Get HDF5 attributes and retrieve them as NumPy objects.

    H5T_SCALAR types will be retrieved as scalar NumPy.
    H5T_ARRAY types will be retrieved as ndarray NumPy objects.

    """

    cdef hsize_t *dims
    cdef H5T_class_t class_id
    cdef size_t type_size
    cdef hid_t mem_type, dset_id, type_id, native_type
    cdef int rank, ret, enumtype
    cdef void *rbuf
    cdef char *str_value
    cdef char **str_values = NULL
    cdef ndarray ndvalue
    cdef object shape, stype_atom, shape_atom, retvalue
    cdef int i, nelements
    cdef char* cattrname = NULL
    cdef bytes encoded_attrname
    cdef int cset = H5T_CSET_DEFAULT

    encoded_attrname = attrname.encode('utf-8')
    # Get the C pointer
    cattrname = encoded_attrname

    # The dataset id of the node
    dset_id = node._v_objectid
    dims = NULL

    ret = H5ATTRget_type_ndims(dset_id, cattrname, &type_id, &class_id,
                               &type_size, &rank )
    if ret < 0:
      raise HDF5ExtError("Can't get type info on attribute %s in node %s." %
                         (attrname, self.name))

    # Call a fast function for scalar values and typical class types
    if (rank == 0 and class_id == H5T_STRING):
      type_size = H5ATTRget_attribute_string(dset_id, cattrname, &str_value,
                                             &cset)
      if type_size == 0:
        if cset == H5T_CSET_UTF8:
          retvalue = np.unicode_(u'')
        else:
          retvalue = np.bytes_(b'')

      elif cset == H5T_CSET_UTF8:
        if type_size == 1 and str_value[0] == 0:
          # compatibility with PyTables <= 3.1.1
          retvalue = np.unicode_(u'')
        retvalue = PyUnicode_DecodeUTF8(str_value, type_size, NULL)
        retvalue = np.unicode_(retvalue)
      else:
        retvalue = PyBytes_FromStringAndSize(str_value, type_size)
        # AV: oct 2012
        # since now we use the string size got form HDF5 we have to strip
        # trailing zeros used for padding.
        # The entire process is quite odd but due to a bug (??) in the way
        # numpy arrays are pickled in python 3 we can't assume that
        # strlen(attr_value) is the actual length of the attibute
        # and np.bytes_(attr_value) can give a truncated pickle sting
        retvalue = retvalue.rstrip(b'\x00')
        retvalue = np.bytes_(retvalue)     # bytes
      # Important to release attr_value, because it has been malloc'ed!
      if str_value:
        free(str_value)
      H5Tclose(type_id)
      return retvalue
    elif (rank == 0 and class_id in (H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT)):
      dtype_ = get_dtype_scalar(type_id, class_id, type_size)
      if dtype_ is None:
        warnings.warn("Unsupported type for attribute '%s' in node '%s'. "
                      "Offending HDF5 class: %d" % (attrname, self.name,
                                                    class_id), DataTypeWarning)
        self._v_unimplemented.append(attrname)
        return None
      shape = ()
    else:
      # General case

      # Get the dimensional info
      dims = <hsize_t *>malloc(rank * sizeof(hsize_t))
      ret = H5ATTRget_dims(dset_id, cattrname, dims)
      if ret < 0:
        raise HDF5ExtError("Can't get dims info on attribute %s in node %s." %
                           (attrname, self.name))
      shape = getshape(rank, dims)
      # dims is not needed anymore
      free(<void *> dims)

      # Get the NumPy dtype from the type_id
      try:
        stype_, shape_ = hdf5_to_np_ext_type(type_id, pure_numpy_types=True)
        dtype_ = np.dtype(stype_, shape_)
      except TypeError:
        if class_id == H5T_STRING and H5Tis_variable_str(type_id):
          nelements = H5ATTRget_attribute_vlen_string_array(dset_id, cattrname,
                                                            &str_values, &cset)
          if nelements < 0:
            raise HDF5ExtError("Can't read attribute %s in node %s." %
                               (attrname, self.name))

          # The following generator expressions do not work with Cython 0.15.1
          if cset == H5T_CSET_UTF8:
            #retvalue = np.fromiter(
            #  PyUnicode_DecodeUTF8(<char*>str_values[i],
            #                        strlen(<char*>str_values[i]),
            #                        NULL)
            #    for i in range(nelements), "O8")
            retvalue = np.array([
              PyUnicode_DecodeUTF8(<char*>str_values[i],
                                    strlen(<char*>str_values[i]),
                                    NULL)
                for i in range(nelements)], "O8")

          else:
            #retvalue = np.fromiter(
            #  <char*>str_values[i] for i in range(nelements), "O8")
            retvalue = np.array(
              [<char*>str_values[i] for i in range(nelements)], "O8")
          retvalue.shape = shape

          # Important to release attr_value, because it has been malloc'ed!
          for i in range(nelements):
            free(str_values[i]);
          free(str_values)

          return retvalue

        # This class is not supported. Instead of raising a TypeError, issue a
        # warning explaining the problem. This will allow to continue browsing
        # native HDF5 files, while informing the user about the problem.
        warnings.warn("Unsupported type for attribute '%s' in node '%s'. "
                      "Offending HDF5 class: %d" % (attrname, self.name,
                                                    class_id), DataTypeWarning)
        self._v_unimplemented.append(attrname)
        return None

    # Get the native type (so that it is HDF5 who is the responsible to deal
    # with non-native byteorders on-disk)
    native_type_id = get_native_type(type_id)

    # Get the container for data
    ndvalue = np.empty(dtype=dtype_, shape=shape)
    # Get the pointer to the buffer data area
    rbuf = ndvalue.data
    # Actually read the attribute from disk
    ret = H5ATTRget_attribute(dset_id, cattrname, native_type_id, rbuf)
    if ret < 0:
      raise HDF5ExtError("Attribute %s exists in node %s, but can't get it." %
                         (attrname, self.name))
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
    cdef char *cattrname = NULL
    cdef bytes encoded_attrname

    encoded_attrname = attrname.encode('utf-8')
    # Get the C pointer
    cattrname = encoded_attrname

    # The dataset id of the node
    dset_id = node._v_objectid

    ret = H5Adelete(dset_id, cattrname)
    if ret < 0:
      raise HDF5ExtError("Attribute '%s' exists in node '%s', but cannot be "
                         "deleted." % (attrname, self.name))


