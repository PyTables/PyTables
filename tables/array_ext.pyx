import numpy as np

from .atom import Atom
from .exceptions import HDF5ExtError
from .utils import correct_byteorder, SizeType
from .utilsextension import atom_from_hdf5_type, atom_to_hdf5_type, platform_byteorder

from cpython cimport PY_MAJOR_VERSION
from numpy cimport ndarray, npy_intp

from libc.stdlib cimport malloc, free

from .hdf5extension cimport Leaf

from .attributes_ext cimport H5ATTRset_attribute, H5ATTRset_attribute_string
from .definitions cimport hid_t, herr_t, hsize_t, hobj_ref_t
from .definitions cimport H5P_DEFAULT
from .definitions cimport H5T_CSET_ASCII, H5T_CSET_UTF8, H5T_NATIVE_INT, H5T_cset_t, H5T_class_t, H5Tget_size, H5Tget_order
from .definitions cimport H5Dopen, H5Dget_space, H5Dread, H5Dwrite, H5D_FILL_VALUE_UNDEFINED
from .definitions cimport H5Screate_simple, H5Sselect_elements, H5Sselect_hyperslab, H5Sselect_all, H5Sclose, H5S_SELECT_SET, H5S_SELECT_AND, H5S_SELECT_NOTB, H5S_seloper_t
from .definitions cimport H5ARRAYget_ndims, H5ARRAYget_info
from .definitions cimport get_len_of_range

from .utilsextension cimport malloc_dims, get_native_type, npy_malloc_dims, cstr_to_pystr, load_reference, getshape

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
    dims = np.array(shape, dtype=np.intp)
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
                                  self.filters.shuffle_bitshuffle,
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
    if isinstance(atom.dflt, np.ndarray) or atom.dflt:
      dflts = np.array(atom.dflt, dtype=atom.dtype)
      fill_data = dflts.data
    else:
      dflts = np.zeros((), dtype=atom.dtype)
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
      self.filters.shuffle_bitshuffle, self.filters.fletcher32, rbuf)
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
        extdim = <ndarray>np.array([self.extdim], dtype="int32")
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
    if atom.dtype != np.object:
      # Get the fill value
      dflts = np.zeros((), dtype=atom.dtype)
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
    start_ = np.array(startl, dtype="i8")
    count_ = np.array(countl, dtype="i8")
    step_ = np.array(stepl, dtype="i8")

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


