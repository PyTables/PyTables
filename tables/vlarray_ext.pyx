from .exceptions import HDF5ExtError
from .utils import correct_byteorder, SizeType
from .utilsextension import atom_from_hdf5_type, atom_to_hdf5_type, platform_byteorder

cimport numpy as np

from cpython cimport PY_MAJOR_VERSION
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free
from numpy cimport ndarray

from .array_ext cimport H5ARRAYget_chunkshape
from .attributes_ext cimport H5ATTRset_attribute_string
from .definitions cimport hid_t, herr_t, hsize_t, hvl_t
from .definitions cimport H5P_DEFAULT
from .definitions cimport H5T_CSET_ASCII, H5T_CSET_UTF8, H5T_cset_t, H5Tget_order
from .definitions cimport H5Dopen, H5Dget_space, H5Dvlen_get_buf_size, H5Dread, H5Dvlen_reclaim
from .definitions cimport H5Sclose, H5Screate_simple, H5Sselect_hyperslab, H5Sselect_elements, H5Sselect_all, H5S_SELECT_SET, H5S_SELECT_AND, H5S_SELECT_NOTB
from .definitions cimport get_len_of_range
from .utilsextension cimport cstr_to_pystr, malloc_dims

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


cdef class VLArray(Leaf):
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
      nparr = np.ndarray(
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


