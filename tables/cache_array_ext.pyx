from h5py import h5s

from .definitions cimport hid_t, herr_t
from .exceptions import HDF5ExtError

cdef extern from "H5ARRAY-opt.h" nogil:
  herr_t H5ARRAYOread_readBoundsSlice(hid_t dataset_id, hid_t mem_space_id, hid_t type_id, hsize_t irow, hsize_t start, hsize_t stop, void *data)


cdef class CacheArray(Array):
  """Container for keeping index caches of 1st and 2nd level."""

  cdef initread(self, int nbounds):
     self.mem_space = h5s.create_simple((1, nbounds))

  cdef read_slice(self, hsize_t nrow, hsize_t start, hsize_t stop, ndarray buf):
    # "Read an slice of bounds."

    if (H5ARRAYOread_readBoundsSlice(
      self.dataset_id, self.mem_space.id, self.type_id,
      nrow, start, stop, buf.data) < 0):
      raise HDF5ExtError("Problems reading the bounds array data.")
    return

# cdef class CacheArray(Array):
#   """Container for keeping index caches of 1st and 2nd level."""
#
#   cpdef initread(self, int nbounds):
#     # "Actions to accelerate the reads afterwards."
#
#     # Precompute the mem_space_id
#     self.mem_space = h5s.create_simple((1, nbounds))
#     self.mem_space_id = self.mem_space.id
#
#
#   cpdef read_slice(self, hsize_t nrow, hsize_t start, hsize_t stop, ndarray rbuf):
#     # "Read an slice of bounds."
#
#     self.dataset = h5d.DatasetID(self.dataset_id)
#     assert self.dataset.is_valid()
#
#     memory_type = h5t.TypeID(self.type_id)
#
#     disk_space = self.dataset.get_space()
#     disk_space.select_hyperslab((nrow, start), (1, stop - start), (1, 1))
#
#     data = self.dataset.read(self.mem_space, disk_space, rbuf, memory_type)
#

