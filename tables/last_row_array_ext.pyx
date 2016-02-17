from numpy cimport ndarray

from .definitions cimport hid_t, hsize_t, herr_t
from .hdf5extension cimport Array
from .index_array_ext cimport IndexArray

from .exceptions import HDF5ExtError

# Functions for optimized operations with ARRAY for indexing purposes
cdef extern from "H5ARRAY-opt.h" nogil:
  herr_t H5ARRAYOreadSliceLR(hid_t dataset_id, hid_t type_id, hsize_t start, hsize_t stop, void *data)


cdef class LastRowArray(Array):
  """
  Container for keeping sorted and indices values of last rows of an index.
  """

  def _read_index_slice(self, hsize_t start, hsize_t stop, ndarray idx):
    """Read the reverse index part of an LR index."""

    with nogil:
        ret = H5ARRAYOreadSliceLR(self.dataset_id, self.type_id,
                                  start, stop, idx.data)

    if ret < 0:
      raise HDF5ExtError("Problems reading the index data in Last Row.")


  def _read_sorted_slice(self, IndexArray sorted, hsize_t start, hsize_t stop):
    """Read the sorted part of an LR index."""

    cdef void  *rbuflb

    rbuflb = sorted.rbuflb  # direct access to rbuflb: very fast.
    with nogil:
        ret = H5ARRAYOreadSliceLR(self.dataset_id, self.type_id,
                                  start, stop, rbuflb)

    if ret < 0:
      raise HDF5ExtError("Problems reading the index data.")
    return sorted.bufferlb[:stop-start]



