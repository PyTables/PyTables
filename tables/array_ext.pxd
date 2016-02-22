from .definitions cimport hid_t, herr_t,  hsize_t
from .hdf5extension cimport Leaf


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


cdef class Array(Leaf):
  cdef int      rank
  cdef hsize_t *maxdims
  cdef hsize_t *dims_chunk

