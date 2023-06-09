#include <hdf5.h>

hid_t H5ARRAYOmake(hid_t loc_id,
                   const char *dset_name,
                   const char *obversion,
                   const int rank,
                   const hsize_t *dims,
                   int extdim,
                   hid_t type_id,
                   hsize_t *dims_chunk,
                   void *fill_data,
                   int compress,
                   char *complib,
                   int shuffle,
                   int fletcher32,
                   hbool_t track_times,
                   hbool_t blosc2_support,
                   const void *data);

herr_t H5ARRAYOwrite_records(hbool_t blosc2_support,
                             hid_t dataset_id,
                             hid_t type_id,
                             const int rank,
                             hsize_t *start,
                             hsize_t *step,
                             hsize_t *count,
                             const void *data);

herr_t write_records_blosc2_ndim(hid_t dataset_id,
                                 hid_t type_id,
                                 const int rank,
                                 hsize_t *start,
                                 hsize_t *step,
                                 hsize_t *count,
                                 const void *data);

herr_t insert_chunk_blosc2_ndim(hid_t dataset_id,
                                hsize_t *start,
                                hsize_t chunksize,
                                const void *data);


herr_t H5ARRAYOreadSlice(char *filename,
                         hbool_t blosc2_support,
                         hid_t dataset_id,
                         hid_t type_id,
                         hsize_t *start,
                         hsize_t *stop,
                         hsize_t *step,
                         void *data);

herr_t read_chunk_blosc2_ndim(char *filename, hid_t dataset_id, hid_t space_id, hsize_t nchunk, hsize_t chunk_start,
                              hsize_t chunksize,
                              uint8_t *data);

herr_t H5ARRAYOinit_readSlice( hid_t dataset_id,
                               hid_t *mem_space_id,
                               hsize_t count );

herr_t H5ARRAYOread_readSlice( hid_t dataset_id,
                               hid_t type_id,
                               hsize_t irow,
                               hsize_t start,
                               hsize_t stop,
                               void *data );

herr_t H5ARRAYOread_readSortedSlice( hid_t dataset_id,
                                     hid_t mem_space_id,
                                     hid_t type_id,
                                     hsize_t irow,
                                     hsize_t start,
                                     hsize_t stop,
                                     void *data );


herr_t H5ARRAYOread_readBoundsSlice( hid_t dataset_id,
                                     hid_t mem_space_id,
                                     hid_t type_id,
                                     hsize_t irow,
                                     hsize_t start,
                                     hsize_t stop,
                                     void *data );

herr_t H5ARRAYOreadSliceLR( hid_t dataset_id,
                            hid_t type_id,
                            hsize_t start,
                            hsize_t stop,
                            void *data );

