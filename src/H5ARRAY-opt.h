#include <hdf5.h>

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

