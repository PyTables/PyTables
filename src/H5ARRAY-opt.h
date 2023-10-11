#ifndef _H5ARRAY_OPT_H
#define _H5ARRAY_OPT_H

#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

hid_t H5ARRAYOmake(hid_t loc_id,
                   const char *dset_name,
                   const char *obversion,
                   const int rank,
                   const hsize_t *dims,
                   int extdim,
                   hid_t type_id,
                   hsize_t *dims_chunk,
                   hsize_t block_size,
                   void *fill_data,
                   int compress,
                   char *complib,
                   int shuffle,
                   int fletcher32,
                   hbool_t track_times,
                   const void *data);

herr_t H5ARRAYOreadSlice(char *filename,
                         hbool_t blosc2_support,
                         hid_t dataset_id,
                         hid_t type_id,
                         hsize_t *slice_start,
                         hsize_t *slice_stop,
                         hsize_t *slice_step,
                         void *slice_data);

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
#ifdef __cplusplus
}
#endif

#endif
