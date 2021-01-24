#ifndef _H5VLARRAY_H
#define _H5VLARRAY_H

#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

hid_t H5VLARRAYmake(  hid_t loc_id,
                      const char *dset_name,
                      const char *obversion,
                      const int rank,
                      const hsize_t *dims,
                      hid_t type_id,
                      hsize_t chunk_size,
                      void  *fill_data,
                      int   compress,
                      char  *complib,
                      int   shuffle,
                      int   fletcher32,
		      hbool_t track_times,
                      const void *data);

herr_t H5VLARRAYappend_records( hid_t dataset_id,
                                hid_t type_id,
                                size_t nobjects,
                                hsize_t nrecords,
                                const void *data );

herr_t H5VLARRAYmodify_records( hid_t dataset_id,
                                hid_t type_id,
                                hsize_t nrow,
                                size_t nobjects,
                                const void *data );

herr_t H5VLARRAYget_info( hid_t   dataset_id,
                          hid_t   type_id,
                          hsize_t *nrecords,
                          char    *base_byteorder);


#ifdef __cplusplus
}
#endif

#endif
