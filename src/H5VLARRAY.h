#ifndef _H5VLARRAY_H
#define _H5VLARRAY_H

#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

herr_t H5VLARRAYmake( hid_t loc_id,
		      const char *dset_name,
		      const char *class_,
		      const char *title,
		      const char *obversion,    /* The Array VERSION number */
		      const int rank,
		      const int scalar,
		      const hsize_t *dims,
		      hid_t type_id,
		      hsize_t chunk_size,
		      void  *fill_data,
		      int   compress,
		      char  *complib,
		      int   shuffle,
		      int   fletcher32,
		      const void *data);

herr_t H5VLARRAYappend_records( hid_t dataset_id,
				hid_t type_id,
				int nobjects,
				hsize_t nrecords,
				const void *data );

herr_t H5VLARRAYmodify_records( hid_t dataset_id,
				hid_t type_id,
				hsize_t nrow,
				int nobjects,
				const void *data );

herr_t H5VLARRAYget_ndims( hid_t dataset_id,
			   hid_t type_id,
			   int *rank );

herr_t H5VLARRAYget_info( hid_t   dataset_id,
			  hid_t   type_id,
			  hsize_t *nrecords,
			  hsize_t *base_dims,
			  hid_t   *base_type_id,
			  char    *base_byteorder);


#ifdef __cplusplus
}
#endif

#endif
