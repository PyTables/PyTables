#ifndef _H5VLARRAY_H
#define _H5VLARRAY_H

#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

herr_t H5VLARRAYmake( hid_t loc_id, 
		      const char *dset_name,
		      const char *title,
		      const char *flavor,
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

herr_t H5VLARRAYappend_records( hid_t loc_id, 
				const char *dset_name,
				int nobjects,
				hsize_t nrecords,
				const void *data );

herr_t H5VLARRAYread( hid_t loc_id, 
		      const char *dset_name,
		      hsize_t start,
		      hsize_t nrows,
		      hsize_t step,
		      hvl_t *data,
		      hsize_t *datalen );

herr_t H5VLARRAYget_ndims( hid_t loc_id, 
			   const char *dset_name,
			   int *rank );

hid_t H5VLARRAYget_info( hid_t   loc_id, 
			 char    *dset_name,
			 hsize_t *nrecords,
			 hsize_t *base_dims,
			 hid_t   *base_type_id,
			 char    *base_byteorder);


#ifdef __cplusplus
}
#endif

#endif
