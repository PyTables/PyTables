#ifndef _H5TBO_H
#define _H5TBO_H
#endif

#include "H5LT.h"
#include "H5TB.h"

#define HLTB_MAX_FIELD_LEN 255

#ifdef __cplusplus
extern "C" {
#endif

herr_t H5TBOopen_read( hid_t *dataset_id,
		       hid_t *space_id,
		       hid_t *mem_type_id,
		       hid_t loc_id, 
		       const char *dset_name,
		       hsize_t nfields,
		       char **field_names,
		       size_t type_size,
		       size_t *field_offset );

herr_t H5TBOread_records( hid_t *dataset_id, hid_t *space_id,
			  hid_t *mem_type_id, hsize_t start,
			  hsize_t nrecords, void *data );

herr_t H5TBOread_elements( hid_t *dataset_id,
			   hid_t *space_id,
			   hid_t *mem_type_id,
			   hsize_t nrecords,
			   void *coords,
			   void *data );

herr_t H5TBOclose_read( hid_t *dataset_id,
			hid_t *space_id,
			hid_t *mem_type_id );


  /* These are maintained here just in case I want to use them in the future.
     F.Altet 2003/04/20  */


herr_t H5TBOopen_append( hid_t *dataset_id,
			 hid_t *mem_type_id,
			 hid_t loc_id, 
			 const char *dset_name,
			 hsize_t nfields,
			 size_t type_size,
			 const size_t *field_offset );

herr_t H5TBOappend_records( hid_t *dataset_id,
			    hid_t *mem_type_id,
			    hsize_t nrecords,
			    hsize_t nrecords_orig,
			    const void *data );

herr_t H5TBOclose_append(hid_t *dataset_id,
			 hid_t *mem_type_id,
			 hsize_t ntotal_records,
			 const char *dset_name,
			 hid_t parent_id);

herr_t H5TBOwrite_records( hid_t loc_id, 
			   const char *dset_name,
			   hsize_t start,
			   hsize_t nrecords,
			   hsize_t step,
			   size_t type_size,
			   const size_t *field_offset,
			   const void *data );
