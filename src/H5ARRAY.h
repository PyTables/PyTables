#ifndef _H5ARRAY_H
#define _H5ARRAY_H

#include <hdf5.h>

#define TESTING(WHAT)	{printf("%-70s", "Testing " WHAT); fflush(stdout);}
#define PASSED()	{puts(" PASSED");fflush(stdout);}
#define H5_FAILED()	{puts("*FAILED*");fflush(stdout);}
#define SKIPPED()	{puts(" -SKIP-");fflush(stdout);}


#ifdef __cplusplus
extern "C" {
#endif

herr_t H5ARRAYmake( hid_t loc_id, 
		    const char *dset_name,
		    const char *title,
		    const char *flavor,
		    const char *obversion,
		    const int rank, 
		    const hsize_t *dims,
		    int   extdim,
		    hid_t type_id,
		    hsize_t max_tuples,
		    void  *fill_data,
		    int   compress,
		    char  *complib,
		    int   shuffle,
		    const void *data);

herr_t H5ARRAYappend_records( hid_t loc_id, 
			      const char *dset_name,
			      const int rank,
			      hsize_t *dims_orig,
			      hsize_t *dims_new,
			      int extdim,
			      const void *data );

herr_t H5ARRAYread( hid_t loc_id, 
		    const char *dset_name,
		    hsize_t start,
		    hsize_t nrows,
		    hsize_t step,
		    int extdim,
		    void *data );

herr_t H5ARRAYreadSlice( hid_t loc_id, 
			 const char *dset_name,
			 hsize_t *start,
			 hsize_t *stop,
			 hsize_t *step,
			 void *data );

herr_t H5ARRAYget_ndims( hid_t loc_id, 
			 const char *dset_name,
			 int *rank );

herr_t H5ARRAYget_info( hid_t loc_id, 
			const char *dset_name,
			hsize_t *dims,
			hid_t *super_type_id,
			H5T_class_t *super_class_id,
			char *byteorder);

#ifdef __cplusplus
}
#endif

#endif
