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
		    const char *title,  /* Added parameter */
		    const char *flavor,  /* Added parameter */
		    const char *obversion,  /* Added parameter */
		    const int atomic,  /* Added parameter */
		    const int rank, 
		    const hsize_t *dims,
		    hid_t type_id,
		    const void *data,
		    const int offset);

herr_t H5ARRAYread( hid_t loc_id, 
		    const char *dset_name,
		    void *data );

herr_t H5ARRAYget_ndims( hid_t loc_id, 
			 const char *dset_name,
			 int *rank );

herr_t H5ARRAYget_info( hid_t loc_id, 
			const char *dset_name,
			hsize_t *dims,
			H5T_class_t *class_id,
			H5T_sign_t *sign,
			char *byteorder,
			size_t *type_size );

#ifdef __cplusplus
}
#endif

#endif
