
/****************************************************************************
 * NCSA HDF                                                                 *
 * Scientific Data Technologies                                             *
 * National Center for Supercomputing Applications                          *
 * University of Illinois at Urbana-Champaign                               *
 * 605 E. Springfield, Champaign IL 61820                                   *
 *                                                                          *
 * For conditions of distribution and use, see the accompanying             *
 * hdf/COPYING file.                                                        *
 *                                                                          *
 ****************************************************************************/


#ifndef _H5LT_H
#define _H5LT_H

#include <hdf5.h>

#define TESTING(WHAT)	{printf("%-70s", "Testing " WHAT); fflush(stdout);}
#define PASSED()	{puts(" PASSED");fflush(stdout);}
#define H5_FAILED()	{puts("*FAILED*");fflush(stdout);}
#define SKIPPED()	{puts(" -SKIP-");fflush(stdout);}


#ifdef __cplusplus
extern "C" {
#endif


/*-------------------------------------------------------------------------
 *
 * Make dataset functions
 *
 *-------------------------------------------------------------------------
 */

herr_t H5LTmake_dataset( hid_t loc_id, 
                         const char *dset_name, 
                         int rank, 
                         const hsize_t *dims,
                         hid_t type_id,
                         const void *buffer );

herr_t H5LTmake_array( hid_t loc_id, 
		       const char *dset_name,
		       const char *title,
		       int rank, 
		       const hsize_t *dims,
		       hid_t type_id,
		       const void *buffer ); 

/*-------------------------------------------------------------------------
 *
 * Read dataset functions
 *
 *-------------------------------------------------------------------------
 */

herr_t H5LTread_dataset( hid_t loc_id, 
                         const char *dset_name,
                         hid_t type_id,
                         void *buffer );

herr_t H5LTread_array( hid_t loc_id, 
		       const char *dset_name,
		       void *buffer );

/*-------------------------------------------------------------------------
 *
 * Query dataset functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5LTget_dataset_ndims( hid_t loc_id, 
                             const char *dset_name,
                             int *rank );

herr_t H5LTget_array_ndims ( hid_t loc_id,
			     const char *dset_name,
			     int *rank );
  
herr_t H5LTget_dataset_info( hid_t loc_id, 
                             const char *dset_name,
                             hsize_t *dims,
                             H5T_class_t *class_id,
                             size_t *type_size );

herr_t H5LTget_dataset_info_mod( hid_t loc_id, 
				 const char *dset_name,
				 hsize_t *dims,
				 H5T_class_t *class_id,
				 H5T_sign_t *sign, /* Added this parameter */
				 size_t *type_size );

herr_t H5LTget_array_info( hid_t loc_id, const char *dset_name,
                           hsize_t *dims, H5T_class_t *class_id,
			   H5T_sign_t *sign, size_t *type_size );

     
herr_t H5LTfind_dataset( hid_t loc_id, const char *name ); 



/*-------------------------------------------------------------------------
 *
 * Set attribute functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5LTset_attribute_string( hid_t loc_id, 
                                 const char *obj_name, 
                                 const char *attr_name,
                                 const char *attr_data );

herr_t H5LTset_attribute_char( hid_t loc_id, 
                               const char *obj_name, 
                               const char *attr_name,
                               char *buffer,
                               size_t size );

herr_t H5LTset_attribute_short( hid_t loc_id, 
                              const char *obj_name, 
                              const char *attr_name,
                              short *buffer,
                              size_t size );

herr_t H5LTset_attribute_int( hid_t loc_id, 
                              const char *obj_name, 
                              const char *attr_name,
                              int *buffer,
                              size_t size );

herr_t H5LTset_attribute_long( hid_t loc_id, 
                             const char *obj_name, 
                             const char *attr_name,
                             long *buffer,
                             size_t size );

herr_t H5LTset_attribute_float( hid_t loc_id, 
                                const char *obj_name, 
                                const char *attr_name,
                                float *buffer,
                                size_t size );

herr_t H5LTset_attribute_double( hid_t loc_id, 
                               const char *obj_name, 
                               const char *attr_name,
                               double *buffer,
                               size_t size );

/*-------------------------------------------------------------------------
 *
 * Get attribute functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5LTget_attribute( hid_t loc_id, 
                          const char *attr_name,
                          void *attr_out );


/*-------------------------------------------------------------------------
 *
 * Query attribute functions
 *
 *-------------------------------------------------------------------------
 */

herr_t H5LTfind_attribute( hid_t loc_id, const char *name ); 

herr_t H5LTget_attribute_ndims( hid_t loc_id, 
                                const char *attr_name,
                                int *rank );

herr_t H5LTget_attribute_info( hid_t loc_id, 
                               const char *attr_name,
                               hsize_t *dims,
                               H5T_class_t *class_id,
                               size_t *type_size );


/*-------------------------------------------------------------------------
 *
 * General functions
 *
 *-------------------------------------------------------------------------
 */


hid_t H5LTcreate_compound_type( hsize_t nfields, size_t size, const char *field_names[], 
                                const size_t *field_offset, const hid_t *field_types );


herr_t H5LTrepack( hsize_t nfields, 
                   hsize_t nrecords, 
                   size_t src_size, 
                   const size_t *src_offset, 
                   const size_t *src_sizes, 
                   size_t dst_size, 
                   const size_t *dst_offset, 
                   const size_t *dst_sizes,
                   unsigned char *src_buf, 
                   unsigned char *dst_buf );

#ifdef __cplusplus
}
#endif

#endif
