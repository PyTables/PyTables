
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
#include "Python.h"

#define TESTING(WHAT)	{printf("%-70s", "Testing " WHAT); fflush(stdout);}
#define PASSED()	{puts(" PASSED");fflush(stdout);}
#define H5_FAILED()	{puts("*FAILED*");fflush(stdout);}
#define SKIPPED()	{puts(" -SKIP-");fflush(stdout);}
#define EXAMPLE(WHAT)	{printf("%-70s", "Example " WHAT); fflush(stdout);}


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

herr_t H5LTmake_dataset_char( hid_t loc_id, 
                              const char *dset_name, 
                              int rank, 
                              const hsize_t *dims,
                              const char *buffer );

herr_t H5LTmake_dataset_short( hid_t loc_id, 
                               const char *dset_name, 
                               int rank, 
                               const hsize_t *dims,
                               const short *buffer );

herr_t H5LTmake_dataset_int( hid_t loc_id, 
                             const char *dset_name, 
                             int rank, 
                             const hsize_t *dims,
                             const int *buffer );

herr_t H5LTmake_dataset_long( hid_t loc_id, 
                              const char *dset_name, 
                              int rank, 
                              const hsize_t *dims,
                              const long *buffer );

herr_t H5LTmake_dataset_float( hid_t loc_id, 
                               const char *dset_name, 
                               int rank, 
                               const hsize_t *dims,
                               const float *buffer );

herr_t H5LTmake_dataset_double( hid_t loc_id, 
                                const char *dset_name, 
                                int rank, 
                                const hsize_t *dims,
                                const double *buffer );


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

herr_t H5LTread_dataset_char( hid_t loc_id, 
                              const char *dset_name,
                              char *buffer );

herr_t H5LTread_dataset_short( hid_t loc_id, 
                               const char *dset_name,
                               short *buffer );

herr_t H5LTread_dataset_int( hid_t loc_id, 
                             const char *dset_name,
                             int *buffer );

herr_t H5LTread_dataset_long( hid_t loc_id, 
                              const char *dset_name,
                              long *buffer );

herr_t H5LTread_dataset_float( hid_t loc_id, 
                               const char *dset_name,
                               float *buffer );

herr_t H5LTread_dataset_double( hid_t loc_id, 
                                const char *dset_name,
                                double *buffer );

/*-------------------------------------------------------------------------
 *
 * Query dataset functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5LTget_dataset_ndims( hid_t loc_id, 
                             const char *dset_name,
                             int *rank );

herr_t H5LTget_dataset_info( hid_t loc_id, 
                             const char *dset_name,
                             hsize_t *dims,
                             H5T_class_t *type_class,
                             size_t *type_size );

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
                          const char *obj_name, 
                          const char *attr_name,
                          hid_t mem_type_id,
                          void *data );

herr_t H5LTget_attribute_string( hid_t loc_id, 
                                 const char *obj_name, 
                                 const char *attr_name,
                                 char *data );

PyObject *H5LTget_attribute_string_sys( hid_t loc_id, 
					const char *obj_name, 
					const char *attr_name);

herr_t H5LTget_attribute_char( hid_t loc_id, 
                               const char *obj_name, 
                               const char *attr_name,
                               char *data );

herr_t H5LTget_attribute_short( hid_t loc_id, 
                                const char *obj_name, 
                                const char *attr_name,
                                short *data );

herr_t H5LTget_attribute_int( hid_t loc_id, 
                              const char *obj_name, 
                              const char *attr_name,
                              int *data );

herr_t H5LTget_attribute_long( hid_t loc_id, 
                               const char *obj_name, 
                               const char *attr_name,
                               long *data );

herr_t H5LTget_attribute_float( hid_t loc_id, 
                                const char *obj_name, 
                                const char *attr_name,
                                float *data );

herr_t H5LTget_attribute_double( hid_t loc_id, 
                                 const char *obj_name, 
                                 const char *attr_name,
                                 double *data );


/*-------------------------------------------------------------------------
 *
 * Query attribute functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5LTget_attribute_ndims( hid_t loc_id, 
                                const char *obj_name, 
                                const char *attr_name,
                                int *rank );

herr_t H5LTget_attribute_info( hid_t loc_id, 
                               const char *obj_name, 
                               const char *attr_name,
                               hsize_t *dims,
                               H5T_class_t *type_class,
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



/*-------------------------------------------------------------------------
 * 
 * Private functions
 * 
 *-------------------------------------------------------------------------
 */


herr_t H5LT_get_attribute_mem( hid_t obj_id, 
                           const char *attr_name,
                           hid_t mem_type_id,
                           void *data );

herr_t H5LT_get_attribute_disk( hid_t obj_id, 
                           const char *attr_name,
                           void *data );

herr_t H5LT_find_attribute( hid_t loc_id, const char *name ); 





#ifdef __cplusplus
}
#endif

#endif
