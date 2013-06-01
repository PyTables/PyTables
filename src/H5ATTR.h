
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
 * Modified versions of H5LT for getting and setting attributes for open
 * groups and leaves.
 * F. Alted 2005/09/29
 *                                                                          *
 ****************************************************************************/


#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif


/*-------------------------------------------------------------------------
 *
 * Set & get attribute functions
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRset_attribute( hid_t obj_id,
                            const char *attr_name,
                            hid_t type_id,
                            size_t rank,
                            hsize_t *dims,
                            const char *attr_data );

herr_t H5ATTRset_attribute_string( hid_t obj_id,
                                   const char *attr_name,
                                   const char *attr_data,
                                   hsize_t attr_size,
                                   int cset );

herr_t H5ATTRget_attribute( hid_t loc_id,
                            const char *attr_name,
                            hid_t type_id,
                            void *data );

hsize_t H5ATTRget_attribute_string( hid_t obj_id,
                                    const char *attr_name,
                                    char **data,
                                    int *cset );

hsize_t H5ATTRget_attribute_vlen_string_array( hid_t obj_id,
                                               const char *attr_name,
                                               char ***data,
                                               int *cset );

/*-------------------------------------------------------------------------
 *
 * Query attribute functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTRfind_attribute( hid_t loc_id,
                             const char* attr_name );

herr_t H5ATTRget_type_ndims( hid_t loc_id,
                             const char *attr_name,
                             hid_t *type_id,
                             H5T_class_t *class_id,
                             size_t *type_size,
                             int *rank );

herr_t H5ATTRget_dims( hid_t loc_id,
                       const char *attr_name,
                       hsize_t *dims );


#ifdef __cplusplus
}
#endif
