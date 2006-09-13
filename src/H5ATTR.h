
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
 * F. Altet 2005/09/29
 *                                                                          *
 ****************************************************************************/


#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif


/*-------------------------------------------------------------------------
 *
 * Set attribute functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTRset_attribute_string( hid_t obj_id,
				   const char *attr_name,
				   const char *attr_data );


herr_t H5ATTRset_attribute_string_CAarray( hid_t obj_id,
					   const char *attr_name,
					   size_t rank,
					   hsize_t *dims,
					   int itemsize,
					   const char *attr_data );

herr_t H5ATTRset_attribute_numerical_NParray( hid_t loc_id,
					      const char *attr_name,
					      size_t rank,
					      hsize_t *dims,
					      hid_t type_id,
					      const void *data );

/*-------------------------------------------------------------------------
 *
 * Get attribute functions
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRget_attribute( hid_t loc_id,
			    const char *attr_name,
			    hid_t mem_type_id,
			    void *data );

herr_t H5ATTRget_attribute_string( hid_t obj_id,
				   const char *attr_name,
				   char **data);

herr_t H5ATTRget_attribute_string_CAarray( hid_t obj_id,
					   const char *attr_name,
					   char *data );

/*-------------------------------------------------------------------------
 *
 * Query attribute functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTRget_attribute_ndims( hid_t loc_id,
				  const char *attr_name,
				  int *rank );

herr_t H5ATTRget_attribute_info( hid_t loc_id,
				 const char *attr_name,
				 hsize_t *dims,
				 H5T_class_t *type_class,
				 size_t *type_size,
				 hid_t *type_id );


/*-------------------------------------------------------------------------
 *
 * Private functions
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTR_get_attribute_mem( hid_t obj_id,
				 const char *attr_name,
				 hid_t mem_type_id,
				 void *data );

herr_t H5ATTR_get_attribute_disk( hid_t obj_id,
				  const char *attr_name,
				  void *data );

herr_t H5ATTR_find_attribute( hid_t loc_id,
			      const char *name );


herr_t H5ATTR_set_attribute_numerical( hid_t obj_id,
				       const char *attr_name,
				       hid_t type_id,
				       const void *data );




#ifdef __cplusplus
}
#endif
