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

#include <string.h>
#include <stdlib.h>

#include "H5ATTR.h"


/*-------------------------------------------------------------------------
 *
 * Set & get attribute functions
 *
 *-------------------------------------------------------------------------
 */

/*-------------------------------------------------------------------------
 * Function: H5ATTRset_attribute
 *
 * Purpose: Create an attribute named attr_name and attach it to the
 * object specified by the name obj_name. This supports general
 * n-dimensional types (rank > 0), but if rank == 0, an H5T_SCALAR is
 * chosen.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted
 *
 * Date: October 18, 2006
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRset_attribute( hid_t obj_id,
                            const char *attr_name,
                            hid_t type_id,
                            size_t rank,
                            hsize_t *dims,
                            const char *attr_data )
{
 hid_t      space_id;
 hid_t      attr_id;
 int        has_attr;

 /* Create the data space for the attribute. */
 if (rank == 0)
   space_id = H5Screate( H5S_SCALAR );
 else
   space_id = H5Screate_simple( rank, dims, NULL );

 /* Verify whether the attribute already exists */
 has_attr = H5ATTRfind_attribute( obj_id, attr_name );

 /* The attribute already exists, delete it */
 if ( has_attr == 1 )
 {
  if ( H5Adelete( obj_id, attr_name ) < 0 )
    goto out;
 }

 /* Create and write the attribute */
 attr_id = H5Acreate( obj_id, attr_name, type_id, space_id, H5P_DEFAULT,
                      H5P_DEFAULT );

 if ( H5Awrite( attr_id, type_id, attr_data ) < 0 )
  goto out;

 H5Aclose( attr_id );

 H5Sclose( space_id );

 return 0;

out:
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTRset_attribute_string
 *
 * Purpose: Creates and writes a string attribute named attr_name and attaches
 *          it to the object specified by the name obj_name.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: July 23, 2001
 *
 * Comments: If the attribute already exists, it is overwritten
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRset_attribute_string( hid_t obj_id,
                                   const char *attr_name,
                                   const char *attr_data,
                                   hsize_t attr_size,
                                   int cset )
{
 hid_t      attr_type;
 /*size_t     attr_size;*/
 hid_t      attr_space_id;
 hid_t      attr_id;
 int        has_attr;

 /* Create the attribute */
 if ( (attr_type = H5Tcopy( H5T_C_S1 )) < 0 )
  goto out;

 if ( ( ( cset == H5T_CSET_ASCII ) || ( cset == H5T_CSET_UTF8 ) ) &&
      ( H5Tset_cset( attr_type, cset ) < 0 ) )
  goto out;

 if ( H5Tset_strpad( attr_type, H5T_STR_NULLTERM ) < 0 )
  goto out;

 if ( attr_size > 0 )
 {
  if (H5Tset_size( attr_type, attr_size) < 0 )
   goto out;
  if ( (attr_space_id = H5Screate( H5S_SCALAR )) < 0 )
   goto out;
 }
 else
 {
  if ( (attr_space_id = H5Screate( H5S_NULL )) < 0 )
   goto out;
 }

 /* Verify if the attribute already exists */
 has_attr = H5ATTRfind_attribute( obj_id, attr_name );

 /* The attribute already exists, delete it */
 if ( has_attr == 1 )
 {
  if ( H5Adelete( obj_id, attr_name ) < 0 )
    goto out;
 }

 /* Create and write the attribute */

 if ( (attr_id = H5Acreate( obj_id, attr_name, attr_type, attr_space_id,
                            H5P_DEFAULT, H5P_DEFAULT )) < 0 )
  goto out;

 if ( H5Awrite( attr_id, attr_type, attr_data ) < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  goto out;

 if ( H5Sclose( attr_space_id ) < 0 )
  goto out;

 if ( H5Tclose(attr_type) < 0 )
  goto out;

 return 0;

out:
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute
 *
 * Purpose: Reads an attribute named attr_name with the memory type type_id
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 19, 2002
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRget_attribute( hid_t obj_id,
                            const char *attr_name,
                            hid_t type_id,
                            void *data )
{

 /* identifiers */
 hid_t attr_id;

 if ( ( attr_id = H5Aopen_by_name(obj_id, ".", attr_name,
                                  H5P_DEFAULT, H5P_DEFAULT) ) < 0 )
  return -1;

 if ( H5Aread( attr_id, type_id, data ) < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return -1;

 return 0;

out:
 H5Aclose( attr_id );
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute_string
 *
 * Purpose: Reads an string attribute named attr_name.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: February 23, 2005
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

hsize_t H5ATTRget_attribute_string( hid_t obj_id,
                                    const char *attr_name,
                                    char **data,
                                    int *cset )
{
 /* identifiers */
 hid_t      attr_id;
 hid_t      attr_type;
 hid_t      space_id;
 hsize_t    type_size = 0;
 htri_t     is_vlstr = 0;

 *data = NULL;
 if ( ( attr_id = H5Aopen_by_name(obj_id, ".", attr_name,
                                  H5P_DEFAULT, H5P_DEFAULT) ) < 0 )
  return -1;

 if ( (attr_type = H5Aget_type( attr_id )) < 0 )
  goto out;

 if ( ( cset != NULL ) && ( ( *cset = H5Tget_cset( attr_type ) ) < 0 ) )
  goto out;

 is_vlstr = H5Tis_variable_str( attr_type );
 if ( is_vlstr == 0 )
 {
  /* Get the size */
  if ( (type_size = H5Tget_size( attr_type )) < 0 )
   goto out;

  if ( (space_id = H5Aget_space( attr_id )) < 0 )
   goto out;

  if ( H5Sget_simple_extent_type( space_id ) == H5S_NULL )
   type_size = 0;

  H5Sclose( space_id );

  /* Malloc space enough for the string, plus 1 for the trailing '\0' */
  *data = (char *)malloc(type_size + 1);

  if ( type_size > 0)
  {
   if ( H5Aread( attr_id, attr_type, *data ) < 0 )
    goto out;
  }

  /* Set the last character to \0 in case we are dealing with space
     padded strings */
  (*data)[type_size] = '\0';
 }
 else
 {
  /* is_vlstr */
  if ( H5Aread( attr_id, attr_type, data ) < 0 )
   goto out;

  type_size = strlen( *data );
 }

 if ( H5Tclose( attr_type ) < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return -1;

 return type_size;

out:
 H5Tclose( attr_type );
 H5Aclose( attr_id );
 if ( (is_vlstr == 0) && (*data != NULL) )
  free(*data);
 *data = NULL;
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute_vlen_string_array
 *
 * Purpose: Reads a variable length string attribute named attr_name.
 *
 * Return: Success: number of elements of the array, Failure: -1
 *
 * Programmer: Antonio Valentino <antonio.valentino@tiscali.it>
 *
 * Date: November 27, 2011
 *
 * Comments: only rank 1 attributes of 8bit strings are supported
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

hsize_t H5ATTRget_attribute_vlen_string_array( hid_t obj_id,
                                               const char *attr_name,
                                               char ***data,
                                               int *cset )
{
 /* identifiers */
 hid_t attr_id = -1, attr_type = -1, space_id = -1;
 hsize_t nelements = 0, *dims = NULL;
 int ndims = 0, i;

 *data = NULL;
 if ( ( attr_id = H5Aopen_by_name( obj_id, ".", attr_name,
                                   H5P_DEFAULT, H5P_DEFAULT ) ) < 0 )
  return -1;

 if ( (attr_type = H5Aget_type( attr_id )) < 0 )
  goto out;

 if ( ( cset != NULL ) && ( ( *cset = H5Tget_cset( attr_type ) ) < 0 ) )
  goto out;

 if ( (space_id = H5Aget_space( attr_id )) < 0 )
  goto out;

 if ( (ndims = H5Sget_simple_extent_ndims( space_id )) < 1 )
  goto out;

 if ( (dims = (hsize_t *)malloc(ndims * sizeof(hsize_t))) == NULL )
  goto out;

 if ( H5Sget_simple_extent_dims( space_id, dims, NULL ) < 0 )
  goto out;

 nelements = 1;
 for ( i = 0; i < ndims; ++i )
  nelements *= dims[i];

 free( dims );
 dims = NULL;

 if ((*data = (char **)malloc( nelements * sizeof(char*))) == NULL )
  goto out;

 if ( H5Aread( attr_id, attr_type, *data ) < 0 )
  goto out;

 if ( H5Tclose( attr_type ) < 0 )
  goto out;

 if ( H5Sclose( space_id ) < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return -1;

 return nelements;

out:
 if ( *data != NULL )
  free( *data );
 *data = NULL;
 if ( dims != NULL )
  free( dims );
 H5Tclose( attr_type );
 H5Sclose( space_id );
 H5Aclose( attr_id );
 return -1;
}


/*-------------------------------------------------------------------------
 *
 * Helper functions
 *
 *-------------------------------------------------------------------------
 */

/*-------------------------------------------------------------------------
 * Function: find_attr
 *
 * Purpose: operator function used by H5ATTRfind_attribute
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: June 21, 2001
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

static herr_t find_attr( hid_t loc_id,
                         const char *name,
                         const H5A_info_t *ainfo,
                         void *op_data)
{

 /* Define a default zero value for return. This will cause the
  * iterator to continue if the palette attribute is not found yet.
  */

 int ret = 0;

 char *attr_name = (char*)op_data;

 /* Shut the compiler up */
 loc_id=loc_id;

 /* Define a positive value for return value if the attribute was
  * found. This will cause the iterator to immediately return that
  * positive value, indicating short-circuit success
  */

 if( strcmp( name, attr_name ) == 0 )
  ret = 1;

 return ret;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTRfind_attribute
 *
 * Purpose: Inquires if an attribute named attr_name exists attached
 * to the object loc_id.
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: June 21, 2001
 *
 * Comments:
 *  The function uses H5Aiterate with the operator function find_attr
 *
 * Return:
 *  Success: The return value of the first operator that
 *              returns non-zero, or zero if all members were
 *              processed with no operator returning non-zero.
 *
 *  Failure: Negative if something goes wrong within the
 *              library, or the negative value returned by one
 *              of the operators.
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRfind_attribute( hid_t loc_id,
                             const char* attr_name )
{

 hsize_t attr_num;
 herr_t  ret;

 attr_num = 0;
 ret = H5Aiterate( loc_id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, &attr_num,
                   find_attr, (void *)attr_name );

 return ret;
}



/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute_ndims
 *
 * Purpose: Gets the dimensionality of an attribute.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 4, 2001
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRget_type_ndims( hid_t obj_id,
                             const char *attr_name,
                             hid_t *type_id,
                             H5T_class_t *class_id,
                             size_t *type_size,
                             int *rank )
{
 hid_t       attr_id;
 hid_t       space_id;

 /* Open the attribute. */
 if ( ( attr_id = H5Aopen_by_name(obj_id, ".", attr_name,
                                  H5P_DEFAULT, H5P_DEFAULT) ) < 0 )
 {
  return -1;
 }

 /* Get an identifier for the datatype. */
 *type_id = H5Aget_type( attr_id );

 /* Get the class. */
 *class_id = H5Tget_class( *type_id );

 /* Get the size. */
 *type_size = H5Tget_size( *type_id );

 /* Get the dataspace handle */
 if ( (space_id = H5Aget_space( attr_id )) < 0 )
  goto out;

 /* Get rank */
 if ( (*rank = H5Sget_simple_extent_ndims( space_id )) < 0 )
  goto out;

 /* Terminate access to the attribute */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 /* End access to the attribute */
 if ( H5Aclose( attr_id ) )
  goto out;;

 return 0;

out:
 H5Tclose( *type_id );
 H5Aclose( attr_id );
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTRget_dims
 *
 * Purpose: Gets information about an attribute.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 4, 2001
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRget_dims( hid_t obj_id,
                       const char *attr_name,
                       hsize_t *dims)
{
 hid_t       attr_id;
 hid_t       space_id;

 /* Open the attribute. */
 if ( ( attr_id = H5Aopen_by_name(obj_id, ".", attr_name,
                                  H5P_DEFAULT, H5P_DEFAULT) ) < 0 )
 {
  return -1;
 }

  /* Get the dataspace handle */
 if ( (space_id = H5Aget_space( attr_id )) < 0 )
  goto out;

 /* Get dimensions */
 if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

  /* End access to the attribute */
 if ( H5Aclose( attr_id ) )
  goto out;

 return 0;

out:
 H5Aclose( attr_id );
 return -1;
}
