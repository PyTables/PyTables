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

#include <string.h>
#include <stdlib.h>

#include "H5ATTR.h"


/*-------------------------------------------------------------------------
 *
 * Set attribute functions
 *
 *-------------------------------------------------------------------------
 */


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
				   const char *attr_data )
{
 hid_t      attr_type;
 hid_t      attr_size;
 hid_t      attr_space_id;
 hid_t      attr_id;
 int        has_attr;

 /* Create the attribute */
 if ( (attr_type = H5Tcopy( H5T_C_S1 )) < 0 )
  goto out;

 attr_size = strlen( attr_data ) + 1; /* extra null term */

 if ( H5Tset_size( attr_type, (size_t)attr_size) < 0 )
  goto out;

 if ( H5Tset_strpad( attr_type, H5T_STR_NULLTERM ) < 0 )
  goto out;

 if ( (attr_space_id = H5Screate( H5S_SCALAR )) < 0 )
  goto out;

 /* Verify if the attribute already exists */
 has_attr = H5ATTR_find_attribute( obj_id, attr_name );

 /* The attribute already exists, delete it */
 if ( has_attr == 1 )
 {
  if ( H5Adelete( obj_id, attr_name ) < 0 )
    goto out;
 }

 /* Create and write the attribute */

 if ( (attr_id = H5Acreate( obj_id, attr_name, attr_type, attr_space_id, H5P_DEFAULT )) < 0 )
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
 * Function: H5ATTRset_attribute_string_CAarray
 *
 * Purpose: Creates and writes an array of strings attribute named
 * attr_name and attaches it to the object specified by the name
 * obj_name.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Altet
 *
 * Date: January 10, 2006
 *
 * Comments: This is the multidimensional version of H5ATTRset_attribute_string
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRset_attribute_string_CAarray( hid_t obj_id,
					   const char *attr_name,
					   size_t rank,
					   hsize_t *dims,
					   int itemsize,
					   const char *attr_data )
{
 hid_t      attr_type;
 hid_t      attr_size;
 hid_t      attr_space_id;
 hid_t      attr_id;
 int        has_attr;

 /* Create the attribute */
 if ( (attr_type = H5Tcopy( H5T_C_S1 )) < 0 )
  goto out;

 attr_size = itemsize;

 if ( H5Tset_size( attr_type, (size_t)attr_size ) < 0 )
  goto out;

 if ( H5Tset_strpad( attr_type, H5T_STR_NULLTERM ) < 0 )
  goto out;

 /* Create the data space for the attribute. */
 if ( (attr_space_id = H5Screate_simple( rank, dims, NULL )) < 0 )
  goto out;

 /* Verify if the attribute already exists */
 has_attr = H5ATTR_find_attribute( obj_id, attr_name );

 /* The attribute already exists, delete it */
 if ( has_attr == 1 )
 {
  if ( H5Adelete( obj_id, attr_name ) < 0 )
    goto out;
 }

 /* Create and write the attribute */

 if ( (attr_id = H5Acreate( obj_id, attr_name, attr_type, attr_space_id, H5P_DEFAULT )) < 0 )
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
 * Function: H5ATTR_set_attribute_numerical
 *
 * Purpose: Private function used by H5ATTRset_attribute_int and H5ATTRset_attribute_float
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: July 25, 2001
 *
 * Comments:
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTR_set_attribute_numerical( hid_t obj_id,
				       const char *attr_name,
				       hid_t type_id,
				       const void *data )
{

 hid_t      space_id, attr_id;
 int        has_attr;

 /* Create the data space for the attribute. */
/*  if ( (space_id = H5Screate_simple( 1, &dim_size, NULL )) < 0 ) */
 /* For File version 1.4 on, the numerical attributes will be saved as
    regular scalars, as they should */
 if ( (space_id = H5Screate( H5S_SCALAR )) < 0 )
  goto out;

  /* Verify if the attribute already exists */
 has_attr = H5ATTR_find_attribute( obj_id, attr_name );

 /* The attribute already exists, delete it */
 if ( has_attr == 1 )
 {
  if ( H5Adelete( obj_id, attr_name ) < 0 )
    goto out;
 }

 /* Create the attribute. */
 if ( (attr_id = H5Acreate( obj_id, attr_name, type_id, space_id, H5P_DEFAULT )) < 0 )
  goto out;

 /* Write the attribute data. */
 if ( H5Awrite( attr_id, type_id, data ) < 0 )
  goto out;

 /* Close the attribute. */
 if ( H5Aclose( attr_id ) < 0 )
  goto out;

 /* Close the dataspace. */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 return 0;

out:
 return -1;
}



/*-------------------------------------------------------------------------
 * Function: H5ATTRset_attribute_numerical_NParray
 *
 * Purpose: write an array attribute
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer:
 *
 * Date: July 25, 2003
 *
 * Comments:
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTRset_attribute_numerical_NParray( hid_t obj_id,
					      const char *attr_name,
					      size_t rank,
					      hsize_t *dims,
					      hid_t type_id,
					      const void *data )
{

 hid_t      space_id, attr_id;
 int        has_attr;

 /* Create the data space for the attribute. */
 if ( (space_id = H5Screate_simple( rank, dims, NULL )) < 0 )
  goto out;

  /* Verify if the attribute already exists */
 has_attr = H5ATTR_find_attribute( obj_id, attr_name );

 /* The attribute already exists, delete it */
 if ( has_attr == 1 )
 {
  if ( H5Adelete( obj_id, attr_name ) < 0 )
    goto out;
 }

 /* Create the attribute. */
 if ( (attr_id = H5Acreate( obj_id, attr_name, type_id, space_id, H5P_DEFAULT )) < 0 )
  goto out;

 /* Write the attribute data. */
 if ( H5Awrite( attr_id, type_id, data ) < 0 )
  goto out;

 /* Close the attribute. */
 if ( H5Aclose( attr_id ) < 0 )
  goto out;

 /* Close the dataspace. */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 return 0;

out:
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: find_attr
 *
 * Purpose: operator function used by H5ATTR_find_attribute
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
 * Function: H5ATTR_find_attribute
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

herr_t H5ATTR_find_attribute( hid_t loc_id,
			      const char* attr_name )
{

 unsigned int attr_num;
 herr_t       ret;

 attr_num = 0;
 ret = H5Aiterate( loc_id, &attr_num, find_attr, (void *)attr_name );

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

herr_t H5ATTRget_attribute_ndims( hid_t obj_id,
				  const char *attr_name,
				  int *rank )
{
 hid_t       attr_id;
 hid_t       space_id;

 /* Open the attribute. */
 if ( ( attr_id = H5Aopen_name( obj_id, attr_name ) ) < 0 )
 {
  return -1;
 }

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
 H5Aclose( attr_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute_info
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

herr_t H5ATTRget_attribute_info( hid_t obj_id,
				 const char *attr_name,
				 hsize_t *dims,
				 H5T_class_t *type_class,
				 size_t *type_size,
				 hid_t *type_id)
{
 hid_t       attr_id;
 hid_t       space_id;

 /* Open the attribute. */
 if ( ( attr_id = H5Aopen_name( obj_id, attr_name ) ) < 0 )
 {
  return -1;
 }

 /* Get an identifier for the datatype. */
 *type_id = H5Aget_type( attr_id );

 /* Get the class. */
  *type_class = H5Tget_class( *type_id );

 /* Get the size. */
  *type_size = H5Tget_size( *type_id );

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
 H5Tclose( *type_id );
 H5Aclose( attr_id );
 return -1;

}


/*-------------------------------------------------------------------------
 *
 * Get attribute functions
 *
 *-------------------------------------------------------------------------
 */


/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute_string
 *
 * Purpose: Reads an attribute named attr_name
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Altet, faltet@carabos.com
 *
 * Date: February 23, 2005
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ATTRget_attribute_string( hid_t obj_id,
				   const char *attr_name,
				   char **data)
{
 /* identifiers */
 hid_t      attr_id;
 hid_t      attr_type;
 size_t     type_size;

 *data = NULL;
 if ( ( attr_id = H5Aopen_name( obj_id, attr_name ) ) < 0 )
  return -1;

 if ( (attr_type = H5Aget_type( attr_id )) < 0 )
  goto out;

 /* Get the size */
 if ( (type_size = H5Tget_size( attr_type )) < 0 )
  goto out;

 /* Malloc space enough for the string, plus 1 for the trailing '\0' */
 *data = (char *)malloc(type_size+1);

 if ( H5Aread( attr_id, attr_type, *data ) < 0 )
  goto out;

 /* Set the last character to \0 in case we are dealing with null
    padded or space padded strings */
 (*data)[type_size] = '\0';

 if ( H5Tclose( attr_type )  < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return -1;

 return 0;

out:
 H5Tclose( attr_type );
 H5Aclose( attr_id );
 if (*data) free(*data);
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute_string
 *
 * Purpose: Reads an attribute named attr_name
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

herr_t H5ATTRget_attribute_string_CAarray( hid_t obj_id,
					   const char *attr_name,
					   char *data )
{

 /* Get the attribute */
 if ( H5ATTR_get_attribute_disk( obj_id, attr_name, data ) < 0 )
  return -1;

 return 0;

}

/*-------------------------------------------------------------------------
 * Function: H5ATTRget_attribute
 *
 * Purpose: Reads an attribute named attr_name with the memory type mem_type_id
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 19, 2002
 *
 * Comments: Private function
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTRget_attribute( hid_t obj_id,
			    const char *attr_name,
			    hid_t mem_type_id,
			    void *data )
{

 /* Get the attribute */
 if ( H5ATTR_get_attribute_mem( obj_id, attr_name, mem_type_id, data ) < 0 )
  return -1;

 return 0;
}


/*-------------------------------------------------------------------------
 * Function: H5ATTR_get_attribute_mem
 *
 * Purpose: Reads an attribute named attr_name with the memory type mem_type_id
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 19, 2002
 *
 * Comments: Private function
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ATTR_get_attribute_mem( hid_t obj_id,
				 const char *attr_name,
				 hid_t mem_type_id,
				 void *data )
{

 /* identifiers */
 hid_t attr_id;

 if ( ( attr_id = H5Aopen_name( obj_id, attr_name ) ) < 0 )
  return -1;

 if ( H5Aread( attr_id, mem_type_id, data ) < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return -1;

 return 0;

out:
 H5Aclose( attr_id );
 return -1;
}



/*-------------------------------------------------------------------------
 * Function: H5ATTR_get_attribute_disk
 *
 * Purpose: Reads an attribute named attr_name with the dattype stored on disk
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


herr_t H5ATTR_get_attribute_disk( hid_t loc_id,
				  const char *attr_name,
				  void *attr_out )
{

 /* identifiers */
 hid_t      attr_id;
 hid_t      attr_type;

 if ( ( attr_id = H5Aopen_name( loc_id, attr_name ) ) < 0 )
  return -1;

 if ( (attr_type = H5Aget_type( attr_id )) < 0 )
  goto out;

 if ( H5Aread( attr_id, attr_type, attr_out ) < 0 )
  goto out;

 if ( H5Tclose( attr_type )  < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return -1;;

 return 0;

out:
 H5Tclose( attr_type );
 H5Aclose( attr_id );
 return -1;
}




