#include "H5TB-opt.h"

#include <stdlib.h>
#include <string.h>

/*-------------------------------------------------------------------------
 * Function: H5TBOopen_read
 *
 * Purpose: Prepare a table to be read incrementally
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@openlc.org
 *
 * Date: April 19, 2003
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOopen_read( hid_t *dataset_id,
		       hid_t *space_id,
		       hid_t *mem_type_id, 
		       hid_t loc_id, 
		       const char *dset_name,
		       hsize_t nfields,
		       char **field_names,
		       size_t type_size,
		       const size_t *field_offset)
{
 hid_t    type_id;    
 hid_t    member_type_id;
 hsize_t  i;

 /* Open the dataset. */
 if ( (*dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
  return -1;

 /* Get the datatype */
 if ( (type_id = H5Dget_type( *dataset_id )) < 0 )
  goto out;

 /* Create the memory data type. */
 if ((*mem_type_id = H5Tcreate (H5T_COMPOUND, type_size )) < 0 )
  return -1;

 /* Insert fields on the memory data type. We use the types from disk */
 for ( i = 0; i < nfields; i++) 
 {

  /* Get the member type */
  if ( ( member_type_id = H5Tget_member_type( type_id, (int) i )) < 0 )
   goto out;

  if ( H5Tinsert(*mem_type_id, field_names[i], field_offset[i], member_type_id ) < 0 )
   goto out;

  /* Release the datatype */
  if ( H5Tclose( member_type_id ) < 0 )
   goto out; 

 }

 /* Release the type */
 if ( H5Tclose( type_id ) < 0 )
  return -1;

 /* Get the dataspace handle */
 if ( (*space_id = H5Dget_space( *dataset_id )) < 0 )
  goto out;
 
return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5TBOread_records
 *
 * Purpose: Read records from an opened table
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@openlc.org
 *
 * Date: April 19, 2003
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOread_records( hid_t *dataset_id,
			  hid_t *space_id,
			  hid_t *mem_type_id,
			  hsize_t start,
			  hsize_t nrecords,
			  void *data )
{

 hsize_t  count[1];    
 hssize_t offset[1];
 hid_t    mem_space_id;
 hsize_t  mem_size[1];

 /* Define a hyperslab in the dataset of the size of the records */
 offset[0] = start;
 count[0]  = nrecords;
 if ( H5Sselect_hyperslab( *space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
  goto out;

 /* Create a memory dataspace handle */
 mem_size[0] = count[0];
 if ( (mem_space_id = H5Screate_simple( 1, mem_size, NULL )) < 0 )
  goto out;

 if ( H5Dread( *dataset_id, *mem_type_id, mem_space_id, *space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5TBOclose_read
 *
 * Purpose: Close a table that has been opened for reading
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@openlc.org
 *
 * Date: April 19, 2003
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOclose_read(hid_t *dataset_id,
		       hid_t *space_id,
		       hid_t *mem_type_id)
{

 /* Terminate access to the dataspace */
 if ( H5Sclose( *space_id ) < 0 )
  goto out;
 
  /* Release the datatype. */
 if ( H5Tclose( *mem_type_id ) < 0 )
  goto out;

 /* End access to the dataset */
 if ( H5Dclose( *dataset_id ) < 0 )
  return -1;

return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}

/* From here on, similar funtions are provided for appending.
 */

/*-------------------------------------------------------------------------
 * Function: H5TBOopen_append
 *
 * Purpose: Prepare a table to append records
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@openlc.org
 *
 * Date: April 20, 2003
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOopen_append( hid_t *dataset_id,
			 hid_t *mem_type_id,
			 hid_t loc_id, 
			 const char *dset_name,
			 hsize_t nfields,
			 size_t type_size,
			 const size_t *field_offset)
{
 hid_t    type_id;    
 char     **field_names;
 hid_t    member_type_id;
 hsize_t  i;

  /* Alocate space */
 field_names = malloc( sizeof(char*) * (size_t)nfields );
 for ( i = 0; i < nfields; i++) 
 {
  field_names[i] = malloc( sizeof(char) * HLTB_MAX_FIELD_LEN );
 }

 /* Get field info */
 if ( H5TBget_field_info( loc_id, dset_name, field_names, NULL, NULL, NULL ) < 0 )
  return -1;

 /* Open the dataset. */
 if ( (*dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
  goto out;

  /* Get the datatype */
 if ( (type_id = H5Dget_type( *dataset_id )) < 0 )
  goto out;

 /* Create the memory data type. */
 if ((*mem_type_id = H5Tcreate (H5T_COMPOUND, type_size )) < 0 )
  return -1;

 /* Insert fields on the memory data type */
 for ( i = 0; i < nfields; i++) 
 {

  /* Get the member type */
  if ( ( member_type_id = H5Tget_member_type( type_id,(int) i )) < 0 )
   goto out;

  if ( H5Tinsert(*mem_type_id, field_names[i], field_offset[i], member_type_id ) < 0 )
   goto out;

  /* Close the member type */
  if ( H5Tclose( member_type_id ) < 0 )
   goto out;

 /* Release resources. */
  free ( field_names[i] );

 }

 /* Release resources. */
 free ( field_names );

 /* Release the datatype. */
 if ( H5Tclose( type_id ) < 0 )
  return -1;

return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5TBOappend_records
 *
 * Purpose: Appends records to a table
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmers: 
 *  Francesc Alted, falted@openlc.org
 *
 * Date: April 20, 2003
 *
 * Comments: Uses memory offsets
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */


herr_t H5TBOappend_records( hid_t *dataset_id,
			    hid_t *mem_type_id,
			    hsize_t nrecords,
			    hsize_t nrecords_orig,
			    const void *data )  
{
 hid_t    space_id;
 hsize_t  count[1];    
 hssize_t offset[1];
 hid_t    mem_space_id;
 int      rank;
 hsize_t  dims[1];
 hsize_t  mem_dims[1];


 /* Extend the dataset */
 dims[0] = nrecords_orig;
 dims[0] += nrecords;
 if ( H5Dextend ( *dataset_id, dims ) < 0 )
  goto out;

 /* Create a simple memory data space */
 mem_dims[0]=nrecords;
 if ( (mem_space_id = H5Screate_simple( 1, mem_dims, NULL )) < 0 )
  return -1;

 /* Get the file data space */
 if ( (space_id = H5Dget_space( *dataset_id )) < 0 )
  return -1;

 /* Get the dimensions */
 if ( (rank = H5Sget_simple_extent_dims( space_id, dims, NULL )) != 1 )
  goto out;

 /* Define a hyperslab in the dataset */
 offset[0] = nrecords_orig;
 count[0]  = nrecords;
 if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
  goto out;

 if ( H5Dwrite( *dataset_id, *mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

 if ( H5Sclose( space_id ) < 0 )
  goto out;
 
return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5TBOclose_append
 *
 * Purpose: Close a table that has been opened for append
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@openlc.org
 *
 * Date: April 20, 2003
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOclose_append(hid_t *dataset_id,
			 hid_t *mem_type_id,
			 int ntotal_records,
			 const char *dset_name,
			 hid_t parent_id)
{
 int nrows;
  
  /* Release the datatype. */
 if ( H5Tclose( *mem_type_id ) < 0 )
  goto out;

 /* End access to the dataset */
 if ( H5Dclose( *dataset_id ) < 0 )
  return -1;

/*-------------------------------------------------------------------------
 * Store the new dimension as an attribute
 *-------------------------------------------------------------------------
 */

 nrows = (int)ntotal_records;
 /* Set the attribute */
 if ( H5LTset_attribute_int(parent_id, dset_name, "NROWS", &nrows, 1 ) < 0 )
   return -1;

return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}

