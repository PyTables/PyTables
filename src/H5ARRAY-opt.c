#include "H5ARRAY-opt.h"

#include <stdlib.h>
#include <string.h>

/*-------------------------------------------------------------------------
 * Function: H5ARRAYOopen_read
 *
 * Purpose: Prepare an array to be read incrementally
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Altet, faltet@carabos.com
 *
 * Date: May 27, 2004
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOopen_readSlice( hid_t *dataset_id,
			       hid_t *space_id,
			       hid_t *type_id,
			       hid_t loc_id, 
			       const char *dset_name)

{

 /* Open the dataset. */
 if ( (*dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
  return -1;
 
 /* Get the datatype */
 if ( (*type_id = H5Dget_type(*dataset_id)) < 0 )
     return -1;
 
  /* Get the dataspace handle */
 if ( (*space_id = H5Dget_space(*dataset_id )) < 0 )
  goto out;
 
 return 0;

out:
 H5Dclose( *dataset_id );
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5ARRAYOread_readSlice
 *
 * Purpose: Read records from an opened Array
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Altet, faltet@carabos.com
 *
 * Date: May 27, 2004
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOread_readSlice( hid_t dataset_id,
			       hid_t space_id,
			       hid_t type_id,
			       hsize_t irow,
			       hsize_t start,
			       hsize_t stop,
			       void *data )
{
 hid_t    mem_space_id;
 hsize_t  count[2] = {1, stop-start};
 int      rank = 2;
 hssize_t offset[2] = {irow, start};
 hsize_t  stride[2] = {1, 1};

 /* Create a memory dataspace handle */
 if ( (mem_space_id = H5Screate_simple( rank, count, NULL )) < 0 )
   goto out;

 /* Define a hyperslab in the dataset of the size of the records */
 if ( H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
   goto out;

 /* Read */
 if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
   goto out;

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
   goto out;

return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYOclose_readSlice
 *
 * Purpose: Close a table that has been opened for reading
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Altet, faltet@carabos.com
 *
 * Date: May 27, 2004
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOclose_readSlice(hid_t dataset_id,
			       hid_t space_id,
			       hid_t type_id)
{

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 /* Close the vlen type */
 if ( H5Tclose( type_id))
   return -1;

 /* End access to the dataset and release resources used by it. */
 if ( H5Dclose( dataset_id ) )
  return -1;

return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}

