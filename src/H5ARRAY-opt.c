#include "H5ARRAY-opt.h"

#include <stdlib.h>
#include <string.h>

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
 hsize_t  count[2];
 int      rank = 2;
 hsize_t  offset[2];
 hsize_t  stride[2] = {1, 1};

 /* To keep HP-UX compiler happy */
 count[0] = 1;
 count[1] = stop - start;
 offset[0] = irow;
 offset[1] = start;

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


