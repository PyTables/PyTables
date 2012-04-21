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
 * Programmer: Francesc Alted, faltet@pytables.com
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
                               hid_t type_id,
                               hsize_t irow,
                               hsize_t start,
                               hsize_t stop,
                               void *data )
{
 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  count[2];
 int      rank = 2;
 hsize_t  offset[2];
 hsize_t  stride[2] = {1, 1};


 count[0] = 1;
 count[1] = stop - start;
 offset[0] = irow;
 offset[1] = start;

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

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

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYOinit_readSlice
 *
 * Purpose: Prepare structures to read specifics arrays faster
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: May 18, 2006
 *
 * Comments:
 *   - The H5ARRAYOinit_readSlice and H5ARRAYOread_readSlice
 *     are intended to read indexes slices only!
 *     F. Alted 2006-05-18
 *
 * Modifications:
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOinit_readSlice( hid_t dataset_id,
                               hid_t *mem_space_id,
                               hsize_t count)

{
 hid_t    space_id;
 int      rank = 2;
 hsize_t  count2[2] = {1, count};

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space(dataset_id )) < 0 )
  goto out;

 /* Create a memory dataspace handle */
 if ( (*mem_space_id = H5Screate_simple(rank, count2, NULL)) < 0 )
   goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 return 0;

out:
 H5Dclose(dataset_id);
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5ARRAYOread_readSortedSlice
 *
 * Purpose: Read records from an opened Array
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: Aug 11, 2005
 *
 * Comments:
 *
 * Modifications:
 *   - Modified to cache the mem_space_id as well.
 *     F. Alted 2005-08-11
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOread_readSortedSlice( hid_t dataset_id,
                                     hid_t mem_space_id,
                                     hid_t type_id,
                                     hsize_t irow,
                                     hsize_t start,
                                     hsize_t stop,
                                     void *data )
{
 hid_t    space_id;
 hsize_t  count[2] = {1, stop-start};
 hsize_t  offset[2] = {irow, start};
 hsize_t  stride[2] = {1, 1};

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space(dataset_id)) < 0 )
  goto out;

 /* Define a hyperslab in the dataset of the size of the records */
 if ( H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
   goto out;

 /* Read */
 if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
   goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYOread_readBoundsSlice
 *
 * Purpose: Read records from an opened Array
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: Aug 19, 2005
 *
 * Comments:  This is exactly the same as H5ARRAYOread_readSortedSlice,
 *    but I just want to distinguish the calls in profiles.
 *
 * Modifications:
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOread_readBoundsSlice( hid_t dataset_id,
                                     hid_t mem_space_id,
                                     hid_t type_id,
                                     hsize_t irow,
                                     hsize_t start,
                                     hsize_t stop,
                                     void *data )
{
 hid_t    space_id;
 hsize_t  count[2] = {1, stop-start};
 hsize_t  offset[2] = {irow, start};
 hsize_t  stride[2] = {1, 1};

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space(dataset_id)) < 0 )
  goto out;

 /* Define a hyperslab in the dataset of the size of the records */
 if ( H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
   goto out;

 /* Read */
 if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
   goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYreadSliceLR
 *
 * Purpose: Reads a slice of LR index cache from disk.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: August 17, 2005
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYOreadSliceLR(hid_t dataset_id,
                           hid_t type_id,
                           hsize_t start,
                           hsize_t stop,
                           void *data)
{
 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  count[1] = {stop - start};
 hsize_t  stride[1] = {1};
 hsize_t  offset[1] = {start};

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space(dataset_id)) < 0 )
  goto out;

 /* Define a hyperslab in the dataset of the size of the records */
 if ( H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
   goto out;

 /* Create a memory dataspace handle */
 if ( (mem_space_id = H5Screate_simple(1, count, NULL)) < 0 )
   goto out;

 /* Read */
 if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
   goto out;

 /* Release resources */

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
   goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
   goto out;

 return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}
