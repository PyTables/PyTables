#include "H5ATTR.h"

#include "tables.h"
#include "utils.h"
#include "H5Zlzo.h"                    /* Import FILTER_LZO */
#include "H5Zbzip2.h"                  /* Import FILTER_BZIP2 */
#include "blosc_filter.h"              /* Import FILTER_BLOSC */

#include <string.h>
#include <stdlib.h>

/*-------------------------------------------------------------------------
 *
 * Public functions
 *
 *-------------------------------------------------------------------------
 */
/*-------------------------------------------------------------------------
 * Function: H5ARRAYmake
 *
 * Purpose: Creates and writes a dataset of a type type_id
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: F. Alted. October 21, 2002
 *
 * Date: March 19, 2001
 *
 * Comments: Modified by F. Alted. November 07, 2003
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYmake( hid_t loc_id,
                    const char *dset_name,
                    const char *obversion,
                    const int rank,
                    const hsize_t *dims,
                    int   extdim,
                    hid_t type_id,
                    hsize_t *dims_chunk,
                    void  *fill_data,
                    int   compress,
                    char  *complib,
                    int   shuffle,
                    int   fletcher32,
                    const void *data)
{

 hid_t   dataset_id, space_id;
 hsize_t *maxdims = NULL;
 hid_t   plist_id = 0;
 unsigned int cd_values[7];
 int     blosc_compcode;
 char    *blosc_compname = NULL;
 int     chunked = 0;
 int     i;

 /* Check whether the array has to be chunked or not */
 if (dims_chunk) {
   chunked = 1;
 }

 if(chunked) {
   maxdims = malloc(rank*sizeof(hsize_t));
   if(!maxdims) return -1;

   for(i=0;i<rank;i++) {
     if (i == extdim) {
       maxdims[i] = H5S_UNLIMITED;
     }
     else {
       maxdims[i] = dims[i] < dims_chunk[i] ? dims_chunk[i] : dims[i];
     }
   }
 }

 /* Create the data space for the dataset. */
 if ( (space_id = H5Screate_simple( rank, dims, maxdims )) < 0 )
   return -1;

 if (chunked) {
   /* Modify dataset creation properties, i.e. enable chunking  */
   plist_id = H5Pcreate (H5P_DATASET_CREATE);
   if ( H5Pset_chunk ( plist_id, rank, dims_chunk ) < 0 )
     return -1;

   /* Set the fill value using a struct as the data type. */
   if (fill_data) {
     if ( H5Pset_fill_value( plist_id, type_id, fill_data ) < 0 )
       return -1;
   }
   else {
     if ( H5Pset_fill_time(plist_id, H5D_FILL_TIME_ALLOC) < 0 )
       return -1;
   }

   /*
      Dataset creation property list is modified to use
   */

   /* Fletcher must be first */
   if (fletcher32) {
     if ( H5Pset_fletcher32( plist_id) < 0 )
       return -1;
   }
   /* Then shuffle (blosc shuffles inplace) */
   if ((shuffle) && (strncmp(complib, "blosc", 5) != 0)) {
     if ( H5Pset_shuffle( plist_id) < 0 )
       return -1;
   }
   /* Finally compression */
   if (compress) {
     cd_values[0] = compress;
     cd_values[1] = (int)(atof(obversion) * 10);
     if (extdim <0)
       cd_values[2] = CArray;
     else
       cd_values[2] = EArray;

     /* The default compressor in HDF5 (zlib) */
     if (strcmp(complib, "zlib") == 0) {
       if ( H5Pset_deflate( plist_id, compress) < 0 )
         return -1;
     }
     /* The Blosc compressor does accept parameters */
     else if (strcmp(complib, "blosc") == 0) {
       cd_values[4] = compress;
       cd_values[5] = shuffle;
       if ( H5Pset_filter( plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 6, cd_values) < 0 )
         return -1;
     }
     /* The Blosc compressor can use other compressors */
     else if (strncmp(complib, "blosc:", 6) == 0) {
       cd_values[4] = compress;
       cd_values[5] = shuffle;
       blosc_compname = complib + 6;
       blosc_compcode = blosc_compname_to_compcode(blosc_compname);
       cd_values[6] = blosc_compcode;
       if ( H5Pset_filter( plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 7, cd_values) < 0 )
	 return -1;
     }
     /* The LZO compressor does accept parameters */
     else if (strcmp(complib, "lzo") == 0) {
       if ( H5Pset_filter( plist_id, FILTER_LZO, H5Z_FLAG_OPTIONAL, 3, cd_values) < 0 )
         return -1;
     }
     /* The bzip2 compress does accept parameters */
     else if (strcmp(complib, "bzip2") == 0) {
       if ( H5Pset_filter( plist_id, FILTER_BZIP2, H5Z_FLAG_OPTIONAL, 3, cd_values) < 0 )
         return -1;
     }
     else {
       /* Compression library not supported */
       fprintf(stderr, "Compression library not supported\n");
       return -1;
     }
   }

   /* Create the (chunked) dataset */
   if ((dataset_id = H5Dcreate(loc_id, dset_name, type_id, space_id,
                               H5P_DEFAULT, plist_id, H5P_DEFAULT )) < 0 )
     goto out;
 }
 else {         /* Not chunked case */
   /* Create the dataset. */
   if ((dataset_id = H5Dcreate(loc_id, dset_name, type_id, space_id,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT )) < 0 )
     goto out;
 }

 /* Write the dataset only if there is data to write */

 if (data)
 {
   if ( H5Dwrite( dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data ) < 0 )
   goto out;
 }

 /* Terminate access to the data space. */
 if ( H5Sclose( space_id ) < 0 )
  return -1;

 /* End access to the property list */
 if (plist_id)
   if ( H5Pclose( plist_id ) < 0 )
     goto out;

 /* Release resources */
 if (maxdims)
   free(maxdims);

 return dataset_id;

out:
 H5Dclose( dataset_id );
 H5Sclose( space_id );
 if (maxdims)
   free(maxdims);
 if (dims_chunk)
   free(dims_chunk);
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYappend_records
 *
 * Purpose: Appends records to an array
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmers:
 *  Francesc Alted
 *
 * Date: October 30, 2003
 *
 * Comments: Uses memory offsets
 *
 * Modifications:
 *
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ARRAYappend_records( hid_t dataset_id,
                              hid_t type_id,
                              const int rank,
                              hsize_t *dims_orig,
                              hsize_t *dims_new,
                              int extdim,
                              const void *data )
{

 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  *dims = NULL;         /* Shut up the compiler */
 hsize_t  *start = NULL;        /* Shut up the compiler */
 int      i;

 /* Compute the arrays for new dimensions and coordinates and extents */
 dims = malloc(rank*sizeof(hsize_t));
 start = malloc(rank*sizeof(hsize_t));
 for(i=0;i<rank;i++) {
   dims[i] = dims_orig[i];
   start[i] = 0;
 }
 dims[extdim] += dims_new[extdim];
 start[extdim] = (hsize_t )dims_orig[extdim];

 /* Extend the dataset */
 if ( H5Dset_extent( dataset_id, dims ) < 0 )
  goto out;

 /* Create a simple memory data space */
 if ( (mem_space_id = H5Screate_simple( rank, dims_new, NULL )) < 0 )
  return -1;

 /* Get the file data space */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  return -1;

 /* Define a hyperslab in the dataset */
 if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, start, NULL, dims_new, NULL) < 0 )
   goto out;

 if ( H5Dwrite( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
     goto out;

 /* Update the original dimensions of the array after a successful append */
 dims_orig[extdim] += dims_new[extdim];

 /* Terminate access to the dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

 if ( H5Sclose( space_id ) < 0 )
  goto out;

 /* Release resources */
 free(start);
 free(dims);

return 0;

out:
 if (start) free(start);
 if (dims) free(dims);
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYwrite_records
 *
 * Purpose: Write records to an array
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmers:
 *  Francesc Alted
 *
 * Date: October 26, 2004
 *
 * Comments: Uses memory offsets
 *
 * Modifications: Norbert Nemec for dealing with arrays of zero dimensions
 *                Date: Wed, 15 Dec 2004 18:48:07 +0100
 *
 *
 *-------------------------------------------------------------------------
 */


herr_t H5ARRAYwrite_records( hid_t dataset_id,
                             hid_t type_id,
                             const int rank,
                             hsize_t *start,
                             hsize_t *step,
                             hsize_t *count,
                             const void *data )
{

 hid_t    space_id;
 hid_t    mem_space_id;

 /* Create a simple memory data space */
 if ( (mem_space_id = H5Screate_simple( rank, count, NULL )) < 0 )
   return -3;

 /* Get the file data space */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  return -4;

 /* Define a hyperslab in the dataset */
 if ( rank != 0 && H5Sselect_hyperslab( space_id, H5S_SELECT_SET, start,
                                        step, count, NULL) < 0 )
  return -5;

 if ( H5Dwrite( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
   return -6;

 /* Terminate access to the dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
   return -7;

 if ( H5Sclose( space_id ) < 0 )
   return -8;

 /* Everything went smoothly */
 return 0;
}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYread
 *
 * Purpose: Reads an array from disk.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: October 22, 2002
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYread( hid_t dataset_id,
                    hid_t type_id,
                    hsize_t start,
                    hsize_t nrows,
                    hsize_t step,
                    int extdim,
                    void *data )
{

 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  *dims = NULL;
 hsize_t  *count = NULL;
 hsize_t  *stride = NULL;
 hsize_t  *offset = NULL;
 int      rank;
 int      i;
 int      _extdim;


 /* If dataset is not extensible, choose the first dimension as selectable */
 if (extdim < 0)
   _extdim = 0;
 else
   _extdim = extdim;

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Get the rank */
 if ( (rank = H5Sget_simple_extent_ndims(space_id)) < 0 )
   goto out;

 if (rank) {                    /* Array case */

   /* Book some memory for the selections */
   dims = (hsize_t *)malloc(rank*sizeof(hsize_t));
   count = (hsize_t *)malloc(rank*sizeof(hsize_t));
   stride = (hsize_t *)malloc(rank*sizeof(hsize_t));
   offset = (hsize_t  *)malloc(rank*sizeof(hsize_t));

   /* Get dataset dimensionality */
   if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
     goto out;

   if ( start + nrows > dims[_extdim] ) {
     printf("Asking for a range of rows exceeding the available ones!.\n");
     goto out;
   }

   /* Define a hyperslab in the dataset of the size of the records */
   for (i=0; i<rank;i++) {
     offset[i] = 0;
     count[i] = dims[i];
     stride[i] = 1;
     /*    printf("dims[%d]: %d\n",i, (int)dims[i]); */
   }
   offset[_extdim] = start;
   count[_extdim]  = nrows;
   stride[_extdim] = step;
   if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
     goto out;

   /* Create a memory dataspace handle */
   if ( (mem_space_id = H5Screate_simple( rank, count, NULL )) < 0 )
     goto out;

   /* Read */
   if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
     goto out;

   /* Release resources */
   free(dims);
   free(count);
   free(stride);
   free(offset);

   /* Terminate access to the memory dataspace */
   if ( H5Sclose( mem_space_id ) < 0 )
     goto out;
 }
 else {                 /* Scalar case */

   /* Read all the dataset */
   if (H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
     goto out;
 }

   /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 return 0;

out:
 if (dims) free(dims);
 if (count) free(count);
 if (stride) free(stride);
 if (offset) free(offset);
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYreadSlice
 *
 * Purpose: Reads a slice of array from disk.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: December 16, 2003
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYreadSlice( hid_t dataset_id,
                         hid_t type_id,
                         hsize_t *start,
                         hsize_t *stop,
                         hsize_t *step,
                         void *data )
{

 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  *dims = NULL;
 hsize_t  *count = NULL;
 hsize_t  *stride = (hsize_t *)step;
 hsize_t  *offset = (hsize_t *)start;
 int      rank;
 int      i;

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Get the rank */
 if ( (rank = H5Sget_simple_extent_ndims(space_id)) < 0 )
   goto out;

 if (rank) {                    /* Array case */

   /* Book some memory for the selections */
   dims = (hsize_t *)malloc(rank*sizeof(hsize_t));
   count = (hsize_t *)malloc(rank*sizeof(hsize_t));

   /* Get dataset dimensionality */
   if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
     goto out;

   for(i=0;i<rank;i++) {
     count[i] = get_len_of_range(start[i], stop[i], step[i]);
     if ( stop[i] > dims[i] ) {
       printf("Asking for a range of rows exceeding the available ones!.\n");
       goto out;
     }
   }

   /* Define a hyperslab in the dataset of the size of the records */
   if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, stride,
                             count, NULL) < 0 )
     goto out;

   /* Create a memory dataspace handle */
   if ( (mem_space_id = H5Screate_simple( rank, count, NULL )) < 0 )
     goto out;

   /* Read */
   if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT,
                 data ) < 0 )
     goto out;

   /* Release resources */
   free(dims);
   free(count);

   /* Terminate access to the memory dataspace */
   if ( H5Sclose( mem_space_id ) < 0 )
     goto out;
 }
 else {                     /* Scalar case */

   /* Read all the dataset */
   if (H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
     goto out;
 }

   /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 return 0;

out:
/*  H5Dclose( dataset_id ); */
 if (dims) free(dims);
 if (count) free(count);
 return -1;

}


/*   The next represents a try to implement getCoords for != operator */
/*   but it turned out to be too difficult, well, at least to me :( */
/*   2004-06-22 */
/*-------------------------------------------------------------------------
 * Function: H5ARRAYreadIndex
 *
 * Purpose: Reads a slice of array from disk for indexing purposes.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
 *
 * Date: June 21, 2004
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYreadIndex( hid_t   dataset_id,
                         hid_t   type_id,
                         int     notequal,
                         hsize_t *start,
                         hsize_t *stop,
                         hsize_t *step,
                         void    *data )
{

 hid_t    mem_space_id;
 hid_t    space_id;
 hsize_t  *dims = NULL;
 hsize_t  *count = NULL;
 hsize_t  *count2 = NULL;
 hsize_t  *offset2 = NULL;
 hsize_t  *stride = (hsize_t *)step;
 hsize_t  *offset = (hsize_t *)start;
 int      rank;
 int      i;

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Get the rank */
 if ( (rank = H5Sget_simple_extent_ndims(space_id)) < 0 )
   goto out;

 if (rank) {                    /* Array case */

   /* Book some memory for the selections */
   dims = (hsize_t *)malloc(rank*sizeof(hsize_t));
   count = (hsize_t *)malloc(rank*sizeof(hsize_t));
   count2 = (hsize_t *)malloc(rank*sizeof(hsize_t));
   offset2 = (hsize_t *)malloc(rank*sizeof(hsize_t));

   /* Get dataset dimensionality */
   if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
     goto out;

   for(i=0;i<rank;i++) {
     count[i] = get_len_of_range(start[i], stop[i], step[i]);
     if ( stop[i] > dims[i] ) {
       printf("Asking for a range of rows exceeding the available ones!.\n");
       goto out;
     }
   }

   /* Define a hyperslab in the dataset of the size of the records */
   if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, stride,
                             count, NULL) < 0 )
     goto out;

   /* If we want the complementary, do a NOTA against all the row */
   if (notequal) {
     offset2[0] = offset[0]; count2[0] = count[0];
     offset2[1] = 0; count2[1] = dims[1]; /* All the row */
     count[0] = 1; count[1] = dims[1] - count[1]; /* For memory dataspace */
     if ( H5Sselect_hyperslab( space_id, H5S_SELECT_NOTA, offset2, stride,
                               count2, NULL) < 0 )
       goto out;
   }

   /* Create a memory dataspace handle */
   if ( (mem_space_id = H5Screate_simple( rank, count, NULL )) < 0 )
     goto out;

   /* Read */
   if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
     goto out;

   /* Release resources */
   free(dims);
   free(count);
   free(offset2);
   free(count2);

   /* Terminate access to the memory dataspace */
   if ( H5Sclose( mem_space_id ) < 0 )
     goto out;
 }
 else {                         /* Scalar case */

   /* Read all the dataset */
   if (H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
     goto out;
 }

   /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 return 0;

out:
 if (dims) free(dims);
 if (count) free(count);
 return -1;

}



/*-------------------------------------------------------------------------
 * Function: H5ARRAYget_ndims
 *
 * Purpose: Gets the dimensionality of an array.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted
 *
 * Date: October 22, 2002
 *
 * Modification: October 13, 2008
 *   This routine not longer returns the dimensionality of data types
 *   in case they are H5T_ARRAY.
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYget_ndims( hid_t dataset_id,
                         int *rank )
{
  hid_t       space_id;

  /* Get the dataspace handle */
  if ( (space_id = H5Dget_space( dataset_id )) < 0 )
    goto out;

  /* Get rank */
  if ( (*rank = H5Sget_simple_extent_ndims( space_id )) < 0 )
    goto out;

  /* Terminate access to the dataspace */
  if ( H5Sclose( space_id ) < 0 )
    goto out;


  return 0;

out:
  return -1;

}



/* Modified version of H5LTget_dataset_info. */

herr_t H5ARRAYget_info( hid_t dataset_id,
                        hid_t type_id,
                        hsize_t *dims,
                        hsize_t *maxdims,
                        H5T_class_t *class_id,
                        char *byteorder)
{
  hid_t       space_id;

  /* Get the class. */
  *class_id = H5Tget_class( type_id );

  /* Get the dataspace handle */
  if ( (space_id = H5Dget_space( dataset_id )) < 0 )
    goto out;

  /* Get dimensions */
  if ( H5Sget_simple_extent_dims( space_id, dims, maxdims) < 0 )
    goto out;

  /* Terminate access to the dataspace */
  if ( H5Sclose( space_id ) < 0 )
    goto out;

  /* Get the byteorder */
  /* Only integer, float, time, enumerate and array classes can be
     byteordered */
  if ((*class_id == H5T_INTEGER) || (*class_id == H5T_FLOAT)
      || (*class_id == H5T_BITFIELD) || (*class_id == H5T_COMPOUND)
      || (*class_id == H5T_TIME) || (*class_id == H5T_ENUM)
      || (*class_id == H5T_ARRAY)) {
    get_order(type_id, byteorder);
  }
  else {
    strcpy(byteorder, "irrelevant");
  }

  return 0;

out:
 return -1;

}



/*-------------------------------------------------------------------------
 * Function: H5ARRAYget_chunkshape
 *
 * Purpose: Gets the chunkshape of a dataset.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted
 *
 * Date: May 20, 2004
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYget_chunkshape( hid_t dataset_id,
                              int rank,
                              hsize_t *dims_chunk)
{
  hid_t        plist_id;
  H5D_layout_t layout;

  /* Get creation properties list */
  if ( (plist_id = H5Dget_create_plist( dataset_id )) < 0 )
    goto out;

  /* Get the dataset layout */
  layout = H5Pget_layout(plist_id);
  if (layout != H5D_CHUNKED) {
    H5Pclose( plist_id );
    return -1;
  }

  /* Get the chunkshape for all dimensions */
  if (H5Pget_chunk(plist_id, rank, dims_chunk ) < 0 )
    goto out;

 /* Terminate access to the datatype */
 if ( H5Pclose( plist_id ) < 0 )
  goto out;

 return 0;

out:
 if (dims_chunk) free(dims_chunk);
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5ARRAYget_fill_value
 *
 * Purpose: Gets the fill value of a dataset.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted
 *
 * Date: Mar 03, 2009
 *
 *-------------------------------------------------------------------------
 */

herr_t H5ARRAYget_fill_value( hid_t dataset_id,
                              hid_t type_id,
                              int *status,
                              void *value)
{
  hid_t        plist_id;

  /* Get creation properties list */
  if ( (plist_id = H5Dget_create_plist(dataset_id)) < 0 )
    goto out;

  /* How the fill value is defined? */
  if ( (H5Pfill_value_defined(plist_id, status)) < 0 )
    goto out;

  if ( *status == H5D_FILL_VALUE_USER_DEFINED ) {
    if ( H5Pget_fill_value(plist_id, type_id, value) < 0 )
      goto out;
  }

  /* Terminate access to the datatype */
  if ( H5Pclose( plist_id ) < 0 )
    goto out;

  return 0;

out:
  return -1;

}
