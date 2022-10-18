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

/* WARNING: This is a highly stripped down and modified version of the
   original H5TB.c that comes with the HDF5 library. These
   modifications has been done in order to serve the needs of
   PyTables, and specially for supporting nested datatypes. In
   particular, the VERSION attribute is out of sync so it is not
   guaranteed that the resulting PyTables objects will be identical
   with those generated with HDF5_HL, although they should remain
   largely compatibles.

   F. Alted  2005/06/09

   Other modifications are that these routines are meant for opened
   nodes, and do not spend time opening and closing datasets.

   F. Alted 2005/09/29

 */

#include <stdlib.h>
#include <string.h>

#include "H5TB-opt.h"
#include "tables.h"
#include "H5Zlzo.h"                    /* Import FILTER_LZO */
#include "H5Zbzip2.h"                  /* Import FILTER_BZIP2 */
#include "blosc_filter.h"              /* Import FILTER_BLOSC */
#include "blosc2_filter.h"             /* Import FILTER_BLOSC2 */

#if defined(__GNUC__)
#define PUSH_ERR(func, minor, str, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, str, ##__VA_ARGS__)
#elif defined(_MSC_VER)
#define PUSH_ERR(func, minor, str, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, str, __VA_ARGS__)
#else
/* This version is portable but it's better to use compiler-supported
   approaches for handling the trailing comma issue when possible. */
#define PUSH_ERR(func, minor, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, __VA_ARGS__)
#endif	/* defined(__GNUC__) */

/* Define this in order to shrink datasets after deleting */
#if 1
#define SHRINK
#endif

/*-------------------------------------------------------------------------
 *
 * Create functions
 *
 *-------------------------------------------------------------------------
 */

/*-------------------------------------------------------------------------
 * Function: H5TBmake_table
 *
 * Purpose: Make a table
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *             Quincey Koziol
 *
 * Date: January 17, 2001
 *
 * Comments: The data is packed
 *  * Heavily modified and not compliant with attributes
 *    May 20, 2005
 *    F. Alted
 *
 * Modifications:
 *  * Modified by A. Cobb. August 21, 2017 (track_times)
 *
 *-------------------------------------------------------------------------
 */


hid_t H5TBOmake_table(  const char *table_title,
                        hid_t loc_id,
                        const char *dset_name,
                        char *version,
                        const char *class_,
                        hid_t type_id,
                        hsize_t nrecords,
                        hsize_t chunk_size,
                        hsize_t block_size,
                        void  *fill_data,
                        int compress,
                        char *complib,
                        int shuffle,
                        int fletcher32,
                        hbool_t track_times,
                        const void *data )
{

 hid_t   dataset_id;
 hid_t   space_id;
 hid_t   plist_id;
 hsize_t dims[1];
 hsize_t dims_chunk[1];
 hsize_t maxdims[1] = { H5S_UNLIMITED };
 unsigned int cd_values[7];
 int     blosc_compcode;
 char    *blosc_compname = NULL;

 dims[0]       = nrecords;
 dims_chunk[0] = chunk_size;

 /* Create a simple data space with unlimited size */
 if ( (space_id = H5Screate_simple( 1, dims, maxdims )) < 0 )
  return -1;

 /* Dataset creation properties */
 plist_id = H5Pcreate (H5P_DATASET_CREATE);

 /* Enable or disable recording dataset times */
 if ( H5Pset_obj_track_times( plist_id, track_times ) < 0 )
   return -1;

 /* Modify dataset creation properties, i.e. enable chunking  */
 if ( H5Pset_chunk ( plist_id, 1, dims_chunk ) < 0 )
  return -1;

 /* Set the fill value using a struct as the data type. */
 if ( fill_data)
   {
     if ( H5Pset_fill_value( plist_id, type_id, fill_data ) < 0 )
       return -1;
   }
 else {
   if ( H5Pset_fill_time(plist_id, H5D_FILL_TIME_ALLOC) < 0 )
     return -1;
 }

 /*
  Dataset creation property list is modified to use filters
  */

 /* Fletcher must be first */
 if (fletcher32) {
   if ( H5Pset_fletcher32( plist_id) < 0 )
     return -1;
 }
 /* Then shuffle (blosc/blosc2 shuffles inplace) */
 if ((shuffle && compress) && (strncmp(complib, "blosc", 5) != 0)) {
   if ( H5Pset_shuffle( plist_id) < 0 )
     return -1;
 }
 /* Finally compression */
 if ( compress )
 {
   cd_values[0] = compress;
   cd_values[1] = (int)(atof(version) * 10);
   cd_values[2] = Table;

   /* The default compressor in HDF5 (zlib) */
   if (strcmp(complib, "zlib") == 0) {
     if ( H5Pset_deflate( plist_id, compress) < 0 )
       return -1;
   }
   /* The Blosc2 compressor does accept parameters */
   else if (strcmp(complib, "blosc2") == 0) {
     cd_values[1] = (unsigned int) block_size;  /* can be useful in the future */
     cd_values[4] = compress;
     cd_values[5] = shuffle;
     if ( H5Pset_filter( plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 6, cd_values) < 0 )
       return -1;
   }
   /* The Blosc2 compressor can use other compressors */
   else if (strncmp(complib, "blosc2:", 7) == 0) {
     cd_values[1] = (unsigned int) block_size;  /* can be useful in the future */
     cd_values[4] = compress;
     cd_values[5] = shuffle;
     blosc_compname = complib + 7;
     blosc_compcode = blosc2_compname_to_compcode(blosc_compname);
     cd_values[6] = blosc_compcode;
     if ( H5Pset_filter( plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 7, cd_values) < 0 )
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
     return -1;
   }

 }

 /* Create the dataset. */
 if ( (dataset_id = H5Dcreate( loc_id, dset_name, type_id, space_id,
                               H5P_DEFAULT, plist_id, H5P_DEFAULT )) < 0 )
  goto out;

 /* Only write if there is something to write */
 if ( data )
 {
   /* Write data to the dataset. */
   if ( H5Dwrite( dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data ) < 0 )
     goto out;

 }

 /* Terminate access to the data space. */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 /* End access to the property list */
 if ( H5Pclose( plist_id ) < 0 )
  goto out;

 /* Return the object unique ID for future references */
 return dataset_id;

/* error zone, gracefully close */
out:
 H5E_BEGIN_TRY {
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Pclose(plist_id);
 } H5E_END_TRY;
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5TBOread_records
 *
 * Purpose: Read records from an opened table
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
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

herr_t H5TBOread_records( char* filename,
                          hbool_t blosc2_support,
                          hid_t dataset_id,
                          hid_t mem_type_id,
                          hsize_t start,
                          hsize_t nrecords,
                          void *data )
{
 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  count[1];
 hsize_t  offset[1];

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;
 if (blosc2_support) {
  /* Try to read using blosc2 (only supports native byteorder) */
  if (read_records_blosc2(filename, dataset_id, mem_type_id, space_id,
                          start, nrecords, (uint8_t*)data) >= 0) {
   goto success;
  }
 }

 /* Define a hyperslab in the dataset of the size of the records */
 offset[0] = start;
 count[0]  = nrecords;
 if ( H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
  goto out;

 /* Create a memory dataspace handle */
 if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
  goto out;

 if ( H5Dread(dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

success:
 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: read_records_blosc2
 *
 * Purpose: Read records from an opened table using blosc2
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, francesc@blosc.org
 *
 * Date: August 12, 2022
 *
 * Comments:
 *
 * Modifications:
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t read_records_blosc2( char* filename,
                            hid_t dataset_id,
                            hid_t mem_type_id,
                            hid_t space_id,
                            hsize_t start,
                            hsize_t nrecords,
                            uint8_t *data )
{
 uint8_t *buffer_out = NULL;
 /* Get the dataset creation property list */
 hid_t dcpl = H5Dget_create_plist(dataset_id);
 if (dcpl == H5I_INVALID_HID) {
  BLOSC_TRACE_ERROR("Fail getting plist");
  goto out;
 }

 /* Get blosc2 params */
 size_t cd_nelmts = 6;
 unsigned cd_values[6];
 char name[7];
 if (H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL, &cd_nelmts, cd_values, 7, name, NULL) < 0) {
  H5Pclose(dcpl);
  BLOSC_TRACE_ERROR("Fail getting blosc2 params");
  goto out;
 }
 if (H5Pclose(dcpl) < 0)
  goto out;

 /* Check that the compressor name is correct */
 if (strcmp(name, "blosc2") != 0) {
  goto out;
 }
 int32_t typesize = cd_values[2];
 int32_t chunksize = cd_values[3];

 /* Buffer for reading a chunk */
 buffer_out = malloc(chunksize);
 if (buffer_out == NULL) {
  BLOSC_TRACE_ERROR("Malloc failed for buffer_out");
  return -1;
 }

 hsize_t total_records = 0;
 int32_t chunkshape = chunksize / typesize;
 hsize_t start_nchunk = start / chunkshape;
 int32_t start_chunk = start % chunkshape;
 hsize_t stop_nchunk = (start + nrecords) / chunkshape;
 if (nrecords % chunkshape) {
  stop_nchunk += 1;
 }
 for (hsize_t nchunk = start_nchunk; nchunk < stop_nchunk && total_records < nrecords; nchunk++) {
  /* Open the schunk on-disk */
  unsigned flt_msk;
  haddr_t address;
  hsize_t cframe_size;
  hsize_t chunk_offset;
  if (H5Dget_chunk_info(dataset_id, space_id, nchunk, &chunk_offset, &flt_msk,
                        &address, &cframe_size) < 0) {
   BLOSC_TRACE_ERROR("Get chunk info failed!\n");
   goto out;
  }
  blosc2_schunk *schunk = blosc2_schunk_open_offset(filename, (int64_t) address);
  if (schunk == NULL) {
   BLOSC_TRACE_ERROR("Cannot open schunk in %s\n", filename);
   goto out;
  }

  /* Get chunk */
  bool needs_free;
  uint8_t *chunk;
  int32_t cbytes = blosc2_schunk_get_lazychunk(schunk, 0, &chunk, &needs_free);
  if (cbytes < 0) {
   BLOSC_TRACE_ERROR("Cannot get lazy chunk %zd in %s\n", nchunk, filename);
   goto out;
  }

  int32_t blocksize;
  if (blosc2_cbuffer_sizes(chunk, NULL, NULL, &blocksize) < 0) {
    goto out;
  }

  blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
  // Experiments say that 4 threads do not harm performance
  dparams.nthreads = 4;
  dparams.schunk = schunk;
  blosc2_context *dctx = blosc2_create_dctx(dparams);

  /* Gather data for the interesting part */
  hsize_t nrecords_chunk = chunkshape - start_chunk;
  if (nrecords_chunk > nrecords - total_records) {
   nrecords_chunk = nrecords - total_records;
  }

  int32_t blockshape = blocksize / typesize;
  int32_t nblocks = chunkshape / blockshape;
  int32_t start_nblock = start_chunk / blockshape;
  int32_t stop_nblock = (int32_t) (start_chunk + nrecords_chunk) / blockshape;

  if (nrecords_chunk > blockshape) {
   /* We have more than 1 block to read, so use a masked read */
   bool *block_maskout = calloc(nblocks, 1);
   if (block_maskout == NULL) {
    BLOSC_TRACE_ERROR("Calloc failed for block_maskout");
    return -1;
   }
   int32_t nblocks_set = 0;
   for (int32_t nblock = 0; nblock < nblocks; nblock++) {
    if ((nblock < start_nblock) || (nblock > stop_nblock)) {
     block_maskout[nblock] = true;
     nblocks_set++;
    }
   }
   if (blosc2_set_maskout(dctx, block_maskout, nblocks) != BLOSC2_ERROR_SUCCESS) {
    BLOSC_TRACE_ERROR("Error setting the maskout");
    goto out;
   }
   int32_t nbytes = blosc2_decompress_ctx(dctx, chunk, cbytes, buffer_out, chunksize);
   if (nbytes < 0) {
    BLOSC_TRACE_ERROR("Cannot decompress lazy chunk");
    goto out;
   }
   /* Copy data to destination */
   int rbytes = (int) nrecords_chunk * typesize;
   memcpy(data, buffer_out + start_chunk * typesize, rbytes);
   data += rbytes;
   total_records += nrecords_chunk;
   free(block_maskout);
  }
  else {
   /* Less than 1 block to read; use a getitem call */
   int rbytes = (int) blosc2_getitem_ctx(dctx, chunk, cbytes, start_chunk, (int) nrecords_chunk, buffer_out, chunksize);
   if (rbytes < 0) {
    BLOSC_TRACE_ERROR("Cannot get items for lazychunk\n");
    goto out;
   }
   /* Copy data to destination */
   memcpy(data, buffer_out, rbytes);
   data += rbytes;
   total_records += nrecords_chunk;
  }

  if (needs_free) {
   free(chunk);
  }
  blosc2_free_ctx(dctx);
  blosc2_schunk_free(schunk);

  /* Next chunk starts at 0 */
  start_chunk = 0;
 }

 free(buffer_out);

 return 0;

 out:
 if (buffer_out != NULL) {
  free(buffer_out);
 }
 return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5TBOread_elements
 *
 * Purpose: Read selected records from an opened table
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, faltet@pytables.com
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

herr_t H5TBOread_elements( hid_t dataset_id,
                           hid_t mem_type_id,
                           hsize_t nrecords,
                           void *coords,
                           void *data )
{

 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  count[1];

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Define a selection of points in the dataset */

 if ( H5Sselect_elements(space_id, H5S_SELECT_SET, (size_t)nrecords, (const hsize_t *)coords) < 0 )
  goto out;

 /* Create a memory dataspace handle */
 count[0] = nrecords;
 if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
  goto out;

 if ( H5Dread( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
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
 *  Francesc Alted, faltet@pytables.com
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

herr_t H5TBOappend_records( hid_t dataset_id,
                            hid_t mem_type_id,
                            hsize_t nrecords,
                            hsize_t nrecords_orig,
                            const void *data )
{
 hid_t    space_id = -1;        /* Shut up the compiler */
 hsize_t  count[1];
 hsize_t  offset[1];
 hid_t    mem_space_id = -1;    /* Shut up the compiler */
 hsize_t  dims[1];
 uint8_t  *data2 = (uint8_t *) data;

 /* Extend the dataset */
 dims[0] = nrecords_orig;
 dims[0] += nrecords;
 if ( H5Dset_extent(dataset_id, dims) < 0 )
  goto out;

 /* Get the dataset creation property list */
 hid_t dcpl = H5Dget_create_plist(dataset_id);
 if (dcpl == H5I_INVALID_HID) {
  goto out;
 }
 size_t cd_nelmts = 6;
 unsigned cd_values[6];
 char name[7];
 if (H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL, &cd_nelmts, cd_values, 7, name, NULL) < 0) {
  if (H5Pclose(dcpl) < 0)
   goto out;
  goto regular_write;
 }
 /* Check if the compressor name is blosc2 */
 if ((strncmp(name, "blosc2", 6)) == 0) {
  int typesize = cd_values[2];
  hsize_t cshape[1];
  H5Pget_chunk(dcpl, 1, cshape);
  if (H5Pclose(dcpl) < 0)
   goto out;
  int chunkshape = (int) cshape[0];
  int cstart = (int) (nrecords_orig / chunkshape);
  int cstop = (int) (nrecords_orig + nrecords - 1) / chunkshape + 1;
  int data_offset = 0;
  for (int ci = cstart; ci < cstop; ci ++) {
   if (ci == cstart) {
    if ((nrecords_orig % chunkshape == 0) && (nrecords >= chunkshape)) {
     if (append_records_blosc2(dataset_id, chunkshape, data) < 0)
      goto out;
    } else {
     /* Create a simple memory data space */
     if ((nrecords_orig % chunkshape) + nrecords <= chunkshape) {
      count[0] = nrecords;
     } else {
      count[0] = chunkshape - (nrecords_orig % chunkshape);
     }
     if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
      return -1;

     /* Get the file data space */
     if ( (space_id = H5Dget_space(dataset_id)) < 0 )
      return -1;

     /* Define a hyperslab in the dataset */
     offset[0] = nrecords_orig;
     if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
      goto out;

     if ( H5Dwrite( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
      goto out;

    }
   } else if (ci == cstop - 1) {
    data_offset = chunkshape - (nrecords_orig % chunkshape) + (ci - cstart - 1) * chunkshape;
    count[0] = nrecords - data_offset;
    if (count[0] == chunkshape) {
     if (append_records_blosc2(dataset_id, count[0],
                              (const void *) (data2 + data_offset * typesize)) < 0)
      goto out;
    } else {
     /* Create a simple memory data space */
     if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
      return -1;

     /* Get the file data space */
     if ( (space_id = H5Dget_space(dataset_id)) < 0 )
      return -1;

     /* Define a hyperslab in the dataset */
     offset[0] = nrecords_orig + data_offset;
     if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
      goto out;

     if ( H5Dwrite( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT,
                    (const void *) (data2 + data_offset * typesize) ) < 0 )
      goto out;
    }
   } else {
    data_offset = chunkshape - (nrecords_orig % chunkshape) + (ci - cstart - 1) * chunkshape;
    if (append_records_blosc2(dataset_id, chunkshape,
                              data2 + data_offset * typesize) < 0)
     goto out;
   }
  }

  goto success;
 }

 regular_write:
 /* Create a simple memory data space */
 count[0]=nrecords;
 if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
  return -1;

 /* Get the file data space */
 if ( (space_id = H5Dget_space(dataset_id)) < 0 )
  return -1;

 /* Define a hyperslab in the dataset */
 offset[0] = nrecords_orig;
 if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
  goto out;

 if ( H5Dwrite( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 success:
return 0;

out:
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: append_records_blosc2
 *
 * Purpose: Append records to a table using blosc2
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, francesc@blosc.org
 *
 * Date: September 5, 2022
 *
 * Comments:
 *
 * Modifications:
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t append_records_blosc2( hid_t dataset_id,
                              hsize_t nrecords,
                              const void *data )
{
 /* Get the dataset creation property list */
 hid_t dcpl = H5Dget_create_plist(dataset_id);
 if (dcpl == H5I_INVALID_HID) {
  BLOSC_TRACE_ERROR("Fail getting plist");
  goto out;
 }

 /* Get blosc2 params*/
 size_t cd_nelmts = 7;
 unsigned cd_values[7];
 char name[7];
 if (H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL, &cd_nelmts, cd_values, 7, name, NULL) < 0) {
  H5Pclose(dcpl);
  BLOSC_TRACE_ERROR("Fail getting blosc2 params");
  goto out;
 }
 int32_t typesize = cd_values[2];
 int32_t chunksize = cd_values[3];
 hsize_t chunkshape;
 H5Pget_chunk(dcpl, 1, &chunkshape);
 if (H5Pclose(dcpl) < 0)
  goto out;

 /* Compress data into superchunk and get frame */
 blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
 // Experiments say that 4 threads do not harm performance
 cparams.nthreads = 1;
 cparams.typesize = typesize;
 if (strncmp(name, "blosc2:", 7) == 0) {
  cparams.clevel = cd_values[4];
  cparams.filters[5] = cd_values[5];
  cparams.compcode = cd_values[6];
 }
 blosc2_context *cctx = blosc2_create_cctx(cparams);
 blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;

 blosc2_storage storage = {.cparams=&cparams, .dparams=&dparams,
                           .contiguous=true};
 int32_t chunk_size = (int32_t) nrecords * typesize;
 blosc2_schunk *sc = blosc2_schunk_new(&storage);
 if (sc == NULL) {
  BLOSC_TRACE_ERROR("Failed creating superchunk");
  goto out;
 }

 if (blosc2_schunk_append_buffer(sc, (void*) data, chunk_size) <= 0) {
  BLOSC_TRACE_ERROR("Failed appending buffer");
  goto out;
 }
 uint8_t* cframe;
 bool needs_free2;
 int cfsize = (int) blosc2_schunk_to_buffer(sc, &cframe, &needs_free2);
 if (cfsize <= 0) {
  BLOSC_TRACE_ERROR("Failed converting schunk to cframe");
  goto out;
 }

 /* Write frame bypassing HDF5 filter pipeline */
 unsigned flt_msk = 0;
 haddr_t offset[8];

 /* Workarround for avoiding the use of H5S_ALL in older HDF5 versions */
 /* H5S_ALL works well with 1.12.2, but not in HDF5 1.10.7 */
 hid_t d_space = H5Dget_space(dataset_id);
 hsize_t num_chunks;
 if (H5Dget_num_chunks(dataset_id, d_space, &num_chunks) < 0) {
  BLOSC_TRACE_ERROR("Failed getting number of chunks");
  goto out;
 }
 if (H5Sclose(d_space) < 0) {
  goto out;
 }

 offset[0] = num_chunks * chunkshape;
 if (H5Dwrite_chunk(dataset_id, H5P_DEFAULT, flt_msk, offset, cfsize, cframe) < 0) {
  BLOSC_TRACE_ERROR("Failed HDF5 writing chunk");
  goto out;
 }

 return 0;

 out:
  return -1;
}


/*-------------------------------------------------------------------------
 * Function: H5TBOwrite_records
 *
 * Purpose: Writes records
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: November 19, 2001
 *
 * Comments: Uses memory offsets
 *
 * Modifications:
 * -  Added a step parameter in order to support strided writing.
 *    Francesc Alted, faltet@pytables.com. 2004-08-12
 *
 * -  Removed the type_size which was unnecessary
 *    Francesc Alted, 2005-10-25
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOwrite_records( hid_t dataset_id,
                           hid_t mem_type_id,
                           hsize_t start,
                           hsize_t nrecords,
                           hsize_t step,
                           const void *data )
{

 hsize_t  count[1];
 hsize_t  stride[1];
 hsize_t  offset[1];
 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  dims[1];

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

  /* Get records */
 if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
  goto out;

/*  if ( start + nrecords > dims[0] ) */
 if ( start + (nrecords-1) * step + 1 > dims[0] )
  goto out;

 /* Define a hyperslab in the dataset of the size of the records */
 offset[0] = start;
 stride[0] = step;
 count[0] = nrecords;
 if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
  goto out;

 /* Create a memory dataspace handle */
 if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
  goto out;

 if ( H5Dwrite( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5TBOwrite_elements
 *
 * Purpose: Writes records on a list of coordinates
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted,
 *
 * Date: October 25, 2005
 *
 * Comments:
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOwrite_elements( hid_t dataset_id,
                            hid_t mem_type_id,
                            hsize_t nrecords,
                            const void *coords,
                            const void *data )
{

 hsize_t  count[1];
 hid_t    space_id;
 hid_t    mem_space_id;

 /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Define a selection of points in the dataset */

 if ( H5Sselect_elements(space_id, H5S_SELECT_SET, (size_t)nrecords, (const hsize_t *)coords) < 0 )
  goto out;

 /* Create a memory dataspace handle */
 count[0] = nrecords;
 if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
  goto out;

 if ( H5Dwrite( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 )
  goto out;

 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

return 0;

out:
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5TBOdelete_records
 *
 * Purpose: Delete records from middle of table ("pulling up" all the records after it)
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 * Modified by: F. Alted
 *
 * Date: November 26, 2001
 *
 * Modifications: April 29, 2003
 * Modifications: February 19, 2004 (buffered rewriting of trailing rows)
 * Modifications: September 28, 2005 (adapted to opened tables)
 *
 *
 *-------------------------------------------------------------------------
 */

herr_t H5TBOdelete_records( char* filename,
                            hbool_t blosc2_support,
                            hid_t   dataset_id,
                            hid_t   mem_type_id,
                            hsize_t ntotal_records,
                            size_t  src_size,
                            hsize_t start,
                            hsize_t nrecords,
                            hsize_t maxtuples)
{

 hsize_t  nrowsread;
 hsize_t  read_start;
 hsize_t  write_start;
 hsize_t  read_nrecords;
 hsize_t  count[1];
 hsize_t  offset[1];
 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  mem_size[1];
 unsigned char *tmp_buf;
 hsize_t  dims[1];
 size_t   read_nbuf;

 /* Shut the compiler up */
 tmp_buf = NULL;

/*-------------------------------------------------------------------------
 * Read the records after the deleted one(s)
 *-------------------------------------------------------------------------
 */

 read_start = start + nrecords;
 write_start = start;
 read_nrecords = ntotal_records - read_start;
 /* This check added for the case that there are no records to be read */
 /* F. Alted  2003/07/16 */
 if (read_nrecords > 0) {
   nrowsread = 0;

   while (nrowsread < read_nrecords) {

     if (nrowsread + maxtuples < read_nrecords)
       read_nbuf = (size_t)maxtuples;
     else
       read_nbuf = (size_t)(read_nrecords - nrowsread);

     tmp_buf = (unsigned char *)malloc(read_nbuf * src_size );

     if ( tmp_buf == NULL )
       return -1;

     /* Read the records after the deleted one(s) */
     if ( H5TBOread_records(filename, blosc2_support, dataset_id, mem_type_id, read_start,
                            read_nbuf, tmp_buf ) < 0 )
       return -1;

/*-------------------------------------------------------------------------
 * Write the records in another position
 *-------------------------------------------------------------------------
 */

     /* Get the dataspace handle */
     if ( (space_id = H5Dget_space( dataset_id )) < 0 )
       goto out;

     /* Define a hyperslab in the dataset of the size of the records */
     offset[0] = write_start;
     count[0]  = read_nbuf;
     if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 )
       goto out;

     /* Create a memory dataspace handle */
     mem_size[0] = count[0];
     if ( (mem_space_id = H5Screate_simple( 1, mem_size, NULL )) < 0 )
       goto out;

     if ( H5Dwrite( dataset_id, mem_type_id, mem_space_id, space_id, H5P_DEFAULT, tmp_buf ) < 0 )
       goto out;

     /* Terminate access to the memory dataspace */
     if ( H5Sclose( mem_space_id ) < 0 )
       goto out;

     /* Release the reading buffer */
     free( tmp_buf );

     /* Terminate access to the dataspace */
     if ( H5Sclose( space_id ) < 0 )
       goto out;

     /* Update the counters */
     read_start += read_nbuf;
     write_start += read_nbuf;
     nrowsread += read_nbuf;
   } /* while (nrowsread < read_nrecords) */
 } /*  if (nread_nrecords > 0) */


/*-------------------------------------------------------------------------
 * Change the table dimension
 *-------------------------------------------------------------------------
 */

#if defined (SHRINK)
 dims[0] = (int)ntotal_records - (int)nrecords;
 if ( H5Dset_extent( dataset_id, dims ) < 0 )
  goto out;
#endif

 return 0;

out:
 return -1;
}
