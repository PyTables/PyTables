#include "H5ARRAY-opt.h"
#include "tables.h"
#include "H5Zlzo.h"                    /* Import FILTER_LZO */
#include "H5Zbzip2.h"                  /* Import FILTER_BZIP2 */
#include "blosc_filter.h"              /* Import FILTER_BLOSC */
#include "blosc2_filter.h"             /* Import FILTER_BLOSC2 */

#include <stdlib.h>
#include <string.h>


herr_t read_chunk_blosc2_ndim(char *filename,
                              hid_t dataset_id,
                              hid_t space_id,
                              hsize_t nchunk,
                              hsize_t *chunk_start,
                              hsize_t *chunk_stop,
                              hsize_t chunksize,
                              uint8_t *data);

herr_t insert_chunk_blosc2_ndim(hid_t dataset_id,
                                hsize_t *start,
                                hsize_t chunksize,
                                const void *data);


herr_t get_set_blosc2_slice(char *filename, // can be NULL when writing
                          hid_t dataset_id,
                          hid_t type_id,
                          const int rank,
                          hsize_t *start,
                          hsize_t *stop,
                          hsize_t *step,
                          const void *data,
                          hbool_t set)
{
  uint8_t *data2 = (uint8_t *) data;
  /* Get the file data space */
  hid_t space_id;
  if ((space_id = H5Dget_space(dataset_id)) < 0)
    return -1;

  /* Get the dataset creation property list */
  hid_t dcpl = H5Dget_create_plist(dataset_id);
  if (dcpl == H5I_INVALID_HID) {
    return -2;
  }
  size_t cd_nelmts = 7;
  unsigned cd_values[7];
  char name[7];
  if (H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL, &cd_nelmts, cd_values, 7, name, NULL) < 0) {
    H5Pclose(dcpl);
    return -3;
  }
  int typesize = cd_values[2];
  hsize_t chunkshape[rank];
  H5Pget_chunk(dcpl, rank, &chunkshape);

  if (H5Pclose(dcpl) < 0)
    return -4;

  hsize_t shape[rank];
  H5Sget_simple_extent_dims(space_id, shape, NULL);
  int64_t extshape[rank];

  for (int i = 0; i < rank; i++) {
    if (shape[i] % chunkshape[i] != 0) {
      extshape[i] = shape[i] + chunkshape[i] - shape[i] % chunkshape[i];
    } else {
      extshape[i] = shape[i];
    }
  }

  int64_t chunks_in_array[rank];
  for (int i = 0; i < rank; ++i) {
    chunks_in_array[i] = extshape[i] / chunkshape[i];
  }
  int64_t chunks_in_array_strides[rank];
  chunks_in_array_strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    chunks_in_array_strides[i] = chunks_in_array_strides[i + 1] * chunks_in_array[i + 1];
  }

  // Compute the number of chunks to update
  int64_t update_start[rank];
  int64_t update_shape[rank];

  int64_t update_nchunks = 1;
  for (int i = 0; i < rank; ++i) {
    int64_t pos = 0;
    while (pos <= start[i]) {
      pos += chunkshape[i];
    }
    update_start[i] = pos / chunkshape[i] - 1;
    while (pos < stop[i]) {
      pos += chunkshape[i];
    }
    update_shape[i] = pos / chunkshape[i] - update_start[i];
    update_nchunks *= update_shape[i];
  }

  for (int update_nchunk = 0; update_nchunk < update_nchunks; ++update_nchunk) {
    int64_t nchunk_ndim[rank];
    blosc2_unidim_to_multidim(rank, update_shape, update_nchunk, nchunk_ndim);
    for (int i = 0; i < rank; ++i) {
      nchunk_ndim[i] += update_start[i];
    }
    int64_t nchunk;
    blosc2_multidim_to_unidim(nchunk_ndim, rank, chunks_in_array_strides, &nchunk);

    // Check if the chunk needs to be updated
    int64_t chunk_start[rank];
    int64_t chunk_stop[rank];
    int32_t chunksize = typesize;
    for (int i = 0; i < rank; ++i) {
      chunk_start[i] = nchunk_ndim[i] * chunkshape[i];
      chunk_stop[i] = chunk_start[i] + chunkshape[i];
      if (chunk_stop[i] > shape[i]) {
        chunk_stop[i] = shape[i];
      }
      chunksize *= (chunk_stop[i] - chunk_start[i]);
    }

    bool dont_read = false;
    for (int i = 0; i < rank; ++i) {
      dont_read |= (chunk_stop[i] <= start[i] || chunk_start[i] >= stop[i]);
    }
    if (dont_read) {
      continue;
    }

    // Check if all the chunk is going to be updated and avoid the decompression
    bool decompress_chunk = false;
    for (int i = 0; i < rank; ++i) {
      decompress_chunk |= (chunk_start[i] < start[i] || chunk_stop[i] > stop[i]);
    }
    /*
    if (decompress_chunk) {
        int err = blosc2_schunk_decompress_chunk(array->sc, nchunk, data, data_nbytes);
        if (err < 0) {
          BLOSC_TRACE_ERROR("Error decompressing chunk");
          BLOSC_ERROR(BLOSC2_ERROR_FAILURE);
        }
    }else {
    }*/
    if (!decompress_chunk) {
      if (!set) {
        read_chunk_blosc2_ndim(filename, dataset_id, space_id, nchunk, chunk_start, chunk_stop, chunksize, data2);
      }
      else {
        insert_chunk_blosc2_ndim(dataset_id, chunk_start, chunksize, data2);
      }
    }
    data2 += chunksize;
  }

  if (H5Sclose(space_id) < 0)
    return -5;

  return 0;
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


herr_t H5ARRAYOwrite_records(hbool_t blosc2_support,
                             hid_t dataset_id,
                             hid_t type_id,
                             const int rank,
                             hsize_t *start,
                             hsize_t *step,
                             hsize_t *count,
                             const void *data) {
  hid_t space_id;
  hid_t mem_space_id;

  /* Check if the compressor is blosc2 */
  long blosc2_filter = 0;
  char *envvar = getenv("BLOSC2_FILTER");
  if (envvar != NULL)
    blosc2_filter = strtol(envvar, NULL, 10);
  if (blosc2_support && !((int) blosc2_filter)) {
    hsize_t stop[rank];
    for (int i = 0; i < rank; ++i) {
      stop[i] = start[i] + count[i];
    }
    if (get_set_blosc2_slice(NULL, dataset_id, type_id, rank, start, stop, step, data, true) == 0)
      return 0;
  }

  /* Create a simple memory data space */
  if ((mem_space_id = H5Screate_simple(rank, count, NULL)) < 0)
    return -3;

  /* Get the file data space */
  if ((space_id = H5Dget_space(dataset_id)) < 0)
    return -4;

  /* Define a hyperslab in the dataset */
  if (rank != 0 && H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start,
                                       step, count, NULL) < 0)
    return -5;

  if (H5Dwrite(dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data) < 0)
    return -6;

  /* Terminate access to the dataspace */
  if (H5Sclose(mem_space_id) < 0)
    return -7;

  if (H5Sclose(space_id) < 0)
    return -8;

  /* Everything went smoothly */
  return 0;
}


herr_t insert_chunk_blosc2_ndim(hid_t dataset_id,
                                hsize_t *start,
                                hsize_t chunksize,
                                const void *data) {
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
  if (H5Pclose(dcpl) < 0)
    goto out;


  /* Compress data into superchunk and get frame */
  blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
  cparams.typesize = typesize;
  cparams.clevel = cd_values[4];
  cparams.filters[5] = cd_values[5];
  if (cd_nelmts >= 7) {
    cparams.compcode = cd_values[6];
  }

  blosc2_storage storage = {.cparams=&cparams, .dparams=NULL,
    .contiguous=true};
  // b2nd_context_t *ctx = b2nd_create_ctx(const blosc2_storage *b2_storage, int8_t ndim, const int64_t *shape, const int32_t *chunkshape,
  //                const int32_t *blockshape, const char *dtype, int8_t dtype_format, NULL,
  //                0)
  // b2nd_zeros(b2nd_context_t *ctx, b2nd_array_t **array)
  // To remove:
  // blosc2_schunk *sc = blosc2_schunk_new(&storage);
  //  if (sc == NULL) {
  //    BLOSC_TRACE_ERROR("Failed creating superchunk");
  //    goto out;
  //  }
  blosc2_schunk *sc = blosc2_schunk_new(&storage);
  if (sc == NULL) {
    BLOSC_TRACE_ERROR("Failed creating superchunk");
    goto out;
  }

  // b2nd_set_slice_cbuffer(data, const int64_t *buffershape, int64_t buffersize,
  //                                        const int64_t *start, const int64_t *stop, b2nd_array_t *array)
  // To remove:
  //   if (blosc2_schunk_append_buffer(sc, (void *) data, chunksize) <= 0) {
  //    BLOSC_TRACE_ERROR("Failed appending buffer");
  //    goto out;
  //  }
  if (blosc2_schunk_append_buffer(sc, (void *) data, chunksize) <= 0) {
    BLOSC_TRACE_ERROR("Failed appending buffer");
    goto out;
  }
  uint8_t *cframe;
  bool needs_free2;
  // blosc2_schunk_to_buffer(array->sc, &cframe, &needs_free2)
  int64_t cfsize = blosc2_schunk_to_buffer(sc, &cframe, &needs_free2);
  if (cfsize <= 0) {
    BLOSC_TRACE_ERROR("Failed converting schunk to cframe");
    goto out;
  }

  /* Write frame bypassing HDF5 filter pipeline */
  unsigned flt_msk = 0;
  if (H5Dwrite_chunk(dataset_id, H5P_DEFAULT, flt_msk, start, (size_t) cfsize, cframe) < 0) {
    BLOSC_TRACE_ERROR("Failed HDF5 writing chunk");
    goto out;
  }

  /* Free resources */
  if (needs_free2) {
    free(cframe);
  }
  blosc2_schunk_free(sc);

  return 0;

  out:
  return -1;
}


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
 *           Modified by A. Cobb. August 21, 2017 (track_times)
 *
 *-------------------------------------------------------------------------
 */

hid_t H5ARRAYOmake(hid_t loc_id,
                   const char *dset_name,
                   const char *obversion,
                   const int rank,
                   const hsize_t *dims,
                   int extdim,
                   hid_t type_id,
                   hsize_t *dims_chunk,
                   void *fill_data,
                   int compress,
                   char *complib,
                   int shuffle,
                   int fletcher32,
                   hbool_t track_times,
                   hbool_t blosc2_support,
                   const void *data) {

  hid_t dataset_id, space_id;
  hsize_t *maxdims = NULL;
  hid_t plist_id = 0;
  unsigned int cd_values[7];
  int blosc_compcode;
  char *blosc_compname = NULL;
  int chunked = 0;
  int i;

  /* Check whether the array has to be chunked or not */
  if (dims_chunk) {
    chunked = 1;
  }

  if (chunked) {
    maxdims = malloc(rank * sizeof(hsize_t));
    if (!maxdims) return -1;

    for (i = 0; i < rank; i++) {
      if (i == extdim) {
        maxdims[i] = H5S_UNLIMITED;
      } else {
        maxdims[i] = dims[i] < dims_chunk[i] ? dims_chunk[i] : dims[i];
      }
    }
  }

  /* Create the data space for the dataset. */
  if ((space_id = H5Screate_simple(rank, dims, maxdims)) < 0)
    return -1;

  /* Create dataset creation property list with default values */
  plist_id = H5Pcreate(H5P_DATASET_CREATE);

  /* Enable or disable recording dataset times */
  if (H5Pset_obj_track_times(plist_id, track_times) < 0)
    return -1;

  if (chunked) {
    /* Modify dataset creation properties, i.e. enable chunking  */
    if (H5Pset_chunk(plist_id, rank, dims_chunk) < 0)
      return -1;

    /* Set the fill value using a struct as the data type. */
    if (fill_data) {
      if (H5Pset_fill_value(plist_id, type_id, fill_data) < 0)
        return -1;
    } else {
      if (H5Pset_fill_time(plist_id, H5D_FILL_TIME_ALLOC) < 0)
        return -1;
    }

    /*
       Dataset creation property list is modified to use
    */

    /* Fletcher must be first */
    if (fletcher32) {
      if (H5Pset_fletcher32(plist_id) < 0)
        return -1;
    }
    /* Then shuffle (blosc shuffles inplace) */
    if ((shuffle && compress) && (strncmp(complib, "blosc", 5) != 0)) {
      if (H5Pset_shuffle(plist_id) < 0)
        return -1;
    }
    /* Finally compression */
    if (compress) {
      cd_values[0] = compress;
      cd_values[1] = (int) (atof(obversion) * 10);
      if (extdim < 0)
        cd_values[2] = CArray;
      else
        cd_values[2] = EArray;

      /* The default compressor in HDF5 (zlib) */
      if (strcmp(complib, "zlib") == 0) {
        if (H5Pset_deflate(plist_id, compress) < 0)
          return -1;
      }
        /* The Blosc2 compressor does accept parameters */
      else if (strcmp(complib, "blosc2") == 0) {
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        if (H5Pset_filter(plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 6, cd_values) < 0)
          return -1;
      }
        /* The Blosc2 compressor can use other compressors */
      else if (strncmp(complib, "blosc2:", 7) == 0) {
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        blosc_compname = complib + 7;
        blosc_compcode = blosc2_compname_to_compcode(blosc_compname);
        cd_values[6] = blosc_compcode;
        if (H5Pset_filter(plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 7, cd_values) < 0)
          return -1;
      }
        /* The Blosc compressor does accept parameters */
      else if (strcmp(complib, "blosc") == 0) {
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        if (H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 6, cd_values) < 0)
          return -1;
      }
        /* The Blosc compressor can use other compressors */
      else if (strncmp(complib, "blosc:", 6) == 0) {
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        blosc_compname = complib + 6;
        blosc_compcode = blosc_compname_to_compcode(blosc_compname);
        cd_values[6] = blosc_compcode;
        if (H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 7, cd_values) < 0)
          return -1;
      }
        /* The LZO compressor does accept parameters */
      else if (strcmp(complib, "lzo") == 0) {
        if (H5Pset_filter(plist_id, FILTER_LZO, H5Z_FLAG_OPTIONAL, 3, cd_values) < 0)
          return -1;
      }
        /* The bzip2 compress does accept parameters */
      else if (strcmp(complib, "bzip2") == 0) {
        if (H5Pset_filter(plist_id, FILTER_BZIP2, H5Z_FLAG_OPTIONAL, 3, cd_values) < 0)
          return -1;
      } else {
        /* Compression library not supported */
        fprintf(stderr, "Compression library not supported\n");
        return -1;
      }
    }

    /* Create the (chunked) dataset */
    if ((dataset_id = H5Dcreate(loc_id, dset_name, type_id, space_id,
                                H5P_DEFAULT, plist_id, H5P_DEFAULT)) < 0)
      goto out;
  } else {         /* Not chunked case */
    /* Create the dataset. */
    if ((dataset_id = H5Dcreate(loc_id, dset_name, type_id, space_id,
                                H5P_DEFAULT, plist_id, H5P_DEFAULT)) < 0)
      goto out;
  }

  /* Write the dataset only if there is data to write */

  if (data) {
    if (H5Dwrite(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
      goto out;
  }

  /* Terminate access to the data space. */
  if (H5Sclose(space_id) < 0)
    return -1;

  /* End access to the property list */
  if (plist_id)
    if (H5Pclose(plist_id) < 0)
      goto out;

  /* Release resources */
  if (maxdims)
    free(maxdims);

  return dataset_id;

  out:
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  if (maxdims)
    free(maxdims);
  if (dims_chunk)
    free(dims_chunk);
  return -1;

}





herr_t read_chunk_blosc2_ndim(char *filename,
                              hid_t dataset_id,
                              hid_t space_id,
                              hsize_t nchunk,
                              hsize_t *chunk_start,
                              hsize_t *chunk_stop,
                              hsize_t chunksize,
                              uint8_t *data) {
  /* Get the address of the schunk on-disk */
  unsigned flt_msk;
  haddr_t address;
  hsize_t cframe_size;
  hsize_t chunk_offset[2];
  if (H5Dget_chunk_info(dataset_id, space_id, nchunk, chunk_offset, &flt_msk,
                        &address, &cframe_size) < 0) {
    BLOSC_TRACE_ERROR("Get chunk info failed!\n");
    goto out;
  }

  /* Open the schunk on-disk */
  // b2nd_open_offset(const char *urlpath, b2nd_array_t **array, (int64_t) address)
  // To remove:
  //   blosc2_schunk *schunk = blosc2_schunk_open_offset(filename, (int64_t) address);
  //  if (schunk == NULL) {
  //    BLOSC_TRACE_ERROR("Cannot open schunk in %s\n", filename);
  //    goto out;
  //  }
  blosc2_schunk *schunk = blosc2_schunk_open_offset(filename, (int64_t) address);
  if (schunk == NULL) {
    BLOSC_TRACE_ERROR("Cannot open schunk in %s\n", filename);
    goto out;
  }

  //  b2nd_set_slice_cbuffer(data, const int64_t *buffershape, int64_t buffersize,
  // const int64_t *start, stop, b2nd_array_t *array)

  /* Get chunk */
  bool needs_free;
  uint8_t *chunk;
  int32_t cbytes = blosc2_schunk_get_lazychunk(schunk, 0, &chunk, &needs_free);
  if (cbytes < 0) {
    BLOSC_TRACE_ERROR("Cannot get lazy chunk %zd in %s\n", nchunk, filename);
    goto out;
  }

  blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
  dparams.schunk = schunk;
  blosc2_context *dctx = blosc2_create_dctx(dparams);

  /* Gather data for the interesting part */
  //int nrecords_chunk = chunksize - chunk_start;

  int rbytes;
  //if (nrecords_chunk == chunklen) {
  rbytes = blosc2_decompress_ctx(dctx, chunk, cbytes, data, chunksize);
  if (rbytes < 0) {
    BLOSC_TRACE_ERROR("Cannot decompress lazy chunk");
    goto out;
  }
  //}
  /*else {
    /* Less than 1 chunk to read; use a getitem call
    rbytes = blosc2_getitem_ctx(dctx, chunk, cbytes, start_chunk, nrecords_chunk, data, chunksize);
    if (rbytes != nrecords_chunk * typesize) {
      BLOSC_TRACE_ERROR("Cannot get (all) items for lazychunk\n");
      goto out;
    }
  }*/


  if (needs_free) {
    free(chunk);
  }
  blosc2_free_ctx(dctx);
  blosc2_schunk_free(schunk);

  return 0;

  out:
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

herr_t H5ARRAYOreadSlice(char *filename,
                         hbool_t blosc2_support,
                         hid_t dataset_id,
                         hid_t type_id,
                         hsize_t *start,
                         hsize_t *stop,
                         hsize_t *step,
                         void *data) {
  hid_t space_id;
  hid_t mem_space_id;
  hsize_t *stride = (hsize_t *) step;
  hsize_t *offset = (hsize_t *) start;
  int rank;
  int i;

  /* Get the dataspace handle */
  if ((space_id = H5Dget_space(dataset_id)) < 0)
    return -1;

  /* Get the rank */
  if ((rank = H5Sget_simple_extent_ndims(space_id)) < 0)
    return -2;

  hsize_t dims[rank];
  hsize_t count[rank];

  if (rank) {                    /* Array case */
    /* Get dataset dimensionality */
    if (H5Sget_simple_extent_dims(space_id, dims, NULL) < 0)
      return -3;

    for (i = 0; i < rank; i++) {
      if (stop[i] > dims[i]) {
        printf("Asking for a range of rows exceeding the available ones!.\n");
        return -4;
      }
      if (step[i] != 1) {
        blosc2_support = false;
      }
    }

    long blosc2_filter = 0;
    char *envvar = getenv("BLOSC2_FILTER");
    if (envvar != NULL)
      blosc2_filter = strtol(envvar, NULL, 10);

    if (blosc2_support && !((int) blosc2_filter)) {
      /* Try to read using blosc2 (only supports native byteorder and step=1 for now) */
      get_set_blosc2_slice(filename, dataset_id, type_id, rank, start, stop, step, data, false);
      goto success;
    }

    /* Define a hyperslab in the dataset of the size of the records */
    if (H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride,
                            count, NULL) < 0)
      return -5;

    /* Create a memory dataspace handle */
    if ((mem_space_id = H5Screate_simple(rank, count, NULL)) < 0)
      return -6;

    /* Read */
    if (H5Dread(dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT,
                data) < 0)
      return -7;

    /* Terminate access to the memory dataspace */
    if (H5Sclose(mem_space_id) < 0)
      return -8;
  } else {                     /* Scalar case */

    /* Read all the dataset */
    if (H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
      return -9;
  }

  success:
  /* Terminate access to the dataspace */
  if (H5Sclose(space_id) < 0){
    return -10;
  }

  return 0;
}




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
