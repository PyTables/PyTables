#include "H5ARRAY-opt.h"

#include "tables.h"
#include "utils.h"
#include "H5Zlzo.h"                    /* Import FILTER_LZO */
#include "H5Zbzip2.h"                  /* Import FILTER_BZIP2 */
#include "blosc_filter.h"              /* Import FILTER_BLOSC */
#include "blosc2_filter.h"             /* Import FILTER_BLOSC2 */
#include "b2nd.h"

#include <stdlib.h>
#include <string.h>


/* Error handling in this module:
 *
 * 1. Declare "retval" with default error value < 0.
 * 2. Declare "hid_t" variables with initial -1 value
 *    (to avoid closing uninitialized identifiers on cleanup).
 * 3. Close HDF5 objects with error detection when no longer needed.
 * 4. Go to "out" (cleanup) on error; set "retval" < 0 before for optional finer reporting.
 * 5. Reach end of function on success (sets non-error "retval" before cleanup).
 * 6. In all cases, normal resources (allocated arrays & Blosc2 objects) are freed on cleanup.
 * 7. On error ("retval" < 0), HDF5 objects are closed without error detection on cleanup.
 */

/* All these macros depend on an "out" tag for cleanup. */
#define IF_TRUE_OUT(_COND) { if (_COND) goto out; }
#define IF_FALSE_OUT(_COND) IF_TRUE_OUT(!(_COND))
#define IF_NEG_OUT(_EXPR) IF_TRUE_OUT((_EXPR) < 0)

#define IF_TRUE_OUT_DO(_COND, _STMT) { if (_COND) { _STMT; goto out; } }
#define IF_FALSE_OUT_DO(_COND, _STMT) IF_TRUE_OUT_DO(!(_COND), _STMT)
#define IF_NEG_OUT_DO(_EXPR, _STMT) IF_TRUE_OUT_DO((_EXPR) < 0, _STMT)

#define IF_TRUE_OUT_BTRACE(_COND, _MESG, ...) \
  IF_TRUE_OUT_DO(_COND, BLOSC_TRACE_ERROR(_MESG, ##__VA_ARGS__))
#define IF_FALSE_OUT_BTRACE(_COND, _MESG, ...) \
  IF_TRUE_OUT_DO(!(_COND), BLOSC_TRACE_ERROR(_MESG, ##__VA_ARGS__))
#define IF_NEG_OUT_BTRACE(_EXPR, _MESG, ...) \
  IF_TRUE_OUT_DO((_EXPR) < 0, BLOSC_TRACE_ERROR(_MESG, ##__VA_ARGS__))

/* These also depend on a "retval" variable of the return type. */
#define IF_TRUE_OUT_RET(_COND, _RETVAL) IF_TRUE_OUT_DO(_COND, retval = (_RETVAL))
#define IF_FALSE_OUT_RET(_COND, _RETVAL) IF_TRUE_OUT_RET(!(_COND), _RETVAL)
#define IF_NEG_OUT_RET(_EXPR, _RETVAL) IF_TRUE_OUT_RET((_EXPR) < 0, _RETVAL)

herr_t read_chunk_blosc2_ndim(const char *filename,
                              hid_t dataset_id,
                              hid_t space_id,
                              hsize_t nchunk,
                              const int64_t *chunkshape,
                              const int64_t *start,
                              const int64_t *stop,
                              hsize_t chunksize,
                              uint8_t *data);


// A return value < -100 means that data may have been altered.
herr_t get_set_blosc2_slice(char *filename, // NULL means write, read otherwise
                          hid_t dataset_id,
                          hid_t type_id,
                          const int rank,
                          hsize_t *start,
                          hsize_t *stop,
                          const void *data)
{
  herr_t retval = -1;
  hid_t space_id = -1;
  hid_t dcpl = -1;

  /* All these have "rank" elements; remember to free them in the "out" block at the end. */
  hsize_t *chunkshape = NULL;
  hsize_t *shape = NULL;
  int64_t *slice_shape = NULL;
  int64_t *chunks_in_array = NULL;
  int64_t *data_strides = NULL;
  int64_t *chunks_in_array_strides = NULL;
  int32_t *blockshape = NULL;
  int64_t *update_start = NULL;
  int64_t *update_shape = NULL;
  int64_t *nchunk_ndim = NULL;
  hsize_t *chunk_start = NULL;
  hsize_t *chunk_stop = NULL;
  int64_t *temp_chunk_shape = NULL;
  int64_t *temp_chunk_strides = NULL;
  int64_t *start_in_temp_chunk = NULL;
  int64_t *stop_in_temp_chunk = NULL;
  uint8_t *temp_chunk = NULL;
  int64_t *slice_chunk_shape = NULL;
  int64_t *slice_chunk_start = NULL;
  int64_t *slice_chunk_stop = NULL;

  /* Get the file data space */
  IF_NEG_OUT_RET(space_id = H5Dget_space(dataset_id), -1);

  /* Get the dataset creation property list */
  dcpl = H5Dget_create_plist(dataset_id);
  IF_TRUE_OUT_RET(dcpl == H5I_INVALID_HID, -2);

  size_t cd_nelmts = 7;
  unsigned cd_values[7];
  char name[7];  // "blosc2\0", unused
  IF_NEG_OUT_RET(H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL,
                                      &cd_nelmts, cd_values, 7, name, NULL), -3);
  int typesize = cd_values[2];
  chunkshape = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  H5Pget_chunk(dcpl, rank, chunkshape);

  IF_NEG_OUT_RET(H5Pclose(dcpl), -4);

  shape = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  H5Sget_simple_extent_dims(space_id, shape, NULL);

  chunks_in_array = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in chunks
  slice_shape = malloc(sizeof(int64_t) * rank);
  for (int i = 0; i < rank; ++i) {
    chunks_in_array[i] = ((shape[i] / chunkshape[i])
                          + ((shape[i] % chunkshape[i]) ? 1 : 0));
    slice_shape[i] = stop[i] - start[i];
  }

  /* Compute data and chunk strides */
  data_strides = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  chunks_in_array_strides = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in chunks
  data_strides[rank - 1] = 1;
  chunks_in_array_strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    data_strides[i] = data_strides[i + 1] * slice_shape[i + 1];
    chunks_in_array_strides[i] = chunks_in_array_strides[i + 1] * chunks_in_array[i + 1];
  }

  blosc2_cparams cparams;
  if (!filename) {  // write
    /* Compute some Blosc2-specific parameters */
    cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = typesize;
    cparams.clevel = cd_values[4];
    cparams.filters[5] = cd_values[5];
    if (cd_nelmts >= 7) {
      cparams.compcode = cd_values[6];
    }

    int32_t chunkshape_i[BLOSC2_MAX_DIM];
    for (int i = 0; i < rank; i++) chunkshape_i[i] = chunkshape[i];
    blockshape = (int32_t *)(malloc(rank * sizeof(int32_t)));  // in items
    cparams.blocksize = compute_b2nd_block_shape(cd_values[1], typesize,
                                                 rank, chunkshape_i, blockshape);
  }

  /* Compute the number of chunks to update */
  update_start = (int64_t *)(malloc(rank * sizeof(int64_t)));  // chunk index
  update_shape = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in chunks
  int64_t update_nchunks = 1;
  for (int i = 0; i < rank; ++i) {
    hsize_t pos = 0;
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

  /* These dimension arrays are completely rewritten on each iteration */
  nchunk_ndim = (int64_t *)(malloc(rank * sizeof(int64_t)));  // chunk index
  chunk_start = (hsize_t *)(malloc(rank * sizeof(hsize_t)));  // in items
  chunk_stop = (hsize_t *)(malloc(rank * sizeof(hsize_t)));  // in items
  temp_chunk_shape = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  temp_chunk_strides = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  start_in_temp_chunk = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  stop_in_temp_chunk = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  slice_chunk_shape = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  slice_chunk_start = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  slice_chunk_stop = (int64_t *)(malloc(rank * sizeof(int64_t)));  // in items
  for (int update_nchunk = 0; update_nchunk < update_nchunks; ++update_nchunk) {
    blosc2_unidim_to_multidim(rank, update_shape, update_nchunk, nchunk_ndim);
    for (int i = 0; i < rank; ++i) {
      nchunk_ndim[i] += update_start[i];
    }
    int64_t nchunk;
    blosc2_multidim_to_unidim(nchunk_ndim, rank, chunks_in_array_strides, &nchunk);

    /* Check if the chunk needs to be updated */
    hsize_t chunk_slice_size = typesize;  // in bytes
    bool slice_overlaps_chunk = true;
    for (int i = 0; i < rank; ++i) {
      chunk_start[i] = nchunk_ndim[i] * chunkshape[i];
      chunk_stop[i] = chunk_start[i] + chunkshape[i];
      if (chunk_stop[i] > shape[i]) {
        chunk_stop[i] = shape[i];
      }
      if (start[i] > chunk_start[i]) {
        slice_chunk_start[i] = 0;
      }
      else {
        slice_chunk_start[i] = chunk_start[i] - start[i];
      }
      if (stop[i] < chunk_stop[i]) {
            slice_chunk_stop[i] = stop[i] - start[i];
      }
      else {
        slice_chunk_stop[i] = chunk_stop[i] - start[i];
      }

      slice_overlaps_chunk &= (start[i] < chunk_stop[i] && chunk_start[i] < stop[i]);

      temp_chunk_shape[i] = chunk_stop[i] - chunk_start[i];

      start_in_temp_chunk[i] = (start[i] > chunk_start[i])
        ? (int64_t)(start[i] - chunk_start[i])
        : 0;
      stop_in_temp_chunk[i] = (stop[i] < chunk_stop[i])
        ? (int64_t)(stop[i] - chunk_start[i])
        : temp_chunk_shape[i];

      slice_chunk_shape[i] = slice_chunk_stop[i] - slice_chunk_start[i];
      chunk_slice_size *= slice_chunk_shape[i];
    }

    temp_chunk_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
      temp_chunk_strides[i] = temp_chunk_strides[i + 1] * slice_chunk_shape[i + 1];
    }

    if (!slice_overlaps_chunk) {
      continue;  // no overlap between chunk and slice
    }

    assert(temp_chunk == NULL);  // previous temp chunk must have been freed
    temp_chunk = (uint8_t *)(malloc(chunk_slice_size * sizeof(uint8_t)));

    herr_t rv;
    IF_NEG_OUT_RET(rv = read_chunk_blosc2_ndim(filename, dataset_id, space_id,
                                             nchunk, slice_chunk_shape,
                                             start_in_temp_chunk, stop_in_temp_chunk,
                                             chunk_slice_size, temp_chunk),
                    rv - 50);

    /* Copy from temp_chunk to data */
    int64_t chunk_start_idx = -1;
    blosc2_multidim_to_unidim((int64_t*)(slice_chunk_start), rank, data_strides, &chunk_start_idx);
    uint8_t *chunk_line = (uint8_t*)(data) + (chunk_start_idx * typesize);
    uint8_t *temp_chunk_line = temp_chunk;
    for (int i = slice_chunk_start[0]; i < slice_chunk_stop[0]; i++) {
        /* As the temporary chunk has no other chunks around it,
        its main stride is the number of items to be copied per chunk line. */
        memcpy(chunk_line, temp_chunk_line, temp_chunk_strides[0] * typesize);

        chunk_line += data_strides[0] * typesize;
        temp_chunk_line += temp_chunk_strides[0] * typesize;
    }

    assert(temp_chunk);
    free(temp_chunk);
    temp_chunk = NULL;
  }

  IF_NEG_OUT_RET(H5Sclose(space_id), -8);

  //out_success:
  retval = 0;

  out:
  if (temp_chunk) free(temp_chunk);
  if (stop_in_temp_chunk) free(stop_in_temp_chunk);
  if (start_in_temp_chunk) free(start_in_temp_chunk);
  if (temp_chunk_strides) free(temp_chunk_strides);
  if (temp_chunk_shape) free(temp_chunk_shape);
  if (chunk_stop) free(chunk_stop);
  if (chunk_start) free(chunk_start);
  if (nchunk_ndim) free(nchunk_ndim);
  if (update_shape) free(update_shape);
  if (update_start) free(update_start);
  if (blockshape) free(blockshape);
  if (chunks_in_array_strides) free(chunks_in_array_strides);
  if (data_strides) free(data_strides);
  if (chunks_in_array) free(chunks_in_array);
  if (shape) free(shape);
  if (chunkshape) free(chunkshape);
  if (slice_chunk_shape) free(slice_chunk_shape);
  if (slice_shape) free(slice_shape);
  if (slice_chunk_start) free(slice_chunk_start);
  if (slice_chunk_stop) free(slice_chunk_stop);
  if (retval >= 0)
    return retval;

  H5E_BEGIN_TRY {
    H5Pclose(dcpl);
    H5Sclose(space_id);
  } H5E_END_TRY;
  return retval;
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
                   hsize_t block_size,
                   void *fill_data,
                   int compress,
                   char *complib,
                   int shuffle,
                   int fletcher32,
                   hbool_t track_times,
                   const void *data) {

  hid_t retval = -1;
  hid_t dataset_id = -1;
  hid_t space_id = -1;
  hid_t plist_id = -1;
  hsize_t *maxdims = NULL;

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
    maxdims = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
    IF_FALSE_OUT(maxdims);

    for (i = 0; i < rank; i++) {
      if (i == extdim) {
        maxdims[i] = H5S_UNLIMITED;
      } else {
        maxdims[i] = dims[i] < dims_chunk[i] ? dims_chunk[i] : dims[i];
      }
    }
  }

  /* Create the data space for the dataset. */
  IF_NEG_OUT(space_id = H5Screate_simple(rank, dims, maxdims));

  /* Create dataset creation property list with default values */
  plist_id = H5Pcreate(H5P_DATASET_CREATE);

  /* Enable or disable recording dataset times */
  IF_NEG_OUT(H5Pset_obj_track_times(plist_id, track_times));

  if (chunked) {
    /* Modify dataset creation properties, i.e. enable chunking  */
    IF_NEG_OUT(H5Pset_chunk(plist_id, rank, dims_chunk));

    /* Set the fill value using a struct as the data type. */
    if (fill_data) {
      IF_NEG_OUT(H5Pset_fill_value(plist_id, type_id, fill_data));
    } else {
      IF_NEG_OUT(H5Pset_fill_time(plist_id, H5D_FILL_TIME_ALLOC));
    }

    /*
       Dataset creation property list is modified to use
    */

    /* Fletcher must be first */
    if (fletcher32) {
      IF_NEG_OUT(H5Pset_fletcher32(plist_id));
    }
    /* Then shuffle (blosc shuffles inplace) */
    if ((shuffle && compress) && (strncmp(complib, "blosc", 5) != 0)) {
      IF_NEG_OUT(H5Pset_shuffle(plist_id));
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
        IF_NEG_OUT(H5Pset_deflate(plist_id, compress));
      }
      /* The Blosc2 compressor does accept parameters (see blosc2_filter.c) */
      else if (strcmp(complib, "blosc2") == 0) {
        size_t type_size = H5Tget_size(type_id);
        IF_NEG_OUT(type_size);
        int32_t dims_chunk_i[BLOSC2_MAX_DIM], dims_block[BLOSC2_MAX_DIM];
        for (int i = 0; i < rank; i++) dims_chunk_i[i] = dims_chunk[i];
        cd_values[1] = compute_b2nd_block_shape(block_size, type_size,
                                                rank, dims_chunk_i, dims_block);
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 6, cd_values));
      }
      /* The Blosc2 compressor can use other compressors (see blosc2_filter.c) */
      else if (strncmp(complib, "blosc2:", 7) == 0) {
        size_t type_size = H5Tget_size(type_id);
        IF_NEG_OUT(type_size);
        int32_t dims_chunk_i[BLOSC2_MAX_DIM], dims_block[BLOSC2_MAX_DIM];
        for (int i = 0; i < rank; i++) dims_chunk_i[i] = dims_chunk[i];
        cd_values[1] = compute_b2nd_block_shape(block_size, type_size,
                                                rank, dims_chunk_i, dims_block);
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        blosc_compname = complib + 7;
        blosc_compcode = blosc2_compname_to_compcode(blosc_compname);
        cd_values[6] = blosc_compcode;
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 7, cd_values));
      }
        /* The Blosc compressor does accept parameters */
      else if (strcmp(complib, "blosc") == 0) {
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 6, cd_values));
      }
        /* The Blosc compressor can use other compressors */
      else if (strncmp(complib, "blosc:", 6) == 0) {
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        blosc_compname = complib + 6;
        blosc_compcode = blosc_compname_to_compcode(blosc_compname);
        cd_values[6] = blosc_compcode;
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 7, cd_values));
      }
        /* The LZO compressor does accept parameters */
      else if (strcmp(complib, "lzo") == 0) {
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_LZO, H5Z_FLAG_OPTIONAL, 3, cd_values));
      }
        /* The bzip2 compress does accept parameters */
      else if (strcmp(complib, "bzip2") == 0) {
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_BZIP2, H5Z_FLAG_OPTIONAL, 3, cd_values));
      } else {
        /* Compression library not supported */
        IF_TRUE_OUT_DO(true, fprintf(stderr, "Compression library not supported\n"));
      }
    }

    /* Create the (chunked) dataset */
    IF_NEG_OUT(dataset_id = H5Dcreate(loc_id, dset_name, type_id, space_id,
                                      H5P_DEFAULT, plist_id, H5P_DEFAULT));
  } else {         /* Not chunked case */
    /* Create the dataset. */
    IF_NEG_OUT(dataset_id = H5Dcreate(loc_id, dset_name, type_id, space_id,
                                      H5P_DEFAULT, plist_id, H5P_DEFAULT));
  }

  /* Write the dataset only if there is data to write */

  if (data) {
    IF_NEG_OUT(H5Dwrite(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
  }

  /* Terminate access to the data space. */
  IF_NEG_OUT(H5Sclose(space_id));

  /* End access to the property list */
  if (plist_id)
    IF_NEG_OUT(H5Pclose(plist_id));

  //out_success:
  retval = dataset_id;

  out:
  if (maxdims) free(maxdims);
  if (retval >= 0)
    return retval;

  H5E_BEGIN_TRY {
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Pclose(plist_id);
  } H5E_END_TRY;
  return retval;

}


herr_t read_chunk_blosc2_ndim(const char *filename,
                              hid_t dataset_id,
                              hid_t space_id,
                              hsize_t nchunk,
                              const int64_t *chunkshape,
                              const int64_t *start,
                              const int64_t *stop,
                              hsize_t chunksize,
                              uint8_t *data) {
  herr_t retval = -1;
  b2nd_array_t *array = NULL;

  /* Get the address of the schunk on-disk */
  unsigned flt_msk;
  haddr_t address;
  hsize_t cframe_size;
  hsize_t chunk_offset;
  IF_NEG_OUT_BTRACE(H5Dget_chunk_info(dataset_id, space_id, nchunk, &chunk_offset, &flt_msk,
                                      &address, &cframe_size),
                    "Failed getting chunk info of array in %s", filename);

  /* Open the schunk on-disk */
  IF_TRUE_OUT_BTRACE(b2nd_open_offset(filename, &array, address) != BLOSC2_ERROR_SUCCESS,
                     "Cannot open array in %s", filename);

  IF_TRUE_OUT_BTRACE(b2nd_get_slice_cbuffer(array, start, stop,
                                            data, chunkshape, chunksize) != BLOSC2_ERROR_SUCCESS,
                     "Failed getting slice of array in %s", filename);

  //out_success:
  retval = 0;

  out:
  if (array) b2nd_free(array);
  return retval;


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
  herr_t retval = -1;
  hid_t space_id = -1;
  hid_t mem_space_id = -1;
  hsize_t *dims = NULL;
  hsize_t *count = NULL;

  hsize_t *stride = (hsize_t *) step;
  hsize_t *offset = (hsize_t *) start;
  int rank;
  int i;

  /* Get the dataspace handle */
  IF_NEG_OUT_RET(space_id = H5Dget_space(dataset_id), -1);

  /* Get the rank */
  IF_NEG_OUT_RET(rank = H5Sget_simple_extent_ndims(space_id), -2);

  if (rank) {                    /* Array case */
    dims = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
    count = (hsize_t *)(malloc(rank * sizeof(hsize_t)));

    /* Get dataset dimensionality */
    IF_NEG_OUT_RET(H5Sget_simple_extent_dims(space_id, dims, NULL), -3);

    for (i = 0; i < rank; i++) {
      count[i] = get_len_of_range(start[i], stop[i], step[i]);
      if (stop[i] > dims[i]) {
        printf("Asking for a range of rows exceeding the available ones!.\n");
        IF_TRUE_OUT_RET(true, -4);
      }
      if (step[i] != 1) {
        blosc2_support = false;  // Blosc2 only supports step=1 for now
      }
    }

    /* Check if the compressor is Blosc2 */
    long blosc2_filter = 0;
    char *envvar = getenv("BLOSC2_FILTER");
    if (envvar != NULL)
      blosc2_filter = strtol(envvar, NULL, 10);

    if (blosc2_support && !((int) blosc2_filter)) {
      herr_t rv = get_set_blosc2_slice(filename, dataset_id, type_id,
                                       rank, start, stop, data);
      if (rv >= 0) {
        IF_NEG_OUT_RET(H5Sclose(space_id), -5);
        goto out_success;
      }
      assert(rv >= -100);
      /* Attempt non-optimized read since Blosc2 still has some limitations,
         and operations up to here should not have altered data */
    }

    /* Define a hyperslab in the dataset of the size of the records */
    IF_NEG_OUT_RET(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride,
                                       count, NULL), -6);

    /* Create a memory dataspace handle */
    IF_NEG_OUT_RET(mem_space_id = H5Screate_simple(rank, count, NULL), -7);

    /* Read */
    IF_NEG_OUT_RET(H5Dread(dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT,
                           data), -8);

    /* Terminate access to the memory dataspace */
    IF_NEG_OUT_RET(H5Sclose(mem_space_id), -9);

    /* Terminate access to the dataspace */
    IF_NEG_OUT_RET(H5Sclose(space_id), -10);
  } else {                     /* Scalar case */

    /* Read all the dataset */
    IF_NEG_OUT_RET(H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), -11);
  }

  out_success:
  retval = 0;

  out:
  if (count) free(count);
  if (dims) free(dims);
  if (retval >= 0)
    return retval;

  H5E_BEGIN_TRY {
    H5Sclose(mem_space_id);
    H5Sclose(space_id);
  } H5E_END_TRY;
  return retval;
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
 herr_t   retval = -1;
 hid_t    space_id = -1;
 hid_t    mem_space_id = -1;

 hsize_t  count[2];
 int      rank = 2;
 hsize_t  offset[2];
 hsize_t  stride[2] = {1, 1};


 count[0] = 1;
 count[1] = stop - start;
 offset[0] = irow;
 offset[1] = start;

 /* Get the dataspace handle */
 IF_NEG_OUT(space_id = H5Dget_space( dataset_id ));

 /* Create a memory dataspace handle */
 IF_NEG_OUT(mem_space_id = H5Screate_simple( rank, count, NULL ));

 /* Define a hyperslab in the dataset of the size of the records */
 IF_NEG_OUT(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL));

 /* Read */
 IF_NEG_OUT(H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ));

 /* Terminate access to the memory dataspace */
 IF_NEG_OUT(H5Sclose( mem_space_id ));

 /* Terminate access to the dataspace */
 IF_NEG_OUT(H5Sclose( space_id ));

 //out_success:
 retval = 0;

 out:
 if (retval >= 0)
   return retval;

 H5E_BEGIN_TRY {
   H5Sclose(mem_space_id);
   H5Sclose(space_id);
 } H5E_END_TRY;
 return retval;

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
 herr_t   retval = -1;
 hid_t    space_id = -1;

 int      rank = 2;
 hsize_t  count2[2] = {1, count};

 /* Get the dataspace handle */
 IF_NEG_OUT(space_id = H5Dget_space(dataset_id ));

 /* Create a memory dataspace handle */
 IF_NEG_OUT(*mem_space_id = H5Screate_simple(rank, count2, NULL));

 /* Terminate access to the dataspace */
 IF_NEG_OUT(H5Sclose( space_id ));

 //out_success:
 retval = 0;

 out:
 if (retval >= 0)
   return retval;

 H5E_BEGIN_TRY {
   H5Sclose(space_id);
 } H5E_END_TRY;
 return retval;

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
 herr_t   retval = -1;
 hid_t    space_id = -1;

 hsize_t  count[2] = {1, stop-start};
 hsize_t  offset[2] = {irow, start};
 hsize_t  stride[2] = {1, 1};

 /* Get the dataspace handle */
 IF_NEG_OUT(space_id = H5Dget_space(dataset_id));

 /* Define a hyperslab in the dataset of the size of the records */
 IF_NEG_OUT(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL));

 /* Read */
 IF_NEG_OUT(H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ));

 /* Terminate access to the dataspace */
 IF_NEG_OUT(H5Sclose( space_id ));

 //out_success:
 retval = 0;

 out:
 if (retval >= 0)
   return retval;

 H5E_BEGIN_TRY {
   H5Sclose(space_id);
 } H5E_END_TRY;
 return retval;

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
 herr_t   retval = -1;
 hid_t    space_id = -1;
 hsize_t  count[2] = {1, stop-start};
 hsize_t  offset[2] = {irow, start};
 hsize_t  stride[2] = {1, 1};

 /* Get the dataspace handle */
 IF_NEG_OUT(space_id = H5Dget_space(dataset_id));

 /* Define a hyperslab in the dataset of the size of the records */
 IF_NEG_OUT(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL));

 /* Read */
 IF_NEG_OUT(H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ));

 /* Terminate access to the dataspace */
 IF_NEG_OUT(H5Sclose( space_id ));

 //out_success:
 retval = 0;

 out:
 if (retval >= 0)
   return retval;

 H5E_BEGIN_TRY {
   H5Sclose(space_id);
 } H5E_END_TRY;
 return retval;

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
 herr_t   retval = -1;
 hid_t    space_id = -1;
 hid_t    mem_space_id = -1;

 hsize_t  count[1] = {stop - start};
 hsize_t  stride[1] = {1};
 hsize_t  offset[1] = {start};

  /* Get the dataspace handle */
 IF_NEG_OUT(space_id = H5Dget_space(dataset_id));

 /* Define a hyperslab in the dataset of the size of the records */
 IF_NEG_OUT(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride, count, NULL));

 /* Create a memory dataspace handle */
 IF_NEG_OUT(mem_space_id = H5Screate_simple(1, count, NULL));

 /* Read */
 IF_NEG_OUT(H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ));

 /* Release resources */

 /* Terminate access to the memory dataspace */
 IF_NEG_OUT(H5Sclose( mem_space_id ));

 /* Terminate access to the dataspace */
 IF_NEG_OUT(H5Sclose( space_id ));

 //out_success:
 retval = 0;

 out:
 if (retval >= 0)
   return retval;

 H5E_BEGIN_TRY {
   H5Sclose(mem_space_id);
   H5Sclose(space_id);
 } H5E_END_TRY;
 return retval;

}
