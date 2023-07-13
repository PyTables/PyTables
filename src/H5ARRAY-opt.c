#include "H5ARRAY-opt.h"

#include "tables.h"
#include "utils.h"
#include "H5Zlzo.h"                    /* Import FILTER_LZO */
#include "H5Zbzip2.h"                  /* Import FILTER_BZIP2 */
#include "blosc_filter.h"              /* Import FILTER_BLOSC */
#include "blosc2_filter.h"             /* Import FILTER_BLOSC2 */

#include <stdlib.h>
#include <string.h>


/* 128KiB should let both the decompressed and the compressed blocks fit in
   the L2 cache of most current CPUs. */
#define BLOSC2_DEFAULT_BLOCK_SIZE (1 << 17)

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
  int64_t *extshape = NULL;
  int64_t *chunks_in_array = NULL;
  int64_t *chunks_in_array_strides = NULL;
  int64_t *update_start = NULL;
  int64_t *update_shape = NULL;
  int64_t *nchunk_ndim = NULL;
  hsize_t *chunk_start = NULL;
  hsize_t *chunk_stop = NULL;

  uint8_t *data2 = (uint8_t *) data;

  /* Get the file data space */
  IF_NEG_OUT_RET(space_id = H5Dget_space(dataset_id), -1);

  /* Get the dataset creation property list */
  dcpl = H5Dget_create_plist(dataset_id);
  IF_TRUE_OUT_RET(dcpl == H5I_INVALID_HID, -2);
  size_t cd_nelmts = 7;
  unsigned cd_values[7];
  char name[7];
  IF_NEG_OUT_RET(H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL,
                                      &cd_nelmts, cd_values, 7, name, NULL), -3);
  int typesize = cd_values[2];
  chunkshape = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  H5Pget_chunk(dcpl, rank, chunkshape);

  IF_NEG_OUT_RET(H5Pclose(dcpl), -4);

  shape = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  H5Sget_simple_extent_dims(space_id, shape, NULL);
  extshape = (int64_t *)(malloc(rank * sizeof(int64_t)));

  for (int i = 0; i < rank; i++) {
    if (shape[i] % chunkshape[i] != 0) {
      extshape[i] = shape[i] + chunkshape[i] - shape[i] % chunkshape[i];
    } else {
      extshape[i] = shape[i];
    }
  }

  chunks_in_array = (int64_t *)(malloc(rank * sizeof(int64_t)));
  for (int i = 0; i < rank; ++i) {
    chunks_in_array[i] = extshape[i] / chunkshape[i];
  }
  chunks_in_array_strides = (int64_t *)(malloc(rank * sizeof(int64_t)));
  chunks_in_array_strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    chunks_in_array_strides[i] = chunks_in_array_strides[i + 1] * chunks_in_array[i + 1];
  }

  // Compute the number of chunks to update
  update_start = (int64_t *)(malloc(rank * sizeof(int64_t)));
  update_shape = (int64_t *)(malloc(rank * sizeof(int64_t)));

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

  /* These dimension arrays are completely rewritten on each iteration. */
  nchunk_ndim = (int64_t *)(malloc(rank * sizeof(int64_t)));
  chunk_start = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  chunk_stop = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  for (int update_nchunk = 0; update_nchunk < update_nchunks; ++update_nchunk) {
    blosc2_unidim_to_multidim(rank, update_shape, update_nchunk, nchunk_ndim);
    for (int i = 0; i < rank; ++i) {
      nchunk_ndim[i] += update_start[i];
    }
    int64_t nchunk;
    blosc2_multidim_to_unidim(nchunk_ndim, rank, chunks_in_array_strides, &nchunk);

    // Check if the chunk needs to be updated
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
      if (filename) {
        read_chunk_blosc2_ndim(filename, dataset_id, space_id, nchunk, chunk_start, chunk_stop, chunksize, data2);
      }
      else {
        insert_chunk_blosc2_ndim(dataset_id, chunk_start, chunksize, data2);
      }
    }
    data2 += chunksize;
  }

  IF_NEG_OUT_RET(H5Sclose(space_id), -5);

  //out_success:
  retval = 0;

  out:
  if (chunk_stop) free(chunk_stop);
  if (chunk_start) free(chunk_start);
  if (nchunk_ndim) free(nchunk_ndim);
  if (update_shape) free(update_shape);
  if (update_start) free(update_start);
  if (chunks_in_array_strides) free(chunks_in_array_strides);
  if (chunks_in_array) free(chunks_in_array);
  if (extshape) free(extshape);
  if (shape) free(shape);
  if (chunkshape) free(chunkshape);
  if (retval >= 0)
    return retval;

  H5E_BEGIN_TRY {
    H5Pclose(dcpl);
    H5Sclose(space_id);
  } H5E_END_TRY;
  return retval;
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
  herr_t retval = -1;
  hid_t space_id = -1;
  hid_t mem_space_id = -1;

  /* Check if the compressor is blosc2 */
  long blosc2_filter = 0;
  char *envvar = getenv("BLOSC2_FILTER");
  if (envvar != NULL)
    blosc2_filter = strtol(envvar, NULL, 10);
  /* blosc2 only supports step=1 for now. */
  for (int i = 0; i < rank; ++i) {
    if (step[i] != 1) {
      blosc2_support = false;
      break;
    }
  }
  if (blosc2_support && !((int) blosc2_filter)) {
    hsize_t *stop = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
    for (int i = 0; i < rank; ++i) {
      stop[i] = start[i] + count[i];
    }
    herr_t rv = get_set_blosc2_slice(NULL, dataset_id, type_id, rank, start, stop, data);
    free(stop);
    if (rv >= 0) {
      goto out_success;
    }
  }

  /* Create a simple memory data space */
  IF_NEG_OUT_RET(mem_space_id = H5Screate_simple(rank, count, NULL), -3);

  /* Get the file data space */
  IF_NEG_OUT_RET(space_id = H5Dget_space(dataset_id), -4);

  /* Define a hyperslab in the dataset */
  if (rank != 0) {
    IF_NEG_OUT_RET(H5Sselect_hyperslab(space_id, H5S_SELECT_SET,
                                       start, step, count, NULL), -5);
  }

  IF_NEG_OUT_RET(H5Dwrite(dataset_id, type_id, mem_space_id, space_id,
                          H5P_DEFAULT, data), -6);

  /* Terminate access to the dataspace */
  IF_NEG_OUT_RET(H5Sclose(mem_space_id), -7);

  IF_NEG_OUT_RET(H5Sclose(space_id), -8);

  out_success:
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


herr_t insert_chunk_blosc2_ndim(hid_t dataset_id,
                                hsize_t *start,
                                hsize_t chunksize,
                                const void *data) {
  herr_t retval = -1;
  hid_t dcpl = -1;
  blosc2_schunk *sc = NULL;
  bool needs_free2 = false;
  uint8_t *cframe = NULL;

  /* Get the dataset creation property list */
  dcpl = H5Dget_create_plist(dataset_id);
  IF_TRUE_OUT_BTRACE(dcpl == H5I_INVALID_HID, "Fail getting plist");

  /* Get blosc2 params*/
  size_t cd_nelmts = 7;
  unsigned cd_values[7];
  char name[7];
  IF_NEG_OUT_BTRACE(H5Pget_filter_by_id2(dcpl, FILTER_BLOSC2, NULL,
                                         &cd_nelmts, cd_values, 7, name, NULL),
                    "Fail getting blosc2 params");
  int32_t typesize = cd_values[2];
  IF_NEG_OUT(H5Pclose(dcpl));


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
  //  blosc2_schunk *sc = blosc2_schunk_new(&storage);
  //  IF_FALSE_OUT_BTRACE(sc, "Failed creating superchunk");
  sc = blosc2_schunk_new(&storage);
  IF_FALSE_OUT_BTRACE(sc, "Failed creating superchunk");

  // b2nd_set_slice_cbuffer(data, const int64_t *buffershape, int64_t buffersize,
  //                                        const int64_t *start, const int64_t *stop, b2nd_array_t *array)
  // To remove:
  //   IF_TRUE_OUT_BTRACE(blosc2_schunk_append_buffer(sc, (void *) data, chunksize) <= 0,
  //                      "Failed appending buffer");
  IF_TRUE_OUT_BTRACE(blosc2_schunk_append_buffer(sc, (void *) data, chunksize) <= 0,
                     "Failed appending buffer");
  // blosc2_schunk_to_buffer(array->sc, &cframe, &needs_free2)
  int64_t cfsize = blosc2_schunk_to_buffer(sc, &cframe, &needs_free2);
  IF_TRUE_OUT_BTRACE(cfsize <= 0, "Failed converting schunk to cframe");

  /* Write frame bypassing HDF5 filter pipeline */
  unsigned flt_msk = 0;
  IF_NEG_OUT_BTRACE(H5Dwrite_chunk(dataset_id, H5P_DEFAULT, flt_msk,
                                   start, (size_t) cfsize, cframe),
                    "Failed HDF5 writing chunk");

  //out_success:
  retval = 0;

  out:
  if (cframe && needs_free2) free(cframe);
  if (sc) blosc2_schunk_free(sc);
  if (retval >= 0)
    return retval;

  H5E_BEGIN_TRY {
    H5Pclose(dcpl);
  } H5E_END_TRY;
  return retval;
}


/* Get the maximum block size which is not greater than the given block_size
 * and fits within the given chunk dimensions dims_chunk. A zero block_size
 * means using an automatic value that fits most L2 CPU caches.
 *
 * Block dimensions start with 2 (unless the respective chunk dimension is 1),
 * and are doubled starting from the innermost (rightmost) ones, to leverage
 * the locality of C array arrangement.
 *
 * Based on Python-Blosc2's blosc2.core.compute_chunks_blocks and
 * compute_partition.
 */
hsize_t compute_block_size(hsize_t block_size,  // desired target, 0 for auto
                           hsize_t type_size,
                           const int rank,
                           const hsize_t *dims_chunk) {
  if (block_size == 0) {
    block_size = BLOSC2_DEFAULT_BLOCK_SIZE;
  }
  hsize_t nitems = block_size / type_size;

  // Start with the smallest possible block dimensions (1 or 2).
  hsize_t *dims_block = (hsize_t *)(malloc(rank * sizeof(hsize_t)));
  hsize_t nitems_new = 1;
  for (int i = 0; i < rank; i++) {
    assert(dims_chunk[i] != 0);
    dims_block[i] = dims_chunk[i] == 1 ? 1 : 2;
    nitems_new *= dims_block[i];
  }

  if (nitems_new > nitems) {
    BLOSC_TRACE_ERROR("Target block size is too small, raising to %lu", nitems_new);
  }
  IF_TRUE_OUT(nitems_new >= nitems);

  // Double block dimensions (bound by chunk dimensions) from right to left
  // while block is under nitems.
  while (nitems_new <= nitems) {
    hsize_t nitems_prev = nitems_new;
    for (int i = rank - 1; i >= 0; i--) {
      if (dims_block[i] * 2 <= dims_chunk[i]) {
        if (nitems_new * 2 <= nitems) {
          nitems_new *= 2;
          dims_block[i] *= 2;
        }
      } else if (dims_block[i] < dims_chunk[i]) {
        nitems_new = (nitems_new / dims_block[i]) * dims_chunk[i];
        dims_block[i] = dims_chunk[i];
      } else {
        assert(dims_block[i] == dims_chunk[i]);  // nothing to change
      }
    }
    if (nitems_new == nitems_prev) {
      break;  // not progressing anymore
    }
  }

  out:
  free(dims_block);
  return nitems_new * type_size;
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
                   hbool_t blosc2_support,
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
        cd_values[1] = (unsigned int) compute_block_size(block_size, type_size, rank, dims_chunk);
        cd_values[4] = compress;
        cd_values[5] = shuffle;
        IF_NEG_OUT(H5Pset_filter(plist_id, FILTER_BLOSC2, H5Z_FLAG_OPTIONAL, 6, cd_values));
      }
      /* The Blosc2 compressor can use other compressors (see blosc2_filter.c) */
      else if (strncmp(complib, "blosc2:", 7) == 0) {
        size_t type_size = H5Tget_size(type_id);
        IF_NEG_OUT(type_size);
        cd_values[1] = (unsigned int) compute_block_size(block_size, type_size, rank, dims_chunk);
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





herr_t read_chunk_blosc2_ndim(char *filename,
                              hid_t dataset_id,
                              hid_t space_id,
                              hsize_t nchunk,
                              hsize_t *chunk_start,
                              hsize_t *chunk_stop,
                              hsize_t chunksize,
                              uint8_t *data) {
  herr_t retval = -1;
  blosc2_schunk *schunk = NULL;
  bool needs_free = false;
  uint8_t *chunk = NULL;
  blosc2_context *dctx = NULL;

  /* Get the address of the schunk on-disk */
  unsigned flt_msk;
  haddr_t address;
  hsize_t cframe_size;
  hsize_t chunk_offset[2];
  IF_NEG_OUT_BTRACE(H5Dget_chunk_info(dataset_id, space_id, nchunk, chunk_offset, &flt_msk,
                                      &address, &cframe_size),
                    "Get chunk info failed!\n");

  /* Open the schunk on-disk */
  // b2nd_open_offset(const char *urlpath, b2nd_array_t **array, (int64_t) address)
  // To remove:
  //   blosc2_schunk *schunk = blosc2_schunk_open_offset(filename, (int64_t) address);
  //  IF_FALSE_OUT_BTRACE(schunk, "Cannot open schunk in %s\n", filename);
  schunk = blosc2_schunk_open_offset(filename, (int64_t) address);
  IF_FALSE_OUT_BTRACE(schunk, "Cannot open schunk in %s\n", filename);

  //  b2nd_set_slice_cbuffer(data, const int64_t *buffershape, int64_t buffersize,
  // const int64_t *start, stop, b2nd_array_t *array)

  /* Get chunk */
  int32_t cbytes = blosc2_schunk_get_lazychunk(schunk, 0, &chunk, &needs_free);
  IF_NEG_OUT_BTRACE(cbytes, "Cannot get lazy chunk %zd in %s\n", nchunk, filename);

  blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
  dparams.schunk = schunk;
  dctx = blosc2_create_dctx(dparams);

  /* Gather data for the interesting part */
  //int nrecords_chunk = chunksize - chunk_start;

  int rbytes;
  //if (nrecords_chunk == chunklen) {
  rbytes = blosc2_decompress_ctx(dctx, chunk, cbytes, data, chunksize);
  IF_NEG_OUT_BTRACE(rbytes, "Cannot decompress lazy chunk");
  //}
  /*else {
    /* Less than 1 chunk to read; use a getitem call
    rbytes = blosc2_getitem_ctx(dctx, chunk, cbytes, start_chunk, nrecords_chunk, data, chunksize);
    IF_TRUE_OUT_BTRACE(rbytes != nrecords_chunk * typesize,
                       "Cannot get (all) items for lazychunk\n");
  }*/

  //out_success:
  retval = 0;

  out:
  if (chunk && needs_free) free(chunk);
  if (dctx) blosc2_free_ctx(dctx);
  if (schunk) blosc2_schunk_free(schunk);
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
        blosc2_support = false;
      }
    }

    long blosc2_filter = 0;
    char *envvar = getenv("BLOSC2_FILTER");
    if (envvar != NULL)
      blosc2_filter = strtol(envvar, NULL, 10);

    if (blosc2_support && !((int) blosc2_filter)) {
      IF_NEG_OUT_RET(H5Sclose(space_id), -40);  // no longer usable here
      /* Try to read using blosc2 (only supports native byteorder and step=1 for now) */
      herr_t rv;
      IF_NEG_OUT_RET(rv = get_set_blosc2_slice(filename, dataset_id, type_id,
                                               rank, start, stop, data),
                     rv - 40);
      goto out_success;
    }

    /* Define a hyperslab in the dataset of the size of the records */
    IF_NEG_OUT_RET(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, stride,
                                       count, NULL), -5);

    /* Create a memory dataspace handle */
    IF_NEG_OUT_RET(mem_space_id = H5Screate_simple(rank, count, NULL), -6);

    /* Read */
    IF_NEG_OUT_RET(H5Dread(dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT,
                           data), -7);

    /* Terminate access to the memory dataspace */
    IF_NEG_OUT_RET(H5Sclose(mem_space_id), -8);

    /* Terminate access to the dataspace */
    IF_NEG_OUT_RET(H5Sclose(space_id), -9);
  } else {                     /* Scalar case */

    /* Read all the dataset */
    IF_NEG_OUT_RET(H5Dread(dataset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), -10);
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
