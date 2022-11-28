/*
    Copyright (C) 2022  Blosc Development Team
    http://blosc.org
    License: MIT (see LICENSE.txt)

    Filter program that allows the use of the Blosc2 filter in HDF5.

*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "hdf5.h"
#include "blosc2_filter.h"

#if defined(__GNUC__)
#define PUSH_ERR(func, minor, str, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, str, ##__VA_ARGS__)
#elif defined(_MSC_VER)
#define PUSH_ERR(func, minor, str, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, str, __VA_ARGS__)
#else
/* This version is portable but it's better to use compiler-supported
   approaches for handling the trailing comma issue when possible. */
#define PUSH_ERR(func, minor, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, __VA_ARGS__)
#endif	/* defined(__GNUC__) */

#define GET_FILTER(a, b, c, d, e, f, g) H5Pget_filter_by_id(a,b,c,d,e,f,g,NULL)


size_t blosc2_filter_function(unsigned flags, size_t cd_nelmts,
                              const unsigned cd_values[], size_t nbytes,
                              size_t* buf_size, void** buf);

herr_t blosc2_set_local(hid_t dcpl, hid_t type, hid_t space);


/* Register the filter, passing on the HDF5 return value */
int register_blosc2(char **version, char **date){

    int retval;

    H5Z_class_t filter_class = {
        H5Z_CLASS_T_VERS,
        (H5Z_filter_t)(FILTER_BLOSC2),
        1, 1,
        "blosc2",
        NULL,
        (H5Z_set_local_func_t)(blosc2_set_local),
        (H5Z_func_t)(blosc2_filter_function)
    };

    retval = H5Zregister(&filter_class);
    if(retval<0){
        PUSH_ERR("register_blosc2", H5E_CANTREGISTER, "Can't register Blosc2 filter");
    }
    if (version != NULL && date != NULL) {
        *version = strdup(BLOSC2_VERSION_STRING);
        *date = strdup(BLOSC2_VERSION_DATE);
    }
    return 1; /* lib is available */
}

/*  Filter setup.  Records the following inside the DCPL:

    1. If version information is not present, set slots 0 and 1 to the filter
       revision and Blosc2 version, respectively.

    2. Compute the type size in bytes and store it in slot 2.

    3. Compute the chunk size in bytes and store it in slot 3.
*/
herr_t blosc2_set_local(hid_t dcpl, hid_t type, hid_t space) {

  int ndims;
  int i;
  herr_t r;

  unsigned int typesize, basetypesize;
  unsigned int bufsize;
  hsize_t chunkdims[32];
  unsigned int flags;
  size_t nelements = 8;
  unsigned int values[] = {0, 0, 0, 0, 0, 0, 0, 0};
  hid_t super_type;
  H5T_class_t classt;

  r = GET_FILTER(dcpl, FILTER_BLOSC2, &flags, &nelements, values, 0, NULL);
  if (r < 0) return -1;

  if (nelements < 4)
    nelements = 4;  /* First 4 slots reserved. */

  /* Set Blosc2 info in first two slots */
  values[0] = FILTER_BLOSC2_VERSION;

  ndims = H5Pget_chunk(dcpl, 32, chunkdims);
  if (ndims < 0)
    return -1;
  if (ndims > 32) {
    PUSH_ERR("blosc2_set_local", H5E_CALLBACK, "Chunk rank exceeds limit");
    return -1;
  }

  typesize = (unsigned int) H5Tget_size(type);
  if (typesize == 0) return -1;
  /* Get the size of the base type, even for ARRAY types */
  classt = H5Tget_class(type);
  if (classt == H5T_ARRAY) {
    /* Get the array base component */
    super_type = H5Tget_super(type);
    basetypesize = (unsigned int) H5Tget_size(super_type);
    /* Release resources */
    H5Tclose(super_type);
  } else {
    basetypesize = typesize;
  }

  /* Limit large typesizes (they are pretty expensive to shuffle
     and, in addition, Blosc2 does not handle typesizes larger than
     255 bytes). */
  /* But for reads, it is useful to have the original typesize here.
   * And there is no harm in storing the original typesize here, as Blosc2
   * will decide internally to reduce it to 1 if > BLOSC_MAX_TYPESIZE.
   */
  /* if (basetypesize > BLOSC_MAX_TYPESIZE)
   *  basetypesize = 1;
   */
  values[2] = basetypesize;

  /* Get the size of the chunk */
  bufsize = typesize;
  for (i = 0; i < ndims; i++) {
    bufsize *= (unsigned int) chunkdims[i];
  }
  values[3] = bufsize;

#ifdef BLOSC2_DEBUG
  fprintf(stderr, "Blosc2: Computed buffer size %d\n", bufsize);
#endif

  r = H5Pmodify_filter(dcpl, FILTER_BLOSC2, flags, nelements, values);
  if (r < 0)
    return -1;

  return 1;
}


/* The filter function */
size_t blosc2_filter_function(unsigned flags, size_t cd_nelmts,
                              const unsigned cd_values[], size_t nbytes,
                              size_t* buf_size, void** buf) {

  void* outbuf = NULL;
  int64_t status = 0;                /* Return code from Blosc2 routines */
  size_t blocksize;
  size_t typesize;
  size_t outbuf_size;
  int clevel = 5;                /* Compression level default */
  int doshuffle = 1;             /* Shuffle default */
  int compcode = BLOSC_BLOSCLZ;  /* Codec by default */

  /* Filter params that are always set */
  blocksize = cd_values[1];      /* The block size */
  typesize = cd_values[2];      /* The datatype size */
  outbuf_size = cd_values[3];   /* Precomputed buffer guess */

  blosc2_init();

  if (!(flags & H5Z_FLAG_REVERSE)) {
    /* We're compressing */
    /* Compression params */
    clevel = cd_values[4];        /* The compression level */
    doshuffle = cd_values[5];     /* SHUFFLE, BITSHUFFLE, others */
    if (cd_nelmts >= 7) {
      compcode = cd_values[6];    /* The Blosc2 compressor used */
      /* Check that we actually have support for the compressor code */
      const char* complist = blosc2_list_compressors();
      const char* compname;
      int code = blosc2_compcode_to_compname(compcode, &compname);
      if (code == -1) {
        PUSH_ERR("blosc2_filter", H5E_CALLBACK,
                 "this Blosc2 library does not have support for "
                 "the '%s' compressor, but only for: %s",
                 compname, complist);
        goto failed;
      }
    }

#ifdef BLOSC2_DEBUG
    fprintf(stderr, "Blosc2: Compress %zd chunk w/buffer %zd\n",
            nbytes, *buf_size);
#endif

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = compcode;
    cparams.typesize = (int32_t) typesize;
    cparams.filters[BLOSC_LAST_FILTER] = doshuffle;
    cparams.clevel = clevel;

    blosc2_context *cctx = blosc2_create_cctx(cparams);
    blosc2_storage storage = {.cparams=&cparams, .contiguous=false};
    blosc2_schunk* schunk = blosc2_schunk_new(&storage);
    if (schunk == NULL) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Cannot create a super-chunk");
      goto failed;
    }

    status = blosc2_schunk_append_buffer(schunk, *buf, (int32_t) nbytes);
    if (status < 0) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Cannot append to buffer");
      goto failed;
    }

    bool needs_free;
    status = blosc2_schunk_to_buffer(schunk, (uint8_t **)&outbuf, &needs_free);
    if (status < 0 || !needs_free) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Cannot convert to buffer");
      goto failed;
    }
    blosc2_schunk_free(schunk);
    blosc2_free_ctx(cctx);

#ifdef BLOSC2_DEBUG
    fprintf(stderr, "Blosc2: Compressed into %zd bytes\n", status);
#endif

  } else {
    /* We're decompressing */
    /* declare dummy variables */
    int32_t cbytes;

    blosc2_schunk* schunk = blosc2_schunk_from_buffer(*buf, (int64_t)nbytes, false);
    if (schunk == NULL) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Cannot get super-chunk from buffer");
      goto failed;
    }

    uint8_t *chunk;
    bool needs_free;
    cbytes = blosc2_schunk_get_lazychunk(schunk, 0, &chunk, &needs_free);
    if (cbytes < 0) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Get chunk error");
      goto failed;
    }

    /* Get the exact outbuf_size from the buffer header */
    int32_t nbytes;
    blosc2_cbuffer_sizes(chunk, &nbytes, NULL, NULL);
    outbuf_size = nbytes;

#ifdef BLOSC2_DEBUG
    fprintf(stderr, "Blosc2: Decompress %zd chunk w/buffer %zd\n", nbytes, outbuf_size);
#endif

    outbuf = malloc(outbuf_size);
    if (outbuf == NULL) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Can't allocate decompression buffer");
      goto failed;
    }

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    blosc2_context *dctx = blosc2_create_dctx(dparams);
    status = blosc2_decompress_ctx(dctx, chunk, cbytes, outbuf, (int32_t) outbuf_size);
    if (status <= 0) {
      PUSH_ERR("blosc2_filter", H5E_CALLBACK, "Blosc2 decompression error");
      goto failed;
    }

    // Cleanup
    if (needs_free) {
      free(chunk);
    }
    blosc2_free_ctx(dctx);
    blosc2_schunk_free(schunk);

  } /* compressing vs decompressing */

  if (status > 0) {
    free(*buf);
    *buf = outbuf;
    *buf_size = outbuf_size;
    return status;  /* Size of compressed/decompressed data */
  }

  failed:
  free(outbuf);
  blosc2_destroy();

  return 0;

} /* End filter function */
