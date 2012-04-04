#include "H5Zbzip2.h"

#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <hdf5.h>


#ifdef HAVE_BZ2_LIB
#include "bzlib.h"
#endif  /* defined HAVE_BZ2_LIB */

size_t bzip2_deflate(unsigned int flags, size_t cd_nelmts,
                     const unsigned int cd_values[], size_t nbytes,
                     size_t *buf_size, void **buf);


int register_bzip2(char **version, char **date)
{
#ifdef HAVE_BZ2_LIB
  char *libver, *versionp, *datep, *sep;

  H5Z_class_t filter_class = {
    H5Z_CLASS_T_VERS,             /* H5Z_class_t version */
    (H5Z_filter_t)(FILTER_BZIP2), /* filter_id */
    1, 1,                         /* Encoding and decoding enabled */
    "bzip2",                      /* comment */
    NULL,                         /* can_apply_func */
    NULL,                         /* set_local_func */
    (H5Z_func_t)(bzip2_deflate)   /* filter_func */
  };

  /* Register the filter class for the bzip2 compressor. */
  H5Zregister(&filter_class);

  /* Get the library major version from the version string. */
  libver = strdup(BZ2_bzlibVersion());
  sep = strchr(libver, ',');
  assert(sep != NULL);
  assert(*(sep + 1) == ' ');
  *sep = '\0';
  versionp = libver;
  datep = sep + 2;  /* after the comma and a space */

  *version = strdup(versionp);
  *date = strdup(datep);

  free(libver);
  return 1;  /* library is available */

#else
  return 0;  /* library is not available */
#endif  /* defined HAVE_BZ2_LIB */

}


size_t bzip2_deflate(unsigned int flags, size_t cd_nelmts,
                     const unsigned int cd_values[], size_t nbytes,
                     size_t *buf_size, void **buf)
{
#ifdef HAVE_BZ2_LIB
  char *outbuf = NULL;
  size_t outbuflen, outdatalen;
  int ret;

  if (flags & H5Z_FLAG_REVERSE) {

    /** Decompress data.
     **
     ** This process is troublesome since the size of uncompressed data
     ** is unknown, so the low-level interface must be used.
     ** Data is decompressed to the output buffer (which is sized
     ** for the average case); if it gets full, its size is doubled
     ** and decompression continues.  This avoids repeatedly trying to
     ** decompress the whole block, which could be really inefficient.
     **/

    bz_stream stream;
    char *newbuf = NULL;
    size_t newbuflen;

    /* Prepare the output buffer. */
    outbuflen = nbytes * 3 + 1;  /* average bzip2 compression ratio is 3:1 */
    outbuf = malloc(outbuflen);
    if (outbuf == NULL) {
      fprintf(stderr, "memory allocation failed for bzip2 decompression\n");
      goto cleanupAndFail;
    }

    /* Use standard malloc()/free() for internal memory handling. */
    stream.bzalloc = NULL;
    stream.bzfree = NULL;
    stream.opaque = NULL;

    /* Start decompression. */
    ret = BZ2_bzDecompressInit(&stream, 0, 0);
    if (ret != BZ_OK) {
      fprintf(stderr, "bzip2 decompression start failed with error %d\n", ret);
      goto cleanupAndFail;
    }

    /* Feed data to the decompression process and get decompressed data. */
    stream.next_out = outbuf;
    stream.avail_out = outbuflen;
    stream.next_in = *buf;
    stream.avail_in = nbytes;
    do {
      ret = BZ2_bzDecompress(&stream);
      if (ret < 0) {
        fprintf(stderr, "BUG: bzip2 decompression failed with error %d\n", ret);
        goto cleanupAndFail;
      }

      if (ret != BZ_STREAM_END && stream.avail_out == 0) {
        /* Grow the output buffer. */
        newbuflen = outbuflen * 2;
        newbuf = realloc(outbuf, newbuflen);
        if (newbuf == NULL) {
          fprintf(stderr, "memory allocation failed for bzip2 decompression\n");
          goto cleanupAndFail;
        }
        stream.next_out = newbuf + outbuflen;  /* half the new buffer behind */
        stream.avail_out = outbuflen;  /* half the new buffer ahead */
        outbuf = newbuf;
        outbuflen = newbuflen;
      }
    } while (ret != BZ_STREAM_END);

    /* End compression. */
    outdatalen = stream.total_out_lo32;
    ret = BZ2_bzDecompressEnd(&stream);
    if (ret != BZ_OK) {
      fprintf(stderr, "bzip2 compression end failed with error %d\n", ret);
      goto cleanupAndFail;
    }

  } else {

    /** Compress data.
     **
     ** This is quite simple, since the size of compressed data in the worst
     ** case is known and it is not much bigger than the size of uncompressed
     ** data.  This allows us to use the simplified one-shot interface to
     ** compression.
     **/

    unsigned int odatalen;  /* maybe not the same size as outdatalen */
    int blockSize100k = 9;

    /* Get compression block size if present. */
    if (cd_nelmts > 0) {
      blockSize100k = cd_values[0];
      if (blockSize100k < 1 || blockSize100k > 9) {
        fprintf(stderr, "invalid compression block size: %d\n", blockSize100k);
        goto cleanupAndFail;
      }
    }

    /* Prepare the output buffer. */
    outbuflen = nbytes + nbytes / 100 + 600;  /* worst case (bzip2 docs) */
    outbuf = malloc(outbuflen);
    if (outbuf == NULL) {
      fprintf(stderr, "memory allocation failed for bzip2 compression\n");
      goto cleanupAndFail;
    }

    /* Compress data. */
    odatalen = outbuflen;
    ret = BZ2_bzBuffToBuffCompress(outbuf, &odatalen, *buf, nbytes,
                                   blockSize100k, 0, 0);
    outdatalen = odatalen;
    if (ret != BZ_OK) {
      fprintf(stderr, "bzip2 compression failed with error %d\n", ret);
      goto cleanupAndFail;
    }
  }

  /* Always replace the input buffer with the output buffer. */
  free(*buf);
  *buf = outbuf;
  *buf_size = outbuflen;
  return outdatalen;

 cleanupAndFail:
  if (outbuf)
    free(outbuf);
  return 0;
#else
  return 0;
#endif  /* defined HAVE_BZ2_LIB */
}
