#include <string.h>
#include <stdlib.h>
#include <hdf5.h>

#include "H5Zlzo.h"
#include "tables.h"

#ifdef HAVE_LZO_LIB
#   include "lzo1x.h"
#endif
#ifdef HAVE_LZO2_LIB
#   include "lzo/lzo1x.h"
#   define HAVE_LZO_LIB  /* The API for LZO and LZO2 is mostly identical */
#endif

/* #undef DEBUG */

/* Activate the checksum. It is safer and takes only a 1% more of
   space and a 2% more of CPU (but sometimes is faster than without
   checksum, which is almost negligible.  F. Alted 2003/07/22

   Added code for pytables 0.5 backward compatibility.
   F. Alted 2003/07/28

   Added code for saving the uncompressed length buffer as well.
   F. Alted 2003/07/29

*/

/* From pytables 0.8 on I decided to let the user select the
   fletcher32 checksum provided in HDF5 1.6 or higher. So, even though
   the CHECKSUM support here seems pretty stable it will be disabled.
   F. Alted 2004/01/02 */
#undef CHECKSUM

size_t lzo_deflate (unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t *buf_size, void **buf);


int register_lzo(char **version, char **date) {

#ifdef HAVE_LZO_LIB

  H5Z_class_t filter_class = {
    H5Z_CLASS_T_VERS,             /* H5Z_class_t version */
    (H5Z_filter_t)(FILTER_LZO),   /* filter_id */
    1, 1,                         /* Encoding and decoding enabled */
    "lzo",                        /* comment */
    NULL,                         /* can_apply_func */
    NULL,                         /* set_local_func */
    (H5Z_func_t)(lzo_deflate)     /* filter_func */
  };

  /* Init the LZO library */
  if (lzo_init()!=LZO_E_OK) {
    fprintf(stderr, "Problems initializing LZO library\n");
    *version = NULL;
    *date = NULL;
    return 0; /* lib is not available */
  }

  /* Register the lzo compressor */
  H5Zregister(&filter_class);

  *version = strdup(LZO_VERSION_STRING);
  *date = strdup(LZO_VERSION_DATE);
  return 1; /* lib is available */

#else
  *version = NULL;
  *date = NULL;
  return 0; /* lib is not available */
#endif /* HAVE_LZO_LIB */

}


size_t lzo_deflate (unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t *buf_size, void **buf)
{
  size_t ret_value = 0;
#ifdef HAVE_LZO_LIB
  void *outbuf = NULL, *wrkmem = NULL;
  int status;
  size_t  nalloc = *buf_size;
  lzo_uint out_len = (lzo_uint) nalloc;
  /* max_len_buffer will keep the likely output buffer size
     after processing the first chunk */
  static unsigned int max_len_buffer = 0;
  /* int complevel = 1; */
#if (defined CHECKSUM || defined DEBUG)
  int object_version = 10;      /* Default version 1.0 */
  int object_type = Table;      /* Default object type */
#endif
#ifdef CHECKSUM
  lzo_uint32 checksum;
#endif

  /* Check arguments */
  /* For Table versions < 20, there were no parameters */
  if (cd_nelmts==1 ) {
    /* complevel = cd_values[0]; */ /* This do nothing right now */
  }
  else if (cd_nelmts==2 ) {
    /* complevel = cd_values[0]; */ /* This do nothing right now */
#if (defined CHECKSUM || defined DEBUG)
    object_version = cd_values[1]; /* The table VERSION attribute */
#endif
  }
  else if (cd_nelmts==3 ) {
    /* complevel = cd_values[0]; */ /* This do nothing right now */
#if (defined CHECKSUM || defined DEBUG)
    object_version = cd_values[1]; /* The table VERSION attribute */
    object_type = cd_values[2]; /* A tag for identifying the object
                                   (see tables.h) */
#endif
  }

#ifdef DEBUG
  printf("Object type: %d. ", object_type);
  printf("object_version:%d\n", object_version);
#endif

  if (flags & H5Z_FLAG_REVERSE) {
    /* Input */

/*     printf("Decompressing chunk with LZO\n"); */
#ifdef CHECKSUM
    if ((object_type == Table && object_version >= 20) ||
        object_type != Table) {
      nbytes -= 4;      /* Point to uncompressed buffer length */
      memcpy(&nalloc, ((unsigned char *)(*buf)+nbytes), 4);
      out_len = nalloc;
      nbytes -= 4;      /* Point to the checksum */
#ifdef DEBUG
      printf("Compressed bytes: %d. Uncompressed bytes: %d\n", nbytes, nalloc);
#endif
    }
#endif

    /* Only allocate the bytes for the outbuf */
    if (max_len_buffer == 0) {
      if (NULL==(outbuf = (void *)malloc(nalloc)))
        fprintf(stderr, "Memory allocation failed for lzo uncompression.\n");
    }
    else {
      if (NULL==(outbuf = (void *)malloc(max_len_buffer)))
        fprintf(stderr, "Memory allocation failed for lzo uncompression.\n");
      out_len = max_len_buffer;
      nalloc =  max_len_buffer;
    }

    while(1) {

#ifdef DEBUG
      printf("nbytes -->%d\n", nbytes);
      printf("nalloc -->%d\n", nalloc);
      printf("max_len_buffer -->%d\n", max_len_buffer);
#endif /* DEBUG */

      /* The assembler version is a 10% slower than the C version with
         gcc 3.2.2 and gcc 3.3.3 */
/*       status = lzo1x_decompress_asm_safe(*buf, (lzo_uint)nbytes, outbuf, */
/*                                          &out_len, NULL); */
      /* The safe and unsafe versions have the same speed more or less */
      status = lzo1x_decompress_safe(*buf, (lzo_uint)nbytes, outbuf,
                                     &out_len, NULL);

      if (status == LZO_E_OK) {
#ifdef DEBUG
        printf("decompressed %lu bytes back into %lu bytes\n",
               (long) nbytes, (long) out_len);
#endif
        max_len_buffer = out_len;
        break; /* done */
      }
      else if (status == LZO_E_OUTPUT_OVERRUN) {
        nalloc *= 2;
        out_len = (lzo_uint) nalloc;
        if (NULL==(outbuf = realloc(outbuf, nalloc))) {
          fprintf(stderr, "Memory allocation failed for lzo uncompression\n");
        }
      }
      else {
        /* this should NEVER happen */
        fprintf(stderr, "internal error - decompression failed: %d\n", status);
        ret_value = 0; /* fail */
        goto done;
      }
    }

#ifdef CHECKSUM
    if ((object_type == Table && object_version >= 20) ||
        object_type != Table) {
#ifdef DEBUG
      printf("Checksum uncompressing...");
#endif
      /* Compute the checksum */
      checksum=lzo_adler32(lzo_adler32(0,NULL,0), outbuf, out_len);

      /* Compare */
      if (memcmp(&checksum, (unsigned char*)(*buf)+nbytes, 4)) {
        ret_value = 0; /*fail*/
        fprintf(stderr,"Checksum failed!.\n");
        goto done;
      }
    }
#endif /* CHECKSUM */

    free(*buf);
    *buf = outbuf;
    outbuf = NULL;
    *buf_size = nalloc;
    ret_value = out_len;

  } else {
    /*
     * Output; compress but fail if the result would be larger than the
     * input.  The library doesn't provide in-place compression, so we
     * must allocate a separate buffer for the result.
     */
    lzo_byte *z_src = (lzo_byte*)(*buf);
    lzo_byte *z_dst;         /*destination buffer            */
    lzo_uint z_src_nbytes = (lzo_uint)(nbytes);
    /* The next was the original computation for worst-case expansion */
    /* I don't know why the difference with LZO1*. Perhaps some wrong docs in
       LZO package? */
/*     lzo_uint z_dst_nbytes = (lzo_uint)(nbytes + (nbytes / 64) + 16 + 3); */
    /* The next is for LZO1* algorithms */
/*     lzo_uint z_dst_nbytes = (lzo_uint)(nbytes + (nbytes / 16) + 64 + 3); */
    /* The next is for LZO2* algorithms. This will be the default */
    lzo_uint z_dst_nbytes = (lzo_uint)(nbytes + (nbytes / 8) + 128 + 3);

#ifdef CHECKSUM
    if ((object_type == Table && object_version >= 20) ||
        object_type != Table) {
      z_dst_nbytes += 4+4;      /* Checksum + buffer size */
    }
#endif

    if (NULL==(z_dst=outbuf=(void *)malloc(z_dst_nbytes))) {
      fprintf(stderr, "Unable to allocate lzo destination buffer.\n");
      ret_value = 0; /* fail */
      goto done;
    }

    /* Compress this buffer */
    wrkmem = malloc(LZO1X_1_MEM_COMPRESS);
    if (wrkmem == NULL) {
      fprintf(stderr, "Memory allocation failed for lzo compression\n");
      ret_value = 0;
      goto done;
    }

    status = lzo1x_1_compress (z_src, z_src_nbytes, z_dst, &z_dst_nbytes,
                               wrkmem);

    free(wrkmem);
    wrkmem = NULL;

#ifdef CHECKSUM
    if ((object_type == Table && object_version >= 20) ||
        object_type != Table) {
#ifdef DEBUG
      printf("Checksum compressing ...");
      printf("src_nbytes: %d, dst_nbytes: %d\n", z_src_nbytes, z_dst_nbytes);
#endif
      /* Append checksum of *uncompressed* data at the end */
      checksum = lzo_adler32(lzo_adler32(0,NULL,0), *buf, nbytes);
      memcpy((unsigned char*)(z_dst)+z_dst_nbytes, &checksum, 4);
      memcpy((unsigned char*)(z_dst)+z_dst_nbytes+4, &nbytes, 4);
      z_dst_nbytes += (lzo_uint)4+4;
      nbytes += 4+4;
    }
#endif

    if (z_dst_nbytes >= nbytes) {
#ifdef DEBUG
      printf("The compressed buffer takes more space than uncompressed!.\n");
#endif
      ret_value = 0; /* fail */
      goto done;
    } else if (LZO_E_OK != status) {
      fprintf(stderr,"lzo library error in compression\n");
      ret_value = 0; /* fail */
      goto done;
    } else {
      free(*buf);
      *buf = outbuf;
      outbuf = NULL;
      *buf_size = z_dst_nbytes;
      ret_value = z_dst_nbytes;
    }
  }

done:
  if(outbuf)
    free(outbuf);

#endif  /* HAVE_LZO_LIB */

  return ret_value;
}
