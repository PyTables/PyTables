#include <stdlib.h>

/* We will be using the LZO1X-1 algorithm, so we have
 * to include <lzo1x.h>
 */

#include "H5Zlzo.h"
#include "utils.h"

#ifdef HAVE_LZO_LIB
#   include "lzo1x.h"

void *wrkmem;

#endif

#undef DEBUG

/* Activate the checksum. It is safer and takes only a 1% more of
   space and a 2% more of CPU (but sometimes is faster than without
   checksum, which is almost negligible.  F. Alted 2003/07/22
  
   Added code for pytables 0.5 backward compatibility.
   F. Alted 2003/07/28
*/

#define CHECKSUM

int register_lzo(void) {

#ifdef HAVE_LZO_LIB

  herr_t status;

  /* Init the LZO library */
  if (lzo_init()!=LZO_E_OK)
    printf("Problems initializing LZO library\n");

  /* Feed the filter_class data structure */
  H5Z_class_t filter_class = {
    (H5Z_filter_t)FILTER_LZO,	/* filter_id */
    "lzo deflate", 		/* comment */
    NULL,                       /* can_apply_func */
    NULL,                       /* set_local_func */
    (H5Z_func_t)lzo_deflate     /* filter_func */
  };

  /* Register the lzo compressor */
  status = H5Zregister(&filter_class);
  
  /* Book a buffer for the compression */
  wrkmem = (void *)malloc(LZO1X_1_MEM_COMPRESS);
   
  return LZO_VERSION; /* lib is available */

#else
  return 0; /* lib is not available */
#endif /* HAVE_LZO_LIB */

}

/* This routine only can be called if LZO is present */
PyObject *getLZOVersionInfo(void) {
  char *info[2];

#ifdef HAVE_LZO_LIB
  info[0] = strdup(LZO_VERSION_STRING);
  info[1] = strdup(LZO_VERSION_DATE);
#else
  info[0] = NULL;
  info[1] = NULL;
#endif /* HAVE_LZO_LIB */
  return createNamesTuple(info, 2);
}


size_t lzo_deflate (unsigned flags, size_t cd_nelmts,
		    const unsigned cd_values[], size_t nbytes,
		    size_t *buf_size, void **buf)
{
  size_t ret_value = 0;
  void *outbuf = NULL;
#ifdef HAVE_LZO_LIB
  int status;
  size_t  nalloc = *buf_size;
  lzo_uint out_len = (lzo_uint) nalloc;
  /* max_len_buffer will keep the likely output buffer size
     after processing the first chunk */
  static unsigned int max_len_buffer = 0;
  int complevel = 1;
  int object_version = 10;    	/* Default version 1.0 */
#ifdef CHECKSUM
  lzo_uint32 checksum;
#endif

  /* Check arguments */
/*   if (cd_nelmts<1 || cd_values[0]>9) { */
/*     printf("invalid deflate aggression level"); */
/*   } */
  /* For versions < 20, there were no parameters */
  if (cd_nelmts==1 ) {
    complevel = cd_values[0];	/* This do nothing right now */
    printf("invalid deflate aggression level");
  }
  else if (cd_nelmts==2 ) {
    complevel = cd_values[0];	/* This do nothing right now */
    object_version = cd_values[1]; /* The table VERSION attribute */
  }

#if DEBUG
  printf("object_version:%d\n", object_version);
#endif

  if (flags & H5Z_FLAG_REVERSE) {
    /* Input */

    /* Only allocate the bytes for the outbuf */
    if (max_len_buffer == 0) {
      if (NULL==(outbuf = (void *)malloc(nalloc)))
	printf("memory allocation failed for deflate uncompression");
    }
    else {
      if (NULL==(outbuf = (void *)malloc(max_len_buffer)))
	printf("memory allocation failed for deflate uncompression");
      out_len = max_len_buffer;
      nalloc =  max_len_buffer;
    }

#ifdef CHECKSUM
    if (object_version >= 20) {
      nbytes -= 4;
    }
#endif
    while(1) {
      /* The assembler version isn't faster than the C version with 
	 gcc 3.2.2, and it's unsafe */
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
	  printf("memory allocation failed for lzo uncompression");
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
    if (object_version >= 20) {
#ifdef DEBUG
      printf("Checksum uncompressing...");
#endif
      /* Compute the checksum */
      checksum=lzo_adler32(lzo_adler32(0,NULL,0), outbuf, out_len);

      /* Compare */
      if (memcmp(&checksum, (char*)(*buf)+nbytes, 4)) {
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
    lzo_uint z_dst_nbytes = (lzo_uint)(nbytes + (nbytes / 64) + 16 + 3 + 4);

    if (NULL==(z_dst=outbuf=(void *)malloc(z_dst_nbytes))) {
	fprintf(stderr, "unable to allocate deflate destination buffer");
	ret_value = 0; /* fail */
	goto done;
    }

    /* Compress this buffer */
    status = lzo1x_1_compress (z_src, z_src_nbytes, z_dst, &z_dst_nbytes,
			       wrkmem);
#ifdef CHECKSUM
    /* Append checksum of *uncompressed* data at the end */
    if (object_version >= 20) {
#ifdef DEBUG
      printf("Checksum compression...");
#endif
      checksum = lzo_adler32(lzo_adler32(0,NULL,0), *buf, nbytes);
      memcpy((char*)(z_dst)+z_dst_nbytes, &checksum, 4);
      z_dst_nbytes += (lzo_uint)4;
      nbytes += 4; 
    }
#endif /* CHECKSUM */

    if (z_dst_nbytes >= nbytes) {
      /* fprintf(stderr,"overflow"); */
      ret_value = 0; /* fail */
      goto done;
    } else if (LZO_E_OK != status) {
      /* fprintf(stderr,"lzo error"); */
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

#endif  /* HAVE_LZO_LIB */

done:
  if(outbuf)
    free(outbuf);

  return ret_value;
}

