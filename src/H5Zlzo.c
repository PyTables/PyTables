#include <stdlib.h>

/* We will be using the LZO1X-1 algorithm, so we have
 * to include <lzo1x.h>
 */

#ifdef HAVE_LZO_LIB
#   include "lzo1x.h"
#   include "H5Zlzo.h"

void *wrkmem;

#endif

#undef DEBUG

#undef CHECKSUM

int register_lzo(void) {

#ifdef HAVE_LZO_LIB

  herr_t status;

  /* Init the LZO library */
  if (lzo_init()!=LZO_E_OK)
    printf("Problems initializing LZO library\n");

  /* Register the lzo compressor */
  status = H5Zregister(FILTER_LZO, "lzo deflate", lzo_deflate);
  
  /* Book a buffer for the compression */
  wrkmem = (void *)malloc(LZO1X_1_MEM_COMPRESS);
   
  return 1; /* lib is available */

#else
  return 0; /* lib is not available */
#endif /* HAVE_LZO_LIB */

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
#ifdef CHECKSUM
  lzo_uint32 checksum;
#endif

  if (flags & H5Z_FLAG_REVERSE) {
    /* Input */

    if (NULL==(outbuf = (void *)malloc(nalloc))) {
      printf("memory allocation failed for deflate uncompression");
    }
#ifdef CHECKSUM
    nbytes -=4;
#endif
    while(1) {
      /* The assembler version isn't faster than the C version with gcc 3.2.2 */
      /*       status = _lzo1x_decompress_asm_fast_safe(*buf,(lzo_uint)nbytes,outbuf,&out_len,NULL); */
      /* lzo1x_decompress seems to be safe to use in this context, but I'll use the _safe
	 version anyway because it's only sligthly slower (1% or 2 %) */
      /*       status = lzo1x_decompress(*buf,(lzo_uint)nbytes,outbuf,&out_len,NULL); */
#if defined __i386__ && 0  /* Change this to 1 if you want the assembler version */
      status = _lzo1x_decompress_asm_fast(*buf,(lzo_uint)nbytes,outbuf,&out_len,NULL);
#else
      status = lzo1x_decompress_safe(*buf,(lzo_uint)nbytes,outbuf,&out_len,NULL);
#endif
      if (status == LZO_E_OK) {
#ifdef DEBUG
	printf("decompressed %lu bytes back into %lu bytes\n",
	       (long) nbytes, (long) out_len);
#endif
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
    /* Compute the checksum */
    checksum=lzo_adler32(lzo_adler32(0,NULL,0), outbuf, out_len);

    /* Compare */
    if (memcmp(&checksum, (char*)(*buf)+nbytes, 4)) {
      ret_value = 0; /*fail*/
      fprintf(stderr,"Checksum failed!.\n");
      goto done;
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
/*     status = lzo1x_1_compress (z_src, z_src_nbytes, z_dst, &z_dst_nbytes, */
/* 			       wrkmem); */
    status = lzo1x_1_compress (z_src, z_src_nbytes, z_dst, &z_dst_nbytes,
			       wrkmem);
#ifdef CHECKSUM
    /* Append checksum of *uncompressed* data at the end */
    checksum = lzo_adler32(lzo_adler32(0,NULL,0), *buf, nbytes);
    memcpy((char*)(z_dst)+z_dst_nbytes, &checksum, 4);
    z_dst_nbytes += (lzo_uint)4;
    nbytes += 4; 
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

