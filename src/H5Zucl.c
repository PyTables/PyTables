/* Version of ucl compressor/decompressor optimized for
   decompression. This works if and only if the HDF5 layer gives a
   large enough buffer to keep the uncompressed data in. If not, it
   should crash. But HDF5 seems to always provide room enough */
#include <stdlib.h>

#ifdef HAVE_UCL_LIB
#   include "ucl/ucl.h"
#   include "H5Zucl.h"
#endif

#undef CHECKSUM

int register_ucl(void) {

#ifdef HAVE_UCL_LIB

   int status;

  /* Init the ucl library */
  if (ucl_init()!=UCL_E_OK)
    printf("Problems initializing UCL library\n");

  /* Register the ucl compressor */
  status = H5Zregister(FILTER_UCL, "ucl deflate", ucl_deflate);
  
  return 1; /* lib is available */

#else
  return 0; /* lib is not available */
#endif /* HAVE_UCL_LIB */

}

size_t ucl_deflate(unsigned int flags, size_t cd_nelmts,
		   const unsigned int cd_values[], size_t nbytes,
		   size_t *buf_size, void **buf)
{
  size_t ret_value = 0;
#ifdef HAVE_UCL_LIB
  int status;
  size_t  nalloc = *buf_size;
  ucl_uint out_len = (ucl_uint) nalloc;
  void *outbuf;
  int complevel = 1;
#ifdef CHECKSUM
  ucl_uint32 checksum;
#endif

  /* Check arguments */
  if (cd_nelmts!=1 || cd_values[0]>9) {
    printf("invalid deflate aggression level");
  }

  complevel = cd_values[0];

  if (flags & H5Z_FLAG_REVERSE) {
    /* Input */

    /* Only allocate the bytes for the outbuf */
    if (NULL==(outbuf = malloc(nalloc))) {
      printf("memory allocation failed for deflate uncompression");
    }

#ifdef CHECKSUM
    nbytes -= 4;
#endif

    /* The assembler version of the decompression routine is 25%
       faster than the C version.  However, this is not automatically
       included on the UCL library (you have to add it by hand), so it
       is safer to call the C one. */
#if defined __i386__ && 0  /* Change this to 1 if you have the assembler version */
    status = ucl_nrv2e_decompress_asm_fast_8(*buf,(ucl_uint)nbytes,outbuf,&out_len,NULL);
#else
    status = ucl_nrv2e_decompress_8(*buf,(ucl_uint)nbytes,outbuf,&out_len,NULL);
#endif
 
    if (status != UCL_E_OK) {
      /* this should NEVER happen */
      fprintf(stderr, "internal error - decompression failed: %d\n", status);
      ret_value = 0; /* fail */
      goto done;
    }

#ifdef CHECKSUM
    /* Compute the checksum */
    checksum=ucl_adler32(ucl_adler32(0,NULL,0), outbuf, out_len);

    /* Compare */
    if (memcmp(&checksum, (char*)(*buf)+nbytes, 4)) {
      ret_value = 0; /*fail*/
      fprintf(stderr,"Checksum failed!.\n");
      goto done;
    }
#endif /* CHECKSUM */

    ucl_free(*buf);
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
    ucl_byte *z_src = (ucl_byte*)(*buf);
    ucl_byte *z_dst;         /*destination buffer            */
    ucl_uint z_src_nbytes = (ucl_uint)(nbytes);
    /* ucl_uint z_dst_nbytes = (ucl_uint)(nbytes + (nbytes / 64) + 16 + 3 + 4); */
    ucl_uint z_dst_nbytes = (ucl_uint)(nbytes + (nbytes / 8) + 256 + 4);

    if (NULL==(z_dst=outbuf=malloc(z_dst_nbytes))) {
	fprintf(stderr, "unable to allocate deflate destination buffer");
	ret_value = 0; /* fail */
	goto done;
    }

    /* Compress this buffer */
    status = ucl_nrv2e_99_compress(z_src,z_src_nbytes,z_dst,&z_dst_nbytes,0,complevel,NULL,NULL);
 
#ifdef CHECKSUM
    /* Append checksum of *uncompressed* data at the end */
    checksum = ucl_adler32(ucl_adler32(0,NULL,0), *buf, nbytes);
    memcpy((char*)(z_dst)+z_dst_nbytes, &checksum, 4);
    z_dst_nbytes += (ucl_uint)4;
    nbytes += 4; 
#endif /* CHECKSUM */

    if (z_dst_nbytes >= nbytes) {
      /* fprintf(stderr,"overflow"); */
      ret_value = 0; /* fail */
      goto done;
    } else if (UCL_E_OK != status) {
      /* fprintf(stderr,"ucl error"); */
      ret_value = 0; /* fail */
      goto done;
    } else {
      ucl_free(*buf);
      *buf = outbuf;
      outbuf = NULL;
      *buf_size = z_dst_nbytes;
      ret_value = z_dst_nbytes;
    }
  }

#endif  /* HAVE_UCL_LIB */

done:
  if(outbuf)
    ucl_free(outbuf);
  return ret_value;
}

