/* Version of ucl compressor/decompressor optimized for
   decompression. This works if and only if the HDF5 layer gives a
   large enough buffer to keep the uncompressed data in. If not, it
   should crash. But HDF5 seems to always provide room enough */
#include <stdlib.h>
#include "H5Zucl.h"
#include "utils.h"
#include "math.h"  		/* For ceil() */

#ifdef HAVE_UCL_LIB
#   include "ucl/ucl.h"
#endif

/* CHECKSUM symbol adds some safety to the code, but does not seems
   actually necessary for common systems. It can be activated under
   special scenarios (like transmitting through a net with no error
   correction capabilities).  Anyway, this takes only a 1% more of
   space and a 2% more of CPU, which is almost negligible.
   F. Alted 2003/07/22
 * Undef CHECKSUM for pytables 0.5 backward compatibility.
 * F. Alted 2003/07/28
*/
#undef CHECKSUM  		       

#undef DEBUG

/* Adding more memory to the nrve seems to make it more resistant to
 seg faults. But I don't fully understand were is exactly the problem,
 anyway.  F. Alted 2003/07/24 
 * 
 * I'm suspucious now about a possible interaction between psyco and
 * ucl nrve that makes the checksum to fail. 2003/07/28
 * 
 */
/* Adding a combination of the zlib method and ucl to the output buffer */
#define H5Z_UCL_SIZE_ADJUST(s) (ceil((double)((s)*1.001))+((s)/8)+256+12)
/* #define H5Z_UCL_SIZE_ADJUST(s) ((s)+((s)/8)+256) */ /* Old value */

int register_ucl(void) {

#ifdef HAVE_UCL_LIB

   int status;

  /* Init the ucl library */
  if (ucl_init()!=UCL_E_OK)
    printf("Problems initializing UCL library\n");

  /* Feed the filter_class data structure */
  H5Z_class_t filter_class = {
    (H5Z_filter_t)FILTER_UCL,	/* filter_id */
    "ucl deflate", 		/* comment */
    NULL,                       /* can_apply_func */
    NULL,                       /* set_local_func */
    (H5Z_func_t)ucl_deflate     /* filter_func */
  };

  /* Register the lzo compressor */
  status = H5Zregister(&filter_class);
  
  return UCL_VERSION; /* lib is available */

#else
  return 0; /* lib is not available */
#endif /* HAVE_UCL_LIB */

}

/* This routine only can be called if UCL is present */
PyObject *getUCLVersionInfo(void) {
  char *info[2];
#ifdef HAVE_UCL_LIB
  info[0] = strdup(UCL_VERSION_STRING);
  info[1] = strdup(UCL_VERSION_DATE);
#else
  info[0] = NULL;
  info[1] = NULL;
#endif /* HAVE_UCL_LIB */
  
  return createNamesTuple(info, 2);
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
  /* max_len_buffer will keep the likely output buffer size
     after processing the first chunk */
  static unsigned int max_len_buffer = 0;
  ucl_uint32 checksum;

  /* Check arguments */
  if (cd_nelmts!=1 || cd_values[0]>9) {
    printf("invalid deflate aggression level");
  }

  complevel = cd_values[0];

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
    nbytes -= 4;
#endif
    while(1) {
      /* The assembler version of the decompression routine is 25%
	 faster than the C version.  However, this is not automatically
	 included on the UCL library (you have to add it by hand), so it
	 is safer to call the C one. */

#ifdef DEBUG
      printf("nbytes -->%d\n", nbytes);
      printf("nalloc -->%d\n", nalloc);
      printf("max_len_buffer -->%d\n", max_len_buffer);
#endif /* DEBUG */
      status = ucl_nrv2e_decompress_safe_8(*buf, (ucl_uint)nbytes, outbuf,
					   &out_len, NULL);
      /* Check if success */
      if (status == UCL_E_OK) {
#ifdef DEBUG
	printf("decompressed %lu bytes back into %lu bytes\n",
	       (long) nbytes, (long) out_len);
#endif
	max_len_buffer = out_len;
	break; /* done */
      }
      /* If not success, double the buffer size and try again */
      else if (status == UCL_E_OUTPUT_OVERRUN) {
	nalloc *= 2;
	out_len = (ucl_uint) nalloc;
	if (NULL==(outbuf = realloc(outbuf, nalloc))) {
	  printf("memory allocation failed for ucl uncompression");
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
#ifdef CHECKSUM
    ucl_uint z_dst_nbytes = (ucl_uint)H5Z_UCL_SIZE_ADJUST(nbytes)+4;
#else
    ucl_uint z_dst_nbytes = (ucl_uint)H5Z_UCL_SIZE_ADJUST(nbytes);
#endif

    if (NULL==(z_dst=outbuf=(void *)ucl_malloc(z_dst_nbytes))) {
	fprintf(stderr, "unable to allocate deflate destination buffer");
	ret_value = 0; /* fail */
	goto done;
    }

    /* Compress this buffer */

    /* The ucl_nrv2e version of the UCL compressor seems to give some
       segmentation faults in certain scenarios.  

       Way to make this code to crash (with the nrv2e compressor):

       $ python table-bench.py -p -l ucl -c 1 -s small -i 1000000 test.h5
       
       The nrv2b and nrv2d seems to give no problems. I'm adopting the
       nrv2d which is slightly better (more compression) than
       nrv2b. The best is, though, the nrv2e.

       F. Alted 2003/07/22 
       
       New note: I've discovered that adding some more space to the
       nrv2e compressor, it seems to work fine. I¡m pretty sure that
       this does not solve the problem, it just makes the seg faults
       harder to appear. I'm adopting this strategy in order to keep
       backward compatibility with the existing files writen with
       pytables 0.5.x.  F. Alted 2003/07/24

*/

    status = ucl_nrv2e_99_compress(z_src, z_src_nbytes, z_dst, &z_dst_nbytes,
				   0, complevel, NULL, NULL);
 
#ifdef CHECKSUM
    /* Append checksum of *uncompressed* data at the end */
    checksum = ucl_adler32(ucl_adler32(0,NULL,0), *buf, nbytes);
    memcpy((char*)(z_dst)+z_dst_nbytes, &checksum, 4);
    z_dst_nbytes += (ucl_uint)4;
    nbytes += 4; 
#endif /* CHECKSUM */

    if (z_dst_nbytes >= nbytes) {
/*       fprintf(stderr,"overflow"); */
      ret_value = 0; /* Incompressible chunk */
      goto done;
    } else if (UCL_E_OK != status) {
      /* This should never happen! */
      fprintf(stderr,"ucl error!. This should not happen!.\n");
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

done:
  if(outbuf)
    ucl_free(outbuf);
#endif  /* HAVE_UCL_LIB */

  return ret_value;
}

