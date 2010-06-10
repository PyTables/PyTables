#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "hdf5.h"
#include "../blosc/blosc.h"
#include "H5Zblosc.h"


/* The conditional below is somewhat messy, but it is necessary because
  the THG team has decided to fix an API inconsistency in the definition
  of the H5Z_class_t structure in version 1.8.3 */
#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR < 7) || \
    (H5_USE_16_API && (H5_VERS_MAJOR > 1 || \
      (H5_VERS_MAJOR == 1 && (H5_VERS_MINOR > 8 || \
        (H5_VERS_MINOR == 8 && H5_VERS_RELEASE >= 3)))))
/* 1.6.x */
#define BLOSC_16API 1
#define PUSH_ERR(func, minor, str)  H5Epush(__FILE__, func, __LINE__, H5E_PLINE, minor, str)
#define GET_FILTER H5Pget_filter_by_id

#else
/* 1.8.x where x < 3 */
#define BLOSC_16API 0
#define PUSH_ERR(func, minor, str)  H5Epush1(__FILE__, func, __LINE__, H5E_PLINE, minor, str)
#define GET_FILTER(a,b,c,d,e,f,g) H5Pget_filter_by_id2(a,b,c,d,e,f,g,NULL)

#endif

size_t blosc_filter(unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t *buf_size, void **buf);

herr_t blosc_set_local(hid_t dcpl, hid_t type, hid_t space);


/* Register the filter, passing on the HDF5 return value */
int register_blosc(char **version, char **date){

    int retval;

#if BLOSC_16API
    H5Z_class_t filter_class = {
        (H5Z_filter_t)(FILTER_BLOSC),
        "blosc",
        NULL,
        (H5Z_set_local_func_t)(blosc_set_local),
        (H5Z_func_t)(blosc_filter)
    };
#else
    H5Z_class_t filter_class = {
        H5Z_CLASS_T_VERS,
        (H5Z_filter_t)(FILTER_BLOSC),
        1, 1,
        "blosc",
        NULL,
        (H5Z_set_local_func_t)(blosc_set_local),
        (H5Z_func_t)(blosc_filter)
    };
#endif

    retval = H5Zregister(&filter_class);
    if(retval<0){
        PUSH_ERR("register_blosc", H5E_CANTREGISTER, "Can't register BLOSC filter");
    }
    *version = strdup(BLOSC_VERSION_STRING);
    *date = strdup(BLOSC_VERSION_DATE);
    return 1; /* lib is available */
}

/*  Filter setup.  Records the following inside the DCPL:

    1.  If version information is not present, set slots 0 and 1 to the filter
        revision and BLOSC version, respectively.

    2. Compute the type size in bytes and store it in slot 2.

    3. Compute the chunk size in bytes and store it in slot 3.
*/
herr_t blosc_set_local(hid_t dcpl, hid_t type, hid_t space){

    int ndims;
    int i;
    herr_t r;

    unsigned int typesize;
    unsigned int bufsize;
    hsize_t chunkdims[32];

    unsigned int flags;
    size_t nelements = 8;
    unsigned int values[] = {0,0,0,0,0,0,0,0};

    r = GET_FILTER(dcpl, FILTER_BLOSC, &flags, &nelements, values, 0, NULL);
    if(r<0) return -1;

    if(nelements < 4) nelements = 4;  /* First 4 slots reserved.  If any higher
                                      slots are used, preserve the contents. */

    /* It seems the H5Z_FLAG_REVERSE flag doesn't work here, so we have to be
       careful not to clobber any existing version info */
    if(values[0]==0) values[0] = FILTER_BLOSC_VERSION;
    if(values[1]==0) values[1] = BLOSC_VERSION_CFORMAT;

    ndims = H5Pget_chunk(dcpl, 32, chunkdims);
    if(ndims<0) return -1;
    if(ndims>32){
        PUSH_ERR("blosc_set_local", H5E_CALLBACK, "Chunk rank exceeds limit");
        return -1;
    }

    typesize = H5Tget_size(type);
    if (typesize==0) return -1;
    values[2] = typesize;

    bufsize = typesize;
    for(i=0;i<ndims;i++){
        bufsize *= chunkdims[i];
    }
    values[3] = bufsize;

#ifdef BLOSC_DEBUG
    fprintf(stderr, "BLOSC: Computed buffer size %d\n", bufsize);
#endif

    r = H5Pmodify_filter(dcpl, FILTER_BLOSC, flags, nelements, values);
    if(r<0) return -1;

    return 1;
}


/* The filter function */
size_t blosc_filter(unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t *buf_size, void **buf){

    void* outbuf = NULL;
    int clevel = 6;
    int doshuffle = 1;
    size_t typesize = 4;
    size_t outbuf_size = 0;
    int status = 0;              /* Return code from blosc routines */

    if (cd_nelmts>=5) {
        typesize = cd_values[2];    /* The datatype size */
        clevel = cd_values[4];      /* The compression level */
        doshuffle = cd_values[5];   /* Shuffle? */
    }

    /* We're compressing */
    if(!(flags & H5Z_FLAG_REVERSE)){

        /* Allocate an output buffer exactly as long as the input data; if
           the result is larger, we simply return 0.  The filter is flagged
           as optional, so HDF5 marks the chunk as uncompressed and
           proceeds.
        */

        outbuf_size = (*buf_size);
        outbuf = malloc(outbuf_size);

        if(outbuf == NULL){
            PUSH_ERR("blosc_filter", H5E_CALLBACK,
                     "Can't allocate compression buffer");
            goto failed;
        }

        status = blosc_compress(clevel, doshuffle, typesize, nbytes,
                                *buf, outbuf, nbytes);
        /* printf("compress status: %d, %zd\n", status, nbytes); */
        if (status < 0) {
          PUSH_ERR("blosc_filter", H5E_CALLBACK, "BLOSC compression error");
          goto failed;
        }

    /* We're decompressing */
    } else {

        if((cd_nelmts>=4)&&(cd_values[3]!=0)){
            outbuf_size = cd_values[3];   /* Precomputed buffer guess */
        }else{
            outbuf_size = (*buf_size);
        }


#ifdef BLOSC_DEBUG
        fprintf(stderr, "Decompress %zd chunk w/buffer %zd\n", nbytes, outbuf_size);
#endif

        free(outbuf);
        outbuf = malloc(outbuf_size);

        if(outbuf == NULL){
          PUSH_ERR("blosc_filter", H5E_CALLBACK, "Can't allocate decompression buffer");
          goto failed;
        }

        status = blosc_decompress(*buf, outbuf, outbuf_size);
        /* printf("decompress status: %d, %zd\n", status, nbytes); */

        if(status <= 0){    /* decompression failed */
          PUSH_ERR("blosc_filter", H5E_CALLBACK, "BLOSC decompression error");
          goto failed;
        } /* if !status */

    } /* compressing vs decompressing */

    if(status != 0){
        free(*buf);
        *buf = outbuf;
        *buf_size = outbuf_size;
        return status;  /* Size of compressed/decompressed data */
    }

 failed:
    free(outbuf);
    return 0;

} /* End filter function */
