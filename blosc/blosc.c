/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted (faltet@pytables.org)
  Creation date: 2009-05-20

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <pthread.h>
#include "blosc.h"
#include "blosclz.h"
#include "shuffle.h"

#if defined(_WIN32) && !defined(__MINGW32__)
  #include <windows.h>
  #include "stdint-windows.h"
#else
  #include <stdint.h>
  #include <unistd.h>
  #include <inttypes.h>
#endif  /* _WIN32 */


/* Minimal buffer size to be compressed */
#define MIN_BUFFERSIZE 128       /* Cannot be smaller than 66 */

/* Maximum typesize before considering buffer as a stream of bytes. */
#define MAX_TYPESIZE 255         /* Cannot be larger than 255 */

/* The maximum number of splits in a block for compression */
#define MAX_SPLITS 16            /* Cannot be larger than 128 */

/* The maximum number of threads (for some static arrays) */
#define MAX_THREADS 64

/* Some useful units */
#define KB 1024
#define MB (1024*KB)

/* The size of L1 cache.  32 KB is quite common nowadays. */
#define L1 (32*KB)


/* Global variables for main logic */
int32_t init_temps_done = 0;    /* temporaries for compr/decompr initialized? */
size_t force_blocksize = 0;     /* should we force the use of a blocksize? */

/* Global variables for threads */
int32_t nthreads = 1;            /* number of desired threads in pool */
int32_t init_threads_done = 0;   /* pool of threads initialized? */
int32_t end_threads = 0;         /* should exisiting threads end? */
int32_t init_sentinels_done = 0; /* sentinels initialized? */
int32_t giveup_code;             /* error code when give up */
int32_t nblock;                  /* block counter */
pthread_t threads[MAX_THREADS];  /* opaque structure for threads */
int32_t tids[MAX_THREADS];       /* ID per each thread */
pthread_attr_t ct_attr;          /* creation time attributes for threads */

#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0
#define _POSIX_BARRIERS_MINE
#endif

/* Syncronization variables */
pthread_mutex_t count_mutex;
#ifdef _POSIX_BARRIERS_MINE
pthread_barrier_t barr_init;
pthread_barrier_t barr_finish;
#else
int32_t count_threads;
pthread_mutex_t count_threads_mutex;
pthread_cond_t count_threads_cv;
#endif


/* Structure for parameters in (de-)compression threads */
struct thread_data {
  size_t typesize;
  size_t blocksize;
  int32_t compress;
  int32_t clevel;
  int32_t flags;
  int32_t memcpyed;
  int32_t ntbytes;
  uint32_t nbytes;
  uint32_t maxbytes;
  uint32_t nblocks;
  uint32_t leftover;
  uint32_t *bstarts;             /* start pointers for each block */
  uint8_t *src;
  uint8_t *dest;
  uint8_t *tmp[MAX_THREADS];
  uint8_t *tmp2[MAX_THREADS];
} params;


/* Structure for parameters meant for keeping track of current temporaries */
struct temp_data {
  int32_t nthreads;
  size_t typesize;
  size_t blocksize;
} current_temp;



/* If `a` is little-endian, return it as-is.  If not, return a copy,
   with the endianness changed */
int32_t sw32(int32_t a)
{
  int32_t tmp;
  char *pa = (char *)&a;
  char *ptmp = (char *)&tmp;
  int i = 1;                    /* for big/little endian detection */
  char *p = (char *)&i;

  if (p[0] != 1) {
    /* big endian */
    ptmp[0] = pa[3];
    ptmp[1] = pa[2];
    ptmp[2] = pa[1];
    ptmp[3] = pa[0];
    return tmp;
  }
  else {
    /* little endian */
    return a;
  }
}


/* Shuffle & compress a single block */
static int blosc_c(size_t blocksize, int32_t leftoverblock,
                   uint32_t ntbytes, uint32_t maxbytes,
                   uint8_t *src, uint8_t *dest, uint8_t *tmp)
{
  size_t j, neblock, nsplits;
  int32_t cbytes;                   /* number of compressed bytes in split */
  int32_t ctbytes = 0;              /* number of compressed bytes in block */
  int32_t maxout;
  uint8_t *_tmp;
  size_t typesize = params.typesize;

  if ((params.flags & BLOSC_DOSHUFFLE) && (typesize > 1)) {
    /* Shuffle this block (this makes sense only if typesize > 1) */
    shuffle(typesize, blocksize, src, tmp);
    _tmp = tmp;
  }
  else {
    _tmp = src;
  }

  /* Compress for each shuffled slice split for this block. */
  /* If typesize is too large, neblock is too small or we are in a
     leftover block, do not split at all. */
  if ((typesize <= MAX_SPLITS) && (blocksize/typesize) >= MIN_BUFFERSIZE &&
      (!leftoverblock)) {
    nsplits = typesize;
  }
  else {
    nsplits = 1;
  }
  neblock = blocksize / nsplits;
  for (j = 0; j < nsplits; j++) {
    dest += sizeof(int32_t);
    ntbytes += sizeof(int32_t);
    ctbytes += sizeof(int32_t);
    maxout = neblock;
    if (ntbytes+maxout > maxbytes) {
      maxout = maxbytes - ntbytes;   /* avoid buffer overrun */
      if (maxout <= 0) {
        return 0;                  /* non-compressible block */
      }
    }
    cbytes = blosclz_compress(params.clevel, _tmp+j*neblock, neblock,
                              dest, maxout);
    if (cbytes >= maxout) {
      /* Buffer overrun caused by blosclz_compress (should never happen) */
      return -1;
    }
    else if (cbytes < 0) {
      /* cbytes should never be negative */
      return -2;
    }
    else if (cbytes == 0) {
      /* The compressor has been unable to compress data significantly. */
      /* Before doing the copy, check that we are not running into a
         buffer overflow. */
      if ((ntbytes+neblock) > maxbytes) {
        return 0;    /* Non-compressible data */
      }
      memcpy(dest, _tmp+j*neblock, neblock);
      cbytes = neblock;
    }
    ((uint32_t *)(dest))[-1] = sw32(cbytes);
    dest += cbytes;
    ntbytes += cbytes;
    ctbytes += cbytes;
  }  /* Closes j < nsplits */

  return ctbytes;
}


/* Decompress & unshuffle a single block */
static int blosc_d(size_t blocksize, int32_t leftoverblock,
                   uint8_t *src, uint8_t *dest, uint8_t *tmp, uint8_t *tmp2)
{
  int32_t j, neblock, nsplits;
  int32_t nbytes;                /* number of decompressed bytes in split */
  int32_t cbytes;                /* number of compressed bytes in split */
  int32_t ctbytes = 0;           /* number of compressed bytes in block */
  int32_t ntbytes = 0;           /* number of uncompressed bytes in block */
  uint8_t *_tmp;
  size_t typesize = params.typesize;

  if ((params.flags & BLOSC_DOSHUFFLE) && (typesize > 1)) {
    _tmp = tmp;
  }
  else {
    _tmp = dest;
  }

  /* Compress for each shuffled slice split for this block. */
  if ((typesize <= MAX_SPLITS) && (blocksize/typesize) >= MIN_BUFFERSIZE &&
      (!leftoverblock)) {
    nsplits = typesize;
  }
  else {
    nsplits = 1;
  }
  neblock = blocksize / nsplits;
  for (j = 0; j < nsplits; j++) {
    cbytes = sw32(((uint32_t *)(src))[0]);   /* amount of compressed bytes */
    src += sizeof(int32_t);
    ctbytes += sizeof(int32_t);
    /* Uncompress */
    if (cbytes == neblock) {
      memcpy(_tmp, src, neblock);
      nbytes = neblock;
    }
    else {
      nbytes = blosclz_decompress(src, cbytes, _tmp, neblock);
      if (nbytes != neblock) {
        return -2;
      }
    }
    src += cbytes;
    ctbytes += cbytes;
    _tmp += nbytes;
    ntbytes += nbytes;
  } /* Closes j < nsplits */

  if ((params.flags & BLOSC_DOSHUFFLE) && (typesize > 1)) {
    if ((uintptr_t)dest % 16 == 0) {
      /* 16-bytes aligned dest.  SSE2 unshuffle will work. */
      unshuffle(typesize, blocksize, tmp, dest);
    }
    else {
      /* dest is not aligned.  Use tmp2, which is aligned, and copy. */
      unshuffle(typesize, blocksize, tmp, tmp2);
      memcpy(dest, tmp2, blocksize);
    }
  }

  /* Return the number of uncompressed bytes */
  return ntbytes;
}


/* Serial version for compression/decompression */
int serial_blosc(void)
{
  uint32_t j, bsize, leftoverblock;
  int32_t cbytes;
  int32_t compress = params.compress;
  size_t blocksize = params.blocksize;
  int32_t ntbytes = params.ntbytes;
  int32_t flags = params.flags;
  uint32_t maxbytes = params.maxbytes;
  uint32_t nblocks = params.nblocks;
  int32_t leftover = params.nbytes % params.blocksize;
  uint32_t *bstarts = params.bstarts;
  uint8_t *src = params.src;
  uint8_t *dest = params.dest;
  uint8_t *tmp = params.tmp[0];     /* tmp for thread 0 */
  uint8_t *tmp2 = params.tmp2[0];   /* tmp2 for thread 0 */

  for (j = 0; j < nblocks; j++) {
    if (compress && !(flags & BLOSC_MEMCPYED)) {
      bstarts[j] = sw32(ntbytes);
    }
    bsize = blocksize;
    leftoverblock = 0;
    if ((j == nblocks - 1) && (leftover > 0)) {
      bsize = leftover;
      leftoverblock = 1;
    }
    if (compress) {
      if (flags & BLOSC_MEMCPYED) {
        /* We want to memcpy only */
        memcpy(dest+BLOSC_MAX_OVERHEAD+j*blocksize, src+j*blocksize, bsize);
        cbytes = bsize;
      }
      else {
        /* Regular compression */
        cbytes = blosc_c(bsize, leftoverblock, ntbytes, maxbytes,
                         src+j*blocksize, dest+ntbytes, tmp);
        if (cbytes == 0) {
          ntbytes = 0;              /* uncompressible data */
          break;
        }
      }
    }
    else {
      if (flags & BLOSC_MEMCPYED) {
        /* We want to memcpy only */
        memcpy(dest+j*blocksize, src+BLOSC_MAX_OVERHEAD+j*blocksize, bsize);
        cbytes = bsize;
      }
      else {
        /* Regular decompression */
        cbytes = blosc_d(bsize, leftoverblock,
                         src+sw32(bstarts[j]), dest+j*blocksize, tmp, tmp2);
      }
    }
    if (cbytes < 0) {
      ntbytes = cbytes;         /* error in blosc_c or blosc_d */
      break;
    }
    ntbytes += cbytes;
  }

  return ntbytes;
}


/* Threaded version for compression/decompression */
int parallel_blosc(void)
{
  int32_t rc;

  /* Synchronization point for all threads (wait for initialization) */
#ifdef _POSIX_BARRIERS_MINE
  rc = pthread_barrier_wait(&barr_init);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier (init)\n");
    exit(-1);
  }
#else
  pthread_mutex_lock(&count_threads_mutex);
  if (count_threads < nthreads) {
    count_threads++;
    pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
  }
  else {
    pthread_cond_broadcast(&count_threads_cv);
  }
  pthread_mutex_unlock(&count_threads_mutex);
#endif

  /* Synchronization point for all threads (wait for finalization) */
#ifdef _POSIX_BARRIERS_MINE
  rc = pthread_barrier_wait(&barr_finish);
  if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    printf("Could not wait on barrier (finish)\n");
    exit(-1);
  }
#else
  pthread_mutex_lock(&count_threads_mutex);
  if (count_threads > 0) {
    count_threads--;
    pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
  }
  else {
    pthread_cond_broadcast(&count_threads_cv);
  }
  pthread_mutex_unlock(&count_threads_mutex);
#endif

  if (giveup_code > 0) {
    /* Return the total bytes (de-)compressed in threads */
    return params.ntbytes;
  }
  else {
    /* Compression/decompression gave up.  Return error code. */
    return giveup_code;
  }
}


/* Convenience functions for creating and releasing temporaries */
void create_temporaries(void)
{
  int32_t tid;
  size_t typesize = params.typesize;
  size_t blocksize = params.blocksize;
  /* Extended blocksize for temporary destination.  Extended blocksize
   is only useful for compression in parallel mode, but it doesn't
   hurt other modes either. */
  size_t ebsize = blocksize + typesize*sizeof(int32_t);
  uint8_t *tmp, *tmp2;

  /* Create temporary area for each thread */
  for (tid = 0; tid < nthreads; tid++) {
#if defined(_WIN32)
    tmp = (uint8_t *)_aligned_malloc(blocksize, 16);
    tmp2 = (uint8_t *)_aligned_malloc(ebsize, 16);
#elif defined __APPLE__
    /* Mac OS X guarantees 16-byte alignment in small allocs */
    tmp = (uint8_t *)malloc(blocksize);
    tmp2 = (uint8_t *)malloc(ebsize);
#else
    posix_memalign((void **)&tmp, 16, blocksize);
    posix_memalign((void **)&tmp2, 16, ebsize);
#endif  /* _WIN32 */
    params.tmp[tid] = tmp;
    params.tmp2[tid] = tmp2;
  }

  init_temps_done = 1;
  /* Update params for current temporaries */
  current_temp.nthreads = nthreads;
  current_temp.typesize = typesize;
  current_temp.blocksize = blocksize;

}


void release_temporaries(void)
{
  int32_t tid;
  uint8_t *tmp, *tmp2;

  /* Release buffers */
  for (tid = 0; tid < nthreads; tid++) {
    tmp = params.tmp[tid];
    tmp2 = params.tmp2[tid];
#if defined(_WIN32)
    _aligned_free(tmp);
    _aligned_free(tmp2);
#else
    free(tmp);
    free(tmp2);
#endif  /* _WIN32 */
  }

  init_temps_done = 0;

}


/* Do the compression or decompression of the buffer depending on the
   global params. */
int do_job(void) {
  int32_t ntbytes;

  /* Initialize/reset temporaries if needed */
  if (!init_temps_done) {
    create_temporaries();
  }
  else if (current_temp.nthreads != nthreads ||
           current_temp.typesize != params.typesize ||
           current_temp.blocksize != params.blocksize) {
    release_temporaries();
    create_temporaries();
  }

  /* Run the serial version when nthreads is 1 or when the buffers are
     not much larger than blocksize */
  if (nthreads == 1 || (params.nbytes / params.blocksize) <= 1) {
    ntbytes = serial_blosc();
  }
  else {
    ntbytes = parallel_blosc();
  }

  return ntbytes;
}


size_t compute_blocksize(int32_t clevel, size_t typesize, size_t nbytes)
{
  size_t blocksize;

  blocksize = nbytes;           /* Start by a whole buffer as blocksize */

  if (force_blocksize) {
    blocksize = force_blocksize;
    /* Check that forced blocksize is not too small nor too large */
    if (blocksize < MIN_BUFFERSIZE) {
      blocksize = MIN_BUFFERSIZE;
    }
  }
  else if (nbytes >= L1*4) {
    blocksize = L1 * 4;
    if (clevel == 0) {
      blocksize /= 16;
    }
    else if (clevel <= 3) {
      blocksize /= 8;
    }
    else if (clevel <= 5) {
      blocksize /= 4;
    }
    else if (clevel <= 6) {
      blocksize /= 2;
    }
    else if (clevel < 9) {
      blocksize *= 1;
    }
    else {
      blocksize *= 2;
    }
  }

  /* Check that blocksize is not too large */
  if (blocksize > nbytes) {
    blocksize = nbytes;
  }

  /* blocksize must be a multiple of the typesize */
  blocksize = blocksize / typesize * typesize;

  return blocksize;
}


/* The public routine for compression.  See blosc.h for docstrings. */
unsigned int blosc_compress(int clevel, int doshuffle, size_t typesize,
                            size_t nbytes, const void *src, void *dest,
                            size_t maxbytes)
{
  uint8_t *_dest=NULL;         /* current pos for destination buffer */
  uint8_t *flags;              /* flags for header.  Currently booked:
                                  - 0: shuffled?
                                  - 1: memcpy'ed? */
  uint32_t nblocks;            /* number of total blocks in buffer */
  uint32_t leftover;           /* extra bytes at end of buffer */
  uint32_t *bstarts;           /* start pointers for each block */
  size_t blocksize;            /* length of the block in bytes */
  uint32_t ntbytes = 0;        /* the number of compressed bytes */
  uint32_t *ntbytes_;          /* placeholder for bytes in output buffer */

  /* Compression level */
  if (clevel < 0 || clevel > 9) {
    /* If clevel not in 0..9, print an error */
    fprintf(stderr, "`clevel` parameter must be between 0 and 9!\n");
    return -10;
  }

  /* Shuffle */
  if (doshuffle != 0 && doshuffle != 1) {
    fprintf(stderr, "`shuffle` parameter must be either 0 or 1!\n");
    return -10;
  }

  /* Get the blocksize */
  blocksize = compute_blocksize(clevel, typesize, nbytes);

  /* Compute number of blocks in buffer */
  nblocks = nbytes / blocksize;
  leftover = nbytes % blocksize;
  nblocks = (leftover>0)? nblocks+1: nblocks;

  /* Check typesize limits */
  if (typesize > MAX_TYPESIZE) {
    /* If typesize is too large, treat buffer as an 1-byte stream. */
    typesize = 1;
  }

  _dest = (uint8_t *)(dest);
  /* Write header for this block */
  _dest[0] = BLOSC_VERSION_FORMAT;         /* blosc format version */
  _dest[1] = BLOSCLZ_VERSION_FORMAT;       /* blosclz format version */
  flags = _dest+2;                         /* flags */
  _dest[2] = 0;                            /* zeroes flags */
  _dest[3] = (uint8_t)typesize;            /* type size */
  _dest += 4;
  ((uint32_t *)_dest)[0] = sw32(nbytes);   /* size of the buffer */
  ((uint32_t *)_dest)[1] = sw32(blocksize);/* block size */
  ntbytes_ = (uint32_t *)(_dest+8);        /* compressed buffer size */
  _dest += sizeof(int32_t)*3;
  bstarts = (uint32_t *)_dest;             /* starts for every block */
  _dest += sizeof(int32_t)*nblocks;        /* space for pointers to blocks */
  ntbytes = _dest - (uint8_t *)dest;

  if (clevel == 0) {
    /* Compression level 0 means buffer to be memcpy'ed */
    *flags |= BLOSC_MEMCPYED;
  }

  if (nbytes < MIN_BUFFERSIZE) {
    /* Buffer is too small.  Try memcpy'ing. */
    *flags |= BLOSC_MEMCPYED;
  }

  if (doshuffle == 1) {
    /* Shuffle is active */
    *flags |= BLOSC_DOSHUFFLE;              /* bit 0 set to one in flags */
  }

  /* Populate parameters for compression routines */
  params.compress = 1;
  params.clevel = clevel;
  params.flags = (int32_t)*flags;
  params.typesize = typesize;
  params.blocksize = blocksize;
  params.ntbytes = ntbytes;
  params.nbytes = nbytes;
  params.maxbytes = maxbytes;
  params.nblocks = nblocks;
  params.leftover = leftover;
  params.bstarts = bstarts;
  params.src = (uint8_t *)src;
  params.dest = (uint8_t *)dest;

  if (!(*flags & BLOSC_MEMCPYED)) {
    /* Do the actual compression */
    ntbytes = do_job();
    if ((ntbytes == 0) && (nbytes+BLOSC_MAX_OVERHEAD <= maxbytes)) {
      /* Last chance for fitting `src` buffer in `dest`.  Update flags
       and do a memcpy later on. */
      *flags |= BLOSC_MEMCPYED;
      params.flags |= BLOSC_MEMCPYED;
    }
  }

  if (*flags & BLOSC_MEMCPYED) {
    if (((nbytes % L1) == 0) || (nthreads > 1)) {
      /* More effective with large buffers that are multiples of the
       cache size or multi-cores */
      params.ntbytes = BLOSC_MAX_OVERHEAD;
      ntbytes = do_job();
    }
    else {
      memcpy((uint8_t *)dest+BLOSC_MAX_OVERHEAD, src, nbytes);
      ntbytes = nbytes + BLOSC_MAX_OVERHEAD;
    }
  }

  /* Set the number of compressed bytes in header */
  *ntbytes_ = sw32(ntbytes);

  assert((int32_t)ntbytes <= (int32_t)maxbytes);
  return ntbytes;
}


/* The public routine for decompression.  See blosc.h for docstrings. */
unsigned int blosc_decompress(const void *src, void *dest, size_t destsize)
{
  uint8_t *_src=NULL;            /* current pos for source buffer */
  uint8_t *_dest=NULL;           /* current pos for destination buffer */
  uint8_t version, versionlz;    /* versions for compressed header */
  uint8_t flags;                 /* flags for header */
  int32_t doshuffle = 0;         /* do unshuffle? */
  int32_t ntbytes;               /* the number of uncompressed bytes */
  uint32_t nblocks;              /* number of total blocks in buffer */
  uint32_t leftover;             /* extra bytes at end of buffer */
  uint32_t *bstarts;             /* start pointers for each block */
  uint32_t typesize, blocksize, nbytes, ctbytes;

  _src = (uint8_t *)(src);
  _dest = (uint8_t *)(dest);

  /* Read the header block */
  version = _src[0];                         /* blosc format version */
  versionlz = _src[1];                       /* blosclz format version */
  flags = _src[2];                           /* flags */
  typesize = (uint32_t)_src[3];              /* typesize */
  _src += 4;
  nbytes = sw32(((uint32_t *)_src)[0]);      /* buffer size */
  blocksize = sw32(((uint32_t *)_src)[1]);   /* block size */
  ctbytes = sw32(((uint32_t *)_src)[2]);     /* compressed buffer size */

  _src += sizeof(int32_t)*3;
  bstarts = (uint32_t *)_src;
  /* Compute some params */
  /* Total blocks */
  nblocks = nbytes / blocksize;
  leftover = nbytes % blocksize;
  nblocks = (leftover>0)? nblocks+1: nblocks;
  _src += sizeof(int32_t)*nblocks;

  /* Check zero typesizes.  From Blosc version format 2 on, this value
   has been reserved for future use. */
  if ((version == 1) && (typesize == 0)) {
    typesize = 256;             /* 0 means 256 in format version 1 */
  }

  if (nbytes > destsize) {
    /* This should never happen but just in case */
    return -1;
  }

  if (flags & BLOSC_DOSHUFFLE) {
    /* Input is shuffled.  Unshuffle it. */
    doshuffle = 1;
  }

  /* Populate parameters for decompression routines */
  params.compress = 0;
  params.clevel = 0;            /* specific for compression */
  params.flags = (int32_t)flags;
  params.typesize = typesize;
  params.blocksize = blocksize;
  params.ntbytes = 0;
  params.nbytes = nbytes;
  params.nblocks = nblocks;
  params.leftover = leftover;
  params.bstarts = bstarts;
  params.src = (uint8_t *)src;
  params.dest = (uint8_t *)dest;

  /* Check whether this buffer is memcpy'ed */
  if (flags & BLOSC_MEMCPYED) {
    if (((nbytes % L1) == 0) || (nthreads > 1)) {
      /* More effective with large buffers that are multiples of the
       cache size or multi-cores */
      ntbytes = do_job();
    }
    else {
      memcpy(dest, (uint8_t *)src+BLOSC_MAX_OVERHEAD, nbytes);
      ntbytes = nbytes;
    }
  }
  else {
    /* Do the actual decompression */
    ntbytes = do_job();
  }


  assert(ntbytes <= (int32_t)destsize);
  return ntbytes;
}


/* Decompress & unshuffle several blocks in a single thread */
void *t_blosc(void *tids)
{
  int32_t tid = *(int32_t *)tids;
  int32_t cbytes, ntdest;
  uint32_t tblocks;              /* number of blocks per thread */
  uint32_t leftover2;
  uint32_t tblock;               /* limit block on a thread */
  uint32_t nblock_;              /* private copy of nblock */
  int32_t rc;
  uint32_t bsize, leftoverblock;
  /* Parameters for threads */
  size_t blocksize;
  size_t ebsize;
  int32_t compress;
  uint32_t maxbytes;
  uint32_t ntbytes;
  uint32_t flags;
  uint32_t nblocks;
  uint32_t leftover;
  uint32_t *bstarts;
  uint8_t *src;
  uint8_t *dest;
  uint8_t *tmp;
  uint8_t *tmp2;

  while (1) {

    init_sentinels_done = 0;     /* sentinels have to be initialised yet */

    /* Meeting point for all threads (wait for initialization) */
#ifdef _POSIX_BARRIERS_MINE
    rc = pthread_barrier_wait(&barr_init);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      printf("Could not wait on barrier (init)\n");
      exit(-1);
    }
#else
    pthread_mutex_lock(&count_threads_mutex);
    if (count_threads < nthreads) {
      count_threads++;
      pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
    }
    else {
      pthread_cond_broadcast(&count_threads_cv);
    }
    pthread_mutex_unlock(&count_threads_mutex);
#endif

    /* Check if thread has been asked to return */
    if (end_threads) {
      return(0);
    }

    pthread_mutex_lock(&count_mutex);
    if (!init_sentinels_done) {
      /* Set sentinels and other global variables */
      giveup_code = 1;            /* no error code initially */
      nblock = -1;                /* block counter */
      init_sentinels_done = 1;    /* sentinels have been initialised */
    }
    pthread_mutex_unlock(&count_mutex);

    /* Get parameters for this thread before entering the main loop */
    blocksize = params.blocksize;
    ebsize = blocksize + params.typesize*sizeof(int32_t);
    compress = params.compress;
    flags = params.flags;
    maxbytes = params.maxbytes;
    nblocks = params.nblocks;
    leftover = params.leftover;
    bstarts = params.bstarts;
    src = params.src;
    dest = params.dest;
    tmp = params.tmp[tid];
    tmp2 = params.tmp2[tid];

    ntbytes = 0;                /* only useful for decompression */

    if (compress && !(flags & BLOSC_MEMCPYED)) {
      /* Compression always has to follow the block order */
      pthread_mutex_lock(&count_mutex);
      nblock++;
      nblock_ = nblock;
      pthread_mutex_unlock(&count_mutex);
      tblock = nblocks;
    }
    else {
      /* Decompression can happen using any order.  We choose
       sequential block order on each thread */

      /* Blocks per thread */
      tblocks = nblocks / nthreads;
      leftover2 = nblocks % nthreads;
      tblocks = (leftover2>0)? tblocks+1: tblocks;

      nblock_ = tid*tblocks;
      tblock = nblock_ + tblocks;
      if (tblock > nblocks) {
        tblock = nblocks;
      }
    }

    /* Loop over blocks */
    leftoverblock = 0;
    while ((nblock_ < tblock) && giveup_code > 0) {
      bsize = blocksize;
      if (nblock_ == (nblocks - 1) && (leftover > 0)) {
        bsize = leftover;
        leftoverblock = 1;
      }
      if (compress) {
        if (flags & BLOSC_MEMCPYED) {
          /* We want to memcpy only */
          memcpy(dest+BLOSC_MAX_OVERHEAD+nblock_*blocksize,
                 src+nblock_*blocksize, bsize);
          cbytes = bsize;
        }
        else {
          /* Regular compression */
          cbytes = blosc_c(bsize, leftoverblock, 0, ebsize,
                           src+nblock_*blocksize, tmp2, tmp);
        }
      }
      else {
        if (flags & BLOSC_MEMCPYED) {
          /* We want to memcpy only */
          memcpy(dest+nblock_*blocksize,
                 src+BLOSC_MAX_OVERHEAD+nblock_*blocksize, bsize);
          cbytes = bsize;
        }
        else {
          cbytes = blosc_d(bsize, leftoverblock,
                           src+sw32(bstarts[nblock_]), dest+nblock_*blocksize,
                           tmp, tmp2);
        }
      }

      /* Check whether current thread has to giveup */
      if (giveup_code <= 0) {
        break;
      }

      /* Check results for the compressed/decompressed block */
      if (cbytes < 0) {            /* compr/decompr failure */
        /* Set giveup_code error */
        pthread_mutex_lock(&count_mutex);
        giveup_code = cbytes;
        pthread_mutex_unlock(&count_mutex);
        break;
      }

      if (compress && !(flags & BLOSC_MEMCPYED)) {
        /* Start critical section */
        pthread_mutex_lock(&count_mutex);
        ntdest = params.ntbytes;
        bstarts[nblock_] = sw32(ntdest);    /* update block start counter */
        if ( (cbytes == 0) || (ntdest+cbytes > (int32_t)maxbytes) ) {
          giveup_code = 0;                  /* uncompressible buffer */
          pthread_mutex_unlock(&count_mutex);
          break;
        }
        nblock++;
        nblock_ = nblock;
        params.ntbytes += cbytes;           /* update return bytes counter */
        pthread_mutex_unlock(&count_mutex);
        /* End of critical section */

        /* Copy the compressed buffer to destination */
        memcpy(dest+ntdest, tmp2, cbytes);
      }
      else {
        nblock_++;
        /* Update counter for this thread */
        ntbytes += cbytes;
      }

    } /* closes while (nblock_) */

    /* Sum up all the bytes decompressed */
    if ((!compress || (flags & BLOSC_MEMCPYED)) && giveup_code > 0) {
      /* Update global counter for all threads (decompression only) */
      pthread_mutex_lock(&count_mutex);
      params.ntbytes += ntbytes;
      pthread_mutex_unlock(&count_mutex);
    }

    /* Meeting point for all threads (wait for finalization) */
#ifdef _POSIX_BARRIERS_MINE
    rc = pthread_barrier_wait(&barr_finish);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      printf("Could not wait on barrier (finish)\n");
      exit(-1);
    }
#else
    pthread_mutex_lock(&count_threads_mutex);
    if (count_threads > 0) {
      count_threads--;
      pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
    }
    else {
      pthread_cond_broadcast(&count_threads_cv);
    }
    pthread_mutex_unlock(&count_threads_mutex);
#endif

  }  /* closes while(1) */

  /* This should never be reached, but anyway */
  return(0);
}


int init_threads(void)
{
  int32_t tid, rc;

  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&count_mutex, NULL);

  /* Barrier initialization */
#ifdef _POSIX_BARRIERS_MINE
  pthread_barrier_init(&barr_init, NULL, nthreads+1);
  pthread_barrier_init(&barr_finish, NULL, nthreads+1);
#else
  pthread_mutex_init(&count_threads_mutex, NULL);
  pthread_cond_init(&count_threads_cv, NULL);
  count_threads = 0;      /* Reset threads counter */
#endif

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&ct_attr);
  pthread_attr_setdetachstate(&ct_attr, PTHREAD_CREATE_JOINABLE);

  /* Finally, create the threads in detached state */
  for (tid = 0; tid < nthreads; tid++) {
    tids[tid] = tid;
    rc = pthread_create(&threads[tid], &ct_attr, t_blosc, (void *)&tids[tid]);
    if (rc) {
      fprintf(stderr, "ERROR; return code from pthread_create() is %d\n", rc);
      fprintf(stderr, "\tError detail: %s\n", strerror(rc));
      exit(-1);
    }
  }

  init_threads_done = 1;                 /* Initialization done! */

  return(0);
}


int blosc_set_nthreads(int nthreads_new)
{
  int32_t nthreads_old = nthreads;
  int32_t t, rc;
  void *status;

  if (nthreads_new > MAX_THREADS) {
    fprintf(stderr, "Error.  nthreads cannot be larger than MAX_THREADS (%d)",
            MAX_THREADS);
    return -1;
  }
  else if (nthreads_new <= 0) {
    fprintf(stderr, "Error.  nthreads must be a positive integer");
    return -1;
  }
  else if (nthreads_new != nthreads) {
    if (nthreads > 1 && init_threads_done) {
      /* Tell all existing threads to finish */
      end_threads = 1;
#ifdef _POSIX_BARRIERS_MINE
      rc = pthread_barrier_wait(&barr_init);
      if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        printf("Could not wait on barrier (init)\n");
        exit(-1);
      }
#else
      pthread_mutex_lock(&count_threads_mutex);
      if (count_threads < nthreads) {
        count_threads++;
        pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
      }
      else {
        pthread_cond_broadcast(&count_threads_cv);
      }
      pthread_mutex_unlock(&count_threads_mutex);
#endif

      /* Join exiting threads */
      for (t=0; t<nthreads; t++) {
        rc = pthread_join(threads[t], &status);
        if (rc) {
          fprintf(stderr, "ERROR; return code from pthread_join() is %d\n", rc);
          fprintf(stderr, "\tError detail: %s\n", strerror(rc));
          exit(-1);
        }
      }
      init_threads_done = 0;
      end_threads = 0;
    }
    nthreads = nthreads_new;
    if (nthreads > 1) {
      /* Launch a new pool of threads */
      init_threads();
    }
  }
  return nthreads_old;
}


/* Free possible memory temporaries and thread resources */
void blosc_free_resources(void)
{
  int32_t t, rc;
  void *status;

  /* Release temporaries */
  if (init_temps_done) {
    release_temporaries();
  }

  /* Finish the possible thread pool */
  if (nthreads > 1 && init_threads_done) {
    /* Tell all existing threads to finish */
    end_threads = 1;
#ifdef _POSIX_BARRIERS_MINE
    rc = pthread_barrier_wait(&barr_init);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        printf("Could not wait on barrier (init)\n");
      exit(-1);
    }
#else
    pthread_mutex_lock(&count_threads_mutex);
    if (count_threads < nthreads) {
      count_threads++;
      pthread_cond_wait(&count_threads_cv, &count_threads_mutex);
    }
    else {
      pthread_cond_broadcast(&count_threads_cv);
    }
    pthread_mutex_unlock(&count_threads_mutex);
#endif

    /* Join exiting threads */
    for (t=0; t<nthreads; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
        fprintf(stderr, "ERROR; return code from pthread_join() is %d\n", rc);
        fprintf(stderr, "\tError detail: %s\n", strerror(rc));
        exit(-1);
      }
    }

    /* Release mutex and condition variable objects */
    pthread_mutex_destroy(&count_mutex);

    /* Barriers */
#ifdef _POSIX_BARRIERS_MINE
    pthread_barrier_destroy(&barr_init);
    pthread_barrier_destroy(&barr_finish);
#else
    pthread_mutex_destroy(&count_threads_mutex);
    pthread_cond_destroy(&count_threads_cv);
#endif

    /* Thread attributes */
    pthread_attr_destroy(&ct_attr);

    init_threads_done = 0;
    end_threads = 0;
  }
}


/* Return `nbytes`, `cbytes` and `blocksize` from a compressed buffer. */
void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes,
                         size_t *cbytes, size_t *blocksize)
{
  uint8_t *_src = (uint8_t *)(cbuffer);    /* current pos for source buffer */
  uint8_t version, versionlz;              /* versions for compressed header */

  /* Read the version info (could be useful in the future) */
  version = _src[0];                         /* blosc format version */
  versionlz = _src[1];                       /* blosclz format version */

  /* Read the interesting values */
  _src += 4;
  *nbytes = (size_t)sw32(((uint32_t *)_src)[0]);  /* uncompressed buffer size */
  *blocksize = (size_t)sw32(((uint32_t *)_src)[1]);   /* block size */
  *cbytes = (size_t)sw32(((uint32_t *)_src)[2]);  /* compressed buffer size */
}


/* Return `typesize` and `flags` from a compressed buffer. */
void blosc_cbuffer_metainfo(const void *cbuffer, size_t *typesize,
                            int *flags)
{
  uint8_t *_src = (uint8_t *)(cbuffer);  /* current pos for source buffer */
  uint8_t version, versionlz;            /* versions for compressed header */

  /* Read the version info (could be useful in the future) */
  version = _src[0];                     /* blosc format version */
  versionlz = _src[1];                   /* blosclz format version */

  /* Read the interesting values */
  *flags = (int)_src[2];                 /* flags */
  *typesize = (size_t)_src[3];           /* typesize */
}


/* Return version information from a compressed buffer. */
void blosc_cbuffer_versions(const void *cbuffer, int *version,
                            int *versionlz)
{
  uint8_t *_src = (uint8_t *)(cbuffer);  /* current pos for source buffer */

  /* Read the version info */
  *version = (int)_src[0];             /* blosc format version */
  *versionlz = (int)_src[1];           /* blosclz format version */
}


/* Force the use of a specific blocksize.  If 0, an automatic
   blocksize will be used (the default). */
void blosc_set_blocksize(size_t size)
{
  force_blocksize = size;
}

