/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted <francesc@blosc.io>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include <limits.h>
#ifdef __cplusplus
extern "C" {
#endif


#ifndef BLOSC_H
#define BLOSC_H

/* Version numbers */
#define BLOSC_VERSION_MAJOR    1    /* for major interface/format changes  */
#define BLOSC_VERSION_MINOR    4    /* for minor interface/format changes  */
#define BLOSC_VERSION_RELEASE  0    /* for tweaks, bug-fixes, or development */

#define BLOSC_VERSION_STRING   "1.4.0"  /* string version.  Sync with above! */
#define BLOSC_VERSION_REVISION "$Rev$"   /* revision version */
#define BLOSC_VERSION_DATE     "$Date:: 2014-07-04 #$"    /* date version */

#define BLOSCLZ_VERSION_STRING "1.0.2"   /* the internal compressor version */

/* The *_FORMAT symbols should be just 1-byte long */
#define BLOSC_VERSION_FORMAT    2   /* Blosc format version, starting at 1 */

/* Minimum header length */
#define BLOSC_MIN_HEADER_LENGTH 16

/* The maximum overhead during compression in bytes.  This equals to
   BLOSC_MIN_HEADER_LENGTH now, but can be higher in future
   implementations */
#define BLOSC_MAX_OVERHEAD BLOSC_MIN_HEADER_LENGTH

/* Maximum buffer size to be compressed */
#define BLOSC_MAX_BUFFERSIZE (INT_MAX - BLOSC_MAX_OVERHEAD)

/* Maximum typesize before considering buffer as a stream of bytes */
#define BLOSC_MAX_TYPESIZE 255         /* Cannot be larger than 255 */

/* The maximum number of threads (for some static arrays) */
#define BLOSC_MAX_THREADS 256

/* Codes for internal flags (see blosc_cbuffer_metainfo) */
#define BLOSC_DOSHUFFLE 0x1
#define BLOSC_MEMCPYED  0x2

/* Codes for the different compressors shipped with Blosc */
#define BLOSC_BLOSCLZ   0
#define BLOSC_LZ4       1
#define BLOSC_LZ4HC     2
#define BLOSC_SNAPPY    3
#define BLOSC_ZLIB      4

/* Names for the different compressors shipped with Blosc */
#define BLOSC_BLOSCLZ_COMPNAME   "blosclz"
#define BLOSC_LZ4_COMPNAME       "lz4"
#define BLOSC_LZ4HC_COMPNAME     "lz4hc"
#define BLOSC_SNAPPY_COMPNAME    "snappy"
#define BLOSC_ZLIB_COMPNAME      "zlib"

/* Codes for the different compression libraries shipped with Blosc */
#define BLOSC_BLOSCLZ_LIB   0
#define BLOSC_LZ4_LIB       1
#define BLOSC_SNAPPY_LIB    2
#define BLOSC_ZLIB_LIB      3

/* Names for the different compression libraries shipped with Blosc */
#define BLOSC_BLOSCLZ_LIBNAME   "BloscLZ"
#define BLOSC_LZ4_LIBNAME       "LZ4"
#define BLOSC_SNAPPY_LIBNAME    "Snappy"
#define BLOSC_ZLIB_LIBNAME      "Zlib"

/* The codes for compressor formats shipped with Blosc (code must be < 8) */
#define BLOSC_BLOSCLZ_FORMAT  BLOSC_BLOSCLZ_LIB
#define BLOSC_LZ4_FORMAT      BLOSC_LZ4_LIB
    /* LZ4HC and LZ4 share the same format */
#define BLOSC_LZ4HC_FORMAT    BLOSC_LZ4_LIB
#define BLOSC_SNAPPY_FORMAT   BLOSC_SNAPPY_LIB
#define BLOSC_ZLIB_FORMAT     BLOSC_ZLIB_LIB


/* The version formats for compressors shipped with Blosc */
/* All versions here starts at 1 */
#define BLOSC_BLOSCLZ_VERSION_FORMAT  1
#define BLOSC_LZ4_VERSION_FORMAT      1
#define BLOSC_LZ4HC_VERSION_FORMAT    1  /* LZ4HC and LZ4 share the same format */
#define BLOSC_SNAPPY_VERSION_FORMAT   1
#define BLOSC_ZLIB_VERSION_FORMAT     1


/**
  Initialize the Blosc library. You must call this previous to any
  other Blosc call, and make sure that you call this in a non-threaded
  environment.  Other Blosc calls can be called in a threaded
  environment, if desired.
  */
void blosc_init(void);


/**
  Destroy the Blosc library environment. You must call this after to
  you are done with all the Blosc calls, and make sure that you call
  this in a non-threaded environment.
  */
void blosc_destroy(void);


/**
  Compress a block of data in the `src` buffer and returns the size of
  compressed block.  The size of `src` buffer is specified by
  `nbytes`.  There is not a minimum for `src` buffer size (`nbytes`).

  `clevel` is the desired compression level and must be a number
  between 0 (no compression) and 9 (maximum compression).

  `doshuffle` specifies whether the shuffle compression preconditioner
  should be applied or not.  0 means not applying it and 1 means
  applying it.

  `typesize` is the number of bytes for the atomic type in binary
  `src` buffer.  This is mainly useful for the shuffle preconditioner.
  For implementation reasons, only a 1 < typesize < 256 will allow the
  shuffle filter to work.  When typesize is not in this range, shuffle
  will be silently disabled.

  The `dest` buffer must have at least the size of `destsize`.  Blosc
  guarantees that if you set `destsize` to, at least,
  (`nbytes`+BLOSC_MAX_OVERHEAD), the compression will always succeed.
  The `src` buffer and the `dest` buffer can not overlap.

  Compression is memory safe and guaranteed not to write the `dest`
  buffer more than what is specified in `destsize`.

  If `src` buffer cannot be compressed into `destsize`, the return
  value is zero and you should discard the contents of the `dest`
  buffer.

  A negative return value means that an internal error happened.  This
  should never happen.  If you see this, please report it back
  together with the buffer data causing this and compression settings.
  */
int blosc_compress(int clevel, int doshuffle, size_t typesize, size_t nbytes,
                   const void *src, void *dest, size_t destsize);


/**
  Decompress a block of compressed data in `src`, put the result in
  `dest` and returns the size of the decompressed block.

  The `src` buffer and the `dest` buffer can not overlap.

  Decompression is memory safe and guaranteed not to write the `dest`
  buffer more than what is specified in `destsize`.

  If an error occurs, e.g. the compressed data is corrupted or the
  output buffer is not large enough, then 0 (zero) or a negative value
  will be returned instead.
  */
int blosc_decompress(const void *src, void *dest, size_t destsize);


/**
  Get `nitems` (of typesize size) in `src` buffer starting in `start`.
  The items are returned in `dest` buffer, which has to have enough
  space for storing all items.

  Returns the number of bytes copied to `dest` or a negative value if
  some error happens.
  */
int blosc_getitem(const void *src, int start, int nitems, void *dest);


/**
  Initialize a pool of threads for compression/decompression.  If
  `nthreads` is 1, then the serial version is chosen and a possible
  previous existing pool is ended.  If this is not called, `nthreads`
  is set to 1 internally.

  Returns the previous number of threads.
  */
int blosc_set_nthreads(int nthreads);


/**
  Select the compressor to be used.  The supported ones are "blosclz",
  "lz4", "lz4hc", "snappy" and "zlib".  If this function is not
  called, then "blosclz" will be used.

  In case the compressor is not recognized, or there is not support
  for it in this build, it returns a -1.  Else it returns the code for
  the compressor (>=0).
  */
int blosc_set_compressor(const char* compname);


/**
  Get the `compname` associated with the `compcode`.

  If the compressor code is not recognized, or there is not support
  for it in this build, -1 is returned.  Else, the compressor code is
  returned.
 */
int blosc_compcode_to_compname(int compcode, char **compname);


/**
  Return the compressor code associated with the compressor name.

  If the compressor name is not recognized, or there is not support
  for it in this build, -1 is returned instead.
 */
int blosc_compname_to_compcode(const char *compname);


/**
  Get a list of compressors supported in the current build.  The
  returned value is a string with a concatenation of "blosclz", "lz4",
  "lz4hc", "snappy" or "zlib" separated by commas, depending on which
  ones are present in the build.

  This function does not leak, so you should not free() the returned
  list.

  This function should always succeed.
  */
char* blosc_list_compressors(void);


/**
  Get info from compression libraries included in the current build.
  In `compname` you pass the compressor name that you want info from.
  In `complib` and `version` you get the compression library name and
  version (if available) as output.

  In `complib` and `version` you get a pointer to the compressor
  library name and the version in string format respectively.  After
  using the name and version, you should free() them so as to avoid
  leaks.

  If the compressor is supported, it returns the code for the library
  (>=0).  If it is not supported, this function returns -1.
  */
int blosc_get_complib_info(char *compname, char **complib, char **version);


/**
  Free possible memory temporaries and thread resources.  Use this
  when you are not going to use Blosc for a long while.  In case of
  problems releasing the resources, it returns a negative number, else
  it returns 0.
  */
int blosc_free_resources(void);


/**
  Return information about a compressed buffer, namely the number of
  uncompressed bytes (`nbytes`) and compressed (`cbytes`).  It also
  returns the `blocksize` (which is used internally for doing the
  compression by blocks).

  You only need to pass the first BLOSC_MIN_HEADER_LENGTH bytes of a
  compressed buffer for this call to work.

  This function should always succeed.
  */
void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes,
                         size_t *cbytes, size_t *blocksize);


/**
  Return information about a compressed buffer, namely the type size
  (`typesize`), as well as some internal `flags`.

  The `flags` is a set of bits, where the currently used ones are:
    * bit 0: whether the shuffle filter has been applied or not
    * bit 1: whether the internal buffer is a pure memcpy or not

  You can use the `BLOSC_DOSHUFFLE` and `BLOSC_MEMCPYED` symbols for
  extracting the interesting bits (e.g. ``flags & BLOSC_DOSHUFFLE``
  says whether the buffer is shuffled or not).

  This function should always succeed.
  */
void blosc_cbuffer_metainfo(const void *cbuffer, size_t *typesize,
                            int *flags);


/**
  Return information about a compressed buffer, namely the internal
  Blosc format version (`version`) and the format for the internal
  Lempel-Ziv compressor used (`versionlz`).

  This function should always succeed.
  */
void blosc_cbuffer_versions(const void *cbuffer, int *version,
                            int *versionlz);


/**
  Return the compressor library/format used in a compressed buffer.

  This function should always succeed.
  */
char *blosc_cbuffer_complib(const void *cbuffer);



/*********************************************************************

  Low-level functions follows.  Use them only if you are an expert!

*********************************************************************/


/**
  Force the use of a specific blocksize.  If 0, an automatic
  blocksize will be used (the default).
  */
void blosc_set_blocksize(size_t blocksize);

#ifdef __cplusplus
}
#endif


#endif
