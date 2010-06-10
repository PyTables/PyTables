/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted (faltet@pytables.org)

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/


#ifndef BLOSC_H
#define BLOSC_H

/* Version numbers */
#define BLOSC_VERSION_MAJOR    0    /* for major interface/format changes  */
#define BLOSC_VERSION_MINOR    9    /* for minor interface/format changes  */
#define BLOSC_VERSION_RELEASE  5    /* for tweaks, bug-fixes, or development */

#define BLOSC_VERSION_STRING   "0.9.5"  /* string version.  Sync with above! */
#define BLOSC_VERSION_REVISION "$Rev: 159 $"   /* revision version */
#define BLOSC_VERSION_DATE     "2010-06-09"    /* date version */

/* The *_VERS_FORMAT should be just 1-byte long */
#define BLOSC_VERSION_FORMAT    2   /* Blosc format version, starting at 1 */
#define BLOSCLZ_VERSION_FORMAT  1   /* Blosclz format version, starting at 1 */

/* The combined blosc and blosclz formats */
#define BLOSC_VERSION_CFORMAT (BLOSC_VERSION_FORMAT << 8) & (BLOSCLZ_VERSION_FORMAT)

/* The maximum overhead during compression in bytes */
#define BLOSC_MAX_OVERHEAD 16


/* Codes for internal flags */
#define BLOSC_DOSHUFFLE 0x1
#define BLOSC_MEMCPYED  0x2



/**
  Initialize a pool of threads for compression/decompression.  If
  `nthreads` is 1, then the serial version is chosen and a possible
  previous existing pool is ended.  Returns the previous number of
  threads.  If this is not called, `nthreads` is set to 1 internally.
*/

int blosc_set_nthreads(int nthreads);


/**
  Compress a block of data in the `src` buffer and returns the size of
  compressed block.  The size of `src` buffer is specified by
  `nbytes`.  There is not a minimum for `src` buffer size (`nbytes`).

  `clevel` is the desired compression level and must be a number
  between 0 (no compression) and 9 (maximum compression).

  `doshuffle` specifies whether the shuffle compression preconditioner
  should be applyied or not.  0 means not applying it and 1 means
  applying it.

  `typesize` is the number of bytes for the atomic type in binary
  `src` buffer.  This is mainly useful for the shuffle preconditioner.
  Only a typesize > 1 will allow the shuffle to work.

  The `dest` buffer must have at least the size of `maxbytes`.  Blosc
  guarantees that if you set `maxbytes` to, at least,
  (`nbytes`+BLOSC_MAX_OVERHEAD), the compression will always succeed.
  The `src` buffer and the `dest` buffer can not overlap.

  If `src` buffer cannot be compressed into `maxbytes`, the return
  value is zero and you should discard the contents of the `dest`
  buffer.

  A negative return value means that an internal error happened.  This
  should never happen.  If you see this, please report it back
  together with the buffer data causing this and compression settings.

  Compression is memory safe and guaranteed not to write the `dest`
  buffer more than what is specified in `maxbytes`.  However, it is
  not re-entrant and not thread-safe (despite the fact that it uses
  threads internally ;-)
 */

unsigned int blosc_compress(int clevel, int doshuffle, size_t typesize,
                            size_t nbytes, const void *src, void *dest,
                            size_t maxbytes);


/**
  Decompress a block of compressed data in `src`, put the result in
  `dest` and returns the size of the decompressed block. If error
  occurs, e.g. the compressed data is corrupted or the output buffer
  is not large enough, then 0 (zero) or a negative value will be
  returned instead.

  The `src` buffer and the `dest` buffer can not overlap.

  Decompression is memory safe and guaranteed not to write the `dest`
  buffer more than what is specified in `dest_size`.  However, it is
  not re-entrant and not thread-safe (despite the fact that it uses
  threads internally ;-)
*/

unsigned int blosc_decompress(const void *src, void *dest, size_t destsize);


/**
  Free possible memory temporaries and thread resources.  Use this
  when you are not going to use Blosc for a long while.
*/

void blosc_free_resources(void);


/**
  Return information about a compressed buffer, namely the number of
  uncompressed bytes (`nbytes`) and compressed (`cbytes`).  It also
  returns the `blocksize` (which is used internally for doing the
  compression by blocks).  This function should always succeed.
*/

void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes,
                         size_t *cbytes, size_t *blocksize);


/**
  Return information about a compressed buffer, namely the type size
  (`typesize`), as well as some internal `flags`.

  The `flags` is a set of bits, where the currently used ones are::
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
  Lempel-Ziv algorithm (`versionlz`).  This function should always
  succeed.
*/

void blosc_cbuffer_versions(const void *cbuffer, int *version,
                            int *versionlz);



/*********************************************************************

  Low-level functions follows.  Use them only if you are an expert!

*********************************************************************/


/**

  Force the use of a specific blocksize.  If 0, an automatic
  blocksize will be used (the default).

*/

void blosc_set_blocksize(size_t blocksize);


#endif
