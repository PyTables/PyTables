#ifndef FILTER_BLOSC2_H
#define FILTER_BLOSC2_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blosc2.h"

/* Filter revision number, starting at 1 */
#define FILTER_BLOSC2_VERSION 1

/* Filter ID registered with the HDF Group */
/* See https://portal.hdfgroup.org/display/support/Filters */
#define FILTER_BLOSC2 32026

/* 256KiB should let both the decompressed and the compressed blocks fit in
   the L2 cache of most current CPUs,
   while resulting in better compression ratios than smaller sizes
   and less overhead because of excessive partitioning. */
#define B2ND_DEFAULT_BLOCK_SIZE (1 << 18)

/* An opaque NumPy data type format for B2ND that respects the type size.
 * The actual type is irrelevant since HDF5 already stores it. */
#define B2ND_OPAQUE_NPDTYPE_FORMAT "|V%zd"

/* The maximum size of a formatted NumPy data type string:
 * "|V18446744073709551616\0". */
#define B2ND_OPAQUE_NPDTYPE_MAXLEN (2 + 20 + 1)

/* Registers the filter with the HDF5 library. */
#if defined(_MSC_VER)
__declspec(dllexport)
#endif	/* defined(_MSC_VER) */
int register_blosc2(char **version, char **date);

/* Get the maximum block size which is not greater than the given block_size
 * and fits within the given chunk dimensions dims_chunk. A zero block_size
 * means using an automatic value that fits most L2 CPU caches.
 *
 * Block dimensions start with 2 (unless the respective chunk dimension is 1),
 * and are doubled starting from the innermost (rightmost) ones, to leverage
 * the locality of C array arrangement.  The resulting block dimensions are
 * placed in the last (output) argument.
 *
 * Based on Python-Blosc2's blosc2.core.compute_chunks_blocks and
 * compute_partition.
 */
#if defined(_MSC_VER)
__declspec(dllexport)
#endif	/* defined(_MSC_VER) */
int32_t compute_b2nd_block_shape(size_t block_size,  // desired target, 0 for auto
                                 size_t type_size,
                                 const int rank,
                                 const int32_t *dims_chunk,
                                 int32_t *dims_block);

#ifdef __cplusplus
}
#endif

#endif
