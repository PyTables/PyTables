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
#define B2ND_DEFAULT_BLOCKSIZE (1 << 18)

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

#ifdef __cplusplus
}
#endif

#endif
