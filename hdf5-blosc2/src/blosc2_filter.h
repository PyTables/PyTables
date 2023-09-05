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

/* Uses the Blosc2 library to decide an adequate block size for a chunk
 * of chunksize bytes (with elements of typesize bytes),
 * given the compression level clevel
 * for the compressor compcode (negative for the default).
 *
 * Return a negative value if there is some error. */
#if defined(_MSC_VER)
__declspec(dllexport)
#endif	/* defined(_MSC_VER) */
int32_t compute_blosc2_blocksize(int32_t chunksize, int32_t typesize,
                                 int clevel, int compcode);

#ifdef __cplusplus
}
#endif

#endif
