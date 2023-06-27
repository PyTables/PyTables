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

/* Registers the filter with the HDF5 library. */
#if defined(_MSC_VER)
__declspec(dllexport)
#endif	/* defined(_MSC_VER) */
int register_blosc2(char **version, char **date);

#ifdef __cplusplus
}
#endif

#endif
