#ifndef FILTER_BLOSC_H
#define FILTER_BLOSC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Filter revision number, starting at 1 */
#define FILTER_BLOSC_VERSION 1

/* Filter ID registered with the HDF Group */
#define FILTER_BLOSC 32001

/* Register the filter with the library */
int register_blosc(char **version, char **date);

#ifdef __cplusplus
}
#endif

#endif

