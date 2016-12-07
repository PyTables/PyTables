#ifndef _H5ARRAY_H
#define _H5ARRAY_H

#include <hdf5.h>

#define TESTING(WHAT)   {printf("%-70s", "Testing " WHAT); fflush(stdout);}
#define PASSED()        {puts(" PASSED");fflush(stdout);}
#define H5_FAILED()     {puts("*FAILED*");fflush(stdout);}
#define SKIPPED()       {puts(" -SKIP-");fflush(stdout);}


#ifdef __cplusplus
extern "C" {
#endif

hid_t H5ARRAYmake(  hid_t loc_id,
                    const char *dset_name,
                    const char *obversion,
                    const int rank,
                    const hsize_t *dims,
                    int   extdim,
                    hid_t type_id,
                    hsize_t *dims_chunk,
                    void  *fill_data,
                    int   compress,
                    char  *complib,
                    int   shuffle,
                    int   fletcher32,
                    const void *data);

herr_t H5ARRAYappend_records( hid_t dataset_id,
                              hid_t type_id,
                              const int rank,
                              hsize_t *dims_orig,
                              hsize_t *dims_new,
                              int extdim,
                              const void *data );

herr_t H5ARRAYwrite_records( hid_t dataset_id,
                             hid_t type_id,
                             const int rank,
                             hsize_t *start,
                             hsize_t *step,
                             hsize_t *count,
                             const void *data );

herr_t H5ARRAYread( hid_t dataset_id,
                    hid_t type_id,
                    hsize_t start,
                    hsize_t nrows,
                    hsize_t step,
                    int extdim,
                    void *data );

herr_t H5ARRAYreadSlice( hid_t dataset_id,
                         hid_t type_id,
                         hsize_t *start,
                         hsize_t *stop,
                         hsize_t *step,
                         void *data );

herr_t H5ARRAYreadIndex( hid_t dataset_id,
                         hid_t type_id,
                         int   notequal,
                         hsize_t *start,
                         hsize_t *stop,
                         hsize_t *step,
                         void *data );

herr_t H5ARRAYget_ndims( hid_t dataset_id,
                         int *rank );

herr_t H5ARRAYget_info( hid_t dataset_id,
                        hid_t type_id,
                        hsize_t *dims,
                        hsize_t *maxdims,
                        H5T_class_t *class_id,
                        char *byteorder);

herr_t H5ARRAYget_chunkshape( hid_t dataset_id,
                              int rank,
                              hsize_t *dims_chunk);

herr_t H5ARRAYget_fill_value( hid_t dataset_id,
                              hid_t type_id,
                              int *status,
                              void *value);


#ifdef __cplusplus
}
#endif

#endif
