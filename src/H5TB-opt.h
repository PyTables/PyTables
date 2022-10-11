#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

hid_t H5TBOmake_table(  const char *table_title,
                        hid_t loc_id,
                        const char *dset_name,
                        char *version,
                        const char *class_,
                        hid_t type_id,
                        hsize_t nrecords,
                        hsize_t chunk_size,
                        hsize_t block_size,
                        void  *fill_data,
                        int compress,
                        char *complib,
                        int shuffle,
                        int fletcher32,
                        hbool_t track_times,
                        const void *data );

herr_t H5TBOread_records( char *filename,
                          hbool_t native_order,
                          hid_t dataset_id,
                          hid_t mem_type_id,
                          hsize_t start,
                          hsize_t nrecords,
                          void *data );

herr_t read_records_blosc2( char* filename,
                            hid_t dataset_id,
                            hid_t mem_type_id,
                            hid_t space_id,
                            hsize_t start,
                            hsize_t nrecords,
                            uint8_t *data );

herr_t H5TBOread_elements( hid_t dataset_id,
                           hid_t mem_type_id,
                           hsize_t nrecords,
                           void *coords,
                           void *data );

herr_t H5TBOappend_records( hid_t dataset_id,
                            hid_t mem_type_id,
                            hsize_t nrecords,
                            hsize_t nrecords_orig,
                            const void *data );

herr_t append_records_blosc2( hid_t dataset_id,
                              hsize_t nrecords,
                              const void *data );

herr_t H5TBOwrite_records( hid_t dataset_id,
                           hid_t mem_type_id,
                           hsize_t start,
                           hsize_t nrecords,
                           hsize_t step,
                           const void *data );

herr_t H5TBOwrite_elements( hid_t dataset_id,
                            hid_t mem_type_id,
                            hsize_t nrecords,
                            const void *coords,
                            const void *data );

herr_t H5TBOdelete_records( char* filename,
                            hbool_t native_order,
                            hid_t   dataset_id,
                            hid_t   mem_type_id,
                            hsize_t ntotal_records,
                            size_t  src_size,
                            hsize_t start,
                            hsize_t nrecords,
                            hsize_t maxtuples );
