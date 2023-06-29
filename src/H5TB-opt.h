#include <hdf5.h>

#ifdef __cplusplus
extern "C" {
#endif

// For H5Dchunk_iter() callback
typedef struct {
  size_t itemsize;
  size_t chunkshape;
  haddr_t *addrs;
} chunk_iter_op;


int fill_chunk_addrs(hid_t dataset_id, hsize_t nchunks, chunk_iter_op *chunk_op);
int clean_chunk_addrs(chunk_iter_op *chunk_op);

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
                        hbool_t blosc2_support,
                        const void *data );

herr_t H5TBOread_records( char *filename,
                          hbool_t blosc2_support,
                          chunk_iter_op chunk_op,
                          hid_t dataset_id,
                          hid_t mem_type_id,
                          hsize_t start,
                          hsize_t nrecords,
                          void *data );

herr_t read_records_blosc2( char* filename,
                            chunk_iter_op chunk_op,
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

herr_t H5TBOappend_records( hbool_t blosc2_support,
                            hid_t dataset_id,
                            hid_t mem_type_id,
                            hsize_t start,
                            hsize_t nrecords,
                            const void *data );

herr_t H5TBOwrite_records( hbool_t blosc2_support,
                           hid_t dataset_id,
                           hid_t mem_type_id,
                           hsize_t start,
                           hsize_t nrecords,
                           hsize_t step,
                           const void *data );

herr_t write_records_blosc2( hid_t dataset_id,
                             hid_t mem_type_id,
                             hsize_t start,
                             hsize_t nrecords,
                             const void *data );

herr_t write_chunks_blosc2( hid_t dataset_id,
                            hsize_t start,
                            hsize_t nrecords,
                            const void *data );

herr_t insert_chunk_blosc2( hid_t dataset_id,
                            hsize_t start,
                            hsize_t nrecords,
                            const void *data );

herr_t H5TBOwrite_elements( hid_t dataset_id,
                            hid_t mem_type_id,
                            hsize_t nrecords,
                            const void *coords,
                            const void *data );

herr_t H5TBOdelete_records( char* filename,
                            hbool_t blosc2_support,
                            chunk_iter_op chunk_op,
                            hid_t   dataset_id,
                            hid_t   mem_type_id,
                            hsize_t ntotal_records,
                            size_t  src_size,
                            hsize_t start,
                            hsize_t nrecords,
                            hsize_t maxtuples );

#ifdef __cplusplus
}
#endif
