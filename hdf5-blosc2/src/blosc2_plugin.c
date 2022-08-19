/*
 * Dynamically loaded filter plugin for HDF5 blosc2 filter.
 *
 * For compiling, use:
 * $ h5cc -fPIC -shared blosc2_plugin.c blosc2_filter.c -o libH5Zblosc2.so -lblosc2
 *
 */


#include "blosc2_plugin.h"
#include "blosc2_filter.h"


/* Prototypes for filter function in blosc2_filter.c. */
size_t blosc2_filter_function(unsigned flags, size_t cd_nelmts,
                              const unsigned cd_values[], size_t nbytes,
                              size_t* buf_size, void** buf);

herr_t blosc2_set_local(hid_t dcpl, hid_t type, hid_t space);


H5Z_class_t blosc2_H5Filter[1] = {
    {
        H5Z_CLASS_T_VERS,
        (H5Z_filter_t)(FILTER_BLOSC2),
        1,                   /* encoder_present flag (set to true) */
        1,                   /* decoder_present flag (set to true) */
        "blosc2",
        /* Filter info  */
        NULL,                           /* The "can apply" callback */
        (H5Z_set_local_func_t)(blosc2_set_local), /* The "set local" callback */
        (H5Z_func_t)(blosc2_filter_function),    /* The filter function */
    }
};


H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_FILTER; }


const void* H5PLget_plugin_info(void) { return blosc2_H5Filter; }
