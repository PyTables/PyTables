#include "hdf5.h"
#include <assert.h>
#include "H5PCORE-mem.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

hvl_t udata = {NULL, 0};

void *image_malloc(size_t size, H5FD_file_image_op_t file_image_op, void *udata) {
    ((hvl_t *) udata)->len = size;
    return (malloc(size));
}

void *image_memcpy(void *dest, const void *src, size_t size,
        H5FD_file_image_op_t file_image_op, void *udata) {
    return (NULL); /* always fails */
}

void *image_realloc(void *ptr, size_t size, H5FD_file_image_op_t file_image_op,
        void *udata) {
    ((hvl_t *) udata)->len = size;
    return (realloc(ptr, size));
}

herr_t image_free(void *ptr, H5FD_file_image_op_t file_image_op, void *udata) {
    ((hvl_t *) udata)->p = ptr;
    return (0); /* if we get here, we must have been successful */
}

void *udata_copy(void *udata) {
    return udata;
}

herr_t udata_free(void *udata) {
    return 0;
}
H5FD_file_image_callbacks_t callbacks = {image_malloc, image_memcpy,
    image_realloc, image_free,
    udata_copy, udata_free,
    (void *) (&udata)};

hid_t H5Fcreate_inmemory(hvl_t *udata)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_core(fapl,65536,false);
    callbacks.udata=udata;
    H5Pset_file_image_callbacks(fapl, &callbacks);
    hid_t file = H5Fcreate("in_memory", 0, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    return file;
}


#if HAVE_HDF5HL_LIB
#include "hdf5_hl.h"
int H5PCOREhasHDF5HL() {
	return true;
}

hid_t H5LTopen_file_image_proxy(void *buf_ptr, size_t buf_size, unsigned flags)
{
	return H5LTopen_file_image(buf_ptr, buf_size, flags);
}
#else
int H5PCOREhasHDF5HL() {
	return false;
}
hid_t H5LTopen_file_image_proxy(void *buf_ptr, size_t buf_size, unsigned flags)
{
	return -1;
}

#endif
