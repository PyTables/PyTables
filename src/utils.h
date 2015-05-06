#include "Python.h"
#include "numpy/arrayobject.h"
#include "hdf5.h"

/* Define this variable for error printings */
/*#define DEBUG 1 */
/* Define this variable for debugging printings */
/*#define PRINT 1 */
/* Define this for compile the main() function */
/* #define MAIN 1 */

/*
 * Status return values for the herr_t' type.
 * Since some unix/c routines use 0 and -1 (or more precisely, non-negative
 * vs. negative) as their return code, and some assumption had been made in
 * the code about that, it is important to keep these constants the same
 * values.  When checking the success or failure of an integer-valued
 * function, remember to compare against zero and not one of these two
 * values.
 */
#define SUCCEED         0
#define FAIL            (-1)
#define UFAIL           (unsigned)(-1)

/*
 * HDF Boolean type.
 */
#ifndef FALSE
#   define FALSE 0
#endif
#ifndef TRUE
#   define TRUE (!FALSE)
#endif

#ifdef H5_HAVE_WINDOWS
#define H5_HAVE_WINDOWS_DRIVER 1
#else
#define H5_HAVE_WINDOWS_DRIVER 0
#endif

#ifdef H5_HAVE_DIRECT
#define H5_HAVE_DIRECT_DRIVER 1
#else
#define H5_HAVE_DIRECT_DRIVER 0
#endif

#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR == 8 && H5_VERS_RELEASE >= 9) || (H5_VERS_MAJOR == 1 && H5_VERS_MINOR > 8)
/* HDF5 version >= 1.8.9 */
#define H5_HAVE_IMAGE_FILE 1
#else
/* HDF5 version < 1.8.9 */
#define H5_HAVE_IMAGE_FILE 0
#endif

/* COMAPTIBILITY: H5_VERSION_LE has been introduced in HDF5 1.8.7 */
#ifndef H5_VERSION_LE
#define H5_VERSION_LE(Maj,Min,Rel) \
       (((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR==Min) && (H5_VERS_RELEASE<=Rel)) || \
        ((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR<Min)) || \
        (H5_VERS_MAJOR<Maj))
#endif


/* Use %ld to print the value because long should cover most cases. */
/* Used to make certain a return value _is_not_ a value */
#define CHECK(ret, val, where) do {                                           \
    if (ret == val) {                                                         \
        printf("*** UNEXPECTED RETURN from %s is %ld at line %4d "            \
               "in %s\n", where, (long)ret, (int)__LINE__, __FILE__);         \
        H5Eprint(H5E_DEFAULT, stdout);                                                    \
    }                                                                         \
    H5Eclear(H5E_DEFAULT);                                                               \
} while(0)

int getLibrary(char *libname);

herr_t set_cache_size(hid_t file_id, size_t cache_size);

PyObject *_getTablesVersion(void);

/* PyObject *getZLIBVersionInfo(void); */

PyObject *getHDF5VersionInfo(void);

PyObject *createNamesTuple(char *buffer[], int nelements);

PyObject *get_filter_names( hid_t loc_id, const char *dset_name);

int get_objinfo(hid_t loc_id, const char *name);

int get_linkinfo(hid_t loc_id, const char *name);

PyObject *Giterate(hid_t parent_id, hid_t loc_id, const char *name);

PyObject *Aiterate(hid_t loc_id);

H5T_class_t getHDF5ClassID(hid_t loc_id,
                           const char *name,
                           H5D_layout_t *layout,
                           hid_t *type_id,
                           hid_t *dataset_id);

PyObject *H5UIget_info( hid_t loc_id,
                        const char *dset_name,
                        char *byteorder);

herr_t set_order(hid_t type_id, const char *byteorder);

int is_complex(hid_t type_id);

size_t get_complex_precision(hid_t type_id);

herr_t get_order(hid_t type_id, char *byteorder);

hid_t create_ieee_float16(const char *byteorder);

hid_t create_ieee_quadprecision_float(const char *byteorder);

hid_t create_ieee_complex64(const char *byteorder);

hid_t create_ieee_complex128(const char *byteorder);

hid_t create_ieee_complex192(const char *byteorder);

hid_t create_ieee_complex256(const char *byteorder);

hsize_t get_len_of_range(hsize_t lo, hsize_t hi, hsize_t step);

herr_t truncate_dset( hid_t dataset_id, const int maindim, const hsize_t size);

/* compatibility */
#ifndef H5_HAVE_DIRECT
herr_t pt_H5Pset_fapl_direct(hid_t fapl_id, size_t alignment,
                             size_t block_size, size_t cbuf_size);
#else /* H5_HAVE_DIRECT */
#define pt_H5Pset_fapl_direct H5Pset_fapl_direct
#endif

#ifndef H5_HAVE_WINDOWS
herr_t pt_H5Pset_fapl_windows(hid_t fapl_id);
#else /* H5_HAVE_WINDOWS */
#define pt_H5Pset_fapl_windows H5Pset_fapl_windows
#endif /* H5_HAVE_WINDOWS */


#if (H5_HAVE_IMAGE_FILE != 1)
/* HDF5 version >= 1.8.9 */
herr_t pt_H5Pset_file_image(hid_t fapl_id, void *buf_ptr, size_t buf_len);
ssize_t pt_H5Fget_file_image(hid_t file_id, void *buf_ptr, size_t buf_len);
#else /* (H5_HAVE_IMAGE_FILE != 1) */
/* HDF5 version < 1.8.9 */
#define pt_H5Pset_file_image H5Pset_file_image
#define pt_H5Fget_file_image H5Fget_file_image
#endif /* (H5_HAVE_IMAGE_FILE != 1) */


#if H5_VERSION_LE(1,8,12)
herr_t pt_H5free_memory(void *buf);
#else
#define pt_H5free_memory H5free_memory
#endif
