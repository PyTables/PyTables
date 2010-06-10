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
 *  * HDF Boolean type.
 *  */
#ifndef FALSE
#   define FALSE 0
#endif
#ifndef TRUE
#   define TRUE (!FALSE)
#endif


/* Use %ld to print the value because long should cover most cases. */
/* Used to make certain a return value _is_not_ a value */
#define CHECK(ret, val, where) do {                                           \
    if (ret == val) {                                                         \
        printf("*** UNEXPECTED RETURN from %s is %ld at line %4d "            \
               "in %s\n", where, (long)ret, (int)__LINE__, __FILE__);         \
        H5Eprint (stdout);                                                    \
    }                                                                         \
    H5Eclear();                                                               \
} while(0)

int getLibrary(char *libname);

herr_t set_cache_size(hid_t file_id, size_t cache_size);

PyObject *_getTablesVersion(void);

/* PyObject *getZLIBVersionInfo(void); */

PyObject *getHDF5VersionInfo(void);

PyObject *createNamesTuple(char *buffer[], int nelements);

PyObject *get_filter_names( hid_t loc_id, const char *dset_name);

int get_objinfo(hid_t loc_id, const char *name);

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

hsize_t getIndicesExt(PyObject *s, hsize_t length,
		      hssize_t *start, hssize_t *stop, hssize_t *step,
		      hsize_t *slicelength);

herr_t set_order(hid_t type_id, const char *byteorder);

int is_complex(hid_t type_id);

size_t get_complex_precision(hid_t type_id);

herr_t get_order(hid_t type_id, char *byteorder);

hid_t create_ieee_complex64(const char *byteorder);

hid_t create_ieee_complex128(const char *byteorder);

hsize_t get_len_of_range(hsize_t lo, hsize_t hi, hsize_t step);

herr_t truncate_dset( hid_t dataset_id, const int maindim, const hsize_t size);

