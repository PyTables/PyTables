#include <Python.h>
#include <numarray/numarray.h>
#include <hdf5.h>

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

/* Maximum dimensions in an array */
/* This number has been taken from numarray/numarray.h */ 
#define MAXDIM  40

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

PyObject *_getTablesVersion(void);

/* PyObject *getZLIBVersionInfo(void); */

PyObject *getHDF5VersionInfo(void);

PyObject *createNamesTuple(char *buffer[], int nelements);

PyObject *get_filter_names( hid_t loc_id, const char *dset_name);

PyObject *Giterate(hid_t parent_id, hid_t loc_id, const char *name);

PyObject *Aiterate(hid_t loc_id);

H5T_class_t getHDF5ClassID(hid_t loc_id,
			   const char *name,
			   H5D_layout_t *layout);

PyObject *H5UIget_info( hid_t loc_id, 
			const char *dset_name,
			char *byteorder);

int GetIndicesEx(PyObject *s, hsize_t length,
		 int *start, int *stop, int *step,
		 int *slicelength);

PyObject *get_attribute_string_sys( hid_t loc_id,
				    const char *obj_name,
				    const char *attr_name);


herr_t set_order(hid_t type_id, const char *byteorder);

int is_complex(hid_t type_id);

size_t get_complex_precision(hid_t type_id);

H5T_order_t get_order(hid_t type_id);

hid_t create_native_complex64(const char *byteorder);

hid_t create_native_complex32(const char *byteorder);

