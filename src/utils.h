/* Include file for calcoffset.c 
 * F. Alted 
 * 2002/09/17 */

#include "Python.h"
#include "hdf5.h"
#include "H5private.h"

/* Define this variable for error printings */
/*#define DEBUG 1 */
/* Define this variable for debugging printings */
/*#define PRINT 1 */
/* Define this for compile the main() function */
/* #define MAIN 1 */

/* General maximum length of names used */
#define NAMELEN     255
#define MAXELINDIR  256

/* Custom group iteration callback data */
typedef struct {
    char name[NAMELEN];     /* The name of the object */
    int type;               /* The type of the object */
} iter_info;

/* Use %ld to print the value because long should cover most cases. */
/* Used to make certain a return value _is_not_ a value */
#define CHECK(ret, val, where) do {                                           \
    if (ret == val) {                                                         \
        print_func("*** UNEXPECTED RETURN from %s is %ld at line %4d "        \
                   "in %s\n", where, (long)ret, (int)__LINE__, __FILE__);     \
        H5Eprint (stdout);                                                    \
    }                                                                         \
    H5Eclear();                                                               \
} while(0)

PyObject *_getTablesVersion(void);

PyObject *createNamesTuple(char *buffer[], int nelements);

PyObject *Giterate(hid_t loc_id, const char *name);
