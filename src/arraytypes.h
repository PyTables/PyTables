#include "hdf5.h"
#include "Python.h"  /* Necessary to import numarray.h */
#include "numarray/numarray.h"

/* Define this variable for error printings */
  /*#define DEBUG 1 */
  /* Define this variable for debugging printings */
  /*#define PRINT 1 */
  /* Define this for compile the main() function */
  /* #define MAIN 1 */
  
/* Functions in arraytypes.c we want to made accessible */

hid_t convArrayType(int fmt, size_t size, char *byteorder);

int getArrayType(H5T_class_t class_id,
		 size_t type_size,
		 size_t type_precision,
		 H5T_sign_t sign,
		 int *fmt);

