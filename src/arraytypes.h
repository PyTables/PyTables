#include "hdf5.h"
#include "Python.h"  /* Necessary to import numpy.h */
#include "numpy/arrayobject.h"

/* Define this variable for error printings */
  /*#define DEBUG 1 */
  /* Define this variable for debugging printings */
  /*#define PRINT 1 */
  /* Define this for compile the main() function */
  /* #define MAIN 1 */
  
/* Functions in arraytypes.c we want to made accessible */

hid_t convArrayType(int nptype, size_t size, char *byteorder);

size_t getArrayType(hid_t type_id, int *nptype);

