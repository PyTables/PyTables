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

hid_t convArrayType(int fmt, size_t size);

int getArrayType(H5T_class_t class_id,
		 size_t type_size,
		 H5T_sign_t sign,
		 int *fmt);

/* Açò hauria de desaparéixer, no? */
/* herr_t H5LTget_dataset_info_mod( hid_t loc_id,
				 const char *dset_name,
				 hsize_t *dims,
				 H5T_class_t *class_id,
				 H5T_sign_t *sign,
				 size_t *type_size );

*/
