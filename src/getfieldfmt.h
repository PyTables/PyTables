/* Include file for calcoffset.c 
 * F. Altet 
 * 2002/08/28 */

#include "H5TB.h"
#include <stdlib.h>
#include <string.h>
#include <Python.h>


/* Functions in Tgetfieldfmt.c we want to make accessible */
herr_t getfieldfmt ( hid_t loc_id, const char *table_name, char *field_names[],
		     size_t *field_sizes, size_t *field_offset,
		     size_t *rowsize, hsize_t *nrecords, hsize_t *nfields,
		     PyObject *shapes, PyObject *sizes,
		     PyObject *types, char *fmt );
  
