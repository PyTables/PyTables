/* Include file for calcoffset.c 
 * F. Altet 
 * 2002/08/28 */

/* For the sake of code simplicity I've stripped out all the alignment
   stuff, as it seems not necessary for HDF5. If it is needed, you can
   find the complete code version in CVS with tag 1.14.
   F. Altet 2004-09-16 */

/* Define this variable for error printings */
/*#define DEBUG 1 */
/* Define this variable for debugging printings */
/*#define PRINT 1 */
/* Define this for compile the main() function */
/* #define MAIN 1 */

/* Functions in calcoffset.c we want accessible */
int calcoffset(char *fmt, int *nattrs, hid_t *types,
	       size_t *size_types, size_t *offsets);

