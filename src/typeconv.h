/***********************************************************************
 *
 *      License: BSD
 *      Created: December 21, 2004
 *      Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
 *
 *      $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/typeconv.h,v $
 *      $Id$
 *
 ***********************************************************************/

/* Type conversion functions for PyTables types which are stored
 * with a different representation between numpy and HDF5.
 */

#ifndef __TYPECONV_H__
#define __TYPECONV_H__ 1

#include "Python.h"

/* Meaning for common arguments:
 *   * base: pointer to data
 *   * byteoffset: offset of first field/element into the data
 *   * bytestride: distance in bytes from a field/record to the next one
 *   * nrecords: number of fields/records to translate
 *   * nelements: number of elements in a field/record
 *   * sense: 0 for Numarray -> HDF5, otherwise HDF5 -> Numarray
 */

void conv_float64_timeval32(void *base,
                            unsigned long byteoffset,
                            unsigned long bytestride,
                            PY_LONG_LONG nrecords,
                            unsigned long nelements,
                            int sense);

#endif /* def __TYPECONV_H__ */
