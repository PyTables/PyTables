/***********************************************************************
 *
 *      License: BSD
 *      Created: December 21, 2004
 *      Author:  Ivan Vilata i Balaguer - reverse:com.carabos@ivilata
 *      Modified: 
 *        Function inlining and some castings for 64-bit adressing
 *        Francesc Altet 2004-12-27
 *
 *      $Source: /cvsroot/pytables/pytables/src/typeconv.c,v $
 *      $Id$
 *
 ***********************************************************************/

/* Type conversion functions for PyTables types which are stored
 * with a different representation between Numarray and HDF5.
 */

#include "typeconv.h"
#include <math.h>
#include <assert.h>


void conv_float64_timeval32(void *base,
			    unsigned long byteoffset,
			    unsigned long bytestride,
			    long long nrecords,
			    unsigned long nelements,
			    const int sense)
{
  long long      record;
  unsigned long  element, gapsize;
  double         *fieldbase;
  union {
    long long i64;
    double    f64;
  } tv;

  assert(bytestride > 0);
  assert(nelements > 0);

  /* Byte distance from end of field to beginning of next field. */
  gapsize = bytestride/sizeof(double) - nelements;

  fieldbase = (double *)(base + byteoffset);
  for (record = 0;  record < nrecords;  record++) {
    for (element = 0;  element < nelements;  element++) {
      if (sense == 0) {
	/* Convert from float64 to timeval32. */
	tv.i64 = (((long long)(*fieldbase) << 32)
		  | (lround((*fieldbase - (int)(*fieldbase)) * 1e+6)
		     & 0x0ffffffff));
	*fieldbase = tv.f64;
      } else {
	/* Convert from timeval32 to float64. */
	tv.f64 = *fieldbase;
	/* the next computation is 64 bit-platforms aware */
	*fieldbase = 1e-6 * (int)tv.i64 + (tv.i64 >> 32);
      }
      fieldbase++;
    }
    fieldbase += gapsize;
  }

  assert(fieldbase == (base + byteoffset + bytestride * nrecords));
}
