/***********************************************************************
 *
 *      License: BSD
 *      Created: December 21, 2004
 *      Author:  Ivan Vilata i Balaguer - reverse:com.carabos@ivilata
 *      Modified: 
 *        Function inlining and some castings for 64-bit adressing
 *        Francesc Altet 2004-12-27
 *
 *      $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/typeconv.c,v $
 *      $Id: typeconv.c,v 1.3 2004/12/27 22:18:37 falted Exp $
 *
 ***********************************************************************/

/* Type conversion functions for PyTables types which are stored
 * with a different representation between Numarray and HDF5.
 */

#include "typeconv.h"
#include <assert.h>


void conv_float64_timeval32(void *base,
			    unsigned long byteoffset,
			    unsigned long bytestride,
			    long long nrecords,
			    unsigned long nelements,
			    int sense)
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
  gapsize = bytestride - nelements * sizeof(double);

  fieldbase = (double *)(base + byteoffset);
  for (record = 0;  record < nrecords;  record++) {
    if (sense == 0) {     /* The if here allows the compiler doing
			     a better optimization job */
      for (element = 0;  element < nelements;  element++) {
	tv.i64 = ((long long)*fieldbase << 32) + 
	         (int)((*fieldbase - (int)*fieldbase) * 1e+6);
	*fieldbase = tv.f64;
	fieldbase++;
      }
    }
    else {
      for (element = 0;  element < nelements;  element++) {
	tv.f64 = *fieldbase;
	/* the next computation is 64 bit-platforms aware */
	*fieldbase = 1e-6 * (int)tv.i64 + (tv.i64 >> 32);
	fieldbase++;
      }
    }
    fieldbase += gapsize;
  }

  assert(fieldbase == (base + byteoffset + bytestride * nrecords));
}
