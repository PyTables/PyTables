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
 * with a different representation between numpy and HDF5.
 */

#include "typeconv.h"
#include <math.h>
#include <assert.h>


#if (!defined _ISOC99_SOURCE && !defined __USE_ISOC99)
long int lround(double x)
{
  double trunx;

  if (x > 0.0) {
    trunx = floor(x);
    if (x - trunx >= 0.5)
      trunx += 1;
  } else {
    trunx = ceil(x);
    if (trunx - x >= 0.5)
      trunx -= 1;
  }

  return (long int)(trunx);
}
#endif  /* !_ISOC99_SOURCE && !__USE_ISOC99 */


void conv_float64_timeval32(void *base,
			    unsigned long byteoffset,
			    unsigned long bytestride,
			    PY_LONG_LONG nrecords,
			    unsigned long nelements,
			    int sense)
{
  PY_LONG_LONG  record;
  unsigned long  element, gapsize;
  double  *fieldbase;
  union {
    PY_LONG_LONG  i64;
    double  f64;
  } tv;

  assert(bytestride > 0);
  assert(nelements > 0);

  /* Byte distance from end of field to beginning of next field. */
  gapsize = bytestride - nelements * sizeof(double);

  fieldbase = (double *)((unsigned char *)(base) + byteoffset);

  for (record = 0;  record < nrecords;  record++) {
    for (element = 0;  element < nelements;  element++) {
      if (sense == 0) {
	/* Convert from float64 to timeval32. */
	tv.i64 = (((PY_LONG_LONG)(*fieldbase) << 32)
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

    fieldbase = (double *)((unsigned char *)(fieldbase) + gapsize);

    /* XXX: Need to check if this works on platforms which require
       64-bit data to be aligned.  ivb(2005-01-07) */
  }

  assert(fieldbase == (base + byteoffset + bytestride * nrecords));
}


void conv_space_null(void *base,
		     unsigned long byteoffset,
		     unsigned long bytestride,
		     PY_LONG_LONG nrecords,
		     unsigned long nelements,
		     unsigned long itemsize,
		     int sense)
{
  PY_LONG_LONG  record;
  unsigned long  element, gapsize;
  char  *fieldbase, *nchar;

  assert(bytestride > 0);
  assert(nelements > 0);

  /* Byte distance from end of field to beginning of next field. */
  gapsize = bytestride - nelements * itemsize;

  fieldbase = (char *)(base) + byteoffset;

  for (record = 0;  record < nrecords;  record++) {
    for (element = 0;  element < nelements;  element++) {
      if (sense == 0) {
	/* Before writing, convert space padding into nulls */
	for (nchar=fieldbase+itemsize-1; nchar > fieldbase; nchar--) {
	  if (*nchar == 32) *nchar = NULL;
	  else  break;
	}
      }
      else {
	/* Also do so after reading */
	for (nchar=fieldbase+itemsize-1; nchar > fieldbase; nchar--) {
	  if (*nchar == 32) *nchar = NULL;
	  else  break;
	}
      }
      fieldbase += itemsize;
    }

    fieldbase = fieldbase + gapsize;

  }

  assert(fieldbase == (base + byteoffset + bytestride * nrecords));
}
