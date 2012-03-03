/***********************************************************************
 *
 *      License: BSD
 *      Created: December 21, 2004
 *      Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
 *      Modified:
 *        Function inlining and some castings for 64-bit adressing
 *        Francesc Alted 2004-12-27
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
      /* Perform an explicit copy of data to avoid errors related to
         unaligned memory access on platforms like AMR, etc.
         Patch submitted by Julian Taylor */
      double fb;
      memcpy(&fb, fieldbase, sizeof(*fieldbase));
      if (sense == 0) {
        /* Convert from float64 to timeval32. */
        tv.i64 = (((PY_LONG_LONG)(fb) << 32)
                  | (lround((fb - (int)(fb)) * 1e+6) & 0x0ffffffff));
        fb = tv.f64;
      } else {
        /* Convert from timeval32 to float64. */
        tv.f64 = fb;
        /* the next computation is 64 bit-platforms aware */
        fb = 1e-6 * (int)tv.i64 + (tv.i64 >> 32);
      }
      memcpy(fieldbase, &fb, sizeof(*fieldbase));
      fieldbase++;
    }

    fieldbase = (double *)((unsigned char *)(fieldbase) + gapsize);
  }

  assert(fieldbase == (base + byteoffset + bytestride * nrecords));
}

