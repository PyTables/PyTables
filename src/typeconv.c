/***********************************************************************
 *
 *      License: BSD
 *      Created: December 21, 2004
 *      Author:  Ivan Vilata i Balaguer - reverse:com.carabos@ivilata
 *
 *      $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/src/typeconv.c,v $
 *      $Id: typeconv.c,v 1.2 2004/12/26 15:53:33 ivilata Exp $
 *
 ***********************************************************************/

/* Type conversion functions for PyTables types which are stored
 * with a different representation between Numarray and HDF5.
 */

#include "typeconv.h"
#include <math.h>
#include <assert.h>


double float64_to_timeval32(double time) {
  long long sec, usec;
  union {
    long long i64;
    double    f64;
  } tv;

  sec = (int)(time);
  usec = lround((time - sec) * 1e+6);

  tv.i64 = (sec << 32) | (usec & 0x00000000ffffffff);

  /* printf("f->tv: %.6f %lld %lld %lld %.6f\n",
   *        time, sec, usec, tv.i64, tv.f64); */
  return tv.f64;
}

double timeval32_to_float64(double tv) {
  long long sec, usec;
  union {
    long long i64;
    double    f64;
  } time;

  time.f64 = tv;

  sec = time.i64 >> 32;
  usec = (long)time.i64;  /* time.i64 & ~0x00000000ffffffff */

  /* printf("tv->f: %.6f %lld %lld %lld %.6f\n",
   *        tv, sec, usec, time.i64, 1e-6 * usec + sec); */
  return 1e-6 * usec + sec;
}


void conv_float64_timeval32(void *base,
			    unsigned long byteoffset,
			    unsigned long bytestride,
			    unsigned long nrecords,
			    unsigned long nelements,
			    int sense)
{
  unsigned long  record, element, gapsize;
  double         *fieldbase;

  assert(bytestride > 0);
  assert(nelements > 0);

  /* Byte distance from end of field to beginning of next field. */
  gapsize = bytestride - nelements * sizeof(double);

  fieldbase = (double *)(base + byteoffset);
  for (record = 0;  record < nrecords;  record++) {
    for (element = 0;  element < nelements;  element++) {
      if (sense == 0) {
	*fieldbase = float64_to_timeval32(*fieldbase);
      } else {
	*fieldbase = timeval32_to_float64(*fieldbase);
      }
      fieldbase++;
    }
    fieldbase += gapsize;
  }

  assert(fieldbase == (base + byteoffset + bytestride * nrecords));
}
