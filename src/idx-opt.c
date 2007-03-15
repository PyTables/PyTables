#include "stdio.h"
#include "idx-opt.h"

/*-------------------------------------------------------------------------
 *
 * Binary search functions
 *
 *-------------------------------------------------------------------------
 */

/*-------------------------------------------------------------------------
 * Function: bisect_{left,right}_optim_*
 *
 * Purpose: Look-up for a value in sorted arrays
 *
 * Return: The index of the value in array
 *
 * Programmer: Francesc Altet
 *
 * Date: August, 2005
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */


/*   Optimised version for int32 */
int bisect_left_i(int *a, int x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x <= a[offset]) return 0;
  if (a[hi-1+offset] < x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (a[mid+offset] < x) lo = mid+1;
    else hi = mid;
  }
  return lo;
}

/*   Optimised version for int32 */
int bisect_right_i(int *a, int x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x < a[offset]) return 0;
  if (a[hi-1+offset] <= x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (x < a[mid+offset]) hi = mid;
    else lo = mid+1;
  }
  return lo;
}

/*   Optimised version for int64 */
int bisect_left_ll(long long *a, long long x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x <= a[offset]) return 0;
  if (a[hi-1+offset] < x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (a[mid+offset] < x) lo = mid+1;
    else hi = mid;
  }
  return lo;
}

/*   Optimised version for int64 */
int bisect_right_ll(long long *a, long long x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x < a[offset]) return 0;
  if (a[hi-1+offset] <= x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (x < a[mid+offset]) hi = mid;
    else lo = mid+1;
  }
  return lo;
}

/*  Optimised version for float32 */
int bisect_left_f(float *a, float x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x <= a[offset]) return 0;
  if (a[hi-1+offset] < x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (a[mid+offset] < x) lo = mid+1;
    else hi = mid;
  }
  return lo;
}

/*   Optimised version for float32 */
int bisect_right_f(float *a, float x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x < a[offset]) return 0;
  if (a[hi-1+offset] <= x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (x < a[mid+offset]) hi = mid;
    else lo = mid+1;
  }
  return lo;
}

/*  Optimised version for float64 */
int bisect_left_d(double *a, double x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x <= a[offset]) return 0;
  if (a[hi-1+offset] < x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (a[mid+offset] < x) lo = mid+1;
    else hi = mid;
  }
  return lo;
}

/*   Optimised version for float64 */
int bisect_right_d(double *a, double x, int hi, int offset) {
  int lo = 0;
  int mid;

  if (x < a[offset]) return 0;
  if (a[hi-1+offset] <= x) return hi;
  while (lo < hi) {
    mid = (lo+hi)/2;
    if (x < a[mid+offset]) hi = mid;
    else lo = mid+1;
  }
  return lo;
}

/* Get indices from sorted values */
int get_sorted_indices(int nrows, long long *rbufC,
		       int *rbufst, int *rbufln, int ssize) {
  int irow, jrow;
  int len1 = 0;

  for (irow = 0; irow < nrows; irow++) {
    for (jrow = 0; jrow < rbufln[irow]; jrow++) {
      rbufC[len1++] = irow * ssize + rbufst[irow] + jrow;
    }
  }
  return 0;
}

/* Convert indices to 64 bit with offsets */
int convert_addr64(int nrows, int nelem,
		   long long *rbufA, int *rbufR, int *rbufln) {
  int irow, jrow;
  int len1 = 0;

  for (irow = 0; irow < nrows; irow++)
    for (jrow = 0; jrow < rbufln[irow]; jrow++) {
      rbufA[len1] = rbufR[len1] + irow * nelem;
      len1++;
    }
  return 0;
}

/*  Now, it follows a series of functions for doing in-place sorting.
  The array that starts at start1 is sorted in-place. array2 is also
  sorted in-place, but following the array1 order.
 */

#define PYA_QS_STACK 100
#define SMALL_QUICKSORT 15
#define SWAP(a,b) {SWAP_temp = (b); (b) = (a); (a) = SWAP_temp;}
#define iSWAP(a,b) {iSWAP_temp = (b); (b) = (a); (a) = iSWAP_temp;}

int keysort_di(double *start1, unsigned int *start2, long num)
{
  double *pl = start1;
  double *pr = start1 + num - 1;
  double vp, SWAP_temp;
  double *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_dll(double *start1, long long *start2, long num)
{
  double *pl = start1;
  double *pr = start1 + num - 1;
  double vp, SWAP_temp;
  double *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_fi(float *start1, unsigned int *start2, long num)
{
  float *pl = start1;
  float *pr = start1 + num - 1;
  float vp, SWAP_temp;
  float *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_fll(float *start1, long long *start2, long num)
{
  float *pl = start1;
  float *pr = start1 + num - 1;
  float vp, SWAP_temp;
  float *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_lli(long long *start1, unsigned int *start2, long num)
{
  long long *pl = start1;
  long long *pr = start1 + num - 1;
  long long vp, SWAP_temp;
  long long *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_llll(long long *start1, long long *start2, long num)
{
  long long *pl = start1;
  long long *pr = start1 + num - 1;
  long long vp, SWAP_temp;
  long long *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_ii(int *start1, unsigned int *start2, long num)
{
  int *pl = start1;
  int *pr = start1 + num - 1;
  int vp, SWAP_temp;
  int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_ill(int *start1, long long *start2, long num)
{
  int *pl = start1;
  int *pr = start1 + num - 1;
  int vp, SWAP_temp;
  int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_si(short int *start1, unsigned int *start2, long num)
{
  short int *pl = start1;
  short int *pr = start1 + num - 1;
  short int vp, SWAP_temp;
  short int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_sll(short int *start1, long long *start2, long num)
{
  short int *pl = start1;
  short int *pr = start1 + num - 1;
  short int vp, SWAP_temp;
  short int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_bi(char *start1, unsigned int *start2, long num)
{
  char *pl = start1;
  char *pr = start1 + num - 1;
  char vp, SWAP_temp;
  char *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_bll(char *start1, long long *start2, long num)
{
  char *pl = start1;
  char *pr = start1 + num - 1;
  char vp, SWAP_temp;
  char *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

/* Unsigned versions of start1 */

int keysort_ulli(unsigned long long *start1, unsigned int *start2, long num)
{
  unsigned long long *pl = start1;
  unsigned long long *pr = start1 + num - 1;
  unsigned long long vp, SWAP_temp;
  unsigned long long *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_ullll(unsigned long long *start1, long long *start2, long num)
{
  unsigned long long *pl = start1;
  unsigned long long *pr = start1 + num - 1;
  unsigned long long vp, SWAP_temp;
  unsigned long long *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_uii(unsigned int *start1, unsigned int *start2, long num)
{
  unsigned int *pl = start1;
  unsigned int *pr = start1 + num - 1;
  unsigned int vp, SWAP_temp;
  unsigned int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_uill(unsigned int *start1, long long *start2, long num)
{
  unsigned int *pl = start1;
  unsigned int *pr = start1 + num - 1;
  unsigned int vp, SWAP_temp;
  unsigned int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_usi(unsigned short int *start1, unsigned int *start2, long num)
{
  unsigned short int *pl = start1;
  unsigned short int *pr = start1 + num - 1;
  unsigned short int vp, SWAP_temp;
  unsigned short int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_usll(unsigned short int *start1, long long *start2, long num)
{
  unsigned short int *pl = start1;
  unsigned short int *pr = start1 + num - 1;
  unsigned short int vp, SWAP_temp;
  unsigned short int *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_ubi(unsigned char *start1, unsigned int *start2, long num)
{
  unsigned char *pl = start1;
  unsigned char *pr = start1 + num - 1;
  unsigned char vp, SWAP_temp;
  unsigned char *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  unsigned int *ipl = start2;
  unsigned int *ipr = start2 + num - 1;
  unsigned int ivp, iSWAP_temp;
  unsigned int *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

int keysort_ubll(unsigned char *start1, long long *start2, long num)
{
  unsigned char *pl = start1;
  unsigned char *pr = start1 + num - 1;
  unsigned char vp, SWAP_temp;
  unsigned char *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  long long *ipl = start2;
  long long *ipr = start2 + num - 1;
  long long ivp, iSWAP_temp;
  long long *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + ((ipr - ipl) >> 1);
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      if (*pr < *pm) { SWAP(*pr,*pm); iSWAP(*ipr,*ipm); }
      if (*pm < *pl) { SWAP(*pm,*pl); iSWAP(*ipm,*ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - 1;
      SWAP(*pm,*pj); iSWAP(*ipm,*ipj);
      for(;;) {
	do { ++pi; ++ipi; } while (*pi < vp);
	do { --pj; --ipj; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi,*pj); iSWAP(*ipi,*ipj);
      }
      SWAP(*pi,*(pr-1)); iSWAP(*ipi,*(ipr-1));
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + 1;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - 1;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - 1;
	pl = pi + 1; ipl = ipi + 1;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + 1; pi <= pr; ++pi, ++ipi) {
      vp = *pi; ivp = *ipi;
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - 1; \
	    pj > pl && vp < *pt;) {
	*pj-- = *pt--; *ipj-- = *ipt--;
      }
      *pj = vp; *ipj = ivp;
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }
  return 0;
}

