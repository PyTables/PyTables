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
 * Programmer: Francesc Alted
 *
 * Date: August, 2005
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */


/*   Optimised version for left/int8 */
int bisect_left_b(npy_int8 *a, long x, int hi, int offset) {
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

/*   Optimised version for left/uint8 */
int bisect_left_ub(npy_uint8 *a, long x, int hi, int offset) {
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

/*   Optimised version for right/int8 */
int bisect_right_b(npy_int8 *a, long x, int hi, int offset) {
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

/*   Optimised version for right/uint8 */
int bisect_right_ub(npy_uint8 *a, long x, int hi, int offset) {
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

/*   Optimised version for left/int16 */
int bisect_left_s(npy_int16 *a, long x, int hi, int offset) {
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

/*   Optimised version for left/uint16 */
int bisect_left_us(npy_uint16 *a, long x, int hi, int offset) {
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

/*   Optimised version for right/int16 */
int bisect_right_s(npy_int16 *a, long x, int hi, int offset) {
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

/*   Optimised version for right/uint16 */
int bisect_right_us(npy_uint16 *a, long x, int hi, int offset) {
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

/*   Optimised version for left/int32 */
int bisect_left_i(npy_int32 *a, long x, int hi, int offset) {
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

/*   Optimised version for left/uint32 */
int bisect_left_ui(npy_uint32 *a, npy_uint32 x, int hi, int offset) {
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

/*   Optimised version for right/int32 */
int bisect_right_i(npy_int32 *a, long x, int hi, int offset) {
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

/*   Optimised version for right/uint32 */
int bisect_right_ui(npy_uint32 *a, npy_uint32 x, int hi, int offset) {
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

/*   Optimised version for left/int64 */
int bisect_left_ll(npy_int64 *a, npy_int64 x, int hi, int offset) {
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

/*   Optimised version for left/uint64 */
int bisect_left_ull(npy_uint64 *a, npy_uint64 x, int hi, int offset) {
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

/*   Optimised version for right/int64 */
int bisect_right_ll(npy_int64 *a, npy_int64 x, int hi, int offset) {
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

/*   Optimised version for right/uint64 */
int bisect_right_ull(npy_uint64 *a, npy_uint64 x, int hi, int offset) {
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

/*  Optimised version for left/float32 */
int bisect_left_f(npy_float32 *a, npy_float64 x, int hi, int offset) {
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

/*   Optimised version for right/float32 */
int bisect_right_f(npy_float32 *a, npy_float64 x, int hi, int offset) {
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

/*  Optimised version for left/float64 */
int bisect_left_d(npy_float64 *a, npy_float64 x, int hi, int offset) {
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

/*   Optimised version for left/float64 */
int bisect_right_d(npy_float64 *a, npy_float64 x, int hi, int offset) {
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


/*  Now, it follows a series of functions for doing in-place sorting.
  The array that starts at start1 is sorted in-place. array2 is also
  sorted in-place, but following the array1 order.
 */

#define PYA_QS_STACK 100
#define SMALL_QUICKSORT 15
#define SWAP(a,b) {SWAP_temp = (b); (b) = (a); (a) = SWAP_temp;}
#define iSWAP(a,b) {iSWAP_temp = (b); (b) = (a); (a) = iSWAP_temp;}

int keysort_di(npy_float64 *start1, npy_uint32 *start2, long num)
{
  npy_float64 *pl = start1;
  npy_float64 *pr = start1 + num - 1;
  npy_float64 vp, SWAP_temp;
  npy_float64 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_dll(npy_float64 *start1, npy_int64 *start2, long num)
{
  npy_float64 *pl = start1;
  npy_float64 *pr = start1 + num - 1;
  npy_float64 vp, SWAP_temp;
  npy_float64 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_fi(npy_float32 *start1, npy_uint32 *start2, long num)
{
  npy_float32 *pl = start1;
  npy_float32 *pr = start1 + num - 1;
  npy_float32 vp, SWAP_temp;
  npy_float32 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_fll(npy_float32 *start1, npy_int64 *start2, long num)
{
  npy_float32 *pl = start1;
  npy_float32 *pr = start1 + num - 1;
  npy_float32 vp, SWAP_temp;
  npy_float32 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_lli(npy_int64 *start1, npy_uint32 *start2, long num)
{
  npy_int64 *pl = start1;
  npy_int64 *pr = start1 + num - 1;
  npy_int64 vp, SWAP_temp;
  npy_int64 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_llll(npy_int64 *start1, npy_int64 *start2, long num)
{
  npy_int64 *pl = start1;
  npy_int64 *pr = start1 + num - 1;
  npy_int64 vp, SWAP_temp;
  npy_int64 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_ii(npy_int32 *start1, npy_uint32 *start2, long num)
{
  npy_int32 *pl = start1;
  npy_int32 *pr = start1 + num - 1;
  npy_int32 vp, SWAP_temp;
  npy_int32 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_ill(npy_int32 *start1, npy_int64 *start2, long num)
{
  npy_int32 *pl = start1;
  npy_int32 *pr = start1 + num - 1;
  npy_int32 vp, SWAP_temp;
  npy_int32 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_si(npy_int16 *start1, npy_uint32 *start2, long num)
{
  npy_int16 *pl = start1;
  npy_int16 *pr = start1 + num - 1;
  npy_int16 vp, SWAP_temp;
  npy_int16 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_sll(npy_int16 *start1, npy_int64 *start2, long num)
{
  npy_int16 *pl = start1;
  npy_int16 *pr = start1 + num - 1;
  npy_int16 vp, SWAP_temp;
  npy_int16 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_bi(npy_int8 *start1, npy_uint32 *start2, long num)
{
  npy_int8 *pl = start1;
  npy_int8 *pr = start1 + num - 1;
  npy_int8 vp, SWAP_temp;
  npy_int8 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_bll(npy_int8 *start1, npy_int64 *start2, long num)
{
  npy_int8 *pl = start1;
  npy_int8 *pr = start1 + num - 1;
  npy_int8 vp, SWAP_temp;
  npy_int8 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_ulli(npy_uint64 *start1, npy_uint32 *start2, long num)
{
  npy_uint64 *pl = start1;
  npy_uint64 *pr = start1 + num - 1;
  npy_uint64 vp, SWAP_temp;
  npy_uint64 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_ullll(npy_uint64 *start1, npy_int64 *start2, long num)
{
  npy_uint64 *pl = start1;
  npy_uint64 *pr = start1 + num - 1;
  npy_uint64 vp, SWAP_temp;
  npy_uint64 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_uii(npy_uint32 *start1, npy_uint32 *start2, long num)
{
  npy_uint32 *pl = start1;
  npy_uint32 *pr = start1 + num - 1;
  npy_uint32 vp, SWAP_temp;
  npy_uint32 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_uill(npy_uint32 *start1, npy_int64 *start2, long num)
{
  npy_uint32 *pl = start1;
  npy_uint32 *pr = start1 + num - 1;
  npy_uint32 vp, SWAP_temp;
  npy_uint32 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_usi(npy_uint16 *start1, npy_uint32 *start2, long num)
{
  npy_uint16 *pl = start1;
  npy_uint16 *pr = start1 + num - 1;
  npy_uint16 vp, SWAP_temp;
  npy_uint16 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_usll(npy_uint16 *start1, npy_int64 *start2, long num)
{
  npy_uint16 *pl = start1;
  npy_uint16 *pr = start1 + num - 1;
  npy_uint16 vp, SWAP_temp;
  npy_uint16 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_ubi(npy_uint8 *start1, npy_uint32 *start2, long num)
{
  npy_uint8 *pl = start1;
  npy_uint8 *pr = start1 + num - 1;
  npy_uint8 vp, SWAP_temp;
  npy_uint8 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_uint32 *ipl = start2;
  npy_uint32 *ipr = start2 + num - 1;
  npy_uint32 ivp, iSWAP_temp;
  npy_uint32 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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

int keysort_ubll(npy_uint8 *start1, npy_int64 *start2, long num)
{
  npy_uint8 *pl = start1;
  npy_uint8 *pr = start1 + num - 1;
  npy_uint8 vp, SWAP_temp;
  npy_uint8 *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pt;
  npy_int64 *ipl = start2;
  npy_int64 *ipr = start2 + num - 1;
  npy_int64 ivp, iSWAP_temp;
  npy_int64 *istack[PYA_QS_STACK], **isptr = istack, *ipm, *ipi, *ipj, *ipt;

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


/* Get indices from sorted values */
int get_sorted_indices(int nrows, npy_int64 *rbufC,
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

