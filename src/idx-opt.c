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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
    mid = lo + (hi-lo)/2;
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
#define SMALL_STRING 16

#define SWAP(a,b) {SWAP_temp = (b); (b) = (a); (a) = SWAP_temp;}

/* This is for swaping strings */
static void
sSWAP(char *s1, char *s2, size_t len)
{
  while(len--) {
    const char t = *s1;
    *s1++ = *s2;
    *s2++ = t;
  }
}

/* The iSWAP macro is safe because it will always work on aligned ints */
#define iSWAP(a,b) {						\
    switch(ts) {						\
    case 8:							\
      *(npy_int64 *)iSWAP_temp = *(npy_int64 *)(b);		\
      *(npy_int64 *)(b) = *(npy_int64 *)(a);			\
      *(npy_int64 *)(a) = *(npy_int64 *)iSWAP_temp;		\
      break;							\
    case 4:							\
      *(npy_int32 *)iSWAP_temp = *(npy_int32 *)(b);		\
      *(npy_int32 *)(b) = *(npy_int32 *)(a);			\
      *(npy_int32 *)(a) = *(npy_int32 *)iSWAP_temp;		\
      break;							\
    case 1:							\
      *(npy_int8 *)iSWAP_temp = *(npy_int8 *)(b);		\
      *(npy_int8 *)(b) = *(npy_int8 *)(a);			\
      *(npy_int8 *)(a) = *(npy_int8 *)iSWAP_temp;		\
      break;							\
    case 2:							\
      *(npy_int16 *)iSWAP_temp = *(npy_int16 *)(b);		\
      *(npy_int16 *)(b) = *(npy_int16 *)(a);			\
      *(npy_int16 *)(a) = *(npy_int16 *)iSWAP_temp;		\
      break;							\
    default:							\
      for (i=0; i<ts; i++) {					\
	((char *)iSWAP_temp)[i] = ((char *)(b))[i];		\
	((char *)(b))[i] = ((char *)(a))[i];			\
	((char *)(a))[i] = ((char *)(iSWAP_temp))[i];		\
      }								\
    }								\
  }


/* #define opt_memcpy(a,b,n) memcpy((a),(b),(n)) */

/* For small values of n, a loop seems faster for memcpy.
   See http://www.mail-archive.com/numpy-discussion@scipy.org/msg06639.html
   and other messages in the same thread

   F. Alted 2008-02-11
 */
static void
opt_memcpy(char *s1, char *s2, size_t len)
{
  size_t i;
  if (len < SMALL_STRING) {
    for (i=0; i < len; i++) {
      s1[i] = s2[i];
    }
  }
  else {
    memcpy(s1, s2, len);
  }
}

static int
opt_strncmp(char *a, char *b, size_t n)
{
    size_t i;
    unsigned char c, d;
    for (i = 0; i < n; i++) {
        c = a[i]; d = b[i];
        if (c != d) return c - d;
    }
    return 0;
}


int keysort_f64(npy_float64 *start1, char *start2, npy_intp num, int ts)
{
  npy_float64 *pl = start1;
  char *ipl = start2;
  npy_float64 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_float64 vp;
  char *ivp;
  npy_float64 SWAP_temp;
  char *iSWAP_temp;
  npy_float64 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_float64 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_f32(npy_float32 *start1, char *start2, npy_intp num, int ts)
{
  npy_float32 *pl = start1;
  char *ipl = start2;
  npy_float32 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_float32 vp;
  char *ivp;
  npy_float32 SWAP_temp;
  char *iSWAP_temp;
  npy_float32 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_float32 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_i64(npy_int64 *start1, char *start2, npy_intp num, int ts)
{
  npy_int64 *pl = start1;
  char *ipl = start2;
  npy_int64 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_int64 vp;
  char *ivp;
  npy_int64 SWAP_temp;
  char *iSWAP_temp;
  npy_int64 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_int64 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_u64(npy_uint64 *start1, char *start2, npy_intp num, int ts)
{
  npy_uint64 *pl = start1;
  char *ipl = start2;
  npy_uint64 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_uint64 vp;
  char *ivp;
  npy_uint64 SWAP_temp;
  char *iSWAP_temp;
  npy_uint64 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_uint64 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_i32(npy_int32 *start1, char *start2, npy_intp num, int ts)
{
  npy_int32 *pl = start1;
  char *ipl = start2;
  npy_int32 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_int32 vp;
  char *ivp;
  npy_int32 SWAP_temp;
  char *iSWAP_temp;
  npy_int32 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_int32 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_u32(npy_uint32 *start1, char *start2, npy_intp num, int ts)
{
  npy_uint32 *pl = start1;
  char *ipl = start2;
  npy_uint32 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_uint32 vp;
  char *ivp;
  npy_uint32 SWAP_temp;
  char *iSWAP_temp;
  npy_uint32 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_uint32 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_i16(npy_int16 *start1, char *start2, npy_intp num, int ts)
{
  npy_int16 *pl = start1;
  char *ipl = start2;
  npy_int16 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_int16 vp;
  char *ivp;
  npy_int16 SWAP_temp;
  char *iSWAP_temp;
  npy_int16 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_int16 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_u16(npy_uint16 *start1, char *start2, npy_intp num, int ts)
{
  npy_uint16 *pl = start1;
  char *ipl = start2;
  npy_uint16 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_uint16 vp;
  char *ivp;
  npy_uint16 SWAP_temp;
  char *iSWAP_temp;
  npy_uint16 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_uint16 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_i8(npy_int8 *start1, char *start2, npy_intp num, int ts)
{
  npy_int8 *pl = start1;
  char *ipl = start2;
  npy_int8 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_int8 vp;
  char *ivp;
  npy_int8 SWAP_temp;
  char *iSWAP_temp;
  npy_int8 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_int8 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_u8(npy_uint8 *start1, char *start2, npy_intp num, int ts)
{
  npy_uint8 *pl = start1;
  char *ipl = start2;
  npy_uint8 *pr = start1 + num - 1;
  char *ipr = start2 + (num - 1) * ts;
  npy_uint8 vp;
  char *ivp;
  npy_uint8 SWAP_temp;
  char *iSWAP_temp;
  npy_uint8 *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  npy_uint8 *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT) {
      /* quicksort partition */
      pm = pl + ((pr - pl) >> 1); ipm = ipl + (((ipr - ipl)/ts) >> 1)*ts;
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      if (*pr < *pm) { SWAP(*pr, *pm); iSWAP(ipr, ipm); }
      if (*pm < *pl) { SWAP(*pm, *pl); iSWAP(ipm, ipl); }
      vp = *pm;
      pi = pl; ipi = ipl;
      pj = pr - 1; ipj = ipr - ts;
      SWAP(*pm, *pj); iSWAP(ipm, ipj);
      for(;;) {
	do { ++pi; ipi += ts; } while (*pi < vp);
	do { --pj; ipj -= ts; } while (vp < *pj);
	if (pi >= pj)  break;
	SWAP(*pi, *pj); iSWAP(ipi, ipj);
      }
      SWAP(*pi, *(pr-1)); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + 1; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - 1; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - 1; *isptr++ = ipi - ts;
	pl = pi + 1; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + 1, ipi = ipl + ts; pi <= pr; ++pi, ipi += ts) {
      vp = *pi; opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - 1, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && vp < *pt;) {
	*pj-- = *pt--; opt_memcpy(ipj, ipt, ts); ipj -= ts; ipt -= ts;
      }
      *pj = vp; opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(ivp);
  free(iSWAP_temp);

  return 0;
}


int keysort_S(char *start1, int ss, char *start2, npy_intp num, int ts)
{
  char *pl = start1;
  char *ipl = start2;
  char *pr = start1 + (num - 1) * ss;
  char *ipr = start2 + (num - 1) * ts;
  char *vp;
  char *ivp;
  char *iSWAP_temp;
  char *stack[PYA_QS_STACK], **sptr = stack;
  char *istack[PYA_QS_STACK], **isptr = istack;
  char *pm, *pi, *pj, *pt;
  char *ipm, *ipi, *ipj, *ipt;
  npy_intp i;

  vp = malloc(ss); ivp = malloc(ts);
  iSWAP_temp = malloc(ts);
  for(;;) {
    while ((pr - pl) > SMALL_QUICKSORT*ss) {
      /* quicksort partition */
      pm = pl + (((pr-pl)/ss) >> 1)*ss; ipm = ipl + (((ipr-ipl)/ts) >> 1)*ts;
      if (opt_strncmp(pm, pl, ss) < 0) { sSWAP(pm, pl, ss); iSWAP(ipm, ipl); }
      if (opt_strncmp(pr, pm, ss) < 0) { sSWAP(pr, pm, ss); iSWAP(ipr, ipm); }
      if (opt_strncmp(pm, pl, ss) < 0) { sSWAP(pm, pl, ss); iSWAP(ipm, ipl); }
      opt_memcpy(vp, pm, ss);
      pi = pl; ipi = ipl;
      pj = pr - ss; ipj = ipr - ts;
      sSWAP(pm, pj, ss); iSWAP(ipm, ipj);
      for(;;) {
	do { pi += ss; ipi += ts; } while (opt_strncmp(pi, vp, ss) < 0);
	do { pj -= ss; ipj -= ts; } while (opt_strncmp(vp, pj, ss) < 0);
	if (pi >= pj)  break;
	sSWAP(pi, pj, ss); iSWAP(ipi, ipj);
      }
      sSWAP(pi, pr-ss, ss); iSWAP(ipi, ipr-ts);
      /* push largest partition on stack */
      if (pi - pl < pr - pi) {
	*sptr++ = pi + ss; *isptr++ = ipi + ts;
	*sptr++ = pr; *isptr++ = ipr;
	pr = pi - ss; ipr = ipi - ts;
      }else{
	*sptr++ = pl; *isptr++ = ipl;
	*sptr++ = pi - ss; *isptr++ = ipi - ts;
	pl = pi + ss; ipl = ipi + ts;
      }
    }
    /* insertion sort */
    for(pi = pl + ss, ipi = ipl + ts; pi <= pr;	pi += ss, ipi += ts) {
      opt_memcpy(vp, pi, ss); opt_memcpy(ivp, ipi, ts);
      for(pj = pi, pt = pi - ss, ipj = ipi, ipt = ipi - ts; \
	  pj > pl && opt_strncmp(vp, pt, ss) < 0;) {
	opt_memcpy(pj, pt, ss); opt_memcpy(ipj, ipt, ts);
	pj -= ss; pt -= ss; ipj -= ts; ipt -= ts;
      }
      opt_memcpy(pj, vp, ss); opt_memcpy(ipj, ivp, ts);
    }
    if (sptr == stack) break;
    pr = *(--sptr); ipr = *(--isptr);
    pl = *(--sptr); ipl = *(--isptr);
  }

  free(vp); free(ivp);
  free(iSWAP_temp);

  return 0;
}


