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

/*  Optimised version for left/float16 */
int bisect_left_e(npy_float16 *a, npy_float64 x, int hi, int offset) {
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

/*   Optimised version for right/float16 */
int bisect_right_e(npy_float16 *a, npy_float64 x, int hi, int offset) {
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

/*   Optimised version for right/float64 */
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

/*  Optimised version for left/longdouble */
int bisect_left_g(npy_longdouble *a, npy_longdouble x, int hi, int offset) {
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

/*   Optimised version for right/longdouble */
int bisect_right_g(npy_longdouble *a, npy_longdouble x, int hi, int offset) {
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

