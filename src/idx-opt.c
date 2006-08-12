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


/*  Optimised version for Float64 */
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

/*   Optimised version for Int32 */
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

/*   Optimised version for Int64 */
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

/*   Optimised version for Float64 */
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

/*   Optimised version for Int32 */
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

/*   Optimised version for Int64 */
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
    for (jrow = 0; jrow < rbufln[irow]; jrow++)
      rbufA[len1++] = rbufR[len1] + irow * nelem;
  return 0;
}
