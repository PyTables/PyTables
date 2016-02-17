# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: May 18, 2006
# Author:  Francesc Alted - faltet@pytables.com
#          The PyTables Team
#
########################################################################


#
# This module contains utility functions (keysort, _bisect_left, _bisect_right)
#

import cython
cimport numpy as np

# Types, constants, functions, classes & other objects from everywhere
from numpy cimport import_array, ndarray, \
    npy_int8, npy_int16, npy_int32, npy_int64, \
    npy_uint8, npy_uint16, npy_uint32, npy_uint64, \
    npy_float32, npy_float64, \
    npy_float, npy_double, npy_longdouble

# These two types are defined in npy_common.h but not in cython's numpy.pxd
ctypedef unsigned char npy_bool
ctypedef npy_uint16 npy_float16

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, strncmp


#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#---------------------------------------------------------------------------

ctypedef fused floating_type:
    npy_float32
    npy_float64
    npy_longdouble


ctypedef fused number_type:
    npy_int8
    npy_int16
    npy_int32
    npy_int64

    npy_uint8
    npy_uint16
    npy_uint32
    npy_uint64

    npy_float32
    npy_float64
    npy_longdouble

#===========================================================================
# Functions
#===========================================================================

#---------------------------------------------------------------------------
# keysort
#---------------------------------------------------------------------------

DEF PYA_QS_STACK = 100
DEF SMALL_QUICKSORT = 15

def keysort(ndarray array1, ndarray array2):
    """Sort array1 in-place. array2 is also sorted following the array1 order.

    array1 can be of any type, except complex or string.  array2 may be made of
    elements on any size.

    """
    cdef size_t size = np.PyArray_SIZE(array1)
    cdef size_t elsize1 = np.PyArray_ITEMSIZE(array1)
    cdef size_t elsize2 = np.PyArray_ITEMSIZE(array2)
    cdef int type_num = np.PyArray_TYPE(array1)

    # floating types
    if type_num == np.NPY_FLOAT16:
        _keysort[npy_float16](<npy_float16*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_FLOAT32:
        _keysort[npy_float32](<npy_float32*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_FLOAT64:
        _keysort[npy_float64](<npy_float64*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_LONGDOUBLE:
        _keysort[npy_longdouble](<npy_longdouble*>array1.data, array2.data, elsize2, size)
    # signed integer types
    elif type_num == np.NPY_INT8:
        _keysort[npy_int8](<npy_int8*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_INT16:
        _keysort[npy_int16](<npy_int16*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_INT32:
        _keysort[npy_int32](<npy_int32*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_INT64:
        _keysort[npy_int64](<npy_int64*>array1.data, array2.data, elsize2, size)
    # unsigned integer types
    elif type_num == np.NPY_UINT8:
        _keysort[npy_uint8](<npy_uint8*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_UINT16:
        _keysort[npy_uint16](<npy_uint16*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_UINT32:
        _keysort[npy_uint32](<npy_uint32*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_UINT64:
        _keysort[npy_uint64](<npy_uint64*>array1.data, array2.data, elsize2, size)
    # other
    elif type_num == np.NPY_BOOL:
        _keysort[npy_bool](<npy_bool*>array1.data, array2.data, elsize2, size)
    elif type_num == np.NPY_STRING:
        _keysort_string(array1.data, elsize1, array2.data, elsize2, size)
    else:
        raise ValueError("Unknown array datatype")


cdef inline void swap_bytes(char *x, char *y, size_t n) nogil:
    if n == 8:
        (<npy_int64*>x)[0], (<npy_int64*>y)[0] = (<npy_int64*>y)[0], (<npy_int64*>x)[0]
    elif n == 4:
        (<npy_int32*>x)[0], (<npy_int32*>y)[0] = (<npy_int32*>y)[0], (<npy_int32*>x)[0]
    elif n == 2:
        (<npy_int16*>x)[0], (<npy_int16*>y)[0] = (<npy_int16*>y)[0], (<npy_int16*>x)[0]
    else:
        for i in range(n):
            x[i], y[i] = y[i], x[i]


cdef inline int less_than(number_type* a, number_type* b) nogil:
    if number_type in floating_type:
        return a[0] < b[0] or (b[0] != b[0] and a[0] == a[0])
    else:
        return a[0] < b[0]


@cython.cdivision(True)
cdef void _keysort(number_type* start1, char* start2, size_t elsize2, size_t n) nogil:
    cdef number_type *pl = start1
    cdef number_type *pr = start1 + (n - 1)

    cdef char *ipl = start2
    cdef char *ipr = start2 + (n - 1) * elsize2

    cdef number_type vp
    cdef char *ivp = <char *> malloc(elsize2)

    cdef number_type *stack[PYA_QS_STACK]
    cdef number_type **sptr = stack

    cdef char *istack[PYA_QS_STACK]
    cdef char **isptr = istack

    cdef size_t stack_index = 0

    cdef number_type *pm
    cdef number_type *pi
    cdef number_type *pj
    cdef number_type *pt
    cdef char *ipm
    cdef char *ipi
    cdef char *ipj
    cdef char *ipt

    while True:
        while pr - pl > SMALL_QUICKSORT:
            pm  = pl + ((pr - pl) >> 1)
            ipm  = ipl + ((ipr - ipl)/elsize2 >> 1)*elsize2

            if less_than(pm, pl):
                pm[0], pl[0] =  pl[0], pm[0]
                swap_bytes(ipm, ipl, elsize2)

            if less_than(pr, pm):
                pr[0], pm[0] =  pm[0], pr[0]
                swap_bytes(ipr, ipm, elsize2)

            if less_than(pm, pl):
                pm[0], pl[0] =  pl[0], pm[0]
                swap_bytes(ipm, ipl, elsize2)

            vp = pm[0]

            pi = pl
            ipi = ipl

            pj = pr - 1
            ipj = ipr - elsize2

            pm[0], pj[0] = pj[0], pm[0]
            swap_bytes(ipm, ipj, elsize2)

            while True:
                pi += 1
                ipi += elsize2
                while less_than(pi, &vp):
                    pi += 1
                    ipi += elsize2

                pj -= 1
                ipj -= elsize2
                while less_than(&vp, pj):
                    pj -= 1
                    ipj -= elsize2

                if pi >= pj:
                    break

                pi[0], pj[0] = pj[0], pi[0]
                swap_bytes(ipi, ipj, elsize2)

            pi[0], (pr-1)[0] = (pr-1)[0], pi[0]
            swap_bytes(ipi, ipr-elsize2, elsize2)

            # push largest partition on stack and proceed with the other
            if (pi - pl) < (pr - pi):
                sptr[0] = pi + 1
                sptr[1] = pr
                sptr += 2

                isptr[0] = ipi + elsize2
                isptr[1] = ipr
                isptr += 2

                pr = pi - 1
                ipr = ipi - elsize2
            else:
                sptr[0] = pl
                sptr[1] = pi - 1
                sptr += 2

                isptr[0] = ipl
                isptr[1] = ipi - elsize2
                isptr += 2

                pl = pi + 1
                ipl = ipi + elsize2

        pi = pl + 1
        ipi = ipl + elsize2
        while pi <= pr:
            vp = pi[0]
            memcpy(ivp, ipi, elsize2)

            pj = pi
            pt = pi - 1

            ipj = ipi
            ipt = ipi - elsize2

            while pj > pl and less_than(&vp, pt):
                pj[0] = pt[0]
                pj -= 1
                pt -= 1

                memcpy(ipj, ipt, elsize2)
                ipj -= elsize2
                ipt -= elsize2

            pj[0] = vp
            memcpy(ipj, ivp, elsize2)

            pi += 1
            ipi += elsize2

        if sptr == stack:
            break

        sptr -= 2
        pl = sptr[0]
        pr = sptr[1]

        isptr -= 2
        ipl = isptr[0]
        ipr = isptr[1]

    free(ivp)


@cython.cdivision(True)
cdef void _keysort_string(char* start1, size_t ss, char* start2, size_t ts, size_t n) nogil:
    cdef char *pl = start1
    cdef char *pr = start1 + (n - 1) * ss

    cdef char *ipl = start2
    cdef char *ipr = start2 + (n - 1) * ts

    cdef char *vp = <char *>malloc(ss)
    cdef char *ivp = <char *>malloc(ts)

    cdef char *stack[PYA_QS_STACK]
    cdef char **sptr = stack

    cdef char *istack[PYA_QS_STACK]
    cdef char **isptr = istack

    cdef size_t stack_index = 0

    cdef char *pm
    cdef char *pi
    cdef char *pj
    cdef char *pt

    cdef char *ipm
    cdef char *ipi
    cdef char *ipj
    cdef char *ipt

    while True:
        while pr - pl > SMALL_QUICKSORT * ss:
            pm  = pl + ((pr - pl)/ss >> 1)*ss
            ipm  = ipl + ((ipr - ipl)/ts >> 1)*ts

            if strncmp(pm, pl, ss) < 0:
                swap_bytes(pm, pl, ss)
                swap_bytes(ipm, ipl, ts)

            if strncmp(pr, pm, ss) < 0:
                swap_bytes(pr, pm, ss)
                swap_bytes(ipr, ipm, ts)

            if strncmp(pm, pl, ss) < 0:
                swap_bytes(pm, pl, ss)
                swap_bytes(ipm, ipl, ts)

            memcpy(vp, pm, ss)

            pi = pl
            ipi = ipl

            pj = pr - ss
            ipj = ipr - ts

            swap_bytes(pm, pj, ss)
            swap_bytes(ipm, ipj, ts)

            while True:
                pi += ss
                ipi += ts
                while strncmp(pi, vp, ss) < 0:
                    pi += ss
                    ipi += ts

                pj -= ss
                ipj -= ts
                while strncmp(vp, pj, ss) < 0:
                    pj -= ss
                    ipj -= ts

                if pi >= pj:
                    break

                swap_bytes(pi, pj, ss)
                swap_bytes(ipi, ipj, ts)

            swap_bytes(pi, pr-ss, ss)
            swap_bytes(ipi, ipr-ts, ts)

            # push largest partition on stack and proceed with the other
            if (pi - pl) < (pr - pi):
                sptr[0] = pi + ss
                sptr[1] = pr
                sptr += 2

                isptr[0] = ipi + ts
                isptr[1] = ipr
                isptr += 2

                pr = pi - ss
                ipr = ipi - ts
            else:
                sptr[0] = pl
                sptr[1] = pi - ss
                sptr += 2

                isptr[0] = ipl
                isptr[1] = ipi - ts
                isptr += 2

                pl = pi + ss
                ipl = ipi + ts

        pi = pl + ss
        ipi = ipl + ts

        while pi <= pr:
            memcpy(vp, pi, ss)
            memcpy(ivp, ipi, ts)

            pj = pi
            pt = pi - ss

            ipj = ipi
            ipt = ipi - ts

            while pj > pl and strncmp(vp, pt, ss) < 0:
                memcpy(pj, pt, ss)
                pj -= ss
                pt -= ss

                memcpy(ipj, ipt, ts)
                ipj -= ts
                ipt -= ts

            memcpy(pj, vp, ss)
            memcpy(ipj, ivp, ts)

            pi += ss
            ipi += ts

        if sptr == stack:
            break

        sptr -= 2
        pl = sptr[0]
        pr = sptr[1]

        isptr -= 2
        ipl = isptr[0]
        ipr = isptr[1]

    free(vp)
    free(ivp)

#---------------------------------------------------------------------------
# bisect
#---------------------------------------------------------------------------

# This has been copied from the standard module bisect.
# Checks for the values out of limits has been added at the beginning
# because I forsee that this should be a very common case.
# 2004-05-20
def _bisect_left(a, x, int hi):
  """Return the index where to insert item x in list a, assuming a is sorted.

  The return value i is such that all e in a[:i] have e < x, and all e in
  a[i:] have e >= x.  So if x already appears in the list, i points just
  before the leftmost x already there.

  """

  cdef int lo, mid

  lo = 0
  if x <= a[0]: return 0
  if a[-1] < x: return hi
  while lo < hi:
      mid = (lo+hi)/2
      if a[mid] < x: lo = mid+1
      else: hi = mid
  return lo


def _bisect_right(a, x, int hi):
  """Return the index where to insert item x in list a, assuming a is sorted.

  The return value i is such that all e in a[:i] have e <= x, and all e in
  a[i:] have e > x.  So if x already appears in the list, i points just
  beyond the rightmost x already there.

  """

  cdef int lo, mid

  lo = 0
  if x < a[0]: return 0
  if a[-1] <= x: return hi
  while lo < hi:
    mid = (lo+hi)/2
    if x < a[mid]: hi = mid
    else: lo = mid+1
  return lo


