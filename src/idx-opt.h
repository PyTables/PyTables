#include "Python.h"
#include "numpy/arrayobject.h"

#ifndef NPY_FLOAT16
typedef npy_uint16 npy_float16;
#endif

#ifndef NPY_FLOAT96
typedef long double npy_float96;
#endif

#ifndef NPY_FLOAT128
typedef long double npy_float128;
#endif

int bisect_left_b(npy_int8 *a, long x, int hi, int offset);
int bisect_left_ub(npy_uint8 *a, long x, int hi, int offset);
int bisect_right_b(npy_int8 *a, long x, int hi, int offset);
int bisect_right_ub(npy_uint8 *a, long x, int hi, int offset);

int bisect_left_s(npy_int16 *a, long x, int hi, int offset);
int bisect_left_us(npy_uint16 *a, long x, int hi, int offset);
int bisect_right_s(npy_int16 *a, long x, int hi, int offset);
int bisect_right_us(npy_uint16 *a, long x, int hi, int offset);

int bisect_left_i(npy_int32 *a, long x, int hi, int offset);
int bisect_left_ui(npy_uint32 *a, npy_uint32 x, int hi, int offset);
int bisect_right_i(npy_int32 *a, long x, int hi, int offset);
int bisect_right_ui(npy_uint32 *a, npy_uint32 x, int hi, int offset);

int bisect_left_ll(npy_int64 *a, npy_int64 x, int hi, int offset);
int bisect_left_ull(npy_uint64 *a, npy_uint64 x, int hi, int offset);
int bisect_right_ll(npy_int64 *a, npy_int64 x, int hi, int offset);
int bisect_right_ull(npy_uint64 *a, npy_uint64 x, int hi, int offset);

int bisect_left_e(npy_float16 *a, npy_float64 x, int hi, int offset);
int bisect_right_e(npy_float16 *a, npy_float64 x, int hi, int offset);

int bisect_left_f(npy_float32 *a, npy_float64 x, int hi, int offset);
int bisect_right_f(npy_float32 *a, npy_float64 x, int hi, int offset);

int bisect_left_d(npy_float64 *a, npy_float64 x, int hi, int offset);
int bisect_right_d(npy_float64 *a, npy_float64 x, int hi, int offset);

int bisect_left_g(npy_longdouble *a, npy_longdouble x, int hi, int offset);
int bisect_right_g(npy_longdouble *a, npy_longdouble x, int hi, int offset);
