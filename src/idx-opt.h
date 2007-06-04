#include "Python.h"
#include "numpy/arrayobject.h"

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

int bisect_left_f(npy_float32 *a, npy_float64 x, int hi, int offset);
int bisect_right_f(npy_float32 *a, npy_float64 x, int hi, int offset);

int bisect_left_d(npy_float64 *a, npy_float64 x, int hi, int offset);
int bisect_right_d(npy_float64 *a, npy_float64 x, int hi, int offset);


int keysort_di(npy_float64 *start1, npy_uint32 *start2, long num);
int keysort_dll(npy_float64 *start1, npy_int64 *start2, long num);
int keysort_fi(npy_float32 *start1, npy_uint32 *start2, long num);
int keysort_fll(npy_float32 *start1, npy_int64 *start2, long num);
int keysort_lli(npy_int64 *start1, npy_uint32 *start2, long num);
int keysort_llll(npy_int64 *start1, npy_int64 *start2, long num);
int keysort_ii(npy_int32 *start1, npy_uint32 *start2, long num);
int keysort_ill(npy_int32 *start1, npy_int64 *start2, long num);
int keysort_si(npy_int16 *start1, npy_uint32 *start2, long num);
int keysort_sll(npy_int16 *start1, npy_int64 *start2, long num);
int keysort_bi(npy_int8 *start1, npy_uint32 *start2, long num);
int keysort_bll(npy_int8 *start1, npy_int64 *start2, long num);
int keysort_ulli(npy_uint64 *start1, npy_uint32 *start2, long num);
int keysort_ullll(npy_uint64 *start1, npy_int64 *start2, long num);
int keysort_uii(npy_uint32 *start1, npy_uint32 *start2, long num);
int keysort_uill(npy_uint32 *start1, npy_int64 *start2, long num);
int keysort_usi(npy_uint16 *start1, npy_uint32 *start2, long num);
int keysort_usll(npy_uint16 *start1, npy_int64 *start2, long num);
int keysort_ubi(npy_uint8 *start1, npy_uint32 *start2, long num);
int keysort_ubll(npy_uint8 *start1, npy_int64 *start2, long num);

int get_sorted_indices(int nrows, npy_int64 *rbufC,
		       int *rbufst, int *rbufln, int ssize);
