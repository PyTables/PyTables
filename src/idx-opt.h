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


int keysort_f64(npy_float64 *start1, char *start2, npy_intp num, int ts);
int keysort_f32(npy_float32 *start1, char *start2, npy_intp num, int ts);
int keysort_i64(npy_int64 *start1, char *start2, npy_intp num, int ts);
int keysort_u64(npy_uint64 *start1, char *start2, npy_intp num, int ts);
int keysort_i32(npy_int32 *start1, char *start2, npy_intp num, int ts);
int keysort_u32(npy_uint32 *start1, char *start2, npy_intp num, int ts);
int keysort_i16(npy_int16 *start1, char *start2, npy_intp num, int ts);
int keysort_u16(npy_uint16 *start1, char *start2, npy_intp num, int ts);
int keysort_i8(npy_int8 *start1, char *start2, npy_intp num, int ts);
int keysort_u8(npy_uint8 *start1, char *start2, npy_intp num, int ts);
int keysort_S(char *start1, int ss, char *start2, npy_intp num, int ts);

