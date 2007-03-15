
int bisect_left_i(int *a, int x, int hi, int offset);
int bisect_right_i(int *a, int x, int hi, int offset);

int bisect_left_ll(long long *a, long long x, int hi, int offset);
int bisect_right_ll(long long *a, long long x, int hi, int offset);

int bisect_left_f(float *a, float x, int hi, int offset);
int bisect_right_f(float *a, float x, int hi, int offset);

int bisect_left_d(double *a, double x, int hi, int offset);
int bisect_right_d(double *a, double x, int hi, int offset);

int get_sorted_indices(int nrows, long long *rbufC,
		       int *rbufst, int *rbufln, int ssize);
int convert_addr64(int nrows, int nelem,
		   long long *rbufA, int *rbufR, int *rbufln);

int keysort_di(double *start1, unsigned int *start2, long num);
int keysort_dll(double *start1, long long *start2, long num);
int keysort_fi(float *start1, unsigned int *start2, long num);
int keysort_fll(float *start1, long long *start2, long num);
int keysort_lli(long long *start1, unsigned int *start2, long num);
int keysort_llll(long long *start1, long long *start2, long num);
int keysort_ii(int *start1, unsigned int *start2, long num);
int keysort_ill(int *start1, long long *start2, long num);
int keysort_si(short int *start1, unsigned int *start2, long num);
int keysort_sll(short int *start1, long long *start2, long num);
int keysort_bi(char *start1, unsigned int *start2, long num);
int keysort_bll(char *start1, long long *start2, long num);
int keysort_ulli(unsigned long long *start1, unsigned int *start2, long num);
int keysort_ullll(unsigned long long *start1, long long *start2, long num);
int keysort_uii(unsigned int *start1, unsigned int *start2, long num);
int keysort_uill(unsigned int *start1, long long *start2, long num);
int keysort_usi(unsigned short int *start1, unsigned int *start2, long num);
int keysort_usll(unsigned short int *start1, long long *start2, long num);
int keysort_ubi(unsigned char *start1, unsigned int *start2, long num);
int keysort_ubll(unsigned char *start1, long long *start2, long num);
