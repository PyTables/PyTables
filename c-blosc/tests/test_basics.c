/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Unit tests for basic features in Blosc.

  Creation date: 2010-06-07
  Author: Francesc Alted <faltet@gmail.com>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include "test_common.h"

int tests_run = 0;

/* Global vars */
void *src, *srccpy, *dest, *dest2;
size_t nbytes, cbytes;
int clevel = 1;
int doshuffle = 0;
size_t typesize = 4;
size_t size = 1000;             /* must be divisible by 4 */


/* Check maxout with maxout < size */
static char *test_maxout_less() {

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size+15);
  mu_assert("ERROR: cbytes is not 0", cbytes == 0);

  return 0;
}

/* Check maxout with maxout == size */
static char *test_maxout_equal() {

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size+16);
  mu_assert("ERROR: cbytes is not correct", cbytes == size+16);

  /* Decompress the buffer */
  nbytes = blosc_decompress(dest, dest2, size);
  mu_assert("ERROR: nbytes incorrect(1)", nbytes == size);

  return 0;
}


/* Check maxout with maxout > size */
static char *test_maxout_great() {
  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size+17);
  mu_assert("ERROR: cbytes is not 0", cbytes == size+16);

  /* Decompress the buffer */
  nbytes = blosc_decompress(dest, dest2, size);
  mu_assert("ERROR: nbytes incorrect(1)", nbytes == size);

  return 0;
}

static char * test_shuffle()
{
  int sizes[] = {7, 64 * 3, 7*256, 500, 8000, 100000, 702713};
  int types[] = {1, 2, 3, 4, 5, 6, 7, 8, 16};
  int i, j, k;
  int ok;
  for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    for (j = 0; j < sizeof(types) / sizeof(types[0]); j++) {
      int n = sizes[i];
      int t = types[j];
      char * d = malloc(t * n);
      char * d2 = malloc(t * n);
      char * o = malloc(t * n + BLOSC_MAX_OVERHEAD);
      for (k = 0; k < n; k++) {
        d[k] = rand();
      }
      blosc_compress(5, 1, t, t * n, d, o, t * n + BLOSC_MAX_OVERHEAD);
      blosc_decompress(o, d2, t * n);
      ok = 1;
      for (k = 0; ok&& k < n; k++) {
        ok = (d[k] == d2[k]);
      }
      free(d);
      free(d2);
      free(o);
      mu_assert("ERROR: multi size test failed", ok);
    }
  }

  return 0;
}

static char *all_tests() {
  mu_run_test(test_maxout_less);
  mu_run_test(test_maxout_equal);
  mu_run_test(test_maxout_great);
  mu_run_test(test_shuffle);
  return 0;
}

int main(int argc, char **argv) {
  size_t i;
  int32_t *_src;
  char *result;

  printf("STARTING TESTS for %s", argv[0]);

  blosc_init();
  blosc_set_nthreads(1);

  /* Initialize buffers */
  src = malloc(size);
  srccpy = malloc(size);
  dest = malloc(size+16);
  dest2 = malloc(size);
  _src = (int32_t *)src;
  for (i=0; i < (size/4); i++) {
    _src[i] = i;
  }
  memcpy(srccpy, src, size);

  /* Run all the suite */
  result = all_tests();
  if (result != 0) {
    printf(" (%s)\n", result);
  }
  else {
    printf(" ALL TESTS PASSED");
  }
  printf("\tTests run: %d\n", tests_run);

  free(src); free(srccpy); free(dest); free(dest2);
  blosc_destroy();

  return result != 0;
}
