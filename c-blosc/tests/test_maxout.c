/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Unit tests for basic features in Blosc.

  Creation date: 2010-06-07
  Author: Francesc Alted <francesc@blosc.org>

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


static char *all_tests() {
  mu_run_test(test_maxout_less);
  mu_run_test(test_maxout_equal);
  mu_run_test(test_maxout_great);

  return 0;
}

#define BUFFER_ALIGN_SIZE   32

int main(int argc, char **argv) {
  int32_t *_src;
  char *result;
  size_t i;

  printf("STARTING TESTS for %s", argv[0]);

  blosc_init();
  blosc_set_nthreads(1);

  /* Initialize buffers */
  src = blosc_test_malloc(BUFFER_ALIGN_SIZE, size);
  srccpy = blosc_test_malloc(BUFFER_ALIGN_SIZE, size);
  dest = blosc_test_malloc(BUFFER_ALIGN_SIZE, size + 16);
  dest2 = blosc_test_malloc(BUFFER_ALIGN_SIZE, size);
  _src = (int32_t *)src;
  for (i=0; i < (size/4); i++) {
    _src[i] = (int32_t)i;
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

  blosc_test_free(src);
  blosc_test_free(srccpy);
  blosc_test_free(dest);
  blosc_test_free(dest2);

  blosc_destroy();

  return result != 0;
}
