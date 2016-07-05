/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Unit tests for BLOSC_COMPRESSOR environment variable in Blosc.

  Creation date: 2016-04-25
  Author: Francesc Alted <francesc@blosc.org>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include "test_common.h"

int tests_run = 0;

/* Global vars */
void *src, *srccpy, *dest, *dest2;
int nbytes, cbytes;
int clevel = 1;
int doshuffle = 1;
size_t typesize = 8;
size_t size = 8 * 1000 * 1000;  /* must be divisible by typesize */


/* Check compressor */
static char *test_compressor() {
  char* compressor;

  /* Before any blosc_compress() the compressor must be blosclz */
  compressor = blosc_get_compressor();
  mu_assert("ERROR: get_compressor (compress, before) incorrect",
	    strcmp(compressor, "blosclz") == 0);

  /* Activate the BLOSC_COMPRESSOR variable */
  setenv("BLOSC_COMPRESSOR", "lz4", 0);

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not correct", cbytes < size);

  compressor = blosc_get_compressor();
  mu_assert("ERROR: get_compressor (compress, after) incorrect",
	    strcmp(compressor, "lz4") == 0);

  /* Reset envvar */
  unsetenv("BLOSC_COMPRESSOR");
  return 0;
}


/* Check compressing + decompressing */
static char *test_compress_decompress() {
  char* compressor;

  /* Activate the BLOSC_COMPRESSOR variable */
  setenv("BLOSC_COMPRESSOR", "lz4", 0);

  compressor = blosc_get_compressor();
  mu_assert("ERROR: get_compressor incorrect",
	    strcmp(compressor, "lz4") == 0);

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not correct", cbytes < size);

  compressor = blosc_get_compressor();
  mu_assert("ERROR: get_compressor incorrect",
	    strcmp(compressor, "lz4") == 0);

  /* Decompress the buffer */
  nbytes = blosc_decompress(dest, dest2, size);
  mu_assert("ERROR: nbytes incorrect(1)", nbytes == size);

  compressor = blosc_get_compressor();
  mu_assert("ERROR: get_compressor incorrect",
	    strcmp(compressor, "lz4") == 0);

  /* Reset envvar */
  unsetenv("BLOSC_COMPRESSOR");
  return 0;
}


/* Check compression level */
static char *test_clevel() {
  int cbytes2;

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not correct", cbytes < size);

  /* Activate the BLOSC_CLEVEL variable */
  setenv("BLOSC_CLEVEL", "9", 0);
  cbytes2 = blosc_compress(clevel, doshuffle, typesize, size, src,
                           dest, size + 16);
  mu_assert("ERROR: BLOSC_CLEVEL does not work correctly", cbytes2 < cbytes);

  /* Reset envvar */
  unsetenv("BLOSC_CLEVEL");
  return 0;
}

/* Check noshuffle */
static char *test_noshuffle() {
  int cbytes2;

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not correct", cbytes < size);

  /* Activate the BLOSC_SHUFFLE variable */
  setenv("BLOSC_SHUFFLE", "NOSHUFFLE", 0);
  cbytes2 = blosc_compress(clevel, doshuffle, typesize, size, src,
                           dest, size + 16);
  mu_assert("ERROR: BLOSC_SHUFFLE=NOSHUFFLE does not work correctly",
            cbytes2 > cbytes);

  /* Reset env var */
  unsetenv("BLOSC_SHUFFLE");
  return 0;
}


/* Check regular shuffle */
static char *test_shuffle() {
  int cbytes2;

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not 0", cbytes < size);

  /* Activate the BLOSC_SHUFFLE variable */
  setenv("BLOSC_SHUFFLE", "SHUFFLE", 0);
  cbytes2 = blosc_compress(clevel, doshuffle, typesize, size, src,
                           dest, size + 16);
  mu_assert("ERROR: BLOSC_SHUFFLE=SHUFFLE does not work correctly",
            cbytes2 == cbytes);

  /* Reset env var */
  unsetenv("BLOSC_SHUFFLE");
  return 0;
}

/* Check bitshuffle */
static char *test_bitshuffle() {
  int cbytes2;

  /* Get a compressed buffer */
  blosc_set_compressor("blosclz");  /* avoid lz4 here for now (see #168) */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not 0", cbytes < size);

  /* Activate the BLOSC_BITSHUFFLE variable */
  setenv("BLOSC_SHUFFLE", "BITSHUFFLE", 0);
  cbytes2 = blosc_compress(clevel, doshuffle, typesize, size, src,
                           dest, size + 16);
  mu_assert("ERROR: BLOSC_SHUFFLE=BITSHUFFLE does not work correctly",
            cbytes2 < cbytes);

  /* Reset env var */
  unsetenv("BLOSC_SHUFFLE");
  return 0;
}


/* Check typesize */
static char *test_typesize() {
  int cbytes2;

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src,
                          dest, size + 16);
  mu_assert("ERROR: cbytes is not correct", cbytes < size);

  /* Activate the BLOSC_TYPESIZE variable */
  setenv("BLOSC_TYPESIZE", "9", 0);
  cbytes2 = blosc_compress(clevel, doshuffle, typesize, size, src,
                           dest, size + 16);
  mu_assert("ERROR: BLOSC_TYPESIZE does not work correctly", cbytes2 > cbytes);

  /* Reset envvar */
  unsetenv("BLOSC_TYPESIZE");
  return 0;
}


static char *all_tests() {
  mu_run_test(test_compressor);
  mu_run_test(test_compress_decompress);
  mu_run_test(test_clevel);
  mu_run_test(test_noshuffle);
  mu_run_test(test_shuffle);
  mu_run_test(test_bitshuffle);
  mu_run_test(test_typesize);

  return 0;
}

#define BUFFER_ALIGN_SIZE   32

int main(int argc, char **argv) {
  int64_t *_src;
  char *result;
  size_t i;

  printf("STARTING TESTS for %s", argv[0]);

  blosc_init();
  blosc_set_compressor("blosclz");

  /* Initialize buffers */
  src = blosc_test_malloc(BUFFER_ALIGN_SIZE, size);
  srccpy = blosc_test_malloc(BUFFER_ALIGN_SIZE, size);
  dest = blosc_test_malloc(BUFFER_ALIGN_SIZE, size + 16);
  dest2 = blosc_test_malloc(BUFFER_ALIGN_SIZE, size);
  _src = (int64_t *)src;
  for (i=0; i < (size / sizeof(int64_t)); i++) {
    _src[i] = (int64_t)i;
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
