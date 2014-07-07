/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Unit tests for Blosc API.

  Creation date: 2010-06-07
  Author: Francesc Alted <francesc@blosc.io>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include "test_common.h"

int tests_run = 0;

/* Global vars */
void *src, *srccpy, *dest, *dest2;
size_t nbytes, cbytes;
int clevel = 3;
int doshuffle = 1;
size_t typesize = 4;
size_t size = 1*MB;



static char *test_cbuffer_sizes() {
  size_t nbytes_, cbytes_, blocksize;

  blosc_cbuffer_sizes(dest, &nbytes_, &cbytes_, &blocksize);
  mu_assert("ERROR: nbytes incorrect(1)", nbytes == size);
  mu_assert("ERROR: nbytes incorrect(2)", nbytes_ == nbytes);
  mu_assert("ERROR: cbytes incorrect", cbytes == cbytes_);
  mu_assert("ERROR: blocksize incorrect", blocksize >= 128);
  return 0;
}

static char *test_cbuffer_metainfo() {
  size_t typesize_;
  int flags;

  blosc_cbuffer_metainfo(dest, &typesize_, &flags);
  mu_assert("ERROR: typesize incorrect", typesize_ == typesize);
  mu_assert("ERROR: shuffle incorrect", (flags & BLOSC_DOSHUFFLE) == doshuffle);
  return 0;
}


static char *test_cbuffer_versions() {
  int version_;
  int versionlz_;

  blosc_cbuffer_versions(dest, &version_, &versionlz_);
  mu_assert("ERROR: version incorrect", version_ == BLOSC_VERSION_FORMAT);
  mu_assert("ERROR: versionlz incorrect", versionlz_ == BLOSC_BLOSCLZ_VERSION_FORMAT);
  return 0;
}


static char *test_cbuffer_complib() {
  char *complib;

  complib = blosc_cbuffer_complib(dest);
  mu_assert("ERROR: complib incorrect", strcmp(complib, "BloscLZ") == 0);
  return 0;
}


static char *all_tests() {
  mu_run_test(test_cbuffer_sizes);
  mu_run_test(test_cbuffer_metainfo);
  mu_run_test(test_cbuffer_versions);
  mu_run_test(test_cbuffer_complib);
  return 0;
}


int main(int argc, char **argv) {
  char *result;

  printf("STARTING TESTS for %s", argv[0]);

  blosc_init();
  blosc_set_nthreads(1);

  /* Initialize buffers */
  src = malloc(size);
  srccpy = malloc(size);
  dest = malloc(size);
  dest2 = malloc(size);
  memset(src, 0, size);
  memcpy(srccpy, src, size);

  /* Get a compressed buffer */
  cbytes = blosc_compress(clevel, doshuffle, typesize, size, src, dest, size);

  /* Get a decompressed buffer */
  nbytes = blosc_decompress(dest, dest2, size);

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

