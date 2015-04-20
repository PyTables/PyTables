/*********************************************************************
  Small benchmark for testing basic capabilities of Blosc.

  You can select different degrees of 'randomness' in input buffer, as
  well as external datafiles (uncomment the lines after "For data
  coming from a file" comment).

  For usage instructions of this benchmark, please see:

    http://blosc.pytables.org/trac/wiki/SyntheticBenchmarks

  I'm collecting speeds for different machines, so the output of your
  benchmarks and your processor specifications are welcome!

  Author: Francesc Alted <francesc@blosc.org>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#if defined(_WIN32) && !defined(__MINGW32__)
  /* For QueryPerformanceCounter(), etc. */
  #include <windows.h>
#elif defined(__unix__)
  #include <unistd.h>
  #if defined(__linux__)
    #include <time.h>
  #else
    #include <sys/time.h>
  #endif
#else
  #error Unable to detect platform.
#endif


struct bench_wrap_args
{
  char *compressor;
  int nthreads;
  int size;
  int elsize;
  int rshift;
  FILE * output_file;
};

void *bench_wrap(void * args);

#include "../blosc/blosc.h"

#define KB  1024
#define MB  (1024*KB)
#define GB  (1024*MB)

#define NCHUNKS (32*1024)       /* maximum number of chunks */
#define MAX_THREADS 16


int nchunks = NCHUNKS;
int niter = 3;                  /* default number of iterations */
double totalsize = 0.;          /* total compressed/decompressed size */

/* System-specific high-precision timing functions. */
#if defined(_WIN32) && !defined(__MINGW32__)

/* The type of timestamp used on this system. */
#define blosc_timestamp_t LARGE_INTEGER

/* Set a timestamp value to the current time. */
void blosc_set_timestamp(blosc_timestamp_t* timestamp) {
  /* Ignore the return value, assume the call always succeeds. */
  QueryPerformanceCounter(timestamp);
}

/* Given two timestamp values, return the difference in microseconds. */
double blosc_elapsed_usecs(blosc_timestamp_t start_time, blosc_timestamp_t end_time) {
  LARGE_INTEGER CounterFreq;
  QueryPerformanceFrequency(&CounterFreq);

  return (double)(end_time.QuadPart - start_time.QuadPart) / ((double)CounterFreq.QuadPart / 1e6);
}

#else

/* The type of timestamp used on this system. */
#define blosc_timestamp_t struct timespec

/* Set a timestamp value to the current time. */
void blosc_set_timestamp(blosc_timestamp_t* timestamp) {
  clock_gettime(CLOCK_MONOTONIC, timestamp);
}

/* Given two timestamp values, return the difference in microseconds. */
double blosc_elapsed_usecs(blosc_timestamp_t start_time, blosc_timestamp_t end_time) {
	return (1e6 * (end_time.tv_sec - start_time.tv_sec))
		+ (1e-3 * (end_time.tv_nsec - start_time.tv_nsec));
}

#endif

/* Given two timeval stamps, return the difference in seconds */
double getseconds(blosc_timestamp_t last, blosc_timestamp_t current) {
  return 1e-6 * blosc_elapsed_usecs(last, current);
}

/* Given two timeval stamps, return the time per chunk in usec */
double get_usec_chunk(blosc_timestamp_t last, blosc_timestamp_t current, int niter, size_t nchunks) {
  double elapsed_usecs = (double)blosc_elapsed_usecs(last, current);
  return elapsed_usecs / (double)(niter * nchunks);
}

/* Define posix_memalign for Windows */
#if defined(_WIN32) && !defined(__MINGW32__)
#include <malloc.h>

int posix_memalign(void **memptr, size_t alignment, size_t size)
{
	*memptr = _aligned_malloc(size, alignment);
	return 0;
}

#endif

int get_value(int i, int rshift) {
  int v;

  v = (i<<26)^(i<<18)^(i<<11)^(i<<3)^i;
  if (rshift < 32) {
    v &= (1 << rshift) - 1;
  }
  return v;
}


void init_buffer(void *src, int size, int rshift) {
  unsigned int i;
  int *_src = (int *)src;

  /* To have reproducible results */
  srand(1);

  /* Initialize the original buffer */
  for (i = 0; i < size/sizeof(int); ++i) {
    /* Choose one below */
    /* _src[i] = 0;
     * _src[i] = 0x01010101;
     * _src[i] = 0x01020304;
     * _src[i] = i * 1/.3;
     * _src[i] = i;
     * _src[i] = rand() >> (32-rshift); */
    _src[i] = get_value(i, rshift);
  }
}


void do_bench(char *compressor, int nthreads, int size, int elsize,
              int rshift, FILE * ofile) {
  void *src, *srccpy;
  void *dest[NCHUNKS], *dest2;
  int nbytes = 0, cbytes = 0;
  int i, j, retcode;
  unsigned char *orig, *round;
  blosc_timestamp_t last, current;
  double tmemcpy, tshuf, tunshuf;
  int clevel, doshuffle=1;

  blosc_set_nthreads(nthreads);
  if(blosc_set_compressor(compressor) < 0){
    printf("Compiled w/o support for compressor: '%s', so sorry.\n",
           compressor);
    exit(1);
  }

  /* Initialize buffers */
  srccpy = malloc(size);
  retcode = posix_memalign( (void **)(&src), 32, size);
  retcode = posix_memalign( (void **)(&dest2), 32, size);

  /* zero src to initialize byte on it, and not only multiples of 4 */
  memset(src, 0, size);
  init_buffer(src, size, rshift);
  memcpy(srccpy, src, size);
  for (j = 0; j < nchunks; j++) {
     retcode = posix_memalign( (void **)(&dest[j]), 32, size+BLOSC_MAX_OVERHEAD);
  }

  fprintf(ofile, "--> %d, %d, %d, %d, %s\n", nthreads, size, elsize, rshift, compressor);
  fprintf(ofile, "********************** Run info ******************************\n");
  fprintf(ofile, "Blosc version: %s (%s)\n", BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);
  fprintf(ofile, "Using synthetic data with %d significant bits (out of 32)\n", rshift);
  fprintf(ofile, "Dataset size: %d bytes\tType size: %d bytes\n", size, elsize);
  fprintf(ofile, "Working set: %.1f MB\t\t", (size*nchunks) / (float)MB);
  fprintf(ofile, "Number of threads: %d\n", nthreads);
  fprintf(ofile, "********************** Running benchmarks *********************\n");

  blosc_set_timestamp(&last);
  for (i = 0; i < niter; i++) {
    for (j = 0; j < nchunks; j++) {
      memcpy(dest[j], src, size);
    }
  }
  blosc_set_timestamp(&current);
  tmemcpy = get_usec_chunk(last, current, niter, nchunks);
  fprintf(ofile, "memcpy(write):\t\t %6.1f us, %.1f MB/s\n",
         tmemcpy, (size * 1e6) / (tmemcpy*MB));

  blosc_set_timestamp(&last);
  for (i = 0; i < niter; i++) {
    for (j = 0; j < nchunks; j++) {
      memcpy(dest2, dest[j], size);
    }
  }
  blosc_set_timestamp(&current);
  tmemcpy = get_usec_chunk(last, current, niter, nchunks);
  fprintf(ofile, "memcpy(read):\t\t %6.1f us, %.1f MB/s\n",
         tmemcpy, (size * 1e6) / (tmemcpy*MB));

  for (clevel=0; clevel<10; clevel++) {

    fprintf(ofile, "Compression level: %d\n", clevel);

    blosc_set_timestamp(&last);
    for (i = 0; i < niter; i++) {
      for (j = 0; j < nchunks; j++) {
        cbytes = blosc_compress(clevel, doshuffle, elsize, size, src,
                                dest[j], size+BLOSC_MAX_OVERHEAD);
      }
    }
    blosc_set_timestamp(&current);
    tshuf = get_usec_chunk(last, current, niter, nchunks);
    fprintf(ofile, "comp(write):\t %6.1f us, %.1f MB/s\t  ",
           tshuf, (size * 1e6) / (tshuf*MB));
    fprintf(ofile, "Final bytes: %d  ", cbytes);
    if (cbytes > 0) {
      fprintf(ofile, "Ratio: %3.2f", size/(float)cbytes);
    }
    fprintf(ofile, "\n");

    /* Compressor was unable to compress.  Copy the buffer manually. */
    if (cbytes == 0) {
      for (j = 0; j < nchunks; j++) {
        memcpy(dest[j], src, size);
      }
    }

    blosc_set_timestamp(&last);
    for (i = 0; i < niter; i++) {
      for (j = 0; j < nchunks; j++) {
        if (cbytes == 0) {
          memcpy(dest2, dest[j], size);
          nbytes = size;
        }
        else {
          nbytes = blosc_decompress(dest[j], dest2, size);
        }
      }
    }
    blosc_set_timestamp(&current);
    tunshuf = get_usec_chunk(last, current, niter, nchunks);
    fprintf(ofile, "decomp(read):\t %6.1f us, %.1f MB/s\t  ",
           tunshuf, (nbytes * 1e6) / (tunshuf*MB));
    if (nbytes < 0) {
      fprintf(ofile, "FAILED.  Error code: %d\n", nbytes);
    }
    /* fprintf(ofile, "Orig bytes: %d\tFinal bytes: %d\n", cbytes, nbytes); */

    /* Check if data has had a good roundtrip.
       Byte-by-byte comparison is slow, so use 'memcmp' to check whether the
       roundtripped data is correct. If not, fall back to the slow path to
       print diagnostic messages. */
    orig = (unsigned char *)srccpy;
    round = (unsigned char *)dest2;
    if (memcmp(orig, round, size) != 0)
    {
      for(i = 0; i<size; ++i){
        if (orig[i] != round[i]) {
          fprintf(ofile, "\nError: Original data and round-trip do not match in pos %d\n",
                 (int)i);
          fprintf(ofile, "Orig--> %x, round-trip--> %x\n", orig[i], round[i]);
          break;
        }
      }
    }
    else { i = size; }

    if (i == size) fprintf(ofile, "OK\n");

  } /* End clevel loop */


  /* To compute the totalsize, we should take into account the 10
     compression levels */
  totalsize += (size * nchunks * niter * 10.);

  free(src); free(srccpy); free(dest2);
  for (i = 0; i < nchunks; i++) {
    free(dest[i]);
  }

}


/* Compute a sensible value for nchunks */
int get_nchunks(int size_, int ws) {
  int nchunks;

  nchunks = ws / size_;
  if (nchunks > NCHUNKS) nchunks = NCHUNKS;
  if (nchunks < 1) nchunks = 1;
  return nchunks;
}

void *bench_wrap(void * args)
{
    struct bench_wrap_args * arg = (struct bench_wrap_args *) args;
    do_bench(arg->compressor, arg->nthreads, arg->size, arg->elsize,
             arg->rshift, arg->output_file);
    return 0;
}

void print_compress_info(void)
{
  char *name = NULL, *version = NULL;
  int ret;

  printf("Blosc version: %s (%s)\n", BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

  printf("List of supported compressors in this build: %s\n",
         blosc_list_compressors());

  printf("Supported compression libraries:\n");
  ret = blosc_get_complib_info("blosclz", &name, &version);
  if (ret >= 0) printf("  %s: %s\n", name, version);
  ret = blosc_get_complib_info("lz4", &name, &version);
  if (ret >= 0) printf("  %s: %s\n", name, version);
  ret = blosc_get_complib_info("snappy", &name, &version);
  if (ret >= 0) printf("  %s: %s\n", name, version);
  ret = blosc_get_complib_info("zlib", &name, &version);
  if (ret >= 0) printf("  %s: %s\n", name, version);

}


int main(int argc, char *argv[]) {
  char compressor[32];
  char bsuite[32];
  int single = 1;
  int suite = 0;
  int hard_suite = 0;
  int extreme_suite = 0;
  int debug_suite = 0;
  int nthreads = 4;                     /* The number of threads */
  int size = 2*MB;                      /* Buffer size */
  int elsize = 8;                       /* Datatype size */
  int rshift = 19;                      /* Significant bits */
  int workingset = 256*MB;              /* The maximum allocated memory */
  int nthreads_, size_, elsize_, rshift_, i;
  FILE * output_file = stdout;
  blosc_timestamp_t last, current;
  float totaltime;
  char usage[256];

  print_compress_info();

  strncpy(usage, "Usage: bench [blosclz | lz4 | lz4hc | snappy | zlib] "
          "[[single | suite | hardsuite | extremesuite | debugsuite] "
          "[nthreads [bufsize(bytes) [typesize [sbits ]]]]]", 255);

  if (argc < 2) {
    printf("%s\n", usage);
    exit(1);
  }

  strcpy(compressor, argv[1]);

  if (strcmp(compressor, "blosclz") != 0 &&
      strcmp(compressor, "lz4") != 0 &&
      strcmp(compressor, "lz4hc") != 0 &&
      strcmp(compressor, "snappy") != 0 &&
      strcmp(compressor, "zlib") != 0) {
    printf("No such compressor: '%s'\n", compressor);
    exit(2);
  }

  if (argc < 3)
    strcpy(bsuite, "single");
  else
    strcpy(bsuite, argv[2]);

  if (strcmp(bsuite, "single") == 0) {
    single = 1;
  }
  else if (strcmp(bsuite, "suite") == 0) {
    suite = 1;
  }
  else if (strcmp(bsuite, "hardsuite") == 0) {
    hard_suite = 1;
    workingset = 64*MB;
    /* Values here are ending points for loops */
    nthreads = 2;
    size = 8*MB;
    elsize = 32;
    rshift = 32;
  }
  else if (strcmp(bsuite, "extremesuite") == 0) {
    extreme_suite = 1;
    workingset = 32*MB;
    niter = 1;
    /* Values here are ending points for loops */
    nthreads = 4;
    size = 16*MB;
    elsize = 32;
    rshift = 32;
  }
  else if (strcmp(bsuite, "debugsuite") == 0) {
    debug_suite = 1;
    workingset = 32*MB;
    niter = 1;
    /* Warning: values here are starting points for loops.  This is
       useful for debugging. */
    nthreads = 1;
    size = 16*KB;
    elsize = 1;
    rshift = 0;
  }
  else {
    printf("%s\n", usage);
    exit(1);
  }

  printf("Using compressor: %s\n", compressor);
  printf("Running suite: %s\n", bsuite);

  if (argc >= 4) {
    nthreads = atoi(argv[3]);
  }
  if (argc >= 5) {
    size = atoi(argv[4]);
  }
  if (argc >= 6) {
    elsize = atoi(argv[5]);
  }
  if (argc >= 7) {
    rshift = atoi(argv[6]);
  }

  if ((argc >= 8) || !(single || suite || hard_suite || extreme_suite)) {
    printf("%s\n", usage);
    exit(1);
  }

  nchunks = get_nchunks(size, workingset);
  blosc_set_timestamp(&last);

  blosc_init();

  if (suite) {
    for (nthreads_=1; nthreads_ <= nthreads; nthreads_++) {
      do_bench(compressor, nthreads_, size, elsize, rshift, output_file);
    }
  }
  else if (hard_suite) {
    /* Let's start the rshift loop by 4 so that 19 is visited.  This
       is to allow a direct comparison with the plain suite, that runs
       precisely at 19 significant bits. */
    for (rshift_ = 4; rshift_ <= rshift; rshift_ += 5) {
      for (elsize_ = 1; elsize_ <= elsize; elsize_ *= 2) {
        /* The next loop is for getting sizes that are not power of 2 */
        for (i = -elsize_; i <= elsize_; i += elsize_) {
          for (size_ = 32*KB; size_ <= size; size_ *= 2) {
            nchunks = get_nchunks(size_+i, workingset);
    	    niter = 1;
            for (nthreads_ = 1; nthreads_ <= nthreads; nthreads_++) {
              do_bench(compressor, nthreads_, size_+i, elsize_, rshift_, output_file);
              blosc_set_timestamp(&current);
              totaltime = (float)getseconds(last, current);
              printf("Elapsed time:\t %6.1f s.  Processed data: %.1f GB\n",
                     totaltime, totalsize / GB);
            }
          }
        }
      }
    }
  }
  else if (extreme_suite) {
    for (rshift_ = 0; rshift_ <= rshift; rshift_++) {
      for (elsize_ = 1; elsize_ <= elsize; elsize_++) {
        /* The next loop is for getting sizes that are not power of 2 */
        for (i = -elsize_*2; i <= elsize_*2; i += elsize_) {
          for (size_ = 32*KB; size_ <= size; size_ *= 2) {
            nchunks = get_nchunks(size_+i, workingset);
            for (nthreads_ = 1; nthreads_ <= nthreads; nthreads_++) {
              do_bench(compressor, nthreads_, size_+i, elsize_, rshift_, output_file);
              blosc_set_timestamp(&current);
              totaltime = (float)getseconds(last, current);
              printf("Elapsed time:\t %6.1f s.  Processed data: %.1f GB\n",
                     totaltime, totalsize / GB);
            }
          }
        }
      }
    }
  }
  else if (debug_suite) {
    for (rshift_ = rshift; rshift_ <= 32; rshift_++) {
      for (elsize_ = elsize; elsize_ <= 32; elsize_++) {
        /* The next loop is for getting sizes that are not power of 2 */
        for (i = -elsize_*2; i <= elsize_*2; i += elsize_) {
          for (size_ = size; size_ <= 16*MB; size_ *= 2) {
            nchunks = get_nchunks(size_+i, workingset);
            for (nthreads_ = nthreads; nthreads_ <= 6; nthreads_++) {
              do_bench(compressor, nthreads_, size_+i, elsize_, rshift_, output_file);
              blosc_set_timestamp(&current);
              totaltime = (float)getseconds(last, current);
              printf("Elapsed time:\t %6.1f s.  Processed data: %.1f GB\n",
                     totaltime, totalsize / GB);
            }
          }
        }
      }
    }
  }
  /* Single mode */
  else {
    do_bench(compressor, nthreads, size, elsize, rshift, output_file);
  }

  /* Print out some statistics */
  blosc_set_timestamp(&current);
  totaltime = (float)getseconds(last, current);
  printf("\nRound-trip compr/decompr on %.1f GB\n", totalsize / GB);
  printf("Elapsed time:\t %6.1f s, %.1f MB/s\n",
         totaltime, totalsize*2*1.1/(MB*totaltime));

  /* Free blosc resources */
  blosc_free_resources();
  blosc_destroy();
  return 0;
}