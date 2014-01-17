/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Unit tests for basic features in Blosc.

  Creation date: 2010-06-07
  Author: Francesc Alted <faltet@gmail.com>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#if defined(_WIN32) && !defined(__MINGW32__)
  #include <time.h>
  #include "win32/stdint-windows.h"
#else
  #include <unistd.h>
  #include <sys/time.h>
#endif
#include <math.h>
#include "../blosc/blosc.h"


/* This is MinUnit in action (http://www.jera.com/techinfo/jtns/jtn002.html) */
#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do \
    { char *message = test(); tests_run++;                          \
      if (message) { printf("%c", 'F'); return message;}            \
      else printf("%c", '.'); } while (0)

extern int tests_run;

#define KB  1024
#define MB  (1024*KB)
#define GB  (1024*MB)
