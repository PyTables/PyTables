/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef SHUFFLE_COMMON_H
#define SHUFFLE_COMMON_H

#include "blosc-export.h"

/* Define the __SSE2__ symbol if compiling with Visual C++ and
   targeting the minimum architecture level supporting SSE2.
   Other compilers define this as expected and emit warnings
   when it is re-defined. */
#if !defined(__SSE2__) && defined(_MSC_VER) && \
    (defined(_M_X64) || (defined(_M_IX86) && _M_IX86_FP >= 2))
  #define __SSE2__
#endif

/* Import standard integer type definitions */
#if defined(_WIN32) && !defined(__MINGW32__)
  #include <windows.h>
  #include "win32/stdint-windows.h"
#else
  #include <stdint.h>
  #include <stddef.h>
  #include <inttypes.h>
  #include <string.h>
#endif  /* _WIN32 */

#endif  /* SHUFFLE_COMMON_H */
