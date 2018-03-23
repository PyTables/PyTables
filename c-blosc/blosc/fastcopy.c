/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>
  Creation date: 2018-01-03

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

/*********************************************************************
  The code in this file is heavily based on memcopy.h, from the
  zlib-ng compression library.  See LICENSES/ZLIB.txt for details.
  See also: https://github.com/Dead2/zlib-ng/blob/develop/zlib.h

  New implementations by Francesc Alted:
    * get_run() and get_match() familiy of functions
    * fast_copy() and safe_copy() functions
    * Support for SSE2/AVX2 copy instructions for these routines
**********************************************************************/

#include <assert.h>
#include "blosc-common.h"


static inline unsigned char *copy_1_bytes(unsigned char *out, const unsigned char *from) {
  *out++ = *from;
  return out;
}

static inline unsigned char *copy_2_bytes(unsigned char *out, const unsigned char *from) {
#if defined(BLOSC_STRICT_ALIGN)
  uint16_t chunk;
  memcpy(&chunk, from, 2);
  memcpy(out, &chunk, 2);
#else
  *(uint16_t *) out = *(uint16_t *) from;
#endif
  return out + 2;
}

static inline unsigned char *copy_3_bytes(unsigned char *out, const unsigned char *from) {
  out = copy_1_bytes(out, from);
  return copy_2_bytes(out, from + 1);
}

static inline unsigned char *copy_4_bytes(unsigned char *out, const unsigned char *from) {
#if defined(BLOSC_STRICT_ALIGN)
  uint32_t chunk;
  memcpy(&chunk, from, 4);
  memcpy(out, &chunk, 4);
#else
  *(uint32_t *) out = *(uint32_t *) from;
#endif
  return out + 4;
}

static inline unsigned char *copy_5_bytes(unsigned char *out, const unsigned char *from) {
  out = copy_1_bytes(out, from);
  return copy_4_bytes(out, from + 1);
}

static inline unsigned char *copy_6_bytes(unsigned char *out, const unsigned char *from) {
  out = copy_2_bytes(out, from);
  return copy_4_bytes(out, from + 2);
}

static inline unsigned char *copy_7_bytes(unsigned char *out, const unsigned char *from) {
  out = copy_3_bytes(out, from);
  return copy_4_bytes(out, from + 3);
}

static inline unsigned char *copy_8_bytes(unsigned char *out, const unsigned char *from) {
#if defined(BLOSC_STRICT_ALIGN)
  uint64_t chunk;
  memcpy(&chunk, from, 8);
  memcpy(out, &chunk, 8);
#else
  *(uint64_t *) out = *(uint64_t *) from;
#endif
  return out + 8;
}


static inline unsigned char *copy_16_bytes(unsigned char *out, const unsigned char *from) {
#if defined(__SSE2__)
  __m128i chunk;
  chunk = _mm_loadu_si128((__m128i*)from);
  _mm_storeu_si128((__m128i*)out, chunk);
  from += 16; out += 16;
#elif !defined(BLOSC_STRICT_ALIGN)
  *(uint64_t*)out = *(uint64_t*)from;
   from += 8; out += 8;
   *(uint64_t*)out = *(uint64_t*)from;
   from += 8; out += 8;
#else
   int i;
   for (i = 0; i < 16; i++) {
     *out++ = *from++;
   }
#endif
  return out;
}

static inline unsigned char *copy_32_bytes(unsigned char *out, const unsigned char *from) {
#if defined(__AVX2__)
  __m256i chunk;
  chunk = _mm256_loadu_si256((__m256i*)from);
  _mm256_storeu_si256((__m256i*)out, chunk);
  from += 32; out += 32;
#elif defined(__SSE2__)
  __m128i chunk;
  chunk = _mm_loadu_si128((__m128i*)from);
  _mm_storeu_si128((__m128i*)out, chunk);
  from += 16; out += 16;
  chunk = _mm_loadu_si128((__m128i*)from);
  _mm_storeu_si128((__m128i*)out, chunk);
  from += 16; out += 16;
#elif !defined(BLOSC_STRICT_ALIGN)
  *(uint64_t*)out = *(uint64_t*)from;
   from += 8; out += 8;
   *(uint64_t*)out = *(uint64_t*)from;
   from += 8; out += 8;
   *(uint64_t*)out = *(uint64_t*)from;
   from += 8; out += 8;
   *(uint64_t*)out = *(uint64_t*)from;
   from += 8; out += 8;
#else
   int i;
   for (i = 0; i < 32; i++) {
     *out++ = *from++;
   }
#endif
  return out;
}

#if defined(__AVX2__)
static inline unsigned char *copy_32_bytes_aligned(unsigned char *out, const unsigned char *from) {
  __m256i chunk;
  chunk = _mm256_load_si256((__m256i*)from);
  _mm256_storeu_si256((__m256i*)out, chunk);
  return out + 32;
}
#endif  // __AVX2__

/* Copy LEN bytes (7 or fewer) from FROM into OUT. Return OUT + LEN. */
static inline unsigned char *copy_bytes(unsigned char *out, const unsigned char *from, unsigned len) {
  assert(len < 8);

#ifdef BLOSC_STRICT_ALIGN
  while (len--) {
    *out++ = *from++;
  }
#else
  switch (len) {
    case 7:
      return copy_7_bytes(out, from);
    case 6:
      return copy_6_bytes(out, from);
    case 5:
      return copy_5_bytes(out, from);
    case 4:
      return copy_4_bytes(out, from);
    case 3:
      return copy_3_bytes(out, from);
    case 2:
      return copy_2_bytes(out, from);
    case 1:
      return copy_1_bytes(out, from);
    case 0:
      return out;
    default:
      assert(0);
  }
#endif /* BLOSC_STRICT_ALIGN */
  return out;
}

/* Byte by byte semantics: copy LEN bytes from FROM and write them to OUT. Return OUT + LEN. */
static inline unsigned char *chunk_memcpy(unsigned char *out, const unsigned char *from, unsigned len) {
  unsigned sz = sizeof(uint64_t);
  unsigned rem = len % sz;
  unsigned by8;

  assert(len >= sz);

  /* Copy a few bytes to make sure the loop below has a multiple of SZ bytes to be copied. */
  copy_8_bytes(out, from);

  len /= sz;
  out += rem;
  from += rem;

  by8 = len % 8;
  len -= by8;
  switch (by8) {
    case 7:
      out = copy_8_bytes(out, from);
      from += sz;
    case 6:
      out = copy_8_bytes(out, from);
      from += sz;
    case 5:
      out = copy_8_bytes(out, from);
      from += sz;
    case 4:
      out = copy_8_bytes(out, from);
      from += sz;
    case 3:
      out = copy_8_bytes(out, from);
      from += sz;
    case 2:
      out = copy_8_bytes(out, from);
      from += sz;
    case 1:
      out = copy_8_bytes(out, from);
      from += sz;
    default:
      break;
  }

  while (len) {
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;
    out = copy_8_bytes(out, from);
    from += sz;

    len -= 8;
  }

  return out;
}

/* 16-byte version of chunk_memcpy() */
static inline unsigned char *chunk_memcpy_16(unsigned char *out, const unsigned char *from, unsigned len) {
  unsigned sz = 16;
  unsigned rem = len % sz;
  unsigned ilen;

  assert(len >= sz);

  /* Copy a few bytes to make sure the loop below has a multiple of SZ bytes to be copied. */
  copy_16_bytes(out, from);

  len /= sz;
  out += rem;
  from += rem;

  for (ilen = 0; ilen < len; ilen++) {
    copy_16_bytes(out, from);
    out += sz;
    from += sz;
  }

  return out;
}

/* 32-byte version of chunk_memcpy() */
static inline unsigned char *chunk_memcpy_32(unsigned char *out, const unsigned char *from, unsigned len) {
  unsigned sz = 32;
  unsigned rem = len % sz;
  unsigned ilen;

  assert(len >= sz);

  /* Copy a few bytes to make sure the loop below has a multiple of SZ bytes to be copied. */
  copy_32_bytes(out, from);

  len /= sz;
  out += rem;
  from += rem;

  for (ilen = 0; ilen < len; ilen++) {
    copy_32_bytes(out, from);
    out += sz;
    from += sz;
  }

  return out;
}

/* 32-byte *unrolled* version of chunk_memcpy() */
static inline unsigned char *chunk_memcpy_32_unrolled(unsigned char *out, const unsigned char *from, unsigned len) {
  unsigned sz = 32;
  unsigned rem = len % sz;
  unsigned by8;

  assert(len >= sz);

  /* Copy a few bytes to make sure the loop below has a multiple of SZ bytes to be copied. */
  copy_32_bytes(out, from);

  len /= sz;
  out += rem;
  from += rem;

  by8 = len % 8;
  len -= by8;
  switch (by8) {
    case 7:
      out = copy_32_bytes(out, from);
      from += sz;
    case 6:
      out = copy_32_bytes(out, from);
      from += sz;
    case 5:
      out = copy_32_bytes(out, from);
      from += sz;
    case 4:
      out = copy_32_bytes(out, from);
      from += sz;
    case 3:
      out = copy_32_bytes(out, from);
      from += sz;
    case 2:
      out = copy_32_bytes(out, from);
      from += sz;
    case 1:
      out = copy_32_bytes(out, from);
      from += sz;
    default:
      break;
  }

  while (len) {
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;
    out = copy_32_bytes(out, from);
    from += sz;

    len -= 8;
  }

  return out;
}


/* SSE2/AVX2 *unaligned* version of chunk_memcpy() */
#if defined(__SSE2__) || defined(__AVX2__)
static inline unsigned char *chunk_memcpy_unaligned(unsigned char *out, const unsigned char *from, unsigned len) {
#if defined(__AVX2__)
  unsigned sz = sizeof(__m256i);
#elif defined(__SSE2__)
  unsigned sz = sizeof(__m128i);
#endif
  unsigned rem = len % sz;
  unsigned ilen;

  assert(len >= sz);

  /* Copy a few bytes to make sure the loop below has a multiple of SZ bytes to be copied. */
#if defined(__AVX2__)
  copy_32_bytes(out, from);
#elif defined(__SSE2__)
  copy_16_bytes(out, from);
#endif

  len /= sz;
  out += rem;
  from += rem;

  for (ilen = 0; ilen < len; ilen++) {
#if defined(__AVX2__)
    copy_32_bytes(out, from);
#elif defined(__SSE2__)
    copy_16_bytes(out, from);
#endif
    out += sz;
    from += sz;
  }

  return out;
}
#endif // __SSE2__ || __AVX2__


#if defined(__SSE2__) || defined(__AVX2__)
/* SSE2/AVX2 *aligned* version of chunk_memcpy() */
static inline unsigned char *chunk_memcpy_aligned(unsigned char *out, const unsigned char *from, unsigned len) {
#if defined(__AVX2__)
  unsigned sz = sizeof(__m256i);
  __m256i chunk;
#elif defined(__SSE2__)
  unsigned sz = sizeof(__m128i);
  __m128i chunk;
#endif
  unsigned bytes_to_align = sz - (unsigned)(((uintptr_t)(const void *)(from)) % sz);
  unsigned corrected_len = len - bytes_to_align;
  unsigned rem = corrected_len % sz;
  unsigned ilen;

  assert(len >= sz);

  /* Copy a few bytes to make sure the loop below has aligned access. */
#if defined(__AVX2__)
  chunk = _mm256_loadu_si256((__m256i *) from);
  _mm256_storeu_si256((__m256i *) out, chunk);
#elif defined(__SSE2__)
  chunk = _mm_loadu_si128((__m128i *) from);
  _mm_storeu_si128((__m128i *) out, chunk);
#endif
  out += bytes_to_align;
  from += bytes_to_align;

  len = corrected_len / sz;
  for (ilen = 0; ilen < len; ilen++) {
#if defined(__AVX2__)
    chunk = _mm256_load_si256((__m256i *) from);  /* *aligned* load */
    _mm256_storeu_si256((__m256i *) out, chunk);
#elif defined(__SSE2__)
    chunk = _mm_load_si128((__m128i *) from);  /* *aligned* load */
    _mm_storeu_si128((__m128i *) out, chunk);
#endif
    out += sz;
    from += sz;
  }

  /* Copy remaining bytes */
  if (rem < 8) {
    out = copy_bytes(out, from, rem);
  }
  else {
    out = chunk_memcpy(out, from, rem);
  }

  return out;
}
#endif // __AVX2__ || __SSE2__


/* Byte by byte semantics: copy LEN bytes from FROM and write them to OUT. Return OUT + LEN. */
unsigned char *fastcopy(unsigned char *out, const unsigned char *from, unsigned len) {
  switch (len) {
    case 32:
      return copy_32_bytes(out, from);
    case 16:
      return copy_16_bytes(out, from);
    case 8:
      return copy_8_bytes(out, from);
    default: {
    }
  }
  if (len < 8) {
    return copy_bytes(out, from, len);
  }
#if defined(__SSE2__)
  if (len < 16) {
    return chunk_memcpy(out, from, len);
  }
#if !defined(__AVX2__)
  return chunk_memcpy_unaligned(out, from, len);
#else
  if (len < 32) {
    return chunk_memcpy_16(out, from, len);
  }
  return chunk_memcpy_unaligned(out, from, len);
#endif  // !__AVX2__
#endif  // __SSE2__
  return chunk_memcpy(out, from, len);
}


/* Same as fastcopy() but without overwriting origin or destination when they overlap */
unsigned char* safecopy(unsigned char *out, const unsigned char *from, unsigned len) {
#if defined(__AVX2__)
  unsigned sz = sizeof(__m256i);
#elif defined(__SSE2__)
  unsigned sz = sizeof(__m128i);
#else
  unsigned sz = sizeof(uint64_t);
#endif
  if (out - sz < from) {
    for (; len; --len) {
      *out++ = *from++;
    }
    return out;
  }
  else {
    return fastcopy(out, from, len);
  }
}
