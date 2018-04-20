/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Author: Francesc Alted <francesc@blosc.org>
  Creation date: 2009-05-20

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

/*********************************************************************
  The code in this file is heavily based on FastLZ, a lightning-fast
  lossless compression library.  See LICENSES/FASTLZ.txt for details.
**********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "blosclz.h"
#include "fastcopy.h"
#include "blosc-common.h"


/*
 * Check for bound when decompressing.
 * It is a good idea to define this while developing.
 */
#undef BLOSCLZ_SAFE

/*
 * Give hints to the compiler for branch prediction optimization.
 */
#if defined(__GNUC__) && (__GNUC__ > 2)
#define BLOSCLZ_EXPECT_CONDITIONAL(c)    (__builtin_expect((c), 1))
#define BLOSCLZ_UNEXPECT_CONDITIONAL(c)  (__builtin_expect((c), 0))
#else
#define BLOSCLZ_EXPECT_CONDITIONAL(c)    (c)
#define BLOSCLZ_UNEXPECT_CONDITIONAL(c)  (c)
#endif

/*
 * Use inlined functions for supported systems.
 */
#if defined(_MSC_VER) && !defined(__cplusplus)   /* Visual Studio */
#define inline __inline  /* Visual C is not C99, but supports some kind of inline */
#endif

#define MAX_COPY 32
#define MAX_DISTANCE 8191
#define MAX_FARDISTANCE (65535 + MAX_DISTANCE - 1)

#ifdef BLOSC_STRICT_ALIGN
  #define BLOSCLZ_READU16(p) ((p)[0] | (p)[1]<<8)
#else
  #define BLOSCLZ_READU16(p) *((const uint16_t*)(p))
#endif


/* Simple, but pretty effective hash function for 3-byte sequence */
#define HASH_FUNCTION(v, p, l) {                     \
  v = BLOSCLZ_READU16(p);                            \
  v ^= BLOSCLZ_READU16(p + 1) ^ ( v >> (16 - l));    \
  v &= (1 << l) - 1;                                 \
}

#define LITERAL(ip, op, op_limit, anchor, copy) {        \
  if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + 2 > op_limit))   \
    goto out;                                            \
  *op++ = *anchor++;                                     \
  ip = anchor;                                           \
  copy++;                                                \
  if(BLOSCLZ_UNEXPECT_CONDITIONAL(copy == MAX_COPY)) {   \
    copy = 0;                                            \
    *op++ = MAX_COPY-1;                                  \
  }                                                      \
  continue;                                              \
}

#define IP_BOUNDARY 2



static inline uint8_t *get_run(uint8_t *ip, const uint8_t *ip_bound, const uint8_t *ref) {
  uint8_t x = ip[-1];
  int64_t value, value2;
  /* Broadcast the value for every byte in a 64-bit register */
  memset(&value, x, 8);
  /* safe because the outer check against ip limit */
  while (ip < (ip_bound - sizeof(int64_t))) {
#if defined(BLOSC_STRICT_ALIGN)
    memcpy(&value2, ref, 8);
#else
    value2 = ((int64_t*)ref)[0];
#endif
    if (value != value2) {
      /* Find the byte that starts to differ */
      while (*ref++ == x) ip++;
      return ip;
    }
    else {
      ip += 8;
      ref += 8;
    }
  }
  /* Look into the remainder */
  while ((ip < ip_bound) && (*ref++ == x)) ip++;
  return ip;
}

#ifdef __SSE2__
static inline uint8_t *get_run_16(uint8_t *ip, const uint8_t *ip_bound, const uint8_t *ref) {
  uint8_t x = ip[-1];
  __m128i value, value2, cmp;

  /* Broadcast the value for every byte in a 128-bit register */
  memset(&value, x, sizeof(__m128i));
  /* safe because the outer check against ip limit */
  while (ip < (ip_bound - sizeof(__m128i))) {
    value2 = _mm_loadu_si128((__m128i *)ref);
    cmp = _mm_cmpeq_epi32(value, value2);
    if (_mm_movemask_epi8(cmp) != 0xFFFF) {
      /* Find the byte that starts to differ */
      while (*ref++ == x) ip++;
      return ip;
    }
    else {
      ip += sizeof(__m128i);
      ref += sizeof(__m128i);
    }
  }
  /* Look into the remainder */
  while ((ip < ip_bound) && (*ref++ == x)) ip++;
  return ip;
}
#endif


#ifdef __AVX2__
static inline uint8_t *get_run_32(uint8_t *ip, const uint8_t *ip_bound, const uint8_t *ref) {
  uint8_t x = ip[-1];
  __m256i value, value2, cmp;

  /* Broadcast the value for every byte in a 256-bit register */
  memset(&value, x, sizeof(__m256i));
  /* safe because the outer check against ip limit */
  while (ip < (ip_bound - (sizeof(__m256i)))) {
    value2 = _mm256_loadu_si256((__m256i *)ref);
    cmp = _mm256_cmpeq_epi64(value, value2);
    if (_mm256_movemask_epi8(cmp) != 0xFFFFFFFF) {
      /* Find the byte that starts to differ */
      while (*ref++ == x) ip++;
      return ip;
    }
    else {
      ip += sizeof(__m256i);
      ref += sizeof(__m256i);
    }
  }
  /* Look into the remainder */
  while ((ip < ip_bound) && (*ref++ == x)) ip++;
  return ip;
}
#endif


/* Find the byte that starts to differ */
uint8_t *get_match(uint8_t *ip, const uint8_t *ip_bound, const uint8_t *ref) {
#if !defined(BLOSC_STRICT_ALIGN)
  while (ip < (ip_bound - sizeof(int64_t))) {
    if (((int64_t*)ref)[0] != ((int64_t*)ip)[0]) {
      /* Find the byte that starts to differ */
      while (*ref++ == *ip++) {}
      return ip;
    }
    else {
      ip += sizeof(int64_t);
      ref += sizeof(int64_t);
    }
  }
#endif
  /* Look into the remainder */
  while ((ip < ip_bound) && (*ref++ == *ip++)) {}
  return ip;
}


#if defined(__SSE2__)
uint8_t *get_match_16(uint8_t *ip, const uint8_t *ip_bound, const uint8_t *ref) {
  __m128i value, value2, cmp;

  while (ip < (ip_bound - sizeof(__m128i))) {
    value = _mm_loadu_si128((__m128i *) ip);
    value2 = _mm_loadu_si128((__m128i *) ref);
    cmp = _mm_cmpeq_epi32(value, value2);
    if (_mm_movemask_epi8(cmp) != 0xFFFF) {
      /* Find the byte that starts to differ */
      while (*ref++ == *ip++) {}
      return ip;
    }
    else {
      ip += sizeof(__m128i);
      ref += sizeof(__m128i);
    }
  }
  /* Look into the remainder */
  while ((ip < ip_bound) && (*ref++ == *ip++)) {}
  return ip;
}
#endif


#if defined(__AVX2__)
uint8_t *get_match_32(uint8_t *ip, const uint8_t *ip_bound, const uint8_t *ref) {
  __m256i value, value2, cmp;

  while (ip < (ip_bound - sizeof(__m256i))) {
    value = _mm256_loadu_si256((__m256i *) ip);
    value2 = _mm256_loadu_si256((__m256i *)ref);
    cmp = _mm256_cmpeq_epi64(value, value2);
    if (_mm256_movemask_epi8(cmp) != 0xFFFFFFFF) {
      /* Find the byte that starts to differ */
      while (*ref++ == *ip++) {}
      return ip;
    }
    else {
      ip += sizeof(__m256i);
      ref += sizeof(__m256i);
    }
  }
  /* Look into the remainder */
  while ((ip < ip_bound) && (*ref++ == *ip++)) {}
  return ip;
}
#endif


int blosclz_compress(const int opt_level, const void* input, int length,
                     void* output, int maxout) {
  uint8_t* ip = (uint8_t*)input;
  uint8_t* ibase = (uint8_t*)input;
  uint8_t* ip_bound = ip + length - IP_BOUNDARY;
  uint8_t* ip_limit = ip + length - 12;
  uint8_t* op = (uint8_t*)output;

  /* Hash table depends on the opt level.  Hash_log cannot be larger than 15. */
  /* The parametrization below is made from playing with the bench suite, like:
     $ bench/bench blosclz single 4
     $ bench/bench blosclz single 4 4194280 12 25
     and taking the minimum times on a i5-3380M @ 2.90GHz.
     Curiously enough, values >= 14 does not always
     get maximum compression, even with large blocksizes. */
  int8_t hash_log_[10] = {-1, 15, 15, 15, 15, 15, 15, 15, 15, 15};
  uint8_t hash_log = hash_log_[opt_level];
  uint16_t hash_size = 1 << hash_log;
  uint16_t* htab;
  uint8_t* op_limit;

  int32_t hval;
  uint8_t copy;

  double maxlength_[10] = {-1, .1, .3, .5, .6, .8, .9, .95, 1.0, 1.0};
  int32_t maxlength = (int32_t)(length * maxlength_[opt_level]);
  if (maxlength > (int32_t)maxout) {
    maxlength = (int32_t)maxout;
  }
  op_limit = op + maxlength;

  /* output buffer cannot be less than 66 bytes or we can get into trouble */
  if (BLOSCLZ_UNEXPECT_CONDITIONAL(maxout < 66 || length < 4)) {
    return 0;
  }

  htab = (uint16_t*)calloc(hash_size, sizeof(uint16_t));

  /* we start with literal copy */
  copy = 2;
  *op++ = MAX_COPY - 1;
  *op++ = *ip++;
  *op++ = *ip++;

  /* main loop */
  while (BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit)) {
    const uint8_t* ref;
    int32_t distance;
    int32_t len = 3;         /* minimum match length */
    uint8_t* anchor = ip;  /* comparison starting-point */

    /* check for a run */
    if (ip[0] == ip[-1] && BLOSCLZ_READU16(ip - 1) == BLOSCLZ_READU16(ip + 1)) {
      distance = 1;
      ip += 3;
      ref = anchor - 1 + 3;
      goto match;
    }

    /* find potential match */
    HASH_FUNCTION(hval, ip, hash_log);
    ref = ibase + htab[hval];

    /* calculate distance to the match */
    distance = (int32_t)(anchor - ref);

    /* update hash table if necessary */
    /* not exactly sure why 0x1F works best, but experiments apparently say so */
    if ((distance & 0x1F) == 0)
      htab[hval] = (uint16_t)(anchor - ibase);

    /* is this a match? check the first 3 bytes */
    if (distance == 0 || (distance >= MAX_FARDISTANCE) ||
        *ref++ != *ip++ || *ref++ != *ip++ || *ref++ != *ip++) {
      LITERAL(ip, op, op_limit, anchor, copy);
    }

    /* far, needs at least 5-byte match */
    if (opt_level >= 5 && distance >= MAX_DISTANCE) {
      if (*ip++ != *ref++ || *ip++ != *ref++) LITERAL(ip, op, op_limit, anchor, copy);
      len += 2;
    }

    match:

    /* last matched byte */
    ip = anchor + len;

    /* distance is biased */
    distance--;

    if (!distance) {
      /* zero distance means a run */
#if defined(__AVX2__)
      ip = get_run_32(ip, ip_bound, ref);
#elif defined(__SSE2__)
      ip = get_run_16(ip, ip_bound, ref);
#else
      ip = get_run(ip, ip_bound, ref);
#endif
    }
    else {
#if defined(__AVX2__)
      /* Experiments show that the SSE2 version is a bit faster, even on AVX2 processors */
      ip = get_match_16(ip, ip_bound + IP_BOUNDARY, ref);
#elif defined(__SSE2__)
      ip = get_match_16(ip, ip_bound + IP_BOUNDARY, ref);
#else
      ip = get_match(ip, ip_bound + IP_BOUNDARY, ref);
#endif
    }

    /* if we have copied something, adjust the copy count */
    if (copy)
      /* copy is biased, '0' means 1 byte copy */
      *(op - copy - 1) = copy - 1;
    else
      /* back, to overwrite the copy count */
      op--;

    /* reset literal counter */
    copy = 0;

    /* length is biased, '1' means a match of 3 bytes */
    ip -= 3;
    len = (int32_t)(ip - anchor);

    /* check that we have space enough to encode the match for all the cases */
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + (len / 255) + 6 > op_limit)) goto out;

    /* encode the match */
    if (distance < MAX_DISTANCE) {
      if (len < 7) {
        *op++ = (len << 5) + (distance >> 8);
        *op++ = (distance & 255);
      }
      else {
        *op++ = (uint8_t)((7 << 5) + (distance >> 8));
        for (len -= 7; len >= 255; len -= 255)
          *op++ = 255;
        *op++ = len;
        *op++ = (distance & 255);
      }
    }
    else {
      /* far away, but not yet in the another galaxy... */
      if (len < 7) {
        distance -= MAX_DISTANCE;
        *op++ = (uint8_t)((len << 5) + 31);
        *op++ = 255;
        *op++ = (uint8_t)(distance >> 8);
        *op++ = distance & 255;
      }
      else {
        distance -= MAX_DISTANCE;
        *op++ = (7 << 5) + 31;
        for (len -= 7; len >= 255; len -= 255)
          *op++ = 255;
        *op++ = len;
        *op++ = 255;
        *op++ = (uint8_t)(distance >> 8);
        *op++ = distance & 255;
      }
    }

    /* update the hash at match boundary */
    HASH_FUNCTION(hval, ip, hash_log);
    htab[hval] = (uint16_t)(ip++ - ibase);
    HASH_FUNCTION(hval, ip, hash_log);
    htab[hval] = (uint16_t)(ip++ - ibase);

    /* assuming literal copy */
    *op++ = MAX_COPY - 1;
  }

  /* left-over as literal copy */
  ip_bound++;
  while (ip <= ip_bound) {
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + 2 > op_limit)) goto out;
    *op++ = *ip++;
    copy++;
    if (copy == MAX_COPY) {
      copy = 0;
      *op++ = MAX_COPY - 1;
    }
  }

  /* if we have copied something, adjust the copy length */
  if (copy)
    *(op - copy - 1) = copy - 1;
  else
    op--;

  /* marker for blosclz */
  *(uint8_t*)output |= (1 << 5);

  free(htab);
  return (int)(op - (uint8_t*)output);

  out:
  free(htab);
  return 0;

}

int blosclz_decompress(const void* input, int length, void* output, int maxout) {
  const uint8_t* ip = (const uint8_t*)input;
  const uint8_t* ip_limit = ip + length;
  uint8_t* op = (uint8_t*)output;
  int32_t ctrl = (*ip++) & 31;
  int32_t loop = 1;
#ifdef BLOSCLZ_SAFE
  uint8_t* op_limit = op + maxout;
#endif

  do {
    uint8_t* ref = op;
    int32_t len = ctrl >> 5;
    int32_t ofs = (ctrl & 31) << 8;

    if (ctrl >= 32) {
      uint8_t code;
      len--;
      ref -= ofs;
      if (len == 7 - 1)
        do {
          code = *ip++;
          len += code;
        } while (code == 255);
      code = *ip++;
      ref -= code;

      /* match from 16-bit distance */
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(code == 255)) if (BLOSCLZ_EXPECT_CONDITIONAL(ofs == (31 << 8))) {
        ofs = (*ip++) << 8;
        ofs += *ip++;
        ref = op - ofs - MAX_DISTANCE;
      }

#ifdef BLOSCLZ_SAFE
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + len + 3 > op_limit)) {
        return 0;
      }

      if (BLOSCLZ_UNEXPECT_CONDITIONAL(ref - 1 < (uint8_t*)output)) {
        return 0;
      }
#endif

      if (BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit))
        ctrl = *ip++;
      else
        loop = 0;

      if (ref == op) {
        /* optimized copy for a run */
        uint8_t b = ref[-1];
        memset(op, b, len + 3);
        op += len + 3;
      }
      else {
        /* copy from reference */
        ref--;
        len += 3;
        op = safecopy(op, ref, (unsigned) len);
      }
    }
    else {
      ctrl++;
#ifdef BLOSCLZ_SAFE
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + ctrl > op_limit)) {
        return 0;
      }
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(ip + ctrl > ip_limit)) {
        return 0;
      }
#endif

      // memcpy(op, ip, ctrl); op += ctrl; ip += ctrl;
      // On GCC-6, fastcopy this is still faster than plain memcpy
      // However, using recent CLANG/LLVM 9.0, there is almost no difference
      // in performance.
      op = fastcopy(op, ip, (unsigned) ctrl);
      ip += ctrl;

      loop = (int32_t)BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit);
      if (loop)
        ctrl = *ip++;
    }
  } while (BLOSCLZ_EXPECT_CONDITIONAL(loop));

  return (int)(op - (uint8_t*)output);
}
