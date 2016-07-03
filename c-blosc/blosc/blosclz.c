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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "blosclz.h"

#if defined(_WIN32) && !defined(__MINGW32__)
  #include <windows.h>

  /* stdint.h only available in VS2010 (VC++ 16.0) and newer */
  #if defined(_MSC_VER) && _MSC_VER < 1600
    #include "win32/stdint-windows.h"
  #else
    #include <stdint.h>
  #endif
  /* llabs only available in VS2013 (VC++ 18.0) and newer */
  #if defined(_MSC_VER) && _MSC_VER < 1800
    #define llabs(v) abs(v)
  #endif
#else
  #include <stdint.h>
#endif  /* _WIN32 */


/*
 * Prevent accessing more than 8-bit at once, except on x86 architectures.
 */
#if !defined(BLOSCLZ_STRICT_ALIGN)
#define BLOSCLZ_STRICT_ALIGN
#if defined(__i386__) || defined(__386) || defined (__amd64)  /* GNU C, Sun Studio */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(__i486__) || defined(__i586__) || defined(__i686__)  /* GNU C */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(_M_IX86) || defined(_M_X64)   /* Intel, MSVC */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(__386)
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(_X86_) /* MinGW */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(__I86__) /* Digital Mars */
#undef BLOSCLZ_STRICT_ALIGN
/* Seems like unaligned access in ARM (at least ARMv6) is pretty
   expensive, so we are going to always enforce strict aligment in ARM.
   If anybody suggest that newer ARMs are better, we can revisit this. */
/* #elif defined(__ARM_FEATURE_UNALIGNED) */  /* ARM, GNU C */
/* #undef BLOSCLZ_STRICT_ALIGN */
#endif
#endif

/*
 * Always check for bound when decompressing.
 * Generally it is best to leave it defined.
 */
#define BLOSCLZ_SAFE

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

#define MAX_COPY       32
#define MAX_DISTANCE 8191
#define MAX_FARDISTANCE (65535+MAX_DISTANCE-1)

#ifdef BLOSCLZ_STRICT_ALIGN
  #define BLOSCLZ_READU16(p) ((p)[0] | (p)[1]<<8)
#else
  #define BLOSCLZ_READU16(p) *((const uint16_t*)(p))
#endif


/*
 * Fast copy macros
 */
#if defined(_WIN32)
  #define CPYSIZE              32
#else
  #define CPYSIZE              8
#endif
#define MCPY(d,s)            { memcpy(d, s, CPYSIZE); d+=CPYSIZE; s+=CPYSIZE; }
#define FASTCOPY(d,s,e)      { do { MCPY(d,s) } while (d<e); }
#define SAFECOPY(d,s,e)      { while (d<e) { MCPY(d,s) } }

/* Copy optimized for copying in blocks */
#define BLOCK_COPY(op, ref, len, op_limit)    \
{ int ilen = len % CPYSIZE;                   \
  uint8_t *cpy = op + len;                    \
  if (cpy + CPYSIZE - ilen <= op_limit) {     \
    FASTCOPY(op, ref, cpy);                   \
    ref -= (op-cpy); op = cpy;                \
  }                                           \
  else {                                      \
    cpy -= ilen;                              \
    SAFECOPY(op, ref, cpy);                   \
    ref -= (op-cpy); op = cpy;                \
    for(; ilen; --ilen)	                      \
        *op++ = *ref++;                       \
  }                                           \
}

#define SAFE_COPY(op, ref, len, op_limit)     \
if (llabs(op-ref) < CPYSIZE) {                \
  for(; len; --len)                           \
    *op++ = *ref++;                           \
}                                             \
else BLOCK_COPY(op, ref, len, op_limit);

/* Copy optimized for GCC 4.8.  Seems like long copy loops are optimal. */
#define GCC_SAFE_COPY(op, ref, len, op_limit) \
if ((len > 32) || (llabs(op-ref) < CPYSIZE)) { \
  for(; len; --len)                           \
    *op++ = *ref++;                           \
}                                             \
else BLOCK_COPY(op, ref, len, op_limit);

/* Simple, but pretty effective hash function for 3-byte sequence */
#define HASH_FUNCTION(v, p, l) {                       \
    v = BLOSCLZ_READU16(p);                            \
    v ^= BLOSCLZ_READU16(p + 1) ^ ( v >> (16 - l));    \
    v &= (1 << l) - 1;                                 \
}

/* Another version which seems to be a bit more effective than the above,
 * but a bit slower.  Could be interesting for high opt_level.
 */
#define MINMATCH 3
#define HASH_FUNCTION2(v, p, l) {                       \
  v = BLOSCLZ_READU16(p);				\
  v = (v * 2654435761U) >> ((MINMATCH * 8) - (l + 1));  \
  v &= (1 << l) - 1;					\
}

#define LITERAL(ip, op, op_limit, anchor, copy) {        \
  if (BLOSCLZ_UNEXPECT_CONDITIONAL(op+2 > op_limit))     \
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


int blosclz_compress(const int opt_level, const void* input, int length,
                     void* output, int maxout, int accel)
{
  uint8_t* ip = (uint8_t*) input;
  uint8_t* ibase = (uint8_t*) input;
  uint8_t* ip_bound = ip + length - IP_BOUNDARY;
  uint8_t* ip_limit = ip + length - 12;
  uint8_t* op = (uint8_t*) output;

  /* Hash table depends on the opt level.  Hash_log cannot be larger than 15. */
  /* The parametrization below is made from playing with the bench suite, like:
     $ bench/bench blosclz single 4
     $ bench/bench blosclz single 4 4194280 12 25
     and taking the minimum times on a i5-3380M @ 2.90GHz.
     Curiously enough, values >= 14 does not always
     get maximum compression, even with large blocksizes. */
  int8_t hash_log_[10] = {-1, 11, 11, 11, 12, 13, 13, 13, 13, 13};
  uint8_t hash_log = hash_log_[opt_level];
  uint16_t hash_size = 1 << hash_log;
  uint16_t *htab;
  uint8_t* op_limit;

  int32_t hval;
  uint8_t copy;

  double maxlength_[10] = {-1, .1, .15, .2, .3, .45, .6, .75, .9, 1.0};
  int32_t maxlength = (int32_t) (length * maxlength_[opt_level]);
  if (maxlength > (int32_t) maxout) {
    maxlength = (int32_t) maxout;
  }
  op_limit = op + maxlength;

  /* output buffer cannot be less than 66 bytes or we can get into trouble */
  if (BLOSCLZ_UNEXPECT_CONDITIONAL(maxlength < 66 || length < 4)) {
    return 0;
  }

  /* prepare the acceleration to be used in condition */
  accel = accel < 1 ? 1 : accel;
  accel -= 1;

  htab = (uint16_t *) calloc(hash_size, sizeof(uint16_t));

  /* we start with literal copy */
  copy = 2;
  *op++ = MAX_COPY-1;
  *op++ = *ip++;
  *op++ = *ip++;

  /* main loop */
  while(BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit)) {
    const uint8_t* ref;
    int32_t distance;
    int32_t len = 3;         /* minimum match length */
    uint8_t* anchor = ip;  /* comparison starting-point */

    /* check for a run */
    if(ip[0] == ip[-1] && BLOSCLZ_READU16(ip-1)==BLOSCLZ_READU16(ip+1)) {
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
    if ((distance & accel) == 0)
      htab[hval] = (uint16_t)(anchor - ibase);

    /* is this a match? check the first 3 bytes */
    if (distance==0 || (distance >= MAX_FARDISTANCE) ||
        *ref++ != *ip++ || *ref++!=*ip++ || *ref++!=*ip++)
      LITERAL(ip, op, op_limit, anchor, copy);

    /* far, needs at least 5-byte match */
    if (opt_level >= 5 && distance >= MAX_DISTANCE) {
      if (*ip++ != *ref++ || *ip++ != *ref++)
        LITERAL(ip, op, op_limit, anchor, copy);
      len += 2;
    }

    match:

    /* last matched byte */
    ip = anchor + len;

    /* distance is biased */
    distance--;

    if(!distance) {
      /* zero distance means a run */
      uint8_t x = ip[-1];
      int64_t value, value2;
      /* Broadcast the value for every byte in a 64-bit register */
      memset(&value, x, 8);
      /* safe because the outer check against ip limit */
      while (ip < (ip_bound - (sizeof(int64_t) - IP_BOUNDARY))) {
#if !defined(BLOSCLZ_STRICT_ALIGN)
        value2 = ((int64_t *)ref)[0];
#else
        memcpy(&value2, ref, 8);
#endif
        if (value != value2) {
          /* Find the byte that starts to differ */
          while (ip < ip_bound) {
            if (*ref++ != x) break; else ip++;
          }
          break;
        }
        else {
          ip += 8;
          ref += 8;
        }
      }
      if (ip > ip_bound) {
        long l = (long)(ip - ip_bound);
        ip -= l;
        ref -= l;
      }   /* End of optimization */
    }
    else {
      for(;;) {
        /* safe because the outer check against ip limit */
        while (ip < (ip_bound - (sizeof(int64_t) - IP_BOUNDARY))) {
#if !defined(BLOSCLZ_STRICT_ALIGN)
          if (((int64_t *)ref)[0] != ((int64_t *)ip)[0]) {
#endif
            /* Find the byte that starts to differ */
            while (ip < ip_bound) {
              if (*ref++ != *ip++) break;
            }
            break;
#if !defined(BLOSCLZ_STRICT_ALIGN)
          } else { ip += 8; ref += 8; }
#endif
        }
        /* Last correction before exiting loop */
        if (ip > ip_bound) {
          int32_t l = (int32_t)(ip - ip_bound);
          ip -= l;
          ref -= l;
        }   /* End of optimization */
        break;
      }
    }

    /* if we have copied something, adjust the copy count */
    if (copy)
      /* copy is biased, '0' means 1 byte copy */
      *(op-copy-1) = copy-1;
    else
      /* back, to overwrite the copy count */
      op--;

    /* reset literal counter */
    copy = 0;

    /* length is biased, '1' means a match of 3 bytes */
    ip -= 3;
    len = (int32_t)(ip - anchor);

    /* check that we have space enough to encode the match for all the cases */
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op+(len/255)+6 > op_limit)) goto out;

    /* encode the match */
    if(distance < MAX_DISTANCE) {
      if(len < 7) {
        *op++ = (len << 5) + (distance >> 8);
        *op++ = (distance & 255);
      }
      else {
        *op++ = (uint8_t)((7 << 5) + (distance >> 8));
        for(len-=7; len >= 255; len-= 255)
          *op++ = 255;
        *op++ = len;
        *op++ = (distance & 255);
      }
    }
    else {
      /* far away, but not yet in the another galaxy... */
      if(len < 7) {
        distance -= MAX_DISTANCE;
        *op++ = (uint8_t)((len << 5) + 31);
        *op++ = 255;
        *op++ = (uint8_t)(distance >> 8);
        *op++ = distance & 255;
      }
      else {
        distance -= MAX_DISTANCE;
        *op++ = (7 << 5) + 31;
        for(len-=7; len >= 255; len-= 255)
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
    *op++ = MAX_COPY-1;
  }

  /* left-over as literal copy */
  ip_bound++;
  while(ip <= ip_bound) {
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op+2 > op_limit)) goto out;
    *op++ = *ip++;
    copy++;
    if(copy == MAX_COPY) {
      copy = 0;
      *op++ = MAX_COPY-1;
    }
  }

  /* if we have copied something, adjust the copy length */
  if(copy)
    *(op-copy-1) = copy-1;
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

int blosclz_decompress(const void* input, int length, void* output, int maxout)
{
  const uint8_t* ip = (const uint8_t*) input;
  const uint8_t* ip_limit  = ip + length;
  uint8_t* op = (uint8_t*) output;
  uint8_t* op_limit = op + maxout;
  int32_t ctrl = (*ip++) & 31;
  int32_t loop = 1;

  do {
    uint8_t* ref = op;
    int32_t len = ctrl >> 5;
    int32_t ofs = (ctrl & 31) << 8;

    if(ctrl >= 32) {
      uint8_t code;
      len--;
      ref -= ofs;
      if (len == 7-1)
        do {
          code = *ip++;
          len += code;
        } while (code==255);
      code = *ip++;
      ref -= code;

      /* match from 16-bit distance */
      if(BLOSCLZ_UNEXPECT_CONDITIONAL(code==255))
      if(BLOSCLZ_EXPECT_CONDITIONAL(ofs==(31 << 8))) {
        ofs = (*ip++) << 8;
        ofs += *ip++;
        ref = op - ofs - MAX_DISTANCE;
      }

#ifdef BLOSCLZ_SAFE
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + len + 3 > op_limit)) {
        return 0;
      }

      if (BLOSCLZ_UNEXPECT_CONDITIONAL(ref-1 < (uint8_t *)output)) {
        return 0;
      }
#endif

      if(BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit))
        ctrl = *ip++;
      else
        loop = 0;

      if(ref == op) {
        /* optimize copy for a run */
        uint8_t b = ref[-1];
        memset(op, b, len+3);
        op += len+3;
      }
      else {
        /* copy from reference */
        ref--;
        len += 3;
#if !defined(_WIN32) && ((defined(__GNUC__) || defined(__INTEL_COMPILER) || !defined(__clang__)))
        GCC_SAFE_COPY(op, ref, len, op_limit);
#else
        SAFE_COPY(op, ref, len, op_limit);
#endif
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

      BLOCK_COPY(op, ip, ctrl, op_limit);

      loop = (int32_t)BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit);
      if(loop)
        ctrl = *ip++;
    }
  } while(BLOSCLZ_EXPECT_CONDITIONAL(loop));

  return (int)(op - (uint8_t*)output);
}
