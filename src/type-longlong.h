#ifdef _MSC_VER
/* #   define LL_TYPE LONG_LONG */
#   define LL_TYPE __int64    	/* Necessary for python 2.3 */
#   define MY_MSC 1
#else /*_MSC_VER*/
#   define LL_TYPE long long
#   define MY_MSC 0
#endif /*_MSC_VER*/


/* It seems python 2.2 defines LONG_LONG and python 2.3 defines
   PY_LONG_LONG.  There's a well published and easy check for this:
 */
#if !defined(PY_LONG_LONG) && defined(LONG_LONG) 
#define PY_LONG_LONG LONG_LONG
#endif
