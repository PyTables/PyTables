#ifdef _MSC_VER
/* #   define LL_TYPE LONG_LONG */
#   define LL_TYPE __int64    	/* Necessary for python 2.3 */
#   define MY_MSC 1
#else /*_MSC_VER*/
#   define LL_TYPE long long
#   define MY_MSC 0
#endif /*_MSC_VER*/

