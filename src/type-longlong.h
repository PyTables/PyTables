#include "Python.h"

/* It seems python 2.2 defines LONG_LONG and python 2.3 defines
   PY_LONG_LONG.  There's a well published and easy check for this:
 */
#if !defined(PY_LONG_LONG) && defined(LONG_LONG) 
#define PY_LONG_LONG LONG_LONG
#endif
