/* Include file for calcoffset.c 
 * F. Alted 
 * 2002/08/28 */

#include "type-longlong.h"
#include "hdf5.h"

/* Define this variable for error printings */
/*#define DEBUG 1 */
/* Define this variable for debugging printings */
/*#define PRINT 1 */
/* Define this for compile the main() function */
/* #define MAIN 1 */

/* Define various structs to figure out the alignments of types */

#ifdef __MWERKS__
/*
** XXXX We have a problem here. There are no unique alignment rules
** on the PowerPC mac. 
*/
#ifdef __powerc
#pragma options align=mac68k
#endif
#endif /* __MWERKS__ */

typedef struct { char c; short x; } s_short;
typedef struct { char c; int x; } s_int;
typedef struct { char c; long x; } s_long;
typedef struct { char c; float x; } s_float;
typedef struct { char c; double x; } s_double;
typedef struct { char c; LL_TYPE x; } s_longlong;
typedef struct { char c; void *x; } s_void_p;

#define SHORT_ALIGN (sizeof(s_short) - sizeof(short))
#define INT_ALIGN (sizeof(s_int) - sizeof(int))
#define LONG_ALIGN (sizeof(s_long) - sizeof(long))
#define FLOAT_ALIGN (sizeof(s_float) - sizeof(float))
#define DOUBLE_ALIGN (sizeof(s_double) - sizeof(double))
#define LONGLONG_ALIGN (sizeof(s_longlong) - sizeof(LL_TYPE))
#define VOID_P_ALIGN (sizeof(s_void_p) - sizeof(void *))

#ifdef __powerc
#pragma options align=reset
#endif

typedef struct _formatdef {
   char format;
   int size;
   int alignment;
} formatdef;

static formatdef native_table[] = {
     {'x',	sizeof(char),   	0},
     {'b',	sizeof(char),   	0},
     {'B',	sizeof(char),   	0},
     {'c',	sizeof(char),		0},
     {'s',	sizeof(char),		0},
     {'p',	sizeof(char),		0},
     {'h',	sizeof(short),		SHORT_ALIGN},
     {'H',	sizeof(short),		SHORT_ALIGN},
     {'i',	sizeof(int),		INT_ALIGN},
     {'I',	sizeof(int),		INT_ALIGN},
     {'l',	sizeof(long),		LONG_ALIGN},
     {'L',	sizeof(long),		LONG_ALIGN},
     {'f',	sizeof(float),		FLOAT_ALIGN},
     {'d',	sizeof(double),		DOUBLE_ALIGN},
     {'q',	sizeof(LL_TYPE),	LONGLONG_ALIGN},
     {'Q',	sizeof(LL_TYPE),	LONGLONG_ALIGN},
/*     {'P',	sizeof(void *),		VOID_P_ALIGN},*/ /* Not supported */

     {0}
};

static formatdef bigendian_table[] = {
     {'x',	1,		0},
     {'b',	1,		0},
     {'B',	1,		0},
     {'c',	1,		0},
     {'s',	1,		0},
     {'h',	2,		0},
     {'H',	2,		0},
     {'i',	4,		0},
     {'I',	4,		0},
     {'l',	4,		0},
     {'L',	4,		0},
     {'f',	4,		0},
     {'d',	8,		0},
     {'q',	8,		0},
     {'Q',	8,		0},
     {0}
};

static formatdef lilendian_table[] = {
     {'x',	1,		0},
     {'b',	1,		0},
     {'B',	1,		0},
     {'c',	1,		0},
     {'s',	1,		0},
     {'h',	2,		0},
     {'H',	2,		0},
     {'i',	4,		0},
     {'I',	4,		0},
     {'l',	4,		0},
     {'L',	4,		0},
     {'f',	4,		0},
     {'d',	8,		0},
     {'q',	8,		0},
     {'Q',	8,		0},
     {0}
};

/* Functions in calcoffset.c we want accessible */
int calcoffset(char *fmt, size_t *offsets);

int calctypes(char *fmt, hid_t *types, size_t *size_types);

