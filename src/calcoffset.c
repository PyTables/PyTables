/* Routine to compute the offsets of a packed struct (package struct in Python)
 * F.Alted
 * 2002/08/28 */

#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include "calcoffset.h"

static const formatdef *whichtable(char **pfmt)
{
	const char *fmt = (*pfmt)++; /* May be backed out of later */
	switch (*fmt) {
	case '<':
		return lilendian_table;
	case '>':
	case '!': /* Network byte order is big-endian */
		return bigendian_table;
	case '=': { /* Host byte order -- different from native in aligment! */
		int n = 1;
		char *p = (char *) &n;
		if (*p == 1)
			return lilendian_table;
		else
			return bigendian_table;
	}
	default:
		--*pfmt; /* Back out of pointer increment */
		/* Fall through */
	case '@':
		return native_table;
	}
}


/* Get the table entry for a format code */

static const formatdef *getentry(int c, const formatdef *f)
{
	for (; f->format != '\0'; f++) {
		if (f->format == c) {
			return f;
		}
	}
#ifdef DEBUG
        printf("Error: bad char <%c> in struct format\n", c);
#endif DEBUG
	return NULL;
}


/* Get the correct HDF5 type for a format code.
 * I can't manage to do the mapping by a table because
 * the HDF5 types are not constant values and are
 * defined by executing a function. 
 * So we do that in a switch case. */

static hid_t conventry(int c, int numel)
{
   hid_t string_type;
   int rank;
   hsize_t dims[1];

   
   rank = 1;
   dims[0] = numel;
   switch(c) {
    case 'c':
      if (numel == 1) {
	return H5T_NATIVE_CHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_CHAR, 1, dims, NULL);
      }
    case 'b':
      if (numel == 1) {
	return H5T_NATIVE_SCHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_SCHAR, 1, dims, NULL);
      }
    case 'B':
      if (numel == 1) {
	return H5T_NATIVE_UCHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_UCHAR, 1, dims, NULL);
      }
    case 'h':
      if (numel == 1) {
	return H5T_NATIVE_SHORT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_SHORT, 1, dims, NULL);
      }
    case 'H':
      if (numel == 1) {
	return H5T_NATIVE_USHORT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_USHORT, 1, dims, NULL);
      }
    case 'i':
      if (numel == 1) {
	return H5T_NATIVE_INT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_INT, 1, dims, NULL);
      }
    case 'I':
      if (numel == 1) {
	return H5T_NATIVE_UINT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_UINT, 1, dims, NULL);
      }
    case 'l':
      if (numel == 1) {
	return H5T_NATIVE_LONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_LONG, 1, dims, NULL);
      }
    case 'L':
      if (numel == 1) {
	return H5T_NATIVE_ULONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_ULONG, 1, dims, NULL);
      }
    case 'q':
      if (numel == 1) {
	return H5T_NATIVE_LLONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_LLONG, 1, dims, NULL);
      }
      return H5T_NATIVE_LLONG;
    case 'Q':
      if (numel == 1) {
	return H5T_NATIVE_ULLONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_ULLONG, 1, dims, NULL);
      }
    case 'f':
      if (numel == 1) {
	return H5T_NATIVE_FLOAT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_FLOAT, 1, dims, NULL);
      }
    case 'd':
      if (numel == 1) {
	return H5T_NATIVE_DOUBLE;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims, NULL);
      }
    case 's':
      string_type = H5Tcopy(H5T_C_S1);
      H5Tset_size(string_type, numel);
      return string_type;
    default:
#ifdef DEBUG
      printf("Error: bad char <%c> in struct format\n", c);
#endif DEBUG
      return -1;
   }
}


/* Align a size according to a format code */

static int align(int size, int c, const formatdef *e)
{
   if (e->alignment) {
      size = ((size + e->alignment - 1)
	      / e->alignment)
	      * e->alignment;
   }
   return size;
}


/* Calculate the offsets of a format string.
 * The format follows strictly the directions for the package struct
 * and is contained as input in fmt.
 * The offsets are computed and returned in the offsets array.
 * calcoffset returns the number of variables in the fmt string */

/* This routine don't have error checking at all
 * If you want this, call struct.calcsize first */

int calcoffset(char *fmt, size_t *offsets)
{
   const formatdef *f, *e;
   const char *s;
   char c;
   size_t offset;
   int nattrib, size,  num, itemsize, x;

   f = whichtable(&fmt);
   s = fmt;
   size = 0;
   nattrib = 0;
   while ((c = *s++) != '\0') {
      if (isspace((int)c))
	continue;
      if ('0' <= c && c <= '9') {
	 num = c - '0';
	 while ('0' <= (c = *s++) && c <= '9') {
	    x = num*10 + (c - '0');
#ifdef DEBUG
	    if (x/10 != num) {
	       printf("overflow in item count\n");
	       return -1;
	    }
#endif DEBUG
	    num = x;
	 }
	 if (c == '\0')
	   break;
      }
      else
	num = 1;
      
      e = getentry(c, f);
      if (e == NULL)
	return -1;
      itemsize = e->size;
      size = align(size, c, e);
      if (num >= 1 && c != 'x') {
	 offset = size;
	 *offsets++ = offset;
	 /* The case for a number before string is special */
	 /*	 if (c != 's') {
	    for (j=0; j < num - 1; j++) {
	       offset += itemsize;
	       *offsets++ = offset;
	    }
	    nattrib += num;
	    } */
	 if (c != 's') {
	    nattrib ++;
	 } 
	 else {
	    /* In case of string, increment nattrib only by one */
	    nattrib++;
	 }
	 
      }
      x = num * itemsize;
      size += x;
#ifdef DEBUG
      if (x/itemsize != num || size < 0) {
	 printf("total struct size too long\n");
	 return -1;
      }
#endif DEBUG
   }

#ifdef PRINT
   printf("The size computed in calcoffset is %d\n", size);
#endif PRINT

   return nattrib;
}


int
calctypes(fmt, types, size_types)
    char *fmt;
    hid_t *types;
    size_t *size_types;
{
   const formatdef *f, *e;
   const char *s;
   char c;
   int nattrib, size,  num, itemsize, x;
   hid_t hdf5type;
   char byteorder;

   byteorder = fmt[0];
   f = whichtable(&fmt);
   s = fmt;
   size = 0;
   nattrib = 0;
   while ((c = *s++) != '\0') {
      if (isspace((int)c))
	continue;
      if ('0' <= c && c <= '9') {
	 num = c - '0';
	 while ('0' <= (c = *s++) && c <= '9') {
	    x = num*10 + (c - '0');
#ifdef DEBUG
	    if (x/10 != num) {
	       printf("overflow in item count\n");
	       return -1;
	    }
#endif DEBUG
	    num = x;
	 }
	 if (c == '\0')
	   break;
      }
      else
	num = 1;
      
      e = getentry(c, f);
      if (e == NULL)
	return -1;
      itemsize = e->size;
      size = align(size, c, e);
      if (num >= 1 && c != 'x') {
	hdf5type = H5Tcopy(conventry(c, num));
	if (hdf5type == -1)
	  return -1;
	if (c != 's') {
	  *size_types++ = num*itemsize;
	  /* Set the byteorder datatype (if needed) */
	  if (byteorder == '<') 
	    H5Tset_order(hdf5type, H5T_ORDER_LE);
	  else if (byteorder == '>') {
	    H5Tset_order(hdf5type, H5T_ORDER_BE );
	  }
	} 
	else { /* Case of string */
	  /* Set the type size equal to the string length */
	  *size_types++ = num;
	}
	/* Increment nattrib only by one */
	nattrib++;
	*types++ = hdf5type;
      }
      
      x = num * itemsize;
      size += x;
#ifdef DEBUG
      if (x/itemsize != num || size < 0) {
	 printf("total struct size too long\n");
	 return -1;
      }
#endif DEBUG
   }

#ifdef PRINT
   printf("The size computed in calctypes is %d\n", size);
#endif PRINT

/*    return nattrib; */
   return size;
}

#ifdef MAIN
int main(int args, char *argv[])
{
   char  format[256] = "hch";
   char  *fmt;
   int   rowsize, nattrib, nattrib2;
   size_t offsets[256], size_types[256];
   hid_t types[256];
   int   i, niter;
   const formatdef *f;
     
   printf("args # --> %d\n", args);
   printf("arg 0 --> %s\n", argv[0]);
   printf("arg 1 --> %s\n", argv[1]);

   fmt = argv[1];
   printf("The format is %s\n", fmt);
   niter = 1;
   if (args == 3) 
     niter = atoi(argv[2]);
   for (i=0;i<niter;i++) {
     nattrib = calcoffset(fmt, offsets);
     rowsize = calctypes(fmt, types, size_types);
   }
   
   if (nattrib < 0)
     return -1;
   else
     printf("# attributes from calcoffset: %d\nOffsets: ", nattrib);
     for (i = 0; i < nattrib; i++)
       printf(" %d,", offsets[i]);
     printf("\n");
     printf("Rowsize from calctype: %d\nType sizes: ", rowsize);
     for (i = 0; i < nattrib; i++)
       printf(" %d,", size_types[i]);
     printf("\n");
}
#endif MAIN
