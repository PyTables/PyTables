/* Routine to compute the offsets of a packed struct (package struct in Python)
 * F.Alted
 * 2002/08/28 */

#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include "calcoffset.h"
#include "utils.h"  /* To access the MAXDIM value */

/* Define this as 1 if native alignment is needed, although this
   doesn't seem to be necessary. Before pytables 0.6 this was
   implicitely set to 1, but the fact is that both values will
   work. Why?.  */

#ifndef ALIGN
#define ALIGN 0
#endif

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
#endif /* DEBUG */
	return NULL;
}


/* Get the correct HDF5 type for a format code.
 * I can't manage to do the mapping by a table because
 * the HDF5 types are not constant values and are
 * defined by executing a function. 
 * So we do that in a switch case. */

static hid_t conventry(int c, int rank, hsize_t *dims)
{
   hid_t string_type;
   hid_t ret_type;
   int native, i;
   hsize_t shape[MAXDIM];

   if (rank == 1 && dims[0] == 1 )
     native = 1;
   else
     native = 0;

   switch(c) {
    case 'c':
      if (native == 1) {
	return H5T_NATIVE_CHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_CHAR, rank, dims, NULL);
      }
    case 'b':
      if (native == 1) {
	return H5T_NATIVE_SCHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_SCHAR, rank, dims, NULL);
      }
    case 'B':
      if (native == 1) {
	return H5T_NATIVE_UCHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_UCHAR, rank, dims, NULL);
      }
    case 'h':
      if (native == 1) {
	return H5T_NATIVE_SHORT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_SHORT, rank, dims, NULL);
      }
    case 'H':
      if (native == 1) {
	return H5T_NATIVE_USHORT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_USHORT, rank, dims, NULL);
      }
    case 'i':
      if (native == 1) {
	return H5T_NATIVE_INT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_INT, rank, dims, NULL);
      }
    case 'I':
      if (native == 1) {
	return H5T_NATIVE_UINT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_UINT, rank, dims, NULL);
      }
    case 'l':
      if (native == 1) {
	return H5T_NATIVE_LONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_LONG, rank, dims, NULL);
      }
    case 'L':
      if (native == 1) {
	return H5T_NATIVE_ULONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_ULONG, rank, dims, NULL);
      }
    case 'q':
      if (native == 1) {
	return H5T_NATIVE_LLONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_LLONG, rank, dims, NULL);
      }
      return H5T_NATIVE_LLONG;
    case 'Q':
      if (native == 1) {
	return H5T_NATIVE_ULLONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_ULLONG, rank, dims, NULL);
      }
    case 'f':
      if (native == 1) {
	return H5T_NATIVE_FLOAT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_FLOAT, rank, dims, NULL);
      }
    case 'd':
      if (native == 1) {
	return H5T_NATIVE_DOUBLE;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_DOUBLE, rank, dims, NULL);
      }
    case 's':
      string_type = H5Tcopy(H5T_C_S1);
      H5Tset_size(string_type, dims[rank-1]);
      if (rank == 1) {
	return string_type;
      }
      else {
	/* Build a shape array with rank-1 elements */
	for(i=0; i<rank-1; i++) {
	  shape[i] = dims[i];
	}
	ret_type = H5Tarray_create(string_type, rank-1, shape, NULL);
	/* Release resources */
	H5Tclose(string_type);
	return ret_type;
      }
    default:
#ifdef DEBUG
      printf("Error: bad char <%c> in struct format\n", c);
#endif /* DEBUG */
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
 * The format follows the directions for the package struct, plus
 * extensions for multimensional elements. The notation for
 * multimensional elements are in the form (####, ####, ...[,]).

 * The number of fields are returned in the nattrs integer.
 * The HDF5 types are computed and returned in the types array.
 * The sizes of types are computed and returned in the size_types array.
 * The offsets are computed and returned in the offsets array.
 * calcoffset returns the row size. */

int calcoffset(char *fmt, int *nattrs, hid_t *types,
	      size_t *size_types, size_t *offsets)
{
   const formatdef *f, *e;
   const char *s;
   hsize_t shape[MAXDIM];
   char c;
   int ndim;
   int size, num, itemsize, x;
   hid_t hdf5type;
   char byteorder;

   byteorder = fmt[0];
   f = whichtable(&fmt);
   s = fmt;
   size = 0;
   *nattrs = 0;
   if (!ALIGN)
     *offsets++ = 0;

   while ((c = *s++) != '\0') {
      ndim = 0;
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
#endif /* DEBUG */
	  num = x;
	}
	shape[0] = num;
	if (c == '\0')
	  break;
      }
      /* Special case for multidimensional cell elements */
      else if (c == '(') {
	while ((c = *s++) != ')') {
	  if (isspace((int)c)) {
	    continue;
	  }
	  if (c == ',') {
	    continue;
	  }
	  if ('0' <= c && c <= '9') {
	    num = c - '0';
	    while ('0' <= (c = *s++) && c <= '9') {
	      x = num*10 + (c - '0');
#ifdef DEBUG
	      if (x/10 != num) {
		printf("overflow in item count\n");
		return -1;
	      }
#endif /* DEBUG */
	      num = x;
	    }
	    shape[ndim++] = num;
	    /* Special case for ...,####,) tuple */
	    if (c == ',' && s[0] == ')') {
	      c = *s++;
	      c = *s++;
	      break;
	    }
	    if (c == ')') {
	      c = *s++;
	      break;
	    }
 	  }
	}
	ndim--;
      }
      else {
	num = 1;
	shape[0] = num; 
      }

#ifdef PRINT
      printf("e--> %c, ", c);
      printf("(");
      for (i=0; i<=ndim; i++) {
	printf("%d,", shape[i]);
      }
      printf(")\n");
#endif
      e = getentry(c, f);
      if (e == NULL)
	return -1;
      /* Get the proper hdf5 type */
      hdf5type = H5Tcopy(conventry(c, ndim+1, shape));
      if (hdf5type == -1)
	return -1;
      itemsize = H5Tget_size(hdf5type);

      /* Feed the return values */
      if(ALIGN)
	size = align(size, c, e);
      else
	size += itemsize;

      *offsets++ = size;
      if (ALIGN)
	size += itemsize;

      *size_types++ = itemsize;
      *nattrs += 1;
      *types++ = hdf5type;

      /* Set the byteorder datatype (if needed) */
      if (c != 's') {
	if (byteorder == '<') 
	  H5Tset_order(hdf5type, H5T_ORDER_LE);
	else if (byteorder == '>') {
	  H5Tset_order(hdf5type, H5T_ORDER_BE );
	}
      }
   }

#ifdef PRINT
   printf("The size computed in calcoffset is %d\n", size);
#endif /* PRINT */

   return size;
}

#ifdef MAIN
int main(int args, char *argv[])
{
   char  format[256] = "hch";
   char  *fmt;
   int   rowsize, nattrs;
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
     rowsize = calcoffset(fmt, &nattrs, types, size_types, offsets);
   }
   
   if (nattrs < 0)
     return -1;
   else
     printf("# attributes from calcoffset: %d\nOffsets: ", nattrs);
     for (i = 0; i < nattrs; i++)
       printf(" %d,", offsets[i]);
     printf("\n");
     printf("Rowsize: %d\nType sizes: ", rowsize);
     for (i = 0; i < nattrs; i++)
       printf(" %d,", size_types[i]);
     printf("\n");
}
#endif /* MAIN */
