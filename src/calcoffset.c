/* Routine to compute the offsets of a packed struct (package struct in Python)
 * F.Altet
 * 2002/08/28 */

#include <stddef.h>
#include <stdlib.h>
#include "hdf5.h"
#include "calcoffset.h"
#include "utils.h"

/* For the sake of code simplicity I've stripped out all the alignment
   stuff. It not necessary because the offset for each element on
   Table struct is computed prior to read the data. If, for any
   reason, it is needed again, you can find the complete code version
   in CVS with tag 1.7.  F. Altet 2004-09-16 */

/* Get the correct HDF5 type for a format code.
 * I can't manage to do the mapping with a table because
 * the HDF5 types are not constant values and are
 * defined by executing a function. 
 * So we do that in a switch case. */

static hid_t conventry(int c, int rank, hsize_t *dims, char *byteorder)
{
   hid_t string_type;
   hid_t ret_type;
   hid_t tid, tid2;
   int atomic, i;
   hsize_t shape[MAXDIM];

   if (rank == 1 && dims[0] == 1 )
     atomic = 1;
   else
     atomic = 0;

   switch(c) {
    case 'c':
      if (atomic == 1) {
	/* 'c' is used to represent a boolean value instead of
	   character, which was unused (we have 's' for strings).
	*/
	ret_type = H5Tcopy(H5T_NATIVE_B8);
	H5Tset_precision(ret_type, 1);
	return ret_type;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_B8, rank, dims, NULL);
      }
    case 'b':
      if (atomic == 1) {
	return H5T_NATIVE_SCHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_SCHAR, rank, dims, NULL);
      }
    case 'B':
      if (atomic == 1) {
	return H5T_NATIVE_UCHAR;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_UCHAR, rank, dims, NULL);
      }
    case 'h':
      if (atomic == 1) {
	return H5T_NATIVE_SHORT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_SHORT, rank, dims, NULL);
      }
    case 'H':
      if (atomic == 1) {
	return H5T_NATIVE_USHORT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_USHORT, rank, dims, NULL);
      }
    case 'i':
      if (atomic == 1) {
	return H5T_NATIVE_INT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_INT, rank, dims, NULL);
      }
    case 'I':
      if (atomic == 1) {
	return H5T_NATIVE_UINT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_UINT, rank, dims, NULL);
      }
    case 'l':
      if (atomic == 1) {
	return H5T_NATIVE_LONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_LONG, rank, dims, NULL);
      }
    case 'L':
      if (atomic == 1) {
	return H5T_NATIVE_ULONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_ULONG, rank, dims, NULL);
      }
    case 'q':
      if (atomic == 1) {
	return H5T_NATIVE_LLONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_LLONG, rank, dims, NULL);
      }
      return H5T_NATIVE_LLONG;
    case 'Q':
      if (atomic == 1) {
	return H5T_NATIVE_ULLONG;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_ULLONG, rank, dims, NULL);
      }
    case 'f':
      if (atomic == 1) {
	return H5T_NATIVE_FLOAT;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_FLOAT, rank, dims, NULL);
      }
    case 'd':
      if (atomic == 1) {
	return H5T_NATIVE_DOUBLE;
      } 
      else {
	 return H5Tarray_create(H5T_NATIVE_DOUBLE, rank, dims, NULL);
      }
    case 'F':
      tid = create_native_complex32(byteorder);
      if (atomic == 1) {
	return tid;
      } 
      else {
	 tid2 = H5Tarray_create(tid, rank, dims, NULL);
	 H5Tclose(tid);
	 return tid2;
      }
    case 'D':
      tid = create_native_complex64(byteorder);
      if (atomic == 1) {
	return tid;
      } 
      else {
	 tid2 = H5Tarray_create(tid, rank, dims, NULL);
	 H5Tclose(tid);
	 return tid2;
      }
    case 's':
      string_type = H5Tcopy(H5T_C_S1);
      H5Tset_size(string_type, (size_t)dims[rank-1]);
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


int calcoffset(char *fmt, int *nattrs, hid_t *types,
	       size_t *size_types, size_t *offsets)
{
   const char *s;
   hsize_t shape[MAXDIM];
   char c;
   int ndim;
   int size, num, itemsize, x;
   hid_t hdf5type;
   char byteorder[10];

   /* Get the byteorder */
   switch (*fmt) {
   case '<':
     strcpy(byteorder, "little");
     break;
   case '>':
     strcpy(byteorder, "big");
     break;
   case '!': /* Network byte order is big-endian */
     strcpy(byteorder, "big");
     break;
   case '=': { /* Host byte order -- different from native in aligment! */
     int n = 1;
     char *p = (char *) &n;
     if (*p == 1)
       strcpy(byteorder, "little");
     else
       strcpy(byteorder, "big");
   }
   }
   
   fmt++; 			/* Skip the alignment info */
   s = fmt;
   size = 0;
   *nattrs = 0;
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

      /* Get the proper hdf5 type */
      hdf5type = H5Tcopy(conventry(c, ndim+1, shape, byteorder));
      if (hdf5type == -1)
	return -1;
      itemsize = H5Tget_size(hdf5type);

      /* Feed the return values */
      size += itemsize;
      *offsets++ = size;
      *size_types++ = itemsize;
      *nattrs += 1;
      *types++ = hdf5type;

      /* Set the byteorder datatype (if needed) */
      if (c != 's' || c != 'F' || c != 'D') {
	set_order(hdf5type, byteorder);
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
