#include "arraytypes.h"

/* Get the correct HDF5 type for a format code.
 * I can't manage to do the mapping with a table because
 * the HDF5 types are not constant values and are
 * defined by executing a function. 
 * So we do that in a switch case. */

hid_t
  convArrayType(fmt)
    char fmt;
{
   hid_t s1;
   
   switch(fmt) {
    case 'c':
      /*      return H5T_NATIVE_CHAR; */
      /* An H5T_NATIVE_CHAR is interpreted as a signed byte by HDF5
       * so, we have to create a string type of lenght 1 so as to
       * represent a char.
       */
      s1 = H5Tcopy(H5T_C_S1);
      /* I use set_strpad instead of set_size as per section 3.6 
       * (Character and String Datatype Issues) of the HDF5 User's Manual,
       * altough they both seems to work well for character types */
      /* H5Tset_size(s1, 1); */
      H5Tset_strpad(s1, H5T_STR_NULLPAD);
      
      return s1;
    case 'b':
      return H5T_NATIVE_UCHAR;
    case '1':
      return H5T_NATIVE_SCHAR;
    case 's':
      return H5T_NATIVE_SHORT;
    case 'w':
      return H5T_NATIVE_USHORT;
    case 'i':
      return H5T_NATIVE_INT;
    case 'u':
      return H5T_NATIVE_UINT;
    case 'l':
      return H5T_NATIVE_LONG;
    case 'f':
      return H5T_NATIVE_FLOAT;
    case 'd':
      return H5T_NATIVE_DOUBLE;
    default:
#ifdef DEBUG
      printf("Error: bad char <%c> in array format\n", fmt);
#endif DEBUG
      return -1;
   }
}

/* Routine to map the atomic type to a Numeric typecode 
 */
int getArrayType(H5T_class_t class_id,
		 size_t type_size,
		 H5T_sign_t sign,
		 char *fmt) 
{
  switch(class_id) {
  case H5T_INTEGER:                /* int (byte, short, long, long long) */
    switch (type_size) {
    case 1:                        /* byte */
      if ( sign )
	*fmt = '1';                /* signed byte */
      else
	*fmt = 'b';                /* unsigned byte */
      break;
    case 2:                        /* short */
      if ( sign )
	 *fmt ='s';                /* signed short */
      else
	*fmt = 'w';                /* unsigned short */
      break;
    case 4:                        /* long */
      if ( sign )
	*fmt = 'i';                /* signed long */
      else
	*fmt = 'u';                /* unsigned long */
      break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_INTEGER */
  case H5T_FLOAT:                   /* float (single or double) */
    switch (type_size) {
    case 4:
	*fmt = 'f';                 /* float */
	break;
    case 8:
	*fmt = 'd';                 /* double */
	break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_FLOAT */
  case H5T_STRING:                  /* char or string */
    if ( type_size == 1 )
      *fmt = 'c';                   /* char */
    else {
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_STRING */
  default: /* Any other class type */
    /* This should never happen with Numeric arrays */
    fprintf(stderr, "class %d don't supported. Sorry!\n", class_id);
    goto out;
  }

  return 0;

 out:
  /* If we reach this line, there should be an error */
  return -1;
  
}

#ifdef MAIN
int main(int args, char *argv[])
{
   char  fmt;
     
   printf("args # --> %d\n", args);
   printf("arg 0 --> %s\n", argv[0]);
   printf("arg 1 --> %s\n", argv[1]);

   fmt = argv[1][0];
   printf("The array format is %c\n", fmt);
   printf("The correspondent HDF5 variable is %d\n", convArrayType(fmt));
}
#endif MAIN
