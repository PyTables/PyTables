#include "arraytypes.h"

/* Get the correct HDF5 type for a format code.
 * I can't manage to do the mapping with a table because
 * the HDF5 types are not constant values and are
 * defined by executing a function. 
 * So we do that in a switch case. */

hid_t
  convArrayType(fmt, size)
    int fmt;
    size_t size;
{
   hid_t s1;
   
   switch(fmt) {
    /* I have this "a" map until a enum NumarrayType is assigned to it!
       */
    case 'a':
      /*      return H5T_NATIVE_CHAR; */
      /* An H5T_NATIVE_CHAR is interpreted as a signed byte by HDF5
       * so, we have to create a string type of lenght 1 so as to
       * represent a char.
       */
      s1 = H5Tcopy(H5T_C_S1);
      /* I use set_strpad instead of set_size as per section 3.6 
       * (Character and String Datatype Issues) of the HDF5 User's Manual,
       * altough they both seems to work well for character types */
      H5Tset_size(s1, size);
      /* H5Tset_strpad(s1, H5T_STR_NULLPAD); */
      
      return s1;
    case tBool:
      return H5T_NATIVE_HBOOL;
    case tInt8:
      return H5T_NATIVE_SCHAR;
    case tUInt8:
      return H5T_NATIVE_UCHAR;
    case tInt16:
      return H5T_NATIVE_SHORT;
    case tUInt16:
      return H5T_NATIVE_USHORT;
    case tInt32:
      return H5T_NATIVE_INT;
    case tUInt32:
      return H5T_NATIVE_UINT;
    case tInt64:
      return H5T_NATIVE_LLONG;
    case tUInt64:
      return H5T_NATIVE_ULLONG;
    case tFloat32:
      return H5T_NATIVE_FLOAT;
    case tFloat64:
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
		 int *fmt) 
{
  switch(class_id) {
  case H5T_INTEGER:                /* int (byte, short, long, long long) */
    switch (type_size) {
    case 1:                        /* byte */
      if ( sign )
	*fmt = tInt8;                /* signed byte */
      else
	*fmt = tUInt8;                /* unsigned byte */
      break;
    case 2:                        /* short */
      if ( sign )
	 *fmt =tInt16;                /* signed short */
      else
	*fmt = tUInt16;                /* unsigned short */
      break;
    case 4:                        /* long */
      if ( sign )
	*fmt = tInt32;                /* signed long */
      else
	*fmt = tUInt32;                /* unsigned long */
      break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_INTEGER */
  case H5T_FLOAT:                   /* float (single or double) */
    switch (type_size) {
    case 4:
	*fmt = tFloat32;                 /* float */
	break;
    case 8:
	*fmt = tFloat64;                 /* double */
	break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_FLOAT */
  case H5T_STRING:                  /* char or string */
    /* I map this to "a" until a enum NumarrayType is assigned to it! */
      *fmt = (int)'a';                   /* chararray */
    break; /* case H5T_STRING */
  default: /* Any other class type */
    /* This should never happen with Numeric arrays */
    fprintf(stderr, "class %d not supported. Sorry!\n", class_id);
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
