#include "arraytypes.h"
#include "utils.h"

/* Routine to map the atomic type to a numpy typecode
 */
size_t getArrayType(hid_t type_id, int *nptype)
{
  H5T_class_t class_id;
  size_t type_size;
  H5T_sign_t sign;

  /* Get the necessary info from the type */
  class_id = H5Tget_class( type_id );
  type_size = H5Tget_size( type_id );
  if ( (class_id == H5T_INTEGER) ) /* Only class integer can be signed */
    sign = H5Tget_sign( type_id );
  else
    sign = -1;		/* Means no sign */

  switch(class_id) {
  case H5T_BITFIELD:
    *nptype = NPY_BOOL;              /* boolean */
    break;
  case H5T_INTEGER:           /* int (bool, byte, short, long, long long) */
    switch (type_size) {
    case 1:                        /* byte */
      if ( sign )
	*nptype = NPY_INT8;                /* signed byte */
      else
	*nptype = NPY_UINT8;             /* unsigned byte */
      break;
    case 2:                        /* short */
      if ( sign )
	 *nptype = NPY_INT16;                /* signed short */
      else
	*nptype = NPY_UINT16;                /* unsigned short */
      break;
    case 4:                        /* long */
      if ( sign )
	*nptype = NPY_INT32;                /* signed long */
      else
	*nptype = NPY_UINT32;                /* unsigned long */
      break;
    case 8:                        /* long long */
      if ( sign )
	*nptype = NPY_INT64;                /* signed long long */
      else
	*nptype = NPY_UINT64;                /* unsigned long long */
      break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_INTEGER */
  case H5T_FLOAT:                   /* float (single or double) */
    switch (type_size) {
    case 4:
	*nptype = NPY_FLOAT32;                 /* float */
	break;
    case 8:
	*nptype = NPY_FLOAT64;                 /* double */
	break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_FLOAT */
  case H5T_COMPOUND:                /* might be complex (single or double) */
    if (is_complex(type_id)) {
      switch (get_complex_precision(type_id)) {
      case 32:
	*nptype = NPY_COMPLEX64;               /* float complex */
	break;
      case 64:
	*nptype = NPY_COMPLEX128;               /* double complex */
	break;
      default:
	/* This should never happen */
	goto out;
      }
    } else {
      fprintf(stderr, "this H5T_COMPOUND class is not a complex number\n");
      goto out;
    }
    break; /* case H5T_COMPOUND */
  case H5T_STRING:                  /* char or string */
      *nptype = NPY_STRING;
    break; /* case H5T_STRING */
  case H5T_TIME:                    /* time (integer or double) */
    switch (type_size) {
    case 4:
	*nptype = (int)'t';            /* integer */
	break;
    case 8:
	*nptype = (int)'T';            /* double */
	break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_TIME */
  case H5T_ENUM:                    /* enumerated type */
    *nptype = (int)('e');           /* will get type from enum description */
    break; /* case H5T_ENUM */
  default: /* Any other class type */
    /* This should never happen with Numeric arrays */
    fprintf(stderr, "class %d not supported. Sorry!\n", class_id);
    goto out;
  }

  return type_size;

 out:
  /* If we reach this line, there should be an error */
  return -1;

}

#ifdef MAIN
int main(int args, char *argv[])
{
   char *byteorder;
   int nptype;
   size_t size;

   printf("args # --> %d\n", args);
   printf("arg 0 --> %s\n", argv[0]);
   printf("arg 1 --> %s\n", argv[1]);
   printf("arg 2 --> %s\n", argv[2]);

   nptype = atoi(argv[1]);
   size = atoi(argv[2]);
   byteorder = "little";
   printf("The array format is %c\n", nptype);
   printf("The correspondent HDF5 variable is %d \n",
	  convArrayType(nptype, size, byteorder));
}
#endif /* MAIN */
