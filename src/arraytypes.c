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
   switch(fmt) {
    case 'c':
      return H5T_NATIVE_CHAR;
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

/* Modified version of H5LTget_dataset_info present on HDF_HL
 * I had to add the capability to get the sign for
 * the array type.
 * I should request to NCSA to add this feature. */

herr_t H5LTget_dataset_info_mod( hid_t loc_id, 
				 const char *dset_name,
				 hsize_t *dims,
				 H5T_class_t *class_id,
				 H5T_sign_t *sign, /* Added this parameter */
				 size_t *type_size )
{
 hid_t       dataset_id;  
 hid_t       type_id;
 hid_t       space_id; 

 /* Open the dataset. */
 if ( (dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
  return -1;

 /* Get an identifier for the datatype. */
 type_id = H5Dget_type( dataset_id );

 /* Get the class. */
    *class_id = H5Tget_class( type_id );

 /* Get the sign in case the class is an integer. */
   if ( (*class_id == H5T_INTEGER) ) /* Only class integer can be signed */
     *sign = H5Tget_sign( type_id );
   else 
     *sign = -1;
   
 /* Get the size. */
    *type_size = H5Tget_size( type_id );
   

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Get dimensions */
 if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

  /* Release the datatype. */
 if ( H5Tclose( type_id ) )
  return -1;

 /* End access to the dataset */
 if ( H5Dclose( dataset_id ) )
  return -1;

 return 0;

out:
 H5Tclose( type_id );
 H5Dclose( dataset_id );
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
