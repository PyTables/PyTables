#include "getfieldfmt.h"
#include <stdio.h>

herr_t getfieldfmt( hid_t loc_id, 
		     const char *dset_name,
		     char *fmt )
{

  hid_t         dataset_id;
  hid_t         type_id;    
  hid_t         member_type_id;
  size_t        size;
  size_t        member_size;
  int           i;
  int           code_id;
  int           nfields;
  H5T_class_t   class;
  H5T_sign_t    sign;
  H5T_order_t   order;

  /* Open the dataset. */
  if ( ( dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
    goto out;
  
  /* Get the datatype */
  if ( ( type_id = H5Dget_type( dataset_id )) < 0 )
    goto out;
  
  /* Get the struct format */
  /* if ( ( code_id = get_struct_fmt( type_id, fmt )) < 0 )
     goto out; */
  
  /* Get the number of members */
  if ( ( nfields = H5Tget_nmembers( type_id )) < 0 )
    goto out;

  /* Get the type size */
  if ( ( size = H5Tget_size( type_id )) < 0 )
    goto out;

  /* Start always the format string with '=' to mean that the data always
     is returned in standard size and alignment */
  strcpy(fmt, "=");
  order = H5T_ORDER_NONE;  /* Initialize the byte order to NONE */
  /* Iterate tru the members */
  for ( i = 0; i < nfields; i++)
    {

      /* Get the member type */
      if ( ( member_type_id = H5Tget_member_type( type_id, i )) < 0 )
	goto out;
  
      switch (order = H5Tget_order(member_type_id)) {
      case H5T_ORDER_LE:
	fmt[0] = '<';
	break;
      case H5T_ORDER_BE:
	fmt[0] = '>';
	break;
      case H5T_ORDER_NONE:
	break; /* Do nothing */
      case H5T_ORDER_VAX:
	/* Python Struct module don't support this. HDF5 do? */
	fprintf(stderr, "Byte order %d don't supported. Sorry!\n", order);
	goto out;
      default:
	/* This should never happen */
	fprintf(stderr, "Error getting byte order.\n");
	goto out;
      }

      /* Get the member size */
      if ( ( member_size = H5Tget_size( member_type_id )) < 0 )
	goto out;

      if ( ( class = H5Tget_class(member_type_id )) < 0)
	goto out;
      /* printf("Class ID --> %d", class); */

      if ( (class == H5T_INTEGER) ) /* Only class integer can be signed */
	sign = H5Tget_sign(member_type_id);
      else
	sign = -1;


      /* Get the member format */
      if ( format_element(class, member_size, sign, i, fmt) < 0)
	 goto out; 
      
      /* Close the member type */
      if ( H5Tclose( member_type_id ) < 0 )
	goto out;

    } /* i */

  /* Release the datatype. */
  if ( H5Tclose( type_id ) < 0 )
    return -1;

  /* End access to the dataset */
  if ( H5Dclose( dataset_id ) < 0 )
    return -1;

  return 0;

 out:
  H5Dclose( dataset_id );
  return -1;
 
}


/* Routine to map the atomic type to a Python struct format 
 * This follows the standard size and alignment */
/* Falta que la rutina detecte si les dades son big-endian o little endian */
int format_element(H5T_class_t class, 
		     size_t member_size,
		     H5T_sign_t sign,
		     int position,
		     char *format) 
{
  char temp[255];
  
  switch(class) {
  case H5T_INTEGER:                /* int (byte, short, long, long long) */
    switch (member_size) {
    case 1:                        /* byte */
      if ( sign )
	strcat( format, "b" );     /* signed byte */
      else
	strcat( format, "B" );     /* unsigned byte */
      break;
    case 2:                        /* short */
      if ( sign )
	strcat( format, "h" );     /* signed short */
      else
	strcat( format, "H" );     /* unsigned short */
      break;
    case 4:                        /* long */
      if ( sign )
	strcat( format, "i" );     /* signed long */
      else
	strcat( format, "I" );     /* unsigned long */
      break;
    case 8:                        /* long long */
      if ( sign )
	strcat( format, "q" );     /* signed long long */
      else
	strcat( format, "Q" );     /* unsigned long long */
      break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_INTEGER */
  case H5T_FLOAT:                   /* float (single or double) */
    switch (member_size) {
    case 4:
	strcat( format, "f" );      /* float */
	break;
    case 8:
	strcat( format, "d" );      /* double */
	break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_FLOAT */
  case H5T_STRING:                  /* char or string */
    if ( member_size == 1 )
      strcat( format, "c" );        /* char */
    else {
      snprintf(temp, 255, "%ds", member_size);
      strcat( format, temp );       /* string */
    }
    break; /* case H5T_STRING */
  default: /* Any other class type */
    /* This should never happen for table (compound type) members */
    fprintf(stderr, "Member number %d: class %d don't supported. Sorry!\n",
	    position, class);
    goto out;
  }

  return 0;

 out:
  /* If we reach this line, there should be an error */
  return -1;
  
}
