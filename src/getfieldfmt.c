#include "getfieldfmt.h"
#include "utils.h"  /* To access the MAXDIM value */
#include <stdio.h>

#ifdef _MSC_VER
        /*
        ** Compiling with Microsoft Visual C++; include TCHAR.H and
        ** use _snprintf instead of snprintf, because Microsoft's stdio.h
	** doesn't define snprintf.
        */
        #include <TCHAR.H>
        #define snprintf _snprintf
#endif /*_MSC_VER*/


/* Routine to map the atomic type to a Python struct format 
 * This follows the standard size and alignment */
int format_element(hid_t type_id,
		   H5T_class_t class, 
		   size_t member_size,
		   H5T_sign_t sign,
		   int position,
		   PyObject *shapes,
		   PyObject *type_sizes,
		   PyObject *types,
		   char *format) 
{
  hsize_t dims[MAXDIM];
  int ndims, i;
  size_t super_type_size;
  hid_t super_type_id; 
  H5T_class_t super_class_id;
  H5T_sign_t super_sign;
  char temp[2048], arrfmt[255] = "", *t;
  PyObject *tuple_temp;

  if (shapes){
    /* Default value for shape */
    PyList_Append(shapes, PyInt_FromLong(1));
    PyList_Append(type_sizes, PyInt_FromLong(member_size));
  }

  switch(class) {
  case H5T_BITFIELD:
    strcat( format, "b1," );     /* boolean */
    PyList_Append(types, PyString_FromString("b1"));
    break;
  case H5T_INTEGER:                /* int (byte, short, long, long long) */
    switch (member_size) {
    case 1:                        /* byte */
      if ( sign ) {
	strcat( format, "i1," );     /* signed byte */
	PyList_Append(types, PyString_FromString("i1"));
      }
      else {
	strcat( format, "u1," );     /* unsigned byte */
	PyList_Append(types, PyString_FromString("u1"));
      }
      break;
    case 2:                        /* short */
      if ( sign ) {
	strcat( format, "i2," );     /* signed short */
	PyList_Append(types, PyString_FromString("i2"));
      }
      else {
	strcat( format, "u2," );     /* unsigned short */
	PyList_Append(types, PyString_FromString("u2"));
      }
      break;
    case 4:                        /* long */
      if ( sign ) {
	strcat( format, "i4," );     /* signed long */
	PyList_Append(types, PyString_FromString("i4"));
      }
      else {
	strcat( format, "u4," );     /* unsigned long */
	PyList_Append(types, PyString_FromString("u4"));
      }
      break;
    case 8:                        /* long long */
      if ( sign ) {
	strcat( format, "i8," );     /* signed long long */
	PyList_Append(types, PyString_FromString("i8"));
      }
      else {
	strcat( format, "u8," );     /* unsigned long long */
	PyList_Append(types, PyString_FromString("u8"));
      }
      break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_INTEGER */
  case H5T_FLOAT:                   /* float (single or double) */
    switch (member_size) {
    case 4:
	strcat( format, "f4," );      /* float */
	PyList_Append(types, PyString_FromString("f4"));
	break;
    case 8:
	strcat( format, "f8," );      /* double */
	PyList_Append(types, PyString_FromString("f8"));
	break;
    default:
      /* This should never happen */
      goto out;
    }
    break; /* case H5T_FLOAT */
  case H5T_COMPOUND:                /* might be complex (single or double) */
    if (is_complex(type_id))
      switch (member_size) {
      case 8:
	strcat( format, "c8," );      /* float complex */
	PyList_Append(types, PyString_FromString("c8"));
	break;
      case 16:
	strcat( format, "c16," );      /* double complex */
	PyList_Append(types, PyString_FromString("c16"));
	break;
      default:
	/* This should never happen */
	goto out;
      }
    break; /* case H5T_COMPOUND */
  case H5T_STRING:                  /* char or string */
    snprintf(temp, 255, "a%d,", (int)member_size);  /* Always a CharArray */
    PyList_Append(types, PyString_FromString("a"));
    strcat( format, temp );       /* string */
    break; /* case H5T_STRING */
  case H5T_ARRAY:
    /* Get the array base component */
    super_type_id = H5Tget_super( type_id );
 
    /* Get the class of base component. */
    super_class_id = H5Tget_class( super_type_id );

    /* Get the sign in case the class is an integer. */
    if ( (super_class_id == H5T_INTEGER) ) /* Only integer can be signed */
      super_sign = H5Tget_sign( super_type_id );
    else 
      super_sign = -1;
   
    /* Get the size. */
    super_type_size = H5Tget_size( super_type_id );
 
    /* Get dimensions */
    if ( (ndims = H5Tget_array_ndims(type_id)) < 0 )
      goto out;
    if ( H5Tget_array_dims(type_id, dims, NULL) < 0 )
      goto out;

    /* Find the super member format */
    if ( format_element(super_type_id, super_class_id, super_type_size,
			super_sign, position, NULL, type_sizes, types, 
			arrfmt) < 0)
	 goto out; 

    /* Overwrite in the super_member type size place */
    PyList_SetItem(type_sizes, position, PyInt_FromLong(super_type_size));

    /* Return this format as well as the array size */
    t = temp;
    if (ndims > 1) {
      tuple_temp = PyTuple_New(ndims);
      sprintf(t++, "(");
      for(i=0;i<ndims;i++) {
	t += sprintf(t, "%d,", (int)dims[i]);
	PyTuple_SetItem(tuple_temp, i, PyInt_FromLong((long)dims[i]) );
      }
      t--; 			/* Delete the trailing comma */
      sprintf(t++, ")");
    }
    else {
      sprintf(temp, "%d", (int)dims[0]);
      tuple_temp = PyInt_FromLong((long)dims[0]);
    }
    /* Modify the shape for this element */
    PyList_SetItem(shapes, position, tuple_temp);

    /* Add the format to the shape */
    strcat(temp, arrfmt);
    strcat( format, temp );       /* array */

    break; /* case H5T_ARRAY */
    
  default: /* Any other class type */
    /* This should never happen for table (compound type) members */
    fprintf(stderr, "Member number %d: class %d not supported. Sorry!\n",
	    position, class);
    goto out;
  }

  return 0;

 out:
  /* If we reach this line, there should be an error */
  return -1;
  
}

herr_t getfieldfmt( hid_t loc_id, 
		    const char *dset_name,
		    char *field_names[],
		    size_t *field_sizes,
		    size_t *field_offset,
		    size_t *rowsize,
		    hsize_t *nrecords,
		    hsize_t *nfields,
		    PyObject *shapes,
		    PyObject *type_sizes,
		    PyObject *types,
		    char *fmt )
{

  hid_t         dataset_id;
  hid_t         type_id;    
  hid_t         member_type_id;
  int           i;
/*   int           has_attr; */
/*   int           n[1]; */
  size_t        itemsize;
  size_t        offset = 0;
  H5T_class_t   class;
  H5T_sign_t    sign;
  H5T_order_t   order;
  hid_t         space_id;
  hsize_t       dims[1];


  /* Open the dataset. */
  if ( ( dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
    goto out;
  
  /* Get the datatype */
  if ( ( type_id = H5Dget_type( dataset_id )) < 0 )
    goto out;
  
  /* Get the number of members */
  if ( ( *nfields = H5Tget_nmembers( type_id )) < 0 )
    goto out;

  /* Get the type size */
  if ( ( *rowsize = H5Tget_size( type_id )) < 0 )
    goto out;

  /* Get records */
  /* Get the dataspace handle */
  if ( (space_id = H5Dget_space( dataset_id )) < 0 )
    goto out;
  /* Get the number of records */
  if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
    goto out;
  /* Terminate access to the dataspace */
  if ( H5Sclose( space_id ) < 0 )
    goto out;
    
  *nrecords = dims[0];

  /* This version of getting the nrecords works, but it's slower,
     because NROWS is not widely implemented yet, and, in addition,
     perhaps reading an atribute maybe slower than calling
     H5Sget_simple_extent_dims.
     2003/09/17 */

/*   /\* Try to find the attribute "NROWS" *\/ */
/*   has_attr = H5LT_find_attribute( dataset_id, "NROWS" ); */

/*   /\* It exists, get it *\/ */
/*   if ( has_attr == 1 ) { */
/*     /\* Get the attribute *\/ */
/*     if ( H5LTget_attribute_int( loc_id, dset_name, "NROWS", n ) < 0 ) */
/*       goto out; */

/*     *nrecords = *n; */

/*   } */
/*   else { */
/*     /\* Get the dataspace handle *\/ */
/*     if ( (space_id = H5Dget_space( dataset_id )) < 0 ) */
/*       goto out; */
/*     /\* Get the number of records *\/ */
/*     if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 ) */
/*       goto out; */
/*     /\* Terminate access to the dataspace *\/ */
/*     if ( H5Sclose( space_id ) < 0 ) */
/*       goto out; */
    
/*     *nrecords = dims[0]; */
/*   } */

  /* Start always the format string with '=' to indicate that the data is
     always returned in standard size and alignment */
  strcpy(fmt, "=");

  order = H5T_ORDER_NONE;  /* Initialize the byte order to NONE */
  /* Iterate thru the members */
  for ( i = 0; i < *nfields; i++)
    {

      /* Get the member name */
      field_names[i] = H5Tget_member_name( type_id, (int)i );

      /* Get the member type */
      if ( ( member_type_id = H5Tget_member_type( type_id, i )) < 0 )
	goto out;
  
/*       switch (order = H5Tget_order(member_type_id)) { */
      switch (order = get_order(member_type_id)) {
      case H5T_ORDER_LE:
	fmt[0] = '<';
	break;
      case H5T_ORDER_BE:
	fmt[0] = '>';
	break;
      case H5T_ORDER_NONE:
	break; /* Do nothing */
      case H5T_ORDER_VAX:
	/* numarray package don't support this. HDF5 do? */
	fprintf(stderr, "Byte order %d not supported. Sorry!\n", order);
	goto out;
      default:
	/* This should never happen */
	fprintf(stderr, "Error getting byte order.\n");
	goto out;
      }

      /* Get the member size */
      if ( ( itemsize = H5Tget_size( member_type_id )) < 0 )
	goto out;
      field_sizes[i] = itemsize;

      /* The offset of this element */
      field_offset[i] = offset;
      offset += itemsize;

      if ( ( class = H5Tget_class(member_type_id )) < 0)
	goto out;

      if ( (class == H5T_INTEGER) ) /* Only class integer can be signed */
	sign = H5Tget_sign(member_type_id);
      else
	sign = -1;

      /* Get the member format */
      if ( format_element(member_type_id, class, itemsize, sign, i,
			  shapes, type_sizes, types, fmt) < 0)
	 goto out; 
      
      /* Close the member type */
      if ( H5Tclose( member_type_id ) < 0 )
	goto out;

    } /* i */

  /* Remove the trailing ',' in format if it exists */
  i = strlen(fmt);
  if (fmt[i-1] == ',')
    fmt[i-1] = '\0'; 		/* Strip out the last comma */

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

