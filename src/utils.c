#include <stdarg.h>
#include "utils.h"
#include "version.h"
#include "H5Zlzo.h"  		       /* Import FILTER_LZO */
#include "H5Zucl.h"  		       /* Import FILTER_UCL */


/*-------------------------------------------------------------------------
 * 
 * Private functions
 * These are a replica of those in H5LT.c, but get_attribute_string_sys
 * needs them, so it is better to copy them here.
 * F. Alted 2004-04-20
 *
 *-------------------------------------------------------------------------
 */

herr_t _open_id( hid_t loc_id, 
		 const char *obj_name, 
		 int obj_type );

herr_t _close_id( hid_t obj_id,
		  int obj_type );

PyObject *_getTablesVersion() {
  return PyString_FromString(PYTABLES_VERSION);
}

/* PyObject *getZLIBVersionInfo(void) { */
/*   long binver; */
/*   PyObject *t; */

/* #ifdef ZLIB_VERNUM		/\* Only available for zlib >= 1.2 *\/ */
/*   binver = ZLIB_VERNUM;  	/\* This is not exactly the user's lib */
/* 				   version but that of the binary */
/* 				   packager version!  However, this */
/* 				   should be not too important *\/ */
/* #else */
/*   binver = 1;  			/\* For version of zlib < 1.2 *\/ */
/* #endif */
/*   t = PyTuple_New(2); */
/*   PyTuple_SetItem(t, 0, PyInt_FromLong(binver)); */
/*   PyTuple_SetItem(t, 1, PyString_FromString(zlibVersion())); */
/*   return t; */
/* } */

PyObject *getHDF5VersionInfo(void) {
  long binver;
  unsigned majnum, minnum, relnum;
  char     strver[16];
  PyObject *t;

  H5get_libversion(&majnum, &minnum, &relnum);
  /* Get a binary number */
  binver = majnum << 16 | minnum << 8 | relnum;
  /* A string number */
  snprintf(strver, 16, "%d.%d.%d", majnum, minnum, relnum);

  t = PyTuple_New(2);
  PyTuple_SetItem(t, 0, PyInt_FromLong(binver));
  PyTuple_SetItem(t, 1, PyString_FromString(strver));
  return t;
}

/****************************************************************
**
**  createNamesTuple(): Create Python tuple from a string of *char.
** 
****************************************************************/
PyObject *createNamesTuple(char *buffer[], int nelements)
{
  int i;
  PyObject *t;

  t = PyTuple_New(nelements);
  for (i = 0; i < nelements; i++) { 
    PyTuple_SetItem(t, i, PyString_FromString(buffer[i]) );
  }
  return t;
}

PyObject *createNamesList(char *buffer[], int nelements)
{
  int i;
  PyObject *t;

  t = PyList_New(nelements);
  for (i = 0; i < nelements; i++) { 
    PyList_SetItem(t, i, PyString_FromString(buffer[i]) );
  }
  return t;
}

/*-------------------------------------------------------------------------
 * Function: get_filter_names
 *
 * Purpose: Get the filter names for the chunks in a dataset
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@pytables.org
 *
 * Date: December 19, 2003
 *
 * Comments: 
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */

PyObject *get_filter_names( hid_t loc_id, 
			    const char *dset_name)
{
 hid_t    dset;
 hid_t    dcpl;           /* dataset creation property list */
/*  hsize_t  chsize[64];     /\* chunk size in elements *\/ */
 int      i, j;
 int      nf;             /* number of filters */
 unsigned filt_flags;     /* filter flags */
 H5Z_filter_t filt_id;       /* filter identification number */
 size_t   cd_nelmts;      /* filter client number of values */
 unsigned cd_values[20];  /* filter client data values */
 char     f_name[256];    /* filter name */
 PyObject *filters;
 PyObject *filter_values;

 /* Open the dataset. */
 if ( (dset = H5Dopen( loc_id, dset_name )) < 0 ) {
/*    Py_INCREF(Py_None);  */
/*    filters = Py_None;  	/\* Not chunked, so return None *\/ */
   goto out;
 }

 /* Get the properties container */
 dcpl = H5Dget_create_plist(dset);
 /* Collect information about filters on chunked storage */
 if (H5D_CHUNKED==H5Pget_layout(dcpl)) {
   filters = PyDict_New();
    nf = H5Pget_nfilters(dcpl);
   if ((nf = H5Pget_nfilters(dcpl))>0) {
     for (i=0; i<nf; i++) {
       cd_nelmts = 20;
       /* 1.6.2 */
       filt_id = H5Pget_filter(dcpl, i, &filt_flags, &cd_nelmts,
			       cd_values, sizeof(f_name), f_name);
       /* 1.7.x */
/*        filt_id = H5Pget_filter(dcpl, i, &filt_flags, &cd_nelmts, */
/* 			       cd_values, sizeof(f_name), f_name, NULL); */
/*        printf("f_name--> %s\n", f_name); */
	/* This code has been added because a 
	 bug in the H5Pget_filter call that
	 returns a null string when DEFLATE filter is active */
       /* The problem seems to have been solved in 1.6.2 though */
	switch (filt_id) {
	 case H5Z_FILTER_DEFLATE:
	   strcpy(f_name, "deflate");
	   break;
	 case H5Z_FILTER_SHUFFLE:
	   strcpy(f_name, "shuffle");
	   break;
	 case H5Z_FILTER_FLETCHER32:
	   strcpy(f_name, "fletcher32");
	   break;
	 case H5Z_FILTER_SZIP:
	   strcpy(f_name, "szip");
	   break;
	 case FILTER_LZO:
	   strcpy(f_name, "lzo");
	   break;
	 case FILTER_UCL:
	   strcpy(f_name, "ucl");
	   break;
	}
	
       filter_values = PyTuple_New(cd_nelmts);
       for (j=0;j<(long)cd_nelmts;j++) {
	 PyTuple_SetItem(filter_values, j, PyInt_FromLong(cd_values[j]));
       }
       PyMapping_SetItemString (filters, f_name, filter_values);
     }
   }
 }
 else {
   /* http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52309 */
   Py_INCREF(Py_None);
   filters = Py_None;  	/* Not chunked, so return None */
 }
 
 H5Pclose(dcpl);
 H5Dclose(dset);

return filters;

out:
 H5Dclose(dset);
 Py_INCREF(Py_None);
 return Py_None;  	/* Not chunked, so return None */

}

/****************************************************************
**
**  gitercb(): Custom group iteration callback routine.
** 
****************************************************************/
herr_t gitercb(hid_t loc_id, const char *name, void *data) {
  PyObject **out_info=(PyObject **)data;
  herr_t     ret;            /* Generic return value         */
  H5G_stat_t statbuf;

    /*
     * Get type of the object and check it.
     */
    ret = H5Gget_objinfo(loc_id, name, FALSE, &statbuf);
/*     CHECK(ret, FAIL, "H5Gget_objinfo"); */

    if (statbuf.type == H5G_GROUP) {
      PyList_Append(out_info[0], PyString_FromString(name));
    }
    else if (statbuf.type == H5G_DATASET) {
      PyList_Append(out_info[1], PyString_FromString(name));
    }
    
    return(0);  /* Loop until no more objects remain in directory */
}

/****************************************************************
**
**  Giterate(): Group iteration routine.
** 
****************************************************************/
PyObject *Giterate(hid_t parent_id, hid_t loc_id, const char *name) {
  int i=0, ret;
  PyObject  *t, *tdir, *tdset;
  PyObject *info[2];

  info[0] = tdir = PyList_New(0);
  info[1] = tdset = PyList_New(0);

  /* Iterate over all the childs behind loc_id (parent_id+loc_id) */
  ret = H5Giterate(parent_id, name, &i, gitercb, info);

  /* Create the tuple with the list of Groups and Datasets */
  t = PyTuple_New(2);
  PyTuple_SetItem(t, 0, tdir );
  PyTuple_SetItem(t, 1, tdset);

  return t;
}

/****************************************************************
**
**  aitercb(): Custom attribute iteration callback routine.
** 
****************************************************************/
static herr_t aitercb( hid_t loc_id, const char *name, void *op_data) {

  /* Return the name of the attribute on op_data */
  PyList_Append(op_data, PyString_FromString(name));
  return(0);    /* Loop until no more attrs remain in object */
} 


/****************************************************************
**
**  Aiterate(): Attribute set iteration routine.
** 
****************************************************************/
PyObject *Aiterate(hid_t loc_id) {
  unsigned int i = 0;
  int ret;
  PyObject *attrlist;                  /* List where the attrnames are put */

  attrlist = PyList_New(0);
  ret = H5Aiterate(loc_id, &i, (H5A_operator_t)aitercb, (void *)attrlist);

  return attrlist;
}


/****************************************************************
**
**  getHDF5ClassID(): Returns class ID for loc_id.name. -1 if error.
** 
****************************************************************/
H5T_class_t getHDF5ClassID(hid_t loc_id,
			   const char *name,
			   H5D_layout_t *layout) {
   hid_t        dataset_id;  
   hid_t        type_id;
   H5T_class_t  class_id;
   hid_t        plist;
     
   /* Open the dataset. */
   if ( (dataset_id = H5Dopen( loc_id, name )) < 0 )
     return -1;
   
   /* Get an identifier for the datatype. */
   type_id = H5Dget_type( dataset_id );
   
   /* Get the class. */
   class_id = H5Tget_class( type_id );
        
   /* Release the datatype. */
   if ( H5Tclose( type_id ) )
     return -1;

   /* Get the layout of the datatype */
   plist = H5Dget_create_plist(dataset_id);
   *layout = H5Pget_layout(plist);
   H5Pclose(plist);
   
   /* End access to the dataset */
   if ( H5Dclose( dataset_id ) )
     return -1;
   
   return class_id;
   
}

/* Helper routine that returns the rank, dims and byteorder for
   UnImplemented objects. 2004
*/

PyObject *H5UIget_info( hid_t loc_id, 
			const char *dset_name,
			char *byteorder)
{
  hid_t       dataset_id;  
  int         rank;
  hsize_t     *dims;
  hid_t       space_id; 
  H5T_class_t class_id;
  H5T_order_t order;
  hid_t       type_id; 
  PyObject    *t;
  int         i;

  /* Open the dataset. */
  if ( (dataset_id = H5Dopen( loc_id, dset_name )) < 0 ) {
    Py_INCREF(Py_None);
    return Py_None;  	/* Not chunked, so return None */
  }

  /* Get an identifier for the datatype. */
  type_id = H5Dget_type( dataset_id );

  /* Get the class. */
  class_id = H5Tget_class( type_id );

  /* Get the dataspace handle */
  if ( (space_id = H5Dget_space( dataset_id )) < 0 )
    goto out;

  /* Get rank */
  if ( (rank = H5Sget_simple_extent_ndims( space_id )) < 0 )
    goto out;

  /* Book resources for dims */
  dims = (hsize_t *)malloc(rank * sizeof(hsize_t));

  /* Get dimensions */
  if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
    goto out;

  /* Assign the dimensions to a tuple */
  t = PyTuple_New(rank);
  for(i=0;i<rank;i++) {
    /* I don't know if I should increase the reference count for dims[i]! */
    PyTuple_SetItem(t, i, PyInt_FromLong((long)dims[i]));
  }
  
  /* Release resources */
  free(dims);

  /* Terminate access to the dataspace */
  if ( H5Sclose( space_id ) < 0 )
    goto out;
 
  /* Get the byteorder */
  /* Only class integer and float can be byteordered */
  if ( (class_id == H5T_INTEGER) || (class_id == H5T_FLOAT)
       || (class_id == H5T_BITFIELD) ) {
    order = H5Tget_order( type_id );
    if (order == H5T_ORDER_LE) 
      strcpy(byteorder, "little");
    else if (order == H5T_ORDER_BE)
      strcpy(byteorder, "big");
    else {
      fprintf(stderr, "Error: unsupported byteorder: %d\n", order);
      goto out;
    }
  }
  else {
    strcpy(byteorder, "non-relevant");
  }

  /* End access to the dataset */
  H5Dclose( dataset_id );

  /* Return the dimensions tuple */
  return t;

out:
 H5Tclose( type_id );
 H5Dclose( dataset_id );
 Py_INCREF(Py_None);
 return Py_None;  	/* Not chunked, so return None */

}

/* This has been copied from the Python 2.3 sources in order to get a
   funtion similar to the method slice.indices(length) introduced in
   python 2.3, but for 2.2 */

/* F. Alted 2004-01-19 */

int GetIndicesEx(PyObject *s, int length,
		 int *start, int *stop, int *step, int *slicelength)
{
	/* this is harder to get right than you might think */

	int defstart, defstop;
	PySliceObject *r = (PySliceObject *) s;

	if (r->step == Py_None) {
		*step = 1;
	} 
	else {
		if (!_PyEval_SliceIndex(r->step, step)) return -1;
		if (*step == 0) {
			PyErr_SetString(PyExc_ValueError,
					"slice step cannot be zero");
			return -1;
		}
	}

	defstart = *step < 0 ? length-1 : 0;
	defstop = *step < 0 ? -1 : length;

	if (r->start == Py_None) {
		*start = defstart;
	}
	else {
		if (!_PyEval_SliceIndex(r->start, start)) return -1;
		if (*start < 0) *start += length;
		if (*start < 0) *start = (*step < 0) ? -1 : 0;
		if (*start >= length) 
			*start = (*step < 0) ? length - 1 : length;
	}

	if (r->stop == Py_None) {
		*stop = defstop;
	}
	else {
		if (!_PyEval_SliceIndex(r->stop, stop)) return -1;
		if (*stop < 0) *stop += length;
		if (*stop < 0) *stop = -1;
		if (*stop > length) *stop = length;
	}

	if ((*step < 0 && *stop >= *start) 
	    || (*step > 0 && *start >= *stop)) {
		*slicelength = 0;
	}
	else if (*step < 0) {
		*slicelength = (*stop-*start+1)/(*step)+1;
	}
	else {
		*slicelength = (*stop-*start-1)/(*step)+1;
	}

	return 0;
}

/*-------------------------------------------------------------------------
 * Function: get_attribute_string_sys
 *
 * Purpose: Reads a attribute specific of PyTables in a fast way
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@pytables.org
 *
 * Date: September 19, 2003
 *
 * Comments:
 *
 * Modifications:
 *
 *-------------------------------------------------------------------------
 */


PyObject *get_attribute_string_sys( hid_t loc_id,
				    const char *obj_name,
				    const char *attr_name)
{

 /* identifiers */
 hid_t      obj_id;
 hid_t      attr_id;
 hid_t      attr_type;
 size_t     attr_size;
 PyObject   *attr_value;
 char       *data;
 H5G_stat_t statbuf;

 /* Get the type of object */
 if (H5Gget_objinfo(loc_id, obj_name, 1, &statbuf)<0)
  return NULL;

 /* Open the object */
 if ((obj_id = _open_id( loc_id, obj_name, statbuf.type )) < 0)
   return NULL;

/*  Check if attribute exists */
 /* This is commented out to make the attribute reading faster */
/*  if (H5LT_find_attribute(obj_id, attr_name) <= 0)  */
 if ( ( attr_id = H5Aopen_name( obj_id, attr_name ) ) < 0 )
   /* If the attribute does not exists, return None */
   /* and do not even warn the user */
   return Py_None;

 if ( (attr_type = H5Aget_type( attr_id )) < 0 )
  goto out;

 /* Get the size. */
 attr_size = H5Tget_size( attr_type );

/*  printf("name: %s. size: %d\n", attr_name, attr_size); */
 /* Allocate memory for the input buffer */
 data = (char *)malloc(attr_size);

 if ( H5Aread( attr_id, attr_type, data ) < 0 )
  goto out;

 attr_value = PyString_FromString(data);
 free(data);

 if ( H5Tclose( attr_type )  < 0 )
  goto out;

 if ( H5Aclose( attr_id ) < 0 )
  return Py_None;

 /* Close the object */
 if ( _close_id( obj_id, statbuf.type ) < 0 )
  return Py_None;

 return attr_value;

out:
 H5Aclose( attr_id );
 H5Aclose( attr_type );
 return Py_None;

}

/*-------------------------------------------------------------------------
 * Function: _open_id
 *
 * Purpose: Private function used by get_attribute_string_sys
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 19, 2002
 *
 * Comments:
 *
 *-------------------------------------------------------------------------
 */



herr_t _open_id( hid_t loc_id, 
		 const char *obj_name, 
		 int obj_type /*basic object type*/ ) 
{

 hid_t   obj_id = -1;  
 
 switch ( obj_type )
 {
  case H5G_DATASET:
    
   /* Open the dataset. */
   if ( (obj_id = H5Dopen( loc_id, obj_name )) < 0 )
    return -1;
   break;

  case H5G_GROUP:

   /* Open the group. */
   if ( (obj_id = H5Gopen( loc_id, obj_name )) < 0 )
    return -1;
   break;

  default:
   return -1; 
 }

 return obj_id; 

}


/*-------------------------------------------------------------------------
 * Function: _close_id
 *
 * Purpose: Private function used by get_attribute_string_sys
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 *
 * Date: September 19, 2002
 *
 * Comments:
 *
 *-------------------------------------------------------------------------
 */



herr_t _close_id( hid_t obj_id,
		  int obj_type /*basic object type*/ ) 
{

 switch ( obj_type )
 {
  case H5G_DATASET:
   /* Close the dataset. */
   if ( H5Dclose( obj_id ) < 0 )
    return -1; 
   break;

  case H5G_GROUP:
  /* Close the group. */
   if ( H5Gclose( obj_id ) < 0 )
    return -1; 
   break;

  default:
   return -1; 
 }

 return 0; 

}


/* The next provides functions to support a complex datatype.
   HDF5 does not provide an atomic type class for complex numbers
   so we make one from a HDF5 compound type class.

   Added by Tom Hedley <thedley@users.sourceforge.net> April 2004.
*/

/* Return the byteorder of a complex datatype.
   It is obtained from the real part, 
   which is the first member. */
static H5T_order_t get_complex_order(hid_t type_id) {
  hid_t class_id, base_type_id;
  hid_t real_type;
  H5T_order_t result;

  class_id = H5Tget_class(type_id);
  if (class_id == H5T_COMPOUND) {
    real_type = H5Tget_member_type(type_id, 0);
  } 
  else if (class_id == H5T_ARRAY) {
    /* Get the array base component */
    base_type_id = H5Tget_super(type_id);
    /* Get the type of real component. */
    real_type = H5Tget_member_type(base_type_id, 0);
    H5Tclose(base_type_id);
  }
  result = H5Tget_order(real_type);
  H5Tclose(real_type);
  return result;
}

/* Test whether the datatype is of class complex 
   return 1 if it corresponds to our complex class, otherwise 0 */
/* It simply checks if its a H5T_COMPOUND type,
   but we could be more strict by checking names and classes
   of the members*/
int is_complex(hid_t type_id) {
  hid_t class_id, base_type_id, base_class_id;
  int result = 0;
  class_id = H5Tget_class(type_id);
  if (class_id == H5T_COMPOUND) {
    result = 1;
  }
  /* Is an Array of Complex? */
  else if (class_id == H5T_ARRAY) {
    /* Get the array base component */
    base_type_id = H5Tget_super(type_id);
    /* Get the class of base component. */
    base_class_id = H5Tget_class(base_type_id);
    if (base_class_id == H5T_COMPOUND)
      result = 1;
  }
  return result;
}

/* Return the byteorder of a HDF5 data type */
/* This is effectively an extension of H5Tget_order
   to handle complex types */
H5T_order_t get_order(hid_t type_id) {
  hid_t class_id, base_type_id;

  class_id = H5Tget_class(type_id);
/*   printf("Class ID-->%d. Iscomplex?:%d\n", class_id, is_complex(type_id)); */
  if (is_complex(type_id)) {
    return get_complex_order(type_id);
  }
  else {
    return H5Tget_order(type_id);
  }
}

/* Set the byteorder of type_id. */
/* This only works for datatypes that are not Complex. However,
   this types should already been created with correct byteorder */
herr_t set_order(hid_t type_id, const char *byteorder) {
  herr_t status=0;
  if (! is_complex(type_id)) {
    if (strcmp(byteorder, "little") == 0) 
      status = H5Tset_order(type_id, H5T_ORDER_LE);
    else if (strcmp(byteorder, "big") == 0) 
      status = H5Tset_order(type_id, H5T_ORDER_BE );
    else {
      fprintf(stderr, "Error: unsupported byteorder <%s>\n", byteorder);
      status = -1;
    }
  }
  return status;
}

/* Create a HDF5 compound datatype that represents complex numbers 
   defined by numarray as Complex64.
   We must set the byteorder before we create the type */
hid_t create_native_complex64(const char *byteorder) {
  hid_t float_id, complex_id;

  float_id = H5Tcopy(H5T_NATIVE_DOUBLE);
  complex_id = H5Tcreate (H5T_COMPOUND, sizeof(Complex64));
  set_order(float_id, byteorder);
  H5Tinsert (complex_id, "r", HOFFSET(Complex64,r),
	     float_id);
  H5Tinsert (complex_id, "i", HOFFSET(Complex64,i),
	     float_id);
  H5Tclose(float_id);
  return complex_id;
}

/* Create a HDF5 compound datatype that represents complex numbers 
   defined by numarray as Complex32.
   We must set the byteorder before we create the type */
hid_t create_native_complex32(const char *byteorder) {
  hid_t float_id, complex_id;
  float_id = H5Tcopy(H5T_NATIVE_FLOAT);
  complex_id = H5Tcreate (H5T_COMPOUND, sizeof(Complex32));
  set_order(float_id, byteorder);
  H5Tinsert (complex_id, "r", HOFFSET(Complex32,r),
	     float_id);
  H5Tinsert (complex_id, "i", HOFFSET(Complex32,i),
	     float_id);
  H5Tclose(float_id);
  return complex_id;
}

/* return the number of significant bits in the 
   real and imaginary parts */
/* This is effectively an extension of H5Tget_precision
   to handle complex types */
size_t get_complex_precision(hid_t type_id) {
  hid_t real_type;
  size_t result;
  real_type = H5Tget_member_type(type_id, 0);
  result = H5Tget_precision(real_type);
  H5Tclose(real_type);
  return result;
}

