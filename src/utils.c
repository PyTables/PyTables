#include <stdarg.h>
#include "utils.h"
#include "version.h"
/* #include "zlib.h" */


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
 * Programmer: Francesc Alted, falted@openlc.org
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
   /*      ndims = H5Pget_chunk(dcpl, 64, chsize/\*out*\/); */
   filters = PyDict_New();
   if ((nf = H5Pget_nfilters(dcpl))>0) {
     for (i=0; i<nf; i++) {
       cd_nelmts = 20;
       filt_id = H5Pget_filter(dcpl, i, &filt_flags, &cd_nelmts,
			       cd_values, sizeof(f_name), f_name);
       f_name[sizeof(f_name)-1] = '\0';
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
    CHECK(ret, FAIL, "H5Gget_objinfo");

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

