#include <stdarg.h>
#include "utils.h"
#include "version.h"

PyObject *_getTablesVersion() {
  return PyString_FromString(PYTABLES_VERSION);
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
 size_t   cd_num;         /* filter client data counter */
 unsigned cd_values[20];  /* filter client data values */
 char     f_name[256];    /* filter name */
 PyObject *filters;
 PyObject *filter_values;

 /* Open the dataset. */
 if ( (dset = H5Dopen( loc_id, dset_name )) < 0 ) {
   Py_INCREF(Py_None); 
   filters = Py_None;  	/* Not chunked, so return None */
 }

 /* Get the properties container */
 dcpl = H5Dget_create_plist(dset);

 filters = PyDict_New();
 /* Collect information about filters on chunked storage */
 if (H5D_CHUNKED==H5Pget_layout(dcpl)) {
   /*      ndims = H5Pget_chunk(dcpl, 64, chsize/\*out*\/); */
   if ((nf = H5Pget_nfilters(dcpl))>0) {
/*      filter_names = PyTuple_New(nf); */
     for (i=0; i<nf; i++) {
       cd_nelmts = 20;
       filt_id = H5Pget_filter(dcpl, i, &filt_flags, &cd_nelmts,
			       cd_values, sizeof(f_name), f_name);
       f_name[sizeof(f_name)-1] = '\0';
       filter_values = PyTuple_New(cd_nelmts);
       for (j=0;j<cd_nelmts;j++) {
	 PyTuple_SetItem(filter_values, j, PyInt_FromLong(cd_values[j]));
       }
/*        PyTuple_SetItem(filter_names, i, PyString_FromString(f_name)); */
       PyMapping_SetItemString (filters, f_name, filter_values);
     }
   }
   else {
     filters = PyDict_New(); /* Return an empty dictionary */
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
H5T_class_t getHDF5ClassID(hid_t loc_id, const char *name) {
   hid_t       dataset_id;  
   hid_t       type_id;
   H5T_class_t class_id;
     
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
   
   /* End access to the dataset */
   if ( H5Dclose( dataset_id ) )
     return -1;
   
   return class_id;
   
}
