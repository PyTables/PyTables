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
