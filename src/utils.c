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
    iter_info *out_info=(iter_info *)data;
    herr_t     ret;            /* Generic return value         */
    H5G_stat_t statbuf;

    strcpy(out_info->name, name);
#ifdef DEBUG
    printf("object name=%s\n",name);
#endif

    /*
     * Get type of the object and check it.
     */
    ret = H5Gget_objinfo(loc_id, name, FALSE, &statbuf);
    CHECK(ret, FAIL, "H5Gget_objinfo");

    out_info->type = statbuf.type;
#ifdef DEBUG
    printf("statbuf.type=%d\n",statbuf.type);
#endif
    
    return(1);     /* Exit after this object is visited */
    /* return(0); */  /* Loop until no more objects remain in directory */
} /* gitercb() */

/****************************************************************
**
**  Giterate(): Group iteration routine.
** 
****************************************************************/
PyObject *Giterate(hid_t parent_id, hid_t loc_id, const char *name) {
  int i, j, k, cg, ret;
  hsize_t num_obj;
  PyObject  *t, *tdir, *tdset;
  iter_info info;                   /* Info of objects in the group */
  char      *namesdir[MAX_CHILDS_IN_GROUP];  /* Names of dirs in the group */
  char      *namesdset[MAX_CHILDS_IN_GROUP]; /* Names of dsets in the group */

  memset(&info, 0, sizeof info);
  i = 0; j = 0; k = 0;
  /* Get the number of objects in loc_id */
  if ( H5Gget_num_objs(loc_id, &num_obj) < 0) {
    fprintf(stderr, "Problems getting the number of childs in group.\n");
    return NULL;
  }
#ifdef DEBUG
  printf("number of objects in group %s --> %d\n", name, (int)num_obj);
#endif
  if (num_obj > MAX_CHILDS_IN_GROUP) {
    fprintf(stderr, "Maximum number of childs in a group exceeded!.");
    fprintf(stderr, " Fetching only a maximum of: %d\n", MAX_CHILDS_IN_GROUP);
    num_obj = MAX_CHILDS_IN_GROUP;
  }
  /* Iterate over all the childs behind loc_id (parent_id+loc_id) */
  for(cg=0;cg<num_obj;cg++) {
    ret = H5Giterate(parent_id, name, &i, gitercb, &info);
#ifdef DEBUG
    printf("object -> %d, ", i);
    printf("Object type ==> %d\n", info.type);
#endif
    if (info.type == H5G_GROUP) {
      namesdir[j++] = strdup(info.name);
#ifdef DEBUG
      printf("Dir name ==> %s\n", info.name);
#endif
    }
    else if (info.type == H5G_DATASET) {
      namesdset[k++] = strdup(info.name);
#ifdef DEBUG
      printf("Dataset name ==> %s\n", info.name);
#endif
    }
  }
  
#ifdef DEBUG
  printf("Total numer of objects ==> %d\n", num_obj);
#endif
  tdir  = createNamesTuple(namesdir, j);
  tdset = createNamesTuple(namesdset, k);
  t = PyTuple_New(2);
  PyTuple_SetItem(t, 0, tdir );
  PyTuple_SetItem(t, 1, tdset);

  /* Release resources */
  for(i=0;i<j;i++) free(namesdir[i]);
  for(i=0;i<k;i++) free(namesdset[i]);
  
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
