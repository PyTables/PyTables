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

/****************************************************************
**
**  createDimsTuple(): Create Python tuple for array dimensions.
** 
****************************************************************/
PyObject *createDimsTuple(int dimensions[], int nelements)
{
  int i;
  PyObject *t;

  t = PyTuple_New(nelements);
  for (i = 0; i < nelements; i++) { 
    PyTuple_SetItem(t, i, PyInt_FromLong(dimensions[i]) );
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
#endif DEBUG

    /*
     * Get type of the object and check it.
     */
    ret = H5Gget_objinfo(loc_id, name, FALSE, &statbuf);
    CHECK(ret, FAIL, "H5Gget_objinfo");

    out_info->type = statbuf.type;
    /* printf("Object name ==> %s\n", out_info->name);
    printf("Object type ==> %d\n", out_info->type); */
#ifdef DEBUG
    printf("statbuf.type=%d\n",statbuf.type);
#endif DEBUG
    
    return(1);     /* Exit after this object is visited */
    /* return(0); */  /* Loop until no more objects remain in directory */
} /* gitercb() */

/****************************************************************
**
**  Giterate(): Group iteration routine.
** 
****************************************************************/
PyObject *Giterate(hid_t loc_id, const char *name) {
  int i, j, k, totalobjects;
  int mcexceed = 0;
  PyObject  *t, *tdir, *tdset;
  iter_info info;                   /* Info of objects in the group */
  char      *namesdir[MAX_CHILDS_IN_GROUP];  /* Names of dirs in the group */
  char      *namesdset[MAX_CHILDS_IN_GROUP]; /* Names of dsets in the group */

  memset(&info, 0, sizeof info);

  i = 0; j = 0; k = 0;
  while (H5Giterate(loc_id, name, &i, gitercb, &info) > 0) {
    /* Check if we are surpassing our buffer capacities */
    if (i <= MAX_CHILDS_IN_GROUP) {
#ifdef DEBUG
      printf("Object type ==> %d\n", info.type);
#endif DEBUG
      if (info.type == H5G_GROUP) {
	namesdir[j++] = strdup(info.name);
#ifdef DEBUG
	printf("Dir name ==> %s\n", info.name);
#endif DEBUG
      }
      else if (info.type == H5G_DATASET) {
	namesdset[k++] = strdup(info.name);
#ifdef DEBUG
	printf("Dataset name ==> %s\n", info.name);
#endif DEBUG
      }
    }
    else {
      fprintf(stderr, "Maximum number of childs exceeded!");
      mcexceed = 1;
      break;
    }
  }
  
  totalobjects = i;
#ifdef DEBUG
  printf("Total numer of objects ==> %d\n", totalobjects);
#endif DEBUG
  tdir  = createNamesTuple(namesdir, j);
  tdset = createNamesTuple(namesdset, k);
  t = PyTuple_New(2);
  PyTuple_SetItem(t, 0, tdir );
  PyTuple_SetItem(t, 1, tdset);
  
  return t;
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
