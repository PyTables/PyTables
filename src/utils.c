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

/*
 * This routine is designed to provide equivalent functionality to 'printf'
 * and allow easy replacement for environments which don't have stdin/stdout
 * available.  (i.e. Windows & the Mac)
 */
int 
print_func(const char *format,...)
{
    va_list                 arglist;
    int                     ret_value;

    va_start(arglist, format);
    ret_value = vprintf(format, arglist);
    va_end(arglist);
    return (ret_value);
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
  int       i, j, k, totalobjects;
  PyObject  *t, *tdir, *tdset;
  iter_info info;                   /* Info of objects in the group */
  char      *namesdir[MAXELINDIR];  /* Names of dirs in the group */
  char      *namesdset[MAXELINDIR]; /* Names of dsets in the group */

  memset(&info, 0, sizeof info);

  i = 0; j = 0; k = 0;
  while (H5Giterate(loc_id, name, &i, gitercb, &info) > 0) {
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

      
