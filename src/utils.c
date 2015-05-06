#include <stdarg.h>
#include "utils.h"
#include "version.h"
#include "H5Zlzo.h"                /* Import FILTER_LZO */
#include "H5Zbzip2.h"              /* Import FILTER_BZIP2 */

#if PY_MAJOR_VERSION > 2
#define PyString_FromString PyUnicode_FromString
#endif

#ifndef NPY_COMPLEX192
typedef npy_cdouble npy_complex192;
#endif

#ifndef NPY_COMPLEX256
typedef npy_cdouble npy_complex256;
#endif

/* ---------------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>

/* This routine is meant to detect whether a dynamic library can be
   loaded on Windows. This is only way to detect its presence without
   harming the user.
*/
int getLibrary(char *libname) {
    HINSTANCE hinstLib;

    /* Load the dynamic library */
    hinstLib = LoadLibrary(TEXT(libname));

    if (hinstLib != NULL) {
      /* Free the dynamic library */
      FreeLibrary(hinstLib);
      return 0;
    }
    else {
      return -1;
    }
}

#else  /* Unix platforms */
#include <dlfcn.h>

/* Routine to detect the existance of shared libraries in UNIX. This
   has to be checked in MacOSX. However, this is not used right now in
   utilsExtension.pyx because UNIX does not complain when trying to
   load an extension library that depends on a shared library that it
   is not in the system (python raises just the ImportError). */
int getLibrary(char *libname) {
    void *hinstLib;

    /* Load the dynamic library */
    hinstLib = dlopen(libname, RTLD_LAZY);

    if (hinstLib != NULL) {
      /* Free the dynamic library */
      dlclose(hinstLib);
      return 0;
    }
    else {
      return -1;
    }
}


#endif  /* Win32 */

herr_t set_cache_size(hid_t file_id, size_t cache_size) {
#if H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 7
  /* MSVS2005 chokes on declarations after statements */
  H5AC_cache_config_t config;
#endif /* if H5_VERSION < "1.7" */
  herr_t code;

  code = 0;

#if H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 7
  config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
  code = H5Fget_mdc_config(file_id, &config);
  config.set_initial_size = TRUE;
  config.initial_size = cache_size;
/*   config.incr_mode = H5C_incr__off; */
/*   config.decr_mode = H5C_decr__off; */
/*   printf("Setting cache size to: %d\n", cache_size); */
  code = H5Fset_mdc_config(file_id, &config);
/*   printf("Return code for H5Fset_mdc_config: %d\n", code); */

#endif /* if H5_VERSION < "1.7" */

  return code;

}

PyObject *_getTablesVersion() {
  return PyString_FromString(PYTABLES_VERSION);
}

PyObject *getHDF5VersionInfo(void) {
  long binver;
  unsigned majnum, minnum, relnum;
  char     strver[16];
  PyObject *t;

/*  H5get_libversion(&majnum, &minnum, &relnum); */
  majnum = H5_VERS_MAJOR;
  minnum = H5_VERS_MINOR;
  relnum = H5_VERS_RELEASE;
  /* Get a binary number */
  binver = majnum << 16 | minnum << 8 | relnum;
  /* A string number */
  if (strcmp(H5_VERS_SUBRELEASE, "")) {
    snprintf(strver, 16, "%d.%d.%d-%s", majnum, minnum, relnum,
             H5_VERS_SUBRELEASE);
  }
  else {
    snprintf(strver, 16, "%d.%d.%d", majnum, minnum, relnum);
  }

  t = PyTuple_New(2);
  PyTuple_SetItem(t, 0, PyLong_FromLong(binver));
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
  PyObject *str;

  t = PyTuple_New(nelements);
  for (i = 0; i < nelements; i++) {
    str = PyString_FromString(buffer[i]);
    PyTuple_SetItem(t, i, str);
    /* PyTuple_SetItem does not need a decref, because it already do this */
/*     Py_DECREF(str); */
  }
  return t;
}

PyObject *createNamesList(char *buffer[], int nelements)
{
  int i;
  PyObject *t;
  PyObject *str;

  t = PyList_New(nelements);
  for (i = 0; i < nelements; i++) {
    str = PyString_FromString(buffer[i]);
    PyList_SetItem(t, i, str);
    /* PyList_SetItem does not need a decref, because it already do this */
/*     Py_DECREF(str); */
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
 * Programmer: Francesc Alted, faltet@pytables.com
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
  size_t   cd_nelmts;      /* filter client number of values */
  unsigned cd_values[20];  /* filter client data values */
  char     f_name[256];    /* filter name */
  PyObject *filters;
  PyObject *filter_values;

  /* Open the dataset. */
  if ( (dset = H5Dopen( loc_id, dset_name, H5P_DEFAULT )) < 0 ) {
    goto out;
  }

  /* Get the properties container */
  dcpl = H5Dget_create_plist(dset);
  /* Collect information about filters on chunked storage */
  if (H5D_CHUNKED==H5Pget_layout(dcpl)) {
    filters = PyDict_New();
    if ((nf = H5Pget_nfilters(dcpl))>0) {
      for (i=0; i<nf; i++) {
        cd_nelmts = 20;
        H5Pget_filter(dcpl, i, &filt_flags, &cd_nelmts,
                      cd_values, sizeof(f_name), f_name, NULL);
        filter_values = PyTuple_New(cd_nelmts);
        for (j=0;j<(long)cd_nelmts;j++) {
          PyTuple_SetItem(filter_values, j, PyLong_FromLong(cd_values[j]));
        }
        PyMapping_SetItemString (filters, f_name, filter_values);
      }
    }
  }
  else {
    /* http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52309 */
    Py_INCREF(Py_None);
    filters = Py_None;   /* Not chunked, so return None */
  }

  H5Pclose(dcpl);
  H5Dclose(dset);

  return filters;

out:
  H5Dclose(dset);
  Py_INCREF(Py_None);
  return Py_None;        /* Not chunked, so return None */
}


/****************************************************************
**
**  get_objinfo(): Get information about the type of a child.
**
****************************************************************/
int get_objinfo(hid_t loc_id, const char *name) {
  herr_t     ret;            /* Generic return value         */
  H5G_stat_t oinfo;
  /* H5O_info_t oinfo; H5Oget_info_by_name seems to have performance issues (see gh-402) */

  /* Get type of the object, without emiting an error in case the
     node does not exist. */
  H5E_BEGIN_TRY {
    ret = H5Gget_objinfo(loc_id, name, FALSE, &oinfo);
    /* H5Oget_info_by_name seems to have performance issues (see gh-402) */
    /*ret = H5Oget_info_by_name(loc_id, name, &oinfo, H5P_DEFAULT);*/
  } H5E_END_TRY;
  if (ret < 0)
    return -2;
  return oinfo.type;
}

/****************************************************************
**
**  get_linkinfo(): Get information about the type of a link.
**
****************************************************************/
int get_linkinfo(hid_t loc_id, const char *name) {
  herr_t     ret;            /* Generic return value         */
  H5L_info_t linfo;

  /* Get type of the link, without emiting an error in case the
     node does not exist. */
  H5E_BEGIN_TRY {
    ret = H5Lget_info(loc_id, name, &linfo, H5P_DEFAULT);
  } H5E_END_TRY;
  if (ret < 0)
    return -2;
  return linfo.type;
}

/****************************************************************
**
**  litercb(): Custom link iteration callback routine.
**
****************************************************************/
herr_t litercb(hid_t loc_id, const char *name, const H5L_info_t *info,
               void *data) {
  PyObject   **out_info=(PyObject **)data;
  PyObject   *strname;
  herr_t     ret;
  H5G_stat_t oinfo;
  /*H5O_info_t oinfo; H5Oget_info_by_name seems to have performance issues (see gh-402) */
  int        namedtypes = 0;

  strname = PyString_FromString(name);

  switch(info->type) {
    case H5L_TYPE_SOFT:
    case H5L_TYPE_EXTERNAL:
      PyList_Append(out_info[2], strname);
      break;
    case H5L_TYPE_ERROR:  /* XXX: check */
      PyList_Append(out_info[3], strname);
      break;
    case H5L_TYPE_HARD:
      /* Get type of the object and check it */
      ret = H5Gget_objinfo(loc_id, name, FALSE, &oinfo);
      if (ret < 0)
        return -1;

      switch(oinfo.type) {
        case H5G_GROUP:
          PyList_Append(out_info[0], strname);
          break;
        case H5G_DATASET:
          PyList_Append(out_info[1], strname);
          break;
        case H5G_TYPE:
          ++namedtypes;
          break;
        case H5G_UNKNOWN:
          PyList_Append(out_info[3], strname);
          break;
        case H5G_LINK:
          /* should not happen */
          PyList_Append(out_info[2], strname);
          break;
        default:
          /* should not happen: assume it is an external link */
          PyList_Append(out_info[2], strname);
      }

      /* H5Oget_info_by_name seems to have performance issues (see gh-402)
      ret = H5Oget_info_by_name(loc_id, name, &oinfo, H5P_DEFAULT);
      if (ret < 0)
        return -1;

      switch(oinfo.type) {
        case H5O_TYPE_GROUP:
          PyList_Append(out_info[0], strname);
          break;
        case H5O_TYPE_DATASET:
          PyList_Append(out_info[1], strname);
          break;
        case H5O_TYPE_NAMED_DATATYPE:
          ++namedtypes;
          break;
        case H5O_TYPE_UNKNOWN:
          PyList_Append(out_info[3], strname);
          break;
        default:
          / * should not happen * /
          PyList_Append(out_info[3], strname);
      }
      */
      break;
    default:
      /* should not happen */
      PyList_Append(out_info[3], strname);
  }
  Py_DECREF(strname);

  return 0 ;  /* Loop until no more objects remain in directory */
}

/****************************************************************
**
**  Giterate(): Group iteration routine.
**
****************************************************************/
PyObject *Giterate(hid_t parent_id, hid_t loc_id, const char *name) {
  hsize_t i=0;
  PyObject  *t, *tgroup, *tleave, *tlink, *tunknown;
  PyObject *info[4];

  info[0] = tgroup = PyList_New(0);
  info[1] = tleave = PyList_New(0);
  info[2] = tlink = PyList_New(0);
  info[3] = tunknown = PyList_New(0);

  /* Iterate over all the childs behind loc_id (parent_id+loc_id).
   * NOTE: using H5_INDEX_CRT_ORDER instead of H5_INDEX_NAME causes failures
   * in the test suite */
  H5Literate_by_name(parent_id, name, H5_INDEX_NAME, H5_ITER_NATIVE,
                     &i, litercb, info, H5P_DEFAULT);

  /* Create the tuple with the list of Groups and Datasets */
  t = PyTuple_New(4);
  PyTuple_SetItem(t, 0, tgroup);
  PyTuple_SetItem(t, 1, tleave);
  PyTuple_SetItem(t, 2, tlink);
  PyTuple_SetItem(t, 3, tunknown);

  return t;
}

/****************************************************************
**
**  aitercb(): Custom attribute iteration callback routine.
**
****************************************************************/
static herr_t aitercb( hid_t loc_id, const char *name,
                       const H5A_info_t *ainfo, void *op_data) {
  PyObject *strname;

  strname = PyString_FromString(name);
  /* Return the name of the attribute on op_data */
  PyList_Append(op_data, strname);
  Py_DECREF(strname);
  return(0);    /* Loop until no more attrs remain in object */
}


/****************************************************************
**
**  Aiterate(): Attribute set iteration routine.
**
****************************************************************/
PyObject *Aiterate(hid_t loc_id) {
  hsize_t i = 0;
  PyObject *attrlist;                  /* List where the attrnames are put */

  attrlist = PyList_New(0);
  H5Aiterate(loc_id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, &i,
             (H5A_operator_t)aitercb, (void *)attrlist);

  return attrlist;
}


/****************************************************************
**
**  getHDF5ClassID(): Returns class ID for loc_id.name. -1 if error.
**
****************************************************************/
H5T_class_t getHDF5ClassID(hid_t loc_id,
                           const char *name,
                           H5D_layout_t *layout,
                           hid_t *type_id,
                           hid_t *dataset_id) {
   H5T_class_t  class_id;
   hid_t        plist;

   /* Open the dataset. */
   if ( (*dataset_id = H5Dopen( loc_id, name, H5P_DEFAULT )) < 0 )
     return -1;

   /* Get an identifier for the datatype. */
   *type_id = H5Dget_type( *dataset_id );

   /* Get the class. */
   class_id = H5Tget_class( *type_id );

   /* Get the layout of the datatype */
   plist = H5Dget_create_plist(*dataset_id);
   *layout = H5Pget_layout(plist);
   H5Pclose(plist);

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
  if ( (dataset_id = H5Dopen( loc_id, dset_name, H5P_DEFAULT )) < 0 ) {
    Py_INCREF(Py_None);
    return Py_None;     /* Not chunked, so return None */
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
    PyTuple_SetItem(t, i, PyLong_FromLong((long)dims[i]));
  }

  /* Release resources */
  free(dims);

  /* Terminate access to the dataspace */
  if ( H5Sclose( space_id ) < 0 )
    goto out;

  /* Get the byteorder */
  /* Only integer, float, time and enum classes can be byteordered */
  if ((class_id == H5T_INTEGER) || (class_id == H5T_FLOAT)
      || (class_id == H5T_BITFIELD) || (class_id == H5T_TIME)
      ||  (class_id == H5T_ENUM)) {
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
    strcpy(byteorder, "irrelevant");
  }

  /* End access to the dataset */
  H5Dclose( dataset_id );

  /* Return the dimensions tuple */
  return t;

out:
 H5Tclose( type_id );
 H5Dclose( dataset_id );
 Py_INCREF(Py_None);
 return Py_None;    /* Not chunked, so return None */

}


/* The next provides functions to support a complex datatype.
   HDF5 does not provide an atomic type class for complex numbers
   so we make one from a HDF5 compound type class.

   Added by Tom Hedley <thedley@users.sourceforge.net> April 2004.
   Adapted to support Tables by F. Alted September 2004.
*/

/* Test whether the datatype is of class complex
   return 1 if it corresponds to our complex class, otherwise 0 */
/* This may be ultimately confused with nested types with 2 components
   called 'r' and 'i' and being floats, but in that case, the user
   most probably wanted to keep a complex type, so getting a complex
   instead of a nested type should not be a big issue (I hope!) :-/
   F. Alted 2005-05-23 */
int is_complex(hid_t type_id) {
  hid_t class_id, base_type_id;
  hid_t class1, class2;
  char *colname1, *colname2;
  int result = 0;
  hsize_t nfields;

  class_id = H5Tget_class(type_id);
  if (class_id == H5T_COMPOUND) {
    nfields = H5Tget_nmembers(type_id);
    if (nfields == 2) {
      colname1 = H5Tget_member_name(type_id, 0);
      colname2 = H5Tget_member_name(type_id, 1);
      if ((strcmp(colname1, "r") == 0) && (strcmp(colname2, "i") == 0)) {
        class1 = H5Tget_member_class(type_id, 0);
        class2 = H5Tget_member_class(type_id, 1);
        if (class1 == H5T_FLOAT && class2 == H5T_FLOAT)
          result = 1;
      }
      pt_H5free_memory(colname1);
      pt_H5free_memory(colname2);
    }
  }
  /* Is an Array of Complex? */
  else if (class_id == H5T_ARRAY) {
    /* Get the array base component */
    base_type_id = H5Tget_super(type_id);
    /* Call is_complex again */
    result = is_complex(base_type_id);
    H5Tclose(base_type_id);
  }
  return result;
}


/* Return the byteorder of a complex datatype.
   It is obtained from the real part, which is the first member. */
static H5T_order_t get_complex_order(hid_t type_id) {
  hid_t class_id, base_type_id;
  hid_t real_type = 0;
  H5T_order_t result = 0;

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
  if ((class_id == H5T_COMPOUND) || (class_id == H5T_ARRAY)) {
    result = H5Tget_order(real_type);
    H5Tclose(real_type);
  }
  return result;
}


/* Return the byteorder of a HDF5 data type */
/* This is actually an extension of H5Tget_order to handle complex types */
herr_t get_order(hid_t type_id, char *byteorder) {
  H5T_order_t h5byteorder;
  /*
  hid_t class_id;

  class_id = H5Tget_class(type_id);
  */

  if (is_complex(type_id)) {
    h5byteorder = get_complex_order(type_id);
  }
  else {
    h5byteorder = H5Tget_order(type_id);
  }
  if (h5byteorder == H5T_ORDER_LE) {
    strcpy(byteorder, "little");
    return h5byteorder;
  }
  else if (h5byteorder == H5T_ORDER_BE ) {
    strcpy(byteorder, "big");
    return h5byteorder;
  }
  else if (h5byteorder == H5T_ORDER_NONE ) {
    strcpy(byteorder, "irrelevant");
    return h5byteorder;
  }
  else {
    /* This should never happen! */
    fprintf(stderr, "Error: unsupported byteorder <%d>\n", h5byteorder);
    strcpy(byteorder, "unsupported");
    return -1;
  }
}


/* Set the byteorder of type_id. */
/* This only works for datatypes that are not Complex. However,
   these types should already been created with correct byteorder */
herr_t set_order(hid_t type_id, const char *byteorder) {
  herr_t status=0;

  if (! is_complex(type_id)) {
    if (strcmp(byteorder, "little") == 0)
      status = H5Tset_order(type_id, H5T_ORDER_LE);
    else if (strcmp(byteorder, "big") == 0)
      status = H5Tset_order(type_id, H5T_ORDER_BE);
    else if (strcmp(byteorder, "irrelevant") == 0) {
      /* Do nothing because 'irrelevant' doesn't require setting the
         byteorder explicitely */
/*       status = H5Tset_order(type_id, H5T_ORDER_NONE ); */
    }
    else {
      fprintf(stderr, "Error: unsupported byteorder <%s>\n", byteorder);
      status = -1;
    }
  }
  return status;
}


/* Create a HDF5 atomic datatype that represents half precision floatting
   point numbers defined by numpy as float16. */
hid_t create_ieee_float16(const char *byteorder) {
  hid_t float_id;

  if (byteorder == NULL)
    float_id = H5Tcopy(H5T_NATIVE_FLOAT);
  else if (strcmp(byteorder, "little") == 0)
    float_id = H5Tcopy(H5T_IEEE_F32LE);
  else
    float_id = H5Tcopy(H5T_IEEE_F32BE);

  if (float_id < 0)
    return float_id;

  if (H5Tset_fields(float_id, 15, 10, 5, 0, 10) < 0)
    return -1;

  if (H5Tset_size(float_id, 2) < 0)
    return -1;

  if (H5Tset_ebias(float_id, 15) < 0)
    return -1;

  return float_id;
}


/* Create a HDF5 atomic datatype that represents quad precision floatting
   point numbers. */
hid_t create_ieee_quadprecision_float(const char *byteorder) {
  hid_t float_id;

  if (byteorder == NULL)
    float_id = H5Tcopy(H5T_NATIVE_DOUBLE);
  else if (strcmp(byteorder, "little") == 0)
    float_id = H5Tcopy(H5T_IEEE_F64LE);
  else
    float_id = H5Tcopy(H5T_IEEE_F64BE);

  if (float_id < 0)
    return float_id;

  if (H5Tset_size(float_id, 16) < 0)
    return -1;

  if ((H5Tset_precision(float_id, 128)) < 0)
    return -1;

  if (H5Tset_fields(float_id , 127, 112, 15, 0, 112) < 0)
    return -1;

  if (H5Tset_ebias(float_id, 16383) < 0)
    return -1;

  return float_id;
}


/* Create a HDF5 compound datatype that represents complex numbers
   defined by numpy as complex64. */
hid_t create_ieee_complex64(const char *byteorder) {
  hid_t float_id, complex_id;

  complex_id = H5Tcreate(H5T_COMPOUND, sizeof(npy_complex64));
  if (byteorder == NULL)
    float_id = H5Tcopy(H5T_NATIVE_FLOAT);
  else if (strcmp(byteorder, "little") == 0)
    float_id = H5Tcopy(H5T_IEEE_F32LE);
  else
    float_id = H5Tcopy(H5T_IEEE_F32BE);

  if (float_id < 0)
  {
    H5Tclose(complex_id);
    return float_id;
  }

  H5Tinsert(complex_id, "r", HOFFSET(npy_complex64, real), float_id);
  H5Tinsert(complex_id, "i", HOFFSET(npy_complex64, imag), float_id);
  H5Tclose(float_id);
  return complex_id;
}


/* Counterpart for complex128 */
hid_t create_ieee_complex128(const char *byteorder) {
  hid_t float_id, complex_id;

  complex_id = H5Tcreate(H5T_COMPOUND, sizeof(npy_complex128));
  if (byteorder == NULL)
    float_id = H5Tcopy(H5T_NATIVE_DOUBLE);
  else if (strcmp(byteorder, "little") == 0)
    float_id = H5Tcopy(H5T_IEEE_F64LE);
  else
    float_id = H5Tcopy(H5T_IEEE_F64BE);

  if (float_id < 0)
  {
    H5Tclose(complex_id);
    return float_id;
  }

  H5Tinsert(complex_id, "r", HOFFSET(npy_complex128, real), float_id);
  H5Tinsert(complex_id, "i", HOFFSET(npy_complex128, imag), float_id);
  H5Tclose(float_id);
  return complex_id;
}


/* Counterpart for complex192 */
hid_t create_ieee_complex192(const char *byteorder) {
  herr_t err = 0;
  hid_t float_id, complex_id;
  H5T_order_t h5order = H5Tget_order(H5T_NATIVE_LDOUBLE);

  complex_id = H5Tcreate(H5T_COMPOUND, sizeof(npy_complex192));
  float_id = H5Tcopy(H5T_NATIVE_LDOUBLE);
  if (float_id < 0)
  {
    H5Tclose(complex_id);
    return float_id;
  }

  if ((strcmp(byteorder, "little") == 0) && (h5order != H5T_ORDER_LE))
    err = H5Tset_order(float_id, H5T_ORDER_LE);
  else if ((strcmp(byteorder, "big") == 0) && (h5order != H5T_ORDER_BE))
    err = H5Tset_order(float_id, H5T_ORDER_BE);

  if (err < 0)
  {
    H5Tclose(complex_id);
    return err;
  }

  H5Tinsert(complex_id, "r", HOFFSET(npy_complex192, real), float_id);
  H5Tinsert(complex_id, "i", HOFFSET(npy_complex192, imag), float_id);
  H5Tclose(float_id);
  return complex_id;
}


/* Counterpart for complex256 */
hid_t create_ieee_complex256(const char *byteorder) {
  herr_t err = 0;
  hid_t float_id, complex_id;
  H5T_order_t h5order = H5Tget_order(H5T_NATIVE_LDOUBLE);

  complex_id = H5Tcreate(H5T_COMPOUND, sizeof(npy_complex256));
  float_id = H5Tcopy(H5T_NATIVE_LDOUBLE);
  if (float_id < 0)
  {
    H5Tclose(complex_id);
    return float_id;
  }

  if ((strcmp(byteorder, "little") == 0) && (h5order != H5T_ORDER_LE))
    err = H5Tset_order(float_id, H5T_ORDER_LE);
  else if ((strcmp(byteorder, "big") == 0) && (h5order != H5T_ORDER_BE))
    err = H5Tset_order(float_id, H5T_ORDER_BE);

  if (err < 0)
  {
    H5Tclose(complex_id);
    return err;
  }

  H5Tinsert(complex_id, "r", HOFFSET(npy_complex256, real), float_id);
  H5Tinsert(complex_id, "i", HOFFSET(npy_complex256, imag), float_id);
  H5Tclose(float_id);
  return complex_id;
}


/* Return the number of significant bits in the real and imaginary parts */
/* This is actually an extension of H5Tget_precision to handle complex types */
size_t get_complex_precision(hid_t type_id) {
  hid_t real_type;
  size_t result;
  real_type = H5Tget_member_type(type_id, 0);
  result = H5Tget_precision(real_type);
  H5Tclose(real_type);
  return result;
}

/* End of complex additions */


/* The get_len_of_range has been taken from Python interpreter */

/* Return number of items in range/xrange (lo, hi, step).  step > 0
 * required.  Return a value < 0 if & only if the true value is too
 * large to fit in a signed long.
 */
hsize_t get_len_of_range(hsize_t lo, hsize_t hi, hsize_t step)
{
  /* -------------------------------------------------------------
     If lo >= hi, the range is empty.
     Else if n values are in the range, the last one is
     lo + (n-1)*step, which must be <= hi-1.  Rearranging,
     n <= (hi - lo - 1)/step + 1, so taking the floor of the RHS gives
     the proper value.  Since lo < hi in this case, hi-lo-1 >= 0, so
     the RHS is non-negative and so truncation is the same as the
     floor.  Letting M be the largest positive long, the worst case
     for the RHS numerator is hi=M, lo=-M-1, and then
     hi-lo-1 = M-(-M-1)-1 = 2*M.  Therefore unsigned long has enough
     precision to compute the RHS exactly.
     Note: We are using here 64 bit ints because PyTables can deal
     with 64-bit addresses even on 32-bit platforms.
     F. Alted 2006-09-25
     ---------------------------------------------------------------*/
  hsize_t n = 0;
  if (lo < hi) {
    hsize_t diff = hi - lo - 1;
    n = (hsize_t)(diff / step + 1);
  }
  return n;
}


/* Truncate the dataset to at most size rows  */
herr_t truncate_dset( hid_t dataset_id,
                      const int maindim,
                      const hsize_t size)
{

 hid_t    space_id;
 hsize_t  *dims = NULL;
 int      rank;

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space(dataset_id)) < 0 )
  goto out;

 /* Get the rank */
 if ( (rank = H5Sget_simple_extent_ndims(space_id)) < 0 )
   goto out;

 if (rank) {    /* multidimensional case */
   /* Book some memory for the selections */
   dims = (hsize_t *)malloc(rank*sizeof(hsize_t));

   /* Get dataset dimensionality */
   if ( H5Sget_simple_extent_dims(space_id, dims, NULL) < 0 )
     goto out;

   /* Truncate the EArray */
   dims[maindim] = size;
   if ( H5Dset_extent(dataset_id, dims) < 0 )
     goto out;

   /* Release resources */
   free(dims);
 }
 else {         /* scalar case (should never enter here) */
     printf("A scalar Array cannot be truncated!.\n");
     goto out;
 }

 /* Free resources */
 if ( H5Sclose(space_id) < 0 )
   return -1;

 return 0;

out:
 if (dims) free(dims);
 return -1;
}


/*
 * Helpers for management of HDF5 drivers
 */

/* DIRECT driver */
#ifndef H5_HAVE_DIRECT

herr_t pt_H5Pset_fapl_direct(hid_t fapl_id, size_t alignment,
                             size_t block_size, size_t cbuf_size)
{
 return -1;
}

#endif /* H5_HAVE_DIRECT */


/* WINDOWS driver */
#ifndef H5_HAVE_WINDOWS

herr_t pt_H5Pset_fapl_windows(hid_t fapl_id)
{
 return -1;
}

#endif /* H5_HAVE_WINDOWS */


#if (H5_HAVE_IMAGE_FILE != 1)
/* HDF5 version < 1.8.9 */

herr_t pt_H5Pset_file_image(hid_t fapl_id, void *buf_ptr, size_t buf_len) {
 return -1;
}

ssize_t pt_H5Fget_file_image(hid_t file_id, void *buf_ptr, size_t buf_len) {
 return -1;
}

#endif /* (H5_HAVE_IMAGE_FILE != 1) */


#if H5_VERSION_LE(1,8,12)

herr_t pt_H5free_memory(void *buf) {
 free(buf);
 return 0;
}

#endif
