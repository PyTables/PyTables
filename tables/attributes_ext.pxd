from .definitions cimport hid_t, hsize_t, herr_t, H5T_class_t

# Specific HDF5 functions for PyTables
cdef extern from "H5ATTR.h" nogil:
  herr_t H5ATTRget_attribute(hid_t loc_id, char *attr_name,
                             hid_t type_id, void *data)
  hsize_t H5ATTRget_attribute_string(hid_t loc_id, char *attr_name,
                                     char **attr_value, int *cset)
  hsize_t H5ATTRget_attribute_vlen_string_array(hid_t loc_id, char *attr_name,
                                                char ***attr_value, int *cset)
  herr_t H5ATTRset_attribute(hid_t obj_id, char *attr_name,
                             hid_t type_id, size_t rank,  hsize_t *dims,
                             char *attr_data)
  herr_t H5ATTRset_attribute_string(hid_t loc_id, char *attr_name,
                                    char *attr_data, hsize_t attr_size,
                                    int cset)
  herr_t H5ATTRfind_attribute(hid_t loc_id, char *attr_name)
  herr_t H5ATTRget_type_ndims(hid_t loc_id, char *attr_name,
                              hid_t *type_id, H5T_class_t *class_id,
                              size_t *type_size, int *rank)
  herr_t H5ATTRget_dims(hid_t loc_id, char *attr_name, hsize_t *dims)


