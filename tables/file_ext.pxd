from .definitions cimport hid_t

cdef class File:
    cdef hid_t   file_id
    cdef hid_t   access_plist
    cdef object  name

