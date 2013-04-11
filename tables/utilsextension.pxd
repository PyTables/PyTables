# -*- coding: utf-8 -*-

########################################################################
#
#       License: BSD
#       Created: March 03, 2008
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id: definitions.pyd 1018 2005-06-20 09:43:34Z faltet $
#
########################################################################

"""
These are declarations for functions in utilsextension.pyx that have to
be shared with other extensions.
"""

from definitions cimport hsize_t, hid_t, const_char


cdef hsize_t *malloc_dims(object)
cdef hid_t get_native_type(hid_t) nogil
cdef str cstr_to_pystr(const_char*)
