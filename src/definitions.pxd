#  Ei!, emacs, this is -*-Python-*- mode
########################################################################
#
#       License: BSD
#       Created: June 20, 2005
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Id: definitions.pyd 1018 2005-06-20 09:43:34Z faltet $
#
########################################################################

"""Here are some definitions for sharing between extensions.

"""


# Functions from numarray API
cdef extern from "numarray/libnumarray.h":

  cdef enum:
    MAXDIM  # Maximum dimensionality for arrays

  object NA_getPythonScalar(object, long)
  int  NA_setFromPythonScalar(object, long, object)
  long NA_getBufferPtrAndSize(object, int, void**)

  # The numarray initialization funtion
  void import_libnumarray()

