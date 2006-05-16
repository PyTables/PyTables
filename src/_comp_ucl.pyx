cdef extern from "stdlib.h":
  void free(void *)

cdef extern from "H5Zucl.h":
  int register_ucl(char **, char **)

def register_():
  cdef char *version, *date

  if not register_ucl(&version, &date):
    return None

  compinfo = (version, date)
  free(version)
  free(date)
  return compinfo
