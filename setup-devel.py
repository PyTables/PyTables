#!/usr/bin/env python2.2
#----------------------------------------------------------------------
# Setup script for the tables package

import sys, os, string
from distutils.core     import setup, Extension
from distutils.dep_util import newer
# Uncomment this if Pyrex installed and want to rebuild everything
from Pyrex.Distutils import build_ext

VERSION = "0.2"

#----------------------------------------------------------------------

debug = '--debug' in sys.argv or '-g' in sys.argv

lflags_arg = []


if os.name == 'posix':
    # Allow setting the HDF5 dir and additional link flags either in
    # the environment or on the command line.
    # First check the environment...
    HDF5_DIR = os.environ.get('HDF5_DIR', '')
    LFLAGS = os.environ.get('LFLAGS', [])
    LIBS = os.environ.get('LIBS', [])

    # ...then the command line.
    # Handle --hdf5=[PATH] --libs=[LIBS] and --lflags=[FLAGS]
    args = sys.argv[:]
    for arg in args:
        if string.find(arg, '--hdf5=') == 0:
            HDF5_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        elif string.find(arg, '--libs=') == 0:
            LIBS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)
        elif string.find(arg, '--lflags=') == 0:
            LFLAGS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)

    if LFLAGS or LIBS:
        lflags_arg = LFLAGS + LIBS

    # If we were not told where it is, go looking for it.
    incdir = libdir = None
    if not HDF5_DIR:
        for instdir in ('/usr/', '/usr/local/'):
            for ext in ('.a', '.so'):
                libhdf5 = os.path.join(instdir, "lib/libhdf5"+ext)
                if os.path.isfile(libhdf5):
                    HDF5_DIR = instdir
                    libdir = os.path.join(instdir, "lib")
                    print "Found HDF5 libraries at " + libdir
                    break

            headerhdf5 = os.path.join(instdir, "include/H5public.h")
            if os.path.isfile(headerhdf5):
                incdir = os.path.join(instdir, "include")
                print "Found HDF5 header files at " + incdir
                break
            else:
                incdir = None


    if not HDF5_DIR and not incdir and not libdir:
        print """\
Can't find a local hdf5 installation.
Please, read carefully the README and remember to
install the hdf5_hl library and headers in the
same place as hdf5 does."""
        
        sys.exit(1)

    # figure out from the base setting where the lib and .h are
    if not incdir: incdir = os.path.join(HDF5_DIR, 'include')
    if not libdir: libdir = os.path.join(HDF5_DIR, 'lib')
    if (not '-lhdf5' in LIBS):
        libnames = ['hdf5']
    else:
        libnames = []

    # Finally, check for Numeric
    try:
        import Numeric
    except:
        print """\
Can't find a local Numeric Python installation.
Please, read carefully the README and remember
that PyTables needs the Numeric package to
compile and run."""
        
        sys.exit(1)
    else:
        print "Found Numeric package installed"
        
elif os.name == 'nt':

    print """I don't know how to cope with this. If you are interested
    in having such a port, and know how to do it, let me know about
    that."""
    
    sys.exit(1)
    
# Update the version .h file if this file is newer
if newer('setup.py', 'src/version.h'):
    open('src/version.h', 'w').write('#define PYTABLES_VERSION "%s"\n' % VERSION)

setup(name = 'tables',
      version = VERSION,
      description = 'Python interface for working with tables in HDF5',
      long_description = """\
At this moment, this module provides limited
support of HDF5 facilities, but I hope to add more
in the short future. By no means this package will
try to be a complete wrapper for all the HDF5
API. Instead, its goal is to allow working with
tables (and hopefully in short term also with
NumArray objects) in a hierarchical structure.
Please see the documents in the doc directory of
the source distribution or at the website for more
details on the types and methods provided.""",
      author = 'Francesc Alted',
      author_email = 'pytables-users@lists.sourceforge.net',
      #author_email = 'falted@openlc.org',
      url = 'http://pytables.sf.net/',

      packages = ['tables'],
      ext_modules = [ Extension("tables.hdf5Extension",
				include_dirs = [incdir],
                                define_macros = [('DEBUG', 1)],
                                sources = ["src/hdf5Extension.pyx",
                                            "src/calcoffset.c",
                                            "src/arraytypes.c",
                                            "src/getfieldfmt.c",
                                            "src/utils.c",
					    "src/H5LT.c",
					    "src/H5TB.c"],
				library_dirs = [libdir],
                                libraries = libnames
                                )],
      # You may uncomment this line if pyrex installed
      cmdclass = {'build_ext': build_ext}
)
