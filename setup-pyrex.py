#!/usr/bin/env python2.2
#----------------------------------------------------------------------
# Setup script for the tables package

import sys, os, string

if not (sys.version_info[0] >= 2 and sys.version_info[1] >= 2):
    print "################################################################"
    print "You need Python 2.2 or greather to install PyTables!. Exiting..."
    print "################################################################"
    sys.exit(1)

from distutils.core     import setup, Extension
from distutils.dep_util import newer
# Uncomment this if Pyrex installed and want to rebuild everything
from Pyrex.Distutils import build_ext

VERSION = "0.4.5"

#----------------------------------------------------------------------

debug = '--debug' in sys.argv or '-g' in sys.argv

lflags_arg = []

if os.name == 'posix':
    # Allow setting the HDF5 dir and additional link flags either in
    # the environment or on the command line.
    # First check the environment...
    HDF5_DIR = os.environ.get('HDF5_DIR', '')
    COMPR_DIR = os.environ.get('COMPR_DIR', '')
    LFLAGS = os.environ.get('LFLAGS', [])
    LIBS = os.environ.get('LIBS', [])

    # ...then the command line.
    # Handle --hdf5=[PATH] --comprdir=[PATH] --libs=[LIBS] and --lflags=[FLAGS]
    args = sys.argv[:]
    for arg in args:
        if string.find(arg, '--hdf5=') == 0:
            HDF5_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--comprdir=') == 0:
            COMPR_DIR = string.split(arg, '=')[1]
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
    hdf5incdir = hdf5libdir = None
    if not HDF5_DIR:
        for instdir in ('/usr/', '/usr/local/'):
            for ext in ('.a', '.so'):
                libhdf5 = os.path.join(instdir, "lib/libhdf5"+ext)
                if os.path.isfile(libhdf5):
                    HDF5_DIR = instdir
                    hdf5libdir = os.path.join(instdir, "lib")
                    print "Found HDF5 libraries at " + hdf5libdir
                    lib_dirs = [os.path.join(HDF5_DIR, 'lib')]
                    break

            headerhdf5 = os.path.join(instdir, "include/H5public.h")
            if os.path.isfile(headerhdf5):
                hdf5incdir = os.path.join(instdir, "include")
                print "Found HDF5 header files at " + hdf5incdir
                inc_dirs = [ os.path.join(HDF5_DIR, 'include')]
                break
            else:
                hdf5incdir = None


    if not HDF5_DIR and not hdf5incdir and not hdf5libdir:
        print """\
Can't find a local hdf5 installation.
Please, read carefully the README and if your
hdf5 libraries are not in a standard place
set the HDF5_DIR environment variable or
use the flag --hdf5 to give a hint of
where they can be found."""
        
        sys.exit(1)
	
    # figure out from the base setting where the lib and .h are
    if not hdf5incdir:
        inc_dirs = [ os.path.join(HDF5_DIR, 'include')]
    if not hdf5libdir:
        lib_dirs = [os.path.join(HDF5_DIR, 'lib')]
    if (not '-lhdf5' in LIBS):
        libnames = ['hdf5']
    else:
        libnames = []

    # Check for numarray
    try:
        import numarray
    except:
        print """\
Can't find a local numarray Python installation.
Please, read carefully the README and remember
that PyTables needs the numarray package to
compile and run."""
        
        sys.exit(1)
    else:
        print "Found numarray %s package installed" % numarray.__version__

    # Look for optional compression libraries (LZO and UCL)
    def_macros = [('NDEBUG', 1)]
    # figure out from the base setting where the lib and .h are
    if COMPR_DIR:
        lookup_directories = (COMPR_DIR, '/usr/', '/usr/local/')
    else:
        lookup_directories = ('/usr/', '/usr/local/')
        
    for instdir in lookup_directories:
        for ext in ('.a', '.so'):
            liblzo = os.path.join(instdir, "lib/liblzo"+ext)
            if os.path.isfile(liblzo):
                LZO_DIR = instdir
                lzolibdir = os.path.join(instdir, "lib")
                print "Found LZO libraries at " + lzolibdir
                lib_dirs.append(lzolibdir)
                break
            else:
                lzolibdir = None

        headerlzo = os.path.join(instdir, "include/lzo1x.h")
        if os.path.isfile(headerlzo):
            lzoincdir = os.path.join(instdir, "include")
            print "Found LZO header files at " + lzoincdir
            inc_dirs.append(os.path.join(instdir, "include"))
            if lzolibdir and (not '-llzo' in LIBS):
                libnames.append('lzo')
                def_macros.append(("HAVE_LZO_LIB", 1))
            break
        else:
            lzoincdir = None

    if not lzolibdir or not lzoincdir:
        print """Optional LZO libraries or include files not found. Disabling \
support for them."""

    for instdir in lookup_directories:
        for ext in ('.a', '.so'):
            libucl = os.path.join(instdir, "lib/libucl"+ext)
            if os.path.isfile(libucl):
                UCL_DIR = instdir
                ucllibdir = os.path.join(instdir, "lib")
                print "Found UCL libraries at " + ucllibdir
                lib_dirs.append(ucllibdir)
                break
            else:
                ucllibdir = None

        headerucl = os.path.join(instdir, "include/ucl/ucl.h")
        if os.path.isfile(headerucl):
            uclincdir = os.path.join(instdir, "include")
            print "Found UCL header files at " + uclincdir
            inc_dirs.append(os.path.join(instdir, "include"))
            if ucllibdir and (not '-lucl' in LIBS):
                libnames.append('ucl')
                def_macros.append(("HAVE_UCL_LIB", 1))
            break
        else:
            uclincdir = None

    if not ucllibdir or not uclincdir:
        print """Optional UCL libraries or include files not found. Disabling \
support for them."""

    # Set the runtime library search path
    rlib_dirs = lib_dirs
        
elif os.name == 'nt':

    print """I don't know how to cope with this. If you are interested
    in having such a port, and know how to do it, let me know about
    that."""
    
    sys.exit(1)
    
# Update the version .h file if this file is newer
if newer('setup-pyrex.py', 'src/version.h'):
    open('src/version.h', 'w').write('#define PYTABLES_VERSION "%s"\n' % VERSION)

setup(name = 'tables',
      version = VERSION,
      description = 'Python interface for working with scientific data tables',
      long_description = """\
At this moment, this module provides limited
support of HDF5 facilities, but I hope to add more
in the short future. By no means this package will
try to be a complete wrapper for all the HDF5
API. Instead, its goal is to allow working with
tables and Numeric objects in a hierarchical structure.
Please see the documents in the doc directory of
the source distribution or at the website for more
details on the objects and methods provided.""",
      author = 'Francesc Alted',
      #author_email = 'pytables-users@lists.sourceforge.net',
      author_email = 'falted@openlc.org',
      url = 'http://pytables.sf.net/',

      packages = ['tables'],
      ext_modules = [ Extension("tables.hdf5Extension",
				include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = ["src/hdf5Extension.pyx",
                                           "src/calcoffset.c",
                                           "src/arraytypes.c",
                                           "src/getfieldfmt.c",
                                           "src/utils.c",
                                           "src/H5Zlzo.c",
                                           "src/H5Zucl.c",
                                           "src/H5ARRAY.c",
                                           "src/H5LT.c",
                                           "src/H5TB.c",
                                           "src/H5TB-opt.c"],
				library_dirs = lib_dirs,
                                libraries = libnames,
                                runtime_library_dirs = rlib_dirs,
                                )],
      # You may uncomment this line if pyrex installed
      cmdclass = {'build_ext': build_ext}
)
