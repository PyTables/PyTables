#!/usr/bin/env python
#----------------------------------------------------------------------
# Setup script for the tables package

import sys, os, string
from os.path import exists

if not (sys.version_info[0] >= 2 and sys.version_info[1] >= 2):
    print "################################################################"
    print "You need Python 2.2 or greater to install PyTables!. Exiting..."
    print "################################################################"
    sys.exit(1)

from distutils.core     import setup, Extension
from distutils.dep_util import newer
# Check if Pyrex is installed or not
try:
    from Pyrex.Distutils import build_ext
    pyrex = 1
    cmdclass = {'build_ext': build_ext}
except:
    pyrex = 0
    cmdclass = {}

VERSION = "0.8b"

#----------------------------------------------------------------------

debug = '--debug' in sys.argv

lflags_arg = []

# Some useful functions

def check_lib(libname, maindir, dll_lib,
              dirstub, libstub,
              dirheader, headerfile):
    "Check if the libraries are completely specified for a Window system"        
    
    # Look for stub libraries
    libdir = os.path.join(maindir, dirstub)
    libfile = os.path.join(libdir, libstub)
    if os.path.isfile(libfile):
        print "Found", libname, "stub libraries at", libdir
        fdirstub = libdir
    else:
        print libname, "stub libraries *not found* at", libdir
        fdirstub = None

    # Look for headers
    headerdir = os.path.join(maindir, dirheader)
    headerfile = os.path.join(headerdir, headerfile)
    if os.path.isfile(headerfile):
        print "Found", libname, "header files at", headerdir
        fdirheader = headerdir
    else:
        print libname, "header files *not found* at", headerdir
        fdirheader = None

    # Look for DLL library in all the paths in the PATH environment variable
    # The user will have to have added the path to it manually
    for instdir in os.environ['PATH'].split(';'):
        pathdll_lib = os.path.join(instdir, dll_lib)
        if os.path.isfile(pathdll_lib):
            print "Found", dll_lib, "library at", instdir
            break
    else:
        print "Warning!:", dll_lib, "library *not found* in PATH."
        print "  Remember to install it after the compilation phase."

        
    # Return the dirs for stub libs and headers (if found)
    return (fdirstub, fdirheader)

#-----------------------------------------------------------------

if os.name == 'posix':
    # Define macros for UNIX platform
    def_macros = [('NDEBUG', 1)]
    #def_macros = [('DEBUG', 1)]  # For debugging (mainly compression filters)

    # Allow setting the HDF5 dir and additional link flags either in
    # the environment or on the command line.
    # First check the environment...
    HDF5_DIR = os.environ.get('HDF5_DIR', '')
    LZO_DIR = os.environ.get('LZO_DIR', '')
    UCL_DIR = os.environ.get('UCL_DIR', '')
    LFLAGS = os.environ.get('LFLAGS', [])
    LIBS = os.environ.get('LIBS', [])

    # ...then the command line.
    # Handle --hdf5=[PATH] --lzo=[PATH] --ucl=[PATH] --libs=[LIBS] and --lflags=[FLAGS] --debug
    args = sys.argv[:]
    for arg in args:
        if string.find(arg, '--hdf5=') == 0:
            HDF5_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--lzo=') == 0:
            LZO_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--ucl=') == 0:
            UCL_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        elif string.find(arg, '--libs=') == 0:
            LIBS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)
        elif string.find(arg, '--lflags=') == 0:
            LFLAGS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)
        elif string.find(arg, '--debug') == 0:
            # For debugging (mainly compression filters)
            def_macros = [('DEBUG', 1)]
            sys.argv.remove(arg)

    libnames = LIBS
    if LFLAGS or LIBS:
        lflags_arg = LFLAGS

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
        libnames.append('hdf5')

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
        if numarray.__version__ >= "0.6":
            print "Found numarray %s package installed" % numarray.__version__
        else:
            print "###########################################################"
            print "You need numarray 0.6 or greater!. Exiting..."
            print "###########################################################"
            sys.exit(1)


    # Look for optional compression libraries (LZO and UCL)
    # figure out from the base setting where the lib and .h are
    if LZO_DIR:
        lookup_directories = (LZO_DIR, '/usr/', '/usr/local/')
    else:
        lookup_directories = ('/usr/', '/usr/local/')
        
    for instdir in lookup_directories:
        for ext in ('.a', '.so'):
            liblzo = os.path.join(instdir, "lib/liblzo"+ext)
            if os.path.isfile(liblzo):
                LZO_DIR = instdir
                lzolibdir = os.path.join(instdir, "lib")
                print "Found LZO libraries at " + lzolibdir
                if lzolibdir not in lib_dirs:
                    lib_dirs.append(lzolibdir)
                break
            else:
                lzolibdir = None

        headerlzo = os.path.join(instdir, "include/lzo1x.h")
        if os.path.isfile(headerlzo):
            lzoincdir = os.path.join(instdir, "include")
            print "Found LZO header files at " + lzoincdir
            if lzoincdir not in inc_dirs:
                inc_dirs.append(lzoincdir)
            if lzolibdir and (not '-llzo' in LIBS):
                libnames.append('lzo')
                def_macros.append(("HAVE_LZO_LIB", 1))
            break
        else:
            lzoincdir = None

    if not lzolibdir or not lzoincdir:
        print """Optional LZO libraries or include files not found. Disabling \
support for them."""

    # Look for optional compression libraries (LZO and UCL)
    # figure out from the base setting where the lib and .h are
    if UCL_DIR:
        lookup_directories = (UCL_DIR, '/usr/', '/usr/local/')
    else:
        lookup_directories = ('/usr/', '/usr/local/')
        
    for instdir in lookup_directories:
        for ext in ('.a', '.so'):
            libucl = os.path.join(instdir, "lib/libucl"+ext)
            if os.path.isfile(libucl):
                UCL_DIR = instdir
                ucllibdir = os.path.join(instdir, "lib")
                print "Found UCL libraries at " + ucllibdir
                if ucllibdir not in lib_dirs:
                    lib_dirs.append(ucllibdir)
                break
            else:
                ucllibdir = None

        uclincdir = None
        if os.path.isfile(os.path.join(instdir, "include/ucl/ucl.h")):
            uclincdir = os.path.join(instdir, "include/ucl")
        elif os.path.isfile(os.path.join(instdir, "include/ucl.h")):
            uclincdir = os.path.join(instdir, "include")
        if uclincdir:
            print "Found UCL header files at " + uclincdir
            if uclincdir not in inc_dirs:
                inc_dirs.append(uclincdir)
            if ucllibdir and (not '-lucl' in LIBS):
                libnames.append('ucl')
                def_macros.append(("HAVE_UCL_LIB", 1))
            break

    if not ucllibdir or not uclincdir:
        print """Optional UCL libraries or include files not found. Disabling \
support for them."""

    # Set the runtime library search path
    # The use of rlib_dirs should be avoided, because debian lintian says that
    # this is not a good practice, although I should further investigate this.
    # 2003/09/30
    #rlib_dirs = lib_dirs
    rlib_dirs = []
        
    # Set the appropriate flavor hdf5Extension.c source file:
    if pyrex:
        hdf5Extension = "src/hdf5Extension.pyx"
    else:
        hdf5Extension = "src/hdf5Extension.c"
        
#-----------------------------------------------------
   
elif os.name == 'nt':
    # Define macros for Windows platform
    def_macros = [('WIN32', 1), ('NDEBUG', 1), ('_HDF5USEDLL_', 1)]

    # Init variables
    lib_dirs = []  # All the libraries has to be on the PATH
    inc_dirs = []
    # Set the runtime library search path
    rlib_dirs = []  # Windows doesn't support that, but it does not complain
                    # with an empty list
    
    # Allow setting the HDF5 dir either in
    # the environment or on the command line.
    # First check the environment...
    HDF5_DIR = os.environ.get('HDF5_DIR', '')
    LZO_DIR = os.environ.get('LZO_DIR', '')
    UCL_DIR = os.environ.get('UCL_DIR', '')
    LFLAGS = os.environ.get('LFLAGS', [])
    LIBS = os.environ.get('LIBS', [])

    # ...then the command line.
    # Handle --hdf5=[PATH] --lzo=[PATH] --ucl=[PATH]
    # and --libs=[LIBS] and --lflags=[FLAGS]
    args = sys.argv[:]
    for arg in args:
        if string.find(arg, '--hdf5=') == 0:
            HDF5_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--lzo=') == 0:
            LZO_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--ucl=') == 0:
            UCL_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        elif string.find(arg, '--libs=') == 0:
            LIBS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)
        elif string.find(arg, '--lflags=') == 0:
            LFLAGS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)

    libnames = LIBS
    if LFLAGS or LIBS:
        lflags_arg = LFLAGS

    # HDF5 library (mandatory)
    (dirstub, dirheader) = (None, None)
    if HDF5_DIR:
        (dirstub, dirheader) = check_lib("HDF5", HDF5_DIR, "hdf5dll.dll",
                                         "dll", "hdf5dll.lib",  # Stubs
                                         "include", "H5public.h") # Headers
    if dirstub and dirheader:
        lib_dirs.append(dirstub)
        inc_dirs.append(dirheader)
        libnames.append("hdf5dll")
    else:
        print "Unable to locate all the required HDF5 files"
        print """
 Please, read carefully the README and make sure
 that you have correctly specified the
 HDF5_DIR environment variable or use the flag
 --hdf5 to give a hint of where they can
 be found."""

        sys.exit(1)

    # LZO library (optional)
    if LZO_DIR:
        (dirstub, dirheader) = check_lib("LZO", LZO_DIR, "lzo.dll",
                                         "lib", "liblzo.lib",  # Stubs
                                         "include", "lzo1x.h") # Headers
        if dirstub and dirheader:
            lib_dirs.append(dirstub)
            inc_dirs.append(dirheader)
            libnames.append('liblzo')
            def_macros.append(("HAVE_LZO_LIB", 1))
        else:
            print """Optional LZO libraries or include files not found. \
Disabling support for them."""

    # UCL library (optional)
    if UCL_DIR:
        (dirstub, dirheader) = check_lib("UCL", UCL_DIR, "ucl.dll",
                                         "lib", "libucl.lib",  # Stubs
                                         "include", "ucl/ucl.h") # Headers
        if dirstub and dirheader:
            lib_dirs.append(dirstub)
            inc_dirs.append(dirheader)        
            libnames.append('libucl')
            def_macros.append(("HAVE_UCL_LIB", 1))
        else:
            print """Optional UCL libraries or include files not found. \
Disabling support for them."""
        
    # Finally, check for numarray
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
        if numarray.__version__ >= "0.6":
            print "Found numarray %s package installed" % numarray.__version__
        else:
            print "###########################################################"
            print "You need numarray 0.6 or greater!. Exiting..."
            print "###########################################################"
            sys.exit(1)

    # Set the appropriate flavor hdf5Extension.c source file:
    if pyrex:
        hdf5Extension = "src/hdf5Extension.pyx"
    else:
        hdf5Extension = "src/hdf5Extension.c"
        
# Update the version.h file if this file is newer
if pyrex:
    if newer('setup.py', 'src/version.h'):
        open('src/version.h', 'w').write('#define PYTABLES_VERSION "%s"\n' % VERSION)
else:
    if newer('setup.py', 'src/version.h'):
        open('src/version.h', 'w').write('#define PYTABLES_VERSION "%s"\n' % VERSION)

#--------------------------------------------------------------------

setup(name = 'tables',
      version = VERSION,
      description = 'A hierarchical database for Python',
      long_description = """\

PyTables is a hierarchical database package
designed to efficently manage very large amounts
of data. PyTables is built on top of the HDF5
library and the numarray package and features an
object-oriented interface that, combined with
C-code generated from Pyrex sources, makes of it a
fast, yet extremely easy to use tool for
interactively save and retrieve large amounts of
data.

""",
      
      author = 'Francesc Alted',
      author_email = 'falted@openlc.org',
      maintainer = 'Francesc Alted',
      maintainer_email = 'falted@openlc.org',
      url = 'http://pytables.sf.net/',

      packages = ['tables'],
      ext_modules = [ Extension("tables.hdf5Extension",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [hdf5Extension,
                                           "src/calcoffset.c",
                                           "src/arraytypes.c",
                                           "src/getfieldfmt.c",
                                           "src/utils.c",
                                           "src/H5Zlzo.c",
                                           "src/H5Zucl.c",
                                           "src/H5ARRAY.c",
                                           "src/H5VLARRAY.c",
                                           "src/H5LT.c",
                                           "src/H5TB.c",
                                           "src/H5TB-opt.c"],
                                library_dirs = lib_dirs,
                                libraries = libnames,
                                extra_link_args = lflags_arg,
                                #runtime_library is not supported on Windows
                                runtime_library_dirs = rlib_dirs,
                                )],
      # You may uncomment this line if pyrex installed
      cmdclass = cmdclass
)
