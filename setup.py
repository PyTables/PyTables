#!/usr/bin/env python
#----------------------------------------------------------------------
# Setup script for the tables package

import sys, os, string
from os.path import exists

if not (sys.version_info[0] >= 2 and sys.version_info[1] >= 2):
    print "#################################################################"
    print "You need Python 2.2.3 or greater to install PyTables!. Exiting..."
    print "#################################################################"
    sys.exit(1)

# To deal with detected problems with python 2.2.1 and Pyrex 0.9
# Now, this should be solved with the hdf5Extension.c, but I'm not sure
# to disable this, because anybody can get into trouble if they use
# Pyrex 0.9 to generate the new hdf5Extension.c
# I definitely think it is safer to let this protection here.
# F. Altet 2004-2-2
if (sys.version_info[0] == 2 and sys.version_info[1] == 2 and
    sys.version_info[2] < 3):
    print "#################################################################"
    print "You need Python 2.2.3 or greater to install PyTables!. Exiting..."
    print "#################################################################"
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

VERSION = "0.9.1"

#----------------------------------------------------------------------

debug = '--debug' in sys.argv

# Global variables
lflags_arg = []
lib_dirs = []
inc_dirs = []

# Some useful functions
def check_lib_unix(where, libname, headername, compulsory):
    "Check if the libraries and headers are to be found on a Unix system"
    global LIBS, lib_dirs, inc_dirs, def_macros

    incdir = libdir = None
    if where:
        lookup_directories = (where,)
    else:
        lookup_directories = ('/usr/', '/usr/local/')
    for instdir in lookup_directories:
        # ".dylib" is the extension for dynamic libraries for MacOSX
        for ext in ('.a', '.so', '.dylib'):
            libfile = os.path.join(instdir, "lib/lib"+libname+ext)
            if os.path.isfile(libfile):
                libdir = os.path.dirname(libfile)
                print "Found "+libname.upper()+" libraries at " + libdir
                # If libraries are in /usr and /usr/local
                # they should be already available on search paths
                if libdir not in ('/usr/lib', '/usr/local/lib'):
                    lib_dirs.append(libdir)
                break

        headerfile = os.path.join(instdir, "include/"+headername)
        if os.path.isfile(headerfile):
            incdir = os.path.dirname(headerfile)
            print "Found "+libname.upper()+" header files at " + incdir
            # If headers are in /usr and /usr/local
            # they should be already available on search paths
            if incdir not in ('/usr/include', '/usr/local/include'):
                inc_dirs.append(incdir)
            break

    if compulsory:
        if not incdir or not libdir:
            print """\
Can't find a local %s installation.
Please, read carefully the README and if your
%s libraries are not in a standard place
set the %s_DIR environment variable or
use the flag --%s to give a hint of
where they can be found.""" % (libname, libname,
                               libname.upper(), libname)
        
            sys.exit(1)
    else:
        if not incdir or not libdir:
            print "Optional %s libraries or include files not found. Disabling support for them." % (libname,)
            return
        else:
            # Necessary to include code for optional libs
            def_macros.append(("HAVE_"+libname.upper()+"_LIB", 1))

    if (not '-l'+libname in LIBS):
        libnames.append(libname)
	
    return

def check_lib_win(libname, maindir, dll_lib,
                  dirstub, libstub,
                  dirheader, headerfile):
    "Check if the libraries and headers are to be found on a Windows system"
    
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
    # We want top get rid of zlib dependency here
    #ZLIB_DIR = os.environ.get('ZLIB_DIR', '')
    LZO_DIR = os.environ.get('LZO_DIR', '')
    UCL_DIR = os.environ.get('UCL_DIR', '')
    LFLAGS = os.environ.get('LFLAGS', [])
    if LFLAGS:
        LFLAGS = string.split(LFLAGS)
    LIBS = os.environ.get('LIBS', [])
    if LIBS:
        LIBS = string.split(LIBS)    

    # ...then the command line.
    # Handle --hdf5=[PATH] --lzo=[PATH] --ucl=[PATH] --lflags=[FLAGS] and debug
    args = sys.argv[:]
    for arg in args:
        if string.find(arg, '--hdf5=') == 0:
            HDF5_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
#         if string.find(arg, '--zlib=') == 0:
#             ZLIB_DIR = string.split(arg, '=')[1]
#             sys.argv.remove(arg)
        if string.find(arg, '--lzo=') == 0:
            LZO_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--ucl=') == 0:
            UCL_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        elif string.find(arg, '--lflags=') == 0:
            LFLAGS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)
        elif string.find(arg, '--debug') == 0:
            # For debugging (mainly compression filters)
            def_macros = [('DEBUG', 1)]
            # Don't delete this argument. It maybe useful for distutils
            # when adding more flags later on
            #sys.argv.remove(arg)

    libnames = LIBS
    if LFLAGS:
        lflags_arg = LFLAGS

    # Look for libraries. After this, inc_dirs, lib_dirs and LIBS are updated
    # Look for HDF5, compulsory
    check_lib_unix(HDF5_DIR, "hdf5", "H5public.h", compulsory=1)
    # Look for ZLIB, compulsory
    # commented out because if HDF5 is there, then so should be ZLIB
    #check_lib_unix(ZLIB_DIR, "z", "zlib.h", compulsory=1)
    # Look for LZO, not compulsory
    check_lib_unix(LZO_DIR, "lzo", "lzo1x.h", compulsory=0)
    # Look for UCL, not compulsory
    check_lib_unix(UCL_DIR, "ucl", "ucl/ucl.h", compulsory=0)
    
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

    #print "lib_dirs-->", lib_dirs
    #print "inc_dirs-->", inc_dirs
        
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
    # We want to get rid of the zlib dependency
    #ZLIB_DIR = os.environ.get('ZLIB_DIR', '')
    LZO_DIR = os.environ.get('LZO_DIR', '')
    UCL_DIR = os.environ.get('UCL_DIR', '')
    LFLAGS = os.environ.get('LFLAGS', [])
    if LFLAGS:
        LFLAGS = string.split(LFLAGS)
    LIBS = os.environ.get('LIBS', [])
    if LIBS:
        LIBS = string.split(LIBS)
        
    # ...then the command line.
    # Handle --hdf5=[PATH] --lzo=[PATH] --ucl=[PATH] --lflags=[FLAGS] and
    # --debug
    args = sys.argv[:]
    for arg in args:
        if string.find(arg, '--hdf5=') == 0:
            HDF5_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
#         if string.find(arg, '--zlib=') == 0:
#             ZLIB_DIR = string.split(arg, '=')[1]
#             sys.argv.remove(arg)
        if string.find(arg, '--lzo=') == 0:
            LZO_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        if string.find(arg, '--ucl=') == 0:
            UCL_DIR = string.split(arg, '=')[1]
            sys.argv.remove(arg)
        elif string.find(arg, '--lflags=') == 0:
            LFLAGS = string.split(string.split(arg, '=')[1])
            sys.argv.remove(arg)
        elif string.find(arg, '--debug') == 0:
            # For debugging (mainly compression filters)
            def_macros = [('DEBUG', 1)]
            # Don't delete this argument. It maybe useful for distutils
            # when adding more flags later on
            #sys.argv.remove(arg)

    libnames = LIBS
    if LFLAGS:
        lflags_arg = LFLAGS

    # HDF5 library (mandatory)
    (dirstub, dirheader) = (None, None)
    if HDF5_DIR:
        (dirstub, dirheader) = check_lib_win("HDF5", HDF5_DIR, "hdf5dll.dll",
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
 --hdf5 to give a hint of where the stubs and 
 headers can be found."""

        sys.exit(1)

#     # ZLIB library (mandatory)
#     dirstub, dirheader = None, None
#     if ZLIB_DIR:
#         (dirstub, dirheader) = check_lib_win("ZLIB", ZLIB_DIR, "zlib.dll",
#                                          #"lib", "zdll.lib",  # Stubs (1.2.1)
#                                          "lib", "zlib.lib",  # Stubs
#                                          "include", "zlib.h") # Headers
#     if dirstub and dirheader:
#         lib_dirs.append(dirstub)
#         inc_dirs.append(dirheader)
#         #libnames.append('zdll') # (1.2.1)
#         libnames.append('zlib')
#         def_macros.append(("HAVE_ZLIB_LIB", 1))
#     else:
#         print "Unable to locate all the required ZLIB files"
#         print """
#  Please, read carefully the README and make sure
#  that you have correctly specified the
#  ZLIB_DIR environment variable or use the flag
#  --zlib to give a hint of where the stubs and 
#  headers can be found."""

#         sys.exit(1)

    # LZO library (optional)
    if LZO_DIR:
        (dirstub, dirheader) = check_lib_win("LZO", LZO_DIR, "lzo.dll",
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
        (dirstub, dirheader) = check_lib_win("UCL", UCL_DIR, "ucl.dll",
                                             "lib", "libucl.lib",  # Stubs
                                             "include", "ucl/ucl.h") # Headers
        if dirstub and dirheader:
            lib_dirs.append(dirstub)
            inc_dirs.append(dirheader)        
            inc_dirs.append(dirheader+"\ucl")        
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
    if numarray.__version__ >= "1.0":
        print "Found numarray %s package installed" % numarray.__version__
    else:
        print "###########################################################"
        print "You need numarray 1.0 or greater!. Exiting..."
        print "###########################################################"
        sys.exit(1)

        
# Update the version.h file if this file is newer
if pyrex:
    hdf5Extension = "src/hdf5Extension.pyx"
else:
    hdf5Extension = "src/hdf5Extension.c"
# Set the appropriate flavor hdf5Extension.c source file:
if newer('setup.py', 'src/version.h'):
    open('src/version.h', 'w').write('#define PYTABLES_VERSION "%s"\n' % VERSION)

#--------------------------------------------------------------------

#Having the Python version included in the package name makes managing a
#system with multiple versions of Python much easier.

def find_name(base = 'tables'):
    '''If "--name-with-python-version" is on the command line then
    append "-pyX.Y" to the base name'''
    name = base
    if '--name-with-python-version' in sys.argv:
        name += '-py%i.%i'%(sys.version_info[0],sys.version_info[0])
        sys.argv.remove('--name-with-python-version')
    return name


name = find_name()

classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""
setup(name = name,
      version = VERSION,
      description = 'Hierarchical datasets for Python',
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
      classifiers = filter(None, classifiers.split("\n")),
      author = 'Francesc Altet',
      author_email = 'faltet@carabos.com',
      maintainer = 'Francesc Altet',
      maintainer_email = 'faltet@carabos.com',
      url = 'http://www.pytables.org/',
      license = 'http://www.opensource.org/licenses/bsd-license.php',
      platforms = ['any'],
      packages = ['tables', 'tables.nodes'],
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
                                           "src/H5ARRAY-opt.c",
                                           "src/H5VLARRAY.c",
                                           "src/H5LT.c",
                                           "src/H5TB.c",
                                           "src/H5TB-opt.c",
                                           "src/typeconv.c"],
                                library_dirs = lib_dirs,
                                libraries = libnames,
                                extra_link_args = lflags_arg,
                                #runtime_library is not supported on Windows
                                runtime_library_dirs = rlib_dirs,
                                )],
      cmdclass = cmdclass,
      scripts = ['utils/ptdump', 'utils/ptrepack', 'utils/nctoh5'],
)
