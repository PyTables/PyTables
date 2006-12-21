#!/usr/bin/env python
#----------------------------------------------------------------------
# Setup script for the tables package

import sys, os
import textwrap
from os.path import exists

# Using ``setuptools`` enables lots of goodies, such as building eggs.
if 'FORCE_SETUPTOOLS' in os.environ:
    from setuptools import setup, find_packages
    has_setuptools = True
else:
    from distutils.core import setup
    has_setuptools = False

from distutils.core     import Extension
from distutils.dep_util import newer
from distutils.util     import convert_path

# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(
        initial_indent='   ', subsequent_indent='   ')

    print ".. %s:: %s" % (kind.upper(), head)
    for line in tw.wrap(body):
        print line

def print_error(head, body=''):
    _print_admonition('error', head, body)

def print_warning(head, body=''):
    _print_admonition('warning', head, body)

# Check for Python
if not (sys.version_info[0] >= 2 and sys.version_info[1] >= 4):
    print_error("You need Python 2.4 or greater to install PyTables!")
    sys.exit(1)

# Check for required Python packages
def check_import(pkgname, pkgver):
    try:
        mod = __import__(pkgname)
    except ImportError:
        print_error(
            "Can't find a local %s Python installation." % pkgname,
            "Please, read carefully the ``README`` file "
            "and remember that PyTables needs the %s package "
            "to compile and run." % pkgname )
        sys.exit(1)
    else:
        if mod.__version__ < pkgver:
            print_error(
                "You need %(pkgname)s %(pkgver)s or greater to run PyTables!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )
            sys.exit(1)

    print ( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': pkgname, 'pkgver': mod.__version__} )
    globals()[pkgname] = mod

check_import('numpy', '1.0')

# Check if Pyrex is installed or not
try:
    from Pyrex.Distutils import build_ext
    pyrex = 1
    cmdclass = {'build_ext': build_ext}
except:
    pyrex = 0
    cmdclass = {}

VERSION = open('VERSION').read().strip()

#----------------------------------------------------------------------

debug = '--debug' in sys.argv

# Global variables
lib_dirs = []
inc_dirs = []
optional_libs = []

default_header_dirs = None
default_library_dirs = None
default_runtime_dirs = None

if os.name == 'posix':
    default_header_dirs = ['/usr/include', '/usr/local/include']
    default_library_dirs = [
        os.path.join(_tree, _arch)
        for _tree in ('/', '/usr', '/usr/local')
        for _arch in ('lib64', 'lib') ]
    default_runtime_dirs = default_library_dirs

elif os.name == 'nt':
    default_header_dirs = []  # no default, must be given explicitly
    default_library_dirs = []  # no default, must be given explicitly
    default_runtime_dirs = [  # look for DLL files in ``%PATH%``
        _path for _path in os.environ['PATH'].split(';') ]
    # Add the \path_to_python\DLLs and tables package to the list
    default_runtime_dirs.extend(
        [os.path.join(sys.prefix, 'DLLs'),
         os.path.join(sys.prefix, 'Lib\\site-packages\\tables') ] )

from numpy.distutils.misc_util import get_numpy_include_dirs
inc_dirs.extend(get_numpy_include_dirs())

# Gcc 4.0.1 on Mac OS X 10.4 does not seem to include the default
# header and library paths.  See ticket #18.
if sys.platform.lower().startswith('darwin'):
    inc_dirs.extend(default_header_dirs)
    lib_dirs.extend(default_library_dirs)

def _find_file_path(name, locations, prefixes=[''], suffixes=['']):
    for prefix in prefixes:
        for suffix in suffixes:
            for location in locations:
                path = os.path.join(location, prefix + name + suffix)
                if os.path.isfile(path):
                    return path
    return None


class Package(object):
    def find_header_path(self, locations=default_header_dirs):
        return _find_file_path(
            self.header_name, locations, suffixes=['.h'] )

    def find_library_path(self, locations=default_library_dirs):
        return _find_file_path(
            self.library_name, locations,
            self._library_prefixes, self._library_suffixes )

    def find_runtime_path(self, locations=default_runtime_dirs):
        # An explicit path can not be provided for runtime libraries.
        # (The argument is accepted for compatibility with previous methods.)
        return _find_file_path(
            self.runtime_name, default_runtime_dirs,
            self._runtime_prefixes, self._runtime_suffixes )

    def find_directories(self, location):
        dirdata = [
            (self.header_name, self.find_header_path, default_header_dirs),
            (self.library_name, self.find_library_path, default_library_dirs),
            (self.runtime_name, self.find_runtime_path, default_runtime_dirs), ]

        locations = []
        if location:
            # The path of a custom install of the package has been
            # provided, so the directories where the components
            # (headers, libraries, runtime) are going to be searched
            # are constructed by appending platform-dependent
            # component directories to the given path.
            locations = [ os.path.join(location, compdir)
                          for compdir in self._component_dirs ]

        directories = [None, None, None]  # headers, libraries, runtime
        for idx, (name, find_path, default_dirs) in enumerate(dirdata):
            path = find_path(locations or default_dirs)
            if path:
                # Take care of not returning a directory component
                # included in the name.  For instance, if name is
                # 'foo/bar' and path is '/path/foo/bar.h', do *not*
                # take '/path/foo', but just '/path'.  This also works
                # for name 'libfoo.so' and path '/path/libfoo.so'.
                # This has been modified to just work over include files.
                # For libraries, its names can be something like 'bzip2'
                # and if they are located in places like:
                #  \stuff\bzip2-1.0.3\lib\bzip2.lib
                # then, the directory will be returned as '\stuff' (!!)
                # F. Altet 2006-02-16
                if idx == 0:
                    directories[idx] = os.path.dirname(path[:path.find(name)])
                else:
                    directories[idx] = os.path.dirname(path)

        return tuple(directories)

class PosixPackage(Package):
    _library_prefixes = ['lib']
    _library_suffixes = ['.so', '.dylib', '.a']
    _runtime_prefixes = _library_prefixes
    _runtime_suffixes = ['.so', '.dylib']

    _component_dirs = ['include', 'lib']

    def __init__(self, name, tag, header_name, library_name):
        self.name = name
        self.tag = tag
        self.header_name = header_name
        self.library_name = library_name
        self.runtime_name = library_name

class WindowsPackage(Package):
    _library_prefixes = ['']
    _library_suffixes = ['.lib']
    _runtime_prefixes = ['']
    _runtime_suffixes = ['.dll']

    # lookup in '.' seems necessary for LZO2
    _component_dirs = ['include', 'lib', 'dll', '.']

    def __init__(self, name, tag, header_name, library_name, runtime_name):
        self.name = name
        self.tag = tag
        self.header_name = header_name
        self.library_name = library_name
        self.runtime_name = runtime_name


_cp = convert_path
if os.name == 'posix':
    _Package = PosixPackage
    _platdep = {  # package tag -> platform-dependent components
        'HDF5': ['hdf5'],
        'LZO2': ['lzo2'],
        'LZO': ['lzo'],
        'BZ2': ['bz2'], }
elif os.name == 'nt':
    _Package = WindowsPackage
    _platdep = {  # package tag -> platform-dependent components
        'HDF5': ['hdf5dll', 'hdf5dll'],
        'LZO2': ['lzo2', 'lzo2'],
        'LZO': ['liblzo', 'lzo1'],
        'BZ2': ['bzip2', 'bzip2'], }
    if '--debug' in sys.argv:
        _platdep['HDF5'] = ['hdf5ddll', 'hdf5ddll']

hdf5_package = _Package("HDF5", 'HDF5', 'H5public', *_platdep['HDF5'])
lzo2_package = _Package("LZO 2", 'LZO2', _cp('lzo/lzo1x'), *_platdep['LZO2'])
lzo1_package = _Package("LZO 1", 'LZO', 'lzo1x', *_platdep['LZO'])
bzip2_package = _Package("bzip2", 'BZ2', 'bzlib', *_platdep['BZ2'])

#-----------------------------------------------------------------

def_macros = [('NDEBUG', 1)]
# Define macros for Windows platform
if os.name == 'nt':
    def_macros.append(('WIN32', 1))
    def_macros.append(('_HDF5USEDLL_', 1))

# Allow setting the HDF5 dir and additional link flags either in
# the environment or on the command line.
# First check the environment...
HDF5_DIR = os.environ.get('HDF5_DIR', '')
LZO_DIR = os.environ.get('LZO_DIR', '')
BZIP2_DIR = os.environ.get('BZIP2_DIR', '')
LFLAGS = os.environ.get('LFLAGS', '').split()
# in GCC-style compilers, -w in extra flags will get rid of copious
# 'uninitialized variable' Pyrex warnings. However, this shouldn't be the default
# as it will suppress *all* the warnings, which definitely is not a good idea.
CFLAGS = os.environ.get('CFLAGS', '').split()
LIBS = os.environ.get('LIBS', '').split()

# ...then the command line.
# Handle --hdf5=[PATH] --lzo=[PATH] --bzip2=[PATH]
# --lflags=[FLAGS] --cflags=[FLAGS] and --debug
args = sys.argv[:]
for arg in args:
    if arg.find('--hdf5=') == 0:
        HDF5_DIR = arg.split('=')[1]
        sys.argv.remove(arg)
    elif arg.find('--lzo=') == 0:
        LZO_DIR = arg.split('=')[1]
        sys.argv.remove(arg)
    elif arg.find('--bzip2=') == 0:
        BZIP2_DIR = arg.split('=')[1]
        sys.argv.remove(arg)
    elif arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--debug') == 0:
        # For debugging (mainly compression filters)
        if os.name != 'nt': # to prevent including dlfcn.h by utils.c!!!
            def_macros = [('DEBUG', 1)]
        # Don't delete this argument. It maybe useful for distutils
        # when adding more flags later on
        #sys.argv.remove(arg)

# Try to locate the compulsory and optional libraries.
lzo2_enabled = False
for (package, location) in [
    (hdf5_package, HDF5_DIR),
    (lzo2_package, LZO_DIR),
    (lzo1_package, LZO_DIR),
    (bzip2_package, BZIP2_DIR), ]:

    if package.tag == 'LZO' and lzo2_enabled:
        print ( "* Skipping detection of %s since %s has already been found."
                % (lzo1_package.name, lzo2_package.name) )
        continue  # do not use LZO 1 if LZO 2 is available

    (hdrdir, libdir, rundir) = package.find_directories(location)

    if not (hdrdir and libdir):
        if package.tag in ['HDF5']:  # these are compulsory!
            pname, ptag = package.name, package.tag
            print_error(
                "Could not find a local %s installation." % pname,
                "You may need to explicitly state "
                "where your local %(name)s headers and library can be found "
                "by setting the ``%(tag)s_DIR`` environment variable "
                "or by using the ``--%(ltag)s`` command-line option."
                % dict(name=pname, tag=ptag, ltag=ptag.lower()) )
            sys.exit(1)
        print ( "* Could not find %s headers and library; "
                "disabling support for it."  % package.name)
        continue  # look for the next library

    print ( "* Found %s headers at ``%s``, library at ``%s``."
            % (package.name, hdrdir, libdir) )

    if hdrdir not in default_header_dirs:
        inc_dirs.append(hdrdir)  # save header directory if needed
    if libdir not in default_library_dirs:
        lib_dirs.append(libdir)  # save library directory if needed

    if package.tag not in ['HDF5']:
        # Keep record of the optional libraries found.
        optional_libs.append(package.tag)
        def_macros.append(('HAVE_%s_LIB' % package.tag, 1))

    if not rundir:
        print_warning(
            "Could not find the %s runtime." % package.name,
            ( "The %(name)s shared library was *not* found "
              + { 'posix': "in the default library paths.",
                  'nt': "in any of the directories in %%PATH%%.", }[os.name]
              + " In case of runtime problems, please remember to install it." )
            % dict(name=package.name) )

    if package.tag == 'LZO2':
        lzo2_enabled = True

if lzo2_enabled:
    lzo_package = lzo2_package
else:
    lzo_package = lzo1_package

#------------------------------------------------------------------------------

# Set the appropriate flavor hdf5Extension.c source file:
if pyrex:
    hdf5Extension = "src/hdf5Extension.pyx"
    TableExtension = "src/TableExtension.pyx"
    indexesExtension = "src/indexesExtension.pyx"
    utilsExtension = "src/utilsExtension.pyx"
    lrucacheExtension = "src/lrucacheExtension.pyx"
    _comp_lzo = "src/_comp_lzo.pyx"
    _comp_bzip2 = "src/_comp_bzip2.pyx"
else:
    hdf5Extension  = "src/hdf5Extension"
    TableExtension = "src/TableExtension"
    indexesExtension = "src/indexesExtension"
    utilsExtension = "src/utilsExtension"
    lrucacheExtension = "src/lrucacheExtension"
    _comp_lzo = "src/_comp_lzo"
    _comp_bzip2 = "src/_comp_bzip2"
    for ext in [hdf5Extension, TableExtension, indexesExtension,
                utilsExtension, lrucacheExtension,
                _comp_lzo, _comp_bzip2]:
        if newer(ext+".pyx", ext+".c"):
            raise RuntimeError, "The '%s.c' file does not exist or is out of date and Pyrex is not available. Please, install Pyrex in order to properly generate the extension." % ext
    hdf5Extension += ".c"
    TableExtension += ".c"
    indexesExtension += ".c"
    utilsExtension += ".c"
    lrucacheExtension += ".c"
    _comp_lzo += ".c"
    _comp_bzip2 += ".c"

# Update the version.h file if this file is newer
if newer('VERSION', 'src/version.h'):
    open('src/version.h', 'w').write('#define PYTABLES_VERSION "%s"\n' % VERSION)

#--------------------------------------------------------------------

# Package information for ``setuptools``.
setuptools_kwargs = {}
if has_setuptools:
    # PyTables contains no data files whatsoever.
    setuptools_kwargs['zip_safe'] = True

    # ``NumPy`` headers are needed for building the extensions.
    setuptools_kwargs['install_requires'] = ['numpy>=1.0']
    # ``NumPy`` is absolutely required for running PyTables.
    setuptools_kwargs['setup_requires'] = ['numpy>=1.0']
    setuptools_kwargs['extras_require'] = {
        'Numeric': ['Numeric>=24.2'],  # for ``Numeric`` support
        'netCDF': ['ScientificPython'],  # for netCDF interchange
        'numarray': ['numarray>=1.5.2'],  # for ``numarray`` support
        }

    # Detect packages automatically.
    setuptools_kwargs['packages'] = find_packages(
        exclude=['*.tests', '*.bench'] )
    # Entry points for automatic creation of scripts.
    setuptools_kwargs['entry_points'] = {
        'console_scripts': [
            'ptdump = tables.scripts.ptdump:main',
            'ptrepack = tables.scripts.ptrepack:main',
            'nctoh5 = tables.scripts.nctoh5:main [netCDF]',
            ],
        }
    setuptools_kwargs['scripts'] = []
    # Test suites.
    setuptools_kwargs['test_suite'] = 'tables.tests.test_all.suite'
else:
    # There is no other chance, these values must be hardwired.
    setuptools_kwargs['packages'] = [
        'tables', 'tables.nodes', 'tables.scripts', 'tables.numexpr']
    setuptools_kwargs['scripts'] = [
        'utils/ptdump', 'utils/ptrepack', 'utils/nctoh5']


#Having the Python version included in the package name makes managing a
#system with multiple versions of Python much easier.

def find_name(base = 'tables-pro'):
    '''If "--name-with-python-version" is on the command line then
    append "-pyX.Y" to the base name'''
    name = base
    if '--name-with-python-version' in sys.argv:
        name += '-py%i.%i'%(sys.version_info[0],sys.version_info[1])
        sys.argv.remove('--name-with-python-version')
    return name


name = find_name()

hdf5Extension_libs = LIBS + [hdf5_package.library_name]
TableExtension_libs = LIBS + [hdf5_package.library_name]
indexesExtension_libs = LIBS + [hdf5_package.library_name]
utilsExtension_libs = LIBS + [hdf5_package.library_name]
lrucacheExtension_libs = []    # Doesn't need external libraries

# Compressor modules only need other libraries if they are enabled.
_comp_lzo_libs = LIBS[:]
_comp_bzip2_libs = LIBS[:]
for (package, complibs) in [
    (lzo_package, _comp_lzo_libs),
    (bzip2_package, _comp_bzip2_libs), ]:

    if package.tag in optional_libs:
        complibs.extend([hdf5_package.library_name, package.library_name])

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

PyTables is a package for managing hierarchical datasets and
designed to efficently cope with extremely large amounts of
data. PyTables is built on top of the HDF5 library and the
NumPy package and features an object-oriented interface
that, combined with C-code generated from Pyrex sources,
makes of it a fast, yet extremely easy to use tool for
interactively save and retrieve large amounts of data.

""",
      classifiers = filter(None, classifiers.split("\n")),
      author = 'Francesc Altet, Ivan Vilata',
      author_email = 'faltet@carabos.com, ivilata@carabos.com',
      maintainer = 'Francesc Altet, Ivan Vilata',
      maintainer_email = 'faltet@carabos.com, ivilata@carabos.com',
      url = 'http://www.pytables.org/',
      license = 'http://www.opensource.org/licenses/bsd-license.php',
      platforms = ['any'],
      ext_modules = [ Extension("tables.hdf5Extension",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [hdf5Extension,
                                           "src/arraytypes.c",
                                           "src/utils.c",
                                           "src/H5ARRAY.c",
                                           "src/H5ARRAY-opt.c",
                                           "src/H5VLARRAY.c",
                                           "src/H5ATTR.c",
                                           ],
                                library_dirs = lib_dirs,
                                libraries = hdf5Extension_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables.TableExtension",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [TableExtension,
                                           "src/H5ATTR.c",
                                           "src/H5TB-opt.c",
                                           "src/utils.c",
                                           ],
                                library_dirs = lib_dirs,
                                libraries = TableExtension_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables.indexesExtension",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [indexesExtension,
                                           "src/H5ARRAY-opt.c",
                                           "src/idx-opt.c",
                                           ],
                                library_dirs = lib_dirs,
                                libraries = indexesExtension_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables.utilsExtension",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [utilsExtension,
                                           "src/utils.c",
                                           "src/arraytypes.c",
                                           "src/typeconv.c",
                                           "src/H5ARRAY.c",
                                           "src/H5ATTR.c",
                                           ],
                                library_dirs = lib_dirs,
                                libraries = utilsExtension_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables.lrucacheExtension",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [lrucacheExtension,
                                           ],
                                library_dirs = lib_dirs,
                                libraries = lrucacheExtension_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables._comp_lzo",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [_comp_lzo,
                                           "src/H5Zlzo.c",
                                           ],
                                library_dirs = lib_dirs,
                                libraries = _comp_lzo_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables._comp_bzip2",
                                include_dirs = inc_dirs,
                                define_macros = def_macros,
                                sources = [_comp_bzip2,
                                           "src/H5Zbzip2.c",
                                           ],
                                library_dirs = lib_dirs,
                                libraries = _comp_bzip2_libs,
                                extra_link_args = LFLAGS,
                                extra_compile_args = CFLAGS,
                                ),
                       Extension("tables.numexpr.interpreter",
                                include_dirs = inc_dirs,
                                sources=["tables/numexpr/interpreter.c"],
                                depends=["tables/numexpr/interp_body.c",
                                         "tables/numexpr/complex_functions.inc"],
                                ),
                      ],
      cmdclass = cmdclass,

      **setuptools_kwargs
)
