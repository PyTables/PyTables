#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for the tables package"""

import os
import re
import sys
import ctypes
import tempfile
import textwrap
import subprocess
from os.path import exists, expanduser
import glob

# Using ``setuptools`` enables lots of goodies, such as building eggs.
if 'FORCE_SETUPTOOLS' in os.environ:
    from setuptools import setup, find_packages
    has_setuptools = True
else:
    from distutils.core import setup
    has_setuptools = False

from distutils.core import Extension
from distutils.dep_util import newer
from distutils.util import convert_path
from distutils.ccompiler import new_compiler
from distutils.version import LooseVersion

cmdclass = {}
setuptools_kwargs = {}

if sys.version_info >= (3,):
    fixer_names = [
        'lib2to3.fixes.fix_basestring',
        'lib2to3.fixes.fix_dict',
        'lib2to3.fixes.fix_imports',
        'lib2to3.fixes.fix_long',
        'lib2to3.fixes.fix_metaclass',
        'lib2to3.fixes.fix_next',
        'lib2to3.fixes.fix_numliterals',
        'lib2to3.fixes.fix_print',
        'lib2to3.fixes.fix_unicode',
        'lib2to3.fixes.fix_xrange',
    ]

    if has_setuptools:
        from lib2to3.refactor import get_fixers_from_package

        all_fixers = set(get_fixers_from_package('lib2to3.fixes'))
        exclude_fixers = sorted(all_fixers.difference(fixer_names))

        setuptools_kwargs['use_2to3'] = True
        setuptools_kwargs['use_2to3_fixers'] = []
        setuptools_kwargs['use_2to3_exclude_fixers'] = exclude_fixers
    else:
        from distutils.command.build_py import build_py_2to3 as build_py
        from distutils.command.build_scripts \
            import build_scripts_2to3 as build_scripts

        build_py.fixer_names = fixer_names
        build_scripts.fixer_names = fixer_names

        cmdclass['build_py'] = build_py
        cmdclass['build_scripts'] = build_scripts


# The minimum required versions
min_numpy_version = None
min_numexpr_version = None
min_cython_version = None
min_hdf5_version = None
min_python_version = (2, 6)
exec(open(os.path.join('tables', 'req_versions.py')).read())


# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(initial_indent='   ', subsequent_indent='   ')

    print(".. %s:: %s" % (kind.upper(), head))
    for line in tw.wrap(body):
        print(line)


def exit_with_error(head, body=''):
    _print_admonition('error', head, body)
    sys.exit(1)


def print_warning(head, body=''):
    _print_admonition('warning', head, body)


# Check for Python
if sys.version_info < min_python_version:
    exit_with_error("You need Python 2.6 or greater to install PyTables!")
print("* Using Python %s" % sys.version.splitlines()[0])


# Check for required Python packages
def check_import(pkgname, pkgver):
    try:
        mod = __import__(pkgname)
    except ImportError:
        exit_with_error(
            "You need %(pkgname)s %(pkgver)s or greater to run PyTables!"
            % {'pkgname': pkgname, 'pkgver': pkgver})
    else:
        if mod.__version__ < LooseVersion(pkgver):
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run PyTables!"
                % {'pkgname': pkgname, 'pkgver': pkgver})

    print("* Found %(pkgname)s %(pkgver)s package installed."
          % {'pkgname': pkgname, 'pkgver': mod.__version__})
    globals()[pkgname] = mod

check_import('numpy', min_numpy_version)
# Check for numexpr only if not using setuptools (see #298)
if not has_setuptools:
    check_import('numexpr', min_numexpr_version)

# Check if Cython is installed or not (requisite)
try:
    from Cython import __version__ as cython_version
    from Cython.Distutils import build_ext
    cmdclass['build_ext'] = build_ext
except ImportError:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to compile PyTables!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version})

if LooseVersion(cython_version) < min_cython_version:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to run PyTables!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version})
else:
    print("* Found %(pkgname)s %(pkgver)s package installed."
          % {'pkgname': 'Cython', 'pkgver': cython_version})

VERSION = open('VERSION').read().strip()

#----------------------------------------------------------------------

debug = '--debug' in sys.argv

# Global variables
lib_dirs = []
inc_dirs = ['c-blosc/hdf5']
optional_libs = []
data_files = []    # list of data files to add to packages (mainly for DLL's)

default_header_dirs = None
default_library_dirs = None
default_runtime_dirs = None


def add_from_path(envname, dirs):
    try:
        dirs.extend(os.environ[envname].split(os.pathsep))
    except KeyError:
        pass


def add_from_flags(envname, flag_key, dirs):
    for flag in os.environ.get(envname, "").split():
        if flag.startswith(flag_key):
            dirs.append(flag[len(flag_key):])

if os.name == 'posix':
    prefixes = ('/usr/local', '/sw', '/opt', '/opt/local', '/usr', '/')

    default_header_dirs = []
    add_from_path("CPATH", default_header_dirs)
    add_from_path("C_INCLUDE_PATH", default_header_dirs)
    add_from_flags("CPPFLAGS", "-I", default_header_dirs)
    default_header_dirs.extend(
        os.path.join(_tree, 'include') for _tree in prefixes
    )

    default_library_dirs = []
    add_from_flags("LDFLAGS", "-L", default_library_dirs)
    default_library_dirs.extend(
        os.path.join(_tree, _arch) for _tree in prefixes
        for _arch in ('lib64', 'lib'))
    default_runtime_dirs = default_library_dirs

elif os.name == 'nt':
    default_header_dirs = []  # no default, must be given explicitly
    default_library_dirs = []  # no default, must be given explicitly
    default_runtime_dirs = [  # look for DLL files in ``%PATH%``
        _path for _path in os.environ['PATH'].split(';')]
    # Add the \Windows\system to the runtime list (necessary for Vista)
    default_runtime_dirs.append('\\windows\\system')
    # Add the \path_to_python\DLLs and tables package to the list
    default_runtime_dirs.extend(
        [os.path.join(sys.prefix, 'Lib\\site-packages\\tables')])

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
    def __init__(self, name, tag, header_name, library_name,
                 target_function=None):
        self.name = name
        self.tag = tag
        self.header_name = header_name
        self.library_name = library_name
        self.runtime_name = library_name
        self.target_function = target_function

    def find_header_path(self, locations=default_header_dirs):
        return _find_file_path(self.header_name, locations, suffixes=['.h'])

    def find_library_path(self, locations=default_library_dirs):
        return _find_file_path(
            self.library_name, locations,
            self._library_prefixes, self._library_suffixes)

    def find_runtime_path(self, locations=default_runtime_dirs):
        """
        returns True if the runtime can be found
        returns None otherwise
        """
        # An explicit path can not be provided for runtime libraries.
        # (The argument is accepted for compatibility with previous methods.)

        # dlopen() won't tell us where the file is, just whether
        # success occurred, so this returns True instead of a filename
        for prefix in self._runtime_prefixes:
            for suffix in self._runtime_suffixes:
                try:
                    ctypes.CDLL(prefix + self.runtime_name + suffix)
                    return True
                except OSError:
                    pass

    def find_directories(self, location):
        dirdata = [
            (self.header_name, self.find_header_path, default_header_dirs),
            (self.library_name, self.find_library_path, default_library_dirs),
            (self.runtime_name, self.find_runtime_path, default_runtime_dirs),
        ]

        locations = []
        if location:
            # The path of a custom install of the package has been
            # provided, so the directories where the components
            # (headers, libraries, runtime) are going to be searched
            # are constructed by appending platform-dependent
            # component directories to the given path.
            # Remove leading and trailing '"' chars that can mislead
            # the finding routines on Windows machines
            locations = [
                os.path.join(location.strip('"'), compdir)
                for compdir in self._component_dirs
            ]

        directories = [None, None, None]  # headers, libraries, runtime
        for idx, (name, find_path, default_dirs) in enumerate(dirdata):
            path = find_path(locations or default_dirs)
            if path:
                if path is True:
                    directories[idx] = True
                    continue

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
                # F. Alted 2006-02-16
                if idx == 0:
                    directories[idx] = os.path.dirname(path[:path.rfind(name)])
                else:
                    directories[idx] = os.path.dirname(path)

        return tuple(directories)


class PosixPackage(Package):
    _library_prefixes = ['lib']
    _library_suffixes = ['.so', '.dylib', '.a']
    _runtime_prefixes = _library_prefixes
    _runtime_suffixes = ['.so', '.dylib']
    _component_dirs = ['include', 'lib']


class WindowsPackage(Package):
    _library_prefixes = ['']
    _library_suffixes = ['.lib']
    _runtime_prefixes = ['']
    _runtime_suffixes = ['.dll']

    # lookup in '.' seems necessary for LZO2
    _component_dirs = ['include', 'lib', 'dll', '.']

    def find_runtime_path(self, locations=default_runtime_dirs):
        # An explicit path can not be provided for runtime libraries.
        # (The argument is accepted for compatibility with previous methods.)
        return _find_file_path(
            self.runtime_name, default_runtime_dirs,
            self._runtime_prefixes, self._runtime_suffixes)


# Get the HDF5 version provided the 'H5public.h' header
def get_hdf5_version(headername):
    major_version = -1
    minor_version = -1
    release_version = -1
    for line in open(headername):
        if 'H5_VERS_MAJOR' in line:
            major_version = int(re.split("\s*", line)[2])
        if 'H5_VERS_MINOR' in line:
            minor_version = int(re.split("\s*", line)[2])
        if 'H5_VERS_RELEASE' in line:
            release_version = int(re.split("\s*", line)[2])
        if (major_version != -1 and minor_version != -1 and
                release_version != -1):
            break
    if (major_version == -1 or minor_version == -1 or
            release_version == -1):
        exit_with_error("Unable to detect HDF5 library version!")
    return (major_version, minor_version, release_version)


_cp = convert_path
if os.name == 'posix':
    _Package = PosixPackage
    _platdep = {  # package tag -> platform-dependent components
        'HDF5': ['hdf5'],
        'LZO2': ['lzo2'],
        'LZO': ['lzo'],
        'BZ2': ['bz2'],
        'BLOSC': ['blosc'],
    }
elif os.name == 'nt':
    _Package = WindowsPackage
    _platdep = {  # package tag -> platform-dependent components
        'HDF5': ['hdf5', 'hdf5'],
        'LZO2': ['lzo2', 'lzo2'],
        'LZO': ['liblzo', 'lzo1'],
        'BZ2': ['bzip2', 'bzip2'],
        'BLOSC': ['blosc', 'blosc'],
    }

    # Copy the next DLL's to binaries by default.
    # Update these paths for your own system!
    dll_files = []
    if '--debug' in sys.argv:
        _platdep['HDF5'] = ['hdf5ddll', 'hdf5ddll']

hdf5_package = _Package("HDF5", 'HDF5', 'H5public', *_platdep['HDF5'])
hdf5_package.target_function = 'H5close'
lzo2_package = _Package("LZO 2", 'LZO2', _cp('lzo/lzo1x'), *_platdep['LZO2'])
lzo2_package.target_function = 'lzo_version_date'
lzo1_package = _Package("LZO 1", 'LZO', 'lzo1x', *_platdep['LZO'])
lzo1_package.target_function = 'lzo_version_date'
bzip2_package = _Package("bzip2", 'BZ2', 'bzlib', *_platdep['BZ2'])
bzip2_package.target_function = 'BZ2_bzlibVersion'
blosc_package = _Package("blosc", 'BLOSC', 'blosc', *_platdep['BLOSC'])
blosc_package.target_function = 'blosc_list_compressors'  # Blosc >= 1.3


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
BLOSC_DIR = os.environ.get('BLOSC_DIR', '')
LFLAGS = os.environ.get('LFLAGS', '').split()
# in GCC-style compilers, -w in extra flags will get rid of copious
# 'uninitialized variable' Cython warnings. However, this shouldn't be
# the default as it will suppress *all* the warnings, which definitely
# is not a good idea.
CFLAGS = os.environ.get('CFLAGS', '').split()
LIBS = os.environ.get('LIBS', '').split()

# ...then the command line.
# Handle --hdf5=[PATH] --lzo=[PATH] --bzip2=[PATH] --blosc=[PATH]
# --lflags=[FLAGS] --cflags=[FLAGS] and --debug
args = sys.argv[:]
for arg in args:
    if arg.find('--hdf5=') == 0:
        HDF5_DIR = expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    elif arg.find('--lzo=') == 0:
        LZO_DIR = expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    elif arg.find('--bzip2=') == 0:
        BZIP2_DIR = expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    elif arg.find('--blosc=') == 0:
        BLOSC_DIR = expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    elif arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--debug') == 0:
        # For debugging (mainly compression filters)
        if os.name != 'nt':  # to prevent including dlfcn.h by utils.c!!!
            def_macros = [('DEBUG', 1)]
        # Don't delete this argument. It maybe useful for distutils
        # when adding more flags later on
        #sys.argv.remove(arg)

# For windows, search for the hdf5 dll in the path and use it if found.
# This is much more convenient than having to manually set an environment
# variable to rebuild pytables
if not HDF5_DIR and os.name == 'nt':
    import ctypes.util
    libdir = ctypes.util.find_library('hdf5dll.dll')
    # Like 'C:\\Program Files\\HDF Group\\HDF5\\1.8.8\\bin\\hdf5dll.dll'
    if libdir:
        # Strip off the filename
        libdir = os.path.dirname(libdir)
        # Strip off the 'bin' directory
        HDF5_DIR = os.path.dirname(libdir)
        print("* Found HDF5 using system PATH ('%s')" % libdir)

# The next flag for the C compiler is needed for finding the C headers for
# the Cython extensions
CFLAGS.append("-Isrc")

# Force the 1.8.x HDF5 API even if the library as been compiled to use the
# 1.6.x API by default
CFLAGS.extend([
    "-DH5Acreate_vers=2",
    "-DH5Aiterate_vers=2",
    "-DH5Dcreate_vers=2",
    "-DH5Dopen_vers=2",
    "-DH5Eclear_vers=2",
    "-DH5Eprint_vers=2",
    "-DH5Epush_vers=2",
    "-DH5Eset_auto_vers=2",
    "-DH5Eget_auto_vers=2",
    "-DH5Ewalk_vers=2",
    "-DH5E_auto_t_vers=2",
    "-DH5Gcreate_vers=2",
    "-DH5Gopen_vers=2",
    "-DH5Pget_filter_vers=2",
    "-DH5Pget_filter_by_id_vers=2",
    #"-DH5Pinsert_vers=2",
    #"-DH5Pregister_vers=2",
    #"-DH5Rget_obj_type_vers=2",
    "-DH5Tarray_create_vers=2",
    #"-DH5Tcommit_vers=2",
    "-DH5Tget_array_dims_vers=2",
    #"-DH5Topen_vers=2",
    "-DH5Z_class_t_vers=2",
])
# H5Oget_info_by_name seems to have performance issues (see gh-402), so we
# need to use teh deprecated H5Gget_objinfo function
# CFLAGS.append("-DH5_NO_DEPRECATED_SYMBOLS")

# Do not use numpy deprecated API
#CFLAGS.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

# Try to locate the compulsory and optional libraries.
lzo2_enabled = False
compiler = new_compiler()
for (package, location) in [(hdf5_package, HDF5_DIR),
                            (lzo2_package, LZO_DIR),
                            (lzo1_package, LZO_DIR),
                            (bzip2_package, BZIP2_DIR),
                            (blosc_package, BLOSC_DIR)]:

    if package.tag == 'LZO' and lzo2_enabled:
        print("* Skipping detection of %s since %s has already been found."
              % (lzo1_package.name, lzo2_package.name))
        continue  # do not use LZO 1 if LZO 2 is available

    (hdrdir, libdir, rundir) = package.find_directories(location)

    # check if the library is in the standard compiler paths
    if not libdir and package.target_function:
        libdir = compiler.has_function(package.target_function,
                                       libraries=(package.library_name,))

    if not (hdrdir and libdir):
        if package.tag in ['HDF5']:  # these are compulsory!
            pname, ptag = package.name, package.tag
            exit_with_error(
                "Could not find a local %s installation." % pname,
                "You may need to explicitly state "
                "where your local %(name)s headers and library can be found "
                "by setting the ``%(tag)s_DIR`` environment variable "
                "or by using the ``--%(ltag)s`` command-line option."
                % dict(name=pname, tag=ptag, ltag=ptag.lower()))
        if package.tag == 'BLOSC':  # this is optional, but comes with sources
            print("* Could not find %s headers and library; "
                  "using internal sources." % package.name)
        else:
            print("* Could not find %s headers and library; "
                  "disabling support for it." % package.name)

        continue  # look for the next library

    if libdir in ("", True):
        print("* Found %s headers at ``%s``, the library is located in the "
              "standard system search dirs." % (package.name, hdrdir))
    else:
        print("* Found %s headers at ``%s``, library at ``%s``."
              % (package.name, hdrdir, libdir))

    if package.tag in ['HDF5']:
        hdf5_header = os.path.join(hdrdir, "H5public.h")
        hdf5_version = get_hdf5_version(hdf5_header)
        if hdf5_version < min_hdf5_version:
            exit_with_error(
                "Unsupported HDF5 version! HDF5 v%s+ required. "
                "Found version v%s" % (
                    '.'.join(map(str, min_hdf5_version)),
                    '.'.join(map(str, hdf5_version))))

    if hdrdir not in default_header_dirs:
        inc_dirs.append(hdrdir)  # save header directory if needed
    if libdir not in default_library_dirs and libdir not in ("", True):
        # save library directory if needed
        if os.name == "nt":
            # Important to quote the libdir for Windows (Vista) systems
            lib_dirs.append('"%s"' % libdir)
        else:
            lib_dirs.append(libdir)

    if package.tag not in ['HDF5']:
        # Keep record of the optional libraries found.
        optional_libs.append(package.tag)
        def_macros.append(('HAVE_%s_LIB' % package.tag, 1))

    if not rundir:
        loc = {
            'posix': "the default library paths.",
            'nt': "any of the directories in %%PATH%%.",
        }[os.name]

        print_warning(
            "Could not find the %s runtime." % package.name,
            "The %(name)s shared library was *not* found in %(loc)s "
            "In case of runtime problems, please remember to install it."
            % dict(name=package.name, loc=loc)
        )

    if os.name == "nt":
        # LZO DLLs cannot be copied to the binary package for license reasons
        if package.tag not in ['LZO', 'LZO2']:
            dll_file = _platdep[package.tag][1] + '.dll'
            # If DLL is not in rundir, do nothing.  This can be useful
            # for BZIP2, that can be linked either statically (.LIB)
            # or dinamically (.DLL)
            if rundir is not None:
                dll_files.append(os.path.join(rundir, dll_file))

    if package.tag == 'LZO2':
        lzo2_enabled = True

if lzo2_enabled:
    lzo_package = lzo2_package
else:
    lzo_package = lzo1_package

#------------------------------------------------------------------------------

cython_extnames = [
    'utilsextension',
    'hdf5extension',
    'tableextension',
    'linkextension',
    '_comp_lzo',
    '_comp_bzip2',
    'lrucacheextension',
    'indexesextension',
]


def get_cython_extfiles(extnames):
    extdir = 'tables'
    extfiles = {}

    for extname in extnames:
        extfile = os.path.join(extdir, extname)
        extpfile = '%s.pyx' % extfile
        extcfile = '%s.c' % extfile

        if not exists(extcfile) or newer(extpfile, extcfile):
            # For some reason, setup in setuptools does not compile
            # Cython files (!)  Do that manually...
            # 2013/08/24: the issue should be fixed in distribute 0.6.15
            # see also https://bitbucket.org/tarek/distribute/issue/195
            print("cythoning %s to %s" % (extpfile, extcfile))
            retcode = subprocess.call(
                [sys.executable, "-m", "cython", extpfile]
            )
            if retcode > 0:
                print("cython aborted compilation with retcode:", retcode)
                sys.exit()
        extfiles[extname] = extcfile

    return extfiles


cython_extfiles = get_cython_extfiles(cython_extnames)

# Update the version.h file if this file is newer
if newer('VERSION', 'src/version.h'):
    open('src/version.h', 'w').write(
        '#define PYTABLES_VERSION "%s"\n' % VERSION)

#--------------------------------------------------------------------

# Package information for ``setuptools``.
if has_setuptools:
    # PyTables contains data files for tests.
    setuptools_kwargs['zip_safe'] = False

    # ``NumPy`` headers are needed for building the extensions, as
    # well as Cython.
    setuptools_kwargs['setup_requires'] = [
        'numpy>=%s' % min_numpy_version,
        'cython>=%s' % min_cython_version,
    ]

    # ``NumPy`` and ``Numexpr`` are absolutely required for running PyTables.
    setuptools_kwargs['install_requires'] = [
        'numpy>=%s' % min_numpy_version,
        'numexpr>=%s' % min_numexpr_version,
    ]

    setuptools_kwargs['extras_require'] = {}

    # Detect packages automatically.
    setuptools_kwargs['packages'] = find_packages(exclude=['*.bench'])
    # Entry points for automatic creation of scripts.
    setuptools_kwargs['entry_points'] = {
        'console_scripts': [
            'ptdump = tables.scripts.ptdump:main',
            'ptrepack = tables.scripts.ptrepack:main',
            'pt2to3 = tables.scripts.pt2to3:main',
            'pttree = tables.scripts.pttree:main',
        ],
    }

    # Test suites.
    setuptools_kwargs['test_suite'] = 'tables.tests.test_all.suite'
    setuptools_kwargs['scripts'] = []
else:
    # The next should work with stock distutils, but it does not!
    # It is better to rely on check_import
    # setuptools_kwargs['requires'] = ['numpy (>= %s)' % min_numpy_version,
    #                                  'numexpr (>= %s)' % min_numexpr_version]
    # There is no other chance, these values must be hardwired.
    setuptools_kwargs['packages'] = [
        'tables', 'tables.nodes', 'tables.scripts', 'tables.misc',
        # Test suites.
        'tables.tests', 'tables.nodes.tests',
    ]
    setuptools_kwargs['scripts'] = [
        'utils/ptdump', 'utils/ptrepack', 'utils/pt2to3', 'utils/pttree']
# Copy additional data for packages that need it.
setuptools_kwargs['package_data'] = {
    'tables.tests': ['*.h5', '*.mat'],
    'tables.nodes.tests': ['*.dat', '*.xbm', '*.h5']}


#Having the Python version included in the package name makes managing a
#system with multiple versions of Python much easier.

def find_name(base='tables'):
    '''If "--name-with-python-version" is on the command line then
    append "-pyX.Y" to the base name'''
    name = base
    if '--name-with-python-version' in sys.argv:
        name += '-py%i.%i' % (sys.version_info[0], sys.version_info[1])
        sys.argv.remove('--name-with-python-version')
    return name


name = find_name()

if os.name == "nt":
    # Add DLL's to the final package for windows
    data_files.extend([
        ('Lib/site-packages/%s' % name, dll_files),
    ])

ADDLIBS = [hdf5_package.library_name]

# List of Blosc file dependencies
blosc_files = ["c-blosc/hdf5/blosc_filter.c"]
if 'BLOSC' not in optional_libs:
    # Compiling everything from sources
    # Blosc + BloscLZ sources
    blosc_files += glob.glob('c-blosc/blosc/*.c')
    # LZ4 sources
    blosc_files += glob.glob('c-blosc/internal-complibs/lz4*/*.c')
    # Snappy sources
    blosc_files += glob.glob('c-blosc/internal-complibs/snappy*/*.cc')
    # Zlib sources
    blosc_files += glob.glob('c-blosc/internal-complibs/zlib*/*.c')
    # Finally, add all the include dirs...
    inc_dirs += [os.path.join('c-blosc', 'blosc')]
    inc_dirs += glob.glob('c-blosc/internal-complibs/*')
    # ...and the macros for all the compressors supported
    def_macros += [('HAVE_LZ4', 1), ('HAVE_SNAPPY', 1), ('HAVE_ZLIB', 1)]

    # Add extra flags for optimizing shuffle in include Blosc
    def compiler_has_flags(compiler, flags):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c',
                                         delete=False) as fd:
            fd.write('int main() {return 0;}')

        try:
            compiler.compile([fd.name], extra_preargs=flags)
        except Exception:
            return False
        else:
            return True
        finally:
            os.remove(fd.name)

    try_flags = ["-march=native", "-msse2"]
    for ff in try_flags:
        if compiler_has_flags(compiler, [ff]):
            print("Setting compiler flag: " + ff)
            CFLAGS.append(ff)
            break
else:
    ADDLIBS += ['blosc']


utilsExtension_libs = LIBS + ADDLIBS
hdf5Extension_libs = LIBS + ADDLIBS
tableExtension_libs = LIBS + ADDLIBS
linkExtension_libs = LIBS + ADDLIBS
indexesExtension_libs = LIBS + ADDLIBS
lrucacheExtension_libs = []    # Doesn't need external libraries

# Compressor modules only need other libraries if they are enabled.
_comp_lzo_libs = LIBS[:]
_comp_bzip2_libs = LIBS[:]
for (package, complibs) in [(lzo_package, _comp_lzo_libs),
                            (bzip2_package, _comp_bzip2_libs)]:

    if package.tag in optional_libs:
        complibs.extend([hdf5_package.library_name, package.library_name])


extensions = [
    Extension("tables.utilsextension",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['utilsextension'],
                       "src/utils.c",
                       "src/H5ARRAY.c",
                       "src/H5ATTR.c",
                       ] + blosc_files,
              library_dirs=lib_dirs,
              libraries=utilsExtension_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables.hdf5extension",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['hdf5extension'],
                       "src/utils.c",
                       "src/typeconv.c",
                       "src/H5ARRAY.c",
                       "src/H5ARRAY-opt.c",
                       "src/H5VLARRAY.c",
                       "src/H5ATTR.c",
                       ] + blosc_files,
              library_dirs=lib_dirs,
              libraries=hdf5Extension_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables.tableextension",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['tableextension'],
                       "src/utils.c",
                       "src/typeconv.c",
                       "src/H5TB-opt.c",
                       "src/H5ATTR.c",
                       ] + blosc_files,
              library_dirs=lib_dirs,
              libraries=tableExtension_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables._comp_lzo",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['_comp_lzo'],
                       "src/H5Zlzo.c"],
              library_dirs=lib_dirs,
              libraries=_comp_lzo_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables._comp_bzip2",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['_comp_bzip2'],
                       "src/H5Zbzip2.c"],
              library_dirs=lib_dirs,
              libraries=_comp_bzip2_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables.linkextension",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['linkextension']],
              library_dirs=lib_dirs,
              libraries=tableExtension_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables.lrucacheextension",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['lrucacheextension']],
              library_dirs=lib_dirs,
              libraries=lrucacheExtension_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

    Extension("tables.indexesextension",
              include_dirs=inc_dirs,
              define_macros=def_macros,
              sources=[cython_extfiles['indexesextension'],
                       "src/H5ARRAY-opt.c",
                       "src/idx-opt.c"],
              library_dirs=lib_dirs,
              libraries=indexesExtension_libs,
              extra_link_args=LFLAGS,
              extra_compile_args=CFLAGS),

]


classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""

setup(
    name=name,
    version=VERSION,
    description='Hierarchical datasets for Python',
    long_description="""\
PyTables is a package for managing hierarchical datasets and
designed to efficently cope with extremely large amounts of
data. PyTables is built on top of the HDF5 library and the
NumPy package and features an object-oriented interface
that, combined with C-code generated from Cython sources,
makes of it a fast, yet extremely easy to use tool for
interactively save and retrieve large amounts of data.

""",
    classifiers=[c for c in classifiers.split("\n") if c],
    author='Francesc Alted, Ivan Vilata, et al.',
    author_email='pytables@pytables.org',
    maintainer='PyTables maintainers',
    maintainer_email='pytables@pytables.org',
    url='http://www.pytables.org/',
    license='http://www.opensource.org/licenses/bsd-license.php',
    download_url="http://sourceforge.net/projects/pytables/files/pytables/",
    platforms=['any'],
    ext_modules=extensions,
    cmdclass=cmdclass,
    data_files=data_files,
    **setuptools_kwargs
)
