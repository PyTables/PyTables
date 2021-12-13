#!/usr/bin/env python

"""Setup script for the tables package"""

import os
import sys
import ctypes
import tempfile
import textwrap
import subprocess
from pathlib import Path

# Using ``setuptools`` enables lots of goodies
from setuptools import setup, find_packages
import pkg_resources

from distutils.core import Extension
from distutils.dep_util import newer
from distutils.util import convert_path
from distutils.ccompiler import new_compiler
from distutils.version import LooseVersion
import distutils.spawn

# We need to avoid importing numpy until we can be sure it's installed
# This approach is based on this SO answer http://stackoverflow.com/a/21621689
# This is also what pandas does.
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    # `cpuinfo.py` uses multiprocessing to check CPUID flags. On Windows, the
    # entire setup script needs to be protected as a result
    # For guessing the capabilities of the CPU for C-Blosc
    try:
        import cpuinfo

        cpu_info = cpuinfo.get_cpu_info()
        cpu_flags = cpu_info["flags"]
    except Exception as e:
        print("cpuinfo failed, assuming no CPU features:", e)
        cpu_flags = []

    # The name for the pkg-config utility
    PKG_CONFIG = "pkg-config"

    # Fetch the requisites
    requirements = (ROOT / "requirements.txt").read_text().splitlines()

    class BuildExtensions(build_ext):
        """Subclass setuptools build_ext command

        BuildExtensions does two things
        1) it makes sure numpy is available
        2) it injects numpy's core/include directory in the include_dirs
           parameter of all extensions
        3) it runs the original build_ext command
        """

        def run(self):
            # According to
            # https://pip.pypa.io/en/stable/reference/pip_install.html#installation-order
            # at this point we can be sure pip has already installed numpy
            numpy_incl = pkg_resources.resource_filename(
                "numpy", "core/include"
            )

            for ext in self.extensions:
                if (
                    hasattr(ext, "include_dirs")
                    and numpy_incl not in ext.include_dirs
                ):
                    ext.include_dirs.append(numpy_incl)

            build_ext.run(self)

    cmdclass = {"build_ext": BuildExtensions}
    setuptools_kwargs = {}

    # Some functions for showing errors and warnings.
    def _print_admonition(kind, head, body):
        tw = textwrap.TextWrapper(
            initial_indent="   ", subsequent_indent="   "
        )

        print(f".. {kind.upper()}:: {head}")
        for line in tw.wrap(body):
            print(line)

    def exit_with_error(head, body=""):
        _print_admonition("error", head, body)
        sys.exit(1)

    def print_warning(head, body=""):
        _print_admonition("warning", head, body)

    # The minimum required versions
    min_python_version = (3, 6)
    # Check for Python
    if sys.version_info < min_python_version:
        exit_with_error("You need Python 3.6 or greater to install PyTables!")
    print(f"* Using Python {sys.version.splitlines()[0]}")

    try:
        import cython
        print(f"* Found cython {cython.__version__}")
        del cython
    except ImportError:
        pass

    # Minimum required versions for numpy, numexpr and HDF5
    _min_versions = {}
    exec((ROOT / "tables" / "req_versions.py").read_text(), _min_versions)
    min_hdf5_version = _min_versions["min_hdf5_version"]
    min_blosc_version = _min_versions["min_blosc_version"]
    min_blosc_bitshuffle_version = _min_versions[
        "min_blosc_bitshuffle_version"
    ]

    VERSION = (ROOT / "VERSION").read_text().strip()

    # ----------------------------------------------------------------------

    debug = "--debug" in sys.argv

    # Global variables
    lib_dirs = []
    inc_dirs = [ROOT / "hdf5-blosc" / "src"]
    optional_libs = []
    data_files = []  # list of data files to add to packages (mainly for DLL's)

    default_header_dirs = None
    default_library_dirs = None
    default_runtime_dirs = None

    def add_from_path(envname, dirs):
        dirs.extend(
            Path(x) for x in os.environ.get(envname, "").split(os.pathsep) if x
        )

    def add_from_flags(envname, flag_key, dirs):
        dirs.extend(
            Path(flag[len(flag_key) :])
            for flag in os.environ.get(envname, "").split()
            if flag.startswith(flag_key)
        )

    if os.name == "posix":
        prefixes = ("/usr/local", "/sw", "/opt", "/opt/local", "/usr", "/")
        prefix_paths = [Path(x) for x in prefixes]

        default_header_dirs = []
        add_from_path("CPATH", default_header_dirs)
        add_from_path("C_INCLUDE_PATH", default_header_dirs)
        add_from_flags("CPPFLAGS", "-I", default_header_dirs)
        add_from_flags("CFLAGS", "-I", default_header_dirs)
        default_header_dirs.extend(_tree / "include" for _tree in prefix_paths)

        default_library_dirs = []
        add_from_flags("LDFLAGS", "-L", default_library_dirs)
        default_library_dirs.extend(
            _tree / _arch
            for _tree in prefix_paths
            for _arch in ("lib64", "lib")
        )
        default_runtime_dirs = default_library_dirs

    elif os.name == "nt":
        default_header_dirs = []  # no default, must be given explicitly
        default_library_dirs = []  # no default, must be given explicitly
        default_runtime_dirs = [  # look for DLL files in ``%PATH%``
            Path(_path) for _path in os.environ["PATH"].split(";")
        ]
        # Add the \Windows\system to the runtime list (necessary for Vista)
        default_runtime_dirs.append(Path("\\windows\\system"))
        # Add the \path_to_python\DLLs and tables package to the list
        default_runtime_dirs.append(
            Path(sys.prefix) / "Lib" / "site-packages" / "tables"
        )

    # Gcc 4.0.1 on Mac OS X 10.4 does not seem to include the default
    # header and library paths.  See ticket #18.
    if sys.platform.lower().startswith("darwin"):
        inc_dirs.extend(default_header_dirs)
        lib_dirs.extend(default_library_dirs)

    def _find_file_path(name, locations, prefixes=("",), suffixes=("",)):
        for prefix in prefixes:
            for suffix in suffixes:
                for location in locations:
                    path = location / f"{prefix}{name}{suffix}"
                    if path.is_file():
                        return str(path)
        return None

    class BasePackage:
        _library_prefixes = []
        _library_suffixes = []
        _runtime_prefixes = []
        _runtime_suffixes = []
        _component_dirs = []

        def __init__(
            self, name, tag, header_name, library_name, target_function=None
        ):
            self.name = name
            self.tag = tag
            self.header_name = header_name
            self.library_name = library_name
            self.runtime_name = library_name
            self.target_function = target_function

        def find_header_path(self, locations=default_header_dirs):
            return _find_file_path(
                self.header_name, locations, suffixes=[".h"]
            )

        def find_library_path(self, locations=default_library_dirs):
            return _find_file_path(
                self.library_name,
                locations,
                self._library_prefixes,
                self._library_suffixes,
            )

        def find_runtime_path(self, locations=default_runtime_dirs):
            """
            returns True if the runtime can be found
            returns None otherwise
            """
            # An explicit path can not be provided for runtime libraries.
            # (The argument is accepted for compatibility with previous
            # methods.)

            # dlopen() won't tell us where the file is, just whether
            # success occurred, so this returns True instead of a filename
            for prefix in self._runtime_prefixes:
                for suffix in self._runtime_suffixes:
                    try:
                        ctypes.CDLL(f"{prefix}{self.runtime_name}{suffix}")
                    except OSError:
                        pass
                    else:
                        return True

        def _pkg_config(self, flags):
            try:
                cmd = [PKG_CONFIG] + flags.split() + [self.library_name]
                config = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except (OSError, subprocess.CalledProcessError):
                return []
            else:
                return config.decode().strip().split()

        def find_directories(self, location, use_pkgconfig=False):
            dirdata = [
                (self.header_name, self.find_header_path, default_header_dirs),
                (
                    self.library_name,
                    self.find_library_path,
                    default_library_dirs,
                ),
                (
                    self.runtime_name,
                    self.find_runtime_path,
                    default_runtime_dirs,
                ),
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
                    Path(str(location).strip('"')) / compdir
                    for compdir in self._component_dirs
                ]

            if use_pkgconfig:
                # header
                pkgconfig_header_dirs = [
                    Path(d[2:])
                    for d in self._pkg_config("--cflags")
                    if d.startswith("-I")
                ]
                if pkgconfig_header_dirs:
                    print(
                        f"* pkg-config header dirs for {self.name}:",
                        ", ".join(str(x) for x in pkgconfig_header_dirs),
                    )

                # library
                pkgconfig_library_dirs = [
                    Path(d[2:])
                    for d in self._pkg_config("--libs-only-L")
                    if d.startswith("-L")
                ]
                if pkgconfig_library_dirs:
                    print(
                        f"* pkg-config library dirs for {self.name}:",
                        ", ".join(str(x) for x in pkgconfig_library_dirs),
                    )

                # runtime
                pkgconfig_runtime_dirs = pkgconfig_library_dirs

                pkgconfig_dirs = [
                    pkgconfig_header_dirs,
                    pkgconfig_library_dirs,
                    pkgconfig_runtime_dirs,
                ]
            else:
                pkgconfig_dirs = [None, None, None]

            directories = [None, None, None]  # headers, libraries, runtime
            for idx, (name, find_path, default_dirs) in enumerate(dirdata):
                path = find_path(
                    pkgconfig_dirs[idx] or locations or default_dirs
                )
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
                        directories[idx] = Path(path[: path.rfind(name)])
                    else:
                        directories[idx] = Path(path).parent

            return tuple(directories)

    class PosixPackage(BasePackage):
        _library_prefixes = ["lib"]
        _library_suffixes = [".so", ".dylib", ".a"]
        _runtime_prefixes = _library_prefixes
        _runtime_suffixes = [".so", ".dylib"]
        _component_dirs = ["include", "lib", "lib64"]

    class WindowsPackage(BasePackage):
        _library_prefixes = [""]
        _library_suffixes = [".lib"]
        _runtime_prefixes = [""]
        _runtime_suffixes = [".dll"]

        # lookup in '.' seems necessary for LZO2
        _component_dirs = ["include", "lib", "dll", "bin", "."]

        def find_runtime_path(self, locations=default_runtime_dirs):
            # An explicit path can not be provided for runtime libraries.
            # (The argument is accepted for compatibility with previous
            # methods.)
            return _find_file_path(
                self.runtime_name,
                default_runtime_dirs,
                self._runtime_prefixes,
                self._runtime_suffixes,
            )

    # Get the HDF5 version provided the 'H5public.h' header
    def get_hdf5_version(headername):
        major, minor, release = None, None, None
        for line in headername.read_text().splitlines():
            if "H5_VERS_MAJOR" in line:
                major = int(line.split()[2])
            elif "H5_VERS_MINOR" in line:
                minor = int(line.split()[2])
            elif "H5_VERS_RELEASE" in line:
                release = int(line.split()[2])
            if None not in (major, minor, release):
                break
        else:
            exit_with_error("Unable to detect HDF5 library version!")
        return LooseVersion(f"{major}.{minor}.{release}")

    # Get the Blosc version provided the 'blosc.h' header
    def get_blosc_version(headername):
        major, minor, release = None, None, None
        for line in headername.read_text().splitlines():
            if "BLOSC_VERSION_MAJOR" in line:
                major = int(line.split()[2])
            elif "BLOSC_VERSION_MINOR" in line:
                minor = int(line.split()[2])
            elif "BLOSC_VERSION_RELEASE" in line:
                release = int(line.split()[2])
            if None not in (major, minor, release):
                break
        else:
            exit_with_error("Unable to detect Blosc library version!")
        return LooseVersion(f"{major}.{minor}.{release}")

    _cp = convert_path
    if os.name == "posix":
        _Package = PosixPackage
        _platdep = {  # package tag -> platform-dependent components
            "HDF5": ["hdf5"],
            "LZO2": ["lzo2"],
            "LZO": ["lzo"],
            "BZ2": ["bz2"],
            "BLOSC": ["blosc"],
        }

    elif os.name == "nt":
        _Package = WindowsPackage
        _platdep = {  # package tag -> platform-dependent components
            "HDF5": ["hdf5", "hdf5"],
            "LZO2": ["lzo2", "lzo2"],
            "LZO": ["liblzo", "lzo1"],
            "BZ2": ["bzip2", "bzip2"],
            "BLOSC": ["blosc", "blosc"],
        }

        # Copy the next DLL's to binaries by default.
        dll_files = [
            # '\\windows\\system\\zlib1.dll',
            # '\\windows\\system\\szip.dll',
        ]

        if os.environ.get("HDF5_USE_PREFIX", None):
            # This is used on CI systems to link against HDF5 library
            # The vendored `hdf5.dll` in a wheel is renamed to:
            # `pytables_hdf5.dll`  This should prevent DLL Hell.
            print(
                "* HDF5_USE_PREFIX: Trying to build against pytables_hdf5.dll"
            )
            _platdep["HDF5"] = ["pytables_hdf5", "pytables_hdf5"]

        if debug:
            _platdep["HDF5"] = ["hdf5_D", "hdf5_D"]

    else:
        _Package = None
        _platdep = {}
        exit_with_error(f"Unsupported OS: {os.name}")

    hdf5_package = _Package("HDF5", "HDF5", "H5public", *_platdep["HDF5"])
    hdf5_package.target_function = "H5close"
    lzo2_package = _Package(
        "LZO 2", "LZO2", _cp("lzo/lzo1x"), *_platdep["LZO2"]
    )
    lzo2_package.target_function = "lzo_version_date"
    lzo1_package = _Package("LZO 1", "LZO", "lzo1x", *_platdep["LZO"])
    lzo1_package.target_function = "lzo_version_date"
    bzip2_package = _Package("bzip2", "BZ2", "bzlib", *_platdep["BZ2"])
    bzip2_package.target_function = "BZ2_bzlibVersion"
    blosc_package = _Package("blosc", "BLOSC", "blosc", *_platdep["BLOSC"])
    blosc_package.target_function = "blosc_list_compressors"  # Blosc >= 1.3

    # -----------------------------------------------------------------

    def_macros = [("NDEBUG", 1)]
    # Define macros for Windows platform
    if os.name == "nt":
        def_macros.append(("WIN32", 1))
        def_macros.append(("_HDF5USEDLL_", 1))
        def_macros.append(("H5_BUILT_AS_DYNAMIC_LIB", 1))

    # Allow setting the HDF5 dir and additional link flags either in
    # the environment or on the command line.
    # First check the environment...
    HDF5_DIR = os.environ.get("HDF5_DIR", "")
    LZO_DIR = os.environ.get("LZO_DIR", "")
    BZIP2_DIR = os.environ.get("BZIP2_DIR", "")
    BLOSC_DIR = os.environ.get("BLOSC_DIR", "")
    LFLAGS = os.environ.get("LFLAGS", "").split()
    # in GCC-style compilers, -w in extra flags will get rid of copious
    # 'uninitialized variable' Cython warnings. However, this shouldn't be
    # the default as it will suppress *all* the warnings, which definitely
    # is not a good idea.
    CFLAGS = os.environ.get("CFLAGS", "").split()
    LIBS = os.environ.get("LIBS", "").split()
    CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
    # We start using pkg-config since some distributions are putting HDF5
    # (and possibly other libraries) in exotic locations.  See issue #442.
    if distutils.spawn.find_executable(PKG_CONFIG):
        USE_PKGCONFIG = os.environ.get("USE_PKGCONFIG", "TRUE")
    else:
        USE_PKGCONFIG = "FALSE"

    # ...then the command line.
    # Handle --hdf5=[PATH] --lzo=[PATH] --bzip2=[PATH] --blosc=[PATH]
    # --lflags=[FLAGS] --cflags=[FLAGS] and --debug
    for arg in list(sys.argv):
        key, _, val = arg.partition("=")
        if key == "--hdf5":
            HDF5_DIR = Path(val).expanduser()
        elif key == "--lzo":
            LZO_DIR = Path(val).expanduser()
        elif key == "--bzip2":
            BZIP2_DIR = Path(val).expanduser()
        elif key == "--blosc":
            BLOSC_DIR = Path(val).expanduser()
        elif key == "--lflags":
            LFLAGS = val.split()
        elif key == "--cflags":
            CFLAGS = val.split()
        elif key == "--debug":
            # For debugging (mainly compression filters)
            if os.name != "nt":  # to prevent including dlfcn.h by utils.c!!!
                def_macros = [("DEBUG", 1)]
            # Don't delete this argument. It maybe useful for distutils
            # when adding more flags later on
            continue
        elif key == "--use-pkgconfig":
            USE_PKGCONFIG = val
            CONDA_PREFIX = ""
        elif key == "--no-conda":
            CONDA_PREFIX = ""
        else:
            continue
        sys.argv.remove(arg)

    USE_PKGCONFIG = USE_PKGCONFIG.upper() == "TRUE"
    print("* USE_PKGCONFIG:", USE_PKGCONFIG)

    # For windows, search for the hdf5 dll in the path and use it if found.
    # This is much more convenient than having to manually set an environment
    # variable to rebuild pytables
    if not HDF5_DIR and os.name == "nt":
        import ctypes.util

        if not debug:
            libdir = ctypes.util.find_library(
                "hdf5.dll"
            ) or ctypes.util.find_library("hdf5dll.dll")
        else:
            libdir = ctypes.util.find_library(
                "hdf5_D.dll"
            ) or ctypes.util.find_library("hdf5ddll.dll")
        # Like 'C:\\Program Files\\HDF Group\\HDF5\\1.8.8\\bin\\hdf5dll.dll'
        if libdir:
            # Strip off the filename and the 'bin' directory
            HDF5_DIR = Path(libdir).parent.parent
            print(f"* Found HDF5 using system PATH ('{libdir}')")

    if CONDA_PREFIX:
        CONDA_PREFIX = Path(CONDA_PREFIX)
        print(f"* Found conda env: ``{CONDA_PREFIX}``")
        if os.name == "nt":
            CONDA_PREFIX = CONDA_PREFIX / "Library"

    # The next flag for the C compiler is needed for finding the C headers for
    # the Cython extensions
    CFLAGS.append("-Isrc")

    # Force the 1.8.x HDF5 API even if the library as been compiled to use the
    # 1.6.x API by default
    CFLAGS.extend(
        [
            "-DH5_USE_18_API",
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
            # "-DH5Pinsert_vers=2",
            # "-DH5Pregister_vers=2",
            # "-DH5Rget_obj_type_vers=2",
            "-DH5Tarray_create_vers=2",
            # "-DH5Tcommit_vers=2",
            "-DH5Tget_array_dims_vers=2",
            # "-DH5Topen_vers=2",
            "-DH5Z_class_t_vers=2",
        ]
    )
    # H5Oget_info_by_name seems to have performance issues (see gh-402), so we
    # need to use teh deprecated H5Gget_objinfo function
    # CFLAGS.append("-DH5_NO_DEPRECATED_SYMBOLS")

    # Do not use numpy deprecated API
    CFLAGS.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

    # Try to locate the compulsory and optional libraries.
    lzo2_enabled = False
    compiler = new_compiler()
    for (package, location) in [
        (hdf5_package, HDF5_DIR),
        (lzo2_package, LZO_DIR),
        (lzo1_package, LZO_DIR),
        (bzip2_package, BZIP2_DIR),
        (blosc_package, BLOSC_DIR),
    ]:

        if package.tag == "LZO" and lzo2_enabled:
            print(
                f"* Skipping detection of {lzo1_package.name} "
                f"since {lzo2_package.name} has already been found."
            )
            continue  # do not use LZO 1 if LZO 2 is available

        # if a package location is not specified, try to find it in conda env
        if not location and CONDA_PREFIX:
            location = CONDA_PREFIX

        # looking for lzo/lzo1x.h but pkgconfig already returns
        # '/usr/include/lzo'
        use_pkgconfig = USE_PKGCONFIG and package.tag != "LZO2"

        (hdrdir, libdir, rundir) = package.find_directories(
            location, use_pkgconfig=use_pkgconfig
        )

        # check if HDF5 library uses old DLL naming scheme
        if hdrdir and package.tag == "HDF5":
            hdf5_version = get_hdf5_version(Path(hdrdir) / "H5public.h")
            if hdf5_version < min_hdf5_version:
                exit_with_error(
                    f"Unsupported HDF5 version! HDF5 v{min_hdf5_version}+ "
                    f"required. Found version v{hdf5_version}"
                )

            if os.name == "nt" and hdf5_version < LooseVersion("1.8.10"):
                # Change in DLL naming happened in 1.8.10
                hdf5_old_dll_name = "hdf5dll" if not debug else "hdf5ddll"
                package.library_name = hdf5_old_dll_name
                package.runtime_name = hdf5_old_dll_name
                _platdep["HDF5"] = [hdf5_old_dll_name, hdf5_old_dll_name]
                _, libdir, rundir = package.find_directories(
                    location, use_pkgconfig=USE_PKGCONFIG
                )

        # check if the library is in the standard compiler paths
        if not libdir and package.target_function:
            libdir = compiler.has_function(
                package.target_function, libraries=(package.library_name,)
            )

        if not (hdrdir and libdir):
            if package.tag in ["HDF5"]:  # these are compulsory!
                pname, ptag = package.name, package.tag
                exit_with_error(
                    f"Could not find a local {pname} installation.",
                    f"You may need to explicitly state where your local "
                    f"{pname} headers and library can be found by setting "
                    f"the ``{ptag}_DIR`` environment variable or by using "
                    f"the ``--{ptag.lower()}`` command-line option.",
                )
            if package.tag == "BLOSC":
                # this is optional, but comes with sources
                print(
                    f"* Could not find {package.name} headers and library; "
                    f"using internal sources."
                )
            else:
                print(
                    f"* Could not find {package.name} headers and library; "
                    f"disabling support for it."
                )

            continue  # look for the next library

        if libdir in ("", True):
            print(
                f"* Found {package.name} headers at ``{hdrdir}``, the library "
                f"is located in the standard system search dirs."
            )
        else:
            print(
                f"* Found {package.name} headers at ``{hdrdir}``, "
                f"library at ``{libdir}``."
            )

        if hdrdir not in default_header_dirs:
            inc_dirs.append(Path(hdrdir))  # save header directory if needed
        if libdir not in default_library_dirs and libdir not in ("", True):
            # save library directory if needed
            lib_dirs.append(Path(libdir))

        if package.tag not in ["HDF5"]:
            # Keep record of the optional libraries found.
            optional_libs.append(package.tag)
            def_macros.append((f"HAVE_{package.tag}_LIB", 1))

        if hdrdir and package.tag == "BLOSC":
            blosc_version = get_blosc_version(Path(hdrdir) / "blosc.h")
            if blosc_version < min_blosc_version:
                optional_libs.pop()  # Remove Blosc from the discovered libs
                print_warning(
                    f"Unsupported Blosc version installed! Blosc "
                    f"{min_blosc_version}+ required. Found version "
                    f"{blosc_version}.  Using internal Blosc sources."
                )
            if blosc_version < min_blosc_bitshuffle_version:
                print_warning(
                    f"This Blosc version does not support the BitShuffle "
                    f"filter. Minimum desirable version is "
                    f"{min_blosc_bitshuffle_version}.  "
                    f"Found version: {blosc_version}"
                )

        if not rundir:
            loc = {
                "posix": "the default library paths",
                "nt": "any of the directories in %%PATH%%",
            }[os.name]

            if "bdist_wheel" in sys.argv and os.name == "nt":
                exit_with_error(
                    f"Could not find the {package.name} runtime.",
                    f"The {package.name} shared library was *not* found in "
                    f"{loc}. Cannot build wheel without the runtime.",
                )
            else:
                print_warning(
                    f"Could not find the {package.name} runtime.",
                    f"The {package.name} shared library was *not* found "
                    f"in {loc}. In case of runtime problems, please "
                    f"remember to install it.",
                )

        if os.name == "nt":
            # LZO DLLs cannot be copied to the binary package for license
            # reasons
            if package.tag not in ["LZO", "LZO2"]:
                dll_file = f"{_platdep[package.tag][1]}.dll"
                # If DLL is not in rundir, do nothing.  This can be useful
                # for BZIP2, that can be linked either statically (.LIB)
                # or dynamically (.DLL)
                if rundir is not None:
                    dll_files.append(Path(rundir) / dll_file)

        if os.name == "nt" and package.tag in ["HDF5"]:
            # hdf5.dll usually depends on zlib.dll
            import ctypes.util
            z_lib_path = ctypes.util.find_library("zlib.dll")
            if z_lib_path:
                print(f"* Adding zlib.dll (hdf5 dependency): ``{z_lib_path}``")
                dll_files.append(z_lib_path)

        if package.tag == "LZO2":
            lzo2_enabled = True

    lzo_package = lzo2_package if lzo2_enabled else lzo1_package

    # ------------------------------------------------------------------------------

    cython_extnames = [
        "utilsextension",
        "hdf5extension",
        "tableextension",
        "linkextension",
        "_comp_lzo",
        "_comp_bzip2",
        "lrucacheextension",
        "indexesextension",
    ]

    def get_cython_extfiles(extnames):
        extdir = ROOT / "tables"
        extfiles = {}

        for extname in extnames:
            extfile = extdir / extname
            extpfile = extfile.with_suffix(".pyx")
            extcfile = extfile.with_suffix(".c")

            if not extcfile.exists() or newer(extpfile, extcfile):
                # This is the only place where Cython is needed, but every
                # developer should have it installed, so it should not be
                # a hard requisite
                from Cython.Build import cythonize

                cythonize(str(extpfile), language_level="2")
            extfiles[extname] = extcfile

        return extfiles

    cython_extfiles = get_cython_extfiles(cython_extnames)

    # Update the version.h file if this file is newer
    if newer(ROOT / "VERSION", ROOT / "src" / "version.h"):
        (ROOT / "src" / "version.h").write_text(
            f'#define PYTABLES_VERSION "{VERSION}"\n'
        )

    # --------------------------------------------------------------------

    # Package information for ``setuptools``
    # PyTables contains data files for tests.
    setuptools_kwargs["zip_safe"] = False

    setuptools_kwargs["install_requires"] = requirements

    # Detect packages automatically.
    setuptools_kwargs["packages"] = find_packages(exclude=["*.bench"])

    # Entry points for automatic creation of scripts.
    setuptools_kwargs["entry_points"] = {
        "console_scripts": [
            "ptdump = tables.scripts.ptdump:main",
            "ptrepack = tables.scripts.ptrepack:main",
            "pt2to3 = tables.scripts.pt2to3:main",
            "pttree = tables.scripts.pttree:main",
        ],
    }

    # Test suites.
    setuptools_kwargs["test_suite"] = "tables.tests.test_all.suite"
    setuptools_kwargs["scripts"] = []

    # Copy additional data for packages that need it.
    setuptools_kwargs["package_data"] = {
        "tables.tests": ["*.h5", "*.mat"],
        "tables.nodes.tests": ["*.dat", "*.xbm", "*.h5"],
    }

    # Having the Python version included in the package name makes managing a
    # system with multiple versions of Python much easier.

    def find_name(base="tables"):
        """If "--name-with-python-version" is on the command line then
        append "-pyX.Y" to the base name"""
        name = base
        if "--name-with-python-version" in sys.argv:
            name += f"-py{sys.version_info[0]}.{sys.version_info[1]}"
            sys.argv.remove("--name-with-python-version")
        return name

    name = find_name()

    if os.name == "nt":
        # Add DLL's to the final package for windows
        data_files.append((Path("Lib") / "site-packages" / name, dll_files))

    ADDLIBS = [hdf5_package.library_name]

    # List of Blosc file dependencies
    blosc_path = ROOT / "c-blosc" / "blosc"
    int_complibs_path = ROOT / "c-blosc" / "internal-complibs"

    blosc_sources = [ROOT / "hdf5-blosc" / "src" / "blosc_filter.c"]
    if "BLOSC" not in optional_libs:
        if not os.environ.get("PYTABLES_NO_EMBEDDED_LIBS", None) is None:
            exit_with_error(
                "Unable to find the blosc library. "
                "The embedded copy of the blosc sources can't be used because "
                "the PYTABLES_NO_EMBEDDED_LIBS environment variable has been "
                "specified)."
            )

        # Compiling everything from sources
        # Blosc + BloscLZ sources
        blosc_sources += [
            f
            for f in blosc_path.glob("*.c")
            if "avx2" not in f.stem and "sse2" not in f.stem
        ]
        blosc_sources += int_complibs_path.glob("lz4*/*.c")  # LZ4 sources
        blosc_sources += int_complibs_path.glob("zlib*/*.c")  # Zlib sources
        blosc_sources += int_complibs_path.glob("zstd*/*/*.c")  # Zstd sources
        # Finally, add all the include dirs...
        inc_dirs += [blosc_path]
        inc_dirs += int_complibs_path.glob("*")
        inc_dirs += int_complibs_path.glob("zstd*/common")
        inc_dirs += int_complibs_path.glob("zstd*")
        # ...and the macros for all the compressors supported
        def_macros += [("HAVE_LZ4", 1), ("HAVE_ZLIB", 1), ("HAVE_ZSTD", 1)]

        # Add extra flags for optimizing shuffle in include Blosc
        def compiler_has_flags(compiler, flags):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".c", delete=False
            ) as fd:
                fd.write("int main() {return 0;}")

            try:
                compiler.compile([fd.name], extra_preargs=flags)
            except Exception:
                return False
            else:
                return True
            finally:
                Path(fd.name).unlink()

        # SSE2
        if "sse2" in cpu_flags:
            print("SSE2 detected and enabled")
            CFLAGS.append("-DSHUFFLE_SSE2_ENABLED")
            if os.name == "nt":
                # Windows always should have support for SSE2
                # (present in all x86/amd64 architectures since 2003)
                def_macros += [("__SSE2__", 1)]
            else:
                # On UNIX, both gcc and clang understand -msse2
                CFLAGS.append("-msse2")
            blosc_sources += blosc_path.glob("*sse2*.c")
        # AVX2
        if "avx2" in cpu_flags and "DISABLE_AVX2" not in os.environ:
            print("AVX2 detected and enabled")
            if os.name == "nt":
                def_macros += [("__AVX2__", 1)]
                CFLAGS.append("-DSHUFFLE_AVX2_ENABLED")
                blosc_sources += blosc_path.glob("*avx2*.c")
            elif compiler_has_flags(compiler, ["-mavx2"]):
                CFLAGS.append("-DSHUFFLE_AVX2_ENABLED")
                CFLAGS.append("-mavx2")
                blosc_sources += blosc_path.glob("*avx2*.c")
    else:
        ADDLIBS += ["blosc"]

    utilsExtension_libs = LIBS + ADDLIBS
    hdf5Extension_libs = LIBS + ADDLIBS
    tableExtension_libs = LIBS + ADDLIBS
    linkExtension_libs = LIBS + ADDLIBS
    indexesExtension_libs = LIBS + ADDLIBS
    lrucacheExtension_libs = []  # Doesn't need external libraries

    # Compressor modules only need other libraries if they are enabled.
    _comp_lzo_libs = LIBS[:]
    _comp_bzip2_libs = LIBS[:]
    for (package, complibs) in [
        (lzo_package, _comp_lzo_libs),
        (bzip2_package, _comp_bzip2_libs),
    ]:

        if package.tag in optional_libs:
            complibs.extend([hdf5_package.library_name, package.library_name])

    # Extension expects strings, so we have to convert Path to str
    blosc_sources = [str(x) for x in blosc_sources]
    inc_dirs = [str(x) for x in inc_dirs]

    extension_kwargs = {
        "extra_compile_args": CFLAGS,
        "extra_link_args": LFLAGS,
        "library_dirs": [str(x) for x in lib_dirs],
        "define_macros": def_macros,
        "include_dirs": [str(x) for x in inc_dirs],
    }

    extensions = [
        Extension(
            "tables.utilsextension",
            sources=[
                str(cython_extfiles["utilsextension"]),
                "src/utils.c",
                "src/H5ARRAY.c",
                "src/H5ATTR.c",
            ]
            + blosc_sources,
            libraries=utilsExtension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.hdf5extension",
            sources=[
                str(cython_extfiles["hdf5extension"]),
                "src/utils.c",
                "src/typeconv.c",
                "src/H5ARRAY.c",
                "src/H5ARRAY-opt.c",
                "src/H5VLARRAY.c",
                "src/H5ATTR.c",
            ]
            + blosc_sources,
            libraries=hdf5Extension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.tableextension",
            sources=[
                str(cython_extfiles["tableextension"]),
                "src/utils.c",
                "src/typeconv.c",
                "src/H5TB-opt.c",
                "src/H5ATTR.c",
            ]
            + blosc_sources,
            libraries=tableExtension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables._comp_lzo",
            sources=[str(cython_extfiles["_comp_lzo"]), "src/H5Zlzo.c"],
            libraries=_comp_lzo_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables._comp_bzip2",
            sources=[str(cython_extfiles["_comp_bzip2"]), "src/H5Zbzip2.c"],
            libraries=_comp_bzip2_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.linkextension",
            sources=[str(cython_extfiles["linkextension"])],
            libraries=tableExtension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.lrucacheextension",
            sources=[str(cython_extfiles["lrucacheextension"])],
            libraries=lrucacheExtension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.indexesextension",
            sources=[
                str(cython_extfiles["indexesextension"]),
                "src/H5ARRAY-opt.c",
                "src/idx-opt.c",
            ],
            libraries=indexesExtension_libs,
            **extension_kwargs,
        ),
    ]

    classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: Microsoft :: Windows
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
"""
    setup(
        name=name,
        version=VERSION,
        description="Hierarchical datasets for Python",
        long_description="""\
PyTables is a package for managing hierarchical datasets and
designed to efficiently cope with extremely large amounts of
data. PyTables is built on top of the HDF5 library and the
NumPy package and features an object-oriented interface
that, combined with C-code generated from Cython sources,
makes of it a fast, yet extremely easy to use tool for
interactively save and retrieve large amounts of data.
""",
        classifiers=[c for c in classifiers.split("\n") if c],
        author=(
            "Francesc Alted, Ivan Vilata,"
            "Antonio Valentino, Anthony Scopatz et al."
        ),
        author_email="pytables@pytables.org",
        maintainer="PyTables maintainers",
        maintainer_email="pytables@pytables.org",
        url="http://www.pytables.org/",
        license="BSD 2-Clause",
        python_requires=">=3.6",
        platforms=["any"],
        ext_modules=extensions,
        cmdclass=cmdclass,
        data_files=[
            (str(parent), [str(file) for file in files])
            for parent, files in data_files
        ],
        extras_require={
            "doc": ["sphinx >= 1.1", "sphinx_rtd_theme", "numpydoc", "ipython"]
        },
        **setuptools_kwargs,
    )

elif __name__ == "__mp_main__":
    pass
