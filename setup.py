#!/usr/bin/env python

"""Setup script for the tables package"""

import os
import sys
import glob
import ctypes
import shutil
import fnmatch
import platform
import tempfile
import textwrap
import subprocess
from typing import NamedTuple
from pathlib import Path

# Using ``setuptools`` enables lots of goodies
from setuptools import setup, Extension
from packaging.version import Version
from setuptools.command.build_ext import build_ext

# The name for the pkg-config utility
PKG_CONFIG = "pkg-config"
COPY_DLLS = bool(os.environ.get("COPY_DLLS", "FALSE") == "TRUE")


# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(initial_indent="   ", subsequent_indent="   ")

    print(f".. {kind.upper()}:: {head}")
    for line in tw.wrap(body):
        print(line)


def exit_with_error(head, body=""):
    _print_admonition("error", head, body)
    sys.exit(1)


def print_warning(head, body=""):
    _print_admonition("warning", head, body)


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
    return Version(f"{major}.{minor}.{release}")


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
    return Version(f"{major}.{minor}.{release}")


# Get the Blosc2 version provided the 'blosc2.h' header
def get_blosc2_version(headername):
    major, minor, release = None, None, None
    for line in headername.read_text().splitlines():
        if "BLOSC2_VERSION_MAJOR" in line and "major" in line:
            major = int(line.split()[2])
        elif "BLOSC2_VERSION_MINOR" in line and "minor" in line:
            minor = int(line.split()[2])
        elif "BLOSC2_VERSION_RELEASE" in line and "tweaks" in line:
            release = int(line.split()[2])
        if None not in (major, minor, release):
            break
    else:
        exit_with_error("Unable to detect Blosc2 library version!")
    return Version(f"{major}.{minor}.{release}")


def get_blosc2_directories():
    """Get Blosc2 directories for the C library"""
    try:
        import blosc2
    except ModuleNotFoundError:
        raise OSError("Cannot import the blosc2 requirement")
    version = blosc2.__version__
    basepath = Path(os.path.dirname(blosc2.__file__))
    recinfo = basepath.parent / f"blosc2-{version}.dist-info" / "RECORD"
    include_path = library_path = runtime_path = None
    for line in open(recinfo):
        path, _ = line.split(",", 1)
        if fnmatch.fnmatch(path, "**/bin/libblosc2*"):
            runtime_path = basepath.parent.joinpath(path).resolve()
            if not runtime_path.is_file():
                raise FileNotFoundError(
                    f"File does not exists: {runtime_path}"
                )
            runtime_path = runtime_path.parent.resolve()
        elif fnmatch.fnmatch(path, "**/libblosc2*"):
            library_path = basepath.parent.joinpath(path).resolve()
            if not library_path.is_file():
                raise FileNotFoundError(
                    f"File does not exists: {library_path}"
                )
            library_path = library_path.parent.resolve()
        elif fnmatch.fnmatch(path, "**/include/blosc2.h") or fnmatch.fnmatch(
            path, "include/blosc2.h"
        ):
            include_path = basepath.parent.joinpath(path).resolve()
            if not include_path.is_file():
                raise FileNotFoundError(
                    f"File does not exists: {include_path}"
                )
            include_path = include_path.parent.resolve()

    if not library_path:
        raise NotADirectoryError("Library directory not found for blosc2!")

    if not include_path:
        raise NotADirectoryError("Header directory not found for blosc2!")

    return include_path, library_path, runtime_path


def blosc2_find_directories_hook():
    print("* Run 'blosc2_find_directories_hook'")
    header_dirs = library_dirs = runtime_dirs = None
    if os.environ.get("PYTABLES_NO_BLOSC2_WHEEL", None) is None:
        try:
            header_dirs, library_dirs, runtime_dirs = get_blosc2_directories()
        except OSError:
            print("* Unable to find blosc2 wheel.")
        else:
            if header_dirs is not None:
                header_dirs = [header_dirs]
            if library_dirs is not None:
                library_dirs = [library_dirs]
            if runtime_dirs is not None:
                runtime_dirs = [runtime_dirs]

    return header_dirs, library_dirs, runtime_dirs


def newer(source, target):
    """Return true if 'source' exists and is more recently modified than
    'target', or if 'source' exists and 'target' doesn't.  Return false if
    both exist and 'target' is the same age or younger than 'source'.
    Raise FileNotFoundError if 'source' does not exist.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"file '{source.absolute()}' does not exist")
    target = Path(target)
    if not target.exists():
        return True
    return source.stat().st_mtime > target.stat().st_mtime


# https://github.com/pypa/setuptools/issues/2806
def new_compiler():
    from setuptools import Distribution

    build_ext = Distribution().get_command_obj("build_ext")
    build_ext.finalize_options()
    # register an extension to ensure a compiler is created
    build_ext.extensions = [Extension("ignored", ["ignored.c"])]
    # disable building fake extensions
    build_ext.build_extensions = lambda: None
    # run to populate self.compiler
    build_ext.run()
    return build_ext.compiler


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


def _find_file_path(name, locations, prefixes=("",), suffixes=("",)):
    for prefix in prefixes:
        for suffix in suffixes:
            for location in locations:
                path = location / f"{prefix}{name}{suffix}"
                if path.is_file():
                    return str(path)
    return None


# We need to avoid importing numpy until we can be sure it's installed
# This approach is based on this SO answer http://stackoverflow.com/a/21621689
# This is also what pandas does.
class BuildExtensions(build_ext):
    """Subclass setuptools build_ext command

    BuildExtensions does two things
    1) it makes sure numpy is available
    2) it injects numpy's core/include directory in the include_dirs
       parameter of all extensions
    3) it runs the original build_ext command
    """

    def run(self):
        from numpy import get_include

        numpy_incl = get_include()

        for ext in self.extensions:
            if (
                hasattr(ext, "include_dirs")
                and numpy_incl not in ext.include_dirs
            ):
                ext.include_dirs.append(numpy_incl)

        build_ext.run(self)


class DefaultDirs(NamedTuple):
    header: Path | None = None
    library: Path | None = None
    runtime: Path | None = None


def get_default_dirs(conda_prefix: Path | None = None) -> DefaultDirs:
    # Order:
    # * C/CXX flags
    # * conda prefix
    # * hardcoded system paths

    header_dirs = None
    library_dirs = None
    runtime_dirs = None
    if conda_prefix:
        conda_prefix = Path(conda_prefix)

    if os.name == "posix":
        prefixes = ("/usr/local", "/sw", "/opt", "/opt/local", "/usr", "/")
        prefix_paths = [Path(x) for x in prefixes]

        header_dirs = []
        # C/CXX flags
        add_from_path("CPATH", header_dirs)
        add_from_path("C_INCLUDE_PATH", header_dirs)
        add_from_flags("CPPFLAGS", "-I", header_dirs)
        add_from_flags("CFLAGS", "-I", header_dirs)
        # conda prefix
        if conda_prefix:
            header_dirs.append(conda_prefix / "include")
        # hardcoded system paths
        header_dirs.extend(_tree / "include" for _tree in prefix_paths)

        library_dirs = []
        # C/CXX flags
        add_from_flags("LDFLAGS", "-L", library_dirs)
        _archs = ("lib64", "lib")
        # conda prefix
        if conda_prefix:
            library_dirs.extend(conda_prefix / _arch for _arch in _archs)
        # hardcoded system paths
        library_dirs.extend(
            _tree / _arch for _tree in prefix_paths for _arch in _archs
        )
        runtime_dirs = library_dirs

    elif os.name == "nt":
        header_dirs = []  # no default, must be given explicitly
        library_dirs = []  # no default, must be given explicitly
        runtime_dirs = []  # look for DLL files in ``%PATH%``
        if conda_prefix:
            runtime_dirs.append(conda_prefix / "Library")
            header_dirs.append(conda_prefix / "Library" / "include")
        runtime_dirs.extend(
            Path(_path) for _path in os.environ["PATH"].split(";")
        )
        # Add the \Windows\system to the runtime list (necessary for Vista)
        runtime_dirs.append(Path("\\windows\\system"))
        # Add the \path_to_python\DLLs and tables package to the list
        runtime_dirs.append(
            Path(sys.prefix) / "Lib" / "site-packages" / "tables"
        )

    return DefaultDirs(header_dirs, library_dirs, runtime_dirs)


class BasePackage:
    _library_prefixes = []
    _library_suffixes = []
    _runtime_prefixes = []
    _runtime_suffixes = []
    _component_dirs = []

    def __init__(
        self,
        name,
        tag,
        header_name,
        library_name,
        runtime_name,
        target_function=None,
        default_dirs=None,
    ):
        self.name = name
        self.tag = tag
        self.header_name = header_name
        self.library_name = library_name
        self.runtime_name = (
            library_name if runtime_name is None else runtime_name
        )
        self.target_function = target_function
        self.default_dirs = (
            DefaultDirs() if default_dirs is None else default_dirs
        )

    def find_header_path(self, locations):
        return _find_file_path(self.header_name, locations, suffixes=[".h"])

    def find_library_path(self, locations):
        return _find_file_path(
            self.library_name,
            locations,
            self._library_prefixes,
            self._library_suffixes,
        )

    def find_runtime_path(self, locations=None):
        """
        returns True if the runtime can be found
        returns None otherwise
        """
        # An explicit path can not be provided for runtime libraries.
        # (The argument is accepted for compatibility with previous
        # methods.)

        # Try just library name, relying on the OS search path.
        # This fails on macOS.
        # dlopen() won't tell us where the file is, just whether
        # success occurred, so this returns True instead of a filename
        # Try this before the absolute path search below: A Valentino
        for prefix in self._runtime_prefixes:
            for suffix in self._runtime_suffixes:
                try:
                    ctypes.CDLL(f"{prefix}{self.runtime_name}{suffix}")
                except OSError:
                    pass
                else:
                    return True

        # If the library was not found by name alone, try absolute pathnames.
        # In this case, we can return the path, rather than just True.
        for location in locations:
            for prefix in self._runtime_prefixes:
                for suffix in self._runtime_suffixes:
                    abs_path = f"{location}/{prefix}{self.runtime_name}{suffix}"
                    # Debug: print("find_runtime_path() trying ", abs_path)
                          
                    try:
                        ctypes.CDLL(abs_path)
                    except OSError:
                        pass
                    else:
                        # Returning path might foul the logic in caller
                        # return abs_path
                        # Be consistent with name-only search above
                        return True

    def _pkg_config(self, flags):
        try:
            cmd = [PKG_CONFIG] + flags.split() + [self.library_name]
            config = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return []
        else:
            return config.strip().split()

    def find_directories(self, location, use_pkgconfig=False, hook=None):
        dirdata = [
            (
                self.header_name,
                self.find_header_path,
                self.default_dirs.header,
            ),
            (
                self.library_name,
                self.find_library_path,
                self.default_dirs.library,
            ),
            (
                self.runtime_name,
                self.find_runtime_path,
                self.default_dirs.runtime,
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

        hook_dirs = hook() if hook is not None else [None, None, None]

        directories = [None, None, None]  # headers, libraries, runtime
        for idx, (name, find_path, default_dirs) in enumerate(dirdata):
            use_locations = (
                locations
                or pkgconfig_dirs[idx]
                or hook_dirs[idx]
                or default_dirs
            )
            # pkgconfig does not list bin/ as the runtime dir
            if (
                name == "blosc"  # blosc
                and CONDA_PREFIX  # conda
                and sys.platform.startswith("win")  # windows
                and idx == 2  # runtime
                and len(use_locations) == 1  # only one location
                and use_locations[0].parts[-1] == "lib"  # misnamed
            ):
                use_locations = list(use_locations)
                use_locations[0] = use_locations[0].parent / "bin"
                print(f"Patching runtime dir: {str(use_locations[0])}")
            path = find_path(use_locations)
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
            #else:
            #    print("Warning: path is not set.")
        return tuple(directories)


class PosixPackage(BasePackage):
    _library_prefixes = ["lib"]
    _library_suffixes = [".so", ".dylib", ".a"]
    _runtime_prefixes = _library_prefixes
    _runtime_suffixes = [".so", ".dylib"]
    _component_dirs = ["include", "lib", "lib64"]


class WindowsPackage(BasePackage):
    _library_prefixes = ["", "lib"]
    _library_suffixes = [".lib"]
    _runtime_prefixes = ["", "lib"]
    _runtime_suffixes = [".dll"]

    # lookup in '.' seems necessary for LZO2
    _component_dirs = ["include", "lib", "dll", "bin", "."]

    def find_runtime_path(self, locations=None):
        # An explicit path can not be provided for runtime libraries.
        # (The argument is accepted for compatibility with previous
        # methods.)
        return _find_file_path(
            self.runtime_name,
            locations,
            self._runtime_prefixes,
            self._runtime_suffixes,
        )


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent

    # `cpuinfo.py` uses multiprocessing to check CPUID flags. On Windows, the
    # entire setup script needs to be protected as a result
    # For guessing the capabilities of the CPU for C-Blosc
    try:
        import cpuinfo

        cpu_info = cpuinfo.get_cpu_info()
        cpu_flags = cpu_info["flags"]
    except Exception as exc:
        print("cpuinfo failed, assuming no CPU features:", exc)
        cpu_flags = []

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
    min_blosc2_version = _min_versions["min_blosc2_version"]

    # ----------------------------------------------------------------------

    # Allow setting the HDF5 dir and additional link flags either in
    # the environment or on the command line.
    # First check the environment...
    HDF5_DIR = os.environ.get("HDF5_DIR", "")
    LZO_DIR = os.environ.get("LZO_DIR", "")
    BZIP2_DIR = os.environ.get("BZIP2_DIR", "")
    BLOSC_DIR = os.environ.get("BLOSC_DIR", "")
    BLOSC2_DIR = os.environ.get("BLOSC2_DIR", "")
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
    if shutil.which(PKG_CONFIG):
        USE_PKGCONFIG = os.environ.get("USE_PKGCONFIG", "TRUE")
    else:
        USE_PKGCONFIG = "FALSE"

    # ...then the command line.
    # Handle --hdf5=[PATH] --lzo=[PATH] --bzip2=[PATH] --blosc=[PATH]
    # --lflags=[FLAGS] --cflags=[FLAGS] and --debug
    debug = False
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
        elif key == "--blosc2":
            BLOSC2_DIR = Path(val).expanduser()
        elif key == "--lflags":
            LFLAGS = val.split()
        elif key == "--cflags":
            CFLAGS = val.split()
        elif key == "--debug":
            debug = True
            # Don't delete this argument. It may be useful for distutils
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

    if CONDA_PREFIX:
        CONDA_PREFIX = Path(CONDA_PREFIX)
        print(f"* Found conda env: ``{CONDA_PREFIX}``")

    # Global variables
    lib_dirs = []
    inc_dirs = [Path("hdf5-blosc/src"), Path("hdf5-blosc2/src")]
    optional_libs = []
    copy_libs = []

    default_dirs = get_default_dirs(conda_prefix=CONDA_PREFIX)

    # Gcc 4.0.1 on Mac OS X 10.4 does not seem to include the default
    # header and library paths.  See ticket #18.
    if sys.platform.lower().startswith("darwin") or CONDA_PREFIX:
        inc_dirs.extend(default_dirs.header)
        lib_dirs.extend(default_dirs.library)

    if os.name == "posix":
        _Package = PosixPackage
        _platdep = {  # package tag -> platform-dependent components
            "HDF5": ["hdf5", None],
            "LZO2": ["lzo2", None],
            "LZO": ["lzo", None],
            "BZ2": ["bz2", None],
            "BLOSC": ["blosc", None],
            "BLOSC2": ["blosc2", None],
        }

    elif os.name == "nt":
        _Package = WindowsPackage
        _platdep = {  # package tag -> platform-dependent components
            # Library name, runtime name
            "HDF5": ["hdf5", "hdf5"],
            "LZO2": ["lzo2", "lzo2"],
            "LZO": ["liblzo", "lzo1"],
            "BZ2": ["bz2", "bz2"],
            "BLOSC": ["blosc", "blosc"],
            "BLOSC2": ["blosc2", "blosc2"],
        }
        # conda recently got rid of bz2.dll in favor of libbz2.dll
        # https://github.com/conda-forge/bzip2-feedstock/pull/24#pullrequestreview-2178290795
        if CONDA_PREFIX:
            _platdep["BZ2"] = ["libbz2", "libbz2"]

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

    hdf5_package = _Package(
        name="HDF5",
        tag="HDF5",
        header_name="H5public",
        library_name=_platdep["HDF5"][0],
        runtime_name=_platdep["HDF5"][1],
        target_function="H5close",
        default_dirs=default_dirs,
    )
    lzo2_package = _Package(
        name="LZO 2",
        tag="LZO2",
        header_name="lzo/lzo1x",
        library_name=_platdep["LZO2"][0],
        runtime_name=_platdep["LZO2"][1],
        target_function="lzo_version_date",
        default_dirs=default_dirs,
    )
    lzo1_package = _Package(
        name="LZO 1",
        tag="LZO",
        header_name="lzo1x",
        library_name=_platdep["LZO"][0],
        runtime_name=_platdep["LZO"][1],
        target_function="lzo_version_date",
        default_dirs=default_dirs,
    )
    bzip2_package = _Package(
        name="bzip2",
        tag="BZ2",
        header_name="bzlib",
        library_name=_platdep["BZ2"][0],
        runtime_name=_platdep["BZ2"][1],
        target_function="BZ2_bzlibVersion",
        default_dirs=default_dirs,
    )
    blosc_package = _Package(
        name="blosc",
        tag="BLOSC",
        header_name="blosc",
        library_name=_platdep["BLOSC"][0],
        runtime_name=_platdep["BLOSC"][1],
        target_function="blosc_list_compressors",
        default_dirs=default_dirs,
    )
    blosc2_package = _Package(
        name="blosc2",
        tag="BLOSC2",
        header_name="blosc2",
        library_name=_platdep["BLOSC2"][0],
        runtime_name=_platdep["BLOSC2"][1],
        target_function="blosc2_list_compressors",
        default_dirs=default_dirs,
    )

    # -----------------------------------------------------------------

    def_macros = [
        ("NDEBUG", 1),
        ("NPY_TARGET_VERSION", "NPY_1_20_API_VERSION"),
    ]

    # Define macros for Windows platform
    if os.name == "nt":
        if debug:
            # For debugging (mainly compression filters)
            # to prevent including dlfcn.h by utils.c!!!
            def_macros = [("DEBUG", 1)]

        def_macros.append(("WIN32", 1))
        def_macros.append(("_HDF5USEDLL_", 1))
        def_macros.append(("H5_BUILT_AS_DYNAMIC_LIB", 1))

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
        # Like 'C:\\Program Files\\HDF Group\\HDF5\\1.10.2\\bin\\hdf5dll.dll'
        if libdir:
            # Strip off the filename and the 'bin' directory
            HDF5_DIR = Path(libdir).parent.parent
            print(f"* Found HDF5 using system PATH ('{libdir}')")

    # The next flag for the C compiler is needed for finding the C headers for
    # the Cython extensions
    CFLAGS.append("-Isrc")

    # Force the 1.10.x HDF5 API even if the library as been compiled to use the
    # 1.6.x API by default
    CFLAGS.extend(
        [
            "-DH5_USE_110_API",
            "-DH5Rdereference_vers=2",
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

    for package, location in [
        (hdf5_package, HDF5_DIR),
        (lzo2_package, LZO_DIR),
        (lzo1_package, LZO_DIR),
        (bzip2_package, BZIP2_DIR),
        (blosc_package, BLOSC_DIR),
        (blosc2_package, BLOSC2_DIR),
    ]:
        if package.tag == "LZO" and lzo2_enabled:
            print(
                f"* Skipping detection of {lzo1_package.name} "
                f"since {lzo2_package.name} has already been found."
            )
            continue  # do not use LZO 1 if LZO 2 is available

        # looking for lzo/lzo1x.h but pkgconfig already returns
        # '/usr/include/lzo'
        use_pkgconfig = USE_PKGCONFIG and package.tag != "LZO2"

        hook = None
        if package.tag == "BLOSC2":
            hook = blosc2_find_directories_hook

        (hdrdir, libdir, rundir) = package.find_directories(
            location, use_pkgconfig=use_pkgconfig, hook=hook
        )

        # check if HDF5 library uses old DLL naming scheme
        if hdrdir and package.tag == "HDF5":
            hdf5_version = get_hdf5_version(Path(hdrdir) / "H5public.h")
            if hdf5_version < min_hdf5_version:
                exit_with_error(
                    f"Unsupported HDF5 version! HDF5 v{min_hdf5_version}+ "
                    f"required. Found version v{hdf5_version}"
                )

        # check if the library is in the standard compiler paths
        if not libdir and package.target_function:
            libdir = compiler.has_function(
                package.target_function,
                includes=(package.header_name + ".h",),
                libraries=(package.library_name,),
                include_dirs=[str(d) for d in inc_dirs],
                library_dirs=[str(d) for d in lib_dirs],
            )

        if not (hdrdir and libdir):
            err_msg = (
                f"Could not find a local {package.name} installation headers "
                f"({hdrdir=}) and/or libdir ({libdir=})"
            )
            if package.tag in ["HDF5", "BLOSC2"]:  # these are compulsory!
                pname, ptag = package.name, package.tag
                exit_with_error(
                    f"{err_msg}. ",
                    f"You may need to explicitly state where your local "
                    f"{pname} headers and library can be found by setting "
                    f"the ``{ptag}_DIR`` environment variable or by using "
                    f"the ``--{ptag.lower()}`` command-line option.",
                )
            elif package.tag == "BLOSC":
                # this is optional, but comes with sources
                print(f"* {err_msg}; using internal sources.")
            else:
                print(f"* {err_msg}; disabling support for it.")

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

        if hdrdir not in default_dirs.header:
            inc_dirs.insert(0, Path(hdrdir))  # save header directory if needed
        if libdir not in default_dirs.library and libdir not in ("", True):
            # save library directory if needed
            lib_dirs.insert(0, Path(libdir))

        if package.tag not in ["HDF5"]:
            # Keep record of the optional libraries found.
            optional_libs.append(package.tag)
            def_macros.append((f"HAVE_{package.tag}_LIB", 1))

        if hdrdir and package.tag == "BLOSC":
            blosc_version = get_blosc_version(Path(hdrdir) / "blosc.h")
            if blosc_version < min_blosc_version:
                optional_libs.pop()
                print_warning(
                    f"Unsupported Blosc version installed! Blosc "
                    f"{min_blosc_version}+ required. Found version "
                    f"{blosc_version}.  Using internal Blosc sources."
                )

        if hdrdir and package.tag == "BLOSC2":
            blosc2_version = get_blosc2_version(Path(hdrdir) / "blosc2.h")
            if blosc2_version < min_blosc2_version:
                optional_libs.pop()
                print_warning(
                    f"Unsupported Blosc2 version installed! Blosc2 "
                    f"{min_blosc2_version}+ required. Found version "
                    f"{blosc2_version}.  "
                    f"Update it via `pip install blosc2 -U`."
                )

        if not rundir:
            loc = {
                "posix": "the default library paths",
                "nt": "any of the directories in %%PATH%%",
            }[os.name]

            if package.name == "blosc2":
                # We will copy this into the tables directory
                print(
                    "  * Copying blosc2 runtime library to 'tables' dir"
                    " because it was not found in standard locations"
                )
                platform_system = platform.system()
                if platform_system == "Linux":
                    shutil.copy(libdir / "libblosc2.so", ROOT / "tables")
                    copy_libs += ["libblosc2.so"]

                    dll_dir = "/tmp/hdf5/lib"
                    # Copy dlls when producing the wheels in CI
                    if "bdist_wheel" in sys.argv and os.path.exists(dll_dir):
                        shared_libs = glob.glob(str(libdir) + "/libblosc2.so*")
                        for lib in shared_libs:
                            shutil.copy(lib, dll_dir)

                elif platform_system == "Darwin":
                    shutil.copy(libdir / "libblosc2.dylib", ROOT / "tables")
                    copy_libs += ["libblosc2.dylib"]

                    dll_dir = "/tmp/hdf5/lib"
                    # Copy dlls when producing the wheels in CI
                    if "bdist_wheel" in sys.argv and os.path.exists(dll_dir):
                        shared_libs = glob.glob(
                            str(libdir) + "/libblosc2*.dylib"
                        )
                        for lib in shared_libs:
                            shutil.copy(lib, dll_dir)

                else:
                    copy_libs += ["libblosc2.dll"]
                    dll_dir = f"{CONDA_PREFIX}\\Library\\bin"
                    # Copy dlls when producing the wheels in CI
                    if "bdist_wheel" in sys.argv and os.path.exists(dll_dir):
                        # If building a wheel and Conda in use, these could be the
                        # same path. Indicative that the ``rundir`` check is probably
                        # not sufficient above, but easy enough to guard against
                        # trying to copy a file to its current location
                        # (thereby avoiding a "are the same file" shutil error).
                        if Path(dll_dir) != libdir.parent / "bin":
                            shutil.copy(
                                libdir.parent / "bin" / "libblosc2.dll",
                                dll_dir,
                            )
                    else:
                        shutil.copy(
                            libdir.parent / "bin" / "libblosc2.dll", "tables"
                        )
            else:
                if "bdist_wheel" in sys.argv and os.name == "nt":
                    exit_with_error(
                        f"Could not find the {package.name} runtime.",
                        f"The {package.name} shared library was *not* found "
                        f"in {loc}. Cannot build wheel without the runtime.",
                    )

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
        extdir = Path("tables")
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

    # --------------------------------------------------------------------
    if os.name == "nt" and COPY_DLLS:
        for dll_file in dll_files:
            shutil.copy(dll_file, "tables")

    ADDLIBS = [hdf5_package.library_name]

    # List of Blosc file dependencies
    blosc_path = Path("c-blosc/blosc")
    int_complibs_path = Path("c-blosc/internal-complibs")

    blosc_sources = [Path("hdf5-blosc/src/blosc_filter.c")]
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
        def_macros += [
            ("HAVE_LZ4", 1),
            ("HAVE_ZLIB", 1),
            ("HAVE_ZSTD", 1),
            ("ZSTD_DISABLE_ASM", 1),
        ]

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

        # Set flags for SSE2 and AVX2 preventing false detection in case
        # of emulation
        if platform.machine() not in ("aarch64", "ARM64"):
            # SSE2
            if "sse2" in cpu_flags and "DISABLE_SSE2" not in os.environ:
                print("* SSE2 detected and enabled")
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
                print("* AVX2 detected and enabled")
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

    if "BLOSC2" in optional_libs:
        blosc_sources += [Path("hdf5-blosc2/src/blosc2_filter.c")]
        ADDLIBS += ["blosc2"]
    else:
        exit_with_error("Unable to find the blosc2 library.")

    utilsextension_libs = LIBS + ADDLIBS
    hdf5extension_libs = LIBS + ADDLIBS
    tableextension_libs = LIBS + ADDLIBS
    linkextension_libs = LIBS + ADDLIBS
    indexesextension_libs = LIBS + ADDLIBS
    lrucacheextension_libs = []  # Doesn't need external libraries

    # Compressor modules only need other libraries if they are enabled.
    _comp_lzo_libs = LIBS[:]
    _comp_bzip2_libs = LIBS[:]
    for package, complibs in [
        (lzo_package, _comp_lzo_libs),
        (bzip2_package, _comp_bzip2_libs),
    ]:
        if package.tag in optional_libs:
            complibs.extend([hdf5_package.library_name, package.library_name])

    # Extension expects strings, so we have to convert Path to str
    # We remove duplicates too
    blosc_sources = list({str(x) for x in blosc_sources})
    inc_dirs = list({str(x) for x in inc_dirs})
    lib_dirs = list({str(x) for x in lib_dirs})

    extension_kwargs = {
        "extra_compile_args": CFLAGS,
        "extra_link_args": LFLAGS,
        "library_dirs": lib_dirs,
        "define_macros": def_macros,
        "include_dirs": inc_dirs,
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
            libraries=utilsextension_libs,
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
            libraries=hdf5extension_libs,
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
            libraries=tableextension_libs,
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
            libraries=tableextension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.lrucacheextension",
            sources=[str(cython_extfiles["lrucacheextension"])],
            libraries=lrucacheextension_libs,
            **extension_kwargs,
        ),
        Extension(
            "tables.indexesextension",
            sources=[
                str(cython_extfiles["indexesextension"]),
                "src/utils.c",
                "src/H5ARRAY-opt.c",
                "src/idx-opt.c",
            ]
            + blosc_sources,
            libraries=indexesextension_libs,
            **extension_kwargs,
        ),
    ]

    setup(
        ext_modules=extensions,
        cmdclass={"build_ext": BuildExtensions},
    )
