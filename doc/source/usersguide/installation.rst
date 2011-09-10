Installation
============
.. epigraph::

    Make things as simple as possible, but not any simpler.

    -- Albert Einstein


The Python Distutils are used to build and install PyTables, so it is fairly
simple to get the application up and running. If you want to install the
package from sources you can go on reading to the next section.

However, if you are running Windows and want to install precompiled binaries,
you can jump straight to :ref:`binaryInstallationDescr`. In addition, binary
packages are available for many different Linux distributions, MacOSX and
other Unices.  Just check the package repository for your preferred operating
system.

Installation from source
------------------------

These instructions are for both Unix/MacOS X and Windows systems. If you are
using Windows, it is assumed that you have a recent version of MS Visual C++
compiler installed.
A GCC compiler is assumed for Unix, but other compilers should work as well.

Extensions in PyTables have been developed in Cython (see
:ref:`[CYTHON] <CYTHON>`) and the C language. You can rebuild everything from
scratch if you have Cython installed, but this is not necessary, as the Cython
compiled source is included in the source distribution.

To compile PyTables you will need a recent version of Python, the HDF5 (C
flavor) library from http://hdfgroup.org, and the NumPy (see
:ref:`[NUMPY] <NUMPY>`) and Numexpr (see  :ref:`[NUMEXPR] <NUMEXPR>`)
packages.
Although you won't need numarray (see :ref:`[NUMARRAY] <NUMARRAY>`) or Numeric
(see :ref:`[NUMERIC] <NUMERIC>`) in order to compile PyTables, they are
supported; you only need a reasonably recent version of them (>= 1.5.2 for
numarray and >= 24.2 for Numeric) if you plan on using them in your
applications. If you already have numarray and/or Numeric installed, the test
driver module will detect them and will run the tests for numarray and/or
Numeric automatically.

.. warning:: The use of numarray and Numeric in PyTables is now deprecated.
   Support for these packages will be dropped in future versions.

Prerequisites
~~~~~~~~~~~~~

First, make sure that you have

* Python >= 2.4 (Python 3.x is not supported currently),
* HDF5 >= 1.6.10,
* NumPy >= 1.4.1,
* Numexpr >= 1.4.1 and
* Cython >= 0.13

installed (for testing purposes, we are using HDF5 1.6.10/1.8.5, NumPy 1.5 and
Numexpr 1.4.1 currently). If you don't, fetch and install them before
proceeding.

.. note:: Currently PyTables does not use setuptools_ so do not expect that
          the setup.py script automatically install all packages PyTables
          depends on.

.. _setuptools: http://pypi.python.org/pypi/setuptools

Compile and install these packages (but see :ref:`prerequisitesBinInst` for
instructions on how to install precompiled binaries if you are not willing to
compile the prerequisites on Windows systems).

For compression (and possibly improved performance), you will need to install
the Zlib (see :ref:`[ZLIB] <ZLIB>`), which is also required by HDF5 as well.
You may also optionally install the excellent LZO compression library (see
:ref:`[LZO] <LZO>` and :ref:`compressionIssues`). The high-performance bzip2
compression library can also be used with PyTables (see
:ref:`[BZIP2] <BZIP2>`).
The Blosc (see :ref:`[BLOSC] <BLOSC>`) compression library is embedded in
PyTables, so you don't need to install it separately.

**Unix**

    setup.py will detect HDF5, LZO, or bzip2 libraries and include files under
    :file:`/usr` or :file:`/usr/local`; this will cover most manual
    installations as well as installations from packages.
    If setup.py can not find libhdf5, libhdf5 (or liblzo, or libbz2 that you
    may wish to use) or if you have several versions of a library installed
    and want to use a particular one, then you can set the path to the
    resource in the environment, by setting the values of the
    :envvar:`HDF5_DIR`, :envvar:`LZO_DIR`, or :envvar:`BZIP2_DIR` environment
    variables to the path to the particular resource. You may also specify the
    locations of the resource root directories on the setup.py command line.
    For example::

        --hdf5=/stuff/hdf5-1.8.5
        --lzo=/stuff/lzo-2.02
        --bzip2=/stuff/bzip2-1.0.5

    If your HDF5 library was built as a shared library not in the runtime load
    path, then you can specify the additional linker flags needed to find the
    shared library on the command line as well. For example::

        --lflags="-Xlinker -rpath -Xlinker /stuff/hdf5-1.8.5/lib"

    You may also want to try setting the :envvar:`LD_LIBRARY_PATH`
    environment variable to point to the directory where the shared libraries
    can be found. Check your compiler and linker documentation as well as the
    Python Distutils documentation for the correct syntax or environment
    variable names.
    It is also possible to link with specific libraries by setting the
    :envvar:`LIBS` environment variable::

        LIBS="hdf5-1.8.5 nsl"

    Finally, you can give additional flags to your compiler by passing them to
    the :option:`--cflags` flag::

        --cflags="-w -O3 -msse2"

    In the above case, a gcc compiler is used and you instructed it to
    suppress all the warnings and set the level 3 of optimization.
    Finally, if you are running Linux in 32-bit mode, and you know that your
    CPU has support for SSE2 vector instructions, you may want to pass the
    :option:`-msse2` flag that will accelerate Blosc operation.

**Windows**

    You can get ready-to-use Windows binaries and other development files for
    most of the following libraries from the GnuWin32 project (see
    :ref:`[GNUWIN32] <GNUWIN32>`).  In case you cannot find the LZO binaries
    in the GnuWin32 repository, you can find them at
    http://sourceforge.net/projects/pytables/files/lzo-win.
    Once you have installed the prerequisites, setup.py needs to know where
    the necessary library *stub* (.lib) and *header* (.h) files are installed.
    You can set the path to the include and dll directories for the HDF5
    (mandatory) and LZO or BZIP2 (optional) libraries in the environment, by
    setting the values of the :envvar:`HDF5_DIR`, :envvar:`LZO_DIR`, or
    :envvar:`BZIP2_DIR` environment variables to the path to the particular
    resource.  For example::

        set HDF5_DIR=c:\\stuff\\hdf5-1.8.5-32bit-VS2008-IVF101\\release
        set LZO_DIR=c:\\Program Files (x86)\\GnuWin32
        set BZIP2_DIR=c:\\Program Files (x86)\\GnuWin32

    You may also specify the locations of the resource root directories on the
    setup.py command line.
    For example::

        --hdf5=c:\\stuff\\hdf5-1.8.5-32bit-VS2008-IVF101\\release
        --lzo=c:\\Program Files (x86)\\GnuWin32
        --bzip2=c:\\Program Files (x86)\\GnuWin32

**Development version (Unix)**

    Installation of the development version is very similar to installation
    from a source package (described above).  There are two main differences:

    #. sources have to be downloaded from the `PyTables source repository`_
       hosted on GitHub_. Git (see :ref:`[GIT] <GIT>`) is used as VCS.
       The following command create a local copy of latest development version
       sources::

        $ git clone https://github.com/PyTables/PyTables.git

    #. sources in the git repository do not include pre-built documentation
       and pre-generated C code of Cython extension modules.  To be able to
       generate them, both Cython (see :ref:`[CYTHON] <CYTHON>`) and
       sphinx >= 1.0.7 (see :ref:`[SPHINX] <SPHINX>`) are mandatory
       prerequisites.

.. _`PyTables source repository`: https://github.com/PyTables/PyTables
.. _GitHub: http://www.github.com


PyTables package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have installed the HDF5 library and the NumPy and Numexpr packages,
you can proceed with the PyTables package itself.

#. Run this command from the main PyTables distribution directory, including
   any extra command line arguments as discussed above::

      python setup.py build_ext --inplace

#. To run the test suite, execute any of these commands.

   **Unix**
      In the sh shell and its variants::

          PYTHONPATH=.:$PYTHONPATH  python tables/tests/test_all.py

      or, if you prefer::

          PYTHONPATH=.:$PYTHONPATH  python -c "import tables; tables.test()"

   **Windows**

      Open the command prompt (cmd.exe or command.com) and type::

          set PYTHONPATH=.;%PYTHONPATH%
          python tables\\tests\\test_all.py

      or::

          set PYTHONPATH=.;%PYTHONPATH%
          python -c "import tables; tables.test()"

   Both commands do the same thing, but the latter still works on an already
   installed PyTables (so, there is no need to set the PYTHONPATH variable for
   this case).  However, before installation, the former is recommended
   because it is more flexible, as you can see below.
   If you would like to see verbose output from the tests simply add the
   :option:`-v` flag and/or the word verbose to the first of the command lines
   above. You can also run only the tests in a particular test module.
   For example, to execute just the test_types test suite, you only have to
   specify it::

      python tables/tests/test_types.py -v  # change to backslashes for win

   You have other options to pass to the :file:`test_all.py` driver::

      python tables/tests/test_all.py --heavy  # change to backslashes for win

   The command above runs every test in the test unit. Beware, it can take a
   lot of time, CPU and memory resources to complete::

      python tables/tests/test_all.py --print-versions  # change to backslashes for win

   The command above shows the versions for all the packages that PyTables
   relies on. Please be sure to include this when reporting bugs::

      python tables/tests/test_all.py --show-memory  # only under Linux 2.6.x

   The command above prints out the evolution of the memory consumption after
   each test module completion. It's useful for locating memory leaks in
   PyTables (or packages behind it). Only valid for Linux 2.6.x kernels.
   And last, but not least, in case a test fails, please run the failing test
   module again and enable the verbose output::

      python tables/tests/test_<module>.py -v verbose

   and, very important, obtain your PyTables version information by using the
   :option:`--print-versions` flag (see above) and send back both outputs to
   developers so that we may continue improving PyTables.
   If you run into problems because Python can not load the HDF5 library or
   other shared libraries.

   **Unix**

      Try setting the LD_LIBRARY_PATH or equivalent environment variable to
      point to the directory where the missing libraries can be found.

   **Windows**

      Put the DLL libraries (hdf5dll.dll and, optionally, lzo1.dll and
      bzip2.dll) in a directory listed in your :envvar:`PATH` environment
      variable. The setup.py installation program will print out a warning to
      that effect if the libraries can not be found.

#. To install the entire PyTables Python package, change back to the root
   distribution directory and run the following command (make sure you have
   sufficient permissions to write to the directories where the PyTables files
   will be installed)::

      python setup.py install

   Of course, you will need super-user privileges if you want to install
   PyTables on a system-protected area. You can select, though, a different
   place to install the package using the :option:`--prefix` flag::

      python setup.py install --prefix="/home/myuser/mystuff"

   Have in mind, however, that if you use the :option:`--prefix` flag to
   install in a non-standard place, you should properly setup your
   :envvar:`PYTHONPATH` environment variable, so that the Python interpreter
   would be able to find your new PyTables installation.
   You have more installation options available in the Distutils package.
   Issue a::

      python setup.py install --help

   for more information on that subject.

That's it! Now you can skip to the next chapter to learn how to use PyTables.


.. _binaryInstallationDescr:

Binary installation (Windows)
-----------------------------

This section is intended for installing precompiled binaries on Windows
platforms. You may also find it useful for instructions on how to install
*binary prerequisites* even if you want to compile PyTables itself on Windows.

.. warning:: Since PyTables 2.2b3, Windows binaries are distributed with
   SSE2 instructions enabled.  If your processor does not have support
   for SSE2, then you will not be able to use these binaries.

.. _prerequisitesBinInst:

Windows prerequisites
~~~~~~~~~~~~~~~~~~~~~

First, make sure that you have Python 2.4, NumPy 1.4.1 and Numexpr 1.4.1 or
higher installed (PyTables binaries have been built using NumPy 1.5 and
Numexpr 1.4.1).  The binaries already include DLLs for HDF5 (1.6.10, 1.8.5),
zlib1 (1.2.3), szlib (2.0, uncompression support only) and bzip2 (1.0.5) for
Windows (2.8.0).
The LZO DLL can't be included because of license issues (but read below for
directives to install it if you want so).

To enable compression with the optional LZO library (see the
:ref:`compressionIssues` for hints about how it may be used to improve
performance), fetch and install the LZO from
http://sourceforge.net/projects/pytables/files/lzo-win (choose v1.x for
Windows 32-bit and v2.x for Windows 64-bit).
Normally, you will only need to fetch that package and copy the included
lzo1.dll/lzo2.dll file in a directory in the PATH environment variable
(for example C:\\WINDOWS\\SYSTEM) or
python_installation_path\\Lib\\site-packages\\tables (the last directory may
not exist yet, so if you want to install the DLL there, you should do so
*after* installing the PyTables package), so that it can be found by the
PyTables extensions.

Please note that PyTables has internal machinery for dealing with uninstalled
optional compression libraries, so, you don't need to install the LZO dynamic
library if you don't want to.

PyTables package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the tables-<version>.win32-py<version>.exe file and execute it.

You can (and *you should*) test your installation by running the next
commands::

    >>> import tables
    >>> tables.test()

on your favorite python shell. If all the tests pass (possibly with a few
warnings, related to the potential unavailability of LZO lib) you already have
a working, well-tested copy of PyTables installed! If any test fails, please
copy the output of the error messages as well as the output of::

    >>> tables.print_versions()

and mail them to the developers so that the problem can be fixed in future
releases.

You can proceed now to the next chapter to see how to use PyTables.

