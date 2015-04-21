Installation
============

.. epigraph::

    Make things as simple as possible, but not any simpler.

    -- Albert Einstein


The Python Distutils are used to build and install PyTables, so it is fairly
simple to get the application up and running. If you want to install the
package from sources you can go on reading to the next section.

However, if you want to go straight to binaries that 'just work' for the main
platforms (Linux, Mac OSX and Windows), you might want to use the excellent
Anaconda_ or Canopy_ distributions.  PyTables usually distributes its own
Windows binaries too; go :ref:`binaryInstallationDescr` for instructions.
Finally `Christoph Gohlke`_ also maintains an excellent suite of a variety of
binary packages for Windows at his site.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Canopy: https://www.enthought.com/products/canopy/
.. _`Christoph Gohlke`: http://www.lfd.uci.edu/~gohlke/pythonlibs/


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
flavor) library from http://www.hdfgroup.org, and the NumPy (see
:ref:`[NUMPY] <NUMPY>`) and Numexpr (see :ref:`[NUMEXPR] <NUMEXPR>`)
packages.


Prerequisites
~~~~~~~~~~~~~

First, make sure that you have

* Python_ >= 2.6 including Python 3.x
* HDF5_ >= 1.8.4 (>=1.8.7 is strongly recommended)
* NumPy_ >= 1.7.1
* Numexpr_ >= 2.4
* Cython_ >= 0.14
* c-blosc_ >= 1.4.1 (it is bundled with PyTables sources but the user can
  use an external version of sources using the :envvar:`BLOSC_DIR` environment
  variable or the :option:`--blosc` flag of the :file:`setup.py`)
* argparse_ (only Python 2.6, it is used by the :program:`pt2to3` utility)
* unittest2_ (only Python 2.6)

installed (for testing purposes, we are using HDF5_ 1.8.12, NumPy_ 1.8.0
and Numexpr_ 2.4.1 currently). If you don't, fetch and install them before
proceeding.

.. _Python: http://www.python.org
.. _HDF5: http://www.hdfgroup.org/HDF5
.. _NumPy: http://www.numpy.org
.. _Numexpr: http://code.google.com/p/numexpr
.. _Cython: http://www.cython.org
.. _c-blosc: http://blosc.org
.. _argparse: http://code.google.com/p/argparse
.. _unittest2: http://pypi.python.org/pypi/unittest2

.. note::

    Currently PyTables does not use setuptools_ by default so do not expect
    that the setup.py script automatically install all packages PyTables
    depends on.

.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _ctypes: https://pypi.python.org/pypi/ctypes

Compile and install these packages (but see :ref:`prerequisitesBinInst` for
instructions on how to install pre-compiled binaries if you are not willing
to compile the prerequisites on Windows systems).

For compression (and possibly improved performance), you will need to install
the Zlib (see :ref:`[ZLIB] <ZLIB>`), which is also required by HDF5 as well.
You may also optionally install the excellent LZO compression library (see
:ref:`[LZO] <LZO>` and :ref:`compressionIssues`). The high-performance bzip2
compression library can also be used with PyTables (see
:ref:`[BZIP2] <BZIP2>`).

The Blosc (see :ref:`[BLOSC] <BLOSC>`) compression library is embedded
in PyTables, so this will be used in case it is not found in the
system.  So, in case the installer warns about not finding it, do not
worry too much ;)

**Unix**

    setup.py will detect HDF5, LZO, or bzip2 libraries and include files under
    :file:`/usr` or :file:`/usr/local`; this will cover most manual
    installations as well as installations from packages.
    If setup.py can not find libhdf5, libhdf5 (or liblzo, or libbz2 that you
    may wish to use) or if you have several versions of a library installed
    and want to use a particular one, then you can set the path to the
    resource in the environment, by setting the values of the
    :envvar:`HDF5_DIR`, :envvar:`LZO_DIR`, :envvar:`BZIP2_DIR` or
    :envvar:`BLOSC_DIR` environment variables to the path to the particular
    resource. You may also specify the locations of the resource root
    directories on the setup.py command line.  For example::

        --hdf5=/stuff/hdf5-1.8.12
        --lzo=/stuff/lzo-2.02
        --bzip2=/stuff/bzip2-1.0.5
        --blosc=/stuff/blosc-1.3.2

    If your HDF5 library was built as a shared library not in the runtime load
    path, then you can specify the additional linker flags needed to find the
    shared library on the command line as well. For example::

        --lflags="-Xlinker -rpath -Xlinker /stuff/hdf5-1.8.12/lib"

    You may also want to try setting the :envvar:`LD_LIBRARY_PATH`
    environment variable to point to the directory where the shared libraries
    can be found. Check your compiler and linker documentation as well as the
    Python Distutils documentation for the correct syntax or environment
    variable names.
    It is also possible to link with specific libraries by setting the
    :envvar:`LIBS` environment variable::

        LIBS="hdf5-1.8.12 nsl"

    Finally, you can give additional flags to your compiler by passing them to
    the :option:`--cflags` flag::

        --cflags="-w -O3 -msse2"

    In the above case, a gcc compiler is used and you instructed it to
    suppress all the warnings and set the level 3 of optimization.
    Finally, if you are running Linux in 32-bit mode, and you know that your
    CPU has support for SSE2 vector instructions, you may want to pass the
    :option:`-msse2` flag that will accelerate Blosc operation.

    .. hint::

        some GNU/Linux distributions provide a packaged version of the HDF5
        libraries with MPI support.  In this case you may need to specify
        the path of the MPH headers as additional include directory.

        On Ubuntu 12.04 the following command has been reported to work::

          $ C_INCLUDE_PATH=/usr/lib/openmpi/include pip install --upgrade tables

**Windows**

    You can get ready-to-use Windows binaries and other development files for
    most of the following libraries from the GnuWin32 project (see
    :ref:`[GNUWIN32] <GNUWIN32>`).  In case you cannot find the LZO binaries
    in the GnuWin32 repository, you can find them at
    http://sourceforge.net/projects/pytables/files/lzo-win.
    Once you have installed the prerequisites, setup.py needs to know where
    the necessary library *stub* (.lib) and *header* (.h) files are installed.
    You can set the path to the include and dll directories for the HDF5
    (mandatory) and LZO, BZIP2, BLOSC (optional) libraries in the environment,
    by setting the values of the :envvar:`HDF5_DIR`, :envvar:`LZO_DIR`,
    :envvar:`BZIP2_DIR` or :envvar:`BLOSC_DIR` environment variables to the
    path to the particular resource.  For example::

        set HDF5_DIR=c:\\stuff\\hdf5-1.8.5-32bit-VS2008-IVF101\\release
        set LZO_DIR=c:\\Program Files (x86)\\GnuWin32
        set BZIP2_DIR=c:\\Program Files (x86)\\GnuWin32
        set BLOSC_DIR=c:\\Program Files (x86)\\Blosc

    You may also specify the locations of the resource root directories on the
    setup.py command line.
    For example::

        --hdf5=c:\\stuff\\hdf5-1.8.5-32bit-VS2008-IVF101\\release
        --lzo=c:\\Program Files (x86)\\GnuWin32
        --bzip2=c:\\Program Files (x86)\\GnuWin32
        --blosc=c:\\Program Files (x86)\\Blosc

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
.. _GitHub: https://github.com


PyTables package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have installed the HDF5 library and the NumPy and Numexpr packages,
you can proceed with the PyTables package itself.

#. Run this command from the main PyTables distribution directory, including
   any extra command line arguments as discussed above::

      $ python setup.py build

#. To run the test suite, execute any of these commands.

   **Unix**
      In the sh shell and its variants::

        $ cd build/lib.linux-x86_64-3.3
        $ env PYTHONPATH=. python tables/tests/test_all.py

      or, if you prefer::

        $ cd build/lib.linux-x86_64-3.3
        $ env PYTHONPATH=. python -c "import tables; tables.test()"

      .. note::

          the syntax used above overrides original contents of the
          :envvar:`PYTHONPATH` environment variable.
          If this is not the desired behaviour and the user just wants to add
          some path before existing ones, then the safest syntax to use is
          the following::

            $ env PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} python tables/tests/test_all.py

          Please refer to your :program:`sh` documentation for details.

   **Windows**

      Open the command prompt (cmd.exe or command.com) and type::

        > cd build\\lib.linux-x86_64-2.7
        > set PYTHONPATH=.;%PYTHONPATH%
        > python tables\\tests\\test_all.py

      or::

        > cd build\\lib.linux-x86_64-2.7
        > set PYTHONPATH=.;%PYTHONPATH%
        > python -c "import tables; tables.test()"

   Both commands do the same thing, but the latter still works on an already
   installed PyTables (so, there is no need to set the :envvar:`PYTHONPATH`
   variable for this case).
   However, before installation, the former is recommended because it is
   more flexible, as you can see below.
   If you would like to see verbose output from the tests simply add the
   :option:`-v` flag and/or the word verbose to the first of the command lines
   above. You can also run only the tests in a particular test module.
   For example, to execute just the test_types test suite, you only have to
   specify it::

      # change to backslashes for win
      $ python tables/tests/test_types.py -v

   You have other options to pass to the :file:`test_all.py` driver::

      # change to backslashes for win
      $ python tables/tests/test_all.py --heavy

   The command above runs every test in the test unit. Beware, it can take a
   lot of time, CPU and memory resources to complete::

      # change to backslashes for win
      $ python tables/tests/test_all.py --print-versions

   The command above shows the versions for all the packages that PyTables
   relies on. Please be sure to include this when reporting bugs::

      # only under Linux 2.6.x
      $ python tables/tests/test_all.py --show-memory

   The command above prints out the evolution of the memory consumption after
   each test module completion. It's useful for locating memory leaks in
   PyTables (or packages behind it). Only valid for Linux 2.6.x kernels.
   And last, but not least, in case a test fails, please run the failing test
   module again and enable the verbose output::

      $ python tables/tests/test_<module>.py -v verbose

   and, very important, obtain your PyTables version information by using the
   :option:`--print-versions` flag (see above) and send back both outputs to
   developers so that we may continue improving PyTables.
   If you run into problems because Python can not load the HDF5 library or
   other shared libraries.

   **Unix**

      Try setting the LD_LIBRARY_PATH or equivalent environment variable to
      point to the directory where the missing libraries can be found.

   **Windows**

      Put the DLL libraries (hdf5dll.dll and, optionally, lzo1.dll,
      bzip2.dll or blosc.dll) in a directory listed in your
      :envvar:`PATH` environment variable. The setup.py installation
      program will print out a warning to that effect if the libraries
      can not be found.

#. To install the entire PyTables Python package, change back to the root
   distribution directory and run the following command (make sure you have
   sufficient permissions to write to the directories where the PyTables files
   will be installed)::

      $ python setup.py install

   Of course, you will need super-user privileges if you want to install
   PyTables on a system-protected area. You can select, though, a different
   place to install the package using the :option:`--prefix` flag::

      $ python setup.py install --prefix="/home/myuser/mystuff"

   Have in mind, however, that if you use the :option:`--prefix` flag to
   install in a non-standard place, you should properly setup your
   :envvar:`PYTHONPATH` environment variable, so that the Python interpreter
   would be able to find your new PyTables installation.
   You have more installation options available in the Distutils package.
   Issue a::

      $ python setup.py install --help

   for more information on that subject.

That's it! Now you can skip to the next chapter to learn how to use PyTables.


Installation with :program:`pip`
--------------------------------

Many users find it useful to use the :program:`pip` program (or similar ones)
to install python packages.

As explained in previous sections the user should in any case ensure that all
dependencies listed in the `Prerequisites`_ section are correctly installed.

The simplest way to install PyTables using :program:`pip` is the following::

  $ pip install tables

The following example shows how to install the latest stable version of
PyTables in the user folder when a older version of the package is already
installed at system level::

  $ pip install --user --upgrade tables

The `--user` option tels to the :program:`pip` tool to install the package in
the user folder (``$HOME/.local`` on GNU/Linux and Unix systems), while the
`--upgrade` option forces the installation of the latest version even if an
older version of the package is already installed.

The :program:`pip` tool can also be used to install packages from a source
tar-ball::

  $ pip install tables-3.0.0.tar.gz

To install the development version of PyTables from the *develop* branch of
the main :program:`git` :ref:`[GIT] <GIT>` repository the command is the
following::

  $ pip install git+https://github.com/PyTables/PyTables.git@develop#egg=tables

A similar command can be used to install a specific tagged fersion::

  $ pip install git+https://github.com/PyTables/PyTables.git@v.2.4.0#egg=tables

Finally, PyTables developers provide a :file:`requirements.txt` file that
can be used by :program:`pip` to install the PyTables dependencies::

  $ wget https://raw.github.com/PyTables/PyTables/develop/requirements.txt
  $ pip install -r requirements.txt

Of course the :file:`requirements.txt` file can be used to install only
python packages.  Other dependencies like the HDF5 library of compression
libraries have to be installed by the user.


.. _binaryInstallationDescr:

Binary installation (Windows)
-----------------------------

This section is intended for installing precompiled binaries on Windows
platforms. You may also find it useful for instructions on how to install
*binary prerequisites* even if you want to compile PyTables itself on Windows.


.. _prerequisitesBinInst:

Windows prerequisites
~~~~~~~~~~~~~~~~~~~~~

First, make sure that you have Python 2.6, NumPy 1.4.1 and Numexpr 2.0 or
higher installed (PyTables binaries have been built using NumPy 1.7 and
Numexpr 2.1).  The binaries already include DLLs for HDF5 (1.8.4, 1.8.9),
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

Binary packahes can be found e.g. at the `Unofficial Windows Binaries for
Python Extension Packages <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables>`_
page.

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

