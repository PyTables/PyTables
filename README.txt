README.txt
tables v0.1 (This is an alpha version, so use it carefully)
http://pytables.sourceforge.net/
October 4, 2002
--------------------------------------

This package is intended to be an easy-to-use HDF5 interface for
Python. To achieve this, the newest improvements introduced in Python
2.2 (like generators or slots and metaclasses in new-brand classes)
has been used. Another important reason to choose Python 2.2 has been
the use of Pyrex to wrap the HDF5 libraries. Pyrex
(http://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) provides a very
convenient way to access the HDF5 C API. So, you will need to use
Python 2.2 or higher to take advantage of this package (but you won't
need to install Pyrex, as I use it only as a development tool).

I've tested this pytables version with HDF5 1.4.4 and HDF5_HL
beta2. Hopefully, it should work well with all HDF5 1.4.x versions,
but you should stick with HDF5_HL beta 2.

At this moment, this module provides limited support of HDF5
facilities, but I hope to add more in the short future. By no means
this package will try to be a complete wrapper for all the HDF5
API. Instead, its goal is to allow working with tables (and hopefully
in short term also with NumArray objects) in a hierarchical structure.

The meaning of the term "tables" in this context follows the
definition stated on HDF5_HL documentation:

  "A table is defined as a collection of records whose values are
   stored in fixed-length fields. All records have the same structure
   and all values in each field have the same data type."

The terms "fixed-lenght" and strict "data types" seems to be quite a
strange requirement for an interpreted language like Python, but it's
fundamental when we want to save *lots* of data (mainly for scientific
applications, but not only that), in a efficient (both in terms of
CPU and I/O requeriments) way.

To emulate records (C structs in HDF5) in Python I've implemented a
special metaclass with the capability to detect errors in field
assignments as well as range overflows. More in documentation (see doc
directory).

I'm using Linux as the main development platform, and that should be
easy the compile/install in other UNIX machines, but I recognize that
more testing is needed to achieve complete portability, so I'd
appreciate input from other platforms. In particular, I forsee
problems on platforms which doesn't support the "long long int" type
(which allows to create files with sizes bigger than 2 GB).


Installation
------------

This are instructions for Unix/Linux system. If you are using Windows,
and get the library working, please, tell me about.

Extensions in PyTables has been made using Pyrex and C. You can
rebuild everything from scratch if you got Pyrex installed, but this
is not necessary, as the Pyrex compiled source is included in the
distribution. But if you want to do that, merely replace setup.py
script in these instructions by setup-pyrex.py.

The Python Distutils are used to build and install tables, so it is
fairly simple to get things ready to go.

1. First, make sure that you have hdf5 1.4.x and hdf5_hl libraries
   installed (I'm using hdf5 1.4.4 and hdf5_hl beta2 currently). If
   not, you can find them at http://hdf.ncsa.uiuc.edu/HDF5;
   compile/install them.

   setup.py will detect these libraries and include files under either
   /usr or /usr/local; this will catch installations from RPMs and
   most hand installations under Unix.  If setup.py can't find your
   libhdf5 and libhdf5_hl or if you have several versions installed
   and wants to select one of them, then you can give it a hint either
   in the environment (using the HDF5_DIR evironment variable) or on
   the command line by specifying the directory containing the include
   and lib directory.  For example:

	    --hdf5=/stuff/hdf5-1.4.4

   The libraries can installed anywhere on the filesystem, but
   remember to always place them together. For example, if libhdf5.so
   is installed in /usr/lib, so does hdf5_hl.so. The same applies to
   the headers.

   If your HDF5 libs were built as shared libraries, and if these
   shared libraries are not on the runtime load path, then you can
   specify the additional linker flags needed to find the shared
   library on the command line as well.  For example:

	   --lflags="-Xlinker -rpath -Xlinker /stuff/hdf5-1.4.4/lib"

   or perhaps just

           --lflags="-R /stuff/hdf5-1.4.4/lib"

   Check your compiler and linker documentation to be sure.

   It is also possible to specify linking against different libraries
   with the --libs switch:

           --libs="-lhdf5-1.4.6 -lhdf5_hl-beta2"
           --libs="-lhdf5-1.4.6 -lhdf5_hl-beta2 -lnsl"


2. From the main pytables distribution directory run this command,
   (plus any extra flags needed as discussed above):

	python setup.py build_ext --inplace

   depending on the compiler flags used when compiling your Python
   executable, it may appear lots of warnings. Don't worry, almost all
   of them are caused by variables declared but never used. That's
   normal in Pyrex extensions.

3. To run the test suite change into the test directory and run this
   command, (assuming your shell is bash or compatible):

	export PYTHONPATH=..
	python test_all.py

   If you would like to see some verbose output from the tests simply
   add the word "verbose" to the command line.  You can also run only
   the tests in a particular test module by themselves.  For example:

	python test_types.py


4. To install the entire PyTables Python package, change back to the
   root distribution directory and run this command as the root user:

	python setup.py install


That's it!


-- Francesc Alted
falted@openlc.org






