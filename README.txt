README.txt
PyTables v0.2 (This is second alpha version)
http://pytables.sf.net/
November 11, 2002
--------------------------------------

################## Note ############### 
#PyTables is still in alpha because I plan to add some new features
#(like unlimited dimension arrays or general python attributes) and
#the API may still evolve a bit. Nevertheless, the present release has
#been thoroughly tested and may be used to do real work right
#now. However, don't complain if methods names or file format are
#(slightly, hopefully) changed in future releases.
#######################################

PyTables goal is to allow dealing easily, but in a powerful way, with
scientific data *tables* and Numeric Python objects (new in 0.2!) in a
hierarchical structure. As a foundation for the underlying hierachical
data organization the excellent HDF5 library
(http://hdf.ncsa.uiuc.edu/HDF5) has been choosed, and although right
now PyTables provides limited support of all the HDF5 facilities, I
hope to add the most interesting ones (for PyTables needs) in the
short future. But it should be clearly stated that by no means this
package will try to be a complete wrapper for all the HDF5 API.

The meaning of the term *tables* in this context follows the
definition stated on HDF5 documentation:

""" A table is defined as a collection of records whose values are
   stored in fixed-length fields. All records have the same structure
   and all values in each field have the same data type.
"""

The terms "fixed-length" and strict "data types" seems to be quite a
strange requirement for an interpreted language like Python, but they
have a principal role if our our goal is to save *lots* of data
(mainly for scientific applications, but not only that), in a
efficient way in terms both of CPU and I/O requeriments.

To emulate records (C structs in HDF5) in Python PyTables implements a
special metaclass with the capability to detect errors in field
assignments as well as range overflows. At same time, it provides a
powerful interface to process table data.

Quite a bit effort has been put to make browsing on the hierarchical
data structure a pleasant experience. Only three (orthogonal) methods
are enough to accomplish that. More in documentation.

I must say that one of the main objectives during the PyTables design has
been the user friendliness. To achieve this, the newest improvements
introduced in Python 2.2 (like generators or slots and metaclasses in
new-brand classes) has been used and abused. Another important reason
to choose Python 2.2 has been my willing to use Pyrex to wrap the HDF5
libraries. Pyrex provides a very convenient way to access the HDF5 C
API. For these reasons, you will need to use Python 2.2 or higher to
take advantage of PyTables (but you won't need to install Pyrex, as I
use it only as a development tool).

I've tested this PyTables version with HDF5 1.4.4 and Numeric 22.0,
but hopefully, it should work well with all HDF5 1.4.x versions and a
relatively new version of Numeric (>= 20.x).

I'm using Linux as the main development platform, but it should be
easy the compile/install PyTables in other UNIX machines, but I
recognize that more testing is needed to achieve complete portability,
so I'd appreciate input from other platforms. In particular, I forsee
problems on platforms which does not support the "long long int" type
(that allows to create files with sizes bigger than 2 GB).


Installation
------------

This are instructions for Unix/Linux system. If you are using Windows,
and get the library working, please, tell me about.

The Python Distutils are used to build and install tables, so it is
fairly simple to get things ready to go.

1. First, make sure that you have HDF5 1.4.x and Numeric Python
   installed (I'm using HDF5 1.4.4 and Numeric 22.0 currently). If
   don't, you can find them at http://hdf.ncsa.uiuc.edu/HDF5 and
   http://www.pfdubois.com/numpy. Compile/install them.

   setup.py will detect HDF5 libraries and include files under either
   /usr or /usr/local; this will catch installations from RPMs and
   most hand installations under Unix.  If setup.py can't find your
   libhdf5 or if you have several versions installed and wants to
   select one of them, then you can give it a hint either in the
   environment (using the HDF5_DIR evironment variable) or on the
   command line by specifying the directory containing the include and
   lib directory.  For example:

	    --hdf5=/stuff/hdf5-1.4.4

   With that, the libraries can installed anywhere on the filesystem.

   If your HDF5 libs were built as shared libraries, and if these
   shared libraries are not in the runtime load path, then you can
   specify the additional linker flags needed to find the shared
   library on the command line as well.  For example:

	   --lflags="-Xlinker -rpath -Xlinker /stuff/hdf5-1.4.4/lib"

   or perhaps just

           --lflags="-R /stuff/hdf5-1.4.4/lib"

   Check your compiler and linker documentation to be sure.

   It is also possible to specify linking against different libraries
   with the --libs switch:

           --libs="-lhdf5-1.4.6"
           --libs="-lhdf5-1.4.6 -lnsl"


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
   add the flag "-v" and/or the word "verbose" to the command line.
   You can also run only the tests in a particular test module by
   themselves.  For example:

	python test_types.py


4. To install the entire PyTables Python package, change back to the
   root distribution directory and run this command as the root user:

	python setup.py install


That's it!


-- Francesc Alted
falted@openlc.org






