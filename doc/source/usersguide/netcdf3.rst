netcdf3 - a PyTables NetCDF3 emulation API (deprecated)
=======================================================

.. warning:: The tables.netcdf3 module is not actively maintained anymore.
   It is deprecated and will be removed in the future versions.

.. currentmodule:: tables.netcdf3

What is netcdf3?
----------------
The netCDF format is a popular format for binary files. It is
portable between machines and self-describing, i.e. it contains the
information necessary to interpret its contents. A free library
provides convenient access to these files (see ). A very nice python interface to that library
is available in the Scientific Python NetCDF module
(see ). Although it is
somewhat less efficient and flexible than HDF5, netCDF is geared for
storing gridded data and is quite easy to use. It has become a de
facto standard for gridded data, especially in meteorology and
oceanography. The next version of netCDF (netCDF 4) will actually be a
software layer on top of HDF5 (see ). The tables.netcdf3
package does not create HDF5 files that are compatible with netCDF 4
(although this is a long-term goal).


Using the tables.netcdf3 package
--------------------------------
The package tables.netcdf3 emulates the
Scientific.IO.NetCDF API using PyTables. It
presents the data in the form of objects that behave very much like
arrays. A tables.netcdf3 file contains any number
of dimensions and variables, both of which have unique names. Each
variable has a shape defined by a set of dimensions, and optionally
attributes whose values can be numbers, number sequences, or strings.
One dimension of a file can be defined as
*unlimited*, meaning that the file can grow along
that direction. In the sections that follow, a step-by-step tutorial
shows how to create and modify a tables.netcdf3
file. All of the code snippets presented here are included in
examples/netCDF_example.py. The
tables.netcdf3 package is designed to be used as a
drop-in replacement for Scientific.IO.NetCDF, with
only minor modifications to existing code. The differences between
tables.netcdf3 and
Scientific.IO.NetCDF are summarized in the last
section of this chapter.


Creating/Opening/Closing a tables.netcdf3 file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To create a tables.netcdf3 file from
python, you simply call the NetCDFFile
constructor. This is also the method used to open an existing
tables.netcdf3 file. The object returned is an
instance of the NetCDFFile class and all future
access must be done through this object. If the file is open for
write access ('w' or 'a'), you
may write any type of new data including new dimensions, variables
and attributes. The optional history keyword
argument can be used to set the history
NetCDFFile global file attribute. Closing the
tables.netcdf3 file is accomplished via the
close method of NetCDFFile
object.

Here's an example::

    >>> import tables.netcdf3 as NetCDF
    >>> import time
    >>> history = 'Created ' + time.ctime(time.time())
    >>> file = NetCDF.NetCDFFile('test.h5', 'w', history=history)
    >>> file.close()

Dimensions in a tables.netcdf3 file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NetCDF defines the sizes of all variables in terms of
dimensions, so before any variables can be created the dimensions
they use must be created first. A dimension is created using the
createDimension method of the
NetCDFFile object. A Python string is used to set
the name of the dimension, and an integer value is used to set the
size. To create an *unlimited* dimension (a
dimension that can be appended to), the size value is set to
None::

    >>> import tables.netcdf3 as NetCDF
    >>> file = NetCDF.NetCDFFile('test.h5', 'a')
    >>> file.NetCDFFile.createDimension('level', 12)
    >>> file.NetCDFFile.createDimension('time', None)
    >>> file.NetCDFFile.createDimension('lat', 90)

All of the dimension names and their associated sizes are
stored in a Python dictionary::

    >>> print file.dimensions
    {'lat': 90, 'time': None, 'level': 12}


Variables in a tables.netcdf3 file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the data in a tables.netcdf3 file
is stored in a netCDF variable (except for global attributes). To
create a netCDF variable, use the createVariable
method of the NetCDFFile object. The
createVariable method has three mandatory
arguments, the variable name (a Python string), the variable
datatype described by a single character Numeric typecode string
which can be one of f (Float32),
d (Float64), i (Int32),
l (Int32), s (Int16),
c (CharType - length 1), F
(Complex32), D (Complex64) or
1 (Int8), and a tuple containing the variable's
dimension names (defined previously with
createDimension). The dimensions themselves are
usually defined as variables, called coordinate variables. The
createVariable method returns an instance of the
NetCDFVariable class whose methods can be used
later to access and set variable data and attributes::

    >>> times = file.createVariable('time','d',('time',))
    >>> levels = file.createVariable('level','i',('level',))
    >>> latitudes = file.createVariable('latitude','f',('lat',))
    >>> temp = file.createVariable('temp','f',('time','level','lat',))
    >>> pressure = file.createVariable('pressure','i',('level','lat',))

All of the variables in the file are stored in a Python
dictionary, in the same way as the dimensions::

    >>> print file.variables
    {'latitude': <tables.netcdf3.NetCDFVariable instance at 0x244f350>,
     'pressure': <tables.netcdf3.NetCDFVariable instance at 0x244f508>,
     'level': <tables.netcdf3.NetCDFVariable instance at 0x244f0d0>,
     'temp': <tables.netcdf3.NetCDFVariable instance at 0x244f3a0>,
     'time': <tables.netcdf3.NetCDFVariable instance at 0x2564c88>}

Attributes in a tables.netcdf3 file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are two types of attributes in a
tables.netcdf3 file, global (or file) and
variable. Global attributes provide information about the dataset,
or file, as a whole. Variable attributes provide information about
one of the variables in the file. Global attributes are set by
assigning values to NetCDFFile instance
variables. Variable attributes are set by assigning values to
NetCDFVariable instance variables.

Attributes can be strings, numbers or sequences. Returning to
our example::

    >>> file.description = 'bogus example to illustrate the use of tables.netcdf3'
    >>> file.source = 'PyTables Users Guide'
    >>> latitudes.units = 'degrees north'
    >>> pressure.units = 'hPa'
    >>> temp.units = 'K'
    >>> times.units = 'days since January 1, 2005'
    >>> times.scale_factor = 1

The ncattrs method of the
NetCDFFile object can be used to retrieve the
names of all the global attributes. This method is provided as a
convenience, since using the built-in dir Python
function will return a bunch of private methods and attributes that
cannot (or should not) be modified by the user. Similarly, the
ncattrs method of a
NetCDFVariable object returns all of the netCDF
variable attribute names. These functions can be used to easily
print all of the attributes currently defined, like this::

    >>> for name in file.ncattrs():
    ...     print 'Global attr', name, '=', getattr(file,name)
    Global attr description = bogus example to illustrate the use of tables.netcdf3
    Global attr history = Created Mon Nov  7 10:30:56 2005
    Global attr source = PyTables Users Guide

Note that the ncattrs function is not part
of the Scientific.IO.NetCDF interface.


Writing data to and retrieving data from a tables.netcdf3 variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now that you have a netCDF variable object, how do you put
data into it? If the variable has no *unlimited*
dimension, you just treat it like a Numeric array object and assign
data to a slice::

    >>> import numpy
    >>> levels[:] = numpy.arange(12)+1
    >>> latitudes[:] = numpy.arange(-89,90,2)
    >>> for lev in levels[:]:
    >>>     pressure[:,:] = 1000.-100.*lev
    >>> print 'levels = ',levels[:]
    levels =  [ 1  2  3  4  5  6  7  8  9 10 11 12]
    >>> print 'latitudes =\n',latitudes[:]
    latitudes =
    [-89. -87. -85. -83. -81. -79. -77. -75. -73. -71. -69. -67. -65. -63.
    -61. -59. -57. -55. -53. -51. -49. -47. -45. -43. -41. -39. -37. -35.
    -33. -31. -29. -27. -25. -23. -21. -19. -17. -15. -13. -11.  -9.  -7.
    -5.  -3.  -1.   1.   3.   5.   7.   9.  11.  13.  15.  17.  19.  21.
    23.  25.  27.  29.  31.  33.  35.  37.  39.  41.  43.  45.  47.  49.
    51.  53.  55.  57.  59.  61.  63.  65.  67.  69.  71.  73.  75.  77.
    79.  81.  83.  85.  87.  89.]

Note that retrieving data from the netCDF variable object
works just like a Numeric array too. If the netCDF variable has an
*unlimited* dimension, and there is not yet an
entry for the data along that dimension, the
append method must be used::

    >>> for n in range(10):
    >>>     times.append(n)
    >>> print 'times = ',times[:]
    times =  [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]

The data you append must have either the same number of
dimensions as the NetCDFVariable, or one less.
The shape of the data you append must be the same as the
NetCDFVariable for all of the dimensions except
the *unlimited* dimension. The length of the data
long the *unlimited* dimension controls how may
entries along the *unlimited* dimension are
appended. If the data you append has one fewer number of dimensions
than the NetCDFVariable, it is assumed that you
are appending one entry along the *unlimited*
dimension. For example, if the NetCDFVariable has
shape (10,50,100) (where the dimension length of
length 10 is the *unlimited*
dimension), and you append an array of shape
(50,100), the NetCDFVariable
will subsequently have a shape of (11,50,100). If
you append an array with shape (5,50,100), the
NetCDFVariable will have a new shape of
(15,50,100). Appending an array whose last two
dimensions do not have a shape (50,100) will
raise an exception. This append method does not
exist in the Scientific.IO.NetCDF interface,
instead entries are appended along the
*unlimited* dimension one at a time by assigning
to a slice. This is the biggest difference between the
tables.netcdf3 and
Scientific.IO.NetCDF interfaces.

Once data has been appended to any variable with an
*unlimited* dimension, the
sync method can be used to synchronize the sizes
of all the other variables with an *unlimited*
dimension. This is done by filling in missing values (given by the
default netCDF _FillValue, which is intended to
indicate that the data was never defined). The
sync method is automatically invoked with a
NetCDFFile object is closed. Once the
sync method has been invoked, the filled-in
values can be assigned real data with slices::

    >>> print 'temp.shape before sync = ',temp.shape
    temp.shape before sync =  (0, 12, 90)
    >>> file.sync()
    >>> print 'temp.shape after sync = ',temp.shape
    temp.shape after sync =  (10, 12, 90)
    >>> from numarray import random_array
    >>> for n in range(10):
    ...     temp[n] = 10.*random_array.random(pressure.shape)
    ...     print 'time, min/max temp, temp[n,0,0] = ',\\
    times[n],min(temp[n].flat),max(temp[n].flat),temp[n,0,0]
    time, min/max temp, temp[n,0,0] = 0.0 0.0122650898993 9.99259281158 6.13053750992
    time, min/max temp, temp[n,0,0] = 1.0 0.00115821603686 9.9915933609 6.68516159058
    time, min/max temp, temp[n,0,0] = 2.0 0.0152112031356 9.98737239838 3.60537290573
    time, min/max temp, temp[n,0,0] = 3.0 0.0112022599205 9.99535560608 6.24249696732
    time, min/max temp, temp[n,0,0] = 4.0 0.00519315246493 9.99831295013 0.225010097027
    time, min/max temp, temp[n,0,0] = 5.0 0.00978941563517 9.9843454361 4.56814193726
    time, min/max temp, temp[n,0,0] = 6.0 0.0159023851156 9.99160385132 6.36837291718
    time, min/max temp, temp[n,0,0] = 7.0 0.0019518379122 9.99939727783 1.42762875557
    time, min/max temp, temp[n,0,0] = 8.0 0.00390585977584 9.9909954071 2.79601073265
    time, min/max temp, temp[n,0,0] = 9.0 0.0106026884168 9.99195957184 8.18835449219

Note that appending data along an
*unlimited* dimension always increases the length
of the variable along that dimension. Assigning data to a variable
with an *unlimited* dimension with a slice
operation does not change its shape. Finally, before closing the
file we can get a summary of its contents simply by printing the
NetCDFFile object. This produces output very
similar to running 'ncdump -h' on a netCDF file::

    >>> print file
    test.h5 {
    dimensions:
    lat = 90 ;
    time = UNLIMITED ; // (10 currently)
    level = 12 ;
    variables:
    float latitude('lat',) ;
    latitude:units = 'degrees north' ;
    int pressure('level', 'lat') ;
    pressure:units = 'hPa' ;
    int level('level',) ;
    float temp('time', 'level', 'lat') ;
    temp:units = 'K' ;
    double time('time',) ;
    time:scale_factor = 1 ;
    time:units = 'days since January 1, 2005' ;
    // global attributes:
    :description = 'bogus example to illustrate the use of tables.netcdf3' ;
    :history = 'Created Wed Nov  9 12:29:13 2005' ;
    :source = 'PyTables Users Guide' ;
    }

Efficient compression of tables.netcdf3 variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data stored in NetCDFVariable objects is
compressed on disk by default. The parameters for the default
compression are determined from a Filters class
instance (see section :ref:`FiltersClassDescr`) with complevel=6,
complib='zlib' and shuffle=True. To change the default
compression, simply pass a Filters instance to
createVariable with the
filters keyword. If your data only has a certain
number of digits of precision (say for example, it is temperature
data that was measured with a precision of 0.1
degrees), you can dramatically improve compression by quantizing (or
truncating) the data using the
least_significant_digit keyword argument to
createVariable. The *least significant
digit* is the power of ten of the smallest decimal place
in the data that is a reliable value. For example if the data has a
precision of 0.1, then setting
least_significant_digit=1 will cause data the
data to be quantized using
numpy.around(scale*data)/scale, where
scale = 2**bits, and bits is determined so that a
precision of 0.1 is retained (in this case
bits=4).

In our example, try replacing the line::

    >>> temp = file.createVariable('temp','f',('time','level','lat',))

with::

    >>> temp = file.createVariable('temp','f',('time','level','lat',), least_significant_digit=1)

and see how much smaller the resulting file is.

The least_significant_digit keyword
argument is not allowed in Scientific.IO.NetCDF,
since netCDF version 3 does not support compression. The flexible,
fast and efficient compression available in HDF5 is the main reason
I wrote the tables.netcdf3 package - my netCDF
files were just getting too big.

The createVariable method has one other
keyword argument not found in
Scientific.IO.NetCDF -
expectedsize. The expectedsize
keyword can be used to set the expected number of entries along the
*unlimited* dimension (default 10000). If you
expect that your data with have an order of magnitude more or less
than 10000 entries along the *unlimited*
dimension, you may consider setting this keyword to improve
efficiency (see :ref:`expectedRowsOptim` for details).


tables.netcdf3 package reference
--------------------------------

.. deprecated:: 2.3

Global constants
~~~~~~~~~~~~~~~~

.. data:: _fillvalue_dict

    Dictionary whose keys are
    NetCDFVariable single character typecodes
    and whose values are the netCDF _FillValue for that
    typecode.

.. data:: ScientificIONetCDF_imported

    True if Scientific.IO.NetCDF is installed and can be imported.


The NetCDFFile class
~~~~~~~~~~~~~~~~~~~~
.. class:: NetCDFFile(filename, mode='r', history=None)

    Opens an existing tables.netcdf3 file (mode
    = 'r' or 'a') or creates a new
    one (mode = 'w'). The history
    keyword can be used to set the NetCDFFile.history
    global attribute (if mode = 'a' or 'w').

    A NetCDFFile object has two standard
    attributes: dimensions and
    variables. The values of both are dictionaries,
    mapping dimension names to their associated lengths and variable
    names to variables. All other attributes correspond to global
    attributes defined in a netCDF file. Global file attributes are
    created by assigning to an attribute of the NetCDFFile object.


NetCDFFile methods
^^^^^^^^^^^^^^^^^^

.. method:: NetCDFFile.close()

    Closes the file (after invoking the sync method).


.. method:: NetCDFFile.sync()

    Synchronizes the size of variables along the
    *unlimited* dimension, by filling in data
    with default netCDF _FillValue. Returns the length of the
    *unlimited* dimension. Invoked automatically
    when the NetCDFFile object is closed.


.. method:: NetCDFFile.ncattrs()

    Returns a list with the names of all currently defined
    netCDF global file attributes.

.. method:: NetCDFFile.createDimension(name, length)

    Creates a netCDF dimension with a name given by the Python
    string name and a size given by the integer
    size. If size = None, the
    dimension is *unlimited* (i.e. it can grow
    dynamically). There can be only one
    *unlimited* dimension in a file.

.. method:: NetCDFFile.createVariable(name, type, dimensions, least_significant_digit= None, expectedsize=10000, filters=None)

    Creates a new variable with the given name, type,
    and dimensions. The type is a one-letter Numeric
    typecode string which can be one of f
    (Float32), d (Float64), i
    (Int32), l (Int32), s
    (Int16), c (CharType - length 1),
    F (Complex32), D
    (Complex64) or 1 (Int8); the predefined type
    constants from Numeric can also be used. The
    F and D types are not
    supported in netCDF or Scientific.IO.NetCDF, if they are used in
    a tables.netcdf3 file, that file cannot be
    converted to a true netCDF file nor can it be shared over the
    Internet with OPeNDAP. Dimensions must be a tuple containing
    dimension names (strings) that have been defined previously by
    createDimensions. The
    least_significant_digit is the power of ten
    of the smallest decimal place in the variable's data that is a
    reliable value. If this keyword is specified, the variable's
    data truncated to this precision to improve compression. The
    expectedsize keyword can be used to set the
    expected number of entries along the
    *unlimited* dimension (default 10000). If you
    expect that your data with have an order of magnitude more or
    less than 10000 entries along the *unlimited*
    dimension, you may consider setting this keyword to improve
    efficiency (see :ref:`expectedRowsOptim` for details). The
    filters keyword is a PyTables
    Filters instance that describes how to store
    the data on disk. The default corresponds to
    complevel=6,
    complib='zlib',
    shuffle=True and fletcher32=False.


.. method:: NetCDFFile.nctoh5(filename, unpackshort=True, filters=None)

    Imports the data in a netCDF version 3 file
    (filename) into a
    NetCDFFile object using
    Scientific.IO.NetCDF
    (ScientificIONetCDF_imported must be True). If
    unpackshort=True, data packed as short
    integers (type s) in the netCDF file will be
    unpacked to type f using the scale_factor and
    add_offset netCDF variable attributes. The
    filters keyword can be set to a PyTables
    Filters instance to change the default
    parameters used to compress the data in the
    tables.netcdf3 file. The default corresponds
    to complevel=6,
    complib='zlib',
    shuffle=True and
    fletcher32=False.


.. method:: NetCDFFile.h5tonc(filename, packshort=False, scale_factor=None, add_offset=None)

    Exports the data in a tables.netcdf3
    file defined by the NetCDFFile instance into
    a netCDF version 3 file using
    Scientific.IO.NetCDF (ScientificIONetCDF_imported must be True). If
    packshort=True> the dictionaries
    scale_factor and
    add_offset are used to pack data of type
    f as  short integers (of types) in the netCDF file. Since netCDF version 3
    does not provide automatic compression, packing as short
    integers is a commonly used way of saving disk space (see this
    `page <http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml>`_
    for more details). The keys of these dictionaries are the
    variable names to pack, the values are the scale_factors and
    offsets to use in the packing. The data are packed so that the
    original Float32 values can be reconstructed by multiplying the
    scale_factor and adding
    add_offset. The resulting netCDF file will
    have the scale_factor and
    add_offset variable attributes set
    appropriately.


The NetCDFVariable class
~~~~~~~~~~~~~~~~~~~~~~~~
.. class:: NetCDFVariable

    The NetCDFVariable constructor is not
    called explicitly, rather an NetCDFVariable
    instance is returned by an invocation of
    NetCDFFile.createVariable.
    NetCDFVariable objects behave like arrays, and
    have the standard attributes of arrays (such as
    shape). Data can be assigned or extracted from
    NetCDFVariable objects via slices.


NetCDFVariable methods
^^^^^^^^^^^^^^^^^^^^^^

.. method:: NetCDFVariable.typecode()

    Returns a single character typecode describing the type of
    the variable, one of f (Float32),
    d (Float64), i (Int32),
    l (Int32), s (Int16),
    c (CharType - length 1), F
    (Complex32), D (Complex64) or
    1 (Int8).


.. method:: NetCDFVariable.append(data)

    Append data to a variable along its
    *unlimited* dimension. The data you append
    must have either the same number of dimensions as the
    NetCDFVariable, or one less. The shape of the
    data you append must be the same as the
    NetCDFVariable for all of the dimensions
    except the *unlimited* dimension. The length
    of the data long the *unlimited* dimension
    controls how may entries along the
    *unlimited* dimension are appended. If the
    data you append has one fewer number of dimensions than the
    NetCDFVariable, it is assumed that you are
    appending one entry along the *unlimited*
    dimension. For variables without an
    *unlimited* dimension, data can simply be
    assigned to a slice without using the append
    method.


.. method:: NetCDFVariable.ncattrs()

    Returns a list with all the names of the currently defined
    netCDF variable attributes.


.. method:: NetCDFVariable.assignValue(data)

    Provided for compatibility with
    Scientific.IO.NetCDF. Assigns data to the
    variable. If the variable has an *unlimited*
    dimension, it is equivalent to append(data).
    If the variable has no *unlimited* dimension,
    it is equivalent to assigning data to the variable with the
    slice [:].


.. method:: NetCDFVariable.getValue()

    Provided for compatibility with
    Scientific.IO.NetCDF. Returns all the data in
    the variable. Equivalent to extracting the slice [:] from the variable.


Converting between true netCDF files and tables.netcdf3 files
-------------------------------------------------------------
If Scientific.IO.NetCDF is installed,
tables.netcdf3 provides facilities for converting
between true netCDF version 3 files and
tables.netcdf3 hdf5 files via the
NetCDFFile.h5tonc() and
NetCDFFile.nctoh5() class methods. Also, the
nctoh5 command-line utility (see :ref:`nctoh5Descr`) uses the
NetCDFFile.nctoh5() class method.

As an example, look how to convert a
tables.netcdf3 hdf5 file to a true netCDF version 3
file (named test.nc)::

    >>> scale_factor = {'temp': 1.75e-4}
    >>> add_offset = {'temp': 5.}
    >>> file.h5tonc('test.nc',packshort=True, \\
    scale_factor=scale_factor,add_offset=add_offset)
    packing temp as short integers ...
    >>> file.close()

The dictionaries scale_factor and
add_offset are used to optionally pack the data as
short integers in the netCDF file. Since netCDF version 3 does not
provide automatic compression, packing as short integers is a commonly
used way of saving disk space (see this `page <http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml>`_
for more details). The keys of these dictionaries are the variable
names to pack, the values are the scale_factors and offsets to use in
the packing. The resulting netCDF file will have the
scale_factor and add_offset
variable attributes set appropriately.

To convert the netCDF file back to a
tables.netcdf3 hdf5 file::

    >>> history = 'Convert from netCDF ' + time.ctime(time.time())
    >>> file = NetCDF.NetCDFFile('test2.h5', 'w', history=history)
    >>> nobjects, nbytes = file.nctoh5('test.nc',unpackshort=True)
    >>> print nobjects,' objects converted from netCDF, totaling',nbytes,'bytes'
    5  objects converted from netCDF, totaling 48008 bytes
    >>> temp = file.variables['temp']
    >>> times = file.variables['time']
    >>> print 'temp.shape after h5 --> netCDF --> h5 conversion = ',temp.shape
    temp.shape after h5 --> netCDF --> h5 conversion =  (10, 12, 90)
    >>> for n in range(10):
    ...     print 'time, min/max temp, temp[n,0,0] = ',\\
    times[n],min(temp[n].flat),max(temp[n].flat),temp[n,0,0]
    time, min/max temp, temp[n,0,0] = 0.0 0.0123250000179 9.99257469177 6.13049983978
    time, min/max temp, temp[n,0,0] = 1.0 0.00130000000354 9.99152469635 6.68507480621
    time, min/max temp, temp[n,0,0] = 2.0 0.0153000000864 9.98732471466 3.60542488098
    time, min/max temp, temp[n,0,0] = 3.0 0.0112749999389 9.99520015717 6.2423248291
    time, min/max temp, temp[n,0,0] = 4.0 0.00532499980181 9.99817466736 0.225124999881
    time, min/max temp, temp[n,0,0] = 5.0 0.00987500045449 9.98417472839 4.56827497482
    time, min/max temp, temp[n,0,0] = 6.0 0.01600000076 9.99152469635 6.36832523346
    time, min/max temp, temp[n,0,0] = 7.0 0.00200000009499 9.99922466278 1.42772495747
    time, min/max temp, temp[n,0,0] = 8.0 0.00392499985173 9.9908246994 2.79605007172
    time, min/max temp, temp[n,0,0] = 9.0 0.0107500003651 9.99187469482 8.18832492828
    >>> file.close()

Setting unpackshort=True tells
nctoh5 to unpack all of the variables which have
the scale_factor and add_offset
attributes back to floating point arrays. Note that
tables.netcdf3 files have some features not
supported in netCDF (such as Complex data types and the ability to
make any dimension *unlimited*).
tables.netcdf3 files which utilize these features
cannot be converted to netCDF using
NetCDFFile.h5tonc.


tables.netcdf3 file structure
-----------------------------
A tables.netcdf3 file consists of array
objects (either EArrays or
CArrays) located in the root group of a pytables
hdf5 file. Each of the array objects must have a
dimensions attribute, consisting of a tuple of
dimension names (the length of this tuple should be the same as the
rank of the array object). Any array objects with one of the supported
datatypes in a pytables file that conforms to this simple structure
can be read with the tables.netcdf3 package.


Sharing data in tables.netcdf3 files over the Internet with OPeNDAP
-------------------------------------------------------------------
tables.netcdf3 datasets can be shared over
the Internet with the OPeNDAP protocol (http://opendap.org), via the python
OPeNDAP module (http://opendap.oceanografia.org).
A plugin for the python opendap server is included with the pytables
distribution (contrib/h5_dap_plugin.py). Simply
copy that file into the plugins directory of the
opendap python module source distribution, run python
setup.py install, point the opendap server to the directory
containing your tables.netcdf3 files, and away you
go. Any OPeNDAP aware client (such as Matlab or IDL) will now be able
to access your data over http as if it were a local disk file. The
only restriction is that your tables.netcdf3 files
must have the extension .h5 or
.hdf5. Unfortunately,
tables.netcdf3 itself cannot act as an OPeNDAP
client, although there is a client included in the opendap python
module, and Scientific.IO.NetCDF can act as an
OPeNDAP client if it is linked with the OPeNDAP netCDF client library.
Either of these python modules can be used to remotely access
tables.netcdf3 datasets with OPeNDAP.


Differences between the Scientific.IO.NetCDF API and the tables.netcdf3 API
---------------------------------------------------------------------------

#. tables.netcdf3 data is stored in an HDF5
   file instead of a netCDF file.
#. Although each variable can have only one
   *unlimited* dimension in a
   tables.netcdf3 file, it need not be the first
   as in a true NetCDF file. Complex data types F
   (Complex32) and D (Complex64) are supported in
   tables.netcdf3, but are not supported in netCDF
   (or Scientific.IO.NetCDF). Files with variables
   that have these datatypes, or an *unlimited*
   dimension other than the first, cannot be converted to netCDF
   using h5tonc.
#. Variables in a tables.netcdf3 file are
   compressed on disk by default using HDF5 zlib compression with the
   *shuffle* filter. If the
   *least_significant_digit* keyword is used when
   a variable is created with the createVariable
   method, data will be truncated (quantized) before being
   written to the file. This can significantly improve compression.
   For example, if least_significant_digit=1, data
   will be quantized using
   numpy.around(scale*data)/scale, where
   scale = 2**bits, and bits is determined so that
   a precision of 0.1 is retained (in this case
   bits=4). From http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml::

       "least_significant_digit -- power of ten of the smallest
        decimal place in unpacked data that is a reliable value."

   Automatic data compression is not available in netCDF version 3,
   and hence is not available in the
   Scientific.IO.NetCDF module.
#. In tables.netcdf3, data must be appended
   to a variable with an *unlimited* dimension
   using the append method of the
   netCDF variable object. In
   Scientific.IO.NetCDF, data can be added along
   an *unlimited* dimension by assigning it to a
   slice (there is no append method). The sync
   method of a tables.netcdf3 NetCDFVariable
   object synchronizes the size of all variables with an
   *unlimited* dimension by filling in data using
   the default netCDF _FillValue. The
   sync method is automatically invoked with a
   NetCDFFile object is closed. In
   Scientific.IO.NetCDF, the
   sync() method flushes the data to disk.
#. The tables.netcdf3 createVariable()
   method has three extra optional keyword arguments not found in the
   Scientific.IO.NetCDF interface,
   *least_significant_digit* (see item (2) above),
   *expectedsize* and
   *filters*. The
   *expectedsize* keyword applies only to
   variables with an *unlimited* dimension, and is
   an estimate of the number of entries that will be added along that
   dimension (default 1000). This estimate is used to optimize HDF5
   file access and memory usage. The *filters*
   keyword is a PyTables filters instance that describes how to store
   the data on disk. The default corresponds to
   complevel=6, complib='zlib',
   shuffle=True and
   fletcher32=False.
#. tables.netcdf3 data can be saved to a
   true netCDF file using the NetCDFFile class
   method h5tonc (if
   Scientific.IO.NetCDF is installed). The
   *unlimited* dimension must be the first (for
   all variables in the file) in order to use the
   h5tonc method. Data can also be imported from a
   true netCDF file and saved in an HDF5
   tables.netcdf3 file using the
   nctoh5 class method.
#. In tables.netcdf3 a list of attributes
   corresponding to global netCDF attributes defined in the file can
   be obtained with the NetCDFFile ncattrs method.
   Similarly, netCDF variable attributes can be obtained with the
   NetCDFVariable ncattrs
   method. These functions are not available in the
   Scientific.IO.NetCDF API.
#. You should not define tables.netcdf3
   global or variable attributes that start with
   _NetCDF_. Those names are reserved for internal use.
#. Output similar to 'ncdump -h' can be obtained by simply
   printing a tables.netcdf3
   NetCDFFile instance.

