.. _datatypes:

Supported data types in PyTables
================================

All PyTables datasets can handle the complete set of data types supported by
the NumPy (see :ref:`[NUMPY] <NUMPY>`) package in Python.
The data types for table fields can be set via instances of the Col class and
its descendants (see :ref:`ColClassDescr`), while the data type of array
elements can be set through the use of the Atom class and its descendants
(see :ref:`AtomClassDescr`).

PyTables uses ordinary strings to represent its *types*, with most of them
matching the names of NumPy scalar types. Usually, a PyTables type consists
of two parts: a *kind* and a *precision* in bits.
The precision may be omitted in types with just one supported precision (like
bool) or with a non-fixed size (like string).

There are eight kinds of types supported by PyTables:

- bool: Boolean (true/false) types.
  Supported precisions: 8 (default) bits.

- int: Signed integer types.
  Supported precisions: 8, 16, 32 (default) and 64 bits.

- uint: Unsigned integer types.
  Supported precisions: 8, 16, 32 (default) and 64 bits.

- float: Floating point types.
  Supported precisions: 16, 32, 64 (default) bits and extended precision
  floating point (see
  :ref:`note on floating point types<floating-point-note>`).

- complex: Complex number types.
  Supported precisions: 64 (32+32), 128 (64+64, default) bits and extended
  precision complex (see
  :ref:`note on floating point types<floating-point-note>`).

- string: Raw string types.
  Supported precisions: 8-bit positive multiples.

- time: Data/time types.
  Supported precisions: 32 and 64 (default) bits.

- enum: Enumerated types.
  Precision depends on base type.

.. _floating-point-note:
.. note:: Floating point types.

   The half precision floating point data type (float16) and extended
   precision ones (fload96, float128, complex192, complex256) are only
   available if numpy_ supports them on the host platform.

   Also, in order to use the half precision floating point type (float16)
   it is required numpy_ >= 1.6.0.

.. _numpy: http://www.numpy.org

The time and enum kinds area little bit special, since they represent HDF5
types which have no direct Python counterpart, though atoms of these kinds
have a more-or-less equivalent NumPy data type.

There are two types of time: 4-byte signed integer (time32) and 8-byte double
precision floating point (time64). Both of them reflect the number of seconds
since the Unix epoch, i.e. Jan 1 00:00:00 UTC 1970. They are stored in memory
as NumPy's int32 and float64, respectively, and in the HDF5 file using the
H5T_TIME class. Integer times are stored on disk as such, while floating
point times are split into two signed integer values representing seconds and
microseconds (beware: smaller decimals will be lost!).

PyTables also supports HDF5 H5T_ENUM *enumerations* (restricted sets of
unique name and unique value pairs). The NumPy representation of an
enumerated value (an Enum, see :ref:`EnumClassDescr`) depends on the concrete
*base type* used to store the enumeration in the HDF5 file.
Currently, only scalar integer values (both signed and unsigned) are
supported in enumerations. This restriction may be lifted when HDF5 supports
other kinds on enumerated values.

Here you have a quick reference to the complete set of supported data types:

.. table:: **Data types supported for array elements and tables columns in
             PyTables.**

    ================== ========================== ====================== =============== ==================
    Type Code          Description                C Type                 Size (in bytes) Python Counterpart
    ================== ========================== ====================== =============== ==================
    bool               boolean                    unsigned char          1               bool
    int8               8-bit integer              signed char            1               int
    uint8              8-bit unsigned integer     unsigned char          1               int
    int16              16-bit integer             short                  2               int
    uint16             16-bit unsigned integer    unsigned short         2               int
    int32              integer                    int                    4               int
    uint32             unsigned integer           unsigned int           4               long
    int64              64-bit integer             long long              8               long
    uint64             unsigned 64-bit integer    unsigned long long     8               long
    float16 [1]_       half-precision float       -                      2               -
    float32            single-precision float     float                  4               float
    float64            double-precision float     double                 8               float
    float96 [1]_ [2]_  extended precision float   -                      12              -
    float128 [1]_ [2]_ extended precision float   -                      16              -
    complex64          single-precision complex   struct {float r, i;}   8               complex
    complex128         double-precision complex   struct {double r, i;}  16              complex
    complex192 [1]_    extended precision complex -                      24              -
    complex256 [1]_    extended precision complex -                      32              -
    string             arbitrary length string    char[]                 *               str
    time32             integer time               POSIX's time_t         4               int
    time64             floating point time        POSIX's struct timeval 8               float
    enum               enumerated value           enum                   -               -
    ================== ========================== ====================== =============== ==================

.. rubric:: Footnotes

.. [1] see the above :ref:`note on floating point types <floating-point-note>`.
.. [2] currently in numpy_. "float96" and "float128" are equivalent of
       "longdouble" i.e. 80 bit extended precision floating point.
