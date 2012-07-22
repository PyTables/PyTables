PyTables File Format
====================
PyTables has a powerful capability to deal with native HDF5 files created
with another tools. However, there are situations were you may want to create
truly native PyTables files with those tools while retaining fully
compatibility with PyTables format. That is perfectly possible, and in this
appendix is presented the format that you should endow to your own-generated
files in order to get a fully PyTables compatible file.

We are going to describe the *2.0 version of PyTables file format*
(introduced in PyTables version 2.0). As time goes by, some changes might be
introduced (and documented here) in order to cope with new necessities.
However, the changes will be carefully pondered so as to ensure backward
compatibility whenever is possible.

A PyTables file is composed with arbitrarily large amounts of HDF5 groups
(Groups in PyTables naming scheme) and datasets (Leaves in PyTables naming
scheme). For groups, the only requirements are that they must have some
*system attributes* available. By convention, system attributes in PyTables
are written in upper case, and user attributes in lower case but this is not
enforced by the software. In the case of datasets, besides the mandatory
system attributes, some conditions are further needed in their storage
layout, as well as in the datatypes used in there, as we will see shortly.

As a final remark, you can use any filter as you want to create a PyTables
file, provided that the filter is a standard one in HDF5, like *zlib*,
*shuffle* or *szip* (although the last one can not be used from within
PyTables to create a new file, datasets compressed with szip can be read,
because it is the HDF5 library which do the decompression transparently).


.. currentmodule:: tables

Mandatory attributes for a File
-------------------------------
The File object is, in fact, an special HDF5 *group* structure that is *root*
for the rest of the objects on the object tree. The next attributes are
mandatory for the HDF5 *root group* structure in PyTables files:

* *CLASS*: This attribute should always be set to 'GROUP' for group
  structures.
* *PYTABLES_FORMAT_VERSION*: It represents the internal format version, and
  currently should be set to the '2.0' string.
* *TITLE*: A string where the user can put some description on what is this
  group used for.
* *VERSION*: Should contains the string '1.0'.


Mandatory attributes for a Group
--------------------------------
The next attributes are mandatory for *group* structures:

* *CLASS*: This attribute should always be set to 'GROUP' for group structures.
* *TITLE*: A string where the user can put some description on what is this
  group used for.
* *VERSION*: Should contains the string '1.0'.


Optional attributes for a Group
-------------------------------
The next attributes are optional for *group* structures:

* *FILTERS*: When present, this attribute contains the filter properties (a
  Filters instance, see section :ref:`FiltersClassDescr`) that may be
  inherited by leaves or groups created immediately under this group. This is
  a packed 64-bit integer structure, where

  - *byte 0* (the least-significant byte) is the compression level
    (complevel).
  - *byte 1* is the compression library used (complib): 0 when irrelevant, 1
    for Zlib, 2 for LZO and 3 for Bzip2.
  - *byte 2* indicates which parameterless filters are enabled (shuffle and
    fletcher32): bit 0 is for *Shuffle* while bit 1 is for*Fletcher32*.
  - other bytes are reserved for future use.


Mandatory attributes, storage layout and supported data types for Leaves
------------------------------------------------------------------------
This depends on the kind of Leaf. The format for each type follows.


.. _TableFormatDescr:

Table format
~~~~~~~~~~~~

Mandatory attributes
^^^^^^^^^^^^^^^^^^^^
The next attributes are mandatory for *table* structures:

* *CLASS*: Must be set to 'TABLE'.
* *TITLE*: A string where the user can put some description on what is this
  dataset used for.
* *VERSION*: Should contain the string '2.6'.
* *FIELD_X_NAME*: It contains the names of the different fields. The X means
  the number of the field, zero-based (beware, order do matter). You should
  add as many attributes of this kind as fields you have in your records.
* *FIELD_X_FILL*: It contains the default values of the different fields. All
  the datatypes are supported natively, except for complex types that are
  currently serialized using Pickle.  The X means the number of the field,
  zero-based (beware, order do matter). You should add as many attributes of
  this kind as fields you have in your records.  These fields are meant for
  saving the default values persistently and their existence is optional.
* *NROWS*: This should contain the number of *compound* data type entries in
  the dataset. It must be an *int* data type.


Storage Layout
^^^^^^^^^^^^^^
A Table has a *dataspace* with a *1-dimensional chunked* layout.

Datatypes supported
^^^^^^^^^^^^^^^^^^^
The datatype of the elements (rows) of Table must be the H5T_COMPOUND
*compound* data type, and each of these compound components must be built
with only the next HDF5 data types *classes*:

* *H5T_BITFIELD*: This class is used to represent the Bool type. Such a type
  must be build using a H5T_NATIVE_B8 datatype, followed by a HDF5
  H5Tset_precision call to set its precision to be just 1 bit.
* *H5T_INTEGER*: This includes the next data types:
    * *H5T_NATIVE_SCHAR*: This represents a *signed char* C type, but it is
      effectively used to represent an Int8 type.
    * *H5T_NATIVE_UCHAR*:  This represents an *unsigned char* C type, but it
      is effectively used to represent an UInt8 type.
    * *H5T_NATIVE_SHORT*: This represents a *short* C type, and it is
      effectively used to represent an Int16 type.
    * *H5T_NATIVE_USHORT*: This represents an *unsigned short* C type, and it
      is effectively used to represent an UInt16 type.
    * *H5T_NATIVE_INT*: This represents an *int* C type, and it is
      effectively used to represent an Int32 type.
    * *H5T_NATIVE_UINT*: This represents an *unsigned int* C type, and it is
      effectively used to represent an UInt32 type.
    * *H5T_NATIVE_LONG*: This represents a *long* C type, and it is
      effectively used to represent an Int32 or an Int64, depending on
      whether you are running a 32-bit or 64-bit architecture.
    * *H5T_NATIVE_ULONG*: This represents an *unsigned long* C type, and it
      is effectively used to represent an UInt32 or an UInt64, depending on
      whether you are running a 32-bit or 64-bit architecture.
    * *H5T_NATIVE_LLONG*: This represents a *long long* C type (__int64, if
      you are using a Windows system) and it is effectively used to represent
      an Int64 type.
    * *H5T_NATIVE_ULLONG*: This represents an *unsigned long long* C type
      (beware: this type does not have a correspondence on Windows systems)
      and it is effectively used to represent an UInt64 type.
* *H5T_FLOAT*: This includes the next datatypes:
    * *H5T_NATIVE_FLOAT*: This represents a *float* C type and it is
      effectively used to represent an Float32 type.
    * *H5T_NATIVE_DOUBLE*: This represents a *double* C type and it is
      effectively used to represent an Float64 type.
* *H5T_TIME*: This includes the next datatypes:
    * *H5T_UNIX_D32*: This represents a POSIX *time_t* C type and it is
      effectively used to represent a 'Time32' aliasing type, which
      corresponds to an Int32 type.
    * *H5T_UNIX_D64*: This represents a POSIX *struct timeval* C type and it
      is effectively used to represent a 'Time64' aliasing type, which
      corresponds to a Float64 type.
* *H5T_STRING*: The datatype used to describe strings in PyTables is H5T_C_S1
  (i.e. a *string* C type) followed with a call to the HDF5 H5Tset_size()
  function to set their length.
* *H5T_ARRAY*: This allows the construction of homogeneous, multidimensional
  arrays, so that you can include such objects in compound records. The types
  supported as elements of H5T_ARRAY data types are the ones described above.
  Currently, PyTables does not support nested H5T_ARRAY types.
* *H5T_COMPOUND*: This allows the support for datatypes that are compounds of
  compounds (this is also known as *nested types* along this manual).

  This support can also be used for defining complex numbers. Its format is
  described below:

  The H5T_COMPOUND type class contains two members. Both members must have
  the H5T_FLOAT atomic datatype class. The name of the first member should be
  "r" and represents the real part. The name of the second member should be
  "i" and represents the imaginary part. The *precision* property of both of
  the H5T_FLOAT members must be either 32 significant bits (e.g.
  H5T_NATIVE_FLOAT) or 64 significant bits (e.g. H5T_NATIVE_DOUBLE). They
  represent Complex32 and Complex64 types respectively.


Array format
~~~~~~~~~~~~

Mandatory attributes
^^^^^^^^^^^^^^^^^^^^
The next attributes are mandatory for *array* structures:

* *CLASS*: Must be set to 'ARRAY'.
* *TITLE*: A string where the user can put some description on what is this
  dataset used for.
* *VERSION*: Should contain the string '2.3'.


Storage Layout
^^^^^^^^^^^^^^
An Array has a *dataspace* with a *N-dimensional contiguous* layout (if you
prefer a *chunked* layout see EArray below).


Datatypes supported
^^^^^^^^^^^^^^^^^^^
The elements of Array must have either HDF5 *atomic* data types or a
*compound* data type representing a complex number. The atomic data types can
currently be one of the next HDF5 data type *classes*: H5T_BITFIELD,
H5T_INTEGER, H5T_FLOAT and H5T_STRING. The H5T_TIME class is also supported
for reading existing Array objects, but not for creating them. See the Table
format description in :ref:`TableFormatDescr` for more info about these
types.

In addition to the HDF5 atomic data types, the Array format supports complex
numbers with the H5T_COMPOUND data type class.
See the Table format description in :ref:`TableFormatDescr` for more info
about this special type.

You should note that H5T_ARRAY class datatypes are not allowed in Array
objects.


CArray format
~~~~~~~~~~~~~

Mandatory attributes
^^^^^^^^^^^^^^^^^^^^
The next attributes are mandatory for *CArray* structures:

* *CLASS*: Must be set to 'CARRAY'.
* *TITLE*: A string where the user can put some description on what is this
  dataset used for.
* *VERSION*: Should contain the string '1.0'.


Storage Layout
^^^^^^^^^^^^^^
An CArray has a *dataspace* with a *N-dimensional chunked* layout.

Datatypes supported
^^^^^^^^^^^^^^^^^^^
The elements of CArray must have either HDF5 *atomic* data types or a
*compound* data type representing a complex number. The atomic data types can
currently be one of the next HDF5 data type *classes*: H5T_BITFIELD,
H5T_INTEGER, H5T_FLOAT and H5T_STRING. The H5T_TIME class is also supported
for reading existing CArray objects, but not for creating them. See the Table
format description in :ref:`TableFormatDescr` for more info about these
types.

In addition to the HDF5 atomic data types, the CArray format supports complex
numbers with the H5T_COMPOUND data type class.
See the Table format description in :ref:`TableFormatDescr` for more info
about this special type.

You should note that H5T_ARRAY class datatypes are not allowed yet in Array
objects.


EArray format
~~~~~~~~~~~~~

Mandatory attributes
^^^^^^^^^^^^^^^^^^^^
The next attributes are mandatory for *earray* structures:

* *CLASS*: Must be set to 'EARRAY'.
* *EXTDIM*: (*Integer*) Must be set to the extendable dimension. Only one
  extendable dimension is supported right now.
* *TITLE*: A string where the user can put some description on what is this
  dataset used for.
* *VERSION*: Should contain the string '1.3'.


Storage Layout
^^^^^^^^^^^^^^
An EArray has a *dataspace* with a *N-dimensional chunked* layout.


Datatypes supported
^^^^^^^^^^^^^^^^^^^
The elements of EArray are allowed to have the same data types as for the
elements in the Array format. They can be one of the HDF5 *atomic* data type
*classes*: H5T_BITFIELD, H5T_INTEGER, H5T_FLOAT, H5T_TIME or H5T_STRING, see
the Table format description in :ref:`TableFormatDescr` for more info about
these types. They can also be a H5T_COMPOUND datatype representing a complex
number, see the Table format description in :ref:`TableFormatDescr`.

You should note that H5T_ARRAY class data types are not allowed in EArray
objects.


.. _VLArrayFormatDescr:

VLArray format
~~~~~~~~~~~~~~

Mandatory attributes
^^^^^^^^^^^^^^^^^^^^
The next attributes are mandatory for *vlarray* structures:

* *CLASS*: Must be set to 'VLARRAY'.
* *PSEUDOATOM*: This is used so as to specify the kind of pseudo-atom (see
  :ref:`VLArrayFormatDescr`) for the VLArray. It can take the values
  'vlstring', 'vlunicode' or 'object'. If your atom is not a pseudo-atom then
  you should not specify it.
* *TITLE*: A string where the user can put some description on what is this
  dataset used for.
* *VERSION*: Should contain the string '1.3'.


Storage Layout
^^^^^^^^^^^^^^
An VLArray has a *dataspace* with a *1-dimensional chunked* layout.


Data types supported
^^^^^^^^^^^^^^^^^^^^
The data type of the elements (rows) of VLArray objects must be the H5T_VLEN
*variable-length* (or VL for short) datatype, and the base datatype specified
for the VL datatype can be of any *atomic* HDF5 datatype that is listed in
the Table format description :ref:`TableFormatDescr`.  That includes the
classes:

- H5T_BITFIELD
- H5T_INTEGER
- H5T_FLOAT
- H5T_TIME
- H5T_STRING
- H5T_ARRAY

They can also be a H5T_COMPOUND data type representing a complex number, see
the Table format description in :ref:`TableFormatDescr` for a detailed
description.

You should note that this does not include another VL datatype, or a compound
datatype that does not fit the description of a complex number. Note as well
that, for object and vlstring pseudo-atoms, the base for the VL datatype is
always a H5T_NATIVE_UCHAR (H5T_NATIVE_UINT for vlunicode). That means that
the complete row entry in the dataset has to be used in order to fully
serialize the object or the variable length string.


Optional attributes for Leaves
------------------------------
The next attributes are optional for *leaves*:

* *FLAVOR*: This is meant to provide the information about the kind of object
  kept in the Leaf, i.e. when the dataset is read, it will be converted to
  the indicated flavor.
  It can take one the next string values:

    * *"numpy"*: Read data (structures arrays, arrays, records, scalars) will
      be returned as NumPy objects.
    * *"python"*: Read data will be returned as Python lists, tuples, or
      scalars.

