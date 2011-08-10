.. _NestedRecArrayClassDescr:

Using nested record arrays (deprecated)
=======================================

Introduction
------------
Nested record arrays are a generalization of the record array
concept as it appears in the numarray
package. Basically, a nested record array is a record array that
supports nested datatypes. It means that columns can contain not only
regular datatypes but also nested datatypes.

.. warning:: PyTables nested record arrays were implemented to overcome a
   limitation of the record arrays in the numarray
   package.  However, as this functionality is already present in
   NumPy, current users should not need the package
   tables.nra anymore and it will be deprecated
   soon.

Each nested record array is a NestedRecArray
object in the tables.nra package. Nested record
arrays are intended to be as compatible as possible with ordinary
record arrays (in fact the NestedRecArray class
inherits from RecArray). As a consequence, the user
can deal with nested record arrays nearly in the same way that he does
with ordinary record arrays.

The easiest way to create a nested record array is to use the
array() function in the
tables.nra package. The only difference between
this function and its non-nested capable analogous is that now, we
*must* provide an structure for the buffer being
stored. For instance::

    >>> from tables.nra import array
    >>> nra1 = array(
    ...     [(1, (0.5, 1.0), ('a1', 1j)), (2, (0, 0), ('a2', 1+.1j))],
    ...     formats=['Int64', '(2,)Float32', ['a2', 'Complex64']])

will create a two rows nested record array with two regular
fields (columns), and one nested field with two sub-fields.

The field structure of the nested record array is specified by
the keyword argument formats. This argument only
supports sequences of strings and other sequences. Each string defines
the shape and type of a non-nested field. Each sequence contains the
formats of the sub-fields of a nested field. Optionally, we can also
pass an additional names keyword argument
containing the names of fields and sub-fields::

    >>> nra2 = array(
    ...     [(1, (0.5, 1.0), ('a1', 1j)), (2, (0, 0), ('a2', 1+.1j))],
    ...     names=['id', 'pos', ('info', ['name', 'value'])],
    ...     formats=['Int64', '(2,)Float32', ['a2', 'Complex64']])

The names argument only supports lists of strings and 2-tuples.
Each string defines the name of a non-nested field. Each 2-tuple
contains the name of a nested field and a list describing the names of
its sub-fields. If the names argument is not passed
then all fields are automatically named (c1,
c2 etc. on each nested field) so, in our first
example, the fields will be named as ['c1', 'c2', ('c3',
['c1', 'c2'])].

Another way to specify the nested record array structure is to
use the descr keyword argument::

    >>> nra3 = array(
    ...     [(1, (0.5, 1.0), ('a1', 1j)), (2, (0, 0), ('a2', 1+.1j))],
    ...     descr=[('id', 'Int64'), ('pos', '(2,)Float32'),
    ...            ('info', [('name', 'a2'), ('value', 'Complex64')])])
    >>>
    >>> nra3
    array(
    [(1L, array([ 0.5,  1. ], type=Float32), ('a1', 1j)),
     (2L, array([ 0.,  0.], type=Float32), ('a2', (1+0.10000000000000001j)))],
    descr=[('id', 'Int64'), ('pos', '(2,)Float32'), ('info', [('name', 'a2'),
    ('value', 'Complex64')])],
    shape=2)
    >>>

The descr argument is a list of 2-tuples,
each of them describing a field. The first value in a tuple is the
name of the field, while the second one is a description of its
structure. If the second value is a string, it defines the format
(shape and type) of a non-nested field. Else, it is a list of 2-tuples
describing the sub-fields of a nested field.

As you can see, the descr list is a mix of
the names and formats arguments.
In fact, this argument is intended to replace
formats and names, so they
cannot be used at the same time.

Of course the structure of all three keyword arguments must
match that of the elements (rows) in the buffer
being stored.

Sometimes it is convenient to create nested arrays by processing
a set of columns. In these cases the function
fromarrays comes handy. This function works in a
very similar way to the array function, but the passed buffer is a
list of columns. For instance::

    >>> from tables.nra import fromarrays
    >>> nra = fromarrays([[1, 2], [4, 5]], descr=[('x', 'f8'),('y', 'f4')])
    >>>
    >>> nra
    array(
    [(1.0, 4.0),
     (2.0, 5.0)],
    descr=[('x', 'f8'), ('y', 'f4')],
    shape=2)

Columns can be passed as nested arrays, what makes really
straightforward to combine different nested arrays to get a new one,
as you can see in the following examples::

    >>> nra1 = fromarrays([nra, [7, 8]], descr=[('2D', [('x', 'f8'), ('y', 'f4')]),
    ... ('z', 'f4')])
    >>>
    >>> nra1
    array(
    [((1.0, 4.0), 7.0),
    ((2.0, 5.0), 8.0)],
    descr=[('2D', [('x', 'f8'), ('y', 'f4')]), ('z', 'f4')],
    shape=2)
    >>>
    >>> nra2 = fromarrays([nra1.field('2D/x'), nra1.field('z')], descr=[('x', 'f8'),
    ('z', 'f4')])
    >>>
    >>> nra2
    array(
    [(1.0, 7.0),
    (2.0, 8.0)],
    descr=[('x', 'f8'), ('z', 'f4')],
    shape=2)

Finally it's worth to mention a small group of utility functions
in the tables.nra.nestedrecords module,
makeFormats, makeNames and
makeDescr, that can be useful to obtain the
structure specification to be used with the array
and fromarrays functions. Given a description list,
makeFormats gets the corresponding
formats list. In the same way
makeNames gets the names list.
On the other hand the descr list can be obtained
from formats and names lists using the
makeDescr function. For example::

    >>> from tables.nra.nestedrecords import makeDescr, makeFormats, makeNames
    >>> descr =[('2D', [('x', 'f8'), ('y', 'f4')]),('z', 'f4')]
    >>>
    >>> formats = makeFormats(descr)
    >>> formats
    [['f8', 'f4'], 'f4']
    >>> names = makeNames(descr)
    >>> names
    [('2D', ['x', 'y']), 'z']
    >>> d1 = makeDescr(formats, names)
    >>> d1
    [('2D', [('x', 'f8'), ('y', 'f4')]), ('z', 'f4')]
    >>> # If no names are passed then they are automatically generated
    >>> d2 = makeDescr(formats)
    >>> d2
    [('c1', [('c1', 'f8'), ('c2', 'f4')]),('c2', 'f4')]


NestedRecArray methods
----------------------
To access the fields in the nested record array use the
field() method::

    >>> print nra2.field('id')
    [1, 2]
    >>>

The field() method accepts also names of
sub-fields. It will consist of several field name components separated
by the string '/'

This way of specifying the names of sub-fields is
*very* specific to the implementation of
numarray nested arrays of PyTables.
Particularly, if you are using NumPy arrays, keep in mind that
sub-fields in such arrays must be accessed one at a time, like
this: numpy_array['info']['name'], and not like
this: numpy_array['info/name']., for instance::

    >>> print nra2.field('info/name')
    ['a1', 'a2']
    >>>

Finally, the top level fields of the nested recarray can be
accessed passing an integer argument to the field()
method::

    >>> print nra2.field(1)
    [[ 0.5 1. ] [ 0.  0. ]]
    >>>

An alternative to the field() method is the
use of the fields attribute. It is intended mainly
for interactive usage in the Python console. For example::

    >>> nra2.fields.id
    [1, 2]
    >>> nra2.fields.info.fields.name
    ['a1', 'a2']
    >>>

Rows of nested recarrays can be read using the typical index
syntax. The rows are retrieved as NestedRecord
objects::

    >>> print nra2[0]
    (1L, array([ 0.5,  1. ], type=Float32), ('a1', 1j))
    >>>
    >>> nra2[0].__class__
    <class tables.nra.nestedrecords.NestedRecord at 0x413cbb9c>

Slicing is also supported in the usual way::

    >>> print nra2[0:2]
    NestedRecArray[
    (1L, array([ 0.5,  1. ], type=Float32), ('a1', 1j)),
    (2L, array([ 0.,  0.], type=Float32), ('a2', (1+0.10000000000000001j)))
    ]
    >>>

Another useful method is asRecArray(). It
converts a nested array to a non-nested equivalent array.

This method creates a new vanilla RecArray
instance equivalent to this one by flattening its fields. Only
bottom-level fields included in the array. Sub-fields are named by
pre-pending the names of their parent fields up to the top-level
fields, using '/' as a separator. The data area of
the array is copied into the new one. For example, calling
nra3.asRecArray() would return the same array as
calling::

    >>> ra = numarray.records.array(
    ...     [(1, (0.5, 1.0), 'a1', 1j), (2, (0, 0), 'a2', 1+.1j)],
    ...     names=['id', 'pos', 'info/name', 'info/value'],
    ...     formats=['Int64', '(2,)Float32', 'a2', 'Complex64'])

Note that the shape of multidimensional fields is kept.


NestedRecord objects
--------------------
Each element of the nested record array is a
NestedRecord, i.e. a Record with
support for nested datatypes. As said before, we can do indexing as
usual::

    >>> print nra1[0]
    (1, (0.5, 1.0), ('a1', 1j))
    >>>

Using NestedRecord objects is quite similar
to using Record objects. To get the data of a field
we use the field() method. As an argument to this
method we pass a field name. Sub-field names can be passed in the way
described for NestedRecArray.field(). The
fields attribute is also present and works as it
does in NestedRecArray.

Field data can be set with the setField()
method. It takes two arguments, the field name and its value.
Sub-field names can be passed as usual. Finally, the
asRecord() method converts a nested record into a
non-nested equivalent record.

