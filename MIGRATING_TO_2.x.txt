==================================
Migrating from PyTables 1.x to 2.x
==================================

:Author: Francesc Alted i Abad
:Contact: faltet@pytables.com
:Author: Ivan Vilata i Balaguer
:Contact: ivan@selidor.net


Next are described a series of issues that you must have in mind when
migrating from PyTables 1.x to PyTables 2.x series.


New type system
===============

In PyTables 2.x all the data types for leaves are described through a couple
of classes:

- ``Atom``: Describes homogeneous types of the atomic components in ``*Array``
   objects (``Array``, ``CArray``, ``EArray`` and ``VLArray``).

- ``Description``: Describes (possibly nested) heterogeneous types in
  ``Table`` objects.

So, in order to upgrade to the new type system, you must perform the next
replacements:

- ``*Array.stype`` --> ``*Array.atom.type`` (PyTables type)
- ``*Array.type`` --> ``*Array.atom.dtype`` (NumPy type)
- ``*Array.itemsize`` --> ``*Array.atom.itemsize`` (the size of the item)

Furthermore, the PyTables types (previously called "string types") have
changed to better adapt to NumPy conventions.  The next changes have been
applied:

- PyTables types are now written in lower case, so 'Type' becomes 'type'.  For
  example, 'Int64' becomes now 'int64'.

- 'CharType' --> 'string'

- 'Complex32', 'Complex64' --> 'complex64', 'complex128'.  Note that the
  numeric part of a 'complex' type refers now to the *size in bits* of the
  type and not to the precision, as before.

See Appendix I of the Users' Manual on supported data types for more
information on the new PyTables types.


Important changes in ``Atom`` specification
===========================================

- The ``dtype`` argument of ``EnumAtom`` and ``EnumCol`` constructors
  has been replaced by the ``base`` argument, which can take a
  full-blown atom, although it accepts bare PyTables types as well.
  This is a *mandatory* argument now.

- ``vlstring`` pseudo-atoms used in ``VLArray`` nodes do no longer imply UTF-8
  (nor any other) encoding, they only store and load *raw strings of bytes*.
  All encoding and decoding is left to the user.  Be warned that reading old
  files may yield raw UTF-8 encoded strings, which may be coverted back to
  Unicode in this way::

      unistr = vlarray[index].decode('utf-8')

  If you need to work with variable-length Unicode strings, you may want to
  use the new ``vlunicode`` pseudo-atom, which fully supports Unicode strings
  with no encoding hassles.

- Finally, ``Atom`` and ``Col`` are now abstract classes, so you can't use
  them to create atoms or column definitions of an arbitrary type.  If you
  know the particular type you need, use the proper subclass; otherwise, use
  the ``Atom.from_*()`` or ``Col.from_*()`` factory methods.  See the section
  on declarative classes in the reference.

  You are also advised to avoid using the inheritance of atoms to check for
  their kind or type; for that purpose, use their ``kind`` and ``type``
  attributes.


New query system
================

- In-kernel conditions, since they are based now in Numexpr, must be written
  *as strings*.  For example, a condition that in 1.x was stated as::

      result = [row['col2'] for row in table.where(table.cols.col1 == 1)]

  now should read::

      result = [row['col2'] for row in table.where('col1 == 1')]

  That means that complex selections are possible now::

      result = [ row['col2'] for row in
                 table.where('(col1 == 1) & (col3**4 > 1)') ]

- For the same reason, conditions for indexed columns must be written as
  strings as well.


New indexing system
===================

The indexing system has been totally rewritten from scratch for PyTables 2.0
Pro Edition (http://www.pytables.com/moin/PyTablesPro).  The new indexing
systemsame has been included into PyTables with release 2.3.  Due to this,
your existing indexes created with PyTables 1.x will be useless, and although
you will be able to continue using the actual data in files, you won't be
able to take advantage of any improvement in speed.

You will be offered the possibility to automatically re-create the indexes
in PyTables 1.x format to the new 2.0 format by using the ``ptrepack``
utility.


New meanings for atom shape and ``*Array`` shape argument
=========================================================

With PyTables 1.x, the atom shape was used for different goals depending on
the context it was used.  For example, in ``createEArray()``, the shape of the
atom was used to specify the *dataset shape* of the object on disk, while in
``CArray`` the same atom shape was used to specify the *chunk shape* of the
dataset on disk.  Moreover, for ``VLArray`` objects, the very same atom shape
specified the *type shape* of the data type.  As you see, all of these was
quite a mess.

Starting with PyTables 2.x, an ``Atom`` only specifies properties of the data
type (à la ``VLArray`` in 1.x).  This lets the door open for specifying
multidimensional data types (that can be part of another layer of
multidimensional datasets) in a consistent way along all the ``*Array``
objects in PyTables.

As a consequence of this, ``File.createCArray()`` and ``File.createVLArray()``
methods have received new parameters in order to make possible to specify the
shapes of the datasets as well as chunk sizes (in fact, it is possible now to
specify the latter for all the chunked leaves, see below).  Please have this
in mind during the migration process.

Another consequence is that, now that the meaning of the atom shape is clearly
defined, it has been chosen as the main object to describe homogeneous data
types in PyTables.  See the Users' Manual for more info on this.


New argument ``chunkshape`` of chunked leaves
=============================================

It is possible now to specify the chunk shape for all the chunked leaves in
PyTables (all except ``Array``).  With PyTables 1.x this value was
automatically calculated so as to achieve decent results in most of the
situations.  However, the user may be interested in specifying its own chunk
shape based on her own needs (although this should be done only by advanced
users).

Of course, if this parameter is not specified, a sensible default is
calculated for the size of the leave (which is recommended).

A new attribute called ``chunkshape`` has been added to all leaves.  It is
read-only (you can't change the size of chunks once you have created a leaf),
but it can be useful for inspection by advanced users.


New flavor specification
========================

As of 2.x, flavors can *only* be set through the ``flavor`` attribute of
leaves, and they are *persistent*, so changing a flavor requires that the file
be writable.

Flavors can no longer be set through ``File.create*()`` methods, nor the
``flavor`` argument previously found in some ``Table`` methods, nor through
``Atom`` constructors or the ``_v_flavor`` attribute of descriptions.


System attributes can be deleted now
====================================

The protection against removing system attributes (like ``FILTERS``,
``FLAVOR`` or ``CLASS``, to name only a few) has been completely removed.  It
is now the responsibility of the user to make a proper use of this freedom.
With this, users can get rid of all proprietary PyTables attributes if they
want to (for example, for making a file to look more like an HDF5 native one).


Byteorder issues
================

Now, all the data coming from reads and internal buffers is always converted
on-the-fly, if needed, to the *native* byteorder.  This represents a big
advantage in terms of speed when operating with objects coming from files that
have been created in machines with a byte ordering different from native.

Besides, all leaf constructors have received a new ``byteorder`` parameter
that allows specifying the byteorder of data on disk.  In particular, a
``_v_byteorder`` entry in a Table description is no longer honored and you
should use the aforementioned ``byteorder`` parameter.


Tunable internal buffer sizes
=============================

You can change the size of the internal buffers for I/O purposes of PyTables
by changing the value of the new public attribute ``nrowsinbuf`` that is
present in all leaves.  By default, this contains a sensible value so as to
achieve a good balance between speed and memory consumption.  Be careful when
changing it, if you don't want to get unwanted results (very slow I/O, huge
memory consumption...).


Changes to module names
=======================

If your application is directly accessing modules under the ``tables``
package, you need to know that *the names of all modules are now all in
lowercase*.  This allows one to tell apart the ``tables.Array`` *class* from
the ``tables.array`` *module* (which was also called ``tables.Array`` before).
This includes subpackages like ``tables.nodes.FileNode``.

On top of that, more-or-less independent modules have also been renamed and
some of them grouped into subpackages.  The most important are:

- The ``tables.netcdf3`` subpackage replaces the old ``tables.NetCDF`` module.
- The ``tables.nra`` subpackage replaces the old ``nestedrecords.py`` with the
  implementation of the ``NestedRecArray`` class.

Also, the ``tables.misc`` package includes utility modules which do not depend
on PyTables.


Other changes
=============

- ``Filters.complib`` is ``None`` for filter properties created with
  ``complevel=0`` (i.e. disabled compression, which is the default).
- 'non-relevant' --> 'irrelevant' (applied to byteorders)
- ``Table.colstypes`` --> ``Table.coltypes``
- ``Table.coltypes`` --> ``Table.coldtypes``
- Added ``Table.coldescr``, dictionary of the ``Col`` descriptions.
- ``Table.colshapes`` has disappeared.  You can get it this way::

       colshapes = dict( (name, col.shape)
                         for (name, col) in table.coldescr.iteritems() )

- ``Table.colitemsizes`` has disappeared.  You can get it this way::

       colitemsizes = dict( (name, col.itemsize)
                            for (name, col) in table.coldescr.iteritems() )

- ``Description._v_totalsize`` --> ``Description._v_itemsize``
- ``Description._v_itemsizes`` and ``Description._v_totalsizes`` have
  disappeared.

- ``Leaf._v_chunksize`` --> ``Leaf.chunkshape``


----

  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 78
.. End:
