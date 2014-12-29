============================
 What's new in PyTables 1.0
============================


:Author: Francesc Altet
:Contact: faltet@carabos.com
:Author: Ivan Vilata i Balaguer
:Contact: ivilata@carabos.com


This document details the modifications to PyTables since version 0.9.1.  Its
main purpose is help you ensure that your programs will be runnable when you
switch from PyTables 0.9.1 to PyTables 1.0.


API additions
=============

- The new ``Table.col()`` method can be used to get a column from a table as a
  ``NumArray`` or ``CharArray`` object.  This is preferred over the syntax
  ``table['colname']``.

- The new ``Table.readCoordinates()`` method reads a set of rows given their
  indexes into an in-memory object.

- The new ``Table.readAppend()`` method Append rows fullfilling the condition
  to a destination table.

Backward-incompatible changes
=============================

- Trying to open a nonexistent file or a file of unknown type raises
  ``IOError`` instead of ``RuntimeError``.  Using an invalid mode raises
  ``ValueError`` instead of ``RuntimeError``.

- Getting a child node from a closed group raises ``ValueError`` instead of
  ``RuntimeError``.

- Running an action on the wrong type of node now (i.e. using
  ``file.listNodes()`` on a leaf) raises a ``TypeError`` instead of a
  ``NodeError``.

- Removing a non-existing child now raises a ``NoSuchNodeError``, instead of
  doing nothing.

- Removing a non-empty child group using ``del group.child`` fails with a
  ``NodeError`` instead of recursively doing the removal.  This is because of
  the potential damage it may cause when used inadvertently.  If a recursive
  behavior is needed, use the ``_f_remove()`` method of the child node.

- The `recursive` flag of ``Group._f_walkNodes()`` is ``True`` by default now.
  Before it was ``False``.

- Now, deleting and getting a non-existing attribute raises an
  ``AttributeError`` instead of a ``RuntimeError``.

- Swapped last two arguments of ``File.copyAttrs()`` to match the other
  methods.  Please use ``File.copyNodeAttrs()`` anyway.

- Failing to infer the size of a string column raises ``ValueError`` instead
  of ``RuntimeError``.

- Excessive table column name length and number of columns now raise
  ``ValueError`` instead of ``IndexError`` and ``NameError``.

- Excessive table row length now raises ``ValueError`` instead of
  ``RuntimeError``.

- ``table[integer]`` returns a ``numarray.records.Record`` object instead of a
  tuple.  This was the original behavior before PyTables 0.9 and proved to be
  more consistent than the last one (tables do not have an explicit ordering
  of columns).

- Specifying a nonexistent column in ``Table.read()`` raises a ``ValueError``
  instead of a ``LookupError``.

- When ``start >= stop`` an empty iterator is returned by ``Table.iterrows()``
  instead of an empty ``RecArray``.  Thanks to Ashley Walsh for noting this.

- The interface of ``isHDF5File()`` and ``isPyTablesFile()`` file has been
  unified so that they both return true or false values on success and raise
  ``HDF5ExtError`` or errors.  The true value in ``isPyTablesFile()`` is the
  format version string of the file.

- ``Table.whereIndexed()`` and ``Table.whereInRange()`` are now *private*
  methods, since the ``Table.where()`` method is able to choose the most
  adequate option.

- The global variables ``ExtVersion`` and ``HDF5Version`` have been renamed to
  ``extVersion`` and ``hdf5Version``, respectively.

- ``whichLibVersion()`` returns ``None`` on querying unavailable libraries,
  and raises ``ValueError`` on unknown ones.

The following modifications, though being (strictly speaking) modifications of
the API, will most probably not cause compatibility problems (but your mileage
may vary):

- The default values for ``name`` and ``classname`` arguments in
  ``File.getNode()`` are now ``None``, although the empty string is still
  allowed for backwards compatibility.  File hierarchy manipulation and
  attribute handling operations using those arguments have changed to reflect
  this.

- Copy operations (``Group._f_copyChildren()``, ``File.copyChildren()``,
  ``File.copyNode()``...) do no longer return a tuple with the new node and
  statistics.  Instead, they only return the new node, and statistics are
  collected via an optional keyword argument.

- The ``copyFile()`` function in ``File.py`` has changed its signature from::

      copyFile(srcfilename=None, dstfilename=None, title=None, filters=None,
               copyuserattrs=True, overwrite=False, stats=None)

  to::

      copyFile(srcfilename, dstfilename, overwrite=False, **kwargs)

  Thus, the function allows the same options as ``File.copyFile()``.

- The ``File.copyFile()`` method has changed its signature from::

      copyFile(self, dstfilename=None, title=None, filters=None,
               copyuserattrs=1, overwrite=0, stats=None):

  to::

      copyFile(self, dstfilename, overwrite=False, **kwargs)

  This enables this method to pass on arbitrary flags and options supported by
  copying methods of inner nodes in the hierarchy.

- The ``File.copyChildren()`` method has changed its signature from::

      copyChildren(self, wheresrc, wheredst, recursive=False, filters=None,
                   copyuserattrs=True, start=0, stop=None, step=1,
                   overwrite=False, stats=None)

  to::

      copyChildren(self, srcgroup, dstgroup, overwrite=False, recursive=False,
                   **kwargs):

  Thus, the function allows the same options as ``Group._f_copyChildren()``.

- The ``Group._f_copyChildren()`` method has changed its signature from::

      _f_copyChildren(self, where, recursive=False, filters=None,
                      copyuserattrs=True, start=0, stop=None, step=1,
                      overwrite=False, stats=None)

  to::

      _f_copyChildren(self, dstgroup, overwrite=False, recursive=False,
                      **kwargs)

  This enables this method to pass on arbitrary flags and options supported by
  copying methods of inner nodes in the group.

- Renamed ``srcFilename`` and ``dstFilename`` arguments in ``copyFile()`` and
  ``File.copyFile()`` to ``srcfilename`` and ``dstfilename``, respectively.
  Renamed ``whereSrc`` and ``whereDst`` arguments in ``File.copyChildren()``
  to ``wheresrc`` and ``wheredst``, respectively.  Renamed ``dstNode``
  argument in ``File.copyAttrs()`` to ``dstnode``.  Tose arguments should be
  easier to type in interactive sessions (although 99% of the time it is not
  necessary to specify them).

- Renamed ``object`` argument in ``EArray.append()`` to ``sequence``.

- The ``rows`` argument in ``Table.append()`` is now compulsory.

- The ``start`` argument in ``Table.removeRows()`` is now compulsory.


API refinements
===============

- The ``isHDF5()`` function has been deprecated in favor of ``isHDF5File()``.

- Node attribute-handling methods in ``File`` have been renamed for a better
  coherence and understanding of their purpose:

  * ``getAttrNode()`` is now called ``getNodeAttr()``
  * ``setAttrNode()`` is now called ``setNodeAttr()``
  * ``delAttrNode()`` is now called ``delNodeAttr()``
  * ``copyAttrs()`` is now called ``copyNodeAttrs()``

  They keep their respective signatures, and the old versions still exist for
  backwards compatibility, though they issue a ``DeprecationWarning``.

- Using ``VLArray.append()`` with multiple arguments is now deprecated for its
  ambiguity.  You should put the arguments in a single sequence object (list,
  tuple, array...) and pass it as the only argument.

- Using ``table['colname']`` is deprecated.  Using ``table.col('colname')``
  (with the new ``col()`` method) is preferred.


Bug fixes (affecting API)
=========================

- ``Table.iterrows()`` returns an empty iterator when no rows are selected,
  instead of returning ``None``.


----

  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 78
.. End:
