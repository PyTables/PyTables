============================
 What's new in PyTables 1.2
============================


:Author: Francesc Altet
:Contact: faltet@carabos.com
:Author: Ivan Vilata i Balaguer
:Contact: ivilata@carabos.com


This document details the modifications to PyTables since version 1.1.  Its
main purpose is help you ensure that your programs will be runnable when you
switch from PyTables 1.1 to PyTables 1.2.


API additions
=============

- The user is now allowed to set arbitrary Python (non-persistent) attributes
  on any instance of ``Node``.  If the name matches that of a child node, the
  later will no longer be accessible via natural naming, but it will still be
  available via ``File.getNode()``, ``Group._f_getChild()`` and the group
  children dictionaries.

  Of course, this allows the user to overwrite internal (``^_[cfgv]_``)
  PyTables variables, but this is the way most Python packages work.

- The new ``Group._f_getChild()`` method allows to get a child node (be it
  visible or not) by its name.  This should be more intuitive that using
  ``getattr()`` or using the group children dictionaries.

- The new ``File.isVisibleNode()``, ``Node._f_isVisible()`` and
  ``Leaf.isVisible()`` methods tell whether a node is visible or not, i.e. if
  the node will appear in listing operations such as ``Group._f_listNodes()``.


Backward-incompatible changes
=============================

- ``File.objects``, ``File.groups`` and ``File.leaves`` can no longer be used
  to iterate over all the nodes in the file.  However, they still may be used
  to access any node by its path.

- ``File.__contains__()`` returns a true value when it is asked for an
  existent node, be it visible or not.  This is more consistent with
  ``Group.__contains__()``.

- Using ``Group.__delattr__()`` to remove a child is no longer supported.
  Please use ``Group._f_remove()`` instead.

- The ``indexprops`` attribute is now present on all ``Table`` instances, be
  they indexed or not.  In the last case, it is ``None``.

- Table.getWhereList() now has flavor parameter equal to "NumArray" by
  default, which is more consistent with other methods. Before, flavor
  defaulted to "List".

- The ``extVersion`` variable does no longer exist.  It did not make much
  sense either, since the canonical version of the whole PyTables package is
  that of ``__version__``.

- The ``Row.nrow()`` has been converted into a property, so you have to
  replace any call to ``Row.nrow()`` into ``Row.nrow``.


Deprecated features
===================

- The ``objects``, ``groups`` and ``leaves`` mappings in ``File`` are retained
  only for compatibility purposes.  Using ``File.getNode()`` is recommended to
  access nodes, ``File.__contains__()`` to check for node existence, and
  ``File.walkNodes()`` for iteration purposes.  Using ``isinstance()`` and
  ``*isVisible*()`` methods is the preferred way of checking node type and
  visibility.

  Please note that the aforementioned mappings use the named methods
  internally, so the former have no special performance gains over the later.


API refinements
===============

- The ``isHDF5File()`` and ``isPyTablesFile()`` functions know how to handle
  nonexistent or unreadable files.  An ``IOError`` is raised in those cases.


Bug fixes (affecting API)
=========================

- None


----

  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 78
.. End:
