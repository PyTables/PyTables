============================
 What's new in PyTables 1.4
============================


:Author: Francesc Altet
:Contact: faltet@carabos.com
:Author: Ivan Vilata i Balaguer
:Contact: ivilata@carabos.com


This document details the modifications to PyTables since version 1.3.  Its
main purpose is help you ensure that your programs will be runnable when you
switch from PyTables 1.3 to PyTables 1.4.


API additions
=============

- The ``Table.getWhereList()`` method has got a new ``sort`` parameter.  The
  default now is to get the list of parameters unsorted.  Set ``sort`` to True
  to get the old behaviour.  We've done this to avoid unnecessary ordering of
  potentially large sets of coordinates.

- Node creation, copying and moving operations have received a new optional
  `createparents` argument.  When true, the necessary groups in the target
  path that don't exist at the time of running the operation are automatically
  created, so that the target group of the operation always exists.


Backward-incompatible changes
=============================

- None


Deprecated features
===================

- None


API refinements
===============

- ``Description._v_walk()`` has been renamed to ``_f_walk()``, since it is a
  public method, not a value.

- ``Table.removeIndex()`` now accepts a column name in addition to an
  ``Index`` instance (the later is deprecated).  This avoids the user having
  to retrieve the needed ``Index`` object.


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
