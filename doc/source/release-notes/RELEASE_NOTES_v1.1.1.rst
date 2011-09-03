==============================
 What's new in PyTables 1.1.1
==============================


:Author: Francesc Altet
:Contact: faltet@carabos.com
:Author: Ivan Vilata i Balaguer
:Contact: ivilata@carabos.com


This document details the modifications to PyTables since version 1.0.
Its main purpose is help you ensure that your programs will be runnable
when you switch from PyTables 1.0 to PyTables 1.1.1.


API additions
=============

- None

Backward-incompatible changes
=============================

- ``Table.read()`` raises a ``KeyError`` instead of a ``ValueError``
  when a nonexistent field name is specified, for consistency with other
  methods.  The same goes for the ``col()`` method.

- ``File.__contains__()`` returns a true value when it is asked for an
  existent node, be it visible or not.  This is more consistent with
  ``Group.__contains__()``.


API refinements
===============

- Using ``table.cols['colname']`` is deprecated.  The usage of
  ``table.cols._f_col('colname')`` (with the new ``Cols._f_col()``
  method) is preferred.

Bug fixes (affecting API)
=========================

- None


----

  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 72
.. End:
