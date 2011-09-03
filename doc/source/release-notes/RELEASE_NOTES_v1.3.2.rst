==============================
 What's new in PyTables 1.3.2
==============================


:Author: Francesc Altet
:Contact: faltet@carabos.com
:Author: Ivan Vilata i Balaguer
:Contact: ivilata@carabos.com


This document details the modifications to PyTables since version 1.2.  Its
main purpose is help you ensure that your programs will be runnable when you
switch from PyTables 1.2 to PyTables 1.3.2.


API additions
=============

- The ``Table.Cols`` accessor has received a new ``__setitem__()`` method that
  allows doing things like::

      table.cols[4] = record
      table.cols.x[4:1000:2] = array   # homogeneous column
      table.cols.Info[4:1000:2] = recarray   # nested column


Backward-incompatible changes
=============================

- None


Deprecated features
===================

- None


API refinements
===============

- ``Table.itersequence()`` has changed the default value for the ``sort``
  parameter.  It is now false by default, as it is not clear if this actually
  accelerates the iterator, so it is better to let the user do the proper
  checks (if interested).


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
