New Backend Interface for PyTables 4
====================================

A group of developers gathered in Perth to define
a new way to access I/O that allows a new version
of PyTables (4) to use different backends.  The main
goal is to use this for interfacing h5py for HDF5
access, but nothing prevents us from creating interfaces
with other backends in the future.

Interface
=========

The whole idea is to define a few abstract classes and
then to provide concrete implementations for them. The
abstract classes that we found form a minimal set are in
tables/abc.py

The concrete implementation is in tables/backend_h5py.
The new high level implementation in PyTables that uses
the new interface is in tables/core.py.

Implementation Plan
===================

We are currently working towards implementing the fundamental
building blocks in order to create basic leaves like Table and
all sorts of Arrays (Array, CArray, EArray, VLArray).  Then
we will proceed with implementing Group so that we can use
hierarchies as well as Attributes, so we can add metadata.

The second phase will be migrating specific features of PyTables
on top of the new infrastructure.  We will tackle out-of-core
computations first, and then the indexing (OPSI) engine.

The final phase will include the rest of the details, and more
specifically making sure than the current test suite is passing.

Help wanted
===========

If you think this is a good plan and want to collaborate, you are
welcome.  Just drop by the pytables-dev@googlegroups.com and say hi.
