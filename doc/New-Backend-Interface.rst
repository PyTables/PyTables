New Backend Interface for PyTables 4
====================================

A group of developers gathered (namely,
Andrea Bedini, Anthony Scopatz, Thomas Caswell, Pablo Larraondo, Rui Yang and Francesc Alted)
in Perth during the days of 8–11 August 2016
to define a new way to access I/O that allows a new version
of PyTables (4) to use different backends.  The main
goal is to use this for interfacing h5py for HDF5
access, but nothing prevents us from creating interfaces
with other backends in the future.

Interface
=========

The whole idea is to define a few abstract classes and
then to provide concrete implementations for them.  For this, we
have created a new
`pt4 branch <https://github.com/PyTables/PyTables/tree/pt4>`_.  The
abstract classes that we found form a minimal set are in
`tables/abc.py <https://github.com/PyTables/PyTables/blob/pt4/tables/abc.py>`_.

The concrete implementation for h5py is in `tables/backend_h5py.py
<https://github.com/PyTables/PyTables/blob/pt4/tables/backend_h5py.py>`_.
The new high level implementation in PyTables that uses
the new interface is in the `tables/core
<https://github.com/PyTables/PyTables/tree/pt4/tables/core>`_ subpackage.

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


Work done during the Perth hackfest
===================================

During our meeting in Perth, we managed to make quite a good progress
in implementing preliminary versions of the Table and Array interfaces on top of h5py.
We initially centered our efforts in what probably is the most important
object in PyTables: the Table class, and we are happy to report that all
the basic tests (including in-kernel queries) are passing already.

In regards of the Table object, we have implemented buffered I/O so that
preliminary benchmarks are showing that appends and regular and in-kernel
queries can reach a performance that is similar to the original PyTables.

Other objects like Array are there, but they are still far from passing a significant
amount of tests.  Besides, we did not started the job for CArray, EArray and VLArray
yet.  As indexes need CArray and EArray, we still need to implement those
before tackling the indexes port.

Another important feature of PyTables, the node cache, is being rewritten in a
much simplified way, and it already works for some simplistic situations,
but there are still some significant corners that need quite a bit of work.

Finally, it is important to remark that we decided to do a major rewrite of many features
of PyTables (most specially the Table iterators, which are now much simplfied and
expressed in terms of Python generators).  Another major decision that we made is that the new
PyTables (probably PyTables 4) will only support Python 3.5 or higher; this will
allow to use many new features introduced in both the language and its standard
library.

Help wanted
===========

Despite the great start in Perth, there are still a lot of things to do,
so if you think this is a good plan and would like to collaborate, you are
welcome.  Just drop by the pytables-dev@googlegroups.com and say hi.

Thanks
======

This first hackfest has been possible mainly thanks to fundings
from Curtin Institute for Computation, but also from Cisco Systems, Inc., NumFocus
and the University of California.

What's Next
===========

In order to make more progress on this task, we would need more hackfests or similar, so
if you want to fund them, please contact us at the pytables-dev@googlegroups.com list.
