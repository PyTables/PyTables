.. _parameter_files:

PyTables parameter files
========================

.. currentmodule:: tables.parameters

PyTables issues warnings when certain limits are exceeded.  Those limits are
not intrinsic limitations of the underlying software, but rather are
proactive measures to avoid large resource consumptions.  The default limits
should be enough for most of cases, and users should try to respect them.
However, in some situations, it can be convenient to increase (or decrease)
these limits.

Also, and in order to get maximum performance, PyTables implements a series
of sophisticated features, like I/O buffers or different kind of caches (for
nodes, chunks and other internal metadata).  These features comes with a
default set of parameters that ensures a decent performance in most of
situations.  But, as there is always a need for every case, it is handy to
have the possibility to fine-tune some of these parameters.

Because of these reasons, PyTables implements a couple of ways to change the
values of these parameters.  All the *tunable* parameters live in the
:file:`tables/parameters.py`.  The user can choose to change them in the
parameter files themselves for a global and persistent change.  Moreover, if
he wants a finer control, he can pass any of these parameters directly to the
:func:`tables.open_file` function, and the new parameters will only take
effect in the corresponding file (the defaults will continue to be in the
parameter files).

A description of all of the tunable parameters follows.  As the defaults
stated here may change from release to release, please check with your actual
parameter files so as to know your actual default values.

.. warning::

    Changing the next parameters may have a very bad effect in the resource
    consumption and performance of your PyTables scripts.

    Please be careful when touching these!


Tunable parameters in parameters.py
-----------------------------------

Recommended maximum values
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autodata:: MAX_COLUMNS

.. autodata:: MAX_NODE_ATTRS

.. autodata:: MAX_GROUP_WIDTH

.. autodata:: MAX_TREE_DEPTH

.. autodata:: MAX_UNDO_PATH_LENGTH


Cache limits
~~~~~~~~~~~~
.. autodata:: CHUNK_CACHE_NELMTS

.. autodata:: CHUNK_CACHE_PREEMPT

.. autodata:: CHUNK_CACHE_SIZE

.. autodata:: COND_CACHE_SLOTS

.. autodata:: METADATA_CACHE_SIZE

.. autodata:: NODE_CACHE_SLOTS


Parameters for the different internal caches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autodata:: BOUNDS_MAX_SIZE

.. autodata:: BOUNDS_MAX_SLOTS

.. autodata:: ITERSEQ_MAX_ELEMENTS

.. autodata:: ITERSEQ_MAX_SIZE

.. autodata:: ITERSEQ_MAX_SLOTS

.. autodata:: LIMBOUNDS_MAX_SIZE

.. autodata:: LIMBOUNDS_MAX_SLOTS

.. autodata:: TABLE_MAX_SIZE

.. autodata:: SORTED_MAX_SIZE

.. autodata:: SORTEDLR_MAX_SIZE

.. autodata:: SORTEDLR_MAX_SLOTS


Parameters for general cache behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. warning::

    The next parameters will not take any effect if passed to the open_file()
    function, so they can only be changed in a *global* way.  You can change
    them in the file, but this is strongly discouraged unless you know well
    what you are doing.

.. autodata:: DISABLE_EVERY_CYCLES

.. autodata:: ENABLE_EVERY_CYCLES

.. autodata:: LOWEST_HIT_RATIO


Parameters for the I/O buffer in Leaf objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autodata:: IO_BUFFER_SIZE

.. autodata:: BUFFER_TIMES


Miscellaneous
~~~~~~~~~~~~~

.. autodata:: EXPECTED_ROWS_EARRAY

.. autodata:: EXPECTED_ROWS_TABLE

.. autodata:: PYTABLES_SYS_ATTRS

.. autodata:: MAX_NUMEXPR_THREADS

.. autodata:: MAX_BLOSC_THREADS


HDF5 driver management
~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: DRIVER

.. autodata:: DRIVER_DIRECT_ALIGNMENT

.. autodata:: DRIVER_DIRECT_BLOCK_SIZE

.. autodata:: DRIVER_DIRECT_CBUF_SIZE

.. autodata:: DRIVER_CORE_INCREMENT

.. autodata:: DRIVER_CORE_BACKING_STORE

.. autodata:: DRIVER_CORE_IMAGE

.. autodata:: DRIVER_SPLIT_META_EXT

.. autodata:: DRIVER_SPLIT_RAW_EXT
