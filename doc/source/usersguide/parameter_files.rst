.. _parameter_files:

PyTables parameter files
========================

.. currentmodule:: tables

PyTables issues warnings when certain limits are exceeded.  Those
limits are not intrinsic limitations of the underlying software, but
rather are proactive measures to avoid large resource consumptions.  The
default limits should be enough for most of cases, and users should try
to respect them.  However, in some situations, it can be convenient to
increase (or decrease) these limits.

Also, and in order to get maximum performance, PyTables implements
a series of sophisticated features, like I/O buffers or different kind
of caches (for nodes, chunks and other internal metadata).  These
features comes with a default set of parameters that ensures a decent
performance in most of situations.  But, as there is always a need for
every case, it is handy to have the possibility to fine-tune some of
these parameters.

Because of these reasons, PyTables implements a couple of ways to
change the values of these parameters.  All
the *tunable* parameters live in
the tables/parameters.py.  The user can choose to
change them in the parameter files themselves for a global and
persistent change.  Moreover, if he wants a finer control, he can pass
any of these parameters directly to the :func:`tables.openFile`
function, and the new parameters
will only take effect in the corresponding file (the defaults will
continue to be in the parameter files).

A description of all of the tunable parameters follows.  As the
defaults stated here may change from release to release, please check
with your actual parameter files so as to know your actual default
values.

.. warning:: Changing the next parameters may have a very bad effect
   in the resource consumption and performance of your PyTables scripts.
   Please be careful when touching these!


.. currentmodule:: tables.parameters

Tunable parameters in parameters.py.
------------------------------------

Recommended maximum values
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: MAX_COLUMNS

    Maximum number of columns in Table
    objects before a PerformanceWarning is
    issued.  This limit is somewhat arbitrary and can be increased.


.. data:: MAX_NODE_ATTRS

    Maximum allowed number of attributes in a node

.. data:: MAX_GROUP_WIDTH

    Maximum depth in object tree allowed.


.. data:: MAX_UNDO_PATH_LENGTH

    Maximum length of paths allowed in undo/redo operations.


Cache limits
~~~~~~~~~~~~

.. data:: CHUNK_CACHE_NELMTS

    Number of elements for HDF5 chunk cache.


.. data:: CHUNK_CACHE_PREEMPT

    Chunk preemption policy.  This value should be between 0
    and 1 inclusive and indicates how much chunks that have been
    fully read are favored for preemption. A value of zero means
    fully read chunks are treated no differently than other
    chunks (the preemption is strictly LRU) while a value of one
    means fully read chunks are always preempted before other chunks.


.. data:: CHUNK_CACHE_SIZE

    Size (in bytes) for HDF5 chunk cache.

.. data:: COND_CACHE_SLOTS

    Maximum number of conditions for table queries to be
    kept in memory.


.. data:: METADATA_CACHE_SIZE

    Size (in bytes) of the HDF5 metadata cache.  This only
    takes effect if using HDF5 1.8.x series.


.. data:: NODE_CACHE_SLOTS

    Maximum number of unreferenced nodes to be kept in
    memory.

    If positive, this is the number
    of *unreferenced* nodes to be kept in the
    metadata cache.  Least recently used nodes are unloaded from
    memory when this number of loaded nodes is reached. To load
    a node again, simply access it as usual. Nodes referenced by
    user variables are not taken into account nor
    unloaded.

    Negative value means that all the touched nodes will be
    kept in an internal dictionary.  This is the faster way to
    load/retrieve nodes.  However, and in order to avoid a large
    memory consumption, the user will be warned when the number
    of loaded nodes will reach
    the -NODE_CACHE_SLOTS value.

    Finally, a value of zero means that any cache mechanism
    is disabled.


Parameters for the different internal caches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: BOUNDS_MAX_SIZE

    The maximum size for bounds values cached during index lookups.


.. data:: BOUNDS_MAX_SLOTS

    The maximum number of slots for the BOUNDS cache.


.. data:: ITERSEQ_MAX_ELEMENTS

    The maximum number of iterator elements cached in data lookups.


.. data:: ITERSEQ_MAX_SIZE

    The maximum space that will take ITERSEQ cache (in bytes).


.. data:: ITERSEQ_MAX_SLOTS

    The maximum number of slots in ITERSEQ cache.

.. data:: LIMBOUNDS_MAX_SIZE

    The maximum size for the query limits (for example,
    (lim1, lim2) in conditions like
    lim1 <= col < lim2) cached during
    index lookups (in bytes).


.. data:: LIMBOUNDS_MAX_SLOTS

    The maximum number of slots for LIMBOUNDS cache.


.. data:: TABLE_MAX_SIZE

    The maximum size for table chunks cached during index queries.


.. data:: SORTED_MAX_SIZE

    The maximum size for sorted values cached during index lookups.


.. data:: SORTEDLR_MAX_SIZE

    The maximum size for chunks in last row cached in index
    lookups (in bytes).


.. data:: SORTEDLR_MAX_SLOTS

    The maximum number of chunks for SORTEDLR cache.


Parameters for general cache behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. warning:: The next parameters will not take any effect if passed to
   the openFile() function, so they can only be
   changed in a *global* way.  You can change
   them in the file, but this is strongly discouraged unless you know
   well what you are doing.

.. data:: DISABLE_EVERY_CYCLES

    The number of cycles in which a cache will be forced to
    be disabled if the hit ratio is lower than the
    LOWEST_HIT_RATIO (see below).  This value
    should provide time enough to check whether the cache is being
    efficient or not.


.. data:: ENABLE_EVERY_CYCLES

    The number of cycles in which a cache will be forced to
    be (re-)enabled, irregardless of the hit ratio. This will
    provide a chance for checking if we are in a better scenario
    for doing caching again.

.. data:: LOWEST_HIT_RATIO

    The minimum acceptable hit ratio for a cache to avoid
    disabling (and freeing) it.


Parameters for the I/O buffer in Leaf objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: IO_BUFFER_SIZE

    The PyTables internal buffer size for I/O purposes.
    Should not exceed the amount of highest level cache size in
    your CPU.


.. data:: BUFFER_TIMES

    The maximum buffersize/rowsize ratio before issuing a
    PerformanceWarning.


Miscellaneous
~~~~~~~~~~~~~

.. data:: EXPECTED_ROWS_EARRAY

    Default expected number of rows for EArray objects.


.. data:: EXPECTED_ROWS_TABLE

    Default expected number of rows for Table objects.


.. data:: PYTABLES_SYS_ATTRS

    Set this to False if you don't want
    to create PyTables system attributes in datasets.  Also, if
    set to False the possible existing system
    attributes are not considered for guessing the class of the
    node during its loading from disk (this work is delegated to
    the PyTables' class discoverer function for general HDF5 files).


.. data:: MAX_THREADS

    The maximum number of threads that PyTables should use
    internally (mainly in Blosc and Numexpr currently).  If
    None, it is automatically set to the
    number of cores in your machine. In general, it is a good
    idea to set this to the number of cores in your machine or,
    when your machine has many of them (e.g. > 4), perhaps one
    less than this.

