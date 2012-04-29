.. _library_reference:

Library Reference
=================
PyTables implements several classes to represent the different
nodes in the object tree. They are named File,
Group, Leaf,
Table, Array,
CArray, EArray,
VLArray and UnImplemented. Another
one allows the user to complement the information on these different
objects; its name is AttributeSet. Finally, another
important class called IsDescription allows to build
a Table record description by declaring a subclass of
it. Many other classes are defined in PyTables, but they can be regarded
as helpers whose goal is mainly to declare the *data type
properties* of the different first class objects and will be
described at the end of this chapter as well.

An important function, called openFile is
responsible to create, open or append to files. In addition, a few
utility functions are defined to guess if the user supplied file is a
*PyTables* or *HDF5* file. These
are called isPyTablesFile() and
isHDF5File(), respectively. There exists also a
function called whichLibVersion() that informs about
the versions of the underlying C libraries (for example, HDF5 or
Zlib) and another called
print_versions() that prints all the versions of the
software that PyTables relies on. Finally, test()
lets you run the complete test suite from a Python console
interactively.

Let's start discussing the first-level variables and functions
available to the user, then the different classes defined in
PyTables.

.. currentmodule:: tables

tables variables and functions
------------------------------

Global variables
~~~~~~~~~~~~~~~~

.. data:: __version__

    The PyTables version number.

.. data:: hdf5Version

    The underlying HDF5 library version number.

.. data:: is_pro

    True for PyTables Professional edition, false otherwise.

    .. note:: PyTables Professional edition has been released
       under an open source license. Starting with version 2.3,
       PyTables includes all features of PyTables Pro.
       In order to reflect the presence of advanced
       features :data:`is_pro` is always
       set to True.  :data:`is_pro` should be
       considered *deprecated*.
       It will be removed in the next major release.

    .. deprecated:: 2.3

Global functions
~~~~~~~~~~~~~~~~
.. function:: copyFile(srcfilename, dstfilename, overwrite=False, **kwargs)

    An easy way of copying one PyTables file to another.

    This function allows you to copy an existing PyTables file
    named srcfilename to another file called
    dstfilename. The source file must exist and be
    readable. The destination file can be overwritten in place if
    existing by asserting the overwrite
    argument.

    This function is a shorthand for the
    :meth:`File.copyFile` method, which acts on an
    already opened file. kwargs takes keyword
    arguments used to customize the copying process. See the
    documentation of :meth:`File.copyFile` for a description of those
    arguments.

.. function:: isHDF5File(filename)

    Determine whether a file is in the HDF5 format.

    When successful, it returns a true value if the file is an
    HDF5 file, false otherwise. If there were problems identifying the
    file, an HDF5ExtError is raised.

.. function:: isPyTablesFile(filename)

    Determine whether a file is in the PyTables format.

    When successful, it returns the format version string if the
    file is a PyTables file, None otherwise.  If
    there were problems identifying the file,
    an HDF5ExtError is raised.

.. function:: lrange([start, ]stop[, step])

    Iterate over long ranges.

    This is similar to xrange(), but it
    allows 64-bit arguments on all platforms.  The results of the
    iteration are sequentially yielded in the form of
    numpy.int64 values, but getting random
    individual items is not supported.

    Because of the Python 32-bit limitation on object lengths,
    the length attribute (which is also a
    numpy.int64 value) should be used instead of
    the len() syntax.

    Default start and step
    arguments are supported in the same way as in
    xrange().  When the standard
    [x]range() Python objects support 64-bit
    arguments, this iterator will be deprecated.


.. function:: openFile(filename, mode='r', title='', rootUEP="/", filters=None, **kwargs)

    Open a PyTables (or generic HDF5) file and return a File object.

    Parameters
    ----------
    filename : str
        The name of the file (supports environment variable
        expansion). It is suggested that file names have any of the
        .h5, .hdf or .hdf5 extensions, although this is not mandatory.
    mode : str
        The mode to open the file. It can be one of the
        following:

        * *'r'*: Read-only; no data can be modified.
        * *'w'*: Write; a new file is created (an existing file with the same name would be deleted).
        * *'a'*: Append; an existing file is opened for reading and writing, and if the file
          does not exist it is created.
        * *'r+'*: It is similar to 'a', but the file must already exist.
    title : str
        If the file is to be created, a
        TITLE string attribute will be set on the
        root group with the given value. Otherwise, the title will
        be read from disk, and this will not have any effect.
    rootUEP : str
        The root User Entry Point. This is a group in the HDF5
        hierarchy which will be taken as the starting point to
        create the object tree. It can be whatever existing group in
        the file, named by its HDF5 path. If it does not exist, an
        HDF5ExtError is issued. Use this if you
        do not want to build the *entire* object
        tree, but rather only a *subtree* of it.
    filters : Filters
        An instance of the Filters (see
        :ref:`FiltersClassDescr`) class that provides
        information about the desired I/O filters applicable to the
        leaves that hang directly from the *root
        group*, unless other filter properties are
        specified for these leaves. Besides, if you do not specify
        filter properties for child groups, they will inherit these
        ones, which will in turn propagate to child nodes.

    Notes
    -----
    In addition, it recognizes the names of parameters present
    in :file:`tables/parameters.py` as additional keyword
    arguments.  See :ref:`parameter_files` for a
    detailed info on the supported parameters.

    .. note:: If you need to deal with a large number of nodes in an
       efficient way, please see :ref:`LRUOptim` for more info and advices about
       the integrated node cache engine.


.. function:: setBloscMaxThreads(nthreads)

    Set the maximum number of threads that Blosc can use.

    This actually overrides the :data:`tables.parameters.MAX_BLOSC_THREADS`
    setting in :mod:`tables.parameters`, so the new value will be effective
    until this function is called again or a new file with a different
    :data:`tables.parameters.MAX_BLOSC_THREADS` value is specified.

    Returns the previous setting for maximum threads.

.. function:: print_versions()

    Print all the versions of software that PyTables relies on.

.. function:: restrict_flavors(keep=['python'])

    Disable all flavors except those in keep.

    Providing an empty keep sequence implies
    disabling all flavors (but the internal one).  If the sequence is
    not specified, only optional flavors are disabled.

    .. important:: Once you disable a flavor, it can not be enabled again.

.. function:: split_type(type)

    Split a PyTables type into a PyTables
    kind and an item size.

    Returns a tuple of (kind, itemsize). If
    no item size is present in the type (in the
    form of a precision), the returned item size is
    None::

        >>> split_type('int32')
        ('int', 4)
        >>> split_type('string')
        ('string', None)
        >>> split_type('int20')
        Traceback (most recent call last):
        ...
        ValueError: precision must be a multiple of 8: 20
        >>> split_type('foo bar')
        Traceback (most recent call last):
        ...
        ValueError: malformed type: 'foo bar'

.. function:: test(verbose=False, heavy=False)

    Run all the tests in the test suite.

    If verbose is set, the test suite will
    emit messages with full verbosity (not recommended unless you are
    looking into a certain problem).

    If heavy is set, the test suite will be
    run in *heavy* mode (you should be careful with
    this because it can take a lot of time and resources from your
    computer).

.. function:: whichLibVersion(name)

    Get version information about a C library.

    If the library indicated by name is
    available, this function returns a 3-tuple containing the major
    library version as an integer, its full version as a string, and
    the version date as a string. If the library is not available,
    None is returned.

    The currently supported library names are
    hdf5, zlib,
    lzo and bzip2. If another
    name is given, a ValueError is raised.

.. function:: silenceHDF5Messages(silence=True)

   Silence (or re-enable) messages from the HDF5 C library.

   The *silence* parameter can be used control the behaviour and reset the
   standard HDF5 logging.

   .. versionadded:: 2.4

.. File class documentation in tables/file.py docstring

.. automodule:: tables.file


.. Node class documentation in tables/node.py docstring

.. automodule:: tables.node


.. Group class documentation in tables/group.py docstring

.. automodule:: tables.group


.. Leaf class documentation in tables/leaf.py docstring

.. automodule:: tables.leaf


.. Table class, Cols class, and Column class documentation in tables/table.py docstring

.. automodule:: tables.table


.. Description class, Col class, and IsDescription class documentation in
   tables/description.py docstring

.. automodule:: tables.description


.. _RowClassDescr:

The Row class
~~~~~~~~~~~~~
.. class:: Row

    Table row iterator and field accessor.

    Instances of this class are used to fetch and set the values
    of individual table fields.  It works very much like a dictionary,
    where keys are the pathnames or positions (extended slicing is
    supported) of the fields in the associated table in a specific row.

    This class provides an *iterator interface*
    so that you can use the same Row instance to
    access successive table rows one after the other.  There are also
    some important methods that are useful for accessing, adding and
    modifying values in tables.


Row instance variables
^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Row.nrow

    The current row number.

    This property is useful for knowing which row is being
    dealt with in the middle of a loop or iterator.


Row methods
^^^^^^^^^^^

.. method:: Row.append()

    Add a new row of data to the end of the dataset.

    Once you have filled the proper fields for the current
    row, calling this method actually appends the new data to the
    *output buffer* (which will eventually be
    dumped to disk).  If you have not set the value of a field, the
    default value of the column will be used.

    Example of use::

        row = table.row
        for i in xrange(nrows):
            row['col1'] = i-1
            row['col2'] = 'a'
            row['col3'] = -1.0
            row.append()
        table.flush()

    .. warning:: After completion of the loop in which :meth:`Row.append` has
       been called, it is always
       convenient to make a call to :meth:`Table.flush`
       in order to avoid losing the last rows that may still remain
       in internal buffers.


.. method:: Row.fetch_all_fields()

    Retrieve all the fields in the current row.

    Contrarily to row[:] (see :ref:`RowSpecialMethods`), this returns row data as a
    NumPy void scalar.  For instance::

        [row.fetch_all_fields() for row in table.where('col1 < 3')]

    will select all the rows that fulfill the given condition
    as a list of NumPy records.


.. method:: Row.update()

    Change the data of the current row in the dataset.

    This method allows you to modify values in a table when
    you are in the middle of a table iterator like
    :meth:`Table.iterrows` or :meth:`Table.where`.

    Once you have filled the proper fields for the current
    row, calling this method actually changes data in the
    *output buffer* (which will eventually be
    dumped to disk).  If you have not set the value of a field, its
    original value will be used.

    Examples of use::

        for row in table.iterrows(step=10):
            row['col1'] = row.nrow
            row['col2'] = 'b'
            row['col3'] = 0.0
            row.update()
        table.flush()

    which modifies every tenth row in table.  Or::

        for row in table.where('col1 > 3'):
            row['col1'] = row.nrow
            row['col2'] = 'b'
            row['col3'] = 0.0
            row.update()
        table.flush()

    which just updates the rows with values bigger than 3 in
    the first column.

    .. warning:: After completion of the loop in which :meth:`Row.update` has
       been called, it is always
       convenient to make a call to :meth:`Table.flush`
       in order to avoid losing changed rows that may still remain in
       internal buffers.


.. _RowSpecialMethods:

Row special methods
^^^^^^^^^^^^^^^^^^^

.. method:: Row.__contains__(item)

    Is item in this row?

    A true value is returned if item is
    found in current row, false otherwise.


.. method:: Row.__getitem__(key)

    Get the row field specified by the key.

    The key can be a string (the name of
    the field), an integer (the position of the field) or a slice
    (the range of field positions). When key is a
    slice, the returned value is a *tuple*
    containing the values of the specified fields.

    Examples of use::

        res = [row['var3'] for row in table.where('var2 < 20')]

    which selects the var3 field for all
    the rows that fulfil the condition. Or::

        res = [row[4] for row in table if row[1] < 20]

    which selects the field in the *4th*
    position for all the rows that fulfil the
    condition. Or::

        res = [row[:] for row in table if row['var2'] < 20]

    which selects the all the fields (in the form of a
    *tuple*) for all the rows that fulfil the
    condition. Or::

        res = [row[1::2] for row in table.iterrows(2, 3000, 3)]

    which selects all the fields in even positions (in the
    form of a *tuple*) for all the rows in the
    slice [2:3000:3].


.. method:: Row.__setitem__(key, value)

    Set the key row field to the specified value.

    Differently from its __getitem__()
    counterpart, in this case key can only be a
    string (the name of the field). The changes done via
    __setitem__() will not take effect on the
    data on disk until any of the :meth:`Row.append` or
    :meth:`Row.update` methods are called.

    Example of use::

        for row in table.iterrows(step=10):
            row['col1'] = row.nrow
            row['col2'] = 'b'
            row['col3'] = 0.0
            row.update()
        table.flush()

    which modifies every tenth row in the table.


.. Array class documentation in tables/array.py

.. automodule:: tables.array


.. CArray class documentation in tables/carray.py

.. automodule:: tables.carray


.. EArray class documentation in tables/earray.py

.. automodule:: tables.earray


.. VLArray class documentation in tables/vlarray.py

.. automodule:: tables.vlarray


.. Link class documentation in tables/link.py

.. automodule:: tables.link


.. UnImplemented class and Unknown class documentation in tables/unimplemented.py

.. automodule:: tables.unimplemented


.. AttributeSet class documentation in tables/attributeset.py

.. automodule:: tables.attributeset


.. All Atom and Pseudoatom class documentation in tables/atom.py

.. automodule:: tables.atom


Helper classes
.. --------------
.. move this section into its own file

This section describes some classes that do not fit in any other
section and that mainly serve for ancillary purposes.


.. Filters class documentation in tables/filters.py

.. automodule:: tables.filters


.. Index class documentation in tables/index.py

.. automodule:: tables.index


.. Enum class documentation in tables/misc/enum.py

.. automodule:: tables.misc.enum



The Expr class - a general-purpose expression evaluator
-------------------------------------------------------
.. class:: Expr

    Expr is a class for evaluating expressions
    containing array-like objects.  With it, you can evaluate expressions
    (like "3*a+4*b") that operate on arbitrary large
    arrays while optimizing the resources required to perform them
    (basically main memory and CPU cache memory).  It is similar to the
    Numexpr package (see :ref:`[NUMEXPR] <NUMEXPR>`), but in addition
    to NumPy objects, it also accepts disk-based homogeneous arrays, like
    the Array, CArray, EArray and Column PyTables objects.

    All the internal computations are performed via the Numexpr
    package, so all the broadcast and upcasting rules of Numexpr applies
    here too.  These rules are very similar to the NumPy ones, but with
    some exceptions due to the particularities of having to deal with
    potentially very large disk-based arrays.  Be sure to read the
    documentation of the
    Expr constructor and methods as well as that of
    Numexpr, if you want to fully grasp these particularities.


Expr instance variables
~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Expr.append_mode

    The appending mode for user-provided output containers.

.. attribute:: Expr.maindim

    Common main dimension for inputs in expression.

.. attribute:: Expr.names

    The names of variables in expression (list).

.. attribute:: Expr.out

    The user-provided container (if any) for the
    expression outcome.

.. attribute:: Expr.o_start

    The start range selection for the user-provided output.

.. attribute:: Expr.o_stop

    The stop range selection for the user-provided output.

.. attribute:: Expr.o_step

    The step range selection for the user-provided output.

.. attribute:: Expr.shape

    Common shape for the arrays in expression.

.. attribute:: Expr.values

    The values of variables in expression (list).


Expr special tuning variables for input/output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. warning:: The next parameters are meant only for advanced
   users.  Please do not touch them if you don't know what you are
   doing.

.. attribute:: Expr.BUFFERTIMES

    The maximum buffersize/rowsize ratio before issuing a
    *PerformanceWarning*.  The default is 1000.

.. attribute:: Expr.CHUNKTIMES

    The number of chunks in the input buffer per each
    variable in expression.  The default is 16.


Expr methods
~~~~~~~~~~~~

.. method:: Expr.__init__(expr, uservars=None, **kwargs)

    Compile the expression and initialize internal structures.

    expr must be specified as a string like "2*a+3*b".

    The uservars mapping may be used to
    define the variable names appearing in expr.
    This mapping should consist of identifier-like strings pointing
    to any Array, CArray, EArray,
    Column or NumPy ndarray instances (or even
    others which will be tried to be converted to ndarrays).

    When uservars is not provided
    or None, the current local and global
    namespace is sought instead of uservars.  It
    is also possible to pass just some of the variables in
    expression via the uservars mapping, and the
    rest will be retrieved from the current local and global
    namespaces.

    *kwargs* is meant to pass additional
    parameters to the Numexpr kernel.  This is basically the same as
    the *kwargs* argument
    in Numexpr.evaluate(), and is mainly meant
    for advanced use.

    After initialized, an Expr instance can
    be evaluated via its eval() method.  This
    class also provides an __iter__() method that
    iterates over all the resulting rows in expression.

    Example of use::

        >>> a = f.createArray('/', 'a', np.array([1,2,3]))
        >>> b = f.createArray('/', 'b', np.array([3,4,5]))
        >>> c = np.array([4,5,6])
        >>> expr = tb.Expr("2*a+b*c")   # initialize the expression
        >>> expr.eval()                 # evaluate it
        array([14, 24, 36])
        >>> sum(expr)                   # use as an iterator
        74

    where you can see that you can mix different containers in
    the expression (whenever shapes are consistent).

    You can also work with multidimensional arrays::

        >>> a2 = f.createArray('/', 'a2', np.array([[1,2],[3,4]]))
        >>> b2 = f.createArray('/', 'b2', np.array([[3,4],[5,6]]))
        >>> c2 = np.array([4,5])           # This will be broadcasted
        >>> expr = tb.Expr("2*a2+b2-c2")
        >>> expr.eval()
        array([[1, 3],
               [7, 9]])
        >>> sum(expr)
        array([ 8, 12])


.. method:: Expr.eval()

    Evaluate the expression and return the outcome.

    Because of performance reasons, the computation order tries
    to go along the common main dimension of all inputs.  If not
    such a common main dimension is found, the iteration will go
    along the leading dimension instead.

    For non-consistent shapes in inputs (i.e. shapes having a
    different number of dimensions), the regular NumPy broadcast
    rules applies.  There is one exception to this rule though: when
    the dimensions orthogonal to the main dimension of the
    expression are consistent, but the main dimension itself differs
    among the inputs, then the shortest one is chosen for doing the
    computations.  This is so because trying to expand very large
    on-disk arrays could be too expensive or simply not
    possible.

    Also, the regular Numexpr casting rules (which are similar
    to those of NumPy, although you should check the Numexpr manual
    for the exceptions) are applied to determine the output type.

    Finally, if the setOuput() method
    specifying a user container has already been called, the output
    is sent to this user-provided container.  If not, a fresh NumPy
    container is returned instead.

    .. warning:: When dealing with large on-disk inputs, failing to
       specify an on-disk container may consume all your available
       memory.

    For some examples of use see the :meth:`Expr.__init__` docs.


.. method:: Expr.setInputsRange(start=None, stop=None, step=None)

    Define a range for all inputs in expression.

    The computation will only take place for the range defined
    by the start, stop and step parameters in the main dimension of
    inputs (or the leading one, if the object lacks the concept of
    main dimension, like a NumPy container).  If not a common main
    dimension exists for all inputs, the leading dimension will be
    used instead.


.. method:: Expr.setOutput(out, append_mode=False)

    Set out as container for output as well as the append_mode.

    The out must be a container that is meant
    to keep the outcome of the expression.  It should be an
    homogeneous type container and can typically be
    an Array, CArray, EArray, Column or a NumPy ndarray.

    The append_mode specifies the way of
    which the output is filled.  If true, the rows of the outcome
    are *appended* to the out
    container.  Of course, for doing this it is necessary
    that out would have
    an append() method (like
    an EArray, for example).

    If append_mode is false, the output is
    set via the __setitem__() method (see
    the Expr.setOutputRange() for info on how to
    select the rows to be updated).  If out is
    smaller than what is required by the expression, only the
    computations that are needed to fill up the container are
    carried out.  If it is larger, the excess elements are
    unaffected.


.. method:: Expr.setOutputRange(start=None, stop=None, step=None)

    Define a range for user-provided output object.

    The output object will only be modified in the range
    specified by the start,
    stop and step parameters
    in the main dimension of output (or the leading one, if the
    object does not have the concept of main dimension, like a NumPy
    container).


Expr special methods
~~~~~~~~~~~~~~~~~~~~

.. method:: Expr.__iter__()

    Iterate over the rows of the outcome of the expression.

    This iterator always returns rows as NumPy objects, so a
    possible out container specified
    in :meth:`Expr.setOutput` method is ignored
    here.

    The :meth:`Expr.eval` documentation for
    details on how the computation is carried out.  Also, for some
    examples of use see the :meth:`Expr.__init__` docs.



.. _ExceptionsDescr:

Exceptions module
-----------------

In the :mod:`exceptions` module exceptions and warnings that are specific
to PyTables are declared.

:HDF5ExtError:
    A low level HDF5 operation failed.
:ClosedNodeError:
    The operation can not be completed because the node is closed.
:ClosedFileError:
    The operation can not be completed because the hosting file is
    closed.
:FileModeError:
    The operation can not be carried out because the mode in which the
    hosting file is opened is not adequate.
:NodeError:
    Invalid hierarchy manipulation operation requested.
:NoSuchNodeError:
    An operation was requested on a node that does not exist.
:UndoRedoError:
    Problems with doing/redoing actions with Undo/Redo feature.
:UndoRedoWarning:
    Issued when an action not supporting undo/redo is run.
:NaturalNameWarning:
    Issued when a non-pythonic name is given for a node.
:PerformanceWarning:
    Warning for operations which may cause a performance drop.
:FlavorError:
    Unsupported or unavailable flavor or flavor conversion.
:FlavorWarning:
    Unsupported or unavailable flavor conversion.
:FiltersWarning:
    Unavailable filters.
:OldIndexWarning:
    Unsupported index format.
:DataTypeWarning:
    Unsupported data type.
:Incompat16Warning:
    Format incompatible with HDF5 1.6.x series.


The HDF5ExtError exception
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. exception:: exceptions.HDF5ExtError(RuntimeError)

    A low level HDF5 operation failed.

    This exception is raised the low level PyTables components used for
    accessing HDF5 files.  It usually signals that something is not
    going well in the HDF5 library or even at the Input/Output level.

    Errors in the HDF5 C library may be accompanied by an extensive
    HDF5 back trace on standard error (see also
    :func:`tables.silenceHDF5Messages`).

    .. versionchanged:: 2.4


HDF5ExtError class variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY

    Default policy for HDF5 backtrace handling:

    * if set to False the HDF5 back trace is ignored and the
      :attr:`HDF5ExtError.h5backtrace` attribute is set to None
    * if set to True the back trace is retrieved from the HDF5
      library and stored in the :attr:`HDF5ExtError.h5backtrace`
      attribute as a list of tuples
    * if set to "VERBOSE" (default) the HDF5 back trace is
      stored in the :attr:`HDF5ExtError.h5backtrace` attribute
      and also included in the string representation of the
      exception

    This parameter can be set using the
    :envvar:`PT_DEFAULT_H5_BACKTRACE_POLICY` environment variable.
    Allowed values are "IGNORE" (or "FALSE"), "SAVE" (or "TRUE") and
    "VERBOSE" to set the policy to False, True and "VERBOSE"
    respectively.  The special value "DEFAULT" can be used to reset
    the policy to the default value

    .. versionadded:: 2.4


HDF5ExtError instance variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: HDF5ExtError.h5backtrace

    Contains the HDF5 back trace as a (possibly empty) list of
    tuples.  Each tuple has the following format::

        (filename, line number, function name, text)

    Depending on the value of the *h5bt* parameter passed to the
    initializer the h5backtrace attribute can be set to None.
    This means that the HDF5 back trace has been simply ignored
    (not retrieved from the HDF5 C library error stack) or that
    there has been an error (silently ignored) during the HDF5 back
    trace retrieval.

    .. versionadded:: 2.4
    .. seealso:: :func:`traceback.format_list`


HDF5ExtError special methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: HDF5ExtError.__init__(message, h5bt=None)

    Initializer parameters:

    :param message:
        error message
    :param h5bt:
        This parameter (keyword only) controls the HDF5 back trace
        handling:

        * if set to False the HDF5 back trace is ignored and the
          :attr:`HDF5ExtError.h5backtrace` attribute is set to None
        * if set to True the back trace is retrieved from the HDF5
          library and stored in the :attr:`HDF5ExtError.h5backtrace`
          attribute as a list of tuples
        * if set to "VERBOSE" (default) the HDF5 back trace is
          stored in the :attr:`HDF5ExtError.h5backtrace` attribute
          and also included in the string representation of the
          exception
        * if not set (or set to None) the default policy is used
          (see :attr:`HDF5ExtError.DEFAULT_H5_BACKTRACE_POLICY`)

    Keyword arguments different from 'h5bt' are ignored.

    .. versionchanged:: 2.4

.. method::  HDF5ExtError.__str__

    Returns a sting representation of the exception.

    The actual result depends on policy set in the initializer
    :meth:`HDF5ExtError.__init__`.

    .. versionadded:: 2.4


HDF5ExtError methods
^^^^^^^^^^^^^^^^^^^^

.. method:: HDF5ExtError.format_h5_backtrace(backtrace=None)

    Convert the HDF5 trace back represented as a list of tuples
    (see :attr:`HDF5ExtError.h5backtrace`) into string.

    .. versionadded:: 2.4

