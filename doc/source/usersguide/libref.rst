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

    This actually overrides the :data:`parameters.MAX_THREADS`
    setting in :file:`tables/parameters.py`, so the new
    value will be effective until this function is called again or a
    new file with a different :data:`parameters.MAX_THREADS` value
    is specified.

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

..function:: whichLibVersion(name)

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

.. _FileClassDescr:

The File Class
--------------
The in-memory representation of a PyTables file.

An instance of this class is returned when a PyTables file is
opened with the :func:`openFile` function. It offers methods to manipulate
(create, rename, delete...) nodes and handle their attributes, as well
as methods to traverse the object tree. The *user entry
point* to the object tree attached to the HDF5 file is
represented in the rootUEP attribute. Other
attributes are available.

File objects support an *Undo/Redo mechanism* which can be enabled with the
:meth:`File.enableUndo` method. Once the Undo/Redo mechanism is
enabled, explicit *marks* (with an optional unique
name) can be set on the state of the database using the
:meth:`File.mark`
method. There are two implicit marks which are always available: the
initial mark (0) and the final mark (-1).  Both the identifier of a
mark and its name can be used in *undo* and
*redo* operations.

Hierarchy manipulation operations (node creation, movement and
removal) and attribute handling operations (setting and deleting) made
after a mark can be undone by using the :meth:`File.undo` method, which returns the database to the
state of a past mark. If undo() is not followed by
operations that modify the hierarchy or attributes, the
:meth:`File.redo` method can
be used to return the database to the state of a future mark. Else,
future states of the database are forgotten.

Note that data handling operations can not be undone nor redone
by now. Also, hierarchy manipulation operations on nodes that do not
support the Undo/Redo mechanism issue an
UndoRedoWarning *before*
changing the database.

The Undo/Redo mechanism is persistent between sessions and can
only be disabled by calling the :meth:`File.disableUndo` method.

File objects can also act as context managers when using the
with statement introduced in Python 2.5.  When
exiting a context, the file is automatically closed.

.. class:: File

    .. attribute:: filename

        The name of the opened file.

    .. attribute:: filters

        Default filter properties for the root group (see :ref:`FiltersClassDescr`).

    .. attribute:: format_version

        The PyTables version number of this file.

    .. attribute:: isopen

        True if the underlying file is open, false otherwise.

    .. attribute:: mode

        The mode in which the file was opened.

    .. attribute:: open_count

        The number of times this file has been opened currently.

    .. attribute:: root

        The *root* of the object tree hierarchy (a Group instance).

    .. attribute:: rootUEP

        The UEP (user entry point) group name in the file (see
        the :func:`openFile` function).

    .. attribute:: title

        The title of the root group in the file.

File methods - file handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. method:: File.close()

    Flush all the alive leaves in object tree and close the file.

.. method:: File.copyFile(dstfilename, overwrite=False, **kwargs)

    Copy the contents of this file to dstfilename.

    dstfilename must be a path string
    indicating the name of the destination file. If it already exists,
    the copy will fail with an IOError, unless the
    overwrite argument is true, in which case the
    destination file will be overwritten in place. In this last case,
    the destination file should be closed or ugly errors will happen.

    Additional keyword arguments may be passed to customize the
    copying process. For instance, title and filters may be changed,
    user attributes may be or may not be copied, data may be
    sub-sampled, stats may be collected, etc. Arguments unknown to
    nodes are simply ignored. Check the documentation for copying
    operations of nodes to see which options they support.

    In addition, it recognizes the names of parameters present
    in :file:`tables/parameters.py` as additional keyword
    arguments.  See :ref:`parameter_files` for a
    detailed info on the supported parameters.

    Copying a file usually has the beneficial side effect of
    creating a more compact and cleaner version of the original
    file.

.. method:: File.flush()

    Flush all the alive leaves in the object tree.

.. method:: File.fileno()

    Return the underlying OS integer file descriptor.

    This is needed for lower-level file interfaces, such as the
    fcntl module.

.. method:: File.__enter__()

    Enter a context and return the same file.

.. method:: File.__exit__([*exc_info])

    Exit a context and close the file.

.. method:: File.__str__()

    Return a short string representation of the object tree.
    Example of use::

        >>> f = tables.openFile('data/test.h5')
        >>> print f
        data/test.h5 (File) 'Table Benchmark'
        Last modif.: 'Mon Sep 20 12:40:47 2004'
        Object Tree:
        / (Group) 'Table Benchmark'
        /tuple0 (Table(100,)) 'This is the table title'
        /group0 (Group) ''
        /group0/tuple1 (Table(100,)) 'This is the table title'
        /group0/group1 (Group) ''
        /group0/group1/tuple2 (Table(100,)) 'This is the table title'
        /group0/group1/group2 (Group) ''

.. method:: File.__repr__()

    Return a detailed string representation of the object tree.


File methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. method:: File.copyChildren(srcgroup, dstgroup, overwrite=False, recursive=False, createparents=False, **kwargs)

    Copy the children of a group into another group.

    This method copies the nodes hanging from the source group
    srcgroup into the destination group
    dstgroup. Existing destination nodes can be
    replaced by asserting the overwrite argument.
    If the recursive argument is true, all
    descendant nodes of srcnode are recursively
    copied. If createparents is true, the needed
    groups for the given destination parent group path to exist will
    be created.

    kwargs takes keyword arguments used to
    customize the copying process. See the documentation of
    :meth:`Group._f_copyChildren` for a description of those
    arguments.

.. method:: File.copyNode(where, newparent=None, newname=None, name=None, overwrite=False, recursive=False, createparents=False, **kwargs)

    Copy the node specified by where and name to newparent/newname.

    Parameters
    ----------
    where : str
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted
        upon.
    newparent : str or Group
        The destination group that the node will be copied
        into (a path name or a Group
        instance). If not specified or None, the
        current parent group is chosen as the new parent.
    newname : str
        The name to be assigned to the new copy in its
        destination (a string).  If it is not specified or
        None, the current name is chosen as the
        new name.
    name : str
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted
        upon.

    Notes
    -----
    Additional keyword arguments may be passed to customize the
    copying process. The supported arguments depend on the kind of
    node being copied. See :meth:`Group._f_copy` and
    :meth:`Leaf.copy` for more information on their
    allowed keyword arguments.

    This method returns the newly created copy of the source
    node (i.e. the destination node).  See
    :meth:`Node._f_copy`
    for further details on the semantics of copying nodes.


.. method:: File.createArray(where, name, object, title='', byteorder=None, createparents=False)

    Create a new array with the given name in where location.
    See the Array class (in :ref:`ArrayClassDescr`) for more information on
    arrays.

    Parameters
    ----------
    object : python object
        The array or scalar to be saved.  Accepted types are
        NumPy arrays and scalars, numarray arrays
        and string arrays (deprecated), Numeric arrays and scalars
        (deprecated), as well as native Python sequences and scalars,
        provided that values are regular (i.e. they are not like
        [[1,2],2]) and homogeneous (i.e. all the
        elements are of the same type).

        Also, objects that have some of their dimensions equal to 0 are not
        supported (use an EArray node (see :ref:`EArrayClassDescr`) if you
        want to store an array with one of its dimensions equal to 0).
    byteorder : str
        The byteorder of the data *on disk*, specified as
        'little' or
        'big'.  If this is not specified, the
        byteorder is that of the given object.

    Notes
    -----
    See :meth:`File.createTable` for more
    information on the rest of parameters.


.. method:: File.createCArray(where, name, atom, shape, title='', filters=None, chunkshape=None, byteorder=None, createparents=False)

    Create a new chunked array with the given name in where location.
    See the CArray class (in :ref:`CArrayClassDescr`) for more information on
    chunked arrays.

    Parameters
    ----------
    atom : Atom
        An Atom (see :ref:`AtomClassDescr`)
        instance representing the *type* and
        *shape* of the atomic objects to be
        saved.
    shape : tuple
        The shape of the new array.
    chunkshape : tuple or number or None
        The shape of the data chunk to be read or written in a
        single HDF5 I/O operation.  Filters are applied to those
        chunks of data.  The dimensionality of
        chunkshape must be the same as that of
        shape.  If None, a
        sensible value is calculated (which is recommended).

    Notes
    -----
    See :meth:`File.createTable` for more
    information on the rest of parameters.



.. method:: File.createEArray(where, name, atom, shape, title='', filters=None, expectedrows=EXPECTED_ROWS_EARRAY, chunkshape=None, byteorder=None, createparents=False)

    Create a new enlargeable array with the given name in where location.
    See the EArray (in :ref:`EArrayClassDescr`) class for more information on
    enlargeable arrays.

    Parameters
    ----------
    atom : Atom
        An Atom (see :ref:`AtomClassDescr`)
        instance representing the *type* and
        *shape* of the atomic objects to be saved.
    shape : tuple
        The shape of the new array.  One (and only one) of the
        shape dimensions *must* be 0.  The
        dimension being 0 means that the resulting
        EArray object can be extended along it.
        Multiple enlargeable dimensions are not supported right now.
    expectedrows
        A user estimate about the number of row elements that
        will be added to the growable dimension in the
        EArray node.  If not provided, the
        default value is EXPECTED_ROWS_EARRAY
        (see tables/parameters.py).  If you plan
        to create either a much smaller or a much bigger array try
        providing a guess; this will optimize the HDF5 B-Tree
        creation and management process time and the amount of
        memory used.
    chunkshape : tuple, numeric, or None
        The shape of the data chunk to be read or written in a
        single HDF5 I/O operation.  Filters are applied to those
        chunks of data.  The dimensionality of
        chunkshape must be the same as that of
        shape (beware: no dimension should be 0
        this time!).  If None, a sensible value
        is calculated based on the expectedrows
        parameter (which is recommended).
    byteorder : str
        The byteorder of the data *on
        disk*, specified as 'little' or
        'big'. If this is not specified, the
        byteorder is that of the platform.

    Notes
    -----
    See :meth:`File.createTable` for more
    information on the rest of parameters.


.. method:: File.createExternalLink(where, name, target, createparents=False, warn16incompat=False)

    Create an external link to a target node
    with the given name
    in where location.  target
    can be a node object in another file or a path string in the
    form `file:/path/to/node`.  If
    createparents is true, the intermediate
    groups required for reaching where are
    created (the default is not doing so).

    The purpose of the warn16incompat
    argument is to avoid an Incompat16Warning
    (see below).  The default is to issue the warning.

    The returned node is an ExternalLink
    instance.  See the
    ExternalLink class (in
    :ref:`ExternalLinkClassDescr`) for more information on external links.

    .. warning:: External links are only supported when PyTables is
       compiled against HDF5 1.8.x series.  When using PyTables with
       HDF5 1.6.x, the *parent* group containing
       external link objects will be mapped to
       an Unknown instance (see :ref:`UnknownClassDescr`)
       and you won't be able to access *any*
       node hanging of this parent group.  It follows that if the
       parent group containing the external link is the root group,
       you won't be able to read *any* information
       contained in the file when using HDF5 1.6.x.


.. method:: File.createGroup(where, name, title='', filters=None, createparents=False)

    Create a new group with the given name in
    where location.  See the
    Group class (in :ref:`GroupClassDescr`) for more information on
    groups.

    Parameters
    ----------
    filters : Filters
        An instance of the Filters class
        (see :ref:`FiltersClassDescr`) that provides information
        about the desired I/O filters applicable to the leaves that
        hang directly from this new group (unless other filter
        properties are specified for these leaves). Besides, if you
        do not specify filter properties for its child groups, they
        will inherit these ones.

    Notes
    -----
    See :meth:`File.createTable` for more
    information on the rest of parameters.


.. method:: File.createHardLink(where, name, target, createparents=False)

    Create a hard link to a target node with
    the given name in where
    location.  target can be a node object or a
    path string.  If createparents is true, the
    intermediate groups required for
    reaching where are created (the default is
    not doing so).

    The returned node is a regular Group
    or Leaf instance.


.. method:: File.createSoftLink(where, name, target, createparents=False)

    Create a soft link (aka symbolic link) to
    a target node with the
    given name in where
    location.  target can be a node object or a
    path string.  If createparents is true, the
    intermediate groups required for
    reaching where are created (the default is
    not doing so).

    The returned node is a SoftLink instance.
    See the SoftLink class (in
    :ref:`SoftLinkClassDescr`)
    for more information on soft links.


.. method:: File.createTable(where, name, description, title='', filters=None, expectedrows=EXPECTED_ROWS_TABLE, chunkshape=None, byteorder=None, createparents=False)

    Create a new table with the given name in
    where location.  See the
    Table (in :ref:`TableClassDescr`) class for more information on
    tables.

    Parameters
    ----------
    where : path or Group
        The parent group where the new table will hang from.
        It can be a path string (for example
        '/level1/leaf5'), or a
        Group instance (see :ref:`GroupClassDescr`).
    name : str
        The name of the new table.
    description : Description
        This is an object that describes the table, i.e. how
        many columns it has, their names, types, shapes, etc.  It
        can be any of the following:

        * *A user-defined class*: This should inherit from the IsDescription
          class (see :ref:`IsDescriptionClassDescr`) where table fields are specified.
        * *A dictionary*: For example, when you do not know beforehand which structure
          your table will have).
        * *A Description instance*: You can use the description attribute of another
          table to create a new one with the same structure.
        * *A NumPy dtype*: A completely general structured NumPy dtype.
        * *A NumPy (record) array instance*: The dtype of this record array will be used
          as the description.  Also, in case the array has actual data, it will be injected
          into the newly created table.
        * *A RecArray instance (deprecated)*: Object from the numarray package.  This does
          not give you the possibility to create a nested table.  Array data is injected into
          the new table.
        * *A NestedRecArray instance (deprecated)*: If you want to have nested columns in
          your table and you are using numarray, you can use this object. Array data is
          injected into the new table.
    title : str
        A description for this node (it sets the TITLE HDF5 attribute on disk).
    filters : Filters
        An instance of the Filters class
        (see :ref:`FiltersClassDescr`) that provides information
        about the desired I/O filters to be applied during the life
        of this object.
    expectedrows : int
        A user estimate of the number of records that will be
        in the table. If not provided, the default value is
        EXPECTED_ROWS_TABLE (see
        :file:`tables/parameters.py`). If you plan to
        create a bigger table try providing a guess; this will
        optimize the HDF5 B-Tree creation and management process
        time and memory used.
    chunkshape
        The shape of the data chunk to be read or written in a
        single HDF5 I/O operation. Filters are applied to those
        chunks of data. The rank of the
        chunkshape for tables must be 1. If
        None, a sensible value is calculated
        based on the expectedrows parameter
        (which is recommended).
    byteorder : str
        The byteorder of data *on disk*,
        specified as 'little' or
        'big'. If this is not specified, the
        byteorder is that of the platform, unless you passed an
        array as the description, in which case
        its byteorder will be used.
    createparents : bool
        Whether to create the needed groups for the parent
        path to exist (not done by default).


.. method:: File.createVLArray(where, name, atom, title='', filters=None, expectedsizeinMB=1.0, chunkshape=None, byteorder=None, createparents=False)

    Create a new variable-length array with the given
    name in where location.  See
    the VLArray (in :ref:`VLArrayClassDescr`) class
    for more information on variable-length arrays.

    Parameters
    ----------
    atom : Atom
        An Atom (see :ref:`AtomClassDescr`)
        instance representing the *type* and
        *shape* of the atomic objects to be
        saved.
    expectedsizeinMB
        An user estimate about the size (in MB) in the final
        VLArray node. If not provided, the
        default value is 1 MB. If you plan to create either a much
        smaller or a much bigger array try providing a guess; this
        will optimize the HDF5 B-Tree creation and management
        process time and the amount of memory used. If you want to
        specify your own chunk size for I/O purposes, see also the
        chunkshape parameter below.
    chunkshape
        The shape of the data chunk to be read or written in a
        single HDF5 I/O operation. Filters are applied to those
        chunks of data. The dimensionality of
        chunkshape must be 1. If
        None, a sensible value is calculated
        (which is recommended).

    Notes
    -----
    See :meth:`File.createTable` for more
    information on the rest of parameters.


.. method:: File.moveNode(where, newparent=None, newname=None, name=None, overwrite=False, createparents=False)

    Move the node specified by where and name to newparent/newname.

    Parameters
    ----------
    where, name : path
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    newparent
        The destination group the node will be moved into (a
        path name or a Group instance). If it is
        not specified or None, the current parent
        group is chosen as the new parent.
    newname
        The new name to be assigned to the node in its
        destination (a string). If it is not specified or
        None, the current name is chosen as the
        new name.

    Notes
    -----
    The other arguments work as in
    :meth:`Node._f_move`.



.. method:: File.removeNode(where, name=None, recursive=False)

    Remove the object node *name* under *where* location.

    Parameters
    ----------
    where, name
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    recursive : bool
        If not supplied or false, the node will be removed
        only if it has no children; if it does, a
        NodeError will be raised. If supplied
        with a true value, the node and all its descendants will be
        completely removed.


.. method:: File.renameNode(where, newname, name=None, overwrite=False)

    Change the name of the node specified by where and name to newname.

    Parameters
    ----------
    where, name
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    newname : str
        The new name to be assigned to the node (a string).
    overwrite : bool
        Whether to recursively remove a node with the same
        newname if it already exists (not done by default).


File methods - tree traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. method:: File.getNode(where, name=None, classname=None)

    Get the node under where with the given name.

    where can be a Node instance (see :ref:`NodeClassDescr`) or a path string leading to a node. If no
    name is specified, that node is returned.

    If a name is specified, this must be a
    string with the name of a node under where.  In
    this case the where argument can only lead to a
    Group (see :ref:`GroupClassDescr`) instance (else a
    TypeError is raised). The node called
    name under the group where
    is returned.

    In both cases, if the node to be returned does not exist, a
    NoSuchNodeError is raised. Please note that
    hidden nodes are also considered.

    If the classname argument is specified,
    it must be the name of a class derived from
    Node. If the node is found but it is not an
    instance of that class, a NoSuchNodeError is
    also raised.


.. method:: File.isVisibleNode(path)

    Is the node under path visible?

    If the node does not exist, a
    NoSuchNodeError is raised.


.. method:: File.iterNodes(where, classname=None)

    Iterate over children nodes hanging from where.

    Parameters
    ----------
    where
        This argument works as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    classname
        If the name of a class derived from
        Node (see :ref:`NodeClassDescr`) is supplied, only instances of
        that class (or subclasses of it) will be returned.

    Notes
    -----
    The returned nodes are alphanumerically sorted by their
    name.  This is an iterator version of
    :meth:`File.listNodes`.


.. method:: File.listNodes(where, classname=None)

    Return a *list* with children nodes
    hanging from where.

    This is a list-returning version of
    :meth:`File.iterNodes`.


.. method:: File.walkGroups(where='/')

    Recursively iterate over groups (not leaves) hanging from where.

    The where group itself is listed first
    (preorder), then each of its child groups (following an
    alphanumerical order) is also traversed, following the same
    procedure.  If where is not supplied, the root
    group is used.

    The where argument can be a path string
    or a Group instance (see :ref:`GroupClassDescr`).



.. method:: File.walkNodes(where="/", classname="")

    Recursively iterate over nodes hanging from where.

    Parameters
    ----------
    where
        If supplied, the iteration starts from (and includes)
        this group. It can be a path string or a
        Group instance (see :ref:`GroupClassDescr`).
    classname
        If the name of a class derived from
        Node (see :ref:`GroupClassDescr`) is supplied, only instances of
        that class (or subclasses of it) will be returned.

    Examples
    --------
    ::

        # Recursively print all the nodes hanging from '/detector'.
        print "Nodes hanging from group '/detector':"
        for node in h5file.walkNodes('/detector', classname='EArray'):
            print node

.. method:: File.__contains__(path)

    Is there a node with that path?

    Returns True if the file has a node with
    the given path (a string),
    False otherwise.


.. method:: File.__iter__()

    Recursively iterate over the nodes in the tree.

    This is equivalent to calling
    :meth:`File.walkNodes` with no arguments.

    Example of use::

        # Recursively list all the nodes in the object tree.
        h5file = tables.openFile('vlarray1.h5')
        print "All nodes in the object tree:"
        for node in h5file:
            print node


File methods - Undo/Redo support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. method:: File.disableUndo()

    Disable the Undo/Redo mechanism.

    Disabling the Undo/Redo mechanism leaves the database in the
    current state and forgets past and future database states. This
    makes :meth:`File.mark`, :meth:`File.undo`, :meth:`File.redo` and other methods fail with an
    UndoRedoError.

    Calling this method when the Undo/Redo mechanism is already
    disabled raises an UndoRedoError.


.. method:: File.enableUndo(filters=Filters( complevel=1))

    Enable the Undo/Redo mechanism.

    This operation prepares the database for undoing and redoing
    modifications in the node hierarchy. This allows
    :meth:`File.mark`, :meth:`File.undo`, :meth:`File.redo` and other methods
    to be called.

    The filters argument, when specified,
    must be an instance of class Filters (see :ref:`FiltersClassDescr`) and is
    meant for setting the compression values for the action log. The
    default is having compression enabled, as the gains in terms of
    space can be considerable. You may want to disable compression if
    you want maximum speed for Undo/Redo operations.

    Calling this method when the Undo/Redo mechanism is already
    enabled raises an UndoRedoError.


.. method:: File.getCurrentMark()

    Get the identifier of the current mark.

    Returns the identifier of the current mark. This can be used
    to know the state of a database after an application crash, or to
    get the identifier of the initial implicit mark after a call to
    :meth:`File.enableUndo`.

    This method can only be called when the Undo/Redo mechanism
    has been enabled. Otherwise, an UndoRedoError
    is raised.


.. method:: File.goto(mark)

    Go to a specific mark of the database.

    Returns the database to the state associated with the
    specified mark. Both the identifier of a mark
    and its name can be used.

    This method can only be called when the Undo/Redo mechanism
    has been enabled. Otherwise, an UndoRedoError
    is raised.


.. method:: File.isUndoEnabled()

    Is the Undo/Redo mechanism enabled?

    Returns True if the Undo/Redo mechanism
    has been enabled for this file, False
    otherwise. Please note that this mechanism is persistent, so a
    newly opened PyTables file may already have Undo/Redo
    support enabled.


.. method:: File.mark(name=None)

    Mark the state of the database.

    Creates a mark for the current state of the database. A
    unique (and immutable) identifier for the mark is returned. An
    optional name (a string) can be assigned to the
    mark. Both the identifier of a mark and its name can be used in
    :meth:`File.undo`
    and :meth:`File.redo` operations. When the name has already been
    used for another mark, an UndoRedoError is raised.

    This method can only be called when the Undo/Redo mechanism
    has been enabled. Otherwise, an UndoRedoError
    is raised.


.. method:: File.redo(mark=None)

    Go to a future state of the database.

    Returns the database to the state associated with the
    specified mark. Both the identifier of a mark
    and its name can be used. If the mark is
    omitted, the next created mark is used. If there are no future
    marks, or the specified mark is not newer than
    the current one, an UndoRedoError is
    raised.

    This method can only be called when the Undo/Redo mechanism
    has been enabled. Otherwise, an UndoRedoError
    is raised.


.. method:: File.undo(mark=None)

    Go to a past state of the database.

    Returns the database to the state associated with the
    specified mark. Both the identifier of a mark
    and its name can be used. If the mark is
    omitted, the last created mark is used. If there are no past
    marks, or the specified mark is not older than
    the current one, an UndoRedoError is
    raised.

    This method can only be called when the Undo/Redo mechanism
    has been enabled. Otherwise, an UndoRedoError
    is raised.


File methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. method:: File.copyNodeAttrs(where, dstnode, name=None)

    Copy PyTables attributes from one node to another.

    Parameters
    ----------
    where, name
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    dstnode
        The destination node where the attributes will be
        copied to. It can be a path string or a
        Node instance (see :ref:`NodeClassDescr`).


.. method:: File.delNodeAttr(where, attrname, name=None)

    Delete a PyTables attribute from the given node.

    Parameters
    ----------
    where, name
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    attrname
        The name of the attribute to delete.  If the named
        attribute does not exist, an
        AttributeError is raised.


.. method:: File.getNodeAttr(where, attrname, name=None)

    Get a PyTables attribute from the given node.

    Parameters
    ----------
    where, name
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    attrname
        The name of the attribute to retrieve.  If the named
        attribute does not exist, an
        AttributeError is raised.


.. method:: File.setNodeAttr(where, attrname, attrvalue, name=None)

    Set a PyTables attribute for the given node.

    Parameters
    ----------
    where, name
        These arguments work as in
        :meth:`File.getNode`, referencing the node to be acted upon.
    attrname
        The name of the attribute to set.
    attrvalue
        The value of the attribute to set. Any kind of Python
        object (like strings, ints, floats, lists, tuples, dicts,
        small NumPy/Numeric/numarray objects...) can be stored as an
        attribute. However, if necessary, cPickle
        is automatically used so as to serialize objects that you
        might want to save. See the AttributeSet
        class (in :ref:`AttributeSetClassDescr`) for details.

    Notes
    -----
    If the node already has a large number of attributes, a
    PerformanceWarning is issued.


.. _NodeClassDescr:

The Node class
--------------
.. class:: Node

    Abstract base class for all PyTables nodes.

    This is the base class for *all* nodes in a
    PyTables hierarchy. It is an abstract class, i.e. it may not be
    directly instantiated; however, every node in the hierarchy is an
    instance of this class.

    A PyTables node is always hosted in a PyTables
    *file*, under a *parent group*,
    at a certain *depth* in the node hierarchy. A node
    knows its own *name* in the parent group and its
    own *path name* in the file.

    All the previous information is location-dependent, i.e. it may
    change when moving or renaming a node in the hierarchy. A node also
    has location-independent information, such as its *HDF5
    object identifier* and its *attribute set*.

    This class gathers the operations and attributes (both
    location-dependent and independent) which are common to all PyTables
    nodes, whatever their type is. Nonetheless, due to natural naming
    restrictions, the names of all of these members start with a reserved
    prefix (see the Group class in :ref:`GroupClassDescr`).

    Sub-classes with no children (e.g. *leaf
    nodes*) may define new methods, attributes and properties to
    avoid natural naming restrictions. For instance,
    _v_attrs may be shortened to
    attrs and _f_rename to
    rename. However, the original methods and
    attributes should still be available.


Node instance variables - location dependent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Node._v_depth

    The depth of this node in the tree (an non-negative integer value).

.. attribute:: Node._v_file

    The hosting File instance (see :ref:`FileClassDescr`).

.. attribute:: Node._v_name

    The name of this node in its parent group (a string).

.. attribute:: Node._v_parent

    The parent Group instance (see :ref:`GroupClassDescr`).

.. attribute:: Node._v_pathname

    The path of this node in the tree (a string).


.. _NodeClassInstanceVariables:

Node instance variables - location independent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Node._v_attrs

    The associated AttributeSet instance
    (see :ref:`AttributeSetClassDescr`).

.. attribute:: Node._v_isopen

    Whether this node is open or not.

.. attribute:: Node._v_objectID

    A node identifier (may change from run to run).



Node instance variables - attribute shorthands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Node._v_title

    A description of this node. A shorthand for TITLE attribute.


Node methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Node._f_close()

    Close this node in the tree.

    This releases all resources held by the node, so it should
    not be used again. On nodes with data, it may be flushed to
    disk.

    You should not need to close nodes manually because they are
    automatically opened/closed when they are loaded/evicted from the
    integrated LRU cache.


.. method:: Node._f_copy(newparent=None, newname=None, overwrite=False, recursive=False, createparents=False, **kwargs)

    Copy this node and return the new node.

    Creates and returns a copy of the node, maybe in a different
    place in the hierarchy. newparent can be a
    Group object (see :ref:`GroupClassDescr`) or a
    pathname in string form. If it is not specified or
    None, the current parent group is chosen as the
    new parent.  newname must be a string with a
    new name. If it is not specified or None, the
    current name is chosen as the new name. If
    recursive copy is stated, all descendants are
    copied as well. If createparents is true, the
    needed groups for the given new parent group path to exist will be
    created.

    Copying a node across databases is supported but can not be
    undone. Copying a node over itself is not allowed, nor it is
    recursively copying a node into itself. These result in a
    NodeError. Copying over another existing node
    is similarly not allowed, unless the optional
    overwrite argument is true, in which case that
    node is recursively removed before copying.

    Additional keyword arguments may be passed to customize the
    copying process. For instance, title and filters may be changed,
    user attributes may be or may not be copied, data may be
    sub-sampled, stats may be collected, etc. See the documentation
    for the particular node type.

    Using only the first argument is equivalent to copying the
    node to a new location without changing its name. Using only the
    second argument is equivalent to making a copy of the node in the
    same group.



.. method:: Node._f_isVisible()

    Is this node visible?


.. method:: Node._f_move(newparent=None, newname=None, overwrite=False, createparents=False)

    Move or rename this node.

    Moves a node into a new parent group, or changes the name of
    the node. newparent can be a
    Group object (see :ref:`GroupClassDescr`) or a
    pathname in string form. If it is not specified or
    None, the current parent group is chosen as the
    new parent.  newname must be a string with a
    new name. If it is not specified or None, the
    current name is chosen as the new name. If
    createparents is true, the needed groups for
    the given new parent group path to exist will be created.

    Moving a node across databases is not allowed, nor it is
    moving a node *into* itself. These result in a
    NodeError. However, moving a node
    *over* itself is allowed and simply does
    nothing. Moving over another existing node is similarly not
    allowed, unless the optional overwrite argument
    is true, in which case that node is recursively removed before
    moving.

    Usually, only the first argument will be used, effectively
    moving the node to a new location without changing its name.
    Using only the second argument is equivalent to renaming the node
    in place.


.. method:: Node._f_remove(recursive=False, force=False)

    Remove this node from the hierarchy.

    If the node has children, recursive removal must be stated
    by giving recursive a true value; otherwise, a
    NodeError will be raised.

    If the node is a link to a Group object,
    and you are sure that you want to delete it, you can do this by
    setting the force flag to true.


.. method:: Node._f_rename(newname, overwrite=False)

    Rename this node in place.

    Changes the name of a node to *newname*
    (a string).  If a node with the same newname
    already exists and overwrite is true,
    recursively remove it before renaming.



Node methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Node._f_delAttr(name)

    Delete a PyTables attribute from this node.

    If the named attribute does not exist, an
    AttributeError is raised.


.. method:: Node._f_getAttr(name)

    Get a PyTables attribute from this node.

    If the named attribute does not exist, an
    AttributeError is raised.


.. method:: Node._f_setAttr(name, value)

    Set a PyTables attribute for this node.

    If the node already has a large number of attributes, a
    PerformanceWarning is issued.



.. _GroupClassDescr:

The Group class
---------------
.. class:: Group

    Basic PyTables grouping structure.

    Instances of this class are grouping structures containing
    *child* instances of zero or more groups or leaves,
    together with supporting metadata. Each group has exactly one
    *parent* group.

    Working with groups and leaves is similar in many ways to
    working with directories and files, respectively, in a Unix
    filesystem. As with Unix directories and files, objects in the object
    tree are often described by giving their full (or absolute) path
    names. This full path can be specified either as a string (like in
    '/group1/group2') or as a complete object path
    written in *natural naming* schema (like in
    file.root.group1.group2).

    A collateral effect of the *natural naming*
    schema is that the names of members in the Group
    class and its instances must be carefully chosen to avoid colliding
    with existing children node names.  For this reason and to avoid
    polluting the children namespace all members in a
    Group start with some reserved prefix, like
    _f_ (for public methods), _g_
    (for private ones), _v_ (for instance variables) or
    _c_ (for class variables). Any attempt to create a
    new child node whose name starts with one of these prefixes will raise
    a ValueError exception.

    Another effect of natural naming is that children named after
    Python keywords or having names not valid as Python identifiers (e.g.
    class, $a or 44) can not be accessed using the
    node.child syntax. You will be forced to use
    node._f_getChild(child) to access them (which is
    recommended for programmatic accesses).

    You will also need to use _f_getChild() to
    access an existing child node if you set a Python attribute in the
    Group with the same name as that node (you will get
    a NaturalNameWarning when doing this).


Group instance variables
~~~~~~~~~~~~~~~~~~~~~~~~
The following instance variables are provided in addition to
those in Node (see :ref:`NodeClassDescr`):

.. attribute:: Group._v_children

    Dictionary with all nodes hanging from this group.

.. attribute:: Group._v_filters

    Default filter properties for child nodes.

    You can (and are encouraged to) use this property to
    get, set and delete the FILTERS HDF5
    attribute of the group, which stores a
    Filters instance (see :ref:`FiltersClassDescr`). When
    the group has no such attribute, a default
    Filters instance is used.


.. attribute:: Group._v_groups

    Dictionary with all groups hanging from this group.


.. attribute:: Group._v_hidden

    Dictionary with all hidden nodes hanging from this group.

.. attribute:: Group._v_leaves

    Dictionary with all leaves hanging from this group.

.. attribute:: Group._v_links

    Dictionary with all links hanging from this group.

.. attribute:: Group._v_nchildren

    The number of children hanging from this group.

.. attribute:: Group._v_unknown

    Dictionary with all unknown nodes hanging from this group.


Group methods
~~~~~~~~~~~~~
.. important:: *Caveat:* The following methods are
    documented for completeness, and they can be used without any
    problem. However, you should use the high-level counterpart methods
    in the File class (see :ref:`FileClassDescr`, because they
    are most used in documentation and examples, and are a bit more
    powerful than those exposed here.

The following methods are provided in addition to those in
Node (see :ref:`NodeClassDescr`):


.. method:: Group._f_close()

    Close this group and all its descendents.

    This method has the behavior described in
    :meth:`Node._f_close`.  It should be noted that this
    operation closes all the nodes descending from this group.

    You should not need to close nodes manually because they are
    automatically opened/closed when they are loaded/evicted from the
    integrated LRU cache.


.. method:: Group._f_copy(newparent, newname, overwrite=False, recursive=False, createparents=False, **kwargs)

    Copy this node and return the new one.

    This method has the behavior described in
    :meth:`Node._f_copy`. In addition, it recognizes the
    following keyword arguments:

    Parameters
    ----------
    title
        The new title for the destination. If omitted or
        None, the original title is used. This
        only applies to the topmost node in recursive copies.
    filters : Filters
        Specifying this parameter overrides the original
        filter properties in the source node. If specified, it must
        be an instance of the Filters class (see
        :ref:`FiltersClassDescr`). The default is to copy the
        filter properties from the source node.
    copyuserattrs
        You can prevent the user attributes from being copied
        by setting this parameter to False. The
        default is to copy them.
    stats
        This argument may be used to collect statistics on the
        copy process. When used, it should be a dictionary with keys
        'groups', 'leaves',
        'links' and
        'bytes' having a numeric value. Their
        values will be incremented to reflect the number of groups,
        leaves and bytes, respectively, that have been copied during
        the operation.



.. method:: Group._f_copyChildren(dstgroup, overwrite=False, recursive=False, createparents=False, **kwargs)

    Copy the children of this group into another group.

    Children hanging directly from this group are copied into
    dstgroup, which can be a
    Group (see :ref:`GroupClassDescr`) object or its pathname in string
    form. If createparents is true, the needed
    groups for the given destination group path to exist will be
    created.

    The operation will fail with a NodeError
    if there is a child node in the destination group with the same
    name as one of the copied children from this one, unless
    overwrite is true; in this case, the former
    child node is recursively removed before copying the later.

    By default, nodes descending from children groups of this
    node are not copied. If the recursive argument
    is true, all descendant nodes of this node are recursively
    copied.

    Additional keyword arguments may be passed to customize the
    copying process. For instance, title and filters may be changed,
    user attributes may be or may not be copied, data may be
    sub-sampled, stats may be collected, etc. Arguments unknown to
    nodes are simply ignored. Check the documentation for copying
    operations of nodes to see which options they support.


.. method:: Group._f_getChild(childname)

    Get the child called childname of this group.

    If the child exists (be it visible or not), it is returned.
    Else, a NoSuchNodeError is raised.

    Using this method is recommended over
    getattr() when doing programmatic accesses to
    children if childname is unknown beforehand or
    when its name is not a valid Python identifier.


.. method:: Group._f_iterNodes(classname=None)

    Iterate over children nodes.

    Child nodes are yielded alphanumerically sorted by node
    name.  If the name of a class derived from Node
    (see :ref:`NodeClassDescr`)
    is supplied in the classname parameter, only
    instances of that class (or subclasses of it) will be
    returned.

    This is an iterator version of
    :meth:`Group._f_listNodes`.


.. method:: Group._f_listNodes(classname=None)

    Return a *list* with children nodes.

    This is a list-returning version of
    :meth:`Group._f_iterNodes`.


.. method:: Group._f_walkGroups()

    Recursively iterate over descendant groups (not leaves).

    This method starts by yielding *self*,
    and then it goes on to recursively iterate over all child groups
    in alphanumerical order, top to bottom (preorder), following the
    same procedure.


.. method:: Group._f_walkNodes(classname=None)

    Iterate over descendant nodes.

    This method recursively walks *self* top
    to bottom (preorder), iterating over child groups in
    alphanumerical order, and yielding nodes.  If
    classname is supplied, only instances of the
    named class are yielded.

    If *classname* is Group, it behaves like
    :meth:`Group._f_walkGroups`, yielding only groups.  If you
    don't want a recursive behavior, use
    :meth:`Group._f_iterNodes` instead.

    Example of use::

        # Recursively print all the arrays hanging from '/'
        print "Arrays in the object tree '/':"
        for array in h5file.root._f_walkNodes('Array', recursive=True):
            print array


Group special methods
~~~~~~~~~~~~~~~~~~~~~
Following are described the methods that automatically trigger
actions when a Group instance is accessed in a
special way.

This class defines the __setattr__,
__getattr__ and __delattr__
methods, and they set, get and delete *ordinary Python
attributes* as normally intended. In addition to that,
__getattr__ allows getting *child
nodes* by their name for the sake of easy interaction on
the command line, as long as there is no Python attribute with the
same name. Groups also allow the interactive completion (when using
readline) of the names of child nodes. For
instance::

    # get a Python attribute
    nchild = group._v_nchildren

    # Add a Table child called 'table' under 'group'.
    h5file.createTable(group, 'table', myDescription)
    table = group.table          # get the table child instance
    group.table = 'foo'          # set a Python attribute

    # (PyTables warns you here about using the name of a child node.)
    foo = group.table            # get a Python attribute
    del group.table              # delete a Python attribute
    table = group.table          # get the table child instance again


.. method:: Group.__contains__(name)

    Is there a child with that name?

    Returns a true value if the group has a child node (visible
    or hidden) with the given *name* (a string),
    false otherwise.


.. method:: Group.__delattr__(name)

    Delete a Python attribute called name.

    This method deletes an *ordinary Python
    attribute* from the object. It does
    *not* remove children nodes from this group;
    for that, use :meth:`File.removeNode` or
    :meth:`Node._f_remove`. It does *neither*
    delete a PyTables node attribute; for that, use
    :meth:`File.delNodeAttr`,
    :meth:`Node._f_delAttr` or :attr:`:attr:`Node._v_attrs``.

    If there is an attribute and a child node with the same
    name, the child node will be made accessible
    again via natural naming.


.. method:: Group.__getattr__(name)

    Get a Python attribute or child node called name.

    If the object has a Python attribute called
    name, its value is returned. Else, if the node
    has a child node called name, it is returned.
    Else, an AttributeError is raised.


.. method:: Group.__iter__()

    Iterate over the child nodes hanging directly from the group.

    This iterator is *not* recursive. Example of use::

        # Non-recursively list all the nodes hanging from '/detector'
        print "Nodes in '/detector' group:"
        for node in h5file.root.detector:
            print node


.. method:: Group.__repr__()

    Return a detailed string representation of the group.

    Example of use::

        >>> f = tables.openFile('data/test.h5')
        >>> f.root.group0
        /group0 (Group) 'First Group'
          children := ['tuple1' (Table), 'group1' (Group)]

.. method:: Group.__setattr__(name, value)

    Set a Python attribute called name with
    the given value.

    This method stores an *ordinary Python
    attribute* in the object. It does
    *not* store new children nodes under this
    group; for that, use the File.create*() methods
    (see the File class in :ref:`FileClassDescr`). It does
    *neither* store a PyTables node attribute; for
    that, use :meth:`File.setNodeAttr`,
    :meth`:Node._f_setAttr` or :attr:`Node._v_attrs`.

    If there is already a child node with the same
    name, a NaturalNameWarning
    will be issued and the child node will not be accessible via
    natural naming nor getattr(). It will still be
    available via :meth:`File.getNode`, :meth:`Group._f_getChild`
    and children dictionaries in the group (if visible).


.. method:: Group.__str__()

    Return a short string representation of the group.

    Example of use::

        >>> f=tables.openFile('data/test.h5')
        >>> print f.root.group0
        /group0 (Group) 'First Group'


.. _LeafClassDescr:

The Leaf class
--------------
.. class:: Leaf

    Abstract base class for all PyTables leaves.

    A leaf is a node (see the Node class in :class:`Node`) which hangs
    from a group (see the Group class in :class:`Group`) but, unlike a
    group, it can not have any further children below it (i.e. it is an
    end node).

    This definition includes all nodes which contain actual data (datasets
    handled by the Table - see :ref:`TableClassDescr`, Array - see
    :ref:`ArrayClassDescr`, CArray - see :ref:`CArrayClassDescr`, EArray -
    see :ref:`EArrayClassDescr`, and VLArray - see :ref:`VLArrayClassDescr`
    classes) and unsupported nodes (the UnImplemented class -
    :ref:`UnImplementedClassDescr`) these classes do in fact inherit from
    Leaf.



.. _LeafInstanceVariables:

Leaf instance variables
~~~~~~~~~~~~~~~~~~~~~~~
These instance variables are provided in addition to those in
Node (see :ref:`NodeClassDescr`):


.. attribute:: Leaf.byteorder

    The byte ordering of the leaf data *on disk*.

.. attribute:: Leaf.chunkshape

    The HDF5 chunk size for chunked leaves (a tuple).

    This is read-only because you cannot change the chunk
    size of a leaf once it has been created.

.. attribute:: Leaf.dtype

    The NumPy dtype that most closely matches this leaf type.

.. attribute:: Leaf.extdim

    The index of the enlargeable dimension (-1 if none).

.. attribute:: Leaf.filters

    Filter properties for this leaf - see
    Filters in :ref:`FiltersClassDescr`.

.. attribute:: Leaf.flavor

    The type of data object read from this leaf.

    It can be any of 'numpy',
    'numarray', 'numeric' or
    'python' (the set of supported flavors
    depends on which packages you have installed on your
    system).

    You can (and are encouraged to) use this property to
    get, set and delete the FLAVOR HDF5
    attribute of the leaf. When the leaf has no such attribute,
    the default flavor is used.

    .. warning:: The 'numarray' and
       'numeric' flavors are deprecated since
       version 2.3. Support for these flavors will be removed in
       future versions.

.. attribute:: Leaf.maindim

    The dimension along which iterators work.

    Its value is 0 (i.e. the first dimension) when the
    dataset is not extendable, and self.extdim
    (where available) for extendable ones.

.. attribute:: Leaf.nrows

    The length of the main dimension of the leaf data.

.. attribute:: Leaf.nrowsinbuf

    The number of rows that fit in internal input buffers.

    You can change this to fine-tune the speed or memory
    requirements of your application.

.. attribute:: Leaf.shape

    The shape of data in the leaf.



Leaf instance variables - aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following are just easier-to-write aliases to their
Node (see :ref:`NodeClassDescr`) counterparts (indicated between
parentheses):

.. attribute:: Leaf.attrs

    The associated AttributeSet instance - see :ref:`AttributeSetClassDescr` - (:attr:`Node._v_attrs`).

.. attribute:: Leaf.name

    The name of this node in its parent group (:attr:`Node._v_name`).

.. attribute:: Leaf.objectID

    A node identifier (may change from run to run). (:attr:`Node._v_objectID`).

.. attribute:: Leaf.title

    A description for this node (:attr:`Node._v_title`).


Leaf methods
~~~~~~~~~~~~

.. method:: Leaf.close(flush=True)

    Close this node in the tree.

    This method is completely equivalent to
    :meth:`Leaf._f_close`.


.. method:: Leaf.copy(newparent, newname, overwrite=False, createparents=False, **kwargs)

    Copy this node and return the new one.

    This method has the behavior described in
    :meth:`Node._f_copy`. Please note that there is no
    recursive flag since leaves do not have child
    nodes.

    Parameters
    ----------
    title
        The new title for the destination. If omitted or
        None, the original title is used.
    filters : Filters
        Specifying this parameter overrides the original
        filter properties in the source node. If specified, it must
        be an instance of the Filters class (see
        :ref:`FiltersClassDescr`). The default is to copy the
        filter properties from the source node.
    copyuserattrs
        You can prevent the user attributes from being copied
        by setting this parameter to False. The
        default is to copy them.
    start, stop, step : int
        Specify the range of rows to be copied; the default is
        to copy all the rows.
    stats
        This argument may be used to collect statistics on the
        copy process. When used, it should be a dictionary with keys
        'groups', 'leaves' and
        'bytes' having a numeric value. Their
        values will be incremented to reflect the number of groups,
        leaves and bytes, respectively, that have been copied during
        the operation.
    chunkshape
        The chunkshape of the new leaf.  It supports a couple
        of special values.  A value of keep means
        that the chunkshape will be the same than original leaf
        (this is the default).  A value of auto
        means that a new shape will be computed automatically in
        order to ensure best performance when accessing the dataset
        through the main dimension.  Any other value should be an
        integer or a tuple matching the dimensions of the
        leaf.

    Notes
    -----
    .. warning:: Note that unknown parameters passed to this method will be
       ignored, so may want to double check the spell of these (i.e. if
       you write them incorrectly, they will most probably be
       ignored).


.. method:: Leaf.delAttr(name)

    Delete a PyTables attribute from this node.

    This method has the behavior described in
    :meth:`Node_f_delAttr`.


.. method:: Leaf.flush()

    Flush pending data to disk.

    Saves whatever remaining buffered data to disk. It also
    releases I/O buffers, so if you are filling many datasets in the
    same PyTables session, please call flush()
    extensively so as to help PyTables to keep memory requirements low.


.. method:: Leaf.getAttr(name)

    Get a PyTables attribute from this node.

    This method has the behavior described in
    :meth:`Node._f_getAttr`.


.. method:: Leaf.isVisible()

    Is this node visible?

    This method has the behavior described in
    :meth:`Node._f_isVisible`.


.. method:: Leaf.move(newparent=None, newname=None, overwrite=False, createparents=False)

    Move or rename this node.

    This method has the behavior described in
    :meth:`Node._f_move`


.. method:: Leaf.rename(newname)

    Rename this node in place.

    This method has the behavior described in
    :meth:`Node._f_rename`.


.. method:: Leaf.remove()

    Remove this node from the hierarchy.

    This method has the behavior described in
    :meth:`Node._f_remove`. Please note that there is no
    recursive flag since leaves do not have child
    nodes.


.. method:: Leaf.setAttr(name, value)

    Set a PyTables attribute for this node.

    This method has the behavior described in
    :meth:`Node._f_setAttr`.


.. method:: Leaf.truncate(size)

    Truncate the main dimension to be size rows.

    If the main dimension previously was larger than this
    size, the extra data is lost.  If the main
    dimension previously was shorter, it is extended, and the extended
    part is filled with the default values.

    The truncation operation can only be applied to
    *enlargeable* datasets, else a
    TypeError will be raised.

    .. warning:: If you are using the HDF5 1.6.x series, and due to
       limitations of them, size must be greater
       than zero (i.e. the dataset can not be completely emptied).  A
       ValueError will be issued if you are using
       HDF5 1.6.x and try to pass a zero size to this method.  Also,
       HDF5 1.6.x has the problem that it cannot work against
       CArray objects (again, a
       ValueError will be issued).  HDF5 1.8.x
       doesn't undergo these problems.


.. method:: Leaf.__len__()

    Return the length of the main dimension of the leaf data.

    Please note that this may raise an
    OverflowError on 32-bit platforms for datasets
    having more than 2**31-1 rows.  This is a limitation of Python
    that you can work around by using the nrows or
    shape attributes.


.. method:: Leaf._f_close(flush=True)

    Close this node in the tree.

    This method has the behavior described in
    :meth:`Node._f_close`.  Besides that, the optional argument
    flush tells whether to flush pending data to
    disk or not before closing.


.. _TableClassDescr:

The Table class
---------------
.. class:: Table

    This class represents heterogeneous datasets in an HDF5 file.

    Tables are leaves (see the Leaf class in
    :ref:`LeafClassDescr`) whose
    data consists of a unidimensional sequence of
    *rows*, where each row contains one or more
    *fields*.  Fields have an associated unique
    *name* and *position*, with the
    first field having position 0.  All rows have the same fields, which
    are arranged in *columns*.

    Fields can have any type supported by the Col
    class (see :ref:`ColClassDescr`)
    and its descendants, which support multidimensional data.  Moreover, a
    field can be *nested* (to an arbitrary depth),
    meaning that it includes further fields inside.  A field named
    x inside a nested field a in a
    table can be accessed as the field a/x (its
    *path name*) from the table.

    The structure of a table is declared by its description, which
    is made available in the Table.description
    attribute (see :ref:`TableInstanceVariablesDescr`).

    This class provides new methods to read, write and search table
    data efficiently.  It also provides special Python methods to allow
    accessing the table as a normal sequence or array (with extended
    slicing supported).

    PyTables supports *in-kernel* searches
    working simultaneously on several columns using complex conditions.
    These are faster than selections using Python expressions.  See the
    :meth:`Tables.where` method for more information on in-kernel searches.

    Non-nested columns can be *indexed*.
    Searching an indexed column can be several times faster than searching
    a non-nested one.  Search methods automatically take advantage of
    indexing where available.

    When iterating a table, an object from the
    Row (see :ref:`RowClassDescr`) class is used.  This object allows to
    read and write data one row at a time, as well as to perform queries
    which are not supported by in-kernel syntax (at a much lower speed, of
    course).

    Objects of this class support access to individual columns via
    *natural naming* through the
    Table.cols accessor (see :ref:`TableInstanceVariablesDescr`).
    Nested columns are mapped to Cols instances, and
    non-nested ones to Column instances.  See the
    Column class in :ref:`ColumnClassDescr` for examples of this feature.


.. _TableInstanceVariablesDescr:

Table instance variables
~~~~~~~~~~~~~~~~~~~~~~~~
The following instance variables are provided in addition to
those in Leaf (see :ref:`LeafClassDescr`).  Please note that there are several
col* dictionaries to ease retrieving information
about a column directly by its path name, avoiding the need to walk
through Table.description or
Table.cols.


.. attribute:: Table.autoIndex

    Automatically keep column indexes up to date?

    Setting this value states whether existing indexes
    should be automatically updated after an append operation or
    recomputed after an index-invalidating operation (i.e. removal
    and modification of rows). The default is true.

    This value gets into effect whenever a column is
    altered. If you don't have automatic indexing activated and
    you want to do an immediate update use
    :meth:`Table.flushRowsToIndex`; for immediate reindexing
    of invalidated indexes, use
    :meth:`Table.reIndexDirty`.

    This value is persistent.

.. attribute:: Table.coldescrs

    Maps the name of a column to its Col
    description (see :ref:`ColClassDescr`).


.. attribute:: Table.coldflts

    Maps the name of a column to its default value.

.. attribute:: Table.coldtypes

    Maps the name of a column to its NumPy data type.


.. attribute:: Table.colindexed

    Is the column which name is used as a key indexed?


.. attribute:: Table.colindexes

    A dictionary with the indexes of the indexed columns.

.. attribute:: Table.colinstances

    Maps the name of a column to its
    Column (see :ref:`ColumnClassDescr`) or
    Cols (see :ref:`ColsClassDescr`) instance.


.. attribute:: Table.colnames

    A list containing the names of *top-level* columns in the table.


.. attribute:: Table.colpathnames

    A list containing the pathnames of
    *bottom-level* columns in the table.

    These are the leaf columns obtained when walking the
    table description left-to-right, bottom-first. Columns inside
    a nested column have slashes (/) separating
    name components in their pathname.

.. attribute:: Table.cols

    A Cols instance that provides
    *natural naming* access to non-nested
    (Column, see :ref:`ColumnClassDescr`) and
    nested (Cols, see :ref:`ColsClassDescr`)
    columns.

.. attribute:: Table.coltypes

    Maps the name of a column to its PyTables
    data type.


.. attribute:: Table.description

    A Description instance (see :ref:`DescriptionClassDescr`)
    reflecting the structure of the table.

.. attribute:: Table.extdim

    The index of the enlargeable dimension (always 0 for tables).


.. attribute:: Table.indexed

    Does this table have any indexed columns?


.. attribute:: Table.indexedcolpathnames

    List of the pathnames of indexed columns in the table.


.. attribute:: Table.nrows

    The current number of rows in the table.


.. attribute:: Table.row

    The associated Row instance (see :ref:`RowClassDescr`).


.. attribute:: Table.rowsize

    The size in bytes of each row in the table.


Table methods - reading
~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Table.col(name)

    Get a column from the table.

    If a column called name exists in the
    table, it is read and returned as a NumPy object, or as a
    numarray object (depending on the flavor of the
    table). If it does not exist, a KeyError is
    raised.

    Example of use::

        narray = table.col('var2')

    That statement is equivalent to:::

        narray = table.read(field='var2')

    Here you can see how this method can be used as a shorthand
    for the :meth:`Table.read` method.


.. method:: Table.iterrows(start=None, stop=None, step=None)

    Iterate over the table using a Row
    instance (see :ref:`RowClassDescr`).

    If a range is not supplied, *all the
    rows* in the table are iterated upon - you can also use
    the :meth:`Table.__iter__` special method for that purpose. If you want to
    iterate over a given *range of rows* in the
    table, you may use the start,
    stop and step parameters,
    which have the same meaning as in :meth:`Table.read`.

    Example of use::

        result = [ row['var2'] for row in table.iterrows(step=5) if row['var1'] <= 20 ]

    .. note:: This iterator can be nested (see :meth:`Table.where` for an
       example).

    .. warning:: When in the middle of a table row iterator, you should not
       use methods that can change the number of rows in the table
       (like :meth:`Table.append` or :meth:`Table.removeRows`)
       or unexpected errors will happen.


.. method:: Table.itersequence(sequence)

    Iterate over a sequence of row coordinates.

    .. note:: This iterator can be nested (see :meth:`Table.where` for an
       example).


.. method:: Table.itersorted(sortby, checkCSI=False, start=None, stop=None, step=None)

    Iterate table data following the order of the index of sortby column.

    sortby column must have associated a
    full index.  If you want to ensure a fully
    sorted order, the index must be a CSI one.  You may want to use
    the checkCSI argument in order to explicitly
    check for the existence of a CSI index.

    The meaning of the start,
    stop and step arguments is
    the same as in :meth:`Table.read`.  However, in this case a negative value
    of step is supported, meaning that the results
    will be returned in reverse sorted order.


.. method:: Table.read(start=None, stop=None, step=None, field=None)

    Get data in the table as a (record) array.

    The start, stop and
    step parameters can be used to select only a
    *range of rows* in the table. Their meanings
    are the same as in the built-in range() Python
    function, except that negative values of step
    are not allowed yet. Moreover, if only start is
    specified, then stop will be set to
    start+1. If you do not specify neither
    start nor stop, then
    *all the rows* in the table are
    selected.

    If field is supplied only the named
    column will be selected.  If the column is not nested, an
    *array* of the current flavor will be returned;
    if it is, a *record array* will be used
    instead.  I no field is specified, all the
    columns will be returned in a record array of the current flavor.

    Columns under a nested column can be specified in the
    field parameter by using a slash character
    (/) as a separator (e.g. 'position/x').


.. method:: Table.readCoordinates(coords, field=None)

    Get a set of rows given their indexes as a (record) array.

    This method works much like the :meth:`Table.read`
    method, but it uses a sequence
    (coords) of row indexes to select the wanted
    columns, instead of a column range.

    The selected rows are returned in an array or record array
    of the current flavor.


.. method:: Table.readSorted(sortby, checkCSI=False, field=None, start=None, stop=None, step=None)

    Read table data following the order of the index of sortby column.

    sortby column must have associated a
    full index.  If you want to ensure a fully
    sorted order, the index must be a CSI one.  You may want to use
    the checkCSI argument in order to explicitly
    check for the existence of a CSI index.

    If field is supplied only the named
    column will be selected.  If the column is not nested, an
    *array* of the current flavor will be returned;
    if it is, a *record array* will be used
    instead.  If no field is specified, all the
    columns will be returned in a record array of the current
    flavor.

    The meaning of the start,
    stop and step arguments is
    the same as in :meth:`Table.read`.  However, in this case a negative value
    of step is supported, meaning that the results
    will be returned in reverse sorted order.


.. method:: Table.__getitem__(key)

    Get a row or a range of rows from the table.

    If key argument is an integer, the
    corresponding table row is returned as a record of the current
    flavor. If key is a slice, the range of rows
    determined by it is returned as a record array of the current
    flavor.

    In addition, NumPy-style point selections are supported.  In
    particular, if key is a list of row
    coordinates, the set of rows determined by it is returned.
    Furthermore, if key is an array of boolean
    values, only the coordinates where key
    is True are returned.  Note that for the latter
    to work it is necessary that key list would
    contain exactly as many rows as the table has.

    Example of use::

        record = table[4]
        recarray = table[4:1000:2]
        recarray = table[[4,1000]]   # only retrieves rows 4 and 1000
        recarray = table[[True, False, ..., True]]

    Those statements are equivalent to::

        record = table.read(start=4)[0]
        recarray = table.read(start=4, stop=1000, step=2)
        recarray = table.readCoordinates([4,1000])
        recarray = table.readCoordinates([True, False, ..., True])

    Here, you can see how indexing can be used as a shorthand
    for the :meth:`Table.read` and :meth:`Table.readCoordinates` methods.


.. method:: Table.__iter__()

    Iterate over the table using a Row
    instance (see :ref:`RowClassDescr`).

    This is equivalent to calling
    :meth:`Table.iterrows` with default arguments, i.e. it
    iterates over *all the rows* in the table.

    Example of use::

        result = [ row['var2'] for row in table if row['var1'] <= 20 ]

    Which is equivalent to::

        result = [ row['var2'] for row in table.iterrows() if row['var1'] <= 20 ]

    .. note:: This iterator can be nested (see :meth:`Table.where` for an
       example).


Table methods - writing
~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Table.append(rows)

    Append a sequence of rows to the end of the table.

    The rows argument may be any object which
    can be converted to a record array compliant with the table
    structure (otherwise, a ValueError is raised).
    This includes NumPy record arrays, RecArray
    (depracated) or NestedRecArray (deprecated)
    objects if numarray is available, lists of
    tuples or array records, and a string or Python buffer.

    Example of use::

        from tables import *

        class Particle(IsDescription):
            name        = StringCol(16, pos=1) # 16-character String
            lati        = IntCol(pos=2)        # integer
            longi       = IntCol(pos=3)        # integer
            pressure    = Float32Col(pos=4)    # float  (single-precision)
            temperature = FloatCol(pos=5)      # double (double-precision)

        fileh = openFile('test4.h5', mode='w')
        table = fileh.createTable(fileh.root, 'table', Particle, "A table")

        # Append several rows in only one call
        table.append([("Particle:     10", 10, 0, 10*10, 10**2),
                      ("Particle:     11", 11, -1, 11*11, 11**2),
                      ("Particle:     12", 12, -2, 12*12, 12**2)])
        fileh.close()



.. method:: Table.modifyColumn(start=None, stop=None, step=1, column=None, colname=None)

    Modify one single column in the row slice [start:stop:step].

    The colname argument specifies the name
    of the column in the table to be modified with the data given in
    column.  This method returns the number of rows
    modified.  Should the modification exceed the length of the table,
    an IndexError is raised before changing data.

    The column argument may be any object
    which can be converted to a (record) array compliant with the
    structure of the column to be modified (otherwise, a
    ValueError is raised).  This includes NumPy
    (record) arrays, NumArray (deprecated),
    RecArray (deprecated) or
    NestedRecArray (deprecated) objects if
    numarray is available, Numeric arrays
    if available (deprecated), lists of scalars, tuples or array
    records, and a string or Python buffer.


.. method:: Table.modifyColumns(start=None, stop=None, step=1, columns=None, names=None)

    Modify a series of columns in the row slice [start:stop:step].

    The names argument specifies the names of
    the columns in the table to be modified with the data given in
    columns.  This method returns the number of
    rows modified.  Should the modification exceed the length of the
    table, an IndexError is raised before changing data.

    The columns argument may be any object
    which can be converted to a record array compliant with the
    structure of the columns to be modified (otherwise, a
    ValueError is raised).  This includes NumPy
    record arrays, RecArray (deprecated) or
    NestedRecArray (deprecated) objects if
    numarray is available, lists of tuples or array
    records, and a string or Python buffer.



.. method:: Table.modifyCoordinates(coords, rows)

    Modify a series of rows in positions specified in coords

    The values in the selected rows will be modified with the
    data given in rows.  This method returns the
    number of rows modified.

    The possible values for the rows argument
    are the same as in :meth:`Table.append`.


.. method:: Table.modifyRows(start=None, stop=None, step=1, rows=None)

    Modify a series of rows in the slice [start:stop:step].

    The values in the selected rows will be modified with the
    data given in rows.  This method returns the
    number of rows modified.  Should the modification exceed the
    length of the table, an IndexError is raised
    before changing data.

    The possible values for the rows argument
    are the same as in :meth:`Table.append`.



.. method:: Table.removeRows(start, stop=None)

    Remove a range of rows in the table.

    If only start is supplied, only this row
    is to be deleted.  If a range is supplied, i.e. both the
    start and stop parameters
    are passed, all the rows in the range are removed. A
    step parameter is not supported, and it is not
    foreseen to be implemented anytime soon.

    Parameters
    ----------
    start : int
        Sets the starting row to be removed. It accepts
        negative values meaning that the count starts from the end.
        A value of 0 means the first row.
    stop : int
        Sets the last row to be removed to
        stop-1, i.e. the end point is omitted (in
        the Python range() tradition). Negative
        values are also accepted. A special value of
        None (the default) means removing just
        the row supplied in start.


.. method:: Table.__setitem__(key, value)

    Set a row or a range of rows in the table.

    It takes different actions depending on the type of the
    key parameter: if it is an integer, the
    corresponding table row is set to value (a
    record or sequence capable of being converted to the table
    structure). If key is a slice, the row slice
    determined by it is set to value (a record
    array or sequence capable of being converted to the table
    structure).

    In addition, NumPy-style point selections are supported.  In
    particular, if key is a list of row
    coordinates, the set of rows determined by it is set
    to value.  Furthermore,
    if key is an array of boolean values, only the
    coordinates where key
    is True are set to values
    from value.  Note that for the latter to work
    it is necessary that key list would contain
    exactly as many rows as the table has.

    Example of use::

        # Modify just one existing row
        table[2] = [456,'db2',1.2]

        # Modify two existing rows
        rows = numpy.rec.array([[457,'db1',1.2],[6,'de2',1.3]], formats='i4,a3,f8')
        table[1:30:2] = rows             # modify a table slice
        table[[1,3]] = rows              # only modifies rows 1 and 3
        table[[True,False,True]] = rows  # only modifies rows 0 and 2

    Which is equivalent to::

        table.modifyRows(start=2, rows=[456,'db2',1.2])
        rows = numpy.rec.array([[457,'db1',1.2],[6,'de2',1.3]], formats='i4,a3,f8')
        table.modifyRows(start=1, stop=3, step=2, rows=rows)
        table.modifyCoordinates([1,3,2], rows)
        table.modifyCoordinates([True, False, True], rows)

    Here, you can see how indexing can be used as a shorthand
    for the :meth:`Table.modifyRows`  and :meth:`Table.modifyCoordinates` methods.


.. _TableMethods_querying:

Table methods - querying
~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Table.getWhereList(condition, condvars=None, sort=False, start=None, stop=None, step=None)

    Get the row coordinates fulfilling the given condition.

    The coordinates are returned as a list of the current
    flavor.  sort means that you want to retrieve
    the coordinates ordered. The default is to not sort them.

    The meaning of the other arguments is the same as in the
    :meth:`Table.where` method.



.. method:: Table.readWhere(condition, condvars=None, field=None, start=None, stop=None, step=None)

    Read table data fulfilling the given *condition*.

    This method is similar to :meth:`Table.read`, having their common arguments
    and return values the same meanings. However, only the rows
    fulfilling the *condition* are included in the
    result.

    The meaning of the other arguments is the same as in the
    :meth:`Table.where` method.


.. method:: Table.where(condition, condvars=None, start=None, stop=None, step=None)

    Iterate over values fulfilling a condition.

    This method returns a Row iterator (see
    :ref:`RowClassDescr`) which
    only selects rows in the table that satisfy the given
    condition (an expression-like string).

    The condvars mapping may be used to
    define the variable names appearing in the
    condition. condvars should
    consist of identifier-like strings pointing to
    Column (see :ref:`ColumnClassDescr`) instances *of this
    table*, or to other values (which will be converted to
    arrays). A default set of condition variables is provided where
    each top-level, non-nested column with an identifier-like name
    appears. Variables in condvars override the
    default ones.

    When condvars is not provided or
    None, the current local and global namespace is
    sought instead of condvars. The previous
    mechanism is mostly intended for interactive usage. To disable it,
    just specify a (maybe empty) mapping as condvars.

    If a range is supplied (by setting some of the
    start, stop or step parameters), only the rows in that range
    and fulfilling the condition
    are used. The meaning of the start,
    stop and step parameters is
    the same as in the range() Python function,
    except that negative values of step are
    not allowed. Moreover, if only
    start is specified, then
    stop will be set to start+1.

    When possible, indexed columns participating in the
    condition will be used to speed up the search. It is recommended
    that you place the indexed columns as left and out in the
    condition as possible. Anyway, this method has always better
    performance than regular Python selections on the table.

    You can mix this method with regular Python selections in
    order to support even more complex queries. It is strongly
    recommended that you pass the most restrictive condition as the
    parameter to this method if you want to achieve maximum
    performance.

    Example of use::

        >>> passvalues = [ row['col3'] for row in
        ...                table.where('(col1 > 0) & (col2 <= 20)', step=5)
        ...                if your_function(row['col2']) ]
        >>> print "Values that pass the cuts:", passvalues

    Note that, from PyTables 1.1 on, you can nest several
    iterators over the same table. For example::

        for p in rout.where('pressure < 16'):
            for q in rout.where('pressure < 9'):
                for n in rout.where('energy < 10'):
                    print "pressure, energy:", p['pressure'], n['energy']

    In this example, iterators returned by
    :meth:`Table.where` have been used, but you may as
    well use any of the other reading iterators that
    Table objects offer. See the file
    :file:`examples/nested-iter.py` for the full code.

    .. warning:: When in the middle of a table row iterator, you should not
       use methods that can change the number of rows in the table
       (like :meth:`Table.append` or :meth:`Table.removeRows`) or unexpected
       errors will happen.


.. method:: Table.whereAppend(dstTable, condition, condvars=None, start=None, stop=None, step=None)

    Append rows fulfilling the condition to the dstTable table.

    dstTable must be capable of taking the
    rows resulting from the query, i.e. it must have columns with the
    expected names and compatible types. The meaning of the other
    arguments is the same as in the :meth:`Table.where`
    method.

    The number of rows appended to dstTable
    is returned as a result.



.. method:: Table.willQueryUseIndexing(condition, condvars=None)

    Will a query for the condition use indexing?

    The meaning of the condition and
    *condvars* arguments is the same as in the
    :meth:`Table.where` method. If condition can use
    indexing, this method returns a frozenset with the path names of
    the columns whose index is usable. Otherwise, it returns an empty
    list.

    This method is mainly intended for testing. Keep in mind
    that changing the set of indexed columns or their dirtiness may
    make this method return different values for the same arguments at
    different times.


Table methods - other
~~~~~~~~~~~~~~~~~~~~~

.. method:: Table.copy(newparent=None, newname=None, overwrite=False, createparents=False, **kwargs)

    Copy this table and return the new one.

    This method has the behavior and keywords described in
    :meth:`Leaf.copy`.
    Moreover, it recognises the following additional keyword
    arguments.

    Parameters
    ----------
    sortby
        If specified, and sortby
        corresponds to a column with an index, then the copy will be
        sorted by this index.  If you want to ensure a fully sorted
        order, the index must be a CSI one.  A reverse sorted copy
        can be achieved by specifying a negative value for
        the step keyword.
        If sortby is omitted or None, the original table order is used.
    checkCSI
        If true and a CSI index does not exist for the
        sortby column, an error will be raised.
        If false (the default), it does nothing.  You can use this
        flag in order to explicitly check for the existence of a
        CSI index.
    propindexes
        If true, the existing indexes in the source table are
        propagated (created) to the new one.  If false (the
        default), the indexes are not propagated.



.. method:: Table.flushRowsToIndex()

    Add remaining rows in buffers to non-dirty indexes.

    This can be useful when you have chosen non-automatic
    indexing for the table (see the :attr:`Table.autoIndex`
    property in :ref:`TableInstanceVariablesDescr`) and you want to update the indexes on it.


.. method:: Table.getEnum(colname)

    Get the enumerated type associated with the named column.

    If the column named colname (a string)
    exists and is of an enumerated type, the corresponding
    Enum instance (see :ref:`EnumClassDescr`) is
    returned. If it is not of an enumerated type, a
    TypeError is raised. If the column does not
    exist, a KeyError is raised.


.. method:: Table.reIndex()

    Recompute all the existing indexes in the table.

    This can be useful when you suspect that, for any reason,
    the index information for columns is no longer valid and want to
    rebuild the indexes on it.



.. method:: Table.reIndexDirty()

    Recompute the existing indexes in table, *if* they are dirty.

    This can be useful when you have set
    :attr:`Table.autoIndex` (see :ref:`TableInstanceVariablesDescr`) to false for the table
    and you want to update the indexes
    after a invalidating index operation
    (:meth:`Table.removeRows`, for example).




.. _DescriptionClassDescr:

The Description class
~~~~~~~~~~~~~~~~~~~~~
.. class:: Description

    This class represents descriptions of the structure of tables.

    An instance of this class is automatically bound to Table (see 
    :ref:`TableClassDescr`) objects when they are created.  It provides a
    browseable representation of the structure of the table, made of
    non-nested (Col - see :ref:`ColClassDescr`) and nested (Description)
    columns. It also contains information that will allow you to build
    NestedRecArray (see :class:`NestedRecArray`) objects suited for the
    different columns in a table (be they nested or not).

    Column definitions under a description can be accessed as
    attributes of it (*natural naming*). For
    instance, if table.description is a
    Description instance with a column named
    col1 under it, the later can be accessed as
    table.description.col1. If
    col1 is nested and contains a
    col2 column, this can be accessed as
    table.description.col1.col2. Because of natural
    naming, the names of members start with special prefixes, like in
    the Group class (see :ref:`GroupClassDescr`).


Description instance variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Description._v_colObjects

    A dictionary mapping the names of the columns hanging
    directly from the associated table or nested column to their
    respective descriptions (Col - see :ref:`ColClassDescr` or
    Description - see :ref:`DescriptionClassDescr` instances).

.. attribute:: Description._v_dflts

    A dictionary mapping the names of non-nested columns
    hanging directly from the associated table or nested column
    to their respective default values.

.. attribute:: Description._v_dtype

    The NumPy type which reflects the structure of this
    table or nested column.  You can use this as the
    dtype argument of NumPy array factories.

.. attribute:: Description._v_dtypes

    A dictionary mapping the names of non-nested columns
    hanging directly from the associated table or nested column
    to their respective NumPy types.

.. attribute:: Description._v_is_nested

    Whether the associated table or nested column contains
    further nested columns or not.

.. attribute:: Description._v_itemsize

    The size in bytes of an item in this table or nested column.

.. attribute:: Description._v_name

    The name of this description group. The name of the
    root group is '/'.

.. attribute:: Description._v_names

    A list of the names of the columns hanging directly
    from the associated table or nested column. The order of the
    names matches the order of their respective columns in the
    containing table.

.. attribute:: Description._v_nestedDescr

    A nested list of pairs of (name, format) tuples for all the columns under
    this table or nested column. You can use this as the dtype and descr
    arguments of NumPy array and NestedRecArray (see 
    :ref:`NestedRecArrayClassDescr`) factories, respectively.

.. attribute:: Description._v_nestedFormats

    A nested list of the NumPy string formats (and shapes)
    of all the columns under this table or nested column. You
    can use this as the formats argument of
    NumPy array and NestedRecArray (see :class:`NestedRecArray`) factories.

.. attribute:: Description._v_nestedlvl

    The level of the associated table or nested column in
    the nested datatype.

.. attribute:: Description._v_nestedNames

    A nested list of the names of all the columns under this table or nested
    column. You can use this as the names argument of NumPy array and
    NestedRecArray (see :ref:`NestedRecArrayClassDescr`) factories.

.. attribute:: Description._v_pathnames

    A list of the pathnames of all the columns under this
    table or nested column (in preorder).  If it does not
    contain nested columns, this is exactly the same as the
    :attr:`Description._v_names` attribute.

.. attribute:: Description._v_types

    A dictionary mapping the names of non-nested columns
    hanging directly from the associated table or nested column
    to their respective PyTables types.



Description methods
^^^^^^^^^^^^^^^^^^^

.. method:: Description._f_walk(type='All')

    Iterate over nested columns.

    If type is 'All'
    (the default), all column description objects
    (Col and Description
    instances) are yielded in top-to-bottom order (preorder).

    If type is 'Col' or
    'Description', only column descriptions of
    that type are yielded.



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



.. _ColsClassDescr:

The Cols class
~~~~~~~~~~~~~~
.. class:: Cols

    Container for columns in a table or nested column.

    This class is used as an *accessor* to the
    columns in a table or nested column.  It supports the
    *natural naming* convention, so that you can
    access the different columns as attributes which lead to
    Column instances (for non-nested columns) or
    other Cols instances (for nested columns).

    For instance, if table.cols is a
    Cols instance with a column named
    col1 under it, the later can be accessed as
    table.cols.col1. If col1 is
    nested and contains a col2 column, this can be
    accessed as table.cols.col1.col2 and so
    on. Because of natural naming, the names of members start with
    special prefixes, like in the Group class (see
    :ref:`GroupClassDescr`).

    Like the Column class (see :ref:`ColumnClassDescr`),
    Cols supports item access to read and write
    ranges of values in the table or nested column.


Cols instance variables
^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Cols._v_colnames

    A list of the names of the columns hanging directly
    from the associated table or nested column.  The order of
    the names matches the order of their respective columns in
    the containing table.

.. attribute:: Cols._v_colpathnames

    A list of the pathnames of all the columns under the
    associated table or nested column (in preorder).  If it does
    not contain nested columns, this is exactly the same as the
    :attr:`Cols._v_colnames` attribute.

.. attribute:: Cols._v_desc

    The associated Description instance
    (see :ref:`DescriptionClassDescr`).

.. attribute:: Cols._v_table

    The parent Table instance (see :ref:`TableClassDescr`).


Cols methods
^^^^^^^^^^^^

.. method:: Cols._f_col(colname)

    Get an accessor to the column colname.

    This method returns a Column instance
    (see :ref:`ColumnClassDescr`) if the requested column is not nested, and a
    Cols instance (see :ref:`ColsClassDescr`) if it is.
    You may use full column pathnames in colname.

    Calling cols._f_col('col1/col2') is
    equivalent to using cols.col1.col2.  However,
    the first syntax is more intended for programmatic use.  It is
    also better if you want to access columns with names that are
    not valid Python identifiers.


.. method:: Cols.__getitem__(key)

    Get a row or a range of rows from a table or nested column.

    If key argument is an integer, the
    corresponding nested type row is returned as a record of the
    current flavor. If key is a slice, the range
    of rows determined by it is returned as a record array of the
    current flavor.

    Example of use::

        record = table.cols[4]  # equivalent to table[4]
        recarray = table.cols.Info[4:1000:2]

    Those statements are equivalent to::

        nrecord = table.read(start=4)[0]
        nrecarray = table.read(start=4, stop=1000, step=2).field('Info')

    Here you can see how a mix of natural naming, indexing and
    slicing can be used as shorthands for the
    :meth:`Table.read` method.


.. method:: Cols.__len__()

    Get the number of top level columns in table.


.. method:: Cols.__setitem__(key)

    Set a row or a range of rows in a table or nested column.

    If key argument is an integer, the
    corresponding row is set to value. If
    key is a slice, the range of rows determined
    by it is set to value.

    Example of use::

        table.cols[4] = record
        table.cols.Info[4:1000:2] = recarray

    Those statements are equivalent to::

        table.modifyRows(4, rows=record)
        table.modifyColumn(4, 1000, 2, colname='Info', column=recarray)

    Here you can see how a mix of natural naming, indexing and
    slicing can be used as shorthands for the
    :meth:`Table.modifyRows` and
    :meth:`Table.modifyColumn` methods.



.. _ColumnClassDescr:

The Column class
~~~~~~~~~~~~~~~~
.. class:: Column

    Accessor for a non-nested column in a table.

    Each instance of this class is associated with one
    *non-nested* column of a table. These instances
    are mainly used to read and write data from the table columns using
    item access (like the Cols class - see :ref:`ColsClassDescr`), but there
    are a few other associated methods to deal with indexes.


Column instance variables
^^^^^^^^^^^^^^^^^^^^^^^^^
.. attribute:: Column.descr

    The Description (see :ref:`DescriptionClassDescr`) instance of the parent table or nested column.

.. attribute:: Column.dtype

    The NumPy dtype that most closely matches this column.

.. attribute:: Column.index

    The Index instance (see :ref:`IndexClassDescr`)
    associated with this column (None if the
    column is not indexed).

.. attribute:: Column.is_indexed

    True if the column is indexed, false otherwise.

.. attribute:: Column.maindim

    The dimension along which iterators work.

    Its value is 0 (i.e. the first dimension).

.. attribute:: Column.name

    The name of the associated column.

.. attribute:: Column.pathname

    The complete pathname of the associated column (the
    same as Column.name if the column is not
    inside a nested column).

.. attribute:: Column.shape

    The shape of this column.

.. attribute:: Column.table

    The parent Table instance (see :ref:`TableClassDescr`).

.. attribute:: Column.type

    The PyTables type of the column (a string).


Column methods
^^^^^^^^^^^^^^

.. method:: Column.createIndex(optlevel=6, kind="medium", filters=None, tmp_dir=None)

    Create an index for this column.

    Parameters
    ----------
    optlevel : int
        The optimization level for building the index.  The
        levels ranges from 0 (no optimization) up to 9 (maximum
        optimization).  Higher levels of optimization mean better
        chances for reducing the entropy of the index at the price
        of using more CPU, memory and I/O resources for creating
        the index.
    kind : str
        The kind of the index to be built.  It can take the
        'ultralight', 'light',
        'medium' or 'full'
        values.  Lighter kinds ('ultralight'
        and 'light') mean that the index takes
        less space on disk, but will perform queries slower.
        Heavier kinds ('medium'
        and 'full') mean better chances for
        reducing the entropy of the index (increasing the query
        speed) at the price of using more disk space as well as
        more CPU, memory and I/O resources for creating the index.

        Note that selecting a full kind
        with an optlevel of 9 (the maximum)
        guarantees the creation of an index with zero entropy,
        that is, a completely sorted index (CSI) - provided that
        the number of rows in the table does not exceed the 2**48
        figure (that is more than 100 trillions of rows).  See
        :meth:`Column.createCSIndex` method for a
        more direct way to create a CSI index.
    filters : Filters
        Specify the Filters instance used
        to compress the index.  If None,
        default index filters will be used (currently, zlib level
        1 with shuffling).
    tmp_dir
        When kind is other
        than 'ultralight', a temporary file is
        created during the index build process.  You can use the
        tmp_dir argument to specify the
        directory for this temporary file.  The default is to
        create it in the same directory as the file containing the
        original table.

    Notes
    -----
    .. warning:: In some situations it is useful to get a completely
       sorted index (CSI).  For those cases, it is best to use
       the :meth:`Column.createCSIndex` method instead.



.. method:: Column.createCSIndex(filters=None, tmp_dir=None)

    Create a completely sorted index (CSI) for this column.

    This method guarantees the creation of an index with zero
    entropy, that is, a completely sorted index (CSI) -- provided
    that the number of rows in the table does not exceed the 2**48
    figure (that is more than 100 trillions of rows).  A CSI index
    is needed for some table methods (like
    :meth:`Table.itersorted` or
    :meth:`Table.readSorted`) in order to ensure
    completely sorted results.

    For the meaning of filters and
    tmp_dir arguments see
    :meth:`Column.createIndex`.

    .. note:: This method is equivalent to
       Column.createIndex(optlevel=9, kind='full', ...).


.. method:: Column.reIndex()

    Recompute the index associated with this column.

    This can be useful when you suspect that, for any reason,
    the index information is no longer valid and you want to rebuild it.

    This method does nothing if the column is not indexed.


.. method:: Column.reIndexDirty()

    Recompute the associated index only if it is dirty.

    This can be useful when you have set
    :attr:`Table.autoIndex` to false for the table and you want to update the column's
    index after an invalidating index operation
    (like :meth:`Table.removeRows`).

    This method does nothing if the column is not indexed.


.. method:: Column.removeIndex()

    Remove the index associated with this column.

    This method does nothing if the column is not indexed. The
    removed index can be created again by calling the
    :meth:`Column.createIndex` method.




Column special methods
^^^^^^^^^^^^^^^^^^^^^^

.. method:: Column.__getitem__(key)

    Get a row or a range of rows from a column.

    If key argument is an integer, the
    corresponding element in the column is returned as an object of
    the current flavor.  If key is a slice, the
    range of elements determined by it is returned as an array of
    the current flavor.

    Example of use::

        print "Column handlers:"
        for name in table.colnames:
            print table.cols._f_col(name)
            print "Select table.cols.name[1]-->", table.cols.name[1]
            print "Select table.cols.name[1:2]-->", table.cols.name[1:2]
            print "Select table.cols.name[:]-->", table.cols.name[:]
            print "Select table.cols._f_col('name')[:]-->", table.cols._f_col('name')[:]

    The output of this for a certain arbitrary table is::

        Column handlers:
        /table.cols.name (Column(), string, idx=None)
        /table.cols.lati (Column(), int32, idx=None)
        /table.cols.longi (Column(), int32, idx=None)
        /table.cols.vector (Column(2,), int32, idx=None)
        /table.cols.matrix2D (Column(2, 2), float64, idx=None)
        Select table.cols.name[1]--> Particle:     11
        Select table.cols.name[1:2]--> ['Particle:     11']
        Select table.cols.name[:]--> ['Particle:     10'
         'Particle:     11' 'Particle:     12'
         'Particle:     13' 'Particle:     14']
        Select table.cols._f_col('name')[:]--> ['Particle:     10'
         'Particle:     11' 'Particle:     12'
         'Particle:     13' 'Particle:     14']

    See the :file:`examples/table2.py` file for a
    more complete example.


.. method:: Column.__len__()

    Get the number of elements in the column.

    This matches the length in rows of the parent table.



.. method:: Column.__setitem__(key, value)

    Set a row or a range of rows in a column.

    If key argument is an integer, the
    corresponding element is set to value.  If
    key is a slice, the range of elements
    determined by it is set to value.

    Example of use::

        # Modify row 1
        table.cols.col1[1] = -1

        # Modify rows 1 and 3
        table.cols.col1[1::2] = [2,3]

    Which is equivalent to::

        # Modify row 1
        table.modifyColumns(start=1, columns=[[-1]], names=['col1'])

        # Modify rows 1 and 3
        columns = numpy.rec.fromarrays([[2,3]], formats='i4')
        table.modifyColumns(start=1, step=2, columns=columns, names=['col1'])


.. _ArrayClassDescr:

The Array class
---------------
.. class:: Array

    This class represents homogeneous datasets in an HDF5 file.

    This class provides methods to write or read data to or from array objects
    in the file. This class does not allow you neither to enlarge nor compress
    the datasets on disk; use the EArray class (see :ref:`EArrayClassDescr`)
    if you want enlargeable dataset support or compression features, or CArray
    (see :ref:`CArrayClassDescr`) if you just want compression.

    An interesting property of the Array class is
    that it remembers the *flavor* of the object that
    has been saved so that if you saved, for example, a
    list, you will get a list during
    readings afterwards; if you saved a NumPy array, you will get a NumPy
    object, and so forth.

    Note that this class inherits all the public attributes and
    methods that Leaf (see :ref:`LeafClassDescr`) already
    provides. However, as Array instances have no
    internal I/O buffers, it is not necessary to use the
    flush() method they inherit from
    Leaf in order to save their internal state to disk.
    When a writing method call returns, all the data is already on disk.


.. _ArrayClassInstanceVariables:

Array instance variables
~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Array.atom

    An Atom (see :ref:`AtomClassDescr`)
    instance representing the *type* and
    *shape* of the atomic objects to be
    saved.

.. attribute:: Array.rowsize

    The size of the rows in dimensions orthogonal to
    *maindim*.

.. attribute:: Array.nrow

    On iterators, this is the index of the current row.


Array methods
~~~~~~~~~~~~~

.. method:: Array.getEnum()

    Get the enumerated type associated with this array.

    If this array is of an enumerated type, the corresponding
    Enum instance (see :ref:`EnumClassDescr`) is
    returned. If it is not of an enumerated type, a
    TypeError is raised.


.. method:: Array.iterrows(start=None, stop=None, step=None)

    Iterate over the rows of the array.

    This method returns an iterator yielding an object of the
    current flavor for each selected row in the array.  The returned
    rows are taken from the *main dimension*.

    If a range is not supplied, *all the
    rows* in the array are iterated upon - you can also use
    the :meth:`Array.__iter__` special method for that purpose.  If you only want
    to iterate over a given *range of rows* in the
    array, you may use the start,
    stop and step parameters,
    which have the same meaning as in :meth:`Array.read`.

    Example of use::

        result = [row for row in arrayInstance.iterrows(step=4)]

.. method:: Array.next()

    Get the next element of the array during an iteration.

    The element is returned as an object of the current flavor.


.. method:: Array.read(start=None, stop=None, step=None)

    Get data in the array as an object of the current flavor.

    The start, stop and
    step parameters can be used to select only a
    *range of rows* in the array.  Their meanings
    are the same as in the built-in range() Python
    function, except that negative values of step
    are not allowed yet. Moreover, if only start is
    specified, then stop will be set to
    start+1. If you do not specify neither
    start nor stop, then
    *all the rows* in the array are
    selected.


Array special methods
~~~~~~~~~~~~~~~~~~~~~
The following methods automatically trigger actions when an
Array instance is accessed in a special way
(e.g. array[2:3,...,::2] will be equivalent to a
call to array.__getitem__((slice(2, 3, None), Ellipsis,
slice(None, None, 2)))).


.. method:: Array.__getitem__(key)

    Get a row, a range of rows or a slice from the array.

    The set of tokens allowed for the key is
    the same as that for extended slicing in Python (including the
    Ellipsis or ... token).  The
    result is an object of the current flavor; its shape depends on
    the kind of slice used as key and the shape of
    the array itself.

    Furthermore, NumPy-style fancy indexing, where a list of
    indices in a certain axis is specified, is also supported.  Note
    that only one list per selection is supported right now.  Finally,
    NumPy-style point and boolean selections are supported as well.

    Example of use::

        array1 = array[4]                       # simple selection
        array2 = array[4:1000:2]                # slice selection
        array3 = array[1, ..., ::2, 1:4, 4:]    # general slice selection
        array4 = array[1, [1,5,10], ..., -1]    # fancy selection
        array5 = array[np.where(array[:] > 4)]  # point selection
        array6 = array[array[:] > 4]            # boolean selection


.. method:: Array.__iter__()

    Iterate over the rows of the array.

    This is equivalent to calling
    :meth:`Array.iterrows` with default arguments, i.e. it
    iterates over *all the rows* in the array.

    Example of use::

        result = [row[2] for row in array]

    Which is equivalent to::

        result = [row[2] for row in array.iterrows()]


.. method:: Array.__setitem__(key, value)

    Set a row, a range of rows or a slice in the array.

    It takes different actions depending on the type of the
    key parameter: if it is an integer, the
    corresponding array row is set to value (the
    value is broadcast when needed).  If key is a
    slice, the row slice determined by it is set to
    value (as usual, if the slice to be updated
    exceeds the actual shape of the array, only the values in the
    existing range are updated).

    If value is a multidimensional object,
    then its shape must be compatible with the shape determined by
    key, otherwise, a ValueError
    will be raised.

    Furthermore, NumPy-style fancy indexing, where a list of
    indices in a certain axis is specified, is also supported.  Note
    that only one list per selection is supported right now.  Finally,
    NumPy-style point and boolean selections are supported as well.

    Example of use::

        a1[0] = 333        # assign an integer to a Integer Array row
        a2[0] = 'b'        # assign a string to a string Array row
        a3[1:4] = 5        # broadcast 5 to slice 1:4
        a4[1:4:2] = 'xXx'  # broadcast 'xXx' to slice 1:4:2

        # General slice update (a5.shape = (4,3,2,8,5,10).
        a5[1, ..., ::2, 1:4, 4:] = numpy.arange(1728, shape=(4,3,2,4,3,6))
        a6[1, [1,5,10], ..., -1] = arr    # fancy selection
        a7[np.where(a6[:] > 4)] = 4       # point selection + broadcast
        a8[arr > 4] = arr2                # boolean selection



.. _CArrayClassDescr:

The CArray class
----------------
.. class:: CArray

    This class represents homogeneous datasets in an HDF5 file.

    The difference between a CArray and a normal Array (see
    :ref:`ArrayClassDescr`), from which it inherits, is that a CArray has a
    chunked layout and, as a consequence, it supports compression.
    You can use datasets of this class to easily save or load arrays to or
    from disk, with compression support included.


Examples of use
~~~~~~~~~~~~~~~
See below a small example of the use of the
CArray class.  The code is available in
:file:`examples/carray1.py`::

    import numpy
    import tables

    fileName = 'carray1.h5'
    shape = (200, 300)
    atom = tables.UInt8Atom()
    filters = tables.Filters(complevel=5, complib='zlib')

    h5f = tables.openFile(fileName, 'w')
    ca = h5f.createCArray(h5f.root, 'carray', atom, shape, filters=filters)

    # Fill a hyperslab in ``ca``.
    ca[10:60, 20:70] = numpy.ones((50, 50))
    h5f.close()

    # Re-open and read another hyperslab
    h5f = tables.openFile(fileName)
    print h5f
    print h5f.root.carray[8:12, 18:22]
    h5f.close()

The output for the previous script is something like::

    carray1.h5 (File) ''
    Last modif.: 'Thu Apr 12 10:15:38 2007'
    Object Tree:
    / (RootGroup) ''
    /carray (CArray(200, 300), shuffle, zlib(5)) ''
    [[0 0 0 0]
    [0 0 0 0]
    [0 0 1 1]
    [0 0 1 1]]


.. _EArrayClassDescr:

The EArray class
----------------
.. class:: EArray

    This class represents extendable, homogeneous datasets in an HDF5 file.

    The main difference between an EArray and a CArray (see 
    :ref:`CArrayClassDescr`), from which it inherits, is that the former can
    be enlarged along one of its dimensions, the *enlargeable dimension*.
    That means that the :attr:`Leaf.extdim` attribute (see
    :ref:`LeafInstanceVariables`) of any EArray instance will always be
    non-negative.
    Multiple enlargeable dimensions might be supported in the future.

    New rows can be added to the end of an enlargeable array by using the
    :meth:`EArray.append` method.


.. _EArrayMethodsDescr:

EArray methods
~~~~~~~~~~~~~~

.. method:: EArray.append(sequence)

    Add a sequence of data to the end of the dataset.

    The sequence must have the same type as the array; otherwise
    a TypeError is raised. In the same way, the
    dimensions of the sequence must conform to the
    shape of the array, that is, all dimensions must match, with the
    exception of the enlargeable dimension, which can be of any length
    (even 0!).  If the shape of the sequence is
    invalid, a ValueError is raised.


Examples of use
~~~~~~~~~~~~~~~
See below a small example of the use of the
EArray class.  The code is available in
:file:`examples/earray1.py`::

    import tables
    import numpy

    fileh = tables.openFile('earray1.h5', mode='w')
    a = tables.StringAtom(itemsize=8)

    # Use ''a'' as the object type for the enlargeable array.
    array_c = fileh.createEArray(fileh.root, 'array_c', a, (0,), "Chars")
    array_c.append(numpy.array(['a'*2, 'b'*4], dtype='S8'))
    array_c.append(numpy.array(['a'*6, 'b'*8, 'c'*10], dtype='S8'))

    # Read the string ''EArray'' we have created on disk.
    for s in array_c:
        print 'array_c[%s] => %r' % (array_c.nrow, s)

    # Close the file.
    fileh.close()

The output for the previous script is something like::

    array_c[0] => 'aa'
    array_c[1] => 'bbbb'
    array_c[2] => 'aaaaaa'
    array_c[3] => 'bbbbbbbb'
    array_c[4] => 'cccccccc'



.. _VLArrayClassDescr:

The VLArray class
-----------------
.. class:: VLArray

    This class represents variable length (ragged) arrays in an HDF5 file.

    Instances of this class represent array objects in the object
    tree with the property that their rows can have a
    *variable* number of homogeneous elements, called
    *atoms*. Like Table datasets (see :ref:`TableClassDescr`),
    variable length arrays can have only one dimension, and the elements
    (atoms) of their rows can be fully multidimensional.
    VLArray objects do also support compression.

    When reading a range of rows from a VLArray,
    you will *always* get a Python list of objects of
    the current flavor (each of them for a row), which may have different
    lengths.

    This class provides methods to write or read data to or from
    variable length array objects in the file. Note that it also inherits
    all the public attributes and methods that Leaf
    (see :ref:`LeafClassDescr`)
    already provides.


VLArray instance variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: VLArray.atom

    An Atom (see :ref:`AtomClassDescr`)
    instance representing the *type* and
    *shape* of the atomic objects to be
    saved. You may use a *pseudo-atom* for
    storing a serialized object or variable length string per row.


.. attribute:: VLArray.flavor

    The type of data object read from this leaf.

    Please note that when reading several rows of
    VLArray data, the flavor only applies to
    the *components* of the returned Python
    list, not to the list itself.


.. attribute:: VLArray.nrow

    On iterators, this is the index of the current row.



VLArray methods
~~~~~~~~~~~~~~~

.. method:: VLArray.append(sequence)

    Add a sequence of data to the end of the dataset.

    This method appends the objects in the
    sequence to a *single row*
    in this array. The type and shape of individual objects must be
    compliant with the atoms in the array. In the case of serialized
    objects and variable length strings, the object or string to
    append is itself the sequence.

.. method:: VLArray.getEnum()

    Get the enumerated type associated with this array.

    If this array is of an enumerated type, the corresponding
    Enum instance (see :ref:`EnumClassDescr`) is
    returned. If it is not of an enumerated type, a
    TypeError is raised.



.. method:: VLArray.iterrows(start=None, stop=None, step=None)

    Iterate over the rows of the array.

    This method returns an iterator yielding an object of the
    current flavor for each selected row in the array.

    If a range is not supplied, *all the
    rows* in the array are iterated upon —you can also use
    the :meth:`VLArray.__iter__` special method for that purpose.
    If you only want to iterate over a given *range of
    rows* in the array, you may use the
    start, stop and
    step parameters, which have the same meaning as
    in :meth:`VLArray.read`.

    Example of use::

        for row in vlarray.iterrows(step=4):
            print '%s[%d]--> %s' % (vlarray.name, vlarray.nrow, row)

.. method:: VLArray.next()

    Get the next element of the array during an iteration.

    The element is returned as a list of objects of the current flavor.


.. method:: VLArray.read(start=None, stop=None, step=1)

    Get data in the array as a list of objects of the current flavor.

    Please note that, as the lengths of the different rows are
    variable, the returned value is a *Python list*
    (not an array of the current flavor), with as many entries as
    specified rows in the range parameters.

    The start, stop and
    step parameters can be used to select only a
    *range of rows* in the array.  Their meanings
    are the same as in the built-in range() Python
    function, except that negative values of step
    are not allowed yet. Moreover, if only start is
    specified, then stop will be set to
    start+1. If you do not specify neither
    start nor stop, then
    *all the rows* in the array are
    selected.



VLArray special methods
~~~~~~~~~~~~~~~~~~~~~~~
The following methods automatically trigger actions when a
VLArray instance is accessed in a special way
(e.g., vlarray[2:5] will be equivalent to a call
to vlarray.__getitem__(slice(2, 5, None)).

.. method:: VLArray.__getitem__(key)

    Get a row or a range of rows from the array.

    If key argument is an integer, the
    corresponding array row is returned as an object of the current
    flavor.  If key is a slice, the range of rows
    determined by it is returned as a list of objects of the current
    flavor.

    In addition, NumPy-style point selections are supported.  In
    particular, if key is a list of row
    coordinates, the set of rows determined by it is returned.
    Furthermore, if key is an array of boolean
    values, only the coordinates where key
    is True are returned.  Note that for the latter
    to work it is necessary that key list would
    contain exactly as many rows as the array has.

    Example of use:::

        a_row = vlarray[4]
        a_list = vlarray[4:1000:2]
        a_list2 = vlarray[[0,2]]   # get list of coords
        a_list3 = vlarray[[0,-2]]  # negative values accepted
        a_list4 = vlarray[numpy.array([True,...,False])]  # array of bools


.. method:: VLArray.__iter__()

    Iterate over the rows of the array.

    This is equivalent to calling
    :meth:`VLArray.iterrows` with default arguments, i.e. it
    iterates over *all the rows* in the array.

    Example of use::

        result = [row for row in vlarray]

    Which is equivalent to::

        result = [row for row in vlarray.iterrows()]


.. method:: VLArray.__setitem__(key, value)

    Set a row, or set of rows, in the array.

    It takes different actions depending on the type of the
    key parameter: if it is an integer, the
    corresponding table row is set to value (a
    record or sequence capable of being converted to the table
    structure). If key is a slice, the row slice
    determined by it is set to value (a record
    array or sequence capable of being converted to the table
    structure).

    In addition, NumPy-style point selections are supported.  In
    particular, if key is a list of row
    coordinates, the set of rows determined by it is set
    to value.  Furthermore,
    if key is an array of boolean values, only the
    coordinates where key
    is True are set to values
    from value.  Note that for the latter to work
    it is necessary that key list would contain
    exactly as many rows as the table has.

    .. note:: When updating the rows of a VLArray
       object which uses a pseudo-atom, there is a problem: you can
       only update values with *exactly* the same
       size in bytes than the original row.  This is very difficult to
       meet with object pseudo-atoms, because
       cPickle applied on a Python object does not
       guarantee to return the same number of bytes than over another
       object, even if they are of the same class.  This effectively
       limits the kinds of objects than can be updated in
       variable-length arrays.

    Example of use::

        vlarray[0] = vlarray[0] * 2 + 3
        vlarray[99] = arange(96) * 2 + 3

        # Negative values for the index are supported.
        vlarray[-99] = vlarray[5] * 2 + 3
        vlarray[1:30:2] = list_of_rows
        vlarray[[1,3]] = new_1_and_3_rows



Examples of use
~~~~~~~~~~~~~~~
See below a small example of the use of the
VLArray class.  The code is available in
:file:`examples/vlarray1.py`::

    import tables
    from numpy import *

    # Create a VLArray:
    fileh = tables.openFile('vlarray1.h5', mode='w')
    vlarray = fileh.createVLArray(fileh.root, 'vlarray1',
    tables.Int32Atom(shape=()),
                    "ragged array of ints",
                    filters=tables.Filters(1))

    # Append some (variable length) rows:
    vlarray.append(array([5, 6]))
    vlarray.append(array([5, 6, 7]))
    vlarray.append([5, 6, 9, 8])

    # Now, read it through an iterator:
    print '-->', vlarray.title
    for x in vlarray:
        print '%s[%d]--> %s' % (vlarray.name, vlarray.nrow, x)

    # Now, do the same with native Python strings.
    vlarray2 = fileh.createVLArray(fileh.root, 'vlarray2',
    tables.StringAtom(itemsize=2),
                        "ragged array of strings",
                        filters=tables.Filters(1))
    vlarray2.flavor = 'python'

    # Append some (variable length) rows:
    print '-->', vlarray2.title
    vlarray2.append(['5', '66'])
    vlarray2.append(['5', '6', '77'])
    vlarray2.append(['5', '6', '9', '88'])

    # Now, read it through an iterator:
    for x in vlarray2:
        print '%s[%d]--> %s' % (vlarray2.name, vlarray2.nrow, x)

    # Close the file.
    fileh.close()

The output for the previous script is something like::

    --> ragged array of ints
    vlarray1[0]--> [5 6]
    vlarray1[1]--> [5 6 7]
    vlarray1[2]--> [5 6 9 8]
    --> ragged array of strings
    vlarray2[0]--> ['5', '66']
    vlarray2[1]--> ['5', '6', '77']
    vlarray2[2]--> ['5', '6', '9', '88']



The Link class
--------------
.. class:: Link

    Abstract base class for all PyTables links.

    A link is a node that refers to another node.
    The Link class inherits
    from Node class and the links that inherits
    from Link are SoftLink
    and ExternalLink.  There is not
    a HardLink subclass because hard links behave
    like a regular Group or Leaf.
    Contrarily to other nodes, links cannot have HDF5 attributes.  This
    is an HDF5 library limitation that might be solved in future
    releases.

    See :ref:`LinksTutorial` for a small tutorial on how
    to work with links.


Link instance variables
~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Link._v_attrs

    A NoAttrs instance replacing the
    typical *AttributeSet* instance of other
    node objects.  The purpose of NoAttrs is
    to make clear that HDF5 attributes are not supported in link
    nodes.


.. attribute:: Link.target

    The path string to the pointed node.


Link methods
~~~~~~~~~~~~
The following methods are useful for copying, moving, renaming
and removing links.


.. method:: Link.copy(newparent=None, newname=None, overwrite=False, createparents=False)

    Copy this link and return the new one.

    See :meth:`Node._f_copy` for a complete explanation of
    the arguments.  Please note that there is no
    recursive flag since links do not have child nodes.


.. method:: Link.move(newparent=None, newname=None, overwrite=False)

    Move or rename this link.

    See :meth:`Node._f_move` for a complete explanation of the arguments.


.. method:: Link.remove()

    Remove this link from the hierarchy.


.. method:: Link.rename(newname=None)

    Rename this link in place.

    See :meth:`Node._f_rename` for a complete explanation of the arguments.



.. _SoftLinkClassDescr:

The SoftLink class
------------------
.. class:: SoftLink

    Represents a soft link (aka symbolic link).

    A soft link is a reference to another node in
    the *same* file hierarchy.  Getting access to the
    pointed node (this action is
    called *dereferrencing*) is done via
    the __call__ special method (see below).


SoftLink special methods
~~~~~~~~~~~~~~~~~~~~~~~~
The following methods are specific for dereferrencing and
representing soft links.


.. method:: SoftLink.__call__()

    Dereference self.target and return the
    pointed object.

    Example of use::

        >>> f=tables.openFile('data/test.h5')
        >>> print f.root.link0
        /link0 (SoftLink) -> /another/path
        >>> print f.root.link0()
        /another/path (Group) ''

.. method:: SoftLink.__str__()

    Return a short string representation of the link.

    Example of use::

        >>> f=tables.openFile('data/test.h5')
        >>> print f.root.link0
        /link0 (SoftLink) -> /path/to/node




.. _ExternalLinkClassDescr:

The ExternalLink class
----------------------
.. class:: ExternalLink

    Represents an external link.

    An external link is a reference to a node
    in *another* file.  Getting access to the pointed
    node (this action is called *dereferrencing*) is
    done via the __call__ special method (see
    below).

    .. warning:: External links are only supported when PyTables is compiled
       against HDF5 1.8.x series.  When using PyTables with HDF5 1.6.x,
       the *parent* group containing external link
       objects will be mapped to an Unknown instance
       (see :ref:`UnknownClassDescr`) and you won't be able to access *any* node
       hanging of this parent group.  It follows that if the parent group
       containing the external link is the root group, you won't be able
       to read *any* information contained in the file
       when using HDF5 1.6.x.


ExternalLink instance variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: ExternalLink.extfile

    The external file handler, if the link has been
    dereferenced.  In case the link has not been dereferenced
    yet, its value is None.



ExternalLink methods
~~~~~~~~~~~~~~~~~~~~

.. method:: ExternalLink.umount()

    Safely unmount self.extfile, if opened.


ExternalLink special methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following methods are specific for dereferrencing and
representing external links.


.. method:: ExternalLink.__call__(**kwargs)

    Dereference self.target and return the
    pointed object.

    You can pass all the arguments supported by the
    :func:`openFile` function
    (except filename, of course) so as to open
    the referenced external file.

    Example of use::

        >>> f=tables.openFile('data1/test1.h5')
        >>> print f.root.link2
        /link2 (ExternalLink) -> data2/test2.h5:/path/to/node
        >>> plink2 = f.root.link2('a')  # open in 'a'ppend mode
        >>> print plink2
        /path/to/node (Group) ''
        >>> print plink2._v_filename
        'data2/test2.h5'        # belongs to referenced file

.. method:: ExternalLink.__str__()

    Return a short string representation of the link.

    Example of use::

        >>> f=tables.openFile('data1/test1.h5')
        >>> print f.root.link2
        /link2 (ExternalLink) -> data2/test2.h5:/path/to/node



.. _UnImplementedClassDescr:

The UnImplemented class
-----------------------
.. class:: UnImplemented

    This class represents datasets not supported by PyTables in an
    HDF5 file.

    When reading a generic HDF5 file (i.e. one that has not been
    created with PyTables, but with some other HDF5 library based tool),
    chances are that the specific combination of datatypes or dataspaces
    in some dataset might not be supported by PyTables yet. In such a
    case, this dataset will be mapped into an
    UnImplemented instance and the user will still be
    able to access the complete object tree of the generic HDF5 file. The
    user will also be able to *read and write the
    attributes* of the dataset, *access some of its
    metadata*, and perform *certain hierarchy
    manipulation operations* like deleting or moving (but not
    copying) the node. Of course, the user will not be able to read the
    actual data on it.

    This is an elegant way to allow users to work with generic HDF5
    files despite the fact that some of its datasets are not supported by
    PyTables. However, if you are really interested in having full access
    to an unimplemented dataset, please get in contact with the developer
    team.

    This class does not have any public instance variables or
    methods, except those inherited from the Leaf class
    (see :ref:`LeafClassDescr`).


.. _UnknownClassDescr:

The Unknown class
-----------------
.. class:: Unknown

    This class represents nodes reported
    as *unknown* by the underlying HDF5 library.

    This class does not have any public instance variables or
    methods, except those inherited from the Node class.



.. _AttributeSetClassDescr:

The AttributeSet class
----------------------
.. class:: AttributeSet

    Container for the HDF5 attributes of a Node
    (see :ref:`NodeClassDescr`).

    This class provides methods to create new HDF5 node attributes,
    and to get, rename or delete existing ones.

    Like in Group instances (see :ref:`GroupClassDescr`),
    AttributeSet instances make use of the
    *natural naming* convention, i.e. you can access
    the attributes on disk as if they were normal Python attributes of the
    AttributeSet instance.

    This offers the user a very convenient way to access HDF5 node
    attributes. However, for this reason and in order not to pollute the
    object namespace, one can not assign *normal*
    attributes to AttributeSet instances, and their
    members use names which start by special prefixes as happens with
    Group objects.


Notes on native and pickled attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The values of most basic types are saved as HDF5 native data
in the HDF5 file.  This includes Python bool,
int, float,
complex and str (but not
long nor unicode) values, as
well as their NumPy scalar versions and homogeneous or
*structured* NumPy arrays of them.  When read,
these values are always loaded as NumPy scalar or array objects, as
needed.

For that reason, attributes in native HDF5 files will be
always mapped into NumPy objects.  Specifically, a multidimensional
attribute will be mapped into a multidimensional
ndarray and a scalar will be mapped into a NumPy
scalar object (for example, a scalar
H5T_NATIVE_LLONG will be read and returned as a
numpy.int64 scalar).

However, other kinds of values are serialized using
cPickle, so you only will be able to correctly
retrieve them using a Python-aware HDF5 library.  Thus, if you want
to save Python scalar values and make sure you are able to read them
with generic HDF5 tools, you should make use of *scalar or
homogeneous/structured array NumPy objects* (for example,
numpy.int64(1) or numpy.array([1, 2, 3],
dtype='int16')).

One more advice: because of the various potential difficulties
in restoring a Python object stored in an attribute, you may end up
getting a cPickle string where a Python object is
expected. If this is the case, you may wish to run
cPickle.loads() on that string to get an idea of
where things went wrong, as shown in this example::

    >>> import os, tempfile
    >>> import tables
    >>>
    >>> class MyClass(object):
    ...   foo = 'bar'
    ...
    >>> myObject = MyClass()  # save object of custom class in HDF5 attr
    >>> h5fname = tempfile.mktemp(suffix='.h5')
    >>> h5f = tables.openFile(h5fname, 'w')
    >>> h5f.root._v_attrs.obj = myObject  # store the object
    >>> print h5f.root._v_attrs.obj.foo  # retrieve it
    bar
    >>> h5f.close()
    >>>
    >>> del MyClass, myObject  # delete class of object and reopen file
    >>> h5f = tables.openFile(h5fname, 'r')
    >>> print repr(h5f.root._v_attrs.obj)
    'ccopy_reg\\n_reconstructor...
    >>> import cPickle  # let's unpickle that to see what went wrong
    >>> cPickle.loads(h5f.root._v_attrs.obj)
    Traceback (most recent call last):
    ...
    AttributeError: 'module' object has no attribute 'MyClass'
    >>> # So the problem was not in the stored object,
    ... # but in the *environment* where it was restored.
    ... h5f.close()
    >>> os.remove(h5fname)


AttributeSet instance variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: AttributeSet._v_attrnames

    A list with all attribute names.


.. attribute:: AttributeSet._v_attrnamessys

    A list with system attribute names.


.. attribute:: AttributeSet._v_attrnamesuser

    A list with user attribute names.

.. attribute:: AttributeSet._v_node

    The Node instance (see :ref:`NodeClassDescr`) this
    attribute set is associated with.


.. attribute:: AttributeSet._v_unimplemented

    A list of attribute names with unimplemented native HDF5 types.


AttributeSet methods
~~~~~~~~~~~~~~~~~~~~
Note that this class overrides the
__getattr__(), __setattr__()
and __delattr__() special methods.  This allows
you to read, assign or delete attributes on disk by just using the
next constructs::

    leaf.attrs.myattr = 'str attr'    # set a string (native support)
    leaf.attrs.myattr2 = 3            # set an integer (native support)
    leaf.attrs.myattr3 = [3, (1, 2)]  # a generic object (Pickled)
    attrib = leaf.attrs.myattr        # get the attribute ``myattr``
    del leaf.attrs.myattr             # delete the attribute ``myattr``

In addition, the dictionary-like
__getitem__(), __setitem__()
and __delitem__() methods are available, so you
may write things like this::

    for name in :attr:`Node._v_attrs`._f_list():
        print "name: %s, value: %s" % (name, :attr:`Node._v_attrs`[name])

Use whatever idiom you prefer to access the attributes.

If an attribute is set on a target node that already has a
large number of attributes, a PerformanceWarning
will be issued.


.. method:: AttributeSet._f_copy(where)

    Copy attributes to the where node.

    Copies all user and certain system attributes to the given
    where node (a Node instance - see :ref:`NodeClassDescr`),
    replacing the existing ones.

.. method:: AttributeSet._f_list(attrset='user')

    Get a list of attribute names.

    The attrset string selects the attribute
    set to be used.  A 'user' value returns only
    user attributes (this is the default).  A 'sys'
    value returns only system attributes.  Finally,
    'all' returns both system and user
    attributes.


.. method:: AttributeSet._f_rename(oldattrname, newattrname)

    Rename an attribute from oldattrname to newattrname.


.. method:: AttributeSet.__contains__(name)

    Is there an attribute with that name?

    A true value is returned if the attribute set has an
    attribute with the given name, false otherwise.


Declarative classes
-------------------
In this section a series of classes that are meant to
*declare* datatypes that are required for creating
primary PyTables datasets are described.


.. _AtomClassDescr:

The Atom class and its descendants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. class:: Atom

    Defines the type of atomic cells stored in a dataset.

    The meaning of *atomic* is that individual
    elements of a cell can not be extracted directly by indexing (i.e.
    __getitem__()) the dataset; e.g. if a dataset has
    shape (2, 2) and its atoms have shape (3,), to get the third element
    of the cell at (1, 0) one should use
    dataset[1,0][2] instead of dataset[1,0,2].

    The Atom class is meant to declare the
    different properties of the *base element* (also
    known as *atom*) of CArray,
    EArray and VLArray datasets,
    although they are also used to describe the base elements of
    Array datasets. Atoms have the property that
    their length is always the same.  However, you can grow datasets
    along the extensible dimension in the case of
    EArray or put a variable number of them on a
    VLArray row. Moreover, they are not
    restricted to scalar values, and they can be *fully
    multidimensional objects*.

    A series of descendant classes are offered in order to make
    the use of these element descriptions easier. You should use a
    particular Atom descendant class whenever you
    know the exact type you will need when writing your code. Otherwise,
    you may use one of the Atom.from_*() factory Methods.


Atom instance variables
^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Atom.dflt

    The default value of the atom.

    If the user does not supply a value for an element
    while filling a dataset, this default value will be written
    to disk. If the user supplies a scalar value for a
    multidimensional atom, this value is automatically
    *broadcast* to all the items in the atom
    cell. If dflt is not supplied, an
    appropriate zero value (or *null* string)
    will be chosen by default.  Please note that default values
    are kept internally as NumPy objects.

.. attribute:: Atom.dtype

    The NumPy dtype that most closely matches this atom.


.. attribute:: Atom.itemsize

    Size in bytes of a single item in the atom.

    Specially useful for atoms of the string kind.


.. attribute:: Atom.kind

    The PyTables kind of the atom (a string).


.. attribute:: Atom.recarrtype

    String type to be used in numpy.rec.array().


.. attribute:: Atom.shape

    The shape of the atom (a tuple for scalar atoms).

.. attribute:: Atom.size

    Total size in bytes of the atom.


.. attribute:: Atom.type

    The PyTables type of the atom (a string).

    Atoms can be compared with atoms and other objects for
    strict (in)equality without having to compare individual
    attributes::

        >>> atom1 = StringAtom(itemsize=10)  # same as ``atom2``
        >>> atom2 = Atom.from_kind('string', 10)  # same as ``atom1``
        >>> atom3 = IntAtom()
        >>> atom1 == 'foo'
        False
        >>> atom1 == atom2
        True
        >>> atom2 != atom1
        False
        >>> atom1 == atom3
        False
        >>> atom3 != atom2
        True



Atom methods
^^^^^^^^^^^^

.. method:: Atom.copy(**override)

    Get a copy of the atom, possibly overriding some arguments.

    Constructor arguments to be overridden must be passed as
    keyword arguments::

        >>> atom1 = StringAtom(itemsize=12)
        >>> atom2 = atom1.copy()
        >>> print atom1
        StringAtom(itemsize=12, shape=(), dflt='')
        >>> print atom2
        StringAtom(itemsize=12, shape=(), dflt='')
        >>> atom1 is atom2
        False
        >>> atom3 = atom1.copy(itemsize=100, shape=(2, 2))
        >>> print atom3
        StringAtom(itemsize=100, shape=(2, 2), dflt='')
        >>> atom1.copy(foobar=42)
        Traceback (most recent call last):
        ...
        TypeError: __init__() got an unexpected keyword argument 'foobar'


Atom factory methods
^^^^^^^^^^^^^^^^^^^^

.. method:: Atom.from_dtype(dtype, dflt=None)

    Create an Atom from a NumPy dtype.

    An optional default value may be specified as the
    dflt argument. Information in the
    dtype not represented in an Atom is ignored::

        >>> import numpy
        >>> Atom.from_dtype(numpy.dtype((numpy.int16, (2, 2))))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_dtype(numpy.dtype('S5'), dflt='hello')
        StringAtom(itemsize=5, shape=(), dflt='hello')
        >>> Atom.from_dtype(numpy.dtype('Float64'))
        Float64Atom(shape=(), dflt=0.0)

.. method:: Atom.from_kind(kind, itemsize=None, shape=(), dflt=None)

    Create an Atom from a PyTables kind.

    Optional item size, shape and default value may be
    specified as the itemsize, shape and dflt
    arguments, respectively. Bear in mind that not all atoms support
    a default item size::

        >>> Atom.from_kind('int', itemsize=2, shape=(2, 2))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_kind('int', shape=(2, 2))
        Int32Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_kind('int', shape=1)
        Int32Atom(shape=(1,), dflt=0)
        >>> Atom.from_kind('string', itemsize=5, dflt='hello')
        StringAtom(itemsize=5, shape=(), dflt='hello')
        >>> Atom.from_kind('string', dflt='hello')
        Traceback (most recent call last):
        ...
        ValueError: no default item size for kind ``string``
        >>> Atom.from_kind('Float')
        Traceback (most recent call last):
        ...
        ValueError: unknown kind: 'Float'

    Moreover, some kinds with atypical constructor signatures
    are not supported; you need to use the proper
    constructor::

        >>> Atom.from_kind('enum')
        Traceback (most recent call last):
        ...
        ValueError: the ``enum`` kind is not supported...


.. method:: Atom.from_sctype(sctype, shape=(), dflt=None)

    Create an Atom from a NumPy scalar type sctype.

    Optional shape and default value may be specified as the
    shape and dflt
    arguments, respectively. Information in the
    sctype not represented in an Atom is ignored::

        >>> import numpy
        >>> Atom.from_sctype(numpy.int16, shape=(2, 2))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_sctype('S5', dflt='hello')
        Traceback (most recent call last):
        ...
        ValueError: unknown NumPy scalar type: 'S5'
        >>> Atom.from_sctype('Float64')
        Float64Atom(shape=(), dflt=0.0)


.. method:: Atom.from_type(type, shape=(), dflt=None)

    Create an Atom from a PyTables type.

    Optional shape and default value may be specified as the
    shape and dflt arguments, respectively::

        >>> Atom.from_type('bool')
        BoolAtom(shape=(), dflt=False)
        >>> Atom.from_type('int16', shape=(2, 2))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_type('string40', dflt='hello')
        Traceback (most recent call last):
        ...
        ValueError: unknown type: 'string40'
        >>> Atom.from_type('Float64')
        Traceback (most recent call last):
        ...
        ValueError: unknown type: 'Float64'



.. _AtomConstructors:

Atom constructors
^^^^^^^^^^^^^^^^^
There are some common arguments for most
Atom-derived constructors:

.. method:: Atom.__init__(*args)

    Parameters
    ----------
    itemsize : int
        For types with a non-fixed size, this sets the size in
        bytes of individual items in the atom.
    shape : tuple
        Sets the shape of the atom. An integer shape of
        N is equivalent to the tuple (N,).
    dflt
        Sets the default value for the atom.


A relation of the different constructors with their
parameters follows.

.. class:: StringAtom(itemsize, shape=(), dflt='')

    Defines an atom of type string.

    The item size is the *maximum* length
    in characters of strings.


.. class:: BoolAtom(shape=(), dflt=False)

    Defines an atom of type bool.


.. class:: IntAtom(itemsize=4, shape=(), dflt=0)

    Defines an atom of a signed integral type (int kind).


.. class:: Int8Atom(shape=(), dflt=0)

    Defines an atom of type int8.


.. class:: Int16Atom(shape=(), dflt=0)

    Defines an atom of type int16.


.. class:: Int32Atom(shape=(), dflt=0)

    Defines an atom of type int32.


.. class:: Int64Atom(shape=(), dflt=0)

    Defines an atom of type int64.


.. class:: UIntAtom(itemsize=4, shape=(), dflt=0)

    Defines an atom of an unsigned integral type (uint kind).


.. class:: UInt8Atom(shape=(), dflt=0)

    Defines an atom of type uint8.


.. class:: UInt16Atom(shape=(), dflt=0)

    Defines an atom of type uint16.


.. class:: UInt32Atom(shape=(), dflt=0)

    Defines an atom of type uint32.


.. class:: UInt64Atom(shape=(), dflt=0)

    Defines an atom of type uint64.


.. class:: Float32Atom(shape=(), dflt=0.0)

    Defines an atom of type float32.


.. class:: Float64Atom(shape=(), dflt=0.0)

    Defines an atom of type float64.


.. class:: ComplexAtom(itemsize, shape=(), dflt=0j)

    Defines an atom of kind complex.

    Allowed item sizes are 8 (single precision) and 16 (double
    precision). This class must be used instead of more concrete
    ones to avoid confusions with numarray-like
    precision specifications used in PyTables 1.X.


.. class:: TimeAtom(itemsize=4, shape=(), dflt=0)

    Defines an atom of time type (time kind).

    There are two distinct supported types of time: a 32 bit
    integer value and a 64 bit floating point value. Both of them
    reflect the number of seconds since the Unix epoch. This atom
    has the property of being stored using the HDF5 time datatypes.


.. class:: Time32Atom(shape=(), dflt=0)

    Defines an atom of type time32.


.. class:: Time64Atom(shape=(), dflt=0.0)

    Defines an atom of type time64.


.. class:: EnumAtom(enum, dflt, base, shape=())

    Description of an atom of an enumerated type.

    Instances of this class describe the atom type used to
    store enumerated values. Those values belong to an enumerated
    type, defined by the first argument (enum) in
    the constructor of the atom, which accepts the same kinds of
    arguments as the Enum class (see :ref:`EnumClassDescr`).  The
    enumerated type is stored in the enum
    attribute of the atom.

    A default value must be specified as the second argument
    (dflt) in the constructor; it must be the
    *name* (a string) of one of the enumerated
    values in the enumerated type. When the atom is created, the
    corresponding concrete value is broadcast and stored in the
    dflt attribute (setting different default
    values for items in a multidimensional atom is not supported
    yet). If the name does not match any value in the enumerated
    type, a KeyError is raised.

    Another atom must be specified as the
    base argument in order to determine the base
    type used for storing the values of enumerated values in memory
    and disk. This *storage atom* is kept in the
    base attribute of the created atom. As a
    shorthand, you may specify a PyTables type instead of the
    storage atom, implying that this has a scalar shape.

    The storage atom should be able to represent each and
    every concrete value in the enumeration. If it is not, a
    TypeError is raised. The default value of the
    storage atom is ignored.

    The type attribute of enumerated atoms
    is always enum.

    Enumerated atoms also support comparisons with other objects::

        >>> enum = ['T0', 'T1', 'T2']
        >>> atom1 = EnumAtom(enum, 'T0', 'int8')  # same as ``atom2``
        >>> atom2 = EnumAtom(enum, 'T0', Int8Atom())  # same as ``atom1``
        >>> atom3 = EnumAtom(enum, 'T0', 'int16')
        >>> atom4 = Int8Atom()
        >>> atom1 == enum
        False
        >>> atom1 == atom2
        True
        >>> atom2 != atom1
        False
        >>> atom1 == atom3
        False
        >>> atom1 == atom4
        False
        >>> atom4 != atom1
        True

    Examples
    --------

    The next C enum construction::

        enum myEnum {
                    T0,
                    T1,
                    T2
                    };

    would correspond to the following PyTables
    declaration::

        >>> myEnumAtom = EnumAtom(['T0', 'T1', 'T2'], 'T0', 'int32')

    Please note the dflt argument with a
    value of 'T0'. Since the concrete value
    matching T0 is unknown right now (we have
    not used explicit concrete values), using the name is the only
    option left for defining a default value for the atom.

    The chosen representation of values for this enumerated
    atom uses unsigned 32-bit integers, which surely wastes quite
    a lot of memory. Another size could be selected by using the
    base argument (this time with a full-blown
    storage atom)::

        >>> myEnumAtom = EnumAtom(['T0', 'T1', 'T2'], 'T0', UInt8Atom())

    You can also define multidimensional arrays for data
    elements::

        >>> myEnumAtom = EnumAtom(
        ...    ['T0', 'T1', 'T2'], 'T0', base='uint32', shape=(3,2))

    for 3x2 arrays of uint32.


Pseudo atoms
^^^^^^^^^^^^
Now, there come three special classes,
ObjectAtom, VLStringAtom and
VLUnicodeAtom, that actually do not descend
from Atom, but which goal is so similar that
they should be described here. Pseudo-atoms can only be used with
VLArray datasets (see :ref:`VLArrayClassDescr`), and
they do not support multidimensional values, nor multiple values
per row.

They can be recognised because they also have
kind, type and
shape attributes, but no
size, itemsize or
dflt ones. Instead, they have a
base atom which defines the elements used for
storage.

See :file:`examples/vlarray1.py` and
:file:`examples/vlarray2.py` for further examples on
VLArray datasets, including object
serialization and string management.


ObjectAtom
..........
.. class:: ObjectAtom()

    Defines an atom of type object.

    This class is meant to fit *any* kind
    of Python object in a row of a VLArray
    dataset by using cPickle behind the
    scenes. Due to the fact that you can not foresee how long will
    be the output of the cPickle serialization
    (i.e. the atom already has a *variable*
    length), you can only fit *one object per
    row*. However, you can still group several objects in
    a single tuple or list and pass it to the
    :meth:`VLArray.append` method.

    Object atoms do not accept parameters and they cause the
    reads of rows to always return Python objects. You can regard
    object atoms as an easy way to save an
    arbitrary number of generic Python objects in a
    VLArray dataset.


.. _VLStringAtom:

VLStringAtom
............
.. class:: VLStringAtom()

    Defines an atom of type vlstring.

    This class describes a *row* of the
    VLArray class, rather than an atom. It
    differs from the StringAtom class in that you
    can only add *one instance of it to one specific
    row*, i.e. the :meth:`VLArray.append`
    method only accepts one
    object when the base atom is of this type.

    Like StringAtom, this class does not
    make assumptions on the encoding of the string, and raw bytes
    are stored as is.  Unicode strings are supported as long as no
    character is out of the ASCII set; otherwise, you will need to
    *explicitly* convert them to strings before
    you can save them.  For full Unicode support, using
    VLUnicodeAtom (see :ref:`VLUnicodeAtom`) is recommended.

    Variable-length string atoms do not accept parameters and
    they cause the reads of rows to always return Python strings.
    You can regard vlstring atoms as an easy way
    to save generic variable length strings.


.. _VLUnicodeAtom:

VLUnicodeAtom
.............
.. class:: VLUnicodeAtom()

    Defines an atom of type vlunicode.

    This class describes a *row* of the
    VLArray class, rather than an atom.  It is
    very similar to VLStringAtom (see :ref:`VLStringAtom`), but it stores Unicode strings (using
    32-bit characters a la UCS-4, so all strings of the same length
    also take up the same space).

    This class does not make assumptions on the encoding of
    plain input strings.  Plain strings are supported as long as no
    character is out of the ASCII set; otherwise, you will need to
    *explicitly* convert them to Unicode before
    you can save them.

    Variable-length Unicode atoms do not accept parameters and
    they cause the reads of rows to always return Python Unicode
    strings.  You can regard vlunicode atoms as
    an easy way to save variable length Unicode strings.


.. _ColClassDescr:

The Col class and its descendants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. class:: Col

    Defines a non-nested column.

    Col instances are used as a means to
    declare the different properties of a non-nested column in a table
    or nested column.  Col classes are descendants of
    their equivalent Atom classes (see :ref:`AtomClassDescr`), but their
    instances have an additional _v_pos attribute
    that is used to decide the position of the column inside its parent
    table or nested column (see the IsDescription
    class in :ref:`IsDescriptionClassDescr` for more information on column positions).

    In the same fashion as Atom, you should use
    a particular Col descendant class whenever you
    know the exact type you will need when writing your code. Otherwise,
    you may use one of the Col.from_*() factory methods.


Col instance variables
^^^^^^^^^^^^^^^^^^^^^^
In addition to the variables that they inherit from the
Atom class, Col instances
have the following attributes.

.. attribute:: Col._v_pos

    The *relative* position of this
    column with regard to its column siblings.


Col factory methods
^^^^^^^^^^^^^^^^^^^
Each factory method inherited from
the Atom class is available with the same
signature, plus an additional pos parameter
(placed in last position) which defaults
to None and that may take an integer value.
This parameter might be used to specify the position of the column
in the table.

Besides, there are the next additional factory methods,
available only for Col objects.


.. method:: Col.from_atom(atom, pos=None)

    Create a Col definition from a PyTables atom.

    An optional position may be specified as the pos argument.


Col constructors
^^^^^^^^^^^^^^^^
There are some common arguments for most
Col-derived constructors.

.. method:: Col.__init__(*args)

    Parameters
    ----------
    itemsize : int
        For types with a non-fixed size, this sets the size in
        bytes of individual items in the column.
    shape : tuple
        Sets the shape of the column. An integer shape of
        N is equivalent to the tuple (N,).
    dflt
        Sets the default value for the column.
    pos : int
        Sets the position of column in table.  If unspecified,
        the position will be randomly selected.


.. class:: StringCol(itemsize, shape=(), dflt='', pos=None)

    Defines an column of type string.


.. class:: BoolCol(shape=(), dflt=False, pos=None)

    Defines an column of type bool.


.. class:: IntCol(itemsize=4, shape=(), dflt=0, pos=None)

    Defines an column of a signed integral type (int kind).


.. class:: Int8Col(shape=(), dflt=0, pos=None)

    Defines an column of type int8.


.. class:: Int16Col(shape=(), dflt=0, pos=None)

    Defines an column of type int16.


.. class:: Int32Col(shape=(), dflt=0, pos=None)

    Defines an column of type int32.


.. class:: Int64Col(shape=(), dflt=0, pos=None)

    Defines an column of type int64.


.. class:: UIntCol(itemsize=4, shape=(), dflt=0, pos=None)

    Defines an column of an unsigned integral type (uint kind).


.. class:: UInt8Col(shape=(), dflt=0, pos=None)

    Defines an column of type uint8.


.. class:: UInt16Col(shape=(), dflt=0, pos=None)

    Defines an column of type uint16.


.. class:: UInt32Col(shape=(), dflt=0, pos=None)

    Defines an column of type uint32.


.. class:: UInt64Col(shape=(), dflt=0, pos=None)

    Defines an column of type uint64.


.. class:: Float32Col(shape=(), dflt=0.0, pos=None)

    Defines an column of type float32.


.. class:: Float64Col(shape=(), dflt=0.0, pos=None)

    Defines an column of type float64.


.. class:: ComplexCol(itemsize, shape=(), dflt=0j, pos=None)

    Defines an column of kind complex.


.. class:: TimeCol(itemsize=4, shape=(), dflt=0, pos=None)

    Defines an column of time type (time kind).


.. class:: Time32Col(shape=(), dflt=0, pos=None)

    Defines an column of type time32.


.. class:: Time64Col(shape=(), dflt=0.0, pos=None)

    Defines an column of type time64.


.. class:: EnumCol(enum, dflt, base, shape=(), pos=None)

    Description of an column of an enumerated type.



.. _IsDescriptionClassDescr:

The IsDescription class
~~~~~~~~~~~~~~~~~~~~~~~
.. class:: IsDescription()

    Description of the structure of a table or nested column.

    This class is designed to be used as an easy, yet meaningful
    way to describe the structure of new Table (see
    :ref:`TableClassDescr`)
    datasets or nested columns through the definition of
    *derived classes*. In order to define such a
    class, you must declare it as descendant of
    IsDescription, with as many attributes as columns
    you want in your table. The name of each attribute will become the
    name of a column, and its value will hold a description of it.

    Ordinary columns can be described using instances of the
    Col class (see :ref:`ColClassDescr`). Nested columns can be described by
    using classes derived from IsDescription,
    instances of it, or name-description dictionaries. Derived classes
    can be declared in place (in which case the column takes the name of
    the class) or referenced by name.

    Nested columns can have a _v_pos special
    attribute which sets the *relative* position of
    the column among sibling columns *also having explicit
    positions*.  The pos constructor
    argument of Col instances is used for the same
    purpose.  Columns with no explicit position will be placed
    afterwards in alphanumeric order.

    Once you have created a description object, you can pass it to
    the Table constructor, where all the information
    it contains will be used to define the table structure.


IsDescription special attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These are the special attributes that the user can specify
*when declaring* an
IsDescription subclass to complement its
*metadata*.

.. attribute:: IsDescription._v_pos

    Sets the position of a possible nested column
    description among its sibling columns.


IsDescription class variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following attributes are *automatically
created* when an IsDescription
subclass is declared.  Please note that declared columns can no
longer be accessed as normal class variables after its
creation.

.. attribute:: IsDescription.columns

    Maps the name of each column in the description to its
    own descriptive object.



Helper classes
--------------
This section describes some classes that do not fit in any other
section and that mainly serve for ancillary purposes.



.. _FiltersClassDescr:

The Filters class
~~~~~~~~~~~~~~~~~
.. class:: Filters()

    Container for filter properties.

    This class is meant to serve as a container that keeps
    information about the filter properties associated with the chunked
    leaves, that is Table, CArray, EArray and VLArray.

    Instances of this class can be directly compared for equality.


Filters instance variables
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Filters.fletcher32

    Whether the *Fletcher32* filter is active or not.


.. attribute:: Filters.complevel

    The compression level (0 disables compression).


.. attribute:: Filters.complib

    The compression filter used (irrelevant when
    compression is not enabled).

.. attribute:: Filters.shuffle

    Whether the *Shuffle* filter is active or not.


Example of use
^^^^^^^^^^^^^^
This is a small example on using the
Filters class::

    import numpy
    from tables import *

    fileh = openFile('test5.h5', mode='w')
    atom = Float32Atom()
    filters = Filters(complevel=1, complib='blosc', fletcher32=True)
    arr = fileh.createEArray(fileh.root, 'earray', atom, (0,2), "A growable array", filters=filters)

    # Append several rows in only one call
    arr.append(numpy.array([[1., 2.],
                            [2., 3.],
                            [3., 4.]], dtype=numpy.float32))

    # Print information on that enlargeable array
    print "Result Array:"
    print repr(arr)
    fileh.close()

This enforces the use of the Blosc library, a compression
level of 1 and a Fletcher32 checksum filter as well. See the
output of this example::

    Result Array:
    /earray (EArray(3, 2), fletcher32, shuffle, blosc(1)) 'A growable array'
    type = float32
    shape = (3, 2)
    itemsize = 4
    nrows = 3
    extdim = 0
    flavor = 'numpy'
    byteorder = 'little'


.. method:: Filters.__init__(complevel=0, complib='zlib', shuffle=True, fletcher32=False)

    Create a new Filters instance.

    Parameters
    ----------
    complevel : int
        Specifies a compression level for data. The allowed
        range is 0-9. A value of 0 (the default) disables
        compression.
    complib : str
        Specifies the compression library to be used. Right
        now, 'zlib' (the default), 'lzo', 'bzip2'
        and 'blosc' are supported.  Specifying a
        compression library which is not available in the system
        issues a FiltersWarning and sets the library to the default one.
    shuffle : bool
        Whether or not to use the *Shuffle*
        filter in the HDF5 library. This is normally used to improve
        the compression ratio. A false value disables shuffling and
        a true one enables it. The default value depends on whether
        compression is enabled or not; if compression is enabled,
        shuffling defaults to be enabled, else shuffling is
        disabled. Shuffling can only be used when compression is enabled.
    fletcher32 : bool
        Whether or not to use the
        *Fletcher32* filter in the HDF5 library.
        This is used to add a checksum on each data chunk. A false
        value (the default) disables the checksum.


.. method:: Filters.copy(override)

    Get a copy of the filters, possibly overriding some arguments.

    Constructor arguments to be overridden must be passed as keyword arguments.

    Using this method is recommended over replacing the
    attributes of an instance, since instances of this class may
    become immutable in the future::

        >>> filters1 = Filters()
        >>> filters2 = filters1.copy()
        >>> filters1 == filters2
        True
        >>> filters1 is filters2
        False
        >>> filters3 = filters1.copy(complevel=1)
        Traceback (most recent call last):
        ...
        ValueError: compression library ``None`` is not supported...
        >>> filters3 = filters1.copy(complevel=1, complib='zlib')
        >>> print filters1
        Filters(complevel=0, shuffle=False, fletcher32=False)
        >>> print filters3
        Filters(complevel=1, complib='zlib', shuffle=False, fletcher32=False)
        >>> filters1.copy(foobar=42)
        Traceback (most recent call last):
        ...
        TypeError: __init__() got an unexpected keyword argument 'foobar'


.. _IndexClassDescr:

The Index class
~~~~~~~~~~~~~~~
.. class:: Index

    Represents the index of a column in a table.

    This class is used to keep the indexing information for
    columns in a Table dataset (see :ref:`TableClassDescr`). It is
    actually a descendant of the Group class (see
    :ref:`GroupClassDescr`), with
    some added functionality. An Index is always
    associated with one and only one column in the table.

    .. note:: This class is mainly intended for internal use, but some of
       its documented attributes and methods may be interesting for the
       programmer.


Index instance variables
^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Index.column

    The Column (see :ref:`ColumnClassDescr`)
    instance for the indexed column.

.. attribute:: Index.dirty

    Whether the index is dirty or not.

    Dirty indexes are out of sync with column data, so
    they exist but they are not usable.


.. attribute:: Index.filters

    Filter properties for this index - see
    Filters in :ref:`FiltersClassDescr`.


.. attribute:: Index.nelements

    The number of currently indexed row for this column.


.. attribute:: Index.is_CSI

    Whether the index is completely sorted or not.


Index methods
^^^^^^^^^^^^^

.. method:: Index.readSorted(start=None, stop=None, step=None)

    Return the sorted values of index in the specified range.

    The meaning of the start, stop and step arguments is
    the same as in :meth:`Table.readSorted`.


.. method:: Index.readIndices(start=None, stop=None, step=None)

    Return the indices values of index in the specified range.

    The meaning of the start, stop and step arguments is
    the same as in :meth:`Table.readSorted`.


Index special methods
^^^^^^^^^^^^^^^^^^^^^

.. method:: Index.__getitem__(key)

    Return the indices values of index in the specified range.

    If key argument is an integer, the
    corresponding index is returned.  If key is a
    slice, the range of indices determined by it is returned.  A
    negative value of step in slice is supported,
    meaning that the results will be returned in reverse
    order.

    This method is equivalent to :meth:`Index.readIndices`.


.. _EnumClassDescr:

The Enum class
~~~~~~~~~~~~~~
.. class:: Enum

    Enumerated type.

    Each instance of this class represents an enumerated type. The
    values of the type must be declared
    *exhaustively* and named with
    *strings*, and they might be given explicit
    concrete values, though this is not compulsory. Once the type is
    defined, it can not be modified.

    There are three ways of defining an enumerated type. Each one
    of them corresponds to the type of the only argument in the
    constructor of Enum:

    - *Sequence of names*: each enumerated
      value is named using a string, and its order is determined by
      its position in the sequence; the concrete value is assigned
      automatically::

          >>> boolEnum = Enum(['True', 'False'])

    - *Mapping of names*: each enumerated
      value is named by a string and given an explicit concrete value.
      All of the concrete values must be different, or a
      ValueError will be raised::

          >>> priority = Enum({'red': 20, 'orange': 10, 'green': 0})
          >>> colors = Enum({'red': 1, 'blue': 1})
          Traceback (most recent call last):
          ...
          ValueError: enumerated values contain duplicate concrete values: 1

    - *Enumerated type*: in that case, a copy
      of the original enumerated type is created. Both enumerated
      types are considered equal::

          >>> prio2 = Enum(priority)
          >>> priority == prio2
          True

    Please note that names starting with _ are
    not allowed, since they are reserved for internal usage::

        >>> prio2 = Enum(['_xx'])
        Traceback (most recent call last):
        ...
        ValueError: name of enumerated value can not start with ``_``: '_xx'

    The concrete value of an enumerated value is obtained by
    getting its name as an attribute of the Enum
    instance (see __getattr__()) or as an item (see
    __getitem__()). This allows comparisons between
    enumerated values and assigning them to ordinary Python
    variables::

        >>> redv = priority.red
        >>> redv == priority['red']
        True
        >>> redv > priority.green
        True
        >>> priority.red == priority.orange
        False

    The name of the enumerated value corresponding to a concrete
    value can also be obtained by using the
    __call__() method of the enumerated type. In this
    way you get the symbolic name to use it later with
    __getitem__()::

        >>> priority(redv)
        'red'
        >>> priority.red == priority[priority(priority.red)]
        True

    (If you ask, the __getitem__() method is
    not used for this purpose to avoid ambiguity in the case of using
    strings as concrete values.)


Enum special methods
^^^^^^^^^^^^^^^^^^^^

.. method:: Enum.__call__(value, *default)

    Get the name of the enumerated value with that concrete value.

    If there is no value with that concrete value in the
    enumeration and a second argument is given as a
    default, this is returned. Else, a ValueError is raised.

    This method can be used for checking that a concrete value
    belongs to the set of concrete values in an enumerated type.

.. method:: Enum.__contains__(name)

    Is there an enumerated value with that
    name in the type?

    If the enumerated type has an enumerated value with that
    name, True is returned.
    Otherwise, False is returned. The
    name must be a string.

    This method does *not* check for
    concrete values matching a value in an enumerated type. For
    that, please use the :meth:`Enum.__call__` method.


.. method:: Enum.__eq__(other)

    Is the other enumerated type equivalent to this one?

    Two enumerated types are equivalent if they have exactly
    the same enumerated values (i.e. with the same names and
    concrete values).


.. method:: Enum.__getattr__(name)

    Get the concrete value of the enumerated value with that
    name.

    The name of the enumerated value must
    be a string. If there is no value with that
    name in the enumeration, an
    AttributeError is raised.


.. method:: Enum.__getitem__(name)

    Get the concrete value of the enumerated value with that name.

    The name of the enumerated value must
    be a string. If there is no value with that
    name in the enumeration, a KeyError is raised.


.. method:: Enum.__iter__()

    Iterate over the enumerated values.

    Enumerated values are returned as (name,
    value) pairs *in no particular order*.


.. method:: Enum.__len__()

    Return the number of enumerated values in the enumerated type.


.. method:: Enum.__repr__()

    Return the canonical string representation of the
    enumeration. The output of this method can be evaluated to give
    a new enumeration object that will compare equal to this one.



The Expr class - a general-purpose expression evaluator
-------------------------------------------------------
.. class:: Expr

    Expr is a class for evaluating expressions
    containing array-like objects.  With it, you can evaluate expressions
    (like "3*a+4*b") that operate on arbitrary large
    arrays while optimizing the resources required to perform them
    (basically main memory and CPU cache memory).  It is similar to the
    Numexpr package (see ), but in addition
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

