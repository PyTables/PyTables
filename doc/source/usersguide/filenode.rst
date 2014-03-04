.. _filenode_usersguide:

filenode - simulating a filesystem with PyTables
================================================

.. currentmodule:: tables.nodes.filenode

What is filenode?
-----------------
filenode is a module which enables you to create a PyTables database of nodes
which can be used like regular opened files in Python. In other words, you
can store a file in a PyTables database, and read and write it as you would
do with any other file in Python. Used in conjunction with PyTables
hierarchical database organization, you can have your database turned into an
open, extensible, efficient, high capacity, portable and metadata-rich
filesystem for data exchange with other systems (including backup purposes).

Between the main features of filenode, one can list:

- *Open:* Since it relies on PyTables, which in turn, sits over HDF5 (see
  :ref:`[HDGG1] <HDFG1>`), a standard hierarchical data format from NCSA.

- *Extensible:* You can define new types of nodes, and their instances will
  be safely preserved (as are normal groups, leafs and attributes) by
  PyTables applications having no knowledge of their types. Moreover, the set
  of possible attributes for a node is not fixed, so you can define your own
  node attributes.

- *Efficient:* Thanks to PyTables' proven extreme efficiency on handling huge
  amounts of data. filenode can make use of PyTables' on-the-fly compression
  and decompression of data.

- *High capacity:* Since PyTables and HDF5 are designed for massive data
  storage (they use 64-bit addressing even where the platform does not
  support it natively).

- *Portable:* Since the HDF5 format has an architecture-neutral design, and
  the HDF5 libraries and PyTables are known to run under a variety of
  platforms. Besides that, a PyTables database fits into a single file, which
  poses no trouble for transportation.

- *Metadata-rich:* Since PyTables can store arbitrary key-value pairs (even
  Python objects!) for every database node. Metadata may include authorship,
  keywords, MIME types and encodings, ownership information, access control
  lists (ACL), decoding functions and anything you can imagine!


Finding a filenode node
-----------------------
filenode nodes can be recognized because they have a NODE_TYPE system
attribute with a 'file' value. It is recommended that you use the
:meth:`File.get_node_attr` method of tables.File class to get the NODE_TYPE
attribute independently of the nature (group or leaf) of the node, so you do
not need to care about.


filenode - simulating files inside PyTables
-------------------------------------------
The filenode module is part of the nodes sub-package of PyTables. The
recommended way to import the module is::

    >>> from tables.nodes import filenode

However, filenode exports very few symbols, so you can import * for
interactive usage. In fact, you will most probably only use the NodeType
constant and the new_node() and open_node() calls.

The NodeType constant contains the value that the NODE_TYPE system attribute
of a node file is expected to contain ('file', as we have seen).
Although this is not expected to change, you should use filenode.NodeType
instead of the literal 'file' when possible.

new_node() and open_node() are the equivalent to the Python file() call (alias
open()) for ordinary files. Their arguments differ from that of file(), but
this is the only point where you will note the difference between working
with a node file and working with an ordinary file.

For this little tutorial, we will assume that we have a PyTables database
opened for writing. Also, if you are somewhat lazy at typing sentences, the
code that we are going to explain is included in the examples/filenodes1.py
file.

You can create a brand new file with these sentences::

    >>> import tables
    >>> h5file = tables.open_file('fnode.h5', 'w')


Creating a new file node
~~~~~~~~~~~~~~~~~~~~~~~~
Creation of a new file node is achieved with the new_node() call. You must
tell it in which PyTables file you want to create it, where in the PyTables
hierarchy you want to create the node and which will be its name. The
PyTables file is the first argument to new_node(); it will be also called the
'host PyTables file'. The other two arguments must be given as keyword
arguments where and name, respectively.
As a result of the call, a brand new appendable and readable file node object
is returned.

So let us create a new node file in the previously opened h5file PyTables
file, named 'fnode_test' and placed right under the root of the database
hierarchy. This is that command::

    >>> fnode = filenode.new_node(h5file, where='/', name='fnode_test')

That is basically all you need to create a file node. Simple, isn't it? From
that point on, you can use fnode as any opened Python file (i.e. you can
write data, read data, lines of text and so on).

new_node() accepts some more keyword arguments. You can give a title to your
file with the title argument. You can use PyTables' compression features with
the filters argument. If you know beforehand the size that your file will
have, you can give its final file size in bytes to the expectedsize argument
so that the PyTables library would be able to optimize the data access.

new_node() creates a PyTables node where it is told to. To prove it, we will
try to get the NODE_TYPE attribute from the newly created node::

    >>> print(h5file.get_node_attr('/fnode_test', 'NODE_TYPE'))
    file


Using a file node
~~~~~~~~~~~~~~~~~
As stated above, you can use the new node file as any other opened file. Let
us try to write some text in and read it::

    >>> print("This is a test text line.", file=fnode)
    >>> print("And this is another one.", file=fnode)
    >>> print(file=fnode)
    >>> fnode.write("Of course, file methods can also be used.")
    >>>
    >>> fnode.seek(0)  # Go back to the beginning of file.
    >>>
    >>> for line in fnode:
    ...     print(repr(line))
    'This is a test text line.\\n'
    'And this is another one.\\n'
    '\\n'
    'Of course, file methods can also be used.'

This was run on a Unix system, so newlines are expressed as '\n'. In fact,
you can override the line separator for a file by setting its line_separator
property to any string you want.

While using a file node, you should take care of closing it *before* you
close the PyTables host file.
Because of the way PyTables works, your data it will not be at a risk, but
every operation you execute after closing the host file will fail with a
ValueError. To close a file node, simply delete it or call its close()
method::

    >>> fnode.close()
    >>> print(fnode.closed)
    True


Opening an existing file node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have a file node that you created using new_node(), you can open it
later by calling open_node(). Its arguments are similar to that of file() or
open(): the first argument is the PyTables node that you want to open (i.e. a
node with a NODE_TYPE attribute having a 'file' value), and the second
argument is a mode string indicating how to open the file. Contrary to
file(), open_node() can not be used to create a new file node.

File nodes can be opened in read-only mode ('r') or in read-and-append mode
('a+'). Reading from a file node is allowed in both modes, but appending is
only allowed in the second one. Just like Python files do, writing data to an
appendable file places it after the file pointer if it is on or beyond the
end of the file, or otherwise after the existing data. Let us see an
example::

    >>> node = h5file.root.fnode_test
    >>> fnode = filenode.open_node(node, 'a+')
    >>> print(repr(fnode.readline()))
    'This is a test text line.\\n'
    >>> print(fnode.tell())
    26
    >>> print("This is a new line.", file=fnode)
    >>> print(repr(fnode.readline()))
    ''

Of course, the data append process places the pointer at the end of the file,
so the last readline() call hit EOF. Let us seek to the beginning of the file
to see the whole contents of our file::

    >>> fnode.seek(0)
    >>> for line in fnode:
    ...   print(repr(line))
    'This is a test text line.\\n'
    'And this is another one.\\n'
    '\\n'
    'Of course, file methods can also be used.This is a new line.\\n'

As you can check, the last string we wrote was correctly appended at the end
of the file, instead of overwriting the second line, where the file pointer
was positioned by the time of the appending.


Adding metadata to a file node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can associate arbitrary metadata to any open node file, regardless of its
mode, as long as the host PyTables file is writable. Of course, you could use
the set_node_attr() method of tables.File to do it directly on the proper node,
but filenode offers a much more comfortable way to do it. filenode objects
have an attrs property which gives you direct access to their corresponding
AttributeSet object.

For instance, let us see how to associate MIME type metadata to our file
node::

    >>> fnode.attrs.content_type = 'text/plain; charset=us-ascii'

As simple as A-B-C. You can put nearly anything in an attribute, which opens
the way to authorship, keywords, permissions and more. Moreover, there is not
a fixed list of attributes.
However, you should avoid names in all caps or starting with '_', since
PyTables and filenode may use them internally. Some valid examples::

    >>> fnode.attrs.author = "Ivan Vilata i Balaguer"
    >>> fnode.attrs.creation_date = '2004-10-20T13:25:25+0200'
    >>> fnode.attrs.keywords_en = ["FileNode", "test", "metadata"]
    >>> fnode.attrs.keywords_ca = ["FileNode", "prova", "metadades"]
    >>> fnode.attrs.owner = 'ivan'
    >>> fnode.attrs.acl = {'ivan': 'rw', '@users': 'r'}

You can check that these attributes get stored by running the ptdump command
on the host PyTables file.

.. code-block:: bash

    $ ptdump -a fnode.h5:/fnode_test
    /fnode_test (EArray(113,)) ''
    /fnode_test.attrs (AttributeSet), 14 attributes:
    [CLASS := 'EARRAY',
    EXTDIM := 0,
    FLAVOR := 'numpy',
    NODE_TYPE := 'file',
    NODE_TYPE_VERSION := 2,
    TITLE := '',
    VERSION := '1.2',
    acl := {'ivan': 'rw', '@users': 'r'},
    author := 'Ivan Vilata i Balaguer',
    content_type := 'text/plain; charset=us-ascii',
    creation_date := '2004-10-20T13:25:25+0200',
    keywords_ca := ['FileNode', 'prova', 'metadades'],
    keywords_en := ['FileNode', 'test', 'metadata'],
    owner := 'ivan']

Note that filenode makes no assumptions about the meaning of your metadata,
so its handling is entirely left to your needs and imagination.


Complementary notes
-------------------
You can use file nodes and PyTables groups to mimic a filesystem with files
and directories. Since you can store nearly anything you want as file
metadata, this enables you to use a PyTables file as a portable compressed
backup, even between radically different platforms. Take this with a grain of
salt, since node files are restricted in their naming (only valid Python
identifiers are valid); however, remember that you can use node titles and
metadata to overcome this limitation. Also, you may need to devise some
strategy to represent special files such as devices, sockets and such (not
necessarily using filenode).

We are eager to hear your opinion about filenode and its potential uses.
Suggestions to improve filenode and create other node types are also welcome.
Do not hesitate to contact us!


Current limitations
-------------------
filenode is still a young piece of software, so it lacks some functionality.
This is a list of known current limitations:

#. Node files can only be opened for read-only or read and append mode. This
   should be enhanced in the future.
#. Near future?
#. Only binary I/O is supported currently (read/write strings of bytes)
#. There is no universal newline support yet. The only new-line character
   used at the moment is ``\n``. This is likely to be improved in a near
   future.
#. Sparse files (files with lots of zeros) are not treated specially; if you
   want them to take less space, you should be better off using compression.

These limitations still make filenode entirely adequate to work with most
binary and text files. Of course, suggestions and patches are welcome.

See :ref:`filenode_classes` for detailed documentation on the filenode
interface.

