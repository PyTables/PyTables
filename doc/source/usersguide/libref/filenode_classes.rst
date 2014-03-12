.. currentmodule:: tables.nodes.filenode

.. _filenode_classes:

Filenode Module
===============

.. automodule:: tables.nodes.filenode


Module constants
----------------

.. autodata:: NodeType

.. autodata:: NodeTypeVersions


Module functions
----------------

.. autofunction:: new_node

.. autofunction:: open_node

.. autofunction:: read_from_filenode

.. autofunction:: save_to_filenode


The RawPyTablesIO base class
----------------------------

.. autoclass:: RawPyTablesIO


RawPyTablesIO attributes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: RawPyTablesIO.mode


RawPyTablesIO methods
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: RawPyTablesIO.tell

.. automethod:: RawPyTablesIO.seek

.. automethod:: RawPyTablesIO.seekable

.. automethod:: RawPyTablesIO.fileno

.. automethod:: RawPyTablesIO.close

.. automethod:: RawPyTablesIO.flush

.. automethod:: RawPyTablesIO.truncate

.. automethod:: RawPyTablesIO.readable

.. automethod:: RawPyTablesIO.writable

.. automethod:: RawPyTablesIO.readinto

.. automethod:: RawPyTablesIO.readline

.. automethod:: RawPyTablesIO.write


The ROFileNode class
--------------------

.. autoclass:: ROFileNode


ROFileNode attributes
~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: ROFileNode.attrs


ROFileNode methods
~~~~~~~~~~~~~~~~~~

.. automethod:: ROFileNode.flush

.. automethod:: ROFileNode.read

.. automethod:: ROFileNode.readline

.. automethod:: ROFileNode.readlines

.. automethod:: ROFileNode.close

.. automethod:: ROFileNode.seek

.. automethod:: ROFileNode.tell

.. automethod:: ROFileNode.readable

.. automethod:: ROFileNode.writable

.. automethod:: ROFileNode.seekable

.. automethod:: ROFileNode.fileno


The RAFileNode class
--------------------

.. autoclass:: RAFileNode


RAFileNode attributes
~~~~~~~~~~~~~~~~~~~~~

.. autoattribute:: RAFileNode.attrs


RAFileNode methods
~~~~~~~~~~~~~~~~~~

.. automethod:: RAFileNode.flush

.. automethod:: RAFileNode.read

.. automethod:: RAFileNode.readline

.. automethod:: RAFileNode.readlines

.. automethod:: RAFileNode.truncate

.. automethod:: RAFileNode.write

.. automethod:: RAFileNode.writelines

.. automethod:: RAFileNode.close

.. automethod:: RAFileNode.seek

.. automethod:: RAFileNode.tell

.. automethod:: RAFileNode.readable

.. automethod:: RAFileNode.writable

.. automethod:: RAFileNode.seekable

.. automethod:: RAFileNode.fileno

