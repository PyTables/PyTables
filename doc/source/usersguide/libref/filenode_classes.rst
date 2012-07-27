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
.. autofunction:: newNode

.. autofunction:: openNode

The FileNode abstract class
---------------------------

.. autoclass:: FileNode

FileNode attributes
~~~~~~~~~~~~~~~~~~~

.. autoattribute:: FileNode.attrs


FileNode methods
~~~~~~~~~~~~~~~~

.. automethod:: FileNode.close

.. automethod:: FileNode.seek

.. automethod:: FileNode.tell


The ROFileNode class
--------------------

.. autoclass:: ROFileNode


ROFileNode methods
~~~~~~~~~~~~~~~~~~

.. automethod:: ROFileNode.flush

.. automethod:: ROFileNode.next

.. automethod:: ROFileNode.read

.. automethod:: ROFileNode.readline

.. automethod:: ROFileNode.readlines

.. automethod:: ROFileNode.xreadlines

.. automethod:: ROFileNode.truncate

.. automethod:: ROFileNode.write

.. automethod:: ROFileNode.writelines



The RAFileNode class
--------------------

.. autoclass:: RAFileNode

RAFileNode methods
~~~~~~~~~~~~~~~~~~

.. automethod:: RAFileNode.flush

.. automethod:: RAFileNode.next

.. automethod:: RAFileNode.read

.. automethod:: RAFileNode.readline

.. automethod:: RAFileNode.readlines

.. automethod:: RAFileNode.xreadlines

.. automethod:: RAFileNode.truncate

.. automethod:: RAFileNode.write

.. automethod:: RAFileNode.writelines

