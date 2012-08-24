.. currentmodule:: tables

File manipulation class
=======================

.. _FileClassDescr:

The File Class
--------------
.. autoclass:: File

..  These are defined in the class docstring.
    This is necessary because attributes created in a class's
    __init__ method can't be documented with autoattribute.
    See Sphinx bug #904.
    https://bitbucket.org/birkenfeld/sphinx/issue/904

    Attributes
    ~~~~~~~~~~
    .. autoattribute:: File.filename
    .. autoattribute:: File.format_version
    .. autoattribute:: File.isopen
    .. autoattribute:: File.mode
    .. autoattribute:: File.root
    .. autoattribute:: File.rootUEP


File properties
~~~~~~~~~~~~~~~
.. autoattribute:: File.title

.. autoattribute:: File.filters

.. autoattribute:: File.open_count


File methods - file handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.close

.. automethod:: File.copyFile

.. automethod:: File.flush

.. automethod:: File.fileno

.. automethod:: File.__enter__

.. automethod:: File.__exit__

.. automethod:: File.__str__

.. automethod:: File.__repr__

.. automethod:: File.get_file_image


File methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.copyChildren

.. automethod:: File.copyNode

.. automethod:: File.createArray

.. automethod:: File.createCArray

.. automethod:: File.createEArray

.. automethod:: File.createExternalLink

.. automethod:: File.createGroup

.. automethod:: File.createHardLink

.. automethod:: File.createSoftLink

.. automethod:: File.createTable

.. automethod:: File.createVLArray

.. automethod:: File.moveNode

.. automethod:: File.removeNode

.. automethod:: File.renameNode


File methods - tree traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.getNode

.. automethod:: File.isVisibleNode

.. automethod:: File.iterNodes

.. automethod:: File.listNodes

.. automethod:: File.walkGroups

.. automethod:: File.walkNodes

.. automethod:: File.__contains__

.. automethod:: File.__iter__


File methods - Undo/Redo support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.disableUndo

.. automethod:: File.enableUndo

.. automethod:: File.getCurrentMark

.. automethod:: File.goto

.. automethod:: File.isUndoEnabled

.. automethod:: File.mark

.. automethod:: File.redo

.. automethod:: File.undo


File methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.copyNodeAttrs

.. automethod:: File.delNodeAttr

.. automethod:: File.getNodeAttr

.. automethod:: File.setNodeAttr
