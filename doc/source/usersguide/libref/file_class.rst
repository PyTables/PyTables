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
    .. autoattribute:: File.root_uep


File properties
~~~~~~~~~~~~~~~
.. autoattribute:: File.title

.. autoattribute:: File.filters

.. autoattribute:: File.open_count


File methods - file handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.close

.. automethod:: File.copy_file

.. automethod:: File.flush

.. automethod:: File.fileno

.. automethod:: File.__enter__

.. automethod:: File.__exit__

.. automethod:: File.__str__

.. automethod:: File.__repr__

.. automethod:: File.get_file_image

.. automethod:: File.get_filesize

.. automethod:: File.get_userblock_size


File methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.copy_children

.. automethod:: File.copy_node

.. automethod:: File.create_array

.. automethod:: File.create_carray

.. automethod:: File.create_earray

.. automethod:: File.create_external_link

.. automethod:: File.create_group

.. automethod:: File.create_hard_link

.. automethod:: File.create_soft_link

.. automethod:: File.create_table

.. automethod:: File.create_vlarray

.. automethod:: File.move_node

.. automethod:: File.remove_node

.. automethod:: File.rename_node


File methods - tree traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.get_node

.. automethod:: File.is_visible_node

.. automethod:: File.iter_nodes

.. automethod:: File.list_nodes

.. automethod:: File.walk_groups

.. automethod:: File.walk_nodes

.. automethod:: File.__contains__

.. automethod:: File.__iter__


File methods - Undo/Redo support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.disable_undo

.. automethod:: File.enable_undo

.. automethod:: File.get_current_mark

.. automethod:: File.goto

.. automethod:: File.is_undo_enabled

.. automethod:: File.mark

.. automethod:: File.redo

.. automethod:: File.undo


File methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.copy_node_attrs

.. automethod:: File.del_node_attr

.. automethod:: File.get_node_attr

.. automethod:: File.set_node_attr
