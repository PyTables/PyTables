.. currentmodule:: tables

Hierarchy definition classes
============================


.. _NodeClassDescr:

The Node class
--------------
.. autoclass:: Node

.. These are defined in class docstring
    .. autoattribute:: Node._v_depth
    .. autoattribute:: Node._v_file
    .. autoattribute:: Node._v_name
    .. autoattribute:: Node._v_pathname
    .. autoattribute:: Node._v_objectid (location independent)

Node instance variables - location dependent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: Node._v_parent


Node instance variables - location independent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: Node._v_attrs

.. autoattribute:: Node._v_isopen


Node instance variables - attribute shorthands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: Node._v_title


Node methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Node._f_close

.. automethod:: Node._f_copy

.. automethod:: Node._f_isvisible

.. automethod:: Node._f_move

.. automethod:: Node._f_remove

.. automethod:: Node._f_rename


Node methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Node._f_delattr

.. automethod:: Node._f_getattr

.. automethod:: Node._f_setattr


.. _GroupClassDescr:

The Group class
---------------
.. autoclass:: Group

..  These are defined in the class docstring
    Group instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The following instance variables are provided in addition to those in Node
    (see :ref:`NodeClassDescr`):

    .. autoattribute:: Group._v_children
    .. autoattribute:: Group._v_groups
    .. autoattribute:: Group._v_hidden
    .. autoattribute:: Group._v_leaves
    .. autoattribute:: Group._v_links
    .. autoattribute:: Group._v_unknown

Group properties
~~~~~~~~~~~~~~~~
.. autoattribute:: Group._v_nchildren

.. autoattribute:: Group._v_filters


Group methods
~~~~~~~~~~~~~

.. important::

    *Caveat:* The following methods are documented for completeness, and they
    can be used without any problem. However, you should use the high-level
    counterpart methods in the File class (see :ref:`FileClassDescr`, because
    they are most used in documentation and examples, and are a bit more
    powerful than those exposed here.

The following methods are provided in addition to those in
Node (see :ref:`NodeClassDescr`):


.. automethod:: Group._f_close

.. automethod:: Group._f_copy

.. automethod:: Group._f_copy_children

.. automethod:: Group._f_get_child

.. automethod:: Group._f_iter_nodes

.. automethod:: Group._f_list_nodes

.. automethod:: Group._f_walk_groups

.. automethod:: Group._f_walknodes


Group special methods
~~~~~~~~~~~~~~~~~~~~~
Following are described the methods that automatically trigger actions when a
Group instance is accessed in a special way.

This class defines the :meth:`__setattr__`, :meth:`__getattr__` and
:meth:`__delattr__` methods, and they set, get and delete *ordinary Python
attributes* as normally intended. In addition to that, :meth:`__getattr__`
allows getting *child nodes* by their name for the sake of easy interaction
on the command line, as long as there is no Python attribute with the same
name. Groups also allow the interactive completion (when using readline) of
the names of child nodes. For instance::

    # get a Python attribute
    nchild = group._v_nchildren

    # Add a Table child called 'table' under 'group'.
    h5file.create_table(group, 'table', my_description)
    table = group.table          # get the table child instance
    group.table = 'foo'          # set a Python attribute

    # (PyTables warns you here about using the name of a child node.)
    foo = group.table            # get a Python attribute
    del group.table              # delete a Python attribute
    table = group.table          # get the table child instance again

.. automethod:: Group.__contains__

.. automethod:: Group.__delattr__

.. automethod:: Group.__getattr__

.. automethod:: Group.__iter__

.. automethod:: Group.__repr__

.. automethod:: Group.__setattr__

.. automethod:: Group.__str__


.. _LeafClassDescr:

The Leaf class
--------------
.. autoclass:: Leaf

..  These are defined in the class docstring
    .. _LeafInstanceVariables:

    Leaf instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~
    These instance variables are provided in addition to those in Node
    (see :ref:`NodeClassDescr`):

    .. autoattribute:: Leaf.byteorder
    .. autoattribute:: Leaf.dtype
    .. autoattribute:: Leaf.extdim
    .. autoattribute:: Leaf.nrows
    .. autoattribute:: Leaf.nrowsinbuf
    .. autoattribute:: Leaf.shape


Leaf properties
~~~~~~~~~~~~~~~
.. autoattribute:: Leaf.chunkshape

.. autoattribute:: Leaf.ndim

.. autoattribute:: Leaf.filters

.. autoattribute:: Leaf.maindim

.. autoattribute:: Leaf.flavor

.. attribute:: Leaf.size_in_memory

    The size of this leaf's data in bytes when it is fully loaded into
    memory.

.. autoattribute:: Leaf.size_on_disk


Leaf instance variables - aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following are just easier-to-write aliases to their Node (see
:ref:`NodeClassDescr`) counterparts (indicated between parentheses):

.. autoattribute:: Leaf.attrs

.. autoattribute:: Leaf.name

.. autoattribute:: Leaf.object_id

.. autoattribute:: Leaf.title


Leaf methods
~~~~~~~~~~~~
.. automethod:: Leaf.close

.. automethod:: Leaf.copy

.. automethod:: Leaf.flush

.. automethod:: Leaf.isvisible

.. automethod:: Leaf.move

.. automethod:: Leaf.rename

.. automethod:: Leaf.remove

.. automethod:: Leaf.get_attr

.. automethod:: Leaf.set_attr

.. automethod:: Leaf.del_attr

.. automethod:: Leaf.truncate

.. automethod:: Leaf.__len__

.. automethod:: Leaf._f_close
