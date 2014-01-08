.. currentmodule:: tables

Homogenous storage classes
==========================

.. _ArrayClassDescr:

The Array class
---------------
.. autoclass:: Array


Array instance variables
~~~~~~~~~~~~~~~~~~~~~~~~
.. attribute:: Array.atom

    An Atom (see :ref:`AtomClassDescr`) instance representing the *type*
    and *shape* of the atomic objects to be saved.

.. autoattribute:: Array.rowsize

.. attribute:: Array.nrow

    On iterators, this is the index of the current row.

.. autoattribute:: Array.nrows


Array methods
~~~~~~~~~~~~~
.. automethod:: Array.get_enum

.. automethod:: Array.iterrows

.. automethod:: Array.next

.. automethod:: Array.read


Array special methods
~~~~~~~~~~~~~~~~~~~~~
The following methods automatically trigger actions when an :class:`Array`
instance is accessed in a special way (e.g. ``array[2:3,...,::2]`` will be
equivalent to a call to
``array.__getitem__((slice(2, 3, None), Ellipsis, slice(None, None, 2))))``.

.. automethod:: Array.__getitem__

.. automethod:: Array.__iter__

.. automethod:: Array.__setitem__


.. _CArrayClassDescr:

The CArray class
----------------
.. autoclass:: CArray


.. _EArrayClassDescr:

The EArray class
----------------
.. autoclass:: EArray


.. _EArrayMethodsDescr:

EArray methods
~~~~~~~~~~~~~~

.. automethod:: EArray.append


.. _VLArrayClassDescr:

The VLArray class
-----------------
.. autoclass:: VLArray

..  These are defined in the class docstring
    VLArray instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: VLArray.atom
    .. autoattribute:: VLArray.flavor
    .. autoattribute:: VLArray.nrow
    .. autoattribute:: VLArray.nrows
    .. autoattribute:: VLArray.extdim
    .. autoattribute:: VLArray.nrows


VLArray properties
~~~~~~~~~~~~~~~~~~
.. autoattribute:: VLArray.size_on_disk

.. autoattribute:: VLArray.size_in_memory


VLArray methods
~~~~~~~~~~~~~~~
.. automethod:: VLArray.append

.. automethod:: VLArray.get_enum

.. automethod:: VLArray.iterrows

.. automethod:: VLArray.next

.. automethod:: VLArray.read

.. automethod:: VLArray.get_row_size


VLArray special methods
~~~~~~~~~~~~~~~~~~~~~~~
The following methods automatically trigger actions when a :class:`VLArray`
instance is accessed in a special way (e.g., vlarray[2:5] will be equivalent
to a call to vlarray.__getitem__(slice(2, 5, None)).

.. automethod:: VLArray.__getitem__

.. automethod:: VLArray.__iter__

.. automethod:: VLArray.__setitem__
