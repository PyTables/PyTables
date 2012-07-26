.. currentmodule:: tables

Declarative classes
===================
In this section a series of classes that are meant to
*declare* datatypes that are required for creating
primary PyTables datasets are described.


.. _AtomClassDescr:

The Atom class and its descendants
----------------------------------
.. autoclass:: Atom

..  These are defined in the class docstring
    Atom instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Atom.dflt
    .. autoattribute:: Atom.dtype
    .. autoattribute:: Atom.itemsize
    .. autoattribute:: Atom.kind
    .. autoattribute:: Atom.shape
    .. autoattribute:: Atom.type


Atom properties
~~~~~~~~~~~~~~~
.. autoattribute:: Atom.ndim

.. autoattribute:: Atom.recarrtype

.. autoattribute:: Atom.size


Atom methods
~~~~~~~~~~~~
.. automethod:: Atom.copy


Atom factory methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: Atom.from_dtype

.. automethod:: Atom.from_kind

.. automethod:: Atom.from_sctype

.. automethod:: Atom.from_type


Atom Sub-classes
~~~~~~~~~~~~~~~~
.. autoclass:: StringAtom
    :members:

.. autoclass:: BoolAtom
    :members:

.. autoclass:: IntAtom
    :members:

.. autoclass:: Int8Atom
    :members:

.. autoclass:: Int16Atom
    :members:

.. autoclass:: Int32Atom
    :members:

.. autoclass:: Int64Atom
    :members:

.. autoclass:: UIntAtom
    :members:

.. autoclass:: UInt8Atom
    :members:

.. autoclass:: UInt16Atom
    :members:

.. autoclass:: UInt32Atom
    :members:

.. autoclass:: UInt64Atom
    :members:

.. autoclass:: FloatAtom
    :members:

.. autoclass:: Float32Atom
    :members:

.. autoclass:: Float64Atom
    :members:

.. autoclass:: ComplexAtom
    :members:

.. autoclass:: Time32Atom
    :members:

.. autoclass:: Time64Atom
    :members:

.. autoclass:: EnumAtom
    :members:


Pseudo atoms
~~~~~~~~~~~~
Now, there come three special classes, ObjectAtom, VLStringAtom and
VLUnicodeAtom, that actually do not descend from Atom, but which goal is so
similar that they should be described here. Pseudo-atoms can only be used with
VLArray datasets (see :ref:`VLArrayClassDescr`), and they do not support
multidimensional values, nor multiple values per row.

They can be recognised because they also have kind, type and shape attributes,
but no size, itemsize or dflt ones. Instead, they have a base atom which
defines the elements used for storage.

See :file:`examples/vlarray1.py` and :file:`examples/vlarray2.py` for further
examples on VLArray datasets, including object serialization and string
management.


ObjectAtom
^^^^^^^^^^
.. autoclass:: ObjectAtom
    :members:


.. _VLStringAtom:

VLStringAtom
^^^^^^^^^^^^
.. autoclass:: VLStringAtom
    :members:


.. _VLUnicodeAtom:

VLUnicodeAtom
^^^^^^^^^^^^^
.. autoclass:: VLUnicodeAtom
    :members:


.. _ColClassDescr:

The Col class and its descendants
---------------------------------
.. autoclass:: Col

..
    Col instance variables
    ^^^^^^^^^^^^^^^^^^^^^^
   .. autoattribute:: _v_pos


Col instance variables
~~~~~~~~~~~~~~~~~~~~~~
In addition to the variables that they inherit from the Atom class, Col
instances have the following attributes.

.. attribute:: Col._v_pos

    The *relative* position of this column with regard to its column
    siblings.


Col factory methods
~~~~~~~~~~~~~~~~~~~
.. automethod:: Col.from_atom


Col sub-classes
~~~~~~~~~~~~~~~
.. autoclass:: StringCol
    :members:

.. autoclass:: BoolCol
    :members:

.. autoclass:: IntCol
    :members:

.. autoclass:: Int8Col
    :members:

.. autoclass:: Int16Col
    :members:

.. autoclass:: Int32Col
    :members:

.. autoclass:: Int64Col
    :members:

.. autoclass:: UIntCol
    :members:

.. autoclass:: UInt8Col
    :members:

.. autoclass:: UInt16Col
    :members:

.. autoclass:: UInt32Col
    :members:

.. autoclass:: UInt64Col
    :members:

.. autoclass:: Float32Col
    :members:

.. autoclass:: Float64Col
    :members:

.. autoclass:: ComplexCol
    :members:

.. autoclass:: TimeCol
    :members:

.. autoclass:: Time32Col
    :members:

.. autoclass:: Time64Col
    :members:

.. autoclass:: EnumCol
    :members:


.. _IsDescriptionClassDescr:

The IsDescription class
-----------------------
.. autoclass:: IsDescription


Description helper functions
----------------------------
.. autofunction:: tables.description.descr_from_dtype

.. autofunction:: tables.description.dtype_from_descr


.. _AttributeSetClassDescr:

The AttributeSet class
----------------------
.. autoclass:: tables.attributeset.AttributeSet

..  These are defined in the class docstring
    AttributeSet attributes
    ~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: tables.attributeset.AttributeSet._v_attrnames
    .. autoattribute:: tables.attributeset.AttributeSet._v_attrnamessys
    .. autoattribute:: tables.attributeset.AttributeSet._v_attrnamesuser
    .. autoattribute:: tables.attributeset.AttributeSet._v_unimplemented

AttributeSet properties
~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: tables.attributeset.AttributeSet._v_node


AttributeSet methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: tables.attributeset.AttributeSet._f_copy

.. automethod:: tables.attributeset.AttributeSet._f_list

.. automethod:: tables.attributeset.AttributeSet._f_rename

.. automethod:: tables.attributeset.AttributeSet.__contains__
