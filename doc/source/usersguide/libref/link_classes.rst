.. currentmodule:: tables

Link classes
============

.. _LinkClassDescr:

The Link class
--------------
.. autoclass:: tables.link.Link

..  These are defined in the class docstring
    .. autoattribute:: tables.link.Link.target

Link instance variables
~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: tables.link.Link._v_attrs


Link methods
~~~~~~~~~~~~
The following methods are useful for copying, moving, renaming and removing
links.

.. automethod:: tables.link.Link.copy

.. automethod:: tables.link.Link.move

.. automethod:: tables.link.Link.remove

.. automethod:: tables.link.Link.rename


.. _SoftLinkClassDescr:

The SoftLink class
------------------
.. autoclass:: tables.link.SoftLink


SoftLink special methods
~~~~~~~~~~~~~~~~~~~~~~~~
The following methods are specific for dereferrencing and representing soft
links.

.. automethod:: tables.link.SoftLink.__call__

.. automethod:: tables.link.SoftLink.__str__


The ExternalLink class
----------------------
.. autoclass:: tables.link.ExternalLink

..  This is defined in the class docstring
    ExternalLink instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: tables.link.ExternalLink.extfile


ExternalLink methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: tables.link.ExternalLink.umount


ExternalLink special methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following methods are specific for dereferrencing and representing
external links.

.. automethod:: tables.link.ExternalLink.__call__

.. automethod:: tables.link.ExternalLink.__str__
