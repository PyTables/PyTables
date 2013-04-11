# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: March 18, 2005
# Author:  Ivan Vilata - reverse:net.selidor@ivan
#
# $Source$
# $Id$
#
########################################################################

"""Miscellaneous mappings used to avoid circular imports.

Variables:

`class_name_dict`
    Node class name to class object mapping.
`class_id_dict`
    Class identifier to class object mapping.

Misc variables:

`__docformat__`
    The format of documentation strings in this module.
"""

from tables._past import previous_api

# Important: no modules from PyTables should be imported here
# (but standard modules are OK), since the main reason for this module
# is avoiding circular imports!

__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

class_name_dict = {}
"""Node class name to class object mapping.

This dictionary maps class names (e.g. ``'Group'``) to actual class
objects (e.g. `Group`).  Classes are registered here when they are
defined, and they are not expected to be unregistered (by now), but they
can be replaced when the module that defines them is reloaded.
"""

class_id_dict = {}
"""Class identifier to class object mapping.

This dictionary maps class identifiers (e.g. ``'GROUP'``) to actual
class objects (e.g. `Group`).  Classes defining a new ``_c_classid``
attribute are registered here when they are defined, and they are not
expected to be unregistered (by now), but they can be replaced when the
module that defines them is reloaded.
"""

def get_class_by_name(className):
    """Get the node class matching the `className`.

    If the name is not registered, a ``TypeError`` is raised.  The empty
    string and ``None`` are also accepted, and mean the ``Node`` class.
    """

    # The empty string is accepted for compatibility
    # with old default arguments.
    if className is None or className == '':
        className = 'Node'

    # Get the class object corresponding to `classname`.
    if className not in class_name_dict:
        raise TypeError( "there is no registered node class named ``%s``"
                         % (className,) )

    return class_name_dict[className]

getClassByName = previous_api(get_class_by_name)

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:






