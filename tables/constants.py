########################################################################
#
#       License: BSD
#       Created: February 25, 2005
#       Author:  Ivan Vilata - reverse:com.carabos@ivilata
#
#       $Source$
#       $Id$
#
########################################################################

"""
Constant values for PyTables.

Variables:

`MAX_TREE_DEPTH`
    Maximum depth tree allowed in PyTables.
`MAX_GROUP_WIDTH`
    Maximum allowed number of children hanging from a group.
`MAX_NODE_ATTRS`
    Maximum allowed number of attributes in a node.
`MAX_UNDO_PATH_LENGTH`
    Maximum length of paths allowed in undo/redo operations.

Misc variables:

`__docformat__`
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
"""

__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""


# Recommended values for maximum number of groups and maximum depth in tree.
# However, these limits are somewhat arbitrary and can be increased.
MAX_TREE_DEPTH = 2048
"""
Maximum depth tree allowed in PyTables. This number should be
supported by all Python interpreters (i.e. their recursion level
should be bigger that this).
"""

MAX_GROUP_WIDTH = 4096
"""Maximum allowed number of children hanging from a group."""

# Maximum allowed number of attributes in a node.
MAX_NODE_ATTRS = 4096
"""Maximum allowed number of attributes in a node."""

# Maximum pathname length for undo/redo operations.
MAX_UNDO_PATH_LENGTH = 10240
"""Maximum length of paths allowed in undo/redo operations."""



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
